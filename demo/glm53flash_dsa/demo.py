"""Run the GLM-5.3-Flash DSA attention decode layer (the pooled lightning indexer and MLA) with the static-schedule compiler.

The demo runs one decode step of T = 8 requests on one TP shard of one layer, with one kernel. With --tp 4, the shard has 16
of the 64 heads. The A projections and the indexer are replicated. Flash differs from GLM-5.3 (demo/glm53_dsa) in:
  MLA      hidden 4096, q_lora 1536, qk_nope 256 and no rope (the cache rows and the query rows are the 512 latent values)
  indexer  the keys are pooled (SGLang's IndexerKPool): every 4 tokens, one pooled key (softmax(gate + ape) over the 4
           LayerNormed keys, Hadamard, e4m3 with a power-of-two scale) goes to the pooled index cache; the last up to 3
           tokens wait in a tail ring per request. The query: wq_b in bf16, Hadamard, e4m3 per head with a power-of-two
           scale; the head weights in fp32. The indexer scores the complete pools and selects the top 512 pools;
           sparse_mla attends to their 2048 positions and to the tail (the up to 3 positions after the last pool).
The layouts are SGLang's (DSATokenToKVPool): the pooled cache [pages, 64 x 132] (64 pools of e4m3 keys, then their fp32
scales), its page table of 64 pools a page, the tails [request slots, RING, 128] bf16 (k and gate) by each token's request
slot (int64).

  graph (19 nodes):
    group_quant(x) -> scaled_gemm_tile [q_a | kv_a] -> dsa_q_norm_quant (bf16 q_c) -> gemm_tile wq_b (bf16) ----------+
    gemm_tile x [wk | gate | weights_proj] -> dsa_kpool_update (tail, pooled cache)                                   |
                   [q_a | kv_a] -> dsa_q_norm_quant (e4m3 q_c) -> scaled_gemm_tile q_b -> dsa_q_absorb               |
                                -> dsa_kv_write (the latent cache row)                                                 |
    dsa_index_score (FLASH: the pools) <------------------------------------------------------------------------------+
      -> dsa_index_topk (count, write; unit 4: pool ids) -> sparse_mla (KPOOL 4: the pools' positions and the tail;
      17 splits of 2 tiles) -> dsa_mla_merge -> dsa_o_absorb (W_UV) -> group_quant(v) -> scaled_gemm_tile o_proj (K parts,
      added) -> fx_bf16 (this rank's o_proj partial sum, bf16 [8, 4096])
    The graph order is the hand plan's order on each SM: the indexer branch is the critical path, so its nodes come first.

  0. weights        The attention tensors of --layer (weights.py) from --model-dir.
  1. state          seq_lens (--seq-lens; 0: an idle slot), a shuffled block table (--page rows per page) and random history:
                    the latent cache (N(0, 1) x kv_a_layernorm), the pooled cache (e4m3 N(0, 1) keys, power-of-two scales),
                    the tails; x random.
  2. compile_plan   --plan: hand (default), solver, rr, or a plan file. The plan files are for one GPU of 148 SMs, with
                    --tp 4 and --maxctx 8192 (2048 pools: 2 chunks of 1024 for the top-k):
                      hand_plan.json    the grids [q_a | kv_a] 16 x 8 (4 K tiles per task), [wk | gate | weights_proj] 3 x 8,
                                        wq_b 32 x 3, q_b 32 x 3, o_proj 32 x 4; the other nodes keep the default grid of
                                        their layer. The tasks go to the SMs in turn, in graph order (rr: the same for
                                        any --maxctx).
                      solver_plan.json  search.search_plan from the hand plan, with the prefetch cost model and the
                                        cumulative solver model, over the cost records of 72 plans at short contexts (20 to
                                        275) on a B200: [wk | gate | weights_proj] 3 x 16, wq_b 32 x 2, dsa_index_score 4
                                        chunks, o_proj 32 x 2 and its own placement. Faster than the hand plan in 200
                                        interleaved launches each: 53.3 vs 56.1 us at contexts 20 to 275, 59.9 vs 64.5 us
                                        at 3346 to 3600 (page 1, -DSTATIC_RESET_IN_KERNEL). Thus the default could be
                                        either; the hand plan is kept as the default for any --maxctx.
                    kernels/<plan>_layer.cu is the layer each one gives.
  3. compile_static OUT/build/layer.cu, built and loaded.
  4. check          The first launch (not timed), against a PyTorch reference with SGLang's rounding points for the indexer
                    and the kernel's for MLA, node by node:
                      pooled keys and tail rows written    bit-equal (from the kernel's k and gate inputs)
                      scores                               relRMS and the equal share (from the kernel's q_c)
                      top-k                                the kernel's selection is the top 512 of its own scores (ties:
                                                           the lowest pool); its overlap with the reference selection
                      MLA (on the kernel's selection)      the latent rows written; relRMS of q, o_lat, v and o, end to end
                                                           (from x) and per node (from the kernel's own input to the node)
  5. timed launches --reps launches with the same inputs (the history restored before each), 512 MB read before each
                    (empties L2); span per launch.
Steps 4-5 run the layer through the test harness (python/mirage/mpk/static_harness.py).

python demo.py --model-dir /raid/catalyst/models/GLM-5.3-Flash [--layer 3] [--tp 4 --rank 0] [--seq-lens ...] [--plan PLAN]
    [--page 64] [--maxctx 8192] [--reps 20] [--nvcc-flag=-DSTATIC_RESET_IN_KERNEL]
"""
import argparse
import itertools
import json
import os
import socket
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from weights import DV, H, ID, IH, KGW, KVL, NOPE, QL, load_indexer, load_layer, packed  # noqa: E402

from mirage.core import float32, int64, uint8  # noqa: E402
from mirage.mpk import compiler  # noqa: E402
from mirage.mpk.static_harness import Harness  # noqa: E402
from mirage.mpk.static_megakernel import StaticMegakernel  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T = 8
KPOOL, POOLS, PPAGE = 4, 512, 64   # positions per pool; pools selected; pools per page of the pooled cache
TILES = 33                         # sparse_mla: 512 x 4 + 3 tail = 2051 index slots, 33 tiles of 64
SPLITS = 17                        # sparse_mla tasks per request: two tiles each
SLOTS = SPLITS * -(-TILES // SPLITS) * 64
REQS, RING = 16, 4                 # the tails: request slots, slots per request (SGLang: index_kpool + its extra slots)
FLUSH_BYTES = 512 << 20


def bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def relrms(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


# ---- the state ----
def make_state(w: dict, seq_lens, maxctx: int, page: int, seed: int, dev) -> dict:
    """x, seq_lens, the block table and the latent cache with random history; the request slots (int64, as SGLang's
    req_pool_indices), the tails and the pooled cache with random history, the pooled page table (one column more than the
    pools need, as SGLang's pooled_real_page_table: its row stride is wider)."""
    g = torch.Generator().manual_seed(seed)
    maxp, maxpp = maxctx // page, maxctx // (KPOOL * PPAGE) + 1
    rows, pages = T * maxp * page, T * maxpp
    cache = bf(torch.randn(rows, KVL, generator=g) * w["kv_a_norm"].cpu())
    st = {"x": bf(torch.randn(T, H, generator=g)).to(dev),
          "seq_lens": torch.tensor(seq_lens, dtype=torch.int32, device=dev),
          "block_table": torch.randperm(T * maxp, generator=g).view(T, maxp).int().to(dev),
          "cache": cache.to(dev)}
    g = torch.Generator().manual_seed(seed + 1)
    keys = (torch.randn(pages, PPAGE * 128, generator=g) * 64).to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.exp2(-torch.randint(4, 9, (pages, PPAGE), generator=g).float())
    pcache = torch.cat([keys, scales.view(torch.uint8).view(pages, PPAGE * 4)], 1)
    st.update({"req_slots": torch.randperm(REQS, generator=g)[:T].to(dev),
               "pooled_bt": torch.randperm(pages, generator=g).view(T, maxpp).int().to(dev),
               "pcache": pcache.contiguous().to(dev),
               "tail_k": bf(torch.randn(REQS, RING, ID, generator=g)).to(dev),
               "tail_score": bf(torch.randn(REQS, RING, ID, generator=g)).to(dev),
               "pool_idx": torch.full((T, POOLS), -1, dtype=torch.int32, device=dev),
               "idx_scores": torch.zeros(T, maxctx // KPOOL, dtype=torch.float32, device=dev)})
    return st


# ---- the reference: MLA with the kernel's rounding points ----
def group_quant(x: torch.Tensor):
    """bf16 [T, K] -> (e4m3 [T, K], fp32 scales [T, K / 128]): s = max(amax, 1e-10) x fp32(1 / 448), q = e4m3(x / s)."""
    TT, K = x.shape
    xf = x.float().view(TT, K // 128, 128)
    inv448 = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(448.0, dtype=torch.float32)
    s = xf.abs().amax(-1).clamp_min(1e-10) * inv448.to(x.device)
    return (xf / s.unsqueeze(-1)).to(torch.float8_e4m3fn).view(TT, K), s


def gemm(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """x bf16 [T, K] through group_quant, times the e4m3 weight [N, K] with its 128 x 128 block scales: fp64 [T, N]."""
    q, s = group_quant(x)
    TT, K = q.shape
    N, KB = w.shape[0], K // 128
    part = torch.einsum("tbk,nbk->tnb", q.double().view(TT, KB, 128), w.double().view(N, KB, 128))
    return (part * s.double().unsqueeze(1) * ws.double().repeat_interleave(128, 0)[:N].unsqueeze(0)).sum(-1)


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """dsa_common fx_rmsnorm_vals: fp32 math on the bf16 input, v x (rrms x gamma), rounded to bf16 once."""
    v = x.float()
    rrms = torch.rsqrt(v.square().sum(-1, keepdim=True) / v.shape[-1] + eps)
    return bf(v * (rrms * gamma))


def cache_row(st: dict, t: int, pos: int, page: int) -> int:
    return int(st["block_table"][t, pos // page]) * page + pos % page


def attention(q: torch.Tensor, cache: torch.Tensor, st: dict, positions: torch.Tensor, seq_lens, page: int) -> torch.Tensor:
    """q [T, NH, 512] bf16 over the given positions [T, n] of each request (-1: none): o_lat [T, NH, 512] bf16 (fp64 math)."""
    o_lat = torch.zeros(q.shape, dtype=torch.bfloat16, device=q.device)
    for t, L in enumerate(seq_lens):
        pos = positions[t].long()
        pos = pos[(pos >= 0) & (pos < L)]
        if not len(pos):
            continue
        c = cache[st["block_table"][t].long()[pos // page] * page + pos % page].double()   # [n, 512]
        prob = torch.softmax((q[t].double() @ c.t()) * NOPE ** -0.5, -1)          # [NH, n]
        o_lat[t] = bf(prob @ c)
    return o_lat


def o_absorb(o_lat: torch.Tensor, w: dict) -> torch.Tensor:
    return bf(torch.einsum("thc,hcv->thv", o_lat.double(), w["w_uv"].double())).reshape(T, -1)


def o_proj(v: torch.Tensor, w: dict) -> torch.Tensor:
    """This rank's o_proj partial sum: fp32, then bf16 (fx_bf16's rounding of the summed K parts)."""
    return bf(gemm(v, w["w_o"], w["w_o_s"]).float())


def reference(w: dict, st: dict, positions: torch.Tensor, cache: torch.Tensor, seq_lens, page: int, eps: float) -> dict:
    """One decode step of MLA over the given positions; writes this step's rows into `cache` (a copy of the history)."""
    NH = w["w_uk"].shape[0]
    qkva = bf(gemm(st["x"], w["w_qkva"], w["w_qkva_s"]))
    c_q, c_kv = rmsnorm(qkva[:, :QL], w["q_a_norm"], eps), rmsnorm(qkva[:, QL:], w["kv_a_norm"], eps)
    for t, L in enumerate(seq_lens):
        if L > 0:
            cache[cache_row(st, t, L - 1, page)] = c_kv[t]
    qb = bf(gemm(c_q, w["w_qb"], w["w_qb_s"])).view(T, NH, NOPE)
    q = bf(torch.einsum("thk,hkc->thc", qb.double(), w["w_uk"].double()))       # [T, NH, 512]
    o_lat = attention(q, cache, st, positions, seq_lens, page)
    v = o_absorb(o_lat, w)
    return {"q": q, "o_lat": o_lat, "v": v, "o": o_proj(v, w), "c_q": c_q}


# ---- the reference: the indexer with SGLang's rounding points ----
def hadamard128(x: torch.Tensor) -> torch.Tensor:
    """fp32 [..., 128] -> the normalized Hadamard transform: the butterfly stages of strides 1, 2, ..., 64 (SGLang's
    _hadamard128: (a, b) -> (a + b, a - b)), then x fp32(128^-1/2)."""
    for s in (1, 2, 4, 8, 16, 32, 64):
        y = x.reshape(*x.shape[:-1], 128 // (2 * s), 2, s)
        a, b = y[..., 0, :], y[..., 1, :]
        x = torch.stack([a + b, a - b], -2).reshape(*x.shape)
    return x * torch.tensor(0.08838834764831845, dtype=torch.float32, device=x.device)


def quant_ue8m0(x: torch.Tensor):
    """fp32 [..., 128] (bf16 values) -> (e4m3 [..., 128], fp32 scale [...]): s = 2^ceil(log2(max(amax, 1e-4) / 448))."""
    r = x.abs().amax(-1).clamp_min(1e-4) / torch.tensor(448.0, device=x.device)
    m, e = torch.frexp(r)
    s = torch.ldexp(torch.ones_like(r), torch.where(m == 0.5, e - 1, e))
    return (x / s.unsqueeze(-1)).to(torch.float8_e4m3fn), s


def kpool_reference(wi: dict, x: torch.Tensor, st: dict, seq_lens, pcache: torch.Tensor, tail_k: torch.Tensor,
                    tail_score: torch.Tensor, ln_eps: float) -> torch.Tensor:
    """dsa_kpool_update / SGLang's kpool decode update on copies of the pooled cache and the tails (written in place).
    Returns the projection x [wk | gate | weights_proj]^T, fp64 [T, 384]."""
    kw = x.double() @ wi["w_kgw"].double().t()
    k, gt = bf(kw[:, :ID]).float(), bf(kw[:, ID:2 * ID]).float()
    mean = k.mean(-1, keepdim=True)
    kn = bf((k - mean) * torch.rsqrt(((k - mean) ** 2).mean(-1, keepdim=True) + ln_eps) * wi["k_norm_w"] + wi["k_norm_b"])
    ring = tail_k.shape[1]
    for t, L in enumerate(seq_lens):
        pos, r = L - 1, int(st["req_slots"][t])
        if pos < 0 or r < 0:
            continue
        if pos % KPOOL == KPOOL - 1:
            sc, kv = [], []
            for s in range(KPOOL):
                ps = (pos - (KPOOL - 1) + s) % ring
                cur = s == KPOOL - 1
                sc.append((gt[t] if cur else tail_score[r, ps].float()) + wi["ape"][s])
                kv.append(kn[t].float() if cur else tail_k[r, ps].float())
            mx = torch.stack(sc).amax(0)
            acc, den = torch.zeros_like(mx), torch.zeros_like(mx)
            for s in range(KPOOL):   # in slot order, as the kernels
                p = torch.exp(sc[s] - mx)
                den = den + p
                acc = acc + kv[s] * p
            q8, s8 = quant_ue8m0(bf(hadamard128(bf(acc / den).float())).float())
            pool = pos // KPOOL
            pg = pcache[int(st["pooled_bt"][t, pool // PPAGE])]
            o = pool % PPAGE
            pg[o * 128:(o + 1) * 128] = q8.view(torch.uint8)
            pg[PPAGE * 128 + 4 * o:PPAGE * 128 + 4 * o + 4] = s8.reshape(1).view(torch.uint8)
        tail_k[r, pos % ring], tail_score[r, pos % ring] = kn[t], bf(gt[t])
    return kw


def topk_of(scores: torch.Tensor, seq_lens) -> torch.Tensor:
    """The top-512 pool ids of the given scores (the selection rule of dsa_index_topk)."""
    out = torch.full((T, POOLS), -1, dtype=torch.int32, device=scores.device)
    for t, L in enumerate(seq_lens):
        n = max(L, 0) // KPOOL
        if 0 < n <= POOLS:
            out[t, :n] = torch.arange(n, dtype=torch.int32)
        elif n > POOLS:
            v = scores[t, :n].tolist()
            order = sorted(range(n), key=lambda i: (-v[i], i))   # the lowest pool first at a tie
            out[t] = torch.tensor(sorted(order[:POOLS]), dtype=torch.int32)
    return out


def score_reference(wi: dict, q_c: torch.Tensor, kw: torch.Tensor, st: dict, seq_lens, pcache: torch.Tensor):
    """The pool scores [T, MAXPOOLS] (bf16 values, 0 past the pools) and the top-512 pool ids [T, 512] (ascending; all
    pools when there are at most 512; ties: the lowest pool)."""
    iq = bf(q_c.double() @ wi["w_iqb"].double().t()).float().view(T, IH, ID)
    q8, qs = quant_ue8m0(bf(hadamard128(iq)).float())                                 # [T, 32, 128], [T, 32]
    wp = kw[:, 2 * ID:2 * ID + IH].float()
    c32 = torch.tensor(IH ** -0.5, dtype=torch.float32, device=q_c.device)
    c128 = torch.tensor(ID ** -0.5, dtype=torch.float32, device=q_c.device)
    wq = wp * c32 * qs * c128                                                           # fp32, SGLang's order
    scores = torch.zeros(T, st["idx_scores"].shape[1], dtype=torch.float32, device=q_c.device)
    for t, L in enumerate(seq_lens):
        n = max(L, 0) // KPOOL
        if n == 0:
            continue
        j = torch.arange(n, device=q_c.device)
        pg = st["pooled_bt"][t].long()[j // PPAGE]
        off = j % PPAGE
        rows = pcache[pg]                                                               # [n, 64 x 132]
        k8 = torch.gather(rows[:, :PPAGE * 128].reshape(n, PPAGE, 128), 1,
                          off.view(n, 1, 1).expand(n, 1, 128)).view(n, 128).view(torch.float8_e4m3fn)
        ks = torch.gather(rows[:, PPAGE * 128:].contiguous().view(torch.float32), 1, off.view(n, 1)).view(n)
        dot = (q8[t].double() @ k8.double().t()).float()                               # [32, n]
        s = (wq[t].double().unsqueeze(1) * dot.clamp_min(0).double()).sum(0).float()
        scores[t, :n] = bf(s * ks).float()
    return scores, topk_of(scores, seq_lens)


def expand(pool_idx: torch.Tensor, seq_lens) -> torch.Tensor:
    """Pool ids [T, 512] -> positions [T, SLOTS] (the pools' 4 positions, then the tail; -1): sparse_mla's load_index."""
    out = torch.full((T, SLOTS), -1, dtype=torch.int32, device=pool_idx.device)
    for t, L in enumerate(seq_lens):
        p = pool_idx[t][pool_idx[t] >= 0].long()
        pos = torch.cat([(p.unsqueeze(1) * KPOOL + torch.arange(KPOOL, device=p.device)).flatten(),
                         torch.arange(max(L, 0) // KPOOL * KPOOL, max(L, 0), device=p.device)])
        out[t, :len(pos)] = pos.int()
    return out


# ---- the static layer ----
def kernel_tensors(w: dict, wi: dict, st: dict) -> dict:
    """The graph tensors by name, in the order the plan files number their nodes by: the packed weights, the state, and the
    outputs the check reads (q_mla, o_lat, v, o, q_c: nodes write them, as these graph tensors bind the node buffers)."""
    NH = w["w_uk"].shape[0]
    dev = st["x"].device
    t = {"x": st["x"], "seq_lens": st["seq_lens"], "block_table": st["block_table"], "cache": st["cache"],
         "w_uk": w["w_uk"], "w_uv": w["w_uv"], "q_a_norm": w["q_a_norm"], "kv_a_norm": w["kv_a_norm"]}
    for k in ("w_qkva", "w_qb", "w_o"):
        t[k], t[k + "_sf"] = packed(w[k], w[k + "_s"])
    t.update({"q_mla": torch.zeros(T, NH, KVL, dtype=torch.bfloat16, device=dev),
              "o_lat": torch.zeros(T, NH, KVL, dtype=torch.bfloat16, device=dev),
              "v": torch.zeros(T, NH * DV, dtype=torch.bfloat16, device=dev),
              "o": torch.zeros(T, H, dtype=torch.bfloat16, device=dev),
              "pool_idx": st["pool_idx"]})
    t.update({k: wi[k] for k in ("w_kgw", "w_iqb", "k_norm_w", "k_norm_b", "ape")})
    t.update({k: st[k] for k in ("req_slots", "pooled_bt", "pcache", "tail_k", "tail_score", "idx_scores")})
    t["q_c"] = torch.zeros(T, QL, dtype=torch.bfloat16, device=dev)
    return t


def build_graph(mpk, t: dict, page: int, eps: float, ln_eps: float) -> dict:
    """The layer's nodes; returns the hand plan's grids {graph_idx: grid} for the GEMMs."""
    a = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in t.items()}
    nt = lambda dims, dtype, name: mpk.new_tensor(dims=dims, dtype=dtype, name=name)
    NH = t["w_uk"].shape[0]
    maxpools = t["idx_scores"].shape[1]
    xq, qkva = nt((T, H), uint8, "x_fp8"), nt((T, QL + KVL), int64, "qkva")
    mpk.group_quant_layer(input=a["x"], output=xq)
    mpk.scaled_gemm_tile_layer(input=xq, weight=a["w_qkva"], weight_scales=a["w_qkva_sf"], output=qkva, combine="add")
    # The graph order is the hand plan's order on each SM, so the nodes go in the order their inputs are ready: the indexer
    # branch (bf16 q_c -> wq_b -> scores -> top-k) is the critical path, and the MLA query (q_b -> q_absorb) is needed only
    # by sparse_mla after the top-k
    kw = nt((T, KGW), int64, "idx_kw")
    mpk.gemm_tile_layer(input=a["x"], weight=a["w_kgw"], output=kw, combine="add")
    mpk.dsa_q_norm_quant_layer(qkva=qkva, gamma=a["q_a_norm"], output=a["q_c"], eps=eps)    # bf16 q_c for wq_b
    qcq = nt((T, QL), uint8, "q_c_fp8")
    mpk.dsa_q_norm_quant_layer(qkva=qkva, gamma=a["q_a_norm"], output=qcq, eps=eps)
    mpk.dsa_kv_write_layer(qkva=qkva, gamma=a["kv_a_norm"], cos_sin=None, seq_lens=a["seq_lens"],
                           block_table=a["block_table"], cache=a["cache"], kv0=QL, page=page, eps=eps, rope=0)
    mpk.dsa_kpool_update_layer(kw=kw, k_norm_weight=a["k_norm_w"], k_norm_bias=a["k_norm_b"], ape=a["ape"],
                               seq_lens=a["seq_lens"], req_slots=a["req_slots"], pooled_block_table=a["pooled_bt"],
                               tail_k=a["tail_k"], tail_score=a["tail_score"], pooled_cache=a["pcache"], eps=ln_eps)
    iq = nt((T, IH * ID), int64, "idx_q")
    mpk.gemm_tile_layer(input=a["q_c"], weight=a["w_iqb"], output=iq, combine="add")
    qb = nt((T, NH * NOPE), int64, "q_b")
    mpk.scaled_gemm_tile_layer(input=qcq, weight=a["w_qb"], weight_scales=a["w_qb_sf"], output=qb, combine="add")
    mpk.dsa_index_score_layer(iq=iq, kw=kw, index_cache=a["pcache"], cos_sin=None, seq_lens=a["seq_lens"],
                              block_table=a["pooled_bt"], output=a["idx_scores"], chunks=maxpools // 1024, flash=True)
    mpk.dsa_index_topk_layer(scores=a["idx_scores"], seq_lens=a["seq_lens"], output=a["pool_idx"], unit=KPOOL)
    mpk.dsa_q_absorb_layer(qb=qb, w_uk=a["w_uk"], cos_sin=None, seq_lens=a["seq_lens"], output=a["q_mla"])
    # mla_split names the sparse_mla buffer: o [T, S, NH, 512] fp32, then lse [T, S, NH] (not the declared shape)
    split = nt((T, NH, KVL), float32, "mla_split")
    mpk.sparse_mla_layer(q=a["q_mla"], cache=a["cache"], indices=a["pool_idx"], block_table=a["block_table"],
                         seq_lens=a["seq_lens"], output=split, softmax_scale=NOPE ** -0.5, splits=SPLITS, page=page,
                         kpool=KPOOL, slots=SLOTS)
    mpk.dsa_mla_merge_layer(input=split, output=a["o_lat"])
    mpk.dsa_o_absorb_layer(input=a["o_lat"], w_uv=a["w_uv"], output=a["v"])
    vq = nt((T, NH * DV), uint8, "v_fp8")
    mpk.group_quant_layer(input=a["v"], output=vq)
    o_fx = nt((T, H), int64, "o_fx")
    mpk.scaled_gemm_tile_layer(input=vq, weight=a["w_o"], weight_scales=a["w_o_sf"], output=o_fx, combine="add")
    mpk.fx_bf16_layer(input=o_fx, output=a["o"])
    nodes = compiler.graph_nodes(mpk)
    sg = [n for n in nodes if n.name == "scaled_gemm_tile"]
    gt = [n for n in nodes if n.name == "gemm_tile"]
    # K in parts of 4 tiles: [q_a | kv_a] 16 x 8, q_b 32 x 3; [wk | gate | weights_proj] 3 x 8, wq_b 32 x 3; o_proj 32 x 4
    return {sg[0].graph_idx: (sg[0].grid[0], 8, 1), sg[1].graph_idx: (sg[1].grid[0], 3, 1),
            gt[0].graph_idx: (gt[0].grid[0], 8, 1), gt[1].graph_idx: (gt[1].grid[0], 3, 1),
            sg[2].graph_idx: (sg[2].grid[0], 4, 1)}


def round_robin_plan(mpk, grids: dict) -> dict:
    """grids {graph_idx: grid} for the nodes they name (the others: their layer's grid); the tasks dealt to the SMs in turn, in
    graph order (x slowest)."""
    num_sms = mpk._num_sms()
    per_sm, cursor = [[] for _ in range(num_sms)], 0
    for n in compiler.graph_nodes(mpk):
        for pos in itertools.product(*(range(d) for d in grids.get(n.graph_idx, n.grid))):
            per_sm[cursor % num_sms].append((n.graph_idx, tuple(pos)))
            cursor += 1
    return {"grids": dict(grids), "lists": [per_sm]}


def select_plan(mpk, grids: dict, plan: str) -> dict:
    """--plan: hand / solver (the plan files, for 148 SMs and --maxctx 8192), rr (the hand plan's round robin for this graph)
    or a plan file."""
    if plan == "rr":
        return round_robin_plan(mpk, grids)
    path = {"hand": os.path.join(HERE, "hand_plan.json"), "solver": os.path.join(HERE, "solver_plan.json")}.get(plan, plan)
    p = compiler.plan_from_file(path)
    sms = len(p["lists"][0])
    assert sms == mpk._num_sms(), f"{path} is for {sms} SMs, this GPU has {mpk._num_sms()} (use --plan rr)"
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--layer", type=int, default=3, help="a deepseek_sparse_attention layer (3, 7, ..., 43)")
    ap.add_argument("--tp", type=int, default=4, help="the TP size (the heads on this GPU: 64 / tp, at most 16)")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--page", type=int, default=64, help="latent cache rows per block-table entry (1: SGLang's token slots)")
    ap.add_argument("--maxctx", type=int, default=8192, help="a multiple of 4096 (the pools: whole 1024-pool chunks); the "
                    "plan files are for 8192")
    ap.add_argument("--seq-lens", type=int, nargs=8, default=[1, 4, 300, 2051, 2052, 4100, 6001, 8192],
                    help="the context length of each request, with the token of this step (0: an idle slot)")
    ap.add_argument("--plan", default="hand", help="hand (default), solver, rr or a plan file")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="build/glm53flash_dsa", help="the directory for the schedules and the build")
    ap.add_argument("--reps", type=int, default=20, help="the timed launches")
    ap.add_argument("--nvcc-flag", action="append", default=[], help="an nvcc flag for the layer (repeatable), e.g. "
                    "--nvcc-flag=-DSTATIC_RESET_IN_KERNEL")
    args = ap.parse_args()
    assert max(args.seq_lens) <= args.maxctx and args.maxctx % 4096 == 0 and args.maxctx % args.page == 0
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device("cuda", 0)
    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg = json.load(f)["text_config"]
    assert cfg["layer_types"][args.layer] == "deepseek_sparse_attention"
    assert (cfg["index_n_heads"], cfg["index_head_dim"], cfg["index_topk"], cfg["index_kpool"]) == (IH, ID, POOLS * KPOOL, KPOOL)
    eps, ln_eps = cfg["rms_norm_eps"], 1e-6

    # 0-1. weights and state
    w = load_layer(args.model_dir, args.layer, args.tp, args.rank, dev)
    wi = load_indexer(args.model_dir, args.layer, dev)
    st = make_state(w, args.seq_lens, args.maxctx, args.page, args.seed, dev)
    hist = {k: st[k].clone() for k in ("cache", "pcache", "tail_k", "tail_score")}
    t = kernel_tensors(w, wi, st)
    # 2-3. the plan and the layer
    mpk = StaticMegakernel(num_gpus=1)
    plan = select_plan(mpk, build_graph(mpk, t, args.page, eps, ln_eps), args.plan)
    paths, info = compiler.compile_plan(mpk, plan, os.path.join(args.out, "schedules"))
    sk = mpk.compile_static(paths, gpu_tensors=[t], out_dir=os.path.join(args.out, "build"), extra_flags=args.nvcc_flag)
    h = Harness(sk)
    # 4. the check
    try:
        h.launch(FLUSH_BYTES, timeout_s=60)
    except (TimeoutError, RuntimeError) as e:
        print("KERNEL DID NOT FINISH:", e, flush=True)
        os._exit(2)
    torch.cuda.synchronize()
    L = args.seq_lens
    live = [i for i, n in enumerate(L) if n > 0]
    print(f"layer {args.layer}, TP{args.tp} rank {args.rank} ({w['w_uk'].shape[0]} heads), page {args.page}, plan {args.plan}, "
          f"seq_lens {L}")
    pc, tk, ts = hist["pcache"].clone(), hist["tail_k"].clone(), hist["tail_score"].clone()
    kw = kpool_reference(wi, st["x"], st, L, pc, tk, ts, ln_eps)
    pool_rows = [(int(st["pooled_bt"][i, (n - 1) // KPOOL // PPAGE]), (n - 1) // KPOOL % PPAGE) for i, n in enumerate(L)
                 if n > 0 and (n - 1) % KPOOL == KPOOL - 1]
    pool_keys = sum(int((t["pcache"][p, o * 128:(o + 1) * 128] == pc[p, o * 128:(o + 1) * 128]).all()) for p, o in pool_rows)
    tail_eq = ((t["tail_k"].view(torch.int16) == tk.view(torch.int16)).float().mean().item(),
               (t["tail_score"].view(torch.int16) == ts.view(torch.int16)).float().mean().item())
    print(f"  pooled keys: {pool_keys} / {len(pool_rows)} pools closed this step bit-equal; tail k / gate equal "
          f"{100 * tail_eq[0]:.3f}% / {100 * tail_eq[1]:.3f}%")
    scores, picks = score_reference(wi, t["q_c"], kw, st, L, t["pcache"])
    valid = torch.zeros_like(scores, dtype=torch.bool)
    for i, n in enumerate(L):
        valid[i, :max(n, 0) // KPOOL] = True
    s_err = relrms(t["idx_scores"][valid], scores[valid])
    s_eq = (t["idx_scores"][valid] == scores[valid]).float().mean().item()
    topk_ok = bool((topk_of(t["idx_scores"], L) == t["pool_idx"]).all())

    def overlap_of(i):   # the share of the reference's pools that the kernel selected (1 with no pools)
        want = set(picks[i].tolist()) - {-1}
        return len(set(t["pool_idx"][i].tolist()) & want) / len(want) if want else 1.0
    overlap = [overlap_of(i) for i in live]
    print(f"  scores (from the kernel's q_c): relRMS {s_err:.2e}, equal {100 * s_eq:.2f}% | top-k = the top 512 of its "
          f"own scores: {topk_ok} | overlap with the reference selection: " + " ".join(f"{o:.3f}" for o in overlap))
    positions = expand(t["pool_idx"], L)   # MLA on the kernel's selection
    ref_cache = hist["cache"].clone()
    r = reference(w, st, positions, ref_cache, L, args.page, eps)
    rows = [cache_row(st, i, n - 1, args.page) for i, n in enumerate(L) if n > 0]
    same = (t["cache"][rows].view(torch.int16) == ref_cache[rows].view(torch.int16)).float().mean().item()
    node = {"q": r["q"], "o_lat": attention(t["q_mla"], t["cache"], st, positions, L, args.page),
            "v": o_absorb(t["o_lat"], w), "o": o_proj(t["v"], w)}
    print(f"  q_c (bf16 output): relRMS {relrms(t['q_c'], r['c_q']):.2e} | latent rows written bit-equal {100 * same:.2f}%")
    errs = {}
    for k, n in (("q", "q_mla"), ("o_lat", "o_lat"), ("v", "v"), ("o", "o")):
        errs[k], errs[k + "_node"] = relrms(t[n][live], r[k][live]), relrms(t[n][live], node[k][live])
        print(f"  {k:6s} relRMS end to end {errs[k]:.2e}, this node alone {errs[k + '_node']:.2e}")
    ok = pool_keys == len(pool_rows) and min(tail_eq) > 0.999 and s_err < 1e-2 and topk_ok and min(overlap) > 0.95 and \
        same > 0.99 and errs["o_lat_node"] < 3e-3 and errs["o_node"] < 1e-3 and errs["o"] < 2e-2
    if not ok:
        print("CHECK FAILED: no timing")
        sk.finalize()
        sys.exit(1)
    print("CHECK PASSED")
    # 5. timed launches
    spans = []
    for _ in range(args.reps):   # the same inputs again: the history is restored before each launch
        for k in ("cache", "pcache", "tail_k", "tail_score"):
            t[k].copy_(hist[k])
        h.launch(FLUSH_BYTES, timeout_s=60)
        spans.append(h.span_us())
    print(f"machine {socket.gethostname()} {torch.cuda.get_device_name(0)}, grids {info['grids']}, {args.reps} launches "
          f"(L2 emptied before each): span min {min(spans):.1f} median {statistics.median(spans):.1f} max {max(spans):.1f} us")
    sk.finalize()


if __name__ == "__main__":
    main()
