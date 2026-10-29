"""GLM-5.3-Flash DSA attention, the whole layer with the sublayer boundary inside (as Kimi K3's fused layers, #811): the
lightning indexer with its pooled keys, MLA (dsa_mla) and the boundary, for one decode step (or one speculative verify
step) of T rows on a TP shard.

  indexer   gemm_tile x [wk | gate | weights_proj] (bf16, 384 rows) -> dsa_kpool_update (LayerNorm k, the tail ring, the
            pool of 4 it closes -> the pooled cache);  dsa_q_norm_quant (bf16 q_c) -> gemm_tile wq_b (bf16) ->
            dsa_index_score (Hadamard q, e4m3, the pooled keys: one score per complete pool) -> dsa_index_topk (512 pools)
            -> sparse_mla expands the pools to their positions and adds the tail (the up to 3 positions after the pools)
  boundary  o_proj's K parts (this GPU's partial sum) -> allreduce_send -> hc_post_mix (the sum over the GPUs, the
            attention's mHC post, the FFN's mHC mixes) -> hc_pre (the FFN's mHC pre with post_attention_layernorm)
  layouts   SGLang's (DSATokenToKVPool): the pooled cache [pages, 64 x 132] (e4m3 keys, then fp32 scales), its page table
            of 64 pools a page, the tail [requests, RING, 128] bf16 (k and gate) indexed by each token's request slot
  reference the indexer with SGLang's rounding points (kpool_fp8_index.py, dsa_indexer_kpool.py): the pooled keys and tail
            rows, the pool scores, the top-512 selection
"""
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsa_mla as mla  # noqa: E402
from dsa_mla import DV, H, KPOOL, KVL, NOPE, POOLS, QL, T, bf  # noqa: E402

from mirage.core import bfloat16, float32, int64, uint8  # noqa: E402
from mirage.mpk import compiler  # noqa: E402
from mirage.mpk.static_megakernel import HC_MIX_STRIDE  # noqa: E402

IH, ID, PPAGE, KGW = 32, 128, 64, 384   # indexer heads x dims; pools per pooled page; [wk | gate | weights_proj] rows (288 -> 384)
TILES = 33                             # sparse_mla: 512 x 4 + 3 tail = 2051 index slots, 33 tiles of 64
SLOTS = TILES * 64
MLA_TILES = 36                         # sparse_mla's slots in tiles (the last 3 empty): every divisor of 36 a split count
FLUSH_BYTES = mla.FLUSH_BYTES


def load_indexer(model_dir: str, layer: int, dev) -> dict:
    """The indexer tensors of `layer` (replicated over the TP ranks; bf16 in the checkpoint)."""
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        where = json.load(f)["weight_map"]
    p = f"model.language_model.layers.{layer}.self_attn.indexer."

    def get(name):
        with safe_open(os.path.join(model_dir, where[p + name]), framework="pt") as f:
            return f.get_tensor(p + name).to(dev)

    wk, gate, wp = get("wk.weight"), get("index_kpool_compress_gate"), get("weights_proj.weight")
    assert wk.shape == (ID, H) and gate.shape == (ID, H) and wp.shape == (IH, H)
    zero = torch.zeros(KGW - 2 * ID - IH, H, dtype=torch.bfloat16, device=dev)
    return {"w_kgw": torch.cat([wk, gate, wp, zero]).to(torch.bfloat16).contiguous(),
            "w_iqb": get("wq_b.weight").to(torch.bfloat16).contiguous(),               # [4096, 1536]
            "k_norm_w": get("k_norm.weight").float(), "k_norm_b": get("k_norm.bias").float(),
            "ape": get("index_kpool_compress_ape").float().contiguous()}               # [4, 128]


def make_indexer_state(seq_lens, maxctx: int, reqs: int, ring: int, seed: int, dev) -> dict:
    """Request slots (int64, as SGLang's req_pool_indices), the tails and the pooled cache with random history, the pooled
    page table (one column more than the pools need, as SGLang's pooled_real_page_table: its row stride is wider)."""
    g = torch.Generator().manual_seed(seed)
    T = len(seq_lens)   # the rows (requests) of the step
    maxpp = maxctx // (KPOOL * PPAGE) + 1
    pages = T * maxpp
    keys = (torch.randn(pages, PPAGE * 128, generator=g) * 64).to(torch.float8_e4m3fn).view(torch.uint8)
    scales = torch.exp2(-torch.randint(4, 9, (pages, PPAGE), generator=g).float())
    pcache = torch.cat([keys, scales.view(torch.uint8).view(pages, PPAGE * 4)], 1)
    return {"req_slots": torch.randperm(reqs, generator=g)[:T].to(dev),
            "pooled_bt": torch.randperm(pages, generator=g).view(T, maxpp).int().to(dev),
            "pcache": pcache.contiguous().to(dev),
            "tail_k": bf(torch.randn(reqs, ring, ID, generator=g)).to(dev),
            "tail_score": bf(torch.randn(reqs, ring, ID, generator=g)).to(dev),
            "pool_idx": torch.full((T, POOLS), -1, dtype=torch.int32, device=dev),
            "idx_scores": torch.zeros(T, maxctx // KPOOL, dtype=torch.float32, device=dev)}


# ---- the reference, with SGLang's rounding points ----
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


# ---- the static layer ----
def build_graph(mpk, t: dict, page: int, splits: int, eps: float, ln_eps: float, fused: dict, o_parts: int = 4) -> dict:
    """The layer's nodes; returns the plan's grids {graph_idx: grid} for the GEMMs. splits: sparse_mla tasks per token
    (a divisor of MLA_TILES: 36 = one tile each, 18 = two, 12 = three, ...). fused: {"rms_eps", "hc_eps", "norm_eps",
    "iters"[, "verify": True: the T rows are "requests" (default 1) requests' consecutive tokens, T / requests each (a
    speculative verify step: dsa_kpool_update runs them in order; per-token seq_lens, the other rows idle)]}. o_proj adds
    its o_parts K parts into t["o"] (int64 fixed point: this GPU's partial sum), then allreduce_send (each 128-column block
    once its parts are added, bf16 to every GPU), hc_post_mix (the sum over the GPUs; the attention's mHC post with
    t["h_post_in"] / t["h_res_in"] on the streams t["R_in"] -> t["R_out"]; the FFN's mHC mixes with t["hc_ffn_fn"]) and
    hc_pre (the FFN's mHC pre with t["hc_ffn_small"]: its input t["x_ffn"] = RMSNorm(sum_i pre_i R'_i) t["ffn_norm"], its
    h_post / h_res into t["h_post_out"] / t["h_res_out"])."""
    a = {k: mpk.attach_input(torch_tensor=v, name=k) for k, v in t.items()}
    nt = lambda dims, dtype, name: mpk.new_tensor(dims=dims, dtype=dtype, name=name)
    NH, T = t["w_uk"].shape[0], t["x"].shape[0]
    maxpools = t["idx_scores"].shape[1]
    xq, qkva = nt((T, H), uint8, "x_fp8"), nt((T, QL + KVL), int64, "qkva")
    mpk.group_quant_layer(input=a["x"], output=xq)
    mpk.scaled_gemm_tile_layer(input=xq, weight=a["w_qkva"], weight_scales=a["w_qkva_sf"], output=qkva, combine="add")
    # The graph order is the round-robin plan's order on each SM, so the nodes go in the order their inputs are ready: the
    # indexer branch (bf16 q_c -> wq_b -> scores -> top-k) is the critical path, and the MLA query (q_b -> q_absorb) is needed
    # only by sparse_mla after the top-k; a q_absorb task before a wq_b task on an SM held it until q_b was done (7 us)
    kw = nt((T, KGW), int64, "idx_kw")
    mpk.gemm_tile_layer(input=a["x"], weight=a["w_kgw"], output=kw, combine="add")
    mpk.dsa_q_norm_quant_layer(qkva=qkva, gamma=a["q_a_norm"], output=a["q_c"], eps=eps)    # bf16 q_c for wq_b
    qcq = nt((T, QL), uint8, "q_c_fp8")
    mpk.dsa_q_norm_quant_layer(qkva=qkva, gamma=a["q_a_norm"], output=qcq, eps=eps)
    mpk.dsa_kv_write_layer(qkva=qkva, gamma=a["kv_a_norm"], cos_sin=None, seq_lens=a["seq_lens"],
                           block_table=a["block_table"], cache=a["cache"], kv0=QL, page=page, eps=eps, rope=0)
    mpk.dsa_kpool_update_layer(kw=kw, k_norm_weight=a["k_norm_w"], k_norm_bias=a["k_norm_b"], ape=a["ape"],
                               seq_lens=a["seq_lens"], req_slots=a["req_slots"], pooled_block_table=a["pooled_bt"],
                               tail_k=a["tail_k"], tail_score=a["tail_score"], pooled_cache=a["pcache"], eps=ln_eps,
                               # a verify step: the T rows are one request's draft tokens in order (the pooled keys they
                               # close and their tail entries, token by token)
                               tokens_per_request=T // int(fused.get("requests", 1)) if fused.get("verify") else 1)
    iq = nt((T, IH * ID), int64, "idx_q")
    mpk.gemm_tile_layer(input=a["q_c"], weight=a["w_iqb"], output=iq, combine="add")
    qb = nt((T, NH * NOPE), int64, "q_b")
    mpk.scaled_gemm_tile_layer(input=qcq, weight=a["w_qb"], weight_scales=a["w_qb_sf"], output=qb, combine="add")
    mpk.dsa_index_score_layer(iq=iq, kw=kw, index_cache=a["pcache"], cos_sin=None, seq_lens=a["seq_lens"],
                              block_table=a["pooled_bt"], output=a["idx_scores"], chunks=maxpools // 1024, flash=True)
    mpk.dsa_index_topk_layer(scores=a["idx_scores"], seq_lens=a["seq_lens"], output=a["pool_idx"], unit=KPOOL)
    mpk.dsa_q_absorb_layer(qb=qb, w_uk=a["w_uk"], cos_sin=None, seq_lens=a["seq_lens"], output=a["q_mla"])
    split = nt((T, NH, KVL), float32, "mla_split")
    mpk.sparse_mla_layer(q=a["q_mla"], cache=a["cache"], indices=a["pool_idx"], block_table=a["block_table"],
                         seq_lens=a["seq_lens"], output=split, softmax_scale=NOPE ** -0.5, splits=splits, page=page,
                         kpool=KPOOL, slots=MLA_TILES * 64)
    mpk.dsa_mla_merge_layer(input=split, output=a["o_lat"])
    mpk.dsa_o_absorb_layer(input=a["o_lat"], w_uv=a["w_uv"], output=a["v"])
    vq = nt((T, NH * DV), uint8, "v_fp8")
    mpk.group_quant_layer(input=a["v"], output=vq)
    # the boundary: all-reduce, the attention's mHC post, the FFN's mHC pre
    assert t["o"].dtype == torch.int64
    mpk.scaled_gemm_tile_layer(input=vq, weight=a["w_o"], weight_scales=a["w_o_sf"], output=a["o"], combine="add")
    sent = nt((T, H), bfloat16, "o_sent")
    mpk.allreduce_send_layer(input=a["o"], output=sent)
    mix = nt((T, HC_MIX_STRIDE), float32, "ffn_hc_mix")
    mpk.hc_post_mix_layer(sent=sent, residual=a["R_in"], coef=(a["h_post_in"], a["h_res_in"]), fn=a["hc_ffn_fn"],
                          output=a["R_out"], mix=mix)
    mpk.hc_pre_layer(mix=mix, residual=a["R_out"], hc_small=a["hc_ffn_small"], gamma=a["ffn_norm"], output=a["x_ffn"],
                     rms_eps=fused["rms_eps"], hc_eps=fused["hc_eps"], norm_eps=fused["norm_eps"],
                     sinkhorn_iters=fused["iters"], export=(a["h_post_out"], a["h_res_out"]))
    nodes = compiler.graph_nodes(mpk)
    sg = [n for n in nodes if n.name == "scaled_gemm_tile"]
    gt = [n for n in nodes if n.name == "gemm_tile"]
    # K in parts of 4 tiles: [q_a | kv_a] 16 x 8, q_b 32 x 3; [wk | gate | weights_proj] 3 x 8, wq_b 32 x 3; o_proj 32 x
    # o_parts
    return {sg[0].graph_idx: (sg[0].grid[0], 8, 1), sg[1].graph_idx: (sg[1].grid[0], 3, 1),
            gt[0].graph_idx: (gt[0].grid[0], 8, 1), gt[1].graph_idx: (gt[1].grid[0], 3, 1),
            sg[2].graph_idx: (sg[2].grid[0], o_parts, 1)}

