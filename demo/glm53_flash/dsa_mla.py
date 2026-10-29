"""GLM-5.3-Flash DSA attention, the MLA part: the shapes, this TP shard's weights from the checkpoint, a decode step's state,
and a PyTorch reference with the kernel's rounding points; the graph tensors of the fused layer (dsa_layer.build_graph) and
the round-robin plan of a graph.

Flash's MLA differs from GLM-5.3's (#806) in its shapes only: hidden 4096, q_lora 1536, qk_nope 256 and no rope (NoPE: no
k_pe in the cache, no q_pe), so the cache rows and the query rows are the 512 latent values.

  state      seq_lens (0: an idle slot), a shuffled block table of `page` rows per page, a latent cache of random history
             (N(0, 1) x kv_a_layernorm), x random N(0, 1)
  reference  group_quant e4m3 with s = amax / 448, the checkpoint's FP8 blocks, bf16 at each node output (act_fp8=False:
             exact activations)
"""
import itertools
import json
import os

import torch
from safetensors import safe_open

from mirage.mpk import compiler
from mirage.mpk.fp8_fp32scale_weights import pack_w2

T = int(os.environ.get("STATIC_TOKENS", "8"))   # tokens (rows) per step: the kernel's T (8 or 16)
H, QL, KVL, NOPE, DV, HEADS = 4096, 1536, 512, 256, 256, 64
KPOOL, POOLS = 4, 512        # the indexer: pools of 4 positions, top 512 pools (+ the tail of at most 3)
FLUSH_BYTES = 512 << 20


def bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16)


def relrms(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def load_layer(model_dir: str, layer: int, tp: int, rank: int, dev) -> dict:
    """The attention tensors of `layer` for shard `rank` of `tp` on dev: the FP8 projections as e4m3 [N, K] with their fp32
    128 x 128 block scales (for the reference), W_UK / W_UV from kv_b_proj (bf16), the norms (fp32)."""
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        where = json.load(f)["weight_map"]
    p = f"model.language_model.layers.{layer}.self_attn."

    def get(name):
        with safe_open(os.path.join(model_dir, where[p + name]), framework="pt") as f:
            return f.get_tensor(p + name).to(dev)

    NH = HEADS // tp
    heads, sb = slice(rank * NH * DV, (rank + 1) * NH * DV), slice(rank * NH * 2, (rank + 1) * NH * 2)
    qa, kva = get("q_a_proj.weight"), get("kv_a_proj_with_mqa.weight")
    assert qa.shape == (QL, H) and kva.shape == (KVL, H), "not the GLM-5.3-Flash MLA (q_lora 1536, kv_lora 512, no rope)"
    kvb = get("kv_b_proj.weight").to(torch.bfloat16).view(HEADS, NOPE + DV, KVL)[rank * NH:(rank + 1) * NH]
    qkva = torch.cat([qa.view(torch.uint8), kva.view(torch.uint8)]).view(torch.float8_e4m3fn)
    return {"w_qkva": qkva, "w_qkva_s": torch.cat([get("q_a_proj.weight_scale_inv"),
                                                                   get("kv_a_proj_with_mqa.weight_scale_inv")]).float(),
            "w_qb": get("q_b_proj.weight")[heads].contiguous(), "w_qb_s": get("q_b_proj.weight_scale_inv")[sb].float().contiguous(),
            "w_o": get("o_proj.weight")[:, heads].contiguous(),
            "w_o_s": get("o_proj.weight_scale_inv")[:, sb].float().contiguous(),
            "w_uk": kvb[:, :NOPE].contiguous(),                    # [NH, 256, 512]
            "w_uv": kvb[:, NOPE:].transpose(1, 2).contiguous(),    # [NH, 512, 256]
            "q_a_norm": get("q_a_layernorm.weight").float(), "kv_a_norm": get("kv_a_layernorm.weight").float()}


def packed(w: torch.Tensor, s: torch.Tensor):
    """e4m3 [N, K] -> (128 x 128 pieces viewed [N, K] uint8, scales [N / 128, K / 128] fp32): scaled_gemm_tile's weight."""
    q, sf = pack_w2(w.view(torch.uint8).contiguous(), s.float().contiguous())
    return q.view(w.shape[0], w.shape[1]), sf


def select_positions(L: int, topk: int, g: torch.Generator) -> torch.Tensor:
    """[topk] int32: every position for L <= 4 x 512 + 3, else 512 random complete pools (4 positions each) and the tail (the
    positions of the incomplete pool); then -1."""
    if L <= KPOOL * POOLS + KPOOL - 1:
        pos = torch.arange(L)
    else:
        full = L // KPOOL
        pools = torch.randperm(full, generator=g)[:POOLS].sort().values
        pos = torch.cat([(pools.unsqueeze(1) * KPOOL + torch.arange(KPOOL)).flatten(), torch.arange(full * KPOOL, L)])
    out = torch.full((topk,), -1, dtype=torch.int32)
    out[:len(pos)] = pos.int()
    return out


def make_state(w: dict, seq_lens, maxctx: int, page: int, topk: int, seed: int, dev) -> dict:
    g = torch.Generator().manual_seed(seed)
    T = len(seq_lens)   # the rows (requests) of the step
    maxp = maxctx // page
    rows = T * maxp * page
    cache = bf(torch.randn(rows, KVL, generator=g) * w["kv_a_norm"].cpu())
    return {"x": bf(torch.randn(T, H, generator=g)).to(dev),
            "seq_lens": torch.tensor(seq_lens, dtype=torch.int32, device=dev),
            "block_table": torch.randperm(T * maxp, generator=g).view(T, maxp).int().to(dev),
            "cache": cache.to(dev),
            "indices": torch.stack([select_positions(L, topk, g) for L in seq_lens]).to(dev)}


# ---- the reference, with the kernel's rounding points ----
def group_quant(x: torch.Tensor):
    """bf16 [T, K] -> (e4m3 [T, K], fp32 scales [T, K / 128]): s = max(amax, 1e-10) x fp32(1 / 448), q = e4m3(x / s)."""
    TT, K = x.shape
    xf = x.float().view(TT, K // 128, 128)
    inv448 = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(448.0, dtype=torch.float32)
    s = xf.abs().amax(-1).clamp_min(1e-10) * inv448.to(x.device)
    return (xf / s.unsqueeze(-1)).to(torch.float8_e4m3fn).view(TT, K), s


def fp8_gemm(q: torch.Tensor, s: torch.Tensor, w: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
    """e4m3 activations [T, K] with scales [T, K / 128] times e4m3 weights [N, K] with block scales [N / 128, K / 128]: fp64
    [T, N]."""
    TT, K = q.shape
    N, KB = w.shape[0], K // 128
    part = torch.einsum("tbk,nbk->tnb", q.double().view(TT, KB, 128), w.double().view(N, KB, 128))
    return (part * s.double().unsqueeze(1) * ws.double().repeat_interleave(128, 0)[:N].unsqueeze(0)).sum(-1)


def gemm(x: torch.Tensor, w: torch.Tensor, ws: torch.Tensor, act_fp8: bool = True) -> torch.Tensor:
    """x bf16 [T, K] times the e4m3 weight [N, K] with its block scales: fp64 [T, N]. act_fp8: x through group_quant first (the
    kernel's rounding), else x as it is (exact activations: the weights' own rounding only)."""
    if act_fp8:
        return fp8_gemm(*group_quant(x), w, ws)
    N, K = w.shape
    return x.double() @ (w.double() * ws.double().repeat_interleave(128, 0)[:N].repeat_interleave(128, 1)[:, :K]).t()


def rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float) -> torch.Tensor:
    """dsa_common fx_rmsnorm_vals: fp32 math on the bf16 input, v x (rrms x gamma), rounded to bf16 once."""
    v = x.float()
    rrms = torch.rsqrt(v.square().sum(-1, keepdim=True) / v.shape[-1] + eps)
    return bf(v * (rrms * gamma))


def cache_row(st: dict, t: int, pos: int, page: int) -> int:
    return int(st["block_table"][t, pos // page]) * page + pos % page


def attention(q: torch.Tensor, cache: torch.Tensor, st: dict, seq_lens, page: int) -> torch.Tensor:
    """q [T, NH, 512] bf16 over the given positions of each request: o_lat [T, NH, 512] bf16 (fp64 math)."""
    o_lat = torch.zeros(q.shape, dtype=torch.bfloat16, device=q.device)
    for t, L in enumerate(seq_lens):
        pos = st["indices"][t].long()
        pos = pos[(pos >= 0) & (pos < L)]
        if not len(pos):
            continue
        c = cache[st["block_table"][t].long()[pos // page] * page + pos % page].double()   # [n, 512]
        prob = torch.softmax((q[t].double() @ c.t()) * NOPE ** -0.5, -1)          # [NH, n]
        o_lat[t] = bf(prob @ c)
    return o_lat


def o_absorb(o_lat: torch.Tensor, w: dict) -> torch.Tensor:
    return bf(torch.einsum("thc,hcv->thv", o_lat.double(), w["w_uv"].double())).reshape(T, -1)


def o_proj(v: torch.Tensor, w: dict, act_fp8: bool = True) -> torch.Tensor:
    return gemm(v, w["w_o"], w["w_o_s"], act_fp8).float()


def reference(w: dict, st: dict, cache: torch.Tensor, seq_lens, page: int, eps: float, act_fp8: bool = True) -> dict:
    """One decode step; writes this step's rows into `cache` (a copy of the history). Returns the node outputs and the
    latent rows of this step (c_kv). act_fp8: the activations of the FP8 GEMMs through group_quant (the kernel's rounding
    points), else exact (gemm)."""
    NH = w["w_uk"].shape[0]
    qkva = bf(gemm(st["x"], w["w_qkva"], w["w_qkva_s"], act_fp8))
    c_q, c_kv = rmsnorm(qkva[:, :QL], w["q_a_norm"], eps), rmsnorm(qkva[:, QL:], w["kv_a_norm"], eps)
    for t, L in enumerate(seq_lens):
        if L > 0:
            cache[cache_row(st, t, L - 1, page)] = c_kv[t]
    qb = bf(gemm(c_q, w["w_qb"], w["w_qb_s"], act_fp8)).view(T, NH, NOPE)
    q = bf(torch.einsum("thk,hkc->thc", qb.double(), w["w_uk"].double()))       # [T, NH, 512]
    o_lat = attention(q, cache, st, seq_lens, page)
    v = o_absorb(o_lat, w)
    return {"q": q, "o_lat": o_lat, "v": v, "o": o_proj(v, w, act_fp8), "c_kv": c_kv}


# ---- the static layer ----
def gpu_tensors(w: dict, st: dict) -> dict:
    """The graph tensors by name: the packed weights, the state, and the outputs the check reads (q, o_lat, v, o: nodes
    write them, as these graph tensors bind the node buffers)."""
    NH = w["w_uk"].shape[0]
    t = {"x": st["x"], "seq_lens": st["seq_lens"], "block_table": st["block_table"], "cache": st["cache"],
         "indices": st["indices"], "w_uk": w["w_uk"], "w_uv": w["w_uv"], "q_a_norm": w["q_a_norm"],
         "kv_a_norm": w["kv_a_norm"]}
    for k in ("w_qkva", "w_qb", "w_o"):
        t[k], t[k + "_sf"] = packed(w[k], w[k + "_s"])
    dev, T = st["x"].device, st["x"].shape[0]
    t["q_mla"] = torch.zeros(T, NH, KVL, dtype=torch.bfloat16, device=dev)
    t["o_lat"] = torch.zeros(T, NH, KVL, dtype=torch.bfloat16, device=dev)
    t["v"] = torch.zeros(T, NH * DV, dtype=torch.bfloat16, device=dev)
    t["o"] = torch.zeros(T, H, dtype=torch.float32, device=dev)
    return t


def round_robin_plan(mpk, grids: dict) -> dict:
    """grids {graph_idx: grid} for the nodes they name (the others: their layer's grid); the tasks dealt to the SMs in turn, in
    graph order (x slowest)."""
    num_sms = mpk._num_sms()
    per_sm, cursor = [[] for _ in range(num_sms)], 0
    for n in compiler.graph_nodes(mpk):
        for pos in itertools.product(*(range(d) for d in grids.get(n.graph_idx, n.grid))):
            per_sm[cursor % num_sms].append([n.graph_idx, list(pos)])
            cursor += 1
    return {"grids": {str(k): list(v) for k, v in grids.items()}, "lists": [per_sm]}

