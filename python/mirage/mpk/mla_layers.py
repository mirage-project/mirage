"""The Kimi K3 MLA layer's own task types for the static compiler (tasks/mla_norm.cuh, mla_absorb.cuh, mla_attention.cuh),
as functions of a StaticMegakernel (the shared layers stay in static_megakernel.py, unchanged). The layer's GEMMs (fused q_a /
kv_a, q_b, g_proj, o_proj with push) are gemm_tile nodes, its aggregation 2 the attn_residual node.

    from mirage.mpk import mla_layers as mla
    kv_new = mla.norm_layer(mpk, latent, wq, wkv, q_a, kv_cache, loc, eps_q, eps_kv)
    mla.absorb_layer(mpk, q_parts, wkcT, q_full)
    mla.attention_layer(mpk, q_full, gate_parts, wvcT, kv_new, kv_cache, page_table, seq_k, o_in, scale)
"""
from mirage.core import bfloat16, float32

from .static_megakernel import COMBINE, StaticMegakernel, float_bits


def _producer(mpk, t):
    from .compiler import graph_nodes, producer_of
    return producer_of(t, graph_nodes(mpk))


def norm_layer(mpk: StaticMegakernel, latent, wq, wkv, q_a, kv_cache, loc, eps_q: float, eps_kv: float, block_dim=(256, 1, 1)) -> int:
    """Task t = token t (tasks/mla_norm.cuh): the fused q_a / kv_a projection's K parts added, rounded to bf16; q_a = RMSNorm of
    its first Q columns with wq -> q_a [T, Q] bf16 (the node's output, 0xFF before each launch: mla_qb polls it); the KV row
    [RMSNorm(next KV columns) with wkv | the last ROPE columns] -> kv_new (a buffer of its own, 0xFF before each launch, polled
    by mla_attention) and the KV cache at row loc[t]. latent: a gemm_tile node's output in slots [T, Q + KV + ROPE] fp32;
    kv_cache [rows, KV + ROPE] bf16; loc [T] int64. Returns kv_new's buffer slot."""
    p = _producer(mpk, latent)
    assert p is not None and p.name == "gemm_tile" and p.params[0] == COMBINE["slots"], "mla norm: a gemm_tile node's K parts in slots"
    T, N, Q, KV = mpk._tokens(latent.dim(0)), latent.dim(1), wq.dim(0), wkv.dim(0)
    ROPE, D = N - Q - KV, kv_cache.dim(1)
    assert D == KV + ROPE and (q_a.dim(0), q_a.dim(1)) == (T, Q) and loc.dim(0) == T
    mpk._dtypes("mla norm_layer", latent=(latent, float32), wq=(wq, bfloat16), wkv=(wkv, bfloat16), q_a=(q_a, bfloat16),
                kv_cache=(kv_cache, bfloat16))
    kv_new = mpk._scratch("mla_kv_new", T * D * 2, reset=0xFF)
    mpk._add_node("mla_norm", (T, 1, 1), block_dim, [latent, wq, wkv, q_a],
                  [(0, -1, -1), (-1, -1, -1), (-1, -1, -1), (0, -1, -1)],
                  [Q, KV, ROPE, float_bits(eps_q), float_bits(eps_kv)],
                  out_buffer=lambda n: (T * Q * 2, 0xFF, False),
                  buf_slots=[mpk._tensor_slot(wq), mpk._tensor_slot(wkv), kv_new, mpk._tensor_slot(kv_cache), mpk._tensor_slot(loc)],
                  cost_inputs=[0])
    return kv_new


def qb_layer(mpk: StaticMegakernel, q_a, wqb, q, block_dim=(256, 1, 1)):
    """Task x = weight rows [64 x, 64 x + 64) (tasks/mla_qb.cuh): q = q_a . wqb^T [T, N] bf16 (0xFF before each launch, polled
    by mla_absorb); the tile's weight copied into shared memory as the task starts, q_a (mla_norm's output) polled."""
    p = _producer(mpk, q_a)
    assert p is not None and p.name == "mla_norm", "mla qb: q_a is mla_norm's output"
    T, K, N = mpk._tokens(q_a.dim(0)), q_a.dim(1), wqb.dim(0)
    assert wqb.dim(1) == K and (q.dim(0), q.dim(1)) == (T, N) and N % 64 == 0
    mpk._dtypes("mla qb_layer", q_a=(q_a, bfloat16), wqb=(wqb, bfloat16), q=(q, bfloat16))
    mpk._add_node("mla_qb", (N // 64, 1, 1), block_dim, [q_a, wqb, q], [(-1, -1, -1), (0, -1, -1), (1, -1, -1)], [K, N],
                  out_buffer=lambda n: (T * N * 2, 0xFF, False), buf_slots=[mpk._tensor_slot(wqb)])


def absorb_layer(mpk: StaticMegakernel, q_parts, wkcT, q_full, heads: int, nope: int = 128, rope: int = 64, block_dim=(256, 1, 1)):
    """Task h = head h (tasks/mla_absorb.cuh): q_b's output (mla_qb, bf16) of head h, q_abs = q_nope . w_kc[h] (wkcT [heads, KV,
    nope] bf16, viewed [heads KV, nope]) -> q_full [T, heads (KV + rope)] bf16 = [q_abs | q_pe] per head (0xFF before each
    launch, polled by mla_attention)."""
    p = _producer(mpk, q_parts)
    assert p is not None and p.name == "mla_qb", "mla absorb: q is mla_qb's output"
    T, KV = mpk._tokens(q_parts.dim(0)), wkcT.dim(0) // heads
    assert q_parts.dim(1) == heads * (nope + rope) and wkcT.dim(1) == nope and q_full.dim(1) == heads * (KV + rope)
    mpk._dtypes("mla absorb_layer", q_parts=(q_parts, bfloat16), wkcT=(wkcT, bfloat16), q_full=(q_full, bfloat16))
    mpk._add_node("mla_absorb", (heads, 1, 1), block_dim, [q_parts, wkcT, q_full],
                  [(1, -1, -1), (0, -1, -1), (1, -1, -1)], [heads, nope, rope, KV],
                  out_buffer=lambda n: (T * heads * (KV + rope) * 2, 0xFF, False),
                  buf_slots=[mpk._tensor_slot(wkcT)], cost_inputs=[0])


def attention_layer(mpk: StaticMegakernel, q_full, gate_parts, wvcT, kv_new: int, kv_cache, page_table, seq_k, attn, o_in,
                    heads: int, scale: float, thr: int, page_size: int = 64, vd: int = 128, splits: int = 1, block_dim=(256, 1, 1)):
    """Task h = head h (tasks/mla_attention.cuh): head h's attention over the request's KV rows (the cache through page_table
    [blocks] int32 for the rows before this step, kv_new (norm_layer's slot) for the step's tokens; seq_k [1] int32 = all
    rows), attn . w_vc[h] (wvcT [heads, vd, KV] bf16, viewed [heads vd, KV]), times bf16(sigmoid(gate)) (g_proj's K parts in
    slots [T, heads vd] fp32) -> o_in [T, heads vd] bf16 (0xFF before each launch, polled by o_proj's warp-MMA tile).
    splits > 1: grid (heads, splits), the KV rows split among them; split 0 combines the others' partials (two buffers of
    their own: the outputs [heads][splits][T][KV] fp32, their m / l [heads][splits][T][2] fp32 reset to 0xFF).
    Contexts over thr rows: the attention is attn [T, heads KV] bf16 (attn_part_layer's output), split 0 of head h reads its
    rows."""
    pq, pg, pa = _producer(mpk, q_full), _producer(mpk, gate_parts), _producer(mpk, attn)
    assert pa is not None and pa.name == "mla_attn_part", "mla attention: attn is mla_attn_part's output"
    assert pq is not None and pq.name == "mla_absorb", "mla attention: q_full is mla_absorb's output"
    assert pg is not None and pg.name == "gemm_tile" and pg.params[0] == COMBINE["slots"], "mla attention: g_proj's K parts in slots"
    T, KV = mpk._tokens(q_full.dim(0)), wvcT.dim(1)
    D = kv_cache.dim(1)
    assert wvcT.dim(0) == heads * vd and gate_parts.dim(1) == heads * vd and q_full.dim(1) == heads * D and o_in.dim(1) == heads * vd
    mpk._dtypes("mla attention_layer", q_full=(q_full, bfloat16), gate_parts=(gate_parts, float32), wvcT=(wvcT, bfloat16),
                kv_cache=(kv_cache, bfloat16), o_in=(o_in, bfloat16))
    parts = []
    if splits > 1:
        parts = [mpk._scratch("mla_attn_part_o", heads * splits * T * KV * 4),
                 mpk._scratch("mla_attn_part_ml", heads * splits * T * 2 * 4, reset=0xFF)]
    mpk._add_node("mla_attention", (heads, splits, 1), block_dim, [q_full, gate_parts, wvcT, attn, o_in],
                  [(1, -1, -1), (1, -1, -1), (0, -1, -1), (1, -1, -1), (1, -1, -1)],
                  [heads, KV, D - KV, vd, page_size, float_bits(scale), thr],
                  out_buffer=lambda n: (T * heads * vd * 2, 0xFF, False),
                  buf_slots=[mpk._tensor_slot(wvcT), kv_new, mpk._tensor_slot(kv_cache), mpk._tensor_slot(page_table),
                             mpk._tensor_slot(seq_k)] + parts)


KV_MAP_ROWS = 1 << 30   # the KV cache map's row count: the framework's cache size is not known when the kernel is generated (the
                        # tasks only read rows its page table names)


def attn_part_layer(mpk: StaticMegakernel, q_full, kv_new: int, kv_cache, page_table, seq_k, attn, heads: int, scale: float,
                    thr: int, group: int = 4, splits: int = 32, page_size: int = 64, late_splits: int = 0, block_dim=(256, 1, 1)):
    """Task (g, s) = heads [group g, + group) x KV split s (tasks/mla_attn_part.cuh): the attention over split s's 64-row
    chunks, the splits combined -> attn [T, heads KV] bf16 (the node's output, 0xFF before each launch), for contexts over
    thr rows (attention_layer's per-head tasks run the shorter ones); late_splits: the last splits run on SMs that start them
    late (the plan's), used only when every split gets 3 chunks. The partials (a fp32,
    no reset; m / l, reset to 0xFF) in buffers of their own; the chunks read through a tensor map of kv_cache."""
    pq = _producer(mpk, q_full)
    assert pq is not None and pq.name == "mla_absorb", "mla attn_part: q_full is mla_absorb's output"
    T, D, G = mpk._tokens(q_full.dim(0)), kv_cache.dim(1), heads // group
    KV = attn.dim(1) // heads
    ROPE = D - KV
    assert heads % group == 0 and attn.dim(1) == heads * KV and q_full.dim(1) == heads * D and splits <= 48
    mpk._dtypes("mla attn_part_layer", q_full=(q_full, bfloat16), kv_cache=(kv_cache, bfloat16), attn=(attn, bfloat16))
    part_o = mpk._scratch("mla_attn_part_o", G * splits * group * T * KV * 4)
    ml = mpk._scratch("mla_attn_part_ml", G * splits * group * T * 2 * 4, reset=0xFF)
    mpk._add_node("mla_attn_part", (G, splits, 1), block_dim, [q_full, attn],
                  [(1, -1, -1), (1, -1, -1)], [heads, group, KV, ROPE, page_size, float_bits(scale), thr] + ([late_splits] if late_splits else []),
                  out_buffer=lambda n: (T * heads * KV * 2, 0xFF, False), concurrent_along_axis=1,   # a group's splits wait for
                  # each other before the combine: on different SMs, met in the same order (10-14: a random plan with two splits
                  # of a group on one SM hung)
                  buf_slots=[kv_new, mpk._tensor_slot(kv_cache), mpk._tensor_slot(page_table), mpk._tensor_slot(seq_k), part_o, ml],
                  map_slots=[mpk._map("bf16", ("t", kv_cache), (KV_MAP_ROWS, D, 64))])

