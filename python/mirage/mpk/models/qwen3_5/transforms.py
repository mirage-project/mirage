"""Qwen3.5 load-time weight transforms.

Every transform in `docs/qwen35/v1-architecture.md` §2.0 and the checkpoint ->
runtime map of `docs/qwen35/vllm-graph.md` §5.2/§5.3 lives here as a pure
function of torch tensors, so each one can be unit-checked against the HF
oracle dumps independently of the graph builder
(`tests/runtime_python/blackwell/sm100_qwen35_builder/test_loader_transforms.py`).

The transforms, and where each is pinned:

| name | pin |
|---|---|
| `gemma_fold`                | `1 + w`, fp32 add, bf16 store [VG §1.2; MG Gap 5] |
| (no fold on GDN `norm`)     | `RMSNormGated`, ones-init, no `+1` [VG §1.2] |
| `pack_alog_dtbias`          | `[2, num_v_heads]` fp32 [v1 §2.0] |
| `conv1d_weight`             | `[C,1,4] -> [C,4]` bf16 [VG §5.2] |
| `concat_ba`                 | `[b | a] -> [64, 2048]` bf16 [VG §5.1-5.2] |
| `permute_head_dim*`         | partial-RoPE column permutation [v1 §4.4] |
| `fuse_qkvg_*`               | kv-group interleave into MPK's packed row [v1 §2.0] |
| `stack_experts_*`           | per-expert 2-D -> `w13`/`w2` + UNEXPANDED scales [VG §5.3] |
| `dequant_blockscale`        | exact `fp8 x scale` expansion (bf16 scaffold + checks) |

Scale convention throughout: the checkpoint ships `weight_scale_inv` as
`[N/128, K/128]` (BF16 on disk); everything here widens it to fp32 and keeps
the block values bit-exact — this is the preserved-scale contract task 279 and
tasks 241/242 consume (`models/utils.prepare_fp8_blockscale_weight`,
`v1-architecture.md` §6.2).
"""

from __future__ import annotations

import torch

BLOCK = 128
FP8_MAX = 448.0
FP8 = torch.float8_e4m3fn
BF16 = torch.bfloat16


# --------------------------------------------------------------------------
# norms
# --------------------------------------------------------------------------
def gemma_fold(weight: torch.Tensor) -> torch.Tensor:
    """`w_eff = 1 + w`, added in fp32 and stored bf16.

    `GemmaRMSNorm` computes `x * rsqrt(mean(x^2)+eps) * (1.0 + w.float())`
    (`vllm-graph.md` §1.2). MPK's norm kernels take the weight pointer as
    `T const *` — bf16 for this model (`norm_sm100.cuh:28`, `:103`) — so the
    folded value is rounded to bf16 exactly once, here. Doing the add in bf16
    instead would be a second, much coarser rounding: `1 + w` needs ~14
    mantissa bits for a typical `w ~ 2^-6`, and bf16 has 8.

    This bf16 store is a property of MPK's weight representation, quantified by
    M2-I6 against the oracle
    (`test_attention_qwen35_oracle.py` "ATTRIB REF(fp32 norm weight...)").
    """
    return (1.0 + weight.float()).to(BF16)


def no_fold_norm(weight: torch.Tensor) -> torch.Tensor:
    """The GDN gated norm: used as-is, fp32, NO `1 +` (`vllm-graph.md` §1.2).

    `RMSNormGated.weight` is `ones_`-initialised and applied directly
    (`layernorm.py:212`, `:257`), and `gdn_recurrent_layer` declares its
    `norm_w` input fp32 (`persistent_kernel.py:1227`).
    """
    return weight.float().contiguous()


# --------------------------------------------------------------------------
# GDN packing
# --------------------------------------------------------------------------
def pack_alog_dtbias(a_log: torch.Tensor, dt_bias: torch.Tensor) -> torch.Tensor:
    """`[2, num_v_heads]` fp32: row 0 `A_log`, row 1 `dt_bias`.

    `A_log` ships fp32 and `dt_bias` bf16 (`vllm-graph.md` §5.2); the recurrence
    task wants one fp32 tensor so the pair costs a single input slot
    (`v1-architecture.md` §2.0, §2.2 row 6).
    """
    assert a_log.shape == dt_bias.shape and a_log.dim() == 1
    return torch.stack([a_log.float(), dt_bias.float()]).contiguous()


def conv1d_weight(w: torch.Tensor) -> torch.Tensor:
    """`[conv_dim, 1, kernel]` -> `[conv_dim, kernel]` bf16.

    vLLM materialises the same view (`qwen_gdn_linear_attn.py:1591-1593`);
    `gdn_conv1d_layer` asserts a 2-D weight (`persistent_kernel.py:1193`).
    """
    if w.dim() == 3:
        assert w.shape[1] == 1, f"unexpected conv1d weight shape {tuple(w.shape)}"
        w = w.view(w.shape[0], w.shape[2])
    assert w.dim() == 2
    return w.to(BF16).contiguous()


def concat_ba(b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """`in_proj_b` over `in_proj_a` -> `[2*num_v_heads, hidden]` bf16.

    Plain `[b | a]` order: Qwen3.5 ships the two projections SEPARATELY and
    vLLM fuses them at load into `in_proj_ba` rows `[0:32]` / `[32:64]`
    (`vllm-graph.md` §5.1). This is NOT Qwen3-Next's GQA-interleaved layout.
    Both shards are in `modules_to_not_convert`, so this stays bf16.
    """
    assert b.dim() == a.dim() == 2 and b.shape == a.shape
    return torch.cat([b, a], dim=0).to(BF16).contiguous()


# --------------------------------------------------------------------------
# partial RoPE column permutation
# --------------------------------------------------------------------------
def rope_perm_src(head_dim: int, rotary_dim: int, device) -> torch.Tensor:
    """`src[i]` = the Qwen head_dim column that becomes MPK column `i`.

    Thin wrapper over `demo/qwen3_5/rope_permutation.py`, which owns the
    derivation and the P4 proof; importing it keeps ONE definition of the
    permutation in the tree.
    """
    from .rope import rope_permutation_src

    return torch.as_tensor(rope_permutation_src(head_dim, rotary_dim),
                           dtype=torch.long, device=device)


def q_proj_dest_from_src(num_q_heads: int, head_dim: int, rotary_dim: int,
                         device) -> torch.Tensor:
    """Row map for a packed `[q|gate]`-per-head `q_proj`.

    `q_proj` is `[num_q_heads * 2 * head_dim, hidden]` laid out
    `[h0_q | h0_gate | h1_q | ...]` (`vllm-graph.md` §2.2.2). Only the q half of
    each head is permuted; the gate is never normed or rotated (§2.2.4).
    """
    src = rope_perm_src(head_dim, rotary_dim, device)
    rows = torch.arange(num_q_heads * 2 * head_dim, dtype=torch.long, device=device)
    out = rows.clone()
    for h in range(num_q_heads):
        base = h * 2 * head_dim
        out[base:base + head_dim] = base + src
    return out


def k_proj_dest_from_src(num_kv_heads: int, head_dim: int, rotary_dim: int,
                         device) -> torch.Tensor:
    """Row map for `k_proj` — every kv head's full 256 columns are permuted."""
    src = rope_perm_src(head_dim, rotary_dim, device)
    out = torch.empty(num_kv_heads * head_dim, dtype=torch.long, device=device)
    for g in range(num_kv_heads):
        out[g * head_dim:(g + 1) * head_dim] = g * head_dim + src
    return out


def permute_head_dim_norm(weight: torch.Tensor, head_dim: int,
                          rotary_dim: int) -> torch.Tensor:
    """Permute an ALREADY-Gemma-folded `q_norm`/`k_norm` weight.

    RMSNorm over the full head_dim is permutation-EQUIVARIANT provided the
    weight moves with the data (`rope_permutation.py` "Why it is exact").
    """
    src = rope_perm_src(head_dim, rotary_dim, weight.device)
    return weight.index_select(0, src).contiguous()


# --------------------------------------------------------------------------
# block-scaled FP8 helpers
# --------------------------------------------------------------------------
def dequant_blockscale(w_fp8: torch.Tensor, scale: torch.Tensor,
                       block: int = BLOCK) -> torch.Tensor:
    """Exact fp32 expansion of a `[N/128, K/128]`-block-scaled fp8 weight."""
    n, k = w_fp8.shape
    return (w_fp8.float().reshape(n // block, block, k // block, block)
            * scale.float()[:, None, :, None]).reshape(n, k)


def permute_fp8_blockscale_rows(w_fp8: torch.Tensor, scale: torch.Tensor,
                                dest_from_src: torch.Tensor,
                                block: int = BLOCK):
    """Row-permute a block-scaled fp8 weight, keeping the scale layout legal.

    THE COLLISION THIS SOLVES.  MPK's SM100 attention rotates NeoX pairs
    `(i, i + head_dim/2) = (i, i+128)` (`attention_sm100.cuh`'s fused rotary
    block); Qwen3.5 rotates `(j, j+32)` with `rotary_dim = 64`
    (`vllm-graph.md` §2.2.3). The load-time column permutation that reconciles
    them (`v1-architecture.md` §4.4) must therefore send the two members of
    every rotated pair — both of which live in the FIRST 64 columns of a head,
    i.e. inside ONE 128-row scale block — into DIFFERENT halves of the head,
    i.e. into different scale blocks. No permutation can avoid this: a scale
    block covers 128 rows and MPK's partner is exactly 128 rows away. So the
    shipped `[N/128, K/128]` block scale cannot simply be permuted alongside
    the rows (task 279's kernel hard-codes `BLOCK_N = 128`,
    `linear_fp8_blockscale_sm100.cuh:57`, `:120`).

    The resolution here keeps the fp8 path and confines the damage: each
    destination 128-row block takes `scale_out = max(contributing source
    scales)` per k-block, and every row whose source scale differs is rescaled
    by `ratio = s_src / s_out <= 1` (so e4m3 can never overflow) and re-rounded
    ONCE. Rows drawn from the max-scale source block come through BIT-EXACT.
    `stats` reports exactly how much was touched, and
    `test_loader_transforms.py` measures the resulting delta against the oracle
    next to the bf16-dequant alternative.

    Args:
        w_fp8: `[N, K]` float8_e4m3fn.
        scale: `[N/block, K/block]` float (the checkpoint's `weight_scale_inv`).
        dest_from_src: `[N]` int64 — destination row `i` takes source row
            `dest_from_src[i]`.

    Returns:
        `(w_out fp8 [N, K], scale_out fp32 [N/block, K/block], stats dict)`.
    """
    assert w_fp8.dtype == FP8, f"expected e4m3 weight, got {w_fp8.dtype}"
    n, k = w_fp8.shape
    assert n % block == 0 and k % block == 0
    assert tuple(scale.shape) == (n // block, k // block), (
        f"scale {tuple(scale.shape)} != block grid {(n // block, k // block)}")
    assert dest_from_src.shape == (n,)

    scale_f = scale.float()
    src_block = torch.div(dest_from_src, block, rounding_mode="floor")   # [N]
    per_row_scale = scale_f.index_select(0, src_block)                   # [N, K/128]
    scale_out = per_row_scale.reshape(n // block, block, k // block).amax(dim=1)
    ratio = per_row_scale / scale_out.repeat_interleave(block, dim=0)    # <= 1

    permuted = w_fp8.index_select(0, dest_from_src).float()
    scaled = (permuted.reshape(n, k // block, block)
              * ratio.unsqueeze(-1)).reshape(n, k)
    w_out = scaled.to(FP8)

    touched = (ratio != 1.0)
    exact = dequant_blockscale(w_fp8, scale_f).index_select(0, dest_from_src)
    got = dequant_blockscale(w_out, scale_out)
    denom = exact.norm().item()
    stats = {
        "rows": int(n),
        "row_kblock_entries": int(ratio.numel()),
        "rescaled_entries": int(touched.sum().item()),
        "rescaled_fraction": float(touched.float().mean().item()),
        "frob_rel_vs_exact_permute": float((got - exact).norm().item() / denom)
        if denom > 0 else 0.0,
        "max_abs_vs_exact_permute": float((got - exact).abs().max().item()),
        "min_ratio": float(ratio.min().item()),
    }
    return w_out.contiguous(), scale_out.contiguous(), stats


def permute_bf16_rows(w_fp8: torch.Tensor, scale: torch.Tensor,
                      dest_from_src: torch.Tensor) -> torch.Tensor:
    """The bf16-dequant alternative: exact dequant, permute, one bf16 rounding.

    Kept as the measured counter-candidate to
    `permute_fp8_blockscale_rows` and as the debugging scaffold
    `v1-architecture.md` §6.2 allows (never an acceptance configuration).
    """
    return (dequant_blockscale(w_fp8, scale)
            .index_select(0, dest_from_src).to(BF16).contiguous())


# --------------------------------------------------------------------------
# kv-group fusion (the packed QKVG row MPK's attention task reads)
# --------------------------------------------------------------------------
def _interleave_groups(tensors, num_groups: int) -> torch.Tensor:
    """`shuffle_tensors` semantics on dim 0, kept local so the loader has no
    dependency on a builder instance (`models/utils.shuffle_tensors`)."""
    chunks = []
    for g in range(num_groups):
        for t in tensors:
            per = t.shape[0] // num_groups
            chunks.append(t[g * per:(g + 1) * per])
    return torch.cat(chunks, dim=0).contiguous()


def fuse_qkvg(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              num_kv_heads: int) -> torch.Tensor:
    """Interleave `[q|gate]`-packed q, k and v into MPK's per-kv-group row.

    The SM100 attention task addresses its row as
    `[group0: (num_qo_per_kv x [q|gate]) k v][group1: ...]` — the layout the
    M2-I6 kernel test validated (`test_attention_qwen35_testmode.py:55-57`,
    and its reference's `group_w` arithmetic at `:128-145`), and the layout the
    layer's own width assert encodes
    (`persistent_kernel.py:988`: `(2*qo_per_kv + 2) * head_dim * num_kv_heads`).

    NOTE a documentation discrepancy worth carrying forward: `v1-architecture.md`
    §2.3 row 2 describes the fused tensor as a PLAIN concat
    `[16x[q|gate] || k(512) || v(512)]`. The kernel reads the kv-group
    interleave above; this function follows the kernel and the I6 test.
    """
    return _interleave_groups([q, k, v], num_kv_heads)


def fuse_qkvg_scale(sq: torch.Tensor, sk: torch.Tensor, sv: torch.Tensor,
                    num_kv_heads: int) -> torch.Tensor:
    """The same interleave applied to the three block-scale tensors.

    Exact: every group boundary is a whole number of 128-row scale blocks
    (q contributes `8*2*256/128 = 32` scale rows per group, k and v two each),
    so the scale permutation is block-aligned — unlike the head_dim RoPE
    permutation above.
    """
    for s, name in ((sq, "q"), (sk, "k"), (sv, "v")):
        assert s.shape[0] % num_kv_heads == 0, (
            f"{name} scale rows {s.shape[0]} not divisible by "
            f"{num_kv_heads} kv groups")
    return _interleave_groups([sq.float(), sk.float(), sv.float()],
                              num_kv_heads).contiguous()


def fuse_gate_up(gate: torch.Tensor, up: torch.Tensor,
                 num_groups: int) -> torch.Tensor:
    """Interleave a `gate_proj`/`up_proj` pair for `silu_mul_layer`.

    `silu_mul` reads `[gate_chunk | up_chunk]` pairs, so the split granularity
    must divide BOTH the weight rows and the scale rows; `num_groups =
    intermediate // 128` is the largest legal choice (the MoE-block test uses
    exactly this clamp, `test_moe_block_testmode.py:122-125`).
    """
    return _interleave_groups([gate, up], num_groups)


# --------------------------------------------------------------------------
# routed experts
# --------------------------------------------------------------------------
def stack_expert_w13(gate_list, up_list) -> torch.Tensor:
    """`[E, 2*inter, hidden]` fp8 with the checkpoint's `[gate; up]` packing.

    Rows `[0:inter)` are gate/w1 and `[inter:2*inter)` are up/w3 — PACKED, not
    interleaved (`vllm-graph.md` §2.3.4, `routed_experts.py:495-500`), which is
    what `moe_silu_mul` expects. Deliberately NOT vLLM's post-load
    W31/BlockMajorK shuffle (`v1-architecture.md` §2.0, `vllm-graph.md` §7
    row 11).
    """
    return torch.stack([torch.cat([g, u], dim=0)
                        for g, u in zip(gate_list, up_list)]).contiguous()


def stack_expert_w13_scale(gate_s_list, up_s_list) -> torch.Tensor:
    """`[E, 2*inter/128, hidden/128]` fp32 — UNEXPANDED.

    Tasks 241/242 (`moe_fp8_blockscale_layer`) consume `weight_scale_inv` in
    its SHIPPED shape and apply it in fp32 at the k-tile boundary
    (`persistent_kernel.py:1903-1930`). The `repeat_interleave(128)` per-row
    expansion belongs to the rejected `moe_w13/w2_fp8_layer` path, whose
    kernel floors the exponent (M2-I13 P2 verdict).
    """
    return torch.stack([torch.cat([g, u], dim=0).float()
                        for g, u in zip(gate_s_list, up_s_list)]).contiguous()


def stack_expert_w2(down_list) -> torch.Tensor:
    """`[E, hidden, inter]` fp8."""
    return torch.stack(list(down_list)).contiguous()


def stack_expert_w2_scale(down_s_list) -> torch.Tensor:
    """`[E, hidden/128, inter/128]` fp32 — UNEXPANDED (see above)."""
    return torch.stack([s.float() for s in down_s_list]).contiguous()
