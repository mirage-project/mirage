# `grouped_lowrank_o_proj_layer`

V4's output projection. Replaces V3's single `o_proj` linear `[n_heads * v_head_dim → dim]`
with a **grouped low-rank** factorization: per-group rank-`o_lora_rank` matrices, then a
shared row-parallel linear back to `dim`.

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Function: `Attention.forward`, lines 537-542
- Constructor: `Attention.__init__` lines 462-463
- Snippet:
  ```python
  o = o.view(bsz, seqlen, n_local_groups, -1)
  wo_a = wo_a.weight.view(n_local_groups, o_lora_rank, -1)
  o = einsum("bsgd,grd->bsgr", o, wo_a)
  x = wo_b(o.flatten(2))
  ```

## Math

With `H = n_local_heads` (=64), `Dh = head_dim` (=512), `G = o_groups` (=8),
`r = o_lora_rank` (=1024), `D = dim` (=4096):

```
o    shape : [B, S, H * Dh]                       # sparse_attn output
o    view  : [B, S, G, (H*Dh)/G]                  # per-group split along the last dim
wo_a view  : [G, r, (H*Dh)/G]                     # per-group rank-r matrix
o_lr       : einsum("bsgd, grd -> bsgr", o, wo_a) # [B, S, G, r]   per-group low-rank lift
o_lr_flat  : o_lr.flatten(start=2)                # [B, S, G*r]
y          : wo_b(o_lr_flat)                      # [B, S, D]      shared row-parallel linear
```

Effectively a two-stage projection: per-group "expand" via `wo_a`, then shared "contract" via
`wo_b`. With `G=8`, `r=1024`, `H*Dh=64*512=32768`, the intermediate dim is `G*r=8192`.

## Python API (proposed)

Two layers in sequence, chosen so each maps to existing MPK GEMM kernels rather than a
brand-new fused kernel:

```python
# Stage A: per-group "wo_a" einsum (treat as G independent linear layers in batch)
pk.grouped_lowrank_o_proj_a_layer(
    input:  DTensor,   # BF16 [B, n_groups, (n_heads*head_dim)/n_groups]
    weight: DTensor,   # FP8  [n_groups, o_lora_rank, (n_heads*head_dim)/n_groups]
    weight_scale: DTensor,  # FP32 (only if FP8)
    output: DTensor,   # BF16 [B, n_groups * o_lora_rank]
    grid_dim: tuple,
    block_dim: tuple,
)

# Stage B: shared "wo_b" linear with residual fusion (HC residual is taken at hc_post_layer)
pk.linear_fp8_layer(
    input=stage_a_out,                       # [B, n_groups * o_lora_rank]
    input_scale=...,
    weight=wo_b_fp8,                          # [dim, n_groups * o_lora_rank]
    weight_scale=wo_b_scale,
    output=...,
    ...,
)                                             # reuse V3
```

Alternatively, a single fused `grouped_lowrank_o_proj_layer` that does A then B in one task —
preferred once stage A is mature.

## Inputs

| name | dtype | shape | source |
| --- | --- | --- | --- |
| stage A `input` | BF16 | `[B, n_groups, (n_heads * head_dim) / n_groups]` | `sparse_attn_*_layer`'s output reshape |
| stage A `weight` | FP8 e4m3 | `[n_groups, o_lora_rank, (n_heads * head_dim) / n_groups]` | `attn.wo_a.weight.view(...)` |
| stage A `weight_scale` | FP8 ue8m0 | `[n_groups, ceil(o_lora_rank/128), ceil((H*Dh/G)/128)]` | corresponding scale tensor |
| stage B `input` | BF16 | `[B, n_groups * o_lora_rank]` | stage A output flattened |
| stage B `weight` | FP8 | `[dim, n_groups * o_lora_rank]` | `attn.wo_b.weight` |

For V4-Flash-Base: `n_heads=64`, `head_dim=512`, `n_groups=8`, `o_lora_rank=1024`,
`dim=4096`. Per-group K dim = `(64*512)/8 = 4096`.

## Outputs

| name | dtype | shape | sink |
| --- | --- | --- | --- |
| stage A `output` | BF16 | `[B, n_groups * o_lora_rank]` (= `[B, 8*1024]` = `[B, 8192]`) | stage B input |
| stage B `output` | BF16 | `[B, dim]` (= `[B, 4096]`) | input to the attn-side `hc_post_layer` |

## Builder usage

Called once per block, immediately after `sparse_attn_*_layer`:

```python
o = pk.sparse_attn_*_layer(...)                                     # [B, n_heads * head_dim]
o = pk.grouped_lowrank_o_proj_a_layer(o.view(...), wo_a, ...)        # [B, n_groups * o_lora_rank]
y = pk.linear_fp8_layer(o, wo_b, ...)                                # [B, dim]
x_hc = pk.hc_post_layer(y, residual_hc, post_attn, comb_attn, ...)   # NEW (no residual here, residual is the saved hc copies)
```

V3 analog: V3 collapses all of this into a single `linear_fp8_with_residual_layer` for
`o_proj` (with residual fusion). V4 cannot do residual fusion at `wo_b` because the
residual addition is part of `hc_post_layer`, not a simple add.

## Notes / risks

- The reference comment (model.py:539-541) notes `wo_a is FP8 in checkpoint; could do FP8
  einsum here for better perf, but using BF16 for simplicity.` In MPK we should run the
  per-group einsum in **FP8** to match V3's quant convention; this means we need a
  **batched FP8 GEMM** (`G` independent matmuls, each `[B, K] x [K, r]`), not a flat
  `[B, G*K] x [G*K, G*r]`. The kernel design must reflect that — implement either as a
  group-batched FP8 GEMM (similar to MoE's group GEMM but with a fixed group axis) or
  loop the existing `linear_fp8_layer` `G` times. For the first pass the loop is fine;
  fuse later.
- `wo_b` is row-parallel under TP — for `world_size > 1` it would need an `allreduce_layer`
  afterward. **First pass uses world_size=1**, so no allreduce.
- Stage A is the new bit; stage B is identical in shape to a regular V3 `o_proj`-style
  linear and can reuse `linear_fp8_layer`. No new task type needed for stage B.
- Verification: numerical match with the V4 reference einsum + linear, BF16 tolerance.
