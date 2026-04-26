# `hc_post_layer`

Hyper-Connection **post** stage. Expands the single-stream sub-block output back into
`hc_mult` parallel copies, mixing it with the saved residual `hc_mult` copies via the
`post` and `comb` weights produced earlier by [`hc_pre_layer`](hc_pre_layer.md).

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Function: `Block.hc_post`, lines 684-687

## Math

```
y = post[..., None] * x[..., None, :] + sum(comb[..., None] * residual[..., None, :], dim=-3)
```

Concretely, with `H = hc_mult`, `D = dim`:

```
post: [B, H], comb: [B, H, H], x: [B, D], residual: [B, H, D]
y[b, h, d] = post[b, h] * x[b, d] + sum_{i} comb[b, i, h] * residual[b, i, d]
```

i.e. each output copy `h` is the sub-block output scaled by `post[h]` plus a
`comb`-weighted mixture of the H residual copies.

## Python API (proposed)

```python
pk.hc_post_layer(
    input_x:        DTensor,       # BF16 [B, D]      (sub-block output, e.g., grouped_lowrank_o_proj_layer's output or moe combine output)
    input_residual: DTensor,       # BF16 [B, hc_mult, D]   (residual_hc saved before hc_pre_layer)
    input_post:     DTensor,       # FP32 [B, hc_mult]      (from hc_pre_layer)
    input_comb:     DTensor,       # FP32 [B, hc_mult, hc_mult]
    output:         DTensor,       # BF16 [B, hc_mult, D]
    grid_dim: tuple,
    block_dim: tuple,
    hc_mult: int = 4,
)
```

## Inputs

| name | dtype | shape | source |
| --- | --- | --- | --- |
| `input_x` | BF16 | `[B, D]` | output of attention's `grouped_lowrank_o_proj_layer` or FFN's `moe_mul_sum_add_layer` |
| `input_residual` | BF16 | `[B, hc_mult, D]` | residual stream snapshot taken before `hc_pre_layer` |
| `input_post` | FP32 | `[B, hc_mult]` | from corresponding `hc_pre_layer` |
| `input_comb` | FP32 | `[B, hc_mult, hc_mult]` | from corresponding `hc_pre_layer` |

For V4-Flash-Base: `hc_mult=4`, `D=4096`.

## Outputs

| name | dtype | shape | sink |
| --- | --- | --- | --- |
| `output` | BF16 | `[B, hc_mult, D]` | next `hc_pre_layer` (FFN side) or next block's attn side |

## Builder usage

Called twice per block, immediately after the attention output projection and after the
MoE combine. See [`hc_pre_layer.md`](hc_pre_layer.md) for the surrounding sequence.

V3 analog: V3 does a fused `linear_with_residual_layer` for the attention side and an
`elementwise_add_layer`/`moe_mul_sum_add_layer` residual add for the FFN side. In V4, both
are replaced by `hc_post_layer`.

## Notes / risks

- The `D=4096` reduction along the H axis (size 4) is tiny; the kernel is bandwidth-bound on
  loading `residual` (`B * H * D` BF16 elements) and `x` (`B * D`).
- `comb` is FP32; convert to BF16 inside the kernel before the multiply to match the
  reference's `y.type_as(x)` cast.
- After the **last** block's FFN-side `hc_post_layer`, the resulting `[B, hc_mult, D]` feeds
  into [`hc_head_layer`](hc_head_layer.md) before the LM-head linear.
- Verification: same test-mode pattern as `hc_pre_layer`; compare to the reference
  `Block.hc_post` output.
