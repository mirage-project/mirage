# `quantize_fp8` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** per-(token, 128-group) absmax scale; `x_fp8 = round(x/scale)`.

**Phase:** both.

**grid_dim:** `(T,1,1) = (128,1,1)`; one CTA per token row; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `x` | `[T,D]` | bf16 | row-major; activations |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `x_fp8` | `[T,D]` | e4m3 | row-major; quantized activations |
| `scale` | `[T, D/128]` logical | **uint32 (UE8M0)** | **Required: packed UE8M0** — 4 exponent-only 8-bit scales per `uint32`, K-major (contiguous along K, strided by token; K axis = `ceil(K_groups/4)` words), batch padded to a tile multiple (`aligned_batch`). fp32 (one `float32`/group) is **optional, not required**. |

**Params:** `group_size=128`. Scale output is packed **UE8M0 uint32** (fp32 optional, not required).

**Shape variants / insertion points**

| insertion point | shape | scale fmt | notes |
|---|---|---|---|
| qkv_a in / o_proj in / MoE in | `[T, H=7168]` | UE8M0 | standalone in v1 (rmsnorm+quant unfused) |
| MoE intermediate (w13→w2) | `[Mtot, I_r=512]` | UE8M0 | between grouped GEMMs |
| q_nope (pre-BMM1) | `[T, H, 128]` per-head | UE8M0 | may fuse into q_b `*_fp8out` |
| attn_out (pre-BMM2) | `[T, H, 512]` per-head | UE8M0 | |
| attn_out_reduced (pre-o_proj) | `[T, H·128]` | UE8M0 | |
| q_b proj (= q_a slice) | `[T, 1536]` | UE8M0 | slice of `qkv_a`, **stride `[2112,1]`**, offset 0 |

**Tensor-view requirement (MUST):** several inputs are `mpk.narrow` column slices — e.g. the
`q_b`-input quantize reads `q_a` = cols `[0:1536)` of `qkv_a [T,2112]` (stride `[2112,1]`). The
kernel must load `D` columns per row via `stride[0]` + view offset, and form the per-128 scale
groups over the `D` **view**-columns only (not the parent width).

## Python API
```python
def quantize_fp8_layer(
    self,
    x: DTensor,               # [T,D] bf16 activations (may be an mpk.narrow column-slice view)
    x_fp8: DTensor,           # [T,D] e4m3, quantized activations (output)
    scale: DTensor,           # [T,D/128] logical, uint32 UE8M0 packed (output)
    grid_dim: tuple,          # (T,1,1), one CTA per token row
    block_dim: tuple = (256, 1, 1),
    *,
    group_size: int = 128,    # absmax group size
    scale_ue8m0: bool = True, # True: packed UE8M0 uint32 (required); False: fp32 (optional)
) -> None
```

**Reuse:** `quantize_fp8_layer`.

Remember that the actual scales are in ue8m0, and every 32 elements share one ue8m0. The group-size here is 128, consisting of 4 blocks (each block is of size 32 elements). And the blocks are the real quantization unit.