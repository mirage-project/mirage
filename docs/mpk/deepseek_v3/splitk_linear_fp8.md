# `splitk_linear_fp8` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** `C = A·Wᵀ (+accumulate)` via **split-K**, FP8 e4m3 block-scaled. For decode
GEMMs with a large K where a single K-pass underfills the GPU (e.g. the absorbed o-down).
Distinct layer from the dense [`linear_fp8`](./linear_fp8.md) (split-K decomposition + `accumulate`).

**Phase:** decode.

**grid_dim:** variant-dependent (see Implementation); block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `A_fp8` | `[M,K]` | e4m3 | row-major; quantized activation |
| `A_scale` | — | uint32 (UE8M0) | K-major per-128-group scale (see [`quantize_fp8`](./quantize_fp8.md)) |
| `W_fp8` | `[N,K]` | e4m3 | row-major; quantized weight |
| `W_scale` | — | uint32 (UE8M0) | per-128-block scale |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `C` | `[M,N]` | bf16 | row-major (pre-zeroed before accumulation) |

**Params:** `split_k`, `accumulate`. (sizes/scale derived from the tensors.)

**Shape variants**

| role | K | N |
|---|---|---|
| decode o_proj / down (split-K) | large (≥1024) | 7168 |

## Python API
```python
def splitk_linear_fp8_layer(
    self,
    input_fp8: DTensor,       # [M,K] e4m3, quantized activation
    input_scale: DTensor,     # uint32 UE8M0, K-major per-128-group
    weight_fp8: DTensor,      # [N,K] e4m3, quantized weight
    weight_scale: DTensor,    # uint32 UE8M0, per-128-block scale
    output: DTensor,          # [M,N] bf16, row-major (pre-zeroed before accumulation)
    grid_dim: tuple,          # variant-dependent ((M_shards,split_k,1) or (num_workers,1,1))
    block_dim: tuple = (256, 1, 1),
    *,
    split_k: int,
    accumulate: bool,         # True: add onto caller's output; False: prepend tensor_init zero-fill
) -> None
```

**Tasks dispatched (by variant)** — the one `splitk_linear_fp8` layer registers one of these tasks (different grids → Python-level L1 select):

| variant | kernel | grid | reduce |
|---|---|---|---|
| TMA reduce-add (UE8M0 — required format) | `linear_splitk_swapAB_fp8_layer` | `(M_shards, split_k, 1)` | grid.y K-slices, TMA reduce |
| atomic-accumulate (fp32 scale — optional) | `fp8_gemm_dense_decode_splitk_layer` | `(num_workers,1,1)` + prepended `tensor_init` | in-kernel K-split, atomic bf16 add |
