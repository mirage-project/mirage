# `linear` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** `C = A·Wᵀ`, bf16 (no residual). Plain dense GEMM.

**Phase:** both.

**grid_dim:** `(N/128, 1, 1)`; block `(256,1,1)` — grid.x tiles output cols.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `A` | `[M,K]` | bf16 | row-major activation (M=T) |
| `W` | `[N,K]` | bf16 | row-major weight |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `C` | `[M,N]` | bf16 | row-major |

**Params:** none — sizes derived from the tensors. **TP-agnostic** (plain `C = A·Wᵀ`).

**Shape variants**

| role | K | N | notes |
|---|---|---|---|
| lm_head (vocab-sharded) | 7168 | `V/8=16160` | column → [`global_argmax`](./global_argmax.md) |
| lm_head (replicated) | 7168 | `V=129280` | → [`argmax_partial`](./argmax_partial.md) |

## Python API
```python
def linear_layer(
    self,
    A: DTensor,               # [M,K] bf16, row-major activation (M=T)
    W: DTensor,               # [N,K] bf16, row-major weight
    C: DTensor,               # [M,N] bf16, row-major (output)
    grid_dim: tuple,          # (N/128,1,1)
    block_dim: tuple = (256, 1, 1),
) -> None
```

**Reuse:** `linear_layer`. *(DSv3 bf16 use = lm_head; everything else is FP8 — see [`linear_fp8`](./linear_fp8.md). For very large `N`, lm_head may instead use [`splitk_linear`](./splitk_linear.md).)*
