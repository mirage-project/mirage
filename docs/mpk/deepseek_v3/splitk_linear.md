# `splitk_linear` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** `C = A·Wᵀ` via **split-K accumulation**, bf16. The K axis is split across
`grid.y` CTAs that reduce-add into the shared output tile. For small-M (decode) GEMMs where a
single K-pass underfills the GPU.

**Phase:** both (decode-oriented).

**grid_dim:** `(N/128, split_k, 1)`; grid.y = K-slices; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `A` | `[M,K]` | bf16 | row-major activation (M=T) |
| `W` | `[N,K]` | bf16 | row-major weight |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `C` | `[M,N]` | bf16 | row-major (pre-zeroed when `accumulate=False`) |

**Params:** `split_k`, `accumulate`. (sizes derived from the tensors.)

**Shape variants**

| role | notes |
|---|---|
| small-M bf16 decode GEMM | distinct from [`linear`](./linear.md) by the split-K decomposition + `accumulate` |

## Python API
```python
def splitk_linear_layer(
    self,
    A: DTensor,               # [M,K] bf16, row-major activation (M=T)
    W: DTensor,               # [N,K] bf16, row-major weight
    C: DTensor,               # [M,N] bf16, row-major (output; pre-zeroed when accumulate=False)
    grid_dim: tuple,          # (N/128, split_k, 1); grid.y = K-slices
    block_dim: tuple = (256, 1, 1),
    *,
    split_k: int,             # number of K-slices (grid.y)
    accumulate: bool,         # True: add onto caller's output; False: prepend tensor_init zero-fill
) -> None
```

**Reuse:** `splitk_linear_layer`. Prepends a `tensor_init` (zeros the output) when `accumulate=False`.
