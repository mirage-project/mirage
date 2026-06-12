# `moe_unpermute` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** scatter expert outputs back to token order, topk-weighted sum, add residual +
shared-expert. **Local only.**

**Phase:** both.

**grid_dim:** `(T, hidden_split, 1) ≈ (16,8,1)`; grid.x tiles token rows, grid.y tiles `H`; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `permuted_out` | `[Mtot,H]` | bf16 | grouped-GEMM(w2) output |
| `meta` | `[2,*]` | int32 | row→token + topk weights |
| `residual` | `[T,H]` | bf16 | pre-MoE hidden |
| `shared_out` (opt) | `[T,H]` | bf16 | shared-expert result |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `out` | `[T,H]` | bf16 | token-ordered routed sum + residual (+ shared); pre-`all_reduce` |

**Params:** none — sizes derived; `shared_out` is an optional input.

**Shape variants**

| variant | dims |
|---|---|
| config-fixed | `Mtot=16384, H=7168, Ep=8` |

## Python API
```python
def moe_unpermute_layer(
    self,
    permuted_out: DTensor,   # [Mtot,H] bf16 grouped-GEMM(w2) output
    meta: DTensor,           # [2,*] int32 row->token + topk weights
    residual: DTensor,       # [T,H] bf16 pre-MoE hidden
    out: DTensor,            # [T,H] bf16 routed sum + residual (+ shared)
    grid_dim: tuple,
    block_dim: tuple = (256, 1, 1),
    *,
    shared_out: DTensor = None,  # opt [T,H] bf16 shared-expert result
) -> None
```

**Reuse:** `moe_unpermute_sm100_layer`. (Shared expert =
[`linear_fp8`](./linear_fp8.md)→[`silu_mul`](./silu_mul.md)→`linear_fp8`, added here.)
