# `all_reduce` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** sum a tensor across all TP×EP ranks (NVSHMEM); optional fused residual at
final store. This is what synchronizes routed-expert (EP) and tensor-parallel (TP)
contributions — replacing explicit dispatch/combine.

**Phase:** both.

**grid_dim:** `(H/128,1,1) = (56,1,1)`; grid.x tiles `H`; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `x` | `[T,H]` | bf16 | local partial (post-o_proj or post-MoE) |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `y` | `[T,H]` | bf16 | globally summed |
| `buffer` | `[W=8,T,H]` | bf16 | scratch (per-rank staging) |

**Params:** `world_size`, `rank`.

**Shape variants**

| call site | shape |
|---|---|
| after o_proj | `[T, H=7168]` |
| after MoE / dense MLP | `[T, H=7168]` |

## Python API
```python
def all_reduce_layer(
    self,
    x: DTensor,              # [T,H] bf16 local partial
    y: DTensor,              # [T,H] bf16 globally summed
    buffer: DTensor,         # [W=8,T,H] bf16 scratch (per-rank staging)
    grid_dim: tuple,
    block_dim: tuple = (256, 1, 1),
    *,
    world_size: int,
    rank: int,
) -> None
```

**Reuse:** `allreduce_layer`.

**Special:** We don't need to support `residual` for allreduce layer for now.
