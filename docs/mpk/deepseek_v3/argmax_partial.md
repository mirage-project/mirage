# `argmax_partial` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** per-worker **partial** argmax over a vocab shard — **stage 1** of greedy token
selection; reduced by [`argmax_reduce`](./argmax_reduce.md).

**Phase:** both.

**grid_dim:** `(num_workers, 1, 1) = (128,1,1)` — grid.x splits the vocab; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `logits` | `[T, Vloc]` | bf16 | row-major; `Vloc=V` (replicated) or `V/8` (TP-sharded) |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `value` | `[T, nt]` | bf16 | per-worker max value |
| `index` | `[T, nt]` | int32 | per-worker argmax index |

**Params:** none — `num_tasks` derived from the output shape.

**Shape variants**

| variant | Vloc | notes |
|---|---|---|
| full vocab (replicated) | 129280 | single-GPU |
| vocab-sharded | 16160 (`V/8`) | partials also feed [`global_argmax`](./global_argmax.md) |

## Python API
```python
def argmax_partial_layer(
    self,
    logits: DTensor,         # [T,Vloc] bf16; Vloc=V (replicated) or V/8 (TP-sharded)
    value: DTensor,          # [T,nt] bf16 per-worker max value
    index: DTensor,          # [T,nt] int32 per-worker argmax index
    grid_dim: tuple,
    block_dim: tuple = (256, 1, 1),
) -> None
```

**Reuse:** `argmax_partial_layer`.
