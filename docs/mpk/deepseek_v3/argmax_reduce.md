# `argmax_reduce` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** reduce the per-worker partials into the final greedy token — **stage 2**; follows
[`argmax_partial`](./argmax_partial.md). (Single-GPU path; the TP/sharded path uses
[`global_argmax`](./global_argmax.md) instead.)

**Phase:** both.

**grid_dim:** `(1, 1, 1)` — single CTA; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `value` | `[T, nt]` | bf16 | per-worker max values from `argmax_partial` |
| `index` | `[T, nt]` | int32 | per-worker argmax indices |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `token` | `[T, 1]` | int64 | final greedy token |

**Params:** none — `partial_output_size` derived from the inputs.

**Shape variants**

| variant | nt |
|---|---|
| single | `num_tasks` (from `argmax_partial`) |

## Python API
```python
def argmax_reduce_layer(
    self,
    value: DTensor,          # [T,nt] bf16 per-worker max values
    index: DTensor,          # [T,nt] int32 per-worker argmax indices
    token: DTensor,          # [T,1] int64 final greedy token
    grid_dim: tuple,
    block_dim: tuple = (256, 1, 1),
) -> None
```

**Reuse:** `argmax_reduce_layer`.
