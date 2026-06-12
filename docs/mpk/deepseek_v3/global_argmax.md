# `global_argmax` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** reduce per-rank partial `(value, index)` into the global argmax token (NVSHMEM).

**Phase:** both.

**grid_dim:** small (per-token reduce); block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `partial_value` | `[T,*]` | bf16 | per-rank partial max value |
| `partial_index` | `[T,*]` | int32 | per-rank argmax index |
| scratch | — | — | NVSHMEM staging |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `token` | `[T,1]` | int64 | global greedy token |

**Params:** `world_size`, `vocab_offset`, `valid_vocab_size`.

**Shape variants**

| world_size | notes |
|---|---|
| 2 / 4 / 8 | reduce across ranks; this config = 8 |

## Python API
```python
def global_argmax_layer(
    self,
    partial_value: DTensor,  # [T,*] bf16 per-rank partial max value
    partial_index: DTensor,  # [T,*] int32 per-rank argmax index
    token: DTensor,          # [T,1] int64 global greedy token
    grid_dim: tuple,
    block_dim: tuple = (256, 1, 1),
    *,
    world_size: int,
    vocab_offset: int,
    valid_vocab_size: int,
) -> None
```

**Reuse:** `nvshmem_global_argmax_layer`.

**Open:** needed only if lm_head logits are vocab-sharded across ranks; confirm whether
deterministic greedy uses sharded lm_head (→ this) or replicated logits
(→ [`argmax_partial`](./argmax_partial.md) → [`argmax_reduce`](./argmax_reduce.md) only).
