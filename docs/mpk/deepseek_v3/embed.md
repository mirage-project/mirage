# `embed` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** `hidden[t] = embed_table[token_ids[t]]`.

**Phase:** both.

**grid_dim:** `(H/128, 1, 1) = (56,1,1)`, block `(256,1,1)` — grid.x tiles `H`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `token_ids` | `[T]` | int64 | row-major; token id per position |
| `embed_table` | `[V,H]` | bf16 | row-major; vocab embedding matrix |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `hidden` | `[T,H]` | bf16 | row-major; per-token embedding |

**Params:** none — sizes (`H`, `V`) derived from the tensors.

**Shape variants**

| variant | dims |
|---|---|
| only | `V=129280, H=7168` |

## Python API
```python
def embed_layer(
    self,
    token_ids: DTensor,       # [T] int64, token id per position
    embed_table: DTensor,     # [V,H] bf16, vocab embedding matrix
    hidden: DTensor,          # [T,H] bf16, per-token embedding (output)
    grid_dim: tuple,          # (H/128,1,1)
    block_dim: tuple = (256, 1, 1),
) -> None
```

**Reuse:** `embed_layer`.
