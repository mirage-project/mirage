# `silu_mul` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** `o = silu(a[:,:I]) · a[:,I:]`, **dense** (dense MLP + shared expert). The MoE
routed activation is a **different layer** — [`moe_silu_mul`](./moe_silu_mul.md) (it adds an
active-expert `meta` mask + row-partition). This dense layer has neither.

**Phase:** both.

**grid_dim:** `(silu_grid, 1, 1)` — grid.x tiles columns; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `a` | `[T, 2I]` | bf16 | gate‖up concatenated |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `o` | `[T, I]` | bf16 | `silu(gate)·up` |

**Params:** none — `I` derived from the tensors.

**Shape variants**

| use | I | input |
|---|---|---|
| dense MLP (layers 0–2) | 2304 (`inter_dense/8`) | `[T, 2I]` |
| shared expert | 256 (`inter_moe/8`) | `[T, 2I]` |

## Python API
```python
def silu_mul_layer(
    self,
    a: DTensor,                    # [T,2I] bf16, gate‖up concatenated
    o: DTensor,                    # [T,I] bf16 out, silu(gate)·up
    grid_dim: tuple,
    block_dim: tuple = (256, 1, 1),
) -> None
```

**Reuse:** `silu_mul_layer` (registers one task). MoE routed silu is the separate
[`moe_silu_mul`](./moe_silu_mul.md).
