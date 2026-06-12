# `moe_mul_sum_add` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** OLD per-expert MoE path — **combine**. Weighted sum over a token's selected
experts + residual: `out[t] = residual[t] + Σ_k weight[t,k] · input[t,k,:]`. This is the OLD-path
analog of the NEW path's [`moe_unpermute`](./moe_unpermute.md).

**Phase:** both.

**grid_dim:** `(MBT, hidden_split, 1)` — grid.x tiles token rows, grid.y tiles `H`; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `input` | `[T, Ep, H]` | bf16 | per-selected-expert outputs (from [`moe_w2_fp8`](./moe_w2_fp8.md)) |
| `weight` | `[T, Ep]` | f32 | top-k gate weights (from [`moe_router`](./moe_router.md)) |
| `residual` | `[T, H]` | bf16 | pre-MoE hidden |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `output` | `[T, H]` | bf16 | token-ordered routed sum + residual; pre-[`all_reduce`](./all_reduce.md) |

**Params:** none — sizes derived from the tensors.

**Shape variants**

| variant | Ep | H |
|---|---|---|
| config-fixed | 8 | 7168 |

## Python API
```python
def moe_mul_sum_add_layer(
    self,
    input: DTensor,          # [T,Ep,H] bf16 per-selected-expert outputs
    weight: DTensor,         # [T,Ep] f32 top-k gate weights
    residual: DTensor,       # [T,H] bf16 pre-MoE hidden
    output: DTensor,         # [T,H] bf16 routed sum + residual
    grid_dim: tuple,
    block_dim: tuple = (256, 1, 1),
) -> None
```

**Reuse:** `moe_mul_sum_add_layer` (registers task `moe_mul_sum_add_sm100`).

*Note:* shared-expert output is added separately (NEW path folds it into `moe_unpermute`; the OLD
path adds it alongside this combine).
