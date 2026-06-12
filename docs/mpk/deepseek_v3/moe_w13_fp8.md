# `moe_w13_fp8` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** OLD per-expert MoE path — fused **gate+up** projection. For each token, for each
of its top-k selected experts `e`, `[gate|up] = input · W13[e]ᵀ`. Operates per-token-per-selected-
expert via `moe_routing_indices`/`moe_mask` (no permutation into expert-contiguous order — that's
the NEW path's [`moe_permute`](./moe_permute.md) + [`grouped_gemm_fp8`](./grouped_gemm_fp8.md)).

**Phase:** both.

**grid_dim:** grid.y/z partition the `2·I` (gate+up) output columns; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `input_fp8` | `[T, H]` | e4m3 | row-major; quantized hidden |
| `input_scale` | `[T, H/128]` | **uint32 (UE8M0)** | per-128-K-group scale, packed 4/word |
| `weight_fp8` | `[E_loc, 2·I, H]` | e4m3 | per-expert gate+up weight |
| `weight_scale` | `[E_loc, 2·I, H/128]` | **uint32 (UE8M0)** | per-128-K-group weight scale |
| `moe_routing_indices` | `[E_loc, T]` | int32 | expert-major token lists (from [`moe_router`](./moe_router.md)) |
| `moe_mask` | `[E_loc+1]` | int32 | per-expert counts / active mask (**1-indexed**) |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `output` | `[T, Ep, 2·I]` | bf16 | per selected expert, gate‖up |

**Params:** none — sizes derived from the tensors. *(scales = **uint32 packed UE8M0**, 4/word; fp32 optional, not required.)*

**Shape variants**

| variant | E_loc | 2·I | H |
|---|---|---|---|
| config-fixed | 128 | 1024 (`=2·I_r`) | 7168 |

## Python API
```python
def moe_w13_fp8_layer(
    self,
    input_fp8: DTensor,            # [T,H] e4m3 quantized hidden
    input_scale: DTensor,          # [T,H/128] uint32 UE8M0 per-128-K-group scale
    weight_fp8: DTensor,           # [E_loc,2·I,H] e4m3 per-expert gate+up weight
    weight_scale: DTensor,         # [E_loc,2·I,H/128] uint32 UE8M0 weight scale
    moe_routing_indices: DTensor,  # [E_loc,T] int32 expert-major token lists
    moe_mask: DTensor,             # [E_loc+1] int32 per-expert counts (1-indexed)
    output: DTensor,               # [T,Ep,2·I] bf16 per selected expert, gate||up
    grid_dim: tuple,
    block_dim: tuple = (256, 1, 1),
) -> None
```

**Reuse:** `moe_w13_fp8_layer` (registers task `moe_w13_fp8_sm100`).

**Open:** the OLD path predates the EP-grouped design; confirm its `E_loc`/`Ep` semantics under
EP=2 (each rank computes its local experts; cross-rank routed contributions combine at
[`all_reduce`](./all_reduce.md)).
