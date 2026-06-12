# `moe_w2_fp8` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** OLD per-expert MoE path — **down** projection. For each token, for each of its
top-k selected experts `e`, `out = silu_out · W2[e]ᵀ`. Per-token-per-selected-expert via
`moe_routing_indices`/`moe_mask` (NEW path uses [`grouped_gemm_fp8`](./grouped_gemm_fp8.md) w2).

**Phase:** both.

**grid_dim:** grid.y/z partition the `H` output columns; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `input_fp8` | `[T, Ep, I]` | e4m3 | per-expert silu output, quantized ([`quantize_fp8`](./quantize_fp8.md)) |
| `input_scale` | `[T, Ep, I/128]` | **uint32 (UE8M0)** | per-128-K-group scale, packed 4/word |
| `weight_fp8` | `[E_loc, H, I]` | e4m3 | per-expert down weight |
| `weight_scale` | `[E_loc, H, I/128]` | **uint32 (UE8M0)** | per-128-K-group weight scale |
| `moe_routing_indices` | `[E_loc, T]` | int32 | expert-major token lists |
| `moe_mask` | `[E_loc+1]` | int32 | per-expert counts / active mask (1-indexed) |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `output` | `[T, Ep, H]` | bf16 | per selected expert, down-projected — combined by [`moe_mul_sum_add`](./moe_mul_sum_add.md) |

**Params:** none — sizes derived from the tensors. *(scales = **uint32 packed UE8M0**; fp32 optional, not required.)*

**Shape variants**

| variant | E_loc | I | H |
|---|---|---|---|
| config-fixed | 128 | 512 (`I_r`) | 7168 |

## Python API
```python
def moe_w2_fp8_layer(
    self,
    input_fp8: DTensor,            # [T,Ep,I] e4m3 per-expert silu output, quantized
    input_scale: DTensor,          # [T,Ep,I/128] uint32 UE8M0 per-128-K-group scale
    weight_fp8: DTensor,           # [E_loc,H,I] e4m3 per-expert down weight
    weight_scale: DTensor,         # [E_loc,H,I/128] uint32 UE8M0 weight scale
    moe_routing_indices: DTensor,  # [E_loc,T] int32 expert-major token lists
    moe_mask: DTensor,             # [E_loc+1] int32 per-expert counts (1-indexed)
    output: DTensor,               # [T,Ep,H] bf16 per selected expert, down-projected
    grid_dim: tuple,
    block_dim: tuple = (256, 1, 1),
) -> None
```

**Reuse:** `moe_w2_fp8_layer` (registers task `moe_w2_fp8_sm100`).
