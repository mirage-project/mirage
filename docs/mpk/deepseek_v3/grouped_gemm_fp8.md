# `grouped_gemm_fp8` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** grouped block-scaled FP8 GEMM; permuted row group `g` uses local-expert weight `b[g]`.

**Phase:** both.

**grid_dim:** `(num_workers,1,1)`, persistent tile distribution (smallm/largem auto-select);
block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `a_fp8` | `[Mtot,K]` | e4m3 | permuted activations |
| `b_fp8` | `[E_loc,N,K]` | e4m3 | per-local-expert weights |
| `a_scale` | — | uint32 (UE8M0) | block scales |
| `b_scale` | — | uint32 (UE8M0) | block scales |
| `m_indices` | `[Mtot]` | int32 | row→expert map |
| `meta` (opt) | — | int32 | active-mask |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `C` | `[Mtot,N]` | bf16 | grouped result |

**Params:** none — sizes derived; `meta` is an optional input.

**Shape variants** (`E_loc=128` local experts; `Mtot=16384`):

| role | K | N |
|---|---|---|
| w13 (gate+up) | 7168 | 1024 (`=2·I_r`) |
| w2 (down) | 512 (`I_r`) | 7168 |

## Python API
```python
def grouped_gemm_fp8_layer(
    self,
    a_fp8: DTensor,          # [Mtot,K] e4m3 permuted activations
    b_fp8: DTensor,          # [E_loc,N,K] e4m3 per-local-expert weights
    a_scale: DTensor,        # uint32 UE8M0 block scales
    b_scale: DTensor,        # uint32 UE8M0 block scales
    m_indices: DTensor,      # [Mtot] int32 row->expert map
    C: DTensor,              # [Mtot,N] bf16 grouped result
    grid_dim: tuple,
    block_dim: tuple = (256, 1, 1),
    *,
    meta: DTensor = None,    # opt int32 active-mask
) -> None
```

**Tasks dispatched (by M-per-expert)** — the one `fp8_group_gemm_layer` layer **dispatches
internally** to one of these tasks: `fp8_group_gemm_smallm_sm100` / `fp8_group_gemm_largem_compact_sm100`
(small vs large M-per-expert).
