# `moe_silu_mul` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** `o = silu(a[:,:I]) · a[:,I:]` for MoE experts. **One layer serving both MoE paths**
(distinct from dense [`silu_mul`](./silu_mul.md) by the per-expert layout):
- **NEW grouped path** — `a` is 2D `[Mtot, 2I]` expert-contiguous (from grouped GEMM), with a
  `meta` active-expert mask (CTA early-returns for inactive-expert blocks); row-partitioned.
- **OLD per-expert path** — `a` is 3D `[T, Ep, 2I]` (per-token, per-selected-expert), **no `meta`**.

**Phase:** both.

**grid_dim:** NEW `(min(num_workers, Mtot), 1, 1)` (grid.x tiles permuted rows = experts);
OLD 3D partitions `(T, Ep)` rows; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `a` | NEW `[Mtot, 2I]` · OLD `[T, Ep, 2I]` | bf16 | gate‖up |
| `meta` (NEW only) | `[2, *]` | int32 | active-expert mask — CTA skips inactive-expert blocks |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `o` | NEW `[Mtot, I]` · OLD `[T, Ep, I]` | bf16 | `silu(gate)·up` |

**Params:** none — sizes derived; `meta` is an optional input.

**Shape variants**

| path | `a` shape | I | meta |
|---|---|---|---|
| NEW grouped | `[Mtot, 2I]`, `Mtot=16384` | 512 (`I_r`) | yes (active-mask) |
| OLD per-expert | `[T, Ep, 2I]`, `Ep=8` | 512 (`I_r`) | no |

## Python API
```python
def moe_silu_mul_layer(
    self,
    a: DTensor,                    # NEW [Mtot,2I] · OLD [T,Ep,2I] bf16, gate‖up
    o: DTensor,                    # NEW [Mtot,I] · OLD [T,Ep,I] bf16 out, silu(gate)·up
    grid_dim: tuple,
    block_dim: tuple = (256, 1, 1),
    *,
    meta: DTensor = None,          # [2,*] int32 active-expert mask (NEW grouped path only)
) -> None
```

**Reuse:** `moe_silu_mul_layer` (one task; selects 2D-vs-3D by input rank and `meta` presence).
