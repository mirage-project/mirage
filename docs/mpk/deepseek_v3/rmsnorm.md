# `rmsnorm` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** `y = x / sqrt(mean(x²)+ε) · weight`, ε=1e-6. The mean/reduction is over the
**`D` (last) axis only** — i.e. over the `D` view-columns, not the parent buffer width.

**Phase:** both.

**grid_dim:** `(ceil(T/rows_per_task), 1, 1) ≈ (128,1,1)` for `T=128`; grid.x tiles token rows; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `x` | `[T,D]` | bf16 | tokens to normalize; **often a column-slice view** of a wider buffer (see Tensor-view requirement) |
| `weight` | `[D]` | bf16 | per-channel gain |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `y` | `[T,D]` | bf16 | normalized tokens; **also may be a column-slice view** (in-place, or into a wider buffer) |

**Params:** none — `D` derived from the tensor shape.

**Tensor-view requirement (MUST):** `x` and `y` are frequently **`mpk.narrow` column slices**
of a wider parent buffer, so the kernel **must address rows by the tensor's `stride[0]` (+ view
offset), never assume contiguous `D`-wide rows.** Canonical DSv3 case — the q_a-norm and kv_a-norm
normalize slices of the fused `qkv_a` output `[T, 2112]` = `[q_a 1536 | c_latent 512 | k_pe 64]`:

- **kv_a-norm**: `x = [T, 512]` with **`stride = [2112, 1]`**, column offset `1536`.
- **q_a-norm**: `x = [T, 1536]` with `stride = [2112, 1]`, column offset `0`.

The runtime sets the per-task base pointer from the view's offset; the kernel reads/writes `D`
contiguous elements per row and advances by `stride[0]` between rows. Reducing over the parent
width (2112) instead of `D` (512) would be a correctness bug.

**Shape variants**

| role | D | x layout |
|---|---|---|
| input / post-attn / final norm | 7168 (`H`) | full `[T,H]`, contiguous |
| q_a-norm | 1536 (`q_lora`) | slice of `qkv_a [T,2112]`, stride 2112, offset 0 |
| kv_a-norm | 512 (`kv_lora`) | slice of `qkv_a [T,2112]`, stride 2112, offset 1536 |

## Python API
```python
def rmsnorm_layer(
    self,
    x: DTensor,               # [T,D] bf16, tokens (often an mpk.narrow column-slice view)
    weight: DTensor,          # [D] bf16, per-channel gain
    y: DTensor,               # [T,D] bf16, normalized tokens (may be in-place / slice view)
    grid_dim: tuple,          # (ceil(T/rows_per_task),1,1)
    block_dim: tuple = (256, 1, 1),
) -> None
```

**Reuse:** `rmsnorm_layer`. 