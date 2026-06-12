# `rope` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** apply rotary embedding to the decoupled rope dims, in-place. **One tensor-driven
kernel serves both Q and K** — the rotation math is identical (matches the official single
`apply_rotary_emb`, applied to both `q_pe` and `k_pe` with the same `freqs_cis`; `model.py:187,239,242`).
The layer rotates the rope dims of whatever tensor it's handed, iterating over that tensor's head
dim — **no `role` parameter**: head count and strides come from the tensor's shape/stride.

**Phase:** both.

**grid_dim:** caller-sized from the tensor (token-tiles × heads); block `(256,1,1)`. No tensor
partition — each CTA rotates its slice.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `x` | `[T, …, 64]` | bf16 | the rope-dim tensor (Q-rope `[T,H,64]` or K-rope `[T,1,64]`); usually an `mpk.narrow` view of a wider buffer; rotated **in-place** |
| `cos` | `[T, 64]` | bf16 | rotary cos table |
| `sin` | `[T, 64]` | bf16 | rotary sin table |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `x` | `[T, …, 64]` | bf16 | rotated **in-place** |

**Params:** none — head count + strides derived from `x` (the rope-dim tensor).

**Tensor-view requirement (MUST):** `x` is an `mpk.narrow` slice of a wider buffer — e.g. `k_pe`
= cols `[2048:2112)` of `qkv_a [T,2112]` (stride `[2112,1]`, offset 2048); the decode `q_pe` =
`q[:,:,512:576]`. The kernel rotates **in place via the view's `stride[0]` + offset**; it must not
assume the rope dims are contiguous across rows.

**Shape variants**

| case | shape | notes |
|---|---|---|
| Q-rope | `[T, Hd, 64]` | per-head; `Hd=16` this config |
| K-rope | `[T, 1, 64]` | single head, shared across heads (MQA-style) |

## Fusion plan — one tensor-driven kernel

`rope_q_fused`, `rope_q_split`, and `rope_k` **fold into a single `rope_layer(x, cos, sin)`**. The
CUDA kernel is *already* one templated device function (`kernel::deepseek_mla_rope_sm100_task_impl`),
all variants registering under the **same `TASK_DEEPSEEK_MLA_ROPE_SM100`**. The rotation math is
shared (confirmed against the official `apply_rotary_emb` — Q and K rope are the same op); the only
per-variant difference is **addressing**, which is **derived from `x`'s shape/stride**, not a
`role` param. The addressing tuple `(row_stride, head_stride, rope_offset_in_row, num_heads)`:

| case | `x` layout | (row_stride, head_stride, rope_offset, num_heads) |
|---|---|---|
| Q fused | `[T,H,576]`, rope in tail | `(H·576, 576, 512, H)` |
| Q split | `[T,H,64]` standalone | `(H·64, 64, 0, H)` |
| K | `[T,64]` shared | `(stride, 0, 0, 1)` |

Every entry is read from `x`'s `stride()`/shape — so **one kernel, no `role` / `DO_Q` / `DO_K`
branch**.

**Target:** collapse the 4 `register_deepseek_mla_rope_*` fns + the `*_layer` methods into one
`rope_layer(x, cos, sin)` + one register fn — no `role`/`num_heads`/stride args are passed (the
kernel reads the tuple from `x`). `phase_gate` (decode/prefill `Q_LEN` early-return) stays a
register-codegen option, not a kernel param.

## Python API
```python
def rope_layer(
    self,
    x: DTensor,               # [T,…,64] bf16, Q/K rope-dim tensor; rotated in-place (also the output)
    cos: DTensor,             # [T,64] bf16, rotary cos table
    sin: DTensor,             # [T,64] bf16, rotary sin table
    grid_dim: tuple,
    block_dim: tuple = (256, 1, 1),
) -> None
```

**Reuse:** single templated kernel `kernel::deepseek_mla_rope_sm100_task_impl` (exists today);
wrap with one tensor-driven `rope_layer(x, cos, sin)` (no `role`/stride params) replacing
`deepseek_mla_rope_q_fused_layer` / `…_q_split_layer` / `…_k_layer`.
