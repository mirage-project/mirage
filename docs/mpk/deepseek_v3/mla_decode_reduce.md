# `mla_decode_reduce` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** log-sum-exp merge of split-KV partials → final latent attention output.

**Phase:** decode. (Prefill's absorbed attention is single-pass — no reduce.)

**grid_dim (compile-time — derived by the layer, NOT a caller argument):** mirrors decode; baked at
build from `decode_q_len`/`max_kv_len`. For **TP=8**:

```
num_groups = ceil(decode_q_len_even / 2)                    # same q-grouping as decode (qpg=2; even-padded)
rd_dv      = 2                                              # D_V (=512 latent) reduce factor
grid_dim   = (ceil(512 / rd_dv), num_groups, num_request)   # = (256, num_groups, 1); block (256,1,1)
```

`rd_dv` (= the **r**e**d**uce kernel's **`D_V`** tile — `RD_DV` in the kernel) = how many of the 512
latent output channels each CTA finalizes. So `grid.x = ceil(512 / rd_dv) = 256` CTAs, and the block
is `RD_TB = rd_dv × 128` threads — one thread per `(channel, row)`: `lane = tid // 128` picks the
channel (`d = blockIdx.x·rd_dv + lane`), `row = tid % 128` is the partial row (= q-token × head,
padded to 128). Each CTA log-sum-exp-merges its `rd_dv` channels across all
`num_splits = ceil(max_kv_len/128)` decode partials. **`rd_dv` is a compile-time tuning knob** (a
power of 2 with `RD_TB ≤ 1024` ⇒ `rd_dv ∈ {1,2,4,8}`); it must match across the kernel (`RD_DV`,
`RD_TB`) *and* the layer (`block_dim`, `grid.x`). Smaller `rd_dv` = more, smaller CTAs = more reduce
parallelism; **`rd_dv=2` benchmarked fastest** (`rd_dv=4` → 128 CTAs but slower). **Example:**
`decode_q_len=4`, `max_kv_len=4096` → `grid=(256, 2, 1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `attn_partial` | `[T,Hd,512,n_split]` | bf16 | per-split context |
| `lse` | `[T,Hd,n_split]` | f32 | per-split log-sum-exp |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `attn_out` | `[T,Hd,512]` | bf16 | head-major latent context (→ BMM2 in [`bmm_fp8`](./bmm_fp8.md)) |

**Params (all compile-time constants):** `tp_degree` (kernel variant), `decode_q_len`, `max_kv_len`
— must match the decode call that produced the partials. They bake the grid at build time (not
runtime lengths); `rd_dv=2` is internal (the D_V reduce factor).

**Shape variants**

| tp_degree | Hd |
|---|---|
| 1 | 128 |
| 2 | 64 |
| 4 | 32 |
| 8 (this config) | 16 |

## Python API
```python
def mla_decode_reduce_layer(
    self,
    attn_partial: DTensor,         # [T,Hd,512,n_split] bf16, per-split context
    lse: DTensor,                  # [T,Hd,n_split] f32, per-split log-sum-exp
    attn_out: DTensor,             # [T,Hd,512] bf16 out, head-major latent context (→ BMM2)
    decode_q_len: int,             # compile-time: same as the decode call (drives num_groups)
    max_kv_len: int,               # compile-time: KV capacity (= max_seq_length); drives num_splits
    *,
    tp_degree: int,                # selector → kernel variant (Hd=128/64/32/16)
) -> None
```
**No `grid_dim`/`block_dim` args** — the layer derives the grid (block `(256,1,1)`) from the
**compile-time** `decode_q_len`/`max_kv_len`/`tp_degree`; see the grid_dim section above.

**Tasks dispatched (by `tp_degree`)** — the one `mla_decode_reduce` layer registers one of these tasks:

| tp_degree | kernel |
|---|---|
| 1 | `mla_mtp_reduce_layer` |
| 2 | `mla_mtp_decode_tp2_reduce_layer` |
| 4 | `mla_mtp_decode_tp4_reduce_layer` |
| 8 (this config) | `mla_mtp_decode_tp8_reduce_layer` |
