# `mla_decode_attn` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** latent-space **absorbed** MLA attention vs the contiguous KV cache, split over KV
length; emits per-split partials + log-sum-exp (→ [`mla_decode_reduce`](./mla_decode_reduce.md)).
**Decode only** (tiny per-step query budget `decode_q_len` 1–8, huge `max_kv_len` → parallelize over KV). Prefill uses the q-tiled
[`mla_prefill_attn`](./mla_prefill_attn.md) — same absorbed math + compressed cache + BMM1/BMM2, but
a different work distribution (kv-split + reduce here vs q-tiled flash there), so a separate layer.

**Phase:** decode.

**grid_dim (compile-time — derived by the layer, NOT a caller argument):** the layer bakes the grid
into the task graph at build time from the **compile-time constants** `decode_q_len` / `max_kv_len` /
`tp_degree`. For **TP=8** (this config):

```
q_len      = (decode_q_len + 1) & ~1                        # pad per-step query budget to even (decode_q_len 1–8 → 2/4/6/8)
num_groups = ceil(q_len / 2)                                # qpg = 2 queries per CTA-group (TP=8)
num_splits = num_splits_override or ceil(max_kv_len / 128)  # KV tile = 128; max_kv_len = max_seq_length
grid_dim   = (num_groups * num_splits, num_request, 1)      # block_dim = (128, 1, 1)
```

**`grid.x` packs `(q-group, KV-split)`** — the kernel recovers `gi = blockIdx.x // num_splits`
(q-group) and `si = blockIdx.x % num_splits` (KV split). So **KV *is* split across CTAs**, just
folded into `grid.x` rather than given its own axis (MPK dispatches tasks by flat linear index).
Each CTA processes `qpg=2` query tokens **× all `NUM_HEADS=16` heads = 32 MMA rows** (heads are
batched into the same MMA — it is *not* 2 rows of work), over its KV chunk of
`ceil(kv_tiles / num_splits)` 128-token tiles (default `num_splits = kv_tiles` ⇒ **1 tile/CTA**, max
KV parallelism). It emits a partial output + LSE → log-sum-exp-merged by
[`mla_decode_reduce`](./mla_decode_reduce.md) across the `num_splits` partials.
`num_request = max_num_batched_requests` (= 1 here).
**Example:** `decode_q_len=4`, `max_kv_len=4096` → `num_groups=2`, `num_splits=32` → `grid=(64,1,1)`.
(TP=1/2/4 differ: `qpg=min(2or4,decode_q_len)` + head-group / V-split factors folded into `grid.x`;
only TP=8 is needed for v1.)

**Compile-time vs runtime (important):** `decode_q_len` and `max_kv_len` are **build-time
constants** — the grid is fixed when the megakernel is compiled. `decode_q_len` is the per-step
query-token *budget* (1 = plain decode; the MTP/speculative verify width otherwise);
`max_kv_len = max_seq_length` sizes `num_splits` for the worst case. The **actual** current sequence
length and the **actual** number of accepted MTP tokens are *runtime* values the kernel reads from
meta-tensors and masks against — they never resize the grid.

**Block-size note:** the reused MTP decode kernel runs **128 threads/CTA**, *not* the catalog's
`(256,1,1)` default — a fixed property of that kernel.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `q` | `[T,Hd,576]` | bf16 | head-major; absorbed query `[latent\|rope]`, assembled **in-place** via views (BMM1 → `[:,:,:512]`, roped `q_pe` → `[:,:,512:576]`); TMA |
| `kv` | `[L,576]` | bf16 | **contiguous** latent KV cache `[0:seq_len]` (no paging — num_request=1, single sequence) |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `attn_partial` | `[T,Hd,512,n_split]` | bf16 | per-split context |
| `lse` | `[T,Hd,n_split]` | f32 | per-split log-sum-exp |

**Params (all compile-time constants):** `tp_degree` (kernel variant), `decode_q_len` (per-step
query-token budget, 1–8), `max_kv_len` (= `max_seq_length`, sizes `num_splits`),
`num_splits_override` (optional). All bake into the grid at build time — **none is a runtime
sequence length** (the live length is read from meta-tensors inside the kernel). `n_split` equals
`num_splits` (also derivable from `attn_partial`'s last dim).

**Requirement (MTP):** `decode_q_len` is a **compile-time** value in **1–8** (1 = plain decode; the
MTP/speculative verify width otherwise — per request, even at num_request=1). The kernel must be
buildable and correct for any width in that range; for a given build it is one fixed constant. Large
prefill query counts are **not** this kernel's job — see [`mla_prefill_attn`](./mla_prefill_attn.md).
No `qo_indptr`/paging — single sequence.

**Shape variants**

| tp_degree | Hd |
|---|---|
| 1 | 128 |
| 2 | 64 |
| 4 | 32 |
| 8 (this config) | 16 |

## Python API
```python
def mla_decode_attn_layer(
    self,
    q: DTensor,                    # [T,Hd,576] bf16, absorbed query [latent|rope]
    kv: DTensor,                   # [L,576] bf16, contiguous latent KV cache [0:seq_len] (no paging)
    attn_partial: DTensor,         # [T,Hd,512,n_split] bf16 out, per-split context
    lse: DTensor,                  # [T,Hd,n_split] f32 out, per-split log-sum-exp
    decode_q_len: int,             # compile-time: per-step query-token budget (1 plain / MTP width 1–8)
    max_kv_len: int,               # compile-time: KV capacity (= max_seq_length); sizes num_splits
    *,
    tp_degree: int,                # selector → kernel variant (Hd=128/64/32/16)
    num_splits_override: int | None = None,
) -> None
```
**No `grid_dim`/`block_dim` args** — the layer derives the grid (block `(128,1,1)`) from the
**compile-time** `decode_q_len`/`max_kv_len`/`tp_degree`; see the grid_dim section above.

**Tasks dispatched (by `tp_degree`)** — the one `mla_decode_attn` layer registers one of these tasks (over the `num_heads`-templated kernel):

| tp_degree | kernel |
|---|---|
| 1 | `mla_mtp_decode_layer` |
| 2 | `mla_mtp_decode_tp2_layer` |
| 4 | `mla_mtp_decode_tp4_layer` |
| 8 (this config) | `mla_mtp_decode_tp8_layer` |

**Special:** For this layer, we only need to make sure it runs successfully under TP=8 setup. You could silumate the TP=8 dimension and setup on a single GPU for testing.
