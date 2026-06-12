# `mla_prefill_attn` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** causal MLA attention in the **absorbed / latent** space (576-dim `q·kv`, softmax,
context accumulated in the 512-dim latent), over the **contiguous compressed KV cache**. Parallelized
**q-tiled** (flash over q-blocks, streaming KV) with an in-kernel online softmax — **single pass, no
cross-CTA reduce**. The output feeds the shared [`bmm_fp8`](./bmm_fp8.md) BMM2 (latent→`v_head=128`).

This is **prefill only**. It shares the absorbed math, the compressed KV cache, and BMM1/BMM2 with
[`mla_decode_attn`](./mla_decode_attn.md) — but differs in **work distribution** (q-tiled flash here
vs decode's kv-split + reduce), so by the catalog's grid-rule it is a **separate layer** (different
`grid_dim`). The **unabsorbed** `kv_b` prefill path is *not* used.

**Phase:** prefill.

**grid_dim (compile-time — caller-provided):** `(Hd, ceil(max_q_len/64), num_request)`,
block `(256,1,1)` — grid.x = head, grid.y = q-tile (BM = 64), grid.z = request. **Single-pass**
(no reduce). All three are **build-time constants**: `Hd` = heads/rank (= `q.shape[1]`; 16 at TP=8),
`max_q_len` = max prompt tokens (= `q.shape[0]` = `max_num_batched_tokens`), `num_request` =
`max_num_batched_requests` (= 1 here). **Example:** `max_q_len=8192`, TP=8 → `grid=(16, 128, 1)`.

**Compile-time vs runtime:** the grid is baked at build from `max_q_len` (the worst-case prompt).
The **actual** prompt length (and actual KV length) at runtime are ≤ these and are masked inside the
kernel (causal mask, `q_start=0`) — they never resize the grid.

Unlike decode (which **derives** its grid internally), this layer **takes `grid_dim` as an
argument** — the caller computes it from the compile-time formula above. `max_kv_len`
(= `kv.shape[0]`) is passed via params for the mask.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `q` | `[S,Hd,576]` | bf16 | head-major absorbed query `[latent\|rope]`, assembled **in-place** via views (BMM1 → `[:,:,:512]`, roped `q_pe` → `[:,:,512:576]`); TMA |
| `kv` | `[L,576]` | bf16 | **contiguous** latent KV cache `[0:S]` (the prompt's own compressed KV, appended by [`kv_cache_gather`](./kv_cache_gather.md); no paging) |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `attn_out` | `[S,Hd,512]` | bf16 | head-major latent context (→ BMM2 un-absorb) |

**Params (compile-time):** `max_kv_len` (KV capacity for the causal mask; = `kv.shape[0]`). `Hd`
(= `q.shape[1]`) and `max_q_len` (= `q.shape[0]`) — used to form `grid_dim` — are also compile-time,
derived from the tensor shapes. None is a runtime length. Causal mask is implicit (`q_start=0`;
query `i` attends `[0:i]`).

**Tensor-view requirement (MUST):** `q` is the same in-place-assembled `[S,Hd,576]` buffer as the
decode path (BMM1 writes `[:,:,:512]`, roped `q_pe` is `[:,:,512:576]`); `kv` is the contiguous
cache window `[0:S]`. Honor `stride[0]` + offset; do not assume contiguity.

**Shape variants**

| tp_degree | Hd | notes |
|---|---|---|
| 8 (this config) | 16 | `Hd` derived from `q`; only TP=8 needed for v1 |

## Python API
```python
def mla_prefill_attn_layer(
    self,
    q: DTensor,                    # [S,Hd,576] bf16, absorbed query [latent|rope]; S = max_q_len
    kv: DTensor,                   # [L,576] bf16, contiguous compressed KV cache; L = max_kv_len
    attn_out: DTensor,             # [S,Hd,512] bf16 out, latent context (→ BMM2)
    grid_dim: tuple,               # compile-time: (Hd, ceil(max_q_len/64), num_request)
    block_dim: tuple = (256, 1, 1),
) -> None
```
All sizing args are **compile-time constants** (the grid is baked at build); the runtime prompt
length is masked, not a grid input.

**Reuse:** `mla_prefill_absorbed_layer` → task `mla_prefill_absorbed_sm100` (absorbed, q-tiled,
single-pass — **no separate reduce**). A split-K long-`kv_len` variant
(`mla_prefill_tp8_chunked_splitk` + `mla_prefill_tp8_chunked_reduce`) exists but is an
**optimization, deferred** for v1; the unabsorbed `mla_prefill_tp8_chunked` (`kv_b`) is dropped.

**Special:** for v1 we only need this to run under the TP=8 setup; the TP=8 head split can be
simulated on a single GPU for testing.
