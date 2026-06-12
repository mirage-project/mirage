# `kv_cache_gather` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** append the new tokens' compressed KV (`[c_latent | k_pe]`, 576/token) into the
**contiguous** KV cache at the current sequence offset. The input is a **single `[T,576]` strided
view** — the `[c_latent | k_pe]` region of the fused `qkv_a` buffer — and the append is a row-wise
copy into `kv_cache[step:step+T]` (a pure memory move; it does not matter that `c_latent` was
rmsnorm'd and `k_pe` was roped — both are already in place in the source row). Attention then reads
`kv_cache[0:seq_len]` directly — **no paging** (num_request=1, single sequence), so there is no
gather/materialize step.

**Phase:** both (decode appends 1–8 tokens; prefill appends the whole prompt once).

**grid_dim:** caller-sized over the new tokens (token-tiles); block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `c_kv_pe` | `[T, 576]` | bf16 | new compressed KV — **one** `mpk.narrow` view of fused `qkv_a [T,2112]` at cols `[1536:2112)`, stride `[2112,1]`; holds `[c_latent (normed) 512 \| k_pe (roped) 64]` |
| `kv_cache` | `[max_seq, 576]` | bf16 | the contiguous cache; new tokens written at `[step : step+T]` |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `kv_cache` | `[max_seq, 576]` | bf16 | updated **in-place**; attention reads `[0:seq_len]` |

**Params:** none — `T` and the write offset (`step`) come from the runtime; sizes from the tensors.

**Tensor-view requirement (MUST):** `c_kv_pe` is a **single** `mpk.narrow` slice of the fused
`qkv_a [T,2112]` — columns `[1536:2112)` (offset 1536, width 576, row stride `[2112,1]`). The 576
columns are contiguous within each row (`c_latent` `[1536:2048)` then `k_pe` `[2048:2112)`), so the
append is one strided row-copy; read via `stride[0]` + offset, not as a contiguous `[T,576]` buffer.

**Shape variants**

| variant | dims |
|---|---|
| single | contiguous cache `[max_seq, 576]`; no decode/prefill output split |

## Python API
```python
def kv_cache_gather_layer(
    self,
    c_kv_pe: DTensor,         # [T,576] bf16, ONE narrow view of qkv_a cols [1536:2112), stride [2112,1] ([c_latent|k_pe])
    kv_cache: DTensor,        # [max_seq,576] bf16, contiguous cache; written at [step:step+T] (in-place)
    grid_dim: tuple,
    block_dim: tuple = (256, 1, 1),
) -> None
```

**Reuse:** `mla_kv_gather_unified_layer` (`mla_kv_gather_unified_sm100`), **adapted**. Today it
takes two slices (`c_latent_new` + `k_pe_new`) with separate row-stride overrides, plus a
paged-cache input and `ckv_sep`/`kpe_sep` materialize outputs. For this contract it collapses to:
**one** `[T,576]` strided input (`c_kv_pe` — `[c_latent|k_pe]` is already contiguous), no paging, no
materialize — a plain strided row-copy into `kv_cache[step:step+T]`.
