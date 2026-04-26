# `compressor_decode_layer`

Decode-path twin of [`compressor_prefill_layer`](compressor_prefill_layer.md): consumes a
single new token's hidden state, accumulates it into the per-batch `kv_state` /
`score_state` rings, and — every `compress_ratio` steps — emits one compressed KV vector
into the compressed segment of `Attention.kv_cache`.

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Class: `Compressor`
- Lines: 343-377 (the `start_pos > 0` branch in `forward`, including the
  `should_compress = (start_pos + 1) % compress_ratio == 0` gate, the overlap and
  non-overlap cases at 346-359, and the shared norm/RoPE/write at 360-377)
- Called from `Attention.forward` lines 530-532

## Math

With `D = head_dim`, `R = compress_ratio`, `overlap = (R == 4)`, `rd = rope_head_dim`:

```
should_compress = (start_pos + 1) % R == 0
x_fp32   = x.float()                                              # [B, 1, dim]
kv_new   = wkv(x_fp32).squeeze(1)                                 # [B, coff*D]
score_new= wgate(x_fp32).squeeze(1) + ape[start_pos % R]          # [B, coff*D]

slot = (R + start_pos % R) if overlap else (start_pos % R)
kv_state   [:, slot] = kv_new
score_state[:, slot] = score_new

if not should_compress:
    return    # no cache write this step
else:
    if overlap:
        # split kv_state along channel axis: first R rows take low half, last R rows take high half
        kv_state_blk    = cat([kv_state[:, :R, :D],    kv_state[:, R:, D:]],    dim=1)   # [B, 2R, D]
        score_state_blk = cat([score_state[:, :R, :D], score_state[:, R:, D:]], dim=1)
        kv_compressed   = sum(kv_state_blk * softmax(score_state_blk, dim=1), dim=1, keepdim=True)
        # Roll: the second half becomes the new first half
        kv_state[:, :R]    = kv_state[:, R:]
        score_state[:, :R] = score_state[:, R:]
    else:
        kv_compressed = sum(kv_state * softmax(score_state, dim=1), dim=1, keepdim=True)

kv_compressed = kv_norm(kv_compressed.to(bf16))                                  # RMSNorm on head_dim
apply_rotary_emb(kv_compressed[..., -rd:], freqs_cis[start_pos + 1 - R].unsqueeze(0))
# (Optional) FP8 simulate non-rope dims — BF16 in MPK first pass
attention_kv_cache[:, start_pos // R] = kv_compressed.squeeze(1)
```

## Python API (proposed)

```python
pk.compressor_decode_layer(
    input_x:           DTensor,  # BF16 [B, 1, dim]   (single-token hidden state)
    wkv:               DTensor,  # FP32 [coff*head_dim, dim]
    wgate:             DTensor,  # FP32 [coff*head_dim, dim]
    ape:               DTensor,  # FP32 [compress_ratio, coff*head_dim]
    kv_norm_weight:    DTensor,  # FP32 [head_dim]
    freqs_cis:         DTensor,  # precomputed [max_seq_len, rope_head_dim]
    inout_kv_state:    DTensor,  # FP32 [B, coff*compress_ratio, coff*head_dim]
    inout_score_state: DTensor,  # FP32 [B, coff*compress_ratio, coff*head_dim]
    inout_kv_cache:    DTensor,  # BF16 [B, max_compressed, head_dim]   (compressed segment)

    grid_dim: tuple,
    block_dim: tuple,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    compress_ratio: int = 4,
    start_pos: int,             # absolute KV cache write position (passed via task metadata, not blockIdx)
)
```

`start_pos` must be a runtime task-metadata field on the persistent kernel's task descriptor
(à la `task_desc->task_metadata.start_pos`), not derived from `blockIdx`, in line with the
MPK rule that tasks are blockIdx-agnostic (CLAUDE.md "Task" section).

## Inputs

| name | dtype | shape | source |
| --- | --- | --- | --- |
| `input_x` | BF16 | `[B, 1, dim]` | block input post-`attn_norm` |
| `wkv`, `wgate`, `ape`, `kv_norm_weight`, `freqs_cis` | — | — | same as `compressor_prefill_layer` |

## In/out (read-modify-write)

| name | dtype | shape | role |
| --- | --- | --- | --- |
| `inout_kv_state` | FP32 | `[B, coff*R, coff*head_dim]` | persistent state ring; one slot updated per step, half-rolled on `should_compress` |
| `inout_score_state` | FP32 | same | same |
| `inout_kv_cache` | BF16 | `[B, max_compressed, head_dim]` | compressed cache slice; one row written every `R` steps |

## Builder usage

Called once per block whose `compress_ratio > 0`, on every decode step:

```python
if compress_ratio > 0 and start_pos > 0:
    pk.compressor_decode_layer(
        input_x=x_after_attn_norm,
        wkv=cmp_wkv, wgate=cmp_wgate, ape=cmp_ape, kv_norm_weight=cmp_kv_norm_w,
        freqs_cis=freqs_cis_compress,
        inout_kv_state=cmp_kv_state, inout_score_state=cmp_score_state,
        inout_kv_cache=kv_cache_compressed_slice,
        start_pos=start_pos, ...)
```

The `should_compress = ((start_pos + 1) % compress_ratio == 0)` decision is made inside the
kernel using the metadata `start_pos`; when false the kernel only updates state buffers and
skips the cache write.

V3 analog: V3 has no equivalent; V3's KV cache is written by `mla_kv_gather_layer` /
`mla_kv_gather_split_layer` which append the raw KV vector each step, with no compression.

## Notes / risks

- The kernel writes to two distinct cache buffers in different cadences: the state ring
  every step, the compressed cache once every `R` steps. Make sure event ordering between
  consecutive blocks doesn't skip a compressed-cache update.
- The "roll" step `kv_state[:, :R] = kv_state[:, R:]` (overlap branch) requires either an
  in-place shift or a double-buffered ring; pick one and document it in the kernel comment.
- `freqs_cis[start_pos + 1 - R]` is a single complex vector — pass the precomputed value as
  a small constant tensor or recompute from `start_pos` with the same YaRN parameters used
  at prefill (preferred, mirrors V3's RoPE-inside-attn pattern).
- Verification: simulate a multi-step decode in test mode and compare the compressed KV cache
  contents against the reference `Compressor.forward` after the same number of steps. The
  state buffers must also match at intermediate steps.
- The `compress_ratio == 128` mode: `overlap=False`, so the simpler branch (lines 355-359)
  applies; the same kernel handles both cases via the `compress_ratio` parameter.
