# `compressor_prefill_layer`

Computes V4's learned-gated KV compression for the **prefill** path (`start_pos == 0`):
collapses every `compress_ratio` consecutive tokens into one compressed KV vector, with
overlapping windows when `compress_ratio == 4`. Writes the result into the compressed KV
cache (the second segment of `Attention.kv_cache`) and seeds the per-batch `kv_state` /
`score_state` ring buffers used by the decode-path twin.

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Class: `Compressor`
- Lines: 279-377 (forward called from `Attention.forward` line 525)
- Reads `start_pos == 0` branch: lines 325-342 (compute) and 360-377 (norm + RoPE + write)
- Helper inside the class:
  - `Compressor.overlap_transform` (lines 307-314) — stitches the rolling-window halves
    when `overlap=True` (i.e., `compress_ratio == 4`)
  - `Compressor.ape` (line 294) — additive position embedding `[ratio, coff*head_dim]`,
    `coff = 1 + overlap`

## Math

With `D = head_dim` (=512), `R = compress_ratio`, `coff = 1 + (R == 4)`, `rd = rope_head_dim` (=64):

```
x_fp32   = x.float()                                     # [B, S, dim]
kv_pre   = wkv(x_fp32)                                   # [B, S, coff*D]
score_pre= wgate(x_fp32)                                 # [B, S, coff*D]

# Tail handling: if S % R != 0, last `remainder` tokens go into kv_state/score_state buffers
remainder = S % R
cutoff    = S - remainder

# Optional seed of overlap half (only if overlap and cutoff >= R)
if overlap and cutoff >= R:
    kv_state[:, :R]   = kv_pre[:, cutoff-R : cutoff]
    score_state[:, :R]= score_pre[:, cutoff-R : cutoff] + ape

# Save tail to state ring (offset = R if overlap else 0)
offset = R if overlap else 0
if remainder > 0:
    kv_pre, kv_state[:, offset : offset+remainder]    = split(kv_pre,    [cutoff, remainder])
    _,      score_state[:, offset : offset+remainder] = split(score_pre, [cutoff, remainder]) + ape[:remainder]
    score_pre = score_pre[:, :cutoff]

# Compress the full chunks
kv    = kv_pre.unflatten(1, (-1, R))     # [B, S/R, R, coff*D]
score = score_pre.unflatten(1, (-1, R)) + ape

if overlap:                                # overlap_transform stitches across boundaries
    kv    = overlap_transform(kv,    fill=0)
    score = overlap_transform(score, fill=-inf)

kv_compressed = sum(kv * softmax(score, dim=2), dim=2)    # [B, S/R, D]
kv_compressed = kv_norm(kv_compressed.to(bf16))           # RMSNorm on head_dim

# RoPE on the rope_head_dim suffix, using freqs_cis[:cutoff:R] (sub-sampled grid)
apply_rotary_emb(kv_compressed[..., -rd:], freqs_cis[:cutoff:R])

# (Optional) FP8 simulation on the non-rope dims for QAT — kept BF16 in MPK first pass
# Write to compressed cache
attention_kv_cache[:, win : win + S//R] = kv_compressed
```

## Python API (proposed)

```python
pk.compressor_prefill_layer(
    input_x:        DTensor,    # BF16 [B, S, dim]
    wkv:            DTensor,    # FP32 [coff*head_dim, dim]
    wgate:          DTensor,    # FP32 [coff*head_dim, dim]
    ape:            DTensor,    # FP32 [compress_ratio, coff*head_dim]
    kv_norm_weight: DTensor,    # FP32 [head_dim]
    freqs_cis:      DTensor,    # FP32 complex (or split cos/sin) [max_seq_len, rope_head_dim/2]

    output_kv_cache:    DTensor,  # BF16 [B, max_kv, head_dim]    -- compressed segment slice [win : win + max_compressed]
    output_kv_state:    DTensor,  # FP32 [B, coff*compress_ratio, coff*head_dim]
    output_score_state: DTensor,  # FP32 [B, coff*compress_ratio, coff*head_dim]

    grid_dim: tuple,
    block_dim: tuple,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    compress_ratio: int = 4,    # 4 or 128
    seqlen: int,                 # known from prefill chunk length
)
```

The Indexer uses its **own internal Compressor** (with `rotate=True` and `head_dim=128`),
so we register a separate copy with different weight/cache pointers but the same kernel
implementation. See [`indexer_layer.md`](indexer_layer.md).

## Inputs

| name | dtype | shape | source |
| --- | --- | --- | --- |
| `input_x` | BF16 | `[B, S, dim]` | block input (post-`hc_pre_layer`/post-`attn_norm`) |
| `wkv` | FP32 | `[coff * head_dim, dim]` | `compressor.wkv.weight` |
| `wgate` | FP32 | `[coff * head_dim, dim]` | `compressor.wgate.weight` |
| `ape` | FP32 | `[compress_ratio, coff * head_dim]` | `compressor.ape` |
| `kv_norm_weight` | FP32 | `[head_dim]` | `compressor.norm.weight` |
| `freqs_cis` | FP32 (cos+sin pair) | `[max_seq_len, rope_head_dim]` | precomputed once, theta = `compress_rope_theta` |

For V4-Flash-Base: `head_dim=512`, `compress_ratio ∈ {4, 128}`, `coff = 1` if 128 else `2`.

## Outputs

| name | dtype | shape | sink |
| --- | --- | --- | --- |
| `output_kv_cache` | BF16 | `[B, S/R, head_dim]` slice of `Attention.kv_cache` at `[:, win :]` | consumed by `sparse_attn_topk_*_prefill_layer` |
| `output_kv_state` | FP32 | `[B, coff*R, coff*head_dim]` | consumed by `compressor_decode_layer` on subsequent steps |
| `output_score_state` | FP32 | `[B, coff*R, coff*head_dim]` | consumed by `compressor_decode_layer` on subsequent steps |

## Builder usage

Called once per block whose `compress_ratio > 0`, at prefill time only:

```python
if compress_ratio > 0:
    pk.compressor_prefill_layer(
        input_x=x_after_attn_norm, wkv=cmp_wkv, wgate=cmp_wgate, ape=cmp_ape,
        kv_norm_weight=cmp_kv_norm_w, freqs_cis=freqs_cis_compress,
        output_kv_cache=kv_cache_compressed_slice,
        output_kv_state=cmp_kv_state, output_score_state=cmp_score_state,
        ...)
```

`Indexer` (used when `compress_ratio == 4`) calls `compressor_prefill_layer` again with its
**own** `wkv`/`wgate`/`ape`/`kv_norm` weights, an `index_head_dim=128` cache, and Hadamard
rotation enabled on the output (the `rotate=True` branch on lines 368-370 — handled inside
`indexer_layer` and deferred from this design pass).

## Notes / risks

- All compute is fp32 except the final cast to BF16 before RoPE; matches the reference's
  `compression need fp32` comment (line 321).
- `freqs_cis` here uses `compress_rope_theta` (=160000 for V4-Flash-Base); see
  [`compress_rope_theta_delta.md`](compress_rope_theta_delta.md). The non-`compress` ramp
  uses YaRN scaling; for `compress_ratio == 0` layers, no Compressor is used, so this kernel
  is not invoked.
- `output_kv_cache` is the slice `Attention.kv_cache[:, window_size:]` — the same buffer as
  the window-segment writes, just a different stride/offset. The runtime must wire this
  slice carefully (paged-cache style, V3 does similar split for MLA paged cache).
- The "tail" branch (`remainder > 0`) only fires when `S % R != 0`; the kernel must handle
  both the full-chunk and tail-only cases without separate registrations.
- Verification: feed the same `(x, wkv, wgate, ape)` into both the V4 reference and the
  MPK kernel; compare the resulting `kv_cache` slice and the `kv_state`/`score_state`
  buffers. Use the test-mode harness from `tests/runtime_python/test_mode/`.
