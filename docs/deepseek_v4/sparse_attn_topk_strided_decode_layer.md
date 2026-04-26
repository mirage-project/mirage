# `sparse_attn_topk_strided_decode_layer`  (compress_ratio == 128, decode)

Decode-step sibling of [`sparse_attn_topk_strided_prefill_layer`](sparse_attn_topk_strided_prefill_layer.md).
Single new query token attends to the window plus algorithmically-selected compressed slots.

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Function: `Attention.forward`, lines 484-543, with `compress_ratio == 128` and
  `start_pos > 0` branches
- Helper: `get_compress_topk_idxs(R=128, B, S=1, start_pos>0, offset)` — at decode it
  returns `arange(0, (start_pos + 1) // R) + offset` (a single 1D vector)

## Math

```
apply_rotary_emb(q[..., -rd:],  freqs_cis_compress[start_pos:start_pos+1])
apply_rotary_emb(kv[..., -rd:], freqs_cis_compress[start_pos:start_pos+1])

topk_idxs_win = get_window_topk_idxs(W, B, 1, start_pos)
topk_idxs_cmp = get_compress_topk_idxs(R=128, B, 1, start_pos, offset=W)
topk_idxs     = cat([topk_idxs_win, topk_idxs_cmp], dim=-1)

kv_cache[:B, start_pos % W] = kv.squeeze(1)
compressor_decode_layer(..., compress_ratio=128)   # writes kv_cache[:, W:] every 128 steps

o = sparse_attn(q, kv_cache[:B], attn_sink, topk_idxs, softmax_scale)
apply_rotary_emb(o[..., -rd:], freqs_cis_compress[start_pos:start_pos+1], inverse=True)
```

## Python API (proposed)

Same shape as the learned-topk decode sibling, with the `cmp` indices coming from the
algorithmic helper:

```python
pk.sparse_attn_topk_strided_decode_layer(
    input_q:             DTensor,   # BF16 [B, 1, n_heads, head_dim]
    input_kv_new:        DTensor,   # BF16 [B, 1, head_dim]
    input_attn_sink:     DTensor,   # FP32 [n_heads]
    input_topk_idxs_win: DTensor,   # INT32[B, 1, window_size]
    input_topk_idxs_cmp: DTensor,   # INT32[B, 1, max_compressed_topk]    ((start_pos+1)//R, padded)
    freqs_cis:           DTensor,   # FP32 [max_seq_len, rope_head_dim]   (compress_rope_theta + YaRN)
    inout_kv_cache:      DTensor,   # BF16 [B, W + max_seq_len // R, head_dim]
    output_o:            DTensor,   # BF16 [B, n_heads * head_dim]
    grid_dim: tuple,
    block_dim: tuple,
    n_heads: int = 64,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    window_size: int = 128,
    compress_ratio: int = 128,
    start_pos: int,
    softmax_scale: float = 1.0 / 512 ** 0.5,
)
```

`max_compressed_topk` is bounded by `max_seq_len // R`.

## Inputs / Outputs / In/Out

Same schema as [`sparse_attn_topk_learned_decode_layer`](sparse_attn_topk_learned_decode_layer.md);
only the source of `input_topk_idxs_cmp` and the `compress_ratio=128` parameter differ.

## Builder usage

```python
if compress_ratios[L] == 128 and start_pos > 0:
    pk.compressor_decode_layer(..., compress_ratio=128)
    topk_idxs_cmp = pk.attach_input(get_compress_topk_idxs(128, B, 1, start_pos, offset=W))
    o = pk.sparse_attn_topk_strided_decode_layer(
        input_q=q, input_kv_new=kv, input_attn_sink=attn_sink,
        input_topk_idxs_win=win_idxs, input_topk_idxs_cmp=topk_idxs_cmp,
        freqs_cis=freqs_cis_compress,
        inout_kv_cache=kv_cache_full, output_o=o_buf,
        start_pos=start_pos, ...)
```

V3 analog: same as the learned-topk decode sibling.

## Notes / risks

- `get_compress_topk_idxs` at decode returns a 1D index vector of length `(start_pos+1)//R`
  — for `start_pos < R-1` this is empty, meaning attention sees window only. The kernel
  must handle the empty case (treat the cmp slice as length 0).
- The compressed segment lookup index is `start_pos // R`, written by
  `compressor_decode_layer` once every 128 steps. Until the first compressed slot is
  written (i.e., `start_pos >= R-1`), there's nothing to attend to in the compressed segment.
- All other caveats match the learned-topk decode sibling.
- Verification: 256+ step decode in test mode, comparing per-step `output_o` and final
  cache state with the reference. Pay particular attention to the boundary
  `start_pos = R-1` (first compressed slot becomes valid).
