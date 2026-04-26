# `sparse_attn_topk_strided_prefill_layer`  (compress_ratio == 128, prefill)

V4 attention compute kernel for blocks where `compress_ratios[layer_id] == 128` and
`start_pos == 0`. Like the `compress_ratio == 4` prefill sibling, but the compressed
top-k indices come from a **non-learned** algorithmic helper (`get_compress_topk_idxs`)
instead of an Indexer — every 128th token's compressed slot is selected.

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Function: `Attention.forward`, lines 484-543, with `compress_ratio == 128` and
  `start_pos == 0` branches
- Helper: `get_compress_topk_idxs` (lines 268-276) — generates strided indices into the
  compressed cache + a causal mask
- Indexer is **not** instantiated for `compress_ratio != 4` (constructor line 470-471
  sets `self.indexer = None`)

## Math

With `S` = prefill chunk length, `R = 128`, `W = window_size`, etc.:

```
apply_rotary_emb(q[..., -rd:],  freqs_cis_compress[0:S])
apply_rotary_emb(kv[..., -rd:], freqs_cis_compress[0:S])

topk_idxs_win = get_window_topk_idxs(W, B, S, 0)                  # [B, S, W]
topk_idxs_cmp = get_compress_topk_idxs(R, B, S, 0, offset=S)      # [B, S, S/R], strided + causal-masked
topk_idxs     = cat([topk_idxs_win, topk_idxs_cmp], dim=-1)

# Cache writes (same as compress_ratio == 4 prefill)
if S <= W:
    kv_cache[:B, :S] = kv
else:
    cutoff = S % W
    kv_cache[:B, cutoff:W], kv_cache[:B, :cutoff] = kv[:, -W:].split([W-cutoff, cutoff])

kv_compressed = compressor_prefill_layer(x, ..., compress_ratio=128)   # writes kv_cache[:, W:]
kv_for_attn   = cat([kv, kv_compressed], dim=1)                          # [B, S + S/R, head_dim]

o = sparse_attn(q, kv_for_attn, attn_sink, topk_idxs, softmax_scale)
apply_rotary_emb(o[..., -rd:], freqs_cis_compress[0:S], inverse=True)
```

`get_compress_topk_idxs` produces `[arange(S/R) for each row, with rows below row*R masked]`
— an all-1s lower-triangular pattern in row-id-divided-by-`R` space. With `R = 128`, this
is a very sparse selection (e.g., for `S = 4096`, only `S/R = 32` compressed entries per
row).

## Python API (proposed)

Same as [`sparse_attn_topk_learned_prefill_layer`](sparse_attn_topk_learned_prefill_layer.md)
with renaming:

```python
pk.sparse_attn_topk_strided_prefill_layer(
    input_q:               DTensor,   # BF16 [B, S, n_heads, head_dim]
    input_kv_window:       DTensor,   # BF16 [B, S, head_dim]
    input_kv_compressed:   DTensor,   # BF16 [B, S // R, head_dim]
    input_attn_sink:       DTensor,   # FP32 [n_heads]
    input_topk_idxs_win:   DTensor,   # INT32[B, S, window_size]
    input_topk_idxs_cmp:   DTensor,   # INT32[B, S, max_compressed_topk]   (here S/R, strided)
    freqs_cis:             DTensor,   # FP32 [max_seq_len, rope_head_dim]  (compress_rope_theta + YaRN)
    output_kv_cache_window:DTensor,   # BF16 [B, window_size, head_dim]
    output_o:              DTensor,   # BF16 [B, S, n_heads * head_dim]
    grid_dim: tuple,
    block_dim: tuple,
    n_heads: int = 64,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    window_size: int = 128,
    compress_ratio: int = 128,
    seqlen: int,
    softmax_scale: float = 1.0 / 512 ** 0.5,
)
```

## Inputs

Identical schema to [`sparse_attn_topk_learned_prefill_layer`](sparse_attn_topk_learned_prefill_layer.md),
with two semantic differences:
- `input_topk_idxs_cmp` comes from `get_compress_topk_idxs` (algorithmic), not from
  `indexer_layer`. The size of its last dim is `ceil(S/R)`, not `index_topk`.
- The `compress_ratio` parameter passed to the upstream Compressor is `128` instead of `4`,
  meaning **no overlap** in the Compressor (line 290: `overlap = compress_ratio == 4`).

## Outputs

Identical to the learned-topk sibling: `output_kv_cache_window` and `output_o`.

## Builder usage

```python
if compress_ratios[L] == 128 and start_pos == 0:
    pk.compressor_prefill_layer(..., compress_ratio=128)         # writes kv_cache[:, W:]
    topk_idxs_cmp = pk.attach_input(get_compress_topk_idxs(128, B, S, 0, offset=S))
    o = pk.sparse_attn_topk_strided_prefill_layer(
        input_q=q, input_kv_window=kv, input_kv_compressed=kv_cache_compressed_slice,
        input_attn_sink=attn_sink,
        input_topk_idxs_win=win_idxs, input_topk_idxs_cmp=topk_idxs_cmp,
        freqs_cis=freqs_cis_compress,
        output_kv_cache_window=kv_cache_window_slice, output_o=o_buf, ...)
```

V3 analog: same as the learned-topk sibling — `mla_prefill_layer`.

## Notes / risks

- `get_compress_topk_idxs` is precomputable (no learned weights, no input data). Generate
  once per `(R, B, S, start_pos)` and bind via `_safe_attach`.
- `S/R` can be small (32 for `S=4096`, `R=128`). Pad / sentinel-fill to a fixed size if
  needed for the kernel to use a fixed loop bound.
- Compressor invocation here uses `compress_ratio = 128` — the same kernel, but
  `overlap = False` and `coff = 1`. Make sure the kernel handles the `coff = 1` branch.
- All other notes (RoPE theta, attn_sink, `-1` sentinels) match the learned-topk sibling.
- Verification: end-to-end numeric match with the reference for a single prefill chunk on
  a layer where `compress_ratios[L] == 128`. Layers 3 and 5 in V4-Flash-Base are the
  earliest such layers.
