# `sparse_attn_topk_learned_prefill_layer`  (compress_ratio == 4, prefill)

V4 attention compute kernel for blocks where `compress_ratios[layer_id] == 4` and
`start_pos == 0`. KV cache layout = `[window_segment, compressed_segment]`. Top-k indices
combine `get_window_topk_idxs` and the **learned** `Indexer.topk_idxs`.

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Function: `Attention.forward`, lines 484-543, with the `compress_ratio == 4` and
  `start_pos == 0` branches
  - lines 508-515: top-k concat (window ∪ Indexer)
  - lines 518-528: prefill cache write (window) and Compressor → compressed cache
  - line 528: `sparse_attn(q, kv, attn_sink, topk_idxs, scale)` where `kv` here is the
    catenation of fresh-batch KV and Compressor's output (line 526)

## Math

With `H = n_local_heads`, `Dh = head_dim`, `rd = rope_head_dim`, `W = window_size`,
`R = compress_ratio = 4`, `K = index_topk = 512`, `S` = prefill chunk:

```
# RoPE on q/kv rope dims (compress_rope_theta + YaRN)
apply_rotary_emb(q[..., -rd:],  freqs_cis_compress[0:S])
apply_rotary_emb(kv[..., -rd:], freqs_cis_compress[0:S])

# Window top-k
topk_idxs_win = get_window_topk_idxs(W, B, S, 0)                 # [B, S, W]

# Indexer top-k (learned)
topk_idxs_cmp = indexer_layer(...)                                # [B, S, K], offset by S (kv.size(1))
topk_idxs     = cat([topk_idxs_win, topk_idxs_cmp], dim=-1)       # [B, S, W + K]

# Cache writes
if S <= W:
    kv_cache[:B, :S] = kv
else:
    cutoff = S % W
    kv_cache[:B, cutoff:W], kv_cache[:B, :cutoff] = kv[:, -W:].split([W - cutoff, cutoff])

kv_compressed = compressor_prefill_layer(x, ...)                  # writes kv_cache[:, W:]
kv_for_attn   = cat([kv, kv_compressed], dim=1)                   # [B, S + S/R, head_dim]

# Attention
o = sparse_attn(q, kv_for_attn, attn_sink, topk_idxs, softmax_scale=Dh**-0.5)
apply_rotary_emb(o[..., -rd:], freqs_cis_compress[0:S], inverse=True)
```

The unique aspect at prefill: the `kv` passed to `sparse_attn` is the concatenation of the
fresh KV (S tokens) and Compressor's output (`S/R` tokens), **not** the persistent
`kv_cache`. The `topk_idxs` reference these positions: window indices in `[0, S)` and
indexer indices in `[S, S + S/R)` (offset added inside Indexer).

## Python API (proposed)

```python
pk.sparse_attn_topk_learned_prefill_layer(
    input_q:               DTensor,   # BF16 [B, S, n_heads, head_dim]
    input_kv_window:       DTensor,   # BF16 [B, S, head_dim]              (fresh window kv)
    input_kv_compressed:   DTensor,   # BF16 [B, S//R, head_dim]           (Compressor output)
    input_attn_sink:       DTensor,   # FP32 [n_heads]
    input_topk_idxs_win:   DTensor,   # INT32[B, S, window_size]
    input_topk_idxs_cmp:   DTensor,   # INT32[B, S, index_topk]
    freqs_cis:             DTensor,   # FP32 [max_seq_len, rope_head_dim]  (compress_rope_theta + YaRN)

    output_kv_cache_window:DTensor,   # BF16 [B, window_size, head_dim]
    output_o:              DTensor,   # BF16 [B, S, n_heads * head_dim]

    grid_dim: tuple,
    block_dim: tuple,
    n_heads: int = 64,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    window_size: int = 128,
    compress_ratio: int = 4,
    seqlen: int,
    softmax_scale: float = 1.0 / 512 ** 0.5,
)
```

The `input_kv_compressed` tensor *is* the slice of `Attention.kv_cache[:, window_size:]`
written by `compressor_prefill_layer` immediately upstream — so the kernel can either
take it as a separate input (cleaner DAG) or read directly from the cache slice.

## Inputs

| name | dtype | shape | source |
| --- | --- | --- | --- |
| `input_q` | BF16 | `[B, S, n_heads, head_dim]` | `q_normalize_layer` |
| `input_kv_window` | BF16 | `[B, S, head_dim]` | `wkv` → `kv_norm` |
| `input_kv_compressed` | BF16 | `[B, S//R, head_dim]` | `compressor_prefill_layer` |
| `input_attn_sink` | FP32 | `[n_heads]` | `attn.attn_sink` |
| `input_topk_idxs_win` | INT32 | `[B, S, window_size]` | `get_window_topk_idxs` (precomputed) |
| `input_topk_idxs_cmp` | INT32 | `[B, S, index_topk]` | `indexer_layer` |
| `freqs_cis` | FP32 | `[max_seq_len, rope_head_dim]` | precomputed with **`compress_rope_theta=160000`** + YaRN |

## Outputs

| name | dtype | shape | sink |
| --- | --- | --- | --- |
| `output_kv_cache_window` | BF16 | `[B, window_size, head_dim]` | persistent across decodes |
| `output_o` | BF16 | `[B, S, n_heads * head_dim]` | `grouped_lowrank_o_proj_a_layer` |

## Builder usage

```python
if compress_ratios[L] == 4 and start_pos == 0:
    pk.compressor_prefill_layer(...)                                      # writes kv_cache[:, W:]
    topk_idxs_cmp = pk.indexer_layer(...)                                  # consumes Compressor's output via its own internal cache
    o = pk.sparse_attn_topk_learned_prefill_layer(
        input_q=q, input_kv_window=kv, input_kv_compressed=kv_cache_compressed_slice,
        input_attn_sink=attn_sink,
        input_topk_idxs_win=win_idxs, input_topk_idxs_cmp=topk_idxs_cmp,
        freqs_cis=freqs_cis_compress,
        output_kv_cache_window=kv_cache_window_slice, output_o=o_buf, ...)
```

V3 analog: `mla_prefill_layer` (with similar prefill semantics, but no top-k gating and
no attn_sink).

## Notes / risks

- The K dim seen by attention is `S + S/R` (not just window) for prefill — but `topk_idxs`
  caps it to `W + K = 128 + 512 = 640` per row. Sizing the kernel around 640 keys per row
  works regardless of `S`.
- Padding/sentinel: `topk_idxs` may include `-1` for masked entries (causal mask in
  Indexer). The kernel must skip these (zero contribution to softmax numerator/denominator).
- RoPE here uses the **compress** theta + YaRN; do not use `freqs_cis_base`. See
  [`compress_rope_theta_delta.md`](compress_rope_theta_delta.md).
- Verification: end-to-end numeric match with the V4 reference for a chunk of length
  `S = 256` (multiple compressed entries per block); allow loose tolerance (~1e-2) due
  to BF16 + FP8 quant.
