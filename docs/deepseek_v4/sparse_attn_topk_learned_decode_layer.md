# `sparse_attn_topk_learned_decode_layer`  (compress_ratio == 4, decode)

Decode-step sibling of [`sparse_attn_topk_learned_prefill_layer`](sparse_attn_topk_learned_prefill_layer.md).
Single new query token attends to a window of `W=128` recent KV slots **plus** up to
`K=512` learned-indexer-selected compressed KV slots from the same `Attention.kv_cache`.

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Function: `Attention.forward`, lines 484-543, with the `compress_ratio == 4` and
  `start_pos > 0` branches (lines 530-534)

## Math

With `S = 1`, `W = window_size`, `R = compress_ratio = 4`, `K = index_topk = 512`, `Dh = head_dim`:

```
apply_rotary_emb(q[..., -rd:],  freqs_cis_compress[start_pos:start_pos+1])
apply_rotary_emb(kv[..., -rd:], freqs_cis_compress[start_pos:start_pos+1])

topk_idxs_win = get_window_topk_idxs(W, B, 1, start_pos)                # [B, 1, W]
topk_idxs_cmp = indexer_layer(...)                                       # [B, 1, K], offset by W
topk_idxs     = cat([topk_idxs_win, topk_idxs_cmp], dim=-1)              # [B, 1, W+K]

# Cache write into the window segment ring
kv_cache[:B, start_pos % W] = kv.squeeze(1)

# Compressor incrementally writes into kv_cache[:, W:] every R steps
compressor_decode_layer(...)

# Attention reads the FULL kv_cache (window segment + compressed segment)
o = sparse_attn(q, kv_cache[:B], attn_sink, topk_idxs, softmax_scale)
apply_rotary_emb(o[..., -rd:], freqs_cis_compress[start_pos:start_pos+1], inverse=True)
```

## Python API (proposed)

```python
pk.sparse_attn_topk_learned_decode_layer(
    input_q:             DTensor,   # BF16 [B, 1, n_heads, head_dim]
    input_kv_new:        DTensor,   # BF16 [B, 1, head_dim]
    input_attn_sink:     DTensor,   # FP32 [n_heads]
    input_topk_idxs_win: DTensor,   # INT32[B, 1, window_size]
    input_topk_idxs_cmp: DTensor,   # INT32[B, 1, index_topk]
    freqs_cis:           DTensor,   # FP32 [max_seq_len, rope_head_dim]   (compress_rope_theta + YaRN)
    inout_kv_cache:      DTensor,   # BF16 [B, window_size + max_seq_len // R, head_dim]
    output_o:            DTensor,   # BF16 [B, n_heads * head_dim]
    grid_dim: tuple,
    block_dim: tuple,
    n_heads: int = 64,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    window_size: int = 128,
    compress_ratio: int = 4,
    start_pos: int,
    softmax_scale: float = 1.0 / 512 ** 0.5,
)
```

## Inputs

| name | dtype | shape | source |
| --- | --- | --- | --- |
| `input_q` | BF16 | `[B, 1, n_heads, head_dim]` | `q_normalize_layer` |
| `input_kv_new` | BF16 | `[B, 1, head_dim]` | `wkv` → `kv_norm` |
| `input_attn_sink` | FP32 | `[n_heads]` | `attn.attn_sink` |
| `input_topk_idxs_win` | INT32 | `[B, 1, window_size]` | `get_window_topk_idxs` |
| `input_topk_idxs_cmp` | INT32 | `[B, 1, index_topk]` | `indexer_layer` |
| `freqs_cis` | FP32 | `[max_seq_len, rope_head_dim]` | `compress_rope_theta` + YaRN |

## In/out

| name | dtype | shape | role |
| --- | --- | --- | --- |
| `inout_kv_cache` | BF16 | `[B, W + max_seq_len // R, head_dim]` | window slot updated this step; compressed segment maintained by `compressor_decode_layer` |

## Outputs

| name | dtype | shape | sink |
| --- | --- | --- | --- |
| `output_o` | BF16 | `[B, n_heads * head_dim]` | `grouped_lowrank_o_proj_a_layer` |

## Builder usage

```python
if compress_ratios[L] == 4 and start_pos > 0:
    pk.compressor_decode_layer(...)
    topk_idxs_cmp = pk.indexer_layer(..., start_pos=start_pos, offset=window_size)
    o = pk.sparse_attn_topk_learned_decode_layer(
        input_q=q, input_kv_new=kv, input_attn_sink=attn_sink,
        input_topk_idxs_win=win_idxs, input_topk_idxs_cmp=topk_idxs_cmp,
        freqs_cis=freqs_cis_compress,
        inout_kv_cache=kv_cache_full, output_o=o_buf,
        start_pos=start_pos, ...)
```

V3 analog: `mla_mtp_decode_layer` + `mla_mtp_reduce_layer`. The K dim here is up to `W+K =
640`, matching V3's typical reduction shape; consider whether to split into
`*_decode + *_reduce` for kernel efficiency (mirror V3 if profiling indicates a single CTA
can't fit the reduction).

## Notes / risks

- The kernel reads the whole `kv_cache` slice but only attends to slots referenced by
  `topk_idxs` (with `-1` sentinel skip).
- Window slots can be empty (when `start_pos < W`) — `get_window_topk_idxs` returns `-1`
  in those cases.
- The compressed segment grows over time; on early decode steps `topk_idxs_cmp` may be
  shorter than `K=512`. The Indexer pads with `-1`.
- Same RoPE-theta caveat as the prefill sibling.
- Verification: multi-step decode in test mode against the V4 reference; compare per-step
  `output_o` and the cache state.
