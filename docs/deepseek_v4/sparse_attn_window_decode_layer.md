# `sparse_attn_window_decode_layer`  (compress_ratio == 0, decode)

Decode-step sibling of [`sparse_attn_window_prefill_layer`](sparse_attn_window_prefill_layer.md).
Single new query token attends to the rolling window-only KV cache.

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Function: `Attention.forward`, lines 484-543, taking
  `compress_ratio == 0` and `start_pos > 0` branches (lines 529-533, with the
  `compress_ratio` block skipped at 531-532)

## Math

With `S = 1`, `W = window_size`, `Dh = head_dim`, `rd = rope_head_dim`, `H = n_local_heads`:

```
apply_rotary_emb(q[..., -rd:], freqs_cis[start_pos : start_pos+1])
apply_rotary_emb(kv[..., -rd:], freqs_cis[start_pos : start_pos+1])

topk_idxs = get_window_topk_idxs(W, B, 1, start_pos)        # [B, 1, W], -1 for masked

# Sliding-window cache write at slot (start_pos % W)
kv_cache[:B, start_pos % W] = kv.squeeze(1)

# Attention reads the *whole* window cache (length W), with topk_idxs gating valid slots
o = sparse_attn(q, kv_cache[:B], attn_sink, topk_idxs, softmax_scale)   # [B, 1, H, Dh]
apply_rotary_emb(o[..., -rd:], freqs_cis[start_pos : start_pos+1], inverse=True)
```

## Python API (proposed)

```python
pk.sparse_attn_window_decode_layer(
    input_q:         DTensor,    # BF16 [B, 1, n_heads, head_dim]
    input_kv_new:    DTensor,    # BF16 [B, 1, head_dim]
    input_attn_sink: DTensor,    # FP32 [n_heads]
    freqs_cis:       DTensor,    # FP32 [max_seq_len, rope_head_dim] (rope_theta=10000, no YaRN)
    inout_kv_cache:  DTensor,    # BF16 [B, window_size, head_dim]
    output_o:        DTensor,    # BF16 [B, 1, n_heads * head_dim]
    grid_dim: tuple,
    block_dim: tuple,
    n_heads: int = 64,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    window_size: int = 128,
    start_pos: int,             # task metadata, not blockIdx-derived
    softmax_scale: float = 1.0 / 512 ** 0.5,
)
```

## Inputs

| name | dtype | shape | source |
| --- | --- | --- | --- |
| `input_q` | BF16 | `[B, 1, n_heads, head_dim]` | `q_normalize_layer` output |
| `input_kv_new` | BF16 | `[B, 1, head_dim]` | `wkv` → `kv_norm` |
| `input_attn_sink` | FP32 | `[n_heads]` | `attn.attn_sink` |
| `freqs_cis` | FP32 | `[max_seq_len, rope_head_dim]` | base RoPE (no YaRN) |

## In/out

| name | dtype | shape | role |
| --- | --- | --- | --- |
| `inout_kv_cache` | BF16 | `[B, window_size, head_dim]` | sliding window ring; one slot per step is overwritten |

## Outputs

| name | dtype | shape | sink |
| --- | --- | --- | --- |
| `output_o` | BF16 | `[B, n_heads * head_dim]` | `grouped_lowrank_o_proj_a_layer` |

## Builder usage

```python
if compress_ratios[L] == 0 and start_pos > 0:
    o = pk.sparse_attn_window_decode_layer(
        input_q=q, input_kv_new=kv, input_attn_sink=attn_sink,
        freqs_cis=freqs_cis_base,
        inout_kv_cache=kv_cache_window, output_o=o_buf,
        start_pos=start_pos, ...)
```

V3 analog: `mla_decode_layer` + `mla_reduce_layer` (single-token decode). For window-only
attention there's no need to split into decode+reduce since `K_LEN = window_size = 128`
is small; one kernel suffices. If profiling shows the K reduction is too long for one CTA,
add a `*_reduce` sibling later (mirroring V3's pattern).

## Notes / risks

- Same RoPE caveat as the prefill sibling: base `rope_theta=10000`, no YaRN.
- `start_pos % W` must be a runtime task-metadata field (CLAUDE.md "Task" rule:
  blockIdx-agnostic).
- Window cache overwrites: only one slot is updated per step. Make sure no parallel decode
  stream writes the same slot.
- Verification: multi-step decode in test mode against the V4 reference, comparing both
  `output_o` per step and the `inout_kv_cache` final state.
