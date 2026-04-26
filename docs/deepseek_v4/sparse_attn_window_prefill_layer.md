# `sparse_attn_window_prefill_layer`  (compress_ratio == 0, prefill)

V4's attention compute kernel for blocks where `compress_ratios[layer_id] == 0` and
`start_pos == 0`. With no Compressor or Indexer, the KV cache only holds the most recent
`window_size` tokens, and attention is a chunked **sliding-window full-attention** over
the prefill chunk.

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Function: `Attention.forward`, lines 484-543, taking the
  `compress_ratio == 0` and `start_pos == 0` branches
  (line 488 sets `ratio=0`; lines 508-515 skip the compressed-topk concat;
  lines 518-528 take the prefill cache-write path with `compress_ratio` falsy)
- Constructor: `Attention.__init__` lines 472-482 — for `compress_ratio == 0`,
  `kv_cache_size = window_size`, RoPE built with `original_seq_len=0` and
  `rope_theta = 10000` (i.e., **no YaRN**, base RoPE)

## Math (single chunk, single block)

With `H = n_local_heads` (=64), `Dh = head_dim` (=512), `rd = rope_head_dim` (=64),
`W = window_size` (=128), `S` = prefill chunk length:

```
# Q (RoPE applied on the rope dims, after q_normalize_layer)
apply_rotary_emb(q[..., -rd:], freqs_cis[0:S])              # in-place

# KV (single MQA head, head_dim = Dh)
apply_rotary_emb(kv[..., -rd:], freqs_cis[0:S])

# Window-only top-k indices (causal, length up to W per row)
topk_idxs = get_window_topk_idxs(W, B, S, 0)                # [B, S, W], -1 for masked

# Cache write (sliding-window ring)
if S <= W:
    kv_cache[:B, :S] = kv
else:
    cutoff = S % W
    kv_cache[:B, cutoff:W], kv_cache[:B, :cutoff] = kv[:, -W:].split([W - cutoff, cutoff])

# sparse_attn (with attn_sink) — kv argument is the just-computed kv (S tokens), not the cache slice
o = sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale=Dh**-0.5)   # [B, S, H, Dh]

# Inverse RoPE on output rope dims
apply_rotary_emb(o[..., -rd:], freqs_cis[0:S], inverse=True)
```

`sparse_attn` (imported from the V4 `kernel` module) implements the dense-on-topk attention
plus an attention-sink term: a learnable `[H]` scalar per head that participates in the
softmax denominator (a "register token" per head). The MPK kernel must implement the same.

## Python API (proposed)

```python
pk.sparse_attn_window_prefill_layer(
    input_q:        DTensor,     # BF16 [B, S, n_heads, head_dim]   (post q_normalize, pre RoPE)
    input_kv:       DTensor,     # BF16 [B, S, head_dim]            (post kv_norm, pre RoPE)
    input_attn_sink:DTensor,     # FP32 [n_heads]
    freqs_cis:      DTensor,     # FP32 [max_seq_len, rope_head_dim]   (rope_theta = 10000, no YaRN)
    output_o:       DTensor,     # BF16 [B, S, n_heads * head_dim]
    output_kv_cache:DTensor,     # BF16 [B, window_size, head_dim]    (slice; will be ring-buffered)
    grid_dim: tuple,
    block_dim: tuple,
    n_heads: int = 64,
    head_dim: int = 512,
    rope_head_dim: int = 64,
    window_size: int = 128,
    seqlen: int,
    softmax_scale: float = 1.0 / 512 ** 0.5,
)
```

## Inputs

| name | dtype | shape | source |
| --- | --- | --- | --- |
| `input_q` | BF16 | `[B, S, n_heads, head_dim]` | `q_normalize_layer`'s output reshape |
| `input_kv` | BF16 | `[B, S, head_dim]` | `wkv` → `kv_norm` (reuse V3 `linear_fp8_layer` + `rmsnorm_layer`) |
| `input_attn_sink` | FP32 | `[n_heads]` | `attn.attn_sink` |
| `freqs_cis` | FP32 | `[max_seq_len, rope_head_dim]` | precomputed with **base** `rope_theta=10000` (no YaRN) |

## Outputs

| name | dtype | shape | sink |
| --- | --- | --- | --- |
| `output_o` | BF16 | `[B, S, n_heads * head_dim]` | input to `grouped_lowrank_o_proj_a_layer` |
| `output_kv_cache` | BF16 | `[B, window_size, head_dim]` | persisted into next decode step |

## Builder usage

Selected in the per-block dispatch when `compress_ratios[layer_id] == 0`:

```python
if compress_ratios[L] == 0:
    if start_pos == 0:
        o = pk.sparse_attn_window_prefill_layer(...)
    else:
        o = pk.sparse_attn_window_decode_layer(...)
```

V3 analog: `mla_prefill_layer` (with adapted shapes for MQA + new attn_sink).

## Notes / risks

- The RoPE constants are different from the `compress_ratio > 0` siblings: this layer uses
  **base** `rope_theta` and **no YaRN**, while the compressed siblings use
  `compress_rope_theta` (=160000) with YaRN. See [`compress_rope_theta_delta.md`](compress_rope_theta_delta.md).
- `attn_sink`: a per-head scalar added to the softmax denominator (acting like a
  zero-key/zero-value entry). Implement as `softmax([scores, attn_sink])` truncated to keys.
- `get_window_topk_idxs` is a small precomputable tensor — generate once and bind via
  `_safe_attach`, the kernel just reads it.
- Inverse RoPE on `o[..., -rd:]` is part of this kernel (not a separate task).
- Verification: end-to-end numeric match with `Attention.forward(... start_pos=0)` for a
  single chunk on a synthetic input.
