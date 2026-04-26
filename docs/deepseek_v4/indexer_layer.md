# `indexer_layer`

Selects the top-k positions in the **compressed** KV cache that the main attention should
attend to, via a learned scorer. Used **only** in blocks with `compress_ratio == 4` (in
V4-Flash-Base: blocks 2, 4, 6, …, 42 minus the few `ratio==128` blocks; see config).

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Class: `Indexer`, lines 380-433
- Called from `Attention.forward` line 511

## Components

The Indexer wraps:
1. `wq_b` — `ColumnParallelLinear(q_lora_rank=1024, n_heads * head_dim = 64 * 128)` —
   maps shared `qr` (the `q_norm`-normalized Q low-rank) to the indexer's per-head Q.
2. `weights_proj` — `ColumnParallelLinear(dim=4096, n_heads=64)` — produces a per-head
   gating scalar from `x`.
3. An **internal `Compressor`** with `head_dim=128`, `compress_ratio=4`, `rotate=True`.
   (Note: this is a separate Compressor instance from the main attention's, with its own
   weights and its own `kv_cache` buffer.) The Hadamard rotation + FP4 quant on this
   internal Compressor's output is **deferred** in the first design pass; we keep BF16.
4. `kv_cache: [B, max_seq_len // ratio, index_head_dim=128]` — the indexer-private
   compressed KV.

## Math

With `H = index_n_heads` (=64), `Dh = index_head_dim` (=128), `rd = rope_head_dim` (=64),
`R = compress_ratio` (=4), `K = index_topk` (=512):

```
# 1. Build indexer Q
q = wq_b(qr).unflatten(-1, (H, Dh))                          # [B, S, H, Dh]
apply_rotary_emb(q[..., -rd:], freqs_cis)                    # in-place RoPE on rope dims
# (Hadamard rotation + fp4 quant on q would happen here in the reference; deferred)

# 2. Update internal compressed KV cache (uses internal Compressor)
indexer_compressor(x, start_pos)   # writes into self.kv_cache (separate from main attn's)

# 3. Per-head scoring weights
weights = weights_proj(x) * (Dh ** -0.5) * (H ** -0.5)       # [B, S, H]

# 4. Score q against the (so-far filled) compressed cache
end_pos = start_pos + seqlen
kv_compressed = self.kv_cache[:B, : end_pos // R]            # [B, T_c, Dh] where T_c = end_pos // R
index_score = einsum("bshd, btd -> bsht", q, kv_compressed)  # [B, S, H, T_c]
index_score = relu(index_score) * weights.unsqueeze(-1)      # in-place ReLU
index_score = index_score.sum(dim=2)                         # [B, S, T_c]

# (TP) all_reduce(index_score) — handled outside this layer when world_size > 1

# 5. Causal mask for prefill
if start_pos == 0:
    mask = arange(S//R)[None, :] >= (arange(1, S+1)[:, None] // R)   # broadcast to [S, T_c]
    index_score += where(mask, -inf, 0)

# 6. Top-k
topk_idxs = index_score.topk(min(K, end_pos // R), dim=-1).indices   # [B, S, K']

# 7. Causal post-mask + offset for prefill
if start_pos == 0:
    mask = topk_idxs >= (arange(1, S+1)[:, None] // R)
    topk_idxs = where(mask, -1, topk_idxs + offset)
else:
    topk_idxs += offset
return topk_idxs
```

`offset` (passed in from `Attention.forward`) is `kv.size(1)` at prefill, `window_size` at
decode — it shifts the compressed-cache indices into the global cache index space (the
`Attention.kv_cache` layout is `[window_segment, compressed_segment]`, so compressed indices
need to be offset by the window-segment size).

## Python API (proposed)

Because the Indexer's compute graph splits cleanly into two phases (compressed-cache update
and score+topk), implement as a single MPK layer that internally calls the indexer-private
Compressor — or, equivalently, register it as a sequence of three MPK tasks chained inside
the builder:

```python
# Option A (recommended): one layer that fuses Q project + compressor write + score + topk
pk.indexer_layer(
    input_x:           DTensor,    # BF16 [B, S, dim]
    input_qr:          DTensor,    # BF16 [B, S, q_lora_rank]   (q_norm output, shared with main attn)
    wq_b:              DTensor,    # FP8  [n_heads*head_dim, q_lora_rank]   (or BF16 if checkpoint-BF16)
    wq_b_scale:        DTensor,    # FP32 (only if FP8)
    weights_proj_w:    DTensor,    # BF16 [n_heads, dim]
    cmp_wkv:           DTensor,    # FP32 [coff*head_dim, dim]   (indexer's internal Compressor)
    cmp_wgate:         DTensor,    # FP32 [coff*head_dim, dim]
    cmp_ape:           DTensor,    # FP32 [compress_ratio, coff*head_dim]
    cmp_kv_norm_w:     DTensor,    # FP32 [head_dim]
    freqs_cis:         DTensor,    # FP32 [max_seq_len, rope_head_dim]   (compress_rope_theta)

    inout_indexer_kv_cache:    DTensor,  # BF16 [B, max_seq_len // ratio, index_head_dim]
    inout_indexer_kv_state:    DTensor,  # FP32 [B, coff*ratio, coff*index_head_dim]
    inout_indexer_score_state: DTensor,  # FP32 [B, coff*ratio, coff*index_head_dim]

    output_topk_idxs:          DTensor,  # INT32 [B, S, index_topk]

    grid_dim: tuple,
    block_dim: tuple,
    n_heads: int = 64,
    head_dim: int = 128,
    rope_head_dim: int = 64,
    compress_ratio: int = 4,
    index_topk: int = 512,
    start_pos: int,
    offset: int,
)
```

Option B (split): expose `compressor_prefill_layer` / `compressor_decode_layer` again with
the indexer's weights, then expose a separate `indexer_score_topk_layer` that takes the
indexer's pre-computed `kv_cache` and Q. Either is fine; the merged option has fewer task
hops and matches V3's "small but fused" attention kernels.

## Inputs

| name | dtype | shape | source |
| --- | --- | --- | --- |
| `input_x` | BF16 | `[B, S, dim]` | `attn_norm`'s output (same as Compressor) |
| `input_qr` | BF16 | `[B, S, q_lora_rank]` | shared with main attention's `q_norm` output (do **not** recompute) |
| `wq_b` (+`scale`) | FP8 / BF16 | `[n_heads * head_dim, q_lora_rank]` | `indexer.wq_b.weight` |
| `weights_proj_w` | BF16 | `[n_heads, dim]` | `indexer.weights_proj.weight` |
| `cmp_*` | — | — | `indexer.compressor.{wkv, wgate, ape, norm.weight}` (separate from main attn's Compressor) |
| `freqs_cis` | FP32 | `[max_seq_len, rope_head_dim]` | precomputed with `compress_rope_theta` |

For V4-Flash-Base: `n_heads=64`, `head_dim=128`, `index_topk=512`, `compress_ratio=4`.

## Outputs

| name | dtype | shape | sink |
| --- | --- | --- | --- |
| `output_topk_idxs` | INT32 | `[B, S, index_topk]` | concatenated with `window_topk_idxs` and consumed by the corresponding `sparse_attn_topk_learned_*_layer` |
| `inout_indexer_kv_cache`, `inout_indexer_kv_state`, `inout_indexer_score_state` | — | persistent state for next decode step |

## Builder usage

Only registered for blocks with `compress_ratio == 4`:

```python
if compress_ratio == 4:
    topk_idxs_compressed = pk.indexer_layer(
        input_x=x_after_attn_norm, input_qr=qr_shared,
        wq_b=ind_wq_b, weights_proj_w=ind_w_proj,
        cmp_wkv=ind_cmp_wkv, cmp_wgate=ind_cmp_wgate, cmp_ape=ind_cmp_ape,
        cmp_kv_norm_w=ind_cmp_kv_norm_w, freqs_cis=freqs_cis_compress,
        inout_indexer_kv_cache=ind_kv_cache,
        inout_indexer_kv_state=ind_kv_state, inout_indexer_score_state=ind_score_state,
        start_pos=start_pos, offset=offset, ...)
elif compress_ratio == 128:
    topk_idxs_compressed = pk.get_compress_topk_idxs(...)   # algorithmic helper, see sparse_attn_topk_strided_*_layer.md
else:
    topk_idxs_compressed = None
```

V3 analog: none. V3's MLA attends to all KV; there is no top-k selection.

## Notes / risks

- `qr` is shared with the main attention's pre-RoPE Q (the output of `q_norm`). The
  builder must ensure this tensor is alive across both the main attn and indexer paths
  (V3 does similar for the `q_a_proj`/`q_a_layernorm` shared output).
- The Indexer's internal Compressor uses `rotate=True` (Hadamard rotation) and fp4-quant
  the result. **First design pass keeps BF16 + skips Hadamard** — accuracy may dip
  versus the reference; verify against a test that allows a small tolerance.
- The Indexer's internal Compressor has its **own** `kv_state`/`score_state`/`kv_cache`
  buffers — distinct from the main attention's. Don't share buffers.
- `index_topk = 512`, but `min(index_topk, end_pos // R)` can be smaller early in the
  sequence; the kernel must mask the unused entries to `-1` (sentinel; the
  `sparse_attn_topk_learned_*_layer` skips entries with index `-1`).
- For prefill the causal mask is dual: masks scores before topk **and** masks the resulting
  indices that exceed each row's causal limit. Both must be implemented.
- Verification: feed the same `(x, qr)` into the V4 reference Indexer and the MPK kernel,
  compare `topk_idxs` (set equality is enough since order within top-k is unspecified at
  ties; use a relaxed sorted comparison).
