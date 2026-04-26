# DELTA: RoPE theta — separate `compress_rope_theta` for compressed-attn layers

V4 uses two **different** sets of RoPE frequencies depending on the layer's
`compress_ratio`. V3 used a single set for all layers.

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Function: `Attention.__init__`, lines 475-482
- Snippet:
  ```python
  if self.compress_ratio:
      original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
  else:
      # disable YaRN and use base rope_theta in pure sliding-window attention
      original_seq_len, rope_theta = 0, args.rope_theta
  freqs_cis = precompute_freqs_cis(self.rope_head_dim, args.max_seq_len, original_seq_len,
                                   rope_theta, args.rope_factor, args.beta_fast, args.beta_slow)
  ```

V4-Flash-Base config:
- `rope_theta = 10000` (base, used when `compress_ratio == 0`; YaRN disabled)
- `compress_rope_theta = 160000` (used when `compress_ratio > 0`; YaRN enabled with
  `factor=16`, `original_max_position_embeddings=65536`)

## V3 baseline

V3's MLA kernels (`mla_decode_sm100.cuh`, `mla_prefill_sm100.cuh`, etc.) compute RoPE
inside the kernel using a single `rope_theta` constant taken from the model config.
There is no per-layer dispatch on RoPE constants.

## What changes for V4

Two separate `freqs_cis` precomputed tables are needed at the model level — one for the
"compress" RoPE and one for the "base" RoPE. The choice of which table is bound into a
given attention task is determined per layer by `compress_ratios[layer_id]`.

Because the V4 sparse-attn kernels accept `freqs_cis` as a tensor input (see all six
`sparse_attn_*_layer.md` docs), this is **not** a new kernel — it's a builder + weight
pre-compute change:

1. At model setup, the V4 builder calls `precompute_freqs_cis(...)` twice and registers
   both `freqs_cis_base` and `freqs_cis_compress` as device-side tensors.
2. For each block, depending on `compress_ratios[L]`, the builder binds the matching
   tensor to that layer's `sparse_attn_*_layer` and `compressor_*_layer` /
   `indexer_layer` calls.

## Proposed builder code path

```python
# Once per model, in the V4 builder __init__
freqs_cis_base = precompute_freqs_cis(
    rope_head_dim, max_seq_len,
    original_seq_len=0, base=rope_theta,
    factor=rope_factor, beta_fast=beta_fast, beta_slow=beta_slow,
)
freqs_cis_compress = precompute_freqs_cis(
    rope_head_dim, max_seq_len,
    original_seq_len=original_seq_len, base=compress_rope_theta,
    factor=rope_factor, beta_fast=beta_fast, beta_slow=beta_slow,
)
self.freqs_cis_base     = self._safe_attach(freqs_cis_base,     "freqs_cis_base")
self.freqs_cis_compress = self._safe_attach(freqs_cis_compress, "freqs_cis_compress")

# Per layer
for L in range(num_layers):
    r = compress_ratios[L]
    fc = self.freqs_cis_compress if r > 0 else self.freqs_cis_base
    # Bind `fc` into all RoPE-using layers for block L:
    # - sparse_attn_*_layer (forward Q/KV RoPE + inverse RoPE on output)
    # - compressor_*_layer (forward RoPE on the compressed KV)
    # - indexer_layer (forward RoPE on the indexer Q + indexer-internal compressor)
```

## Notes / risks

- `freqs_cis` is precomputed and immutable; binding two copies costs `2 × max_seq_len ×
  rope_head_dim × 8 bytes` (complex64) ≈ ~1 GB at `max_seq_len = 1M`. For first-pass
  testing, use a smaller `max_seq_len` (e.g., 8192) to keep memory bounded; bump for
  long-context evals.
- `precompute_freqs_cis` is `lru_cache(2)`-decorated in the reference (line 199); two
  parameter combos are exactly what V4 needs.
- `original_seq_len` for the **base** RoPE is `0` (YaRN disabled). For the **compress**
  RoPE it's the config's `original_max_position_embeddings = 65536`.
- The output stage of every sparse-attn kernel applies inverse RoPE on the rope dims
  using the same per-layer `freqs_cis`; ensure both directions use the same bound table.
- Verification: numerically the difference shows up as a phase difference in Q/K rope dims
  between layers with different `compress_ratio`. A unit test should pass `Q/K` for a
  ratio-0 layer and a ratio-4 layer and confirm the kernel produces the same output as
  the reference for both.
