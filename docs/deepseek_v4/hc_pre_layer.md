# `hc_pre_layer`

Hyper-Connection **pre** stage. Reduces `hc_mult` parallel hidden-state copies to a single
representation that feeds into the attn / FFN sub-block, and emits two sets of mixing weights
(`post`, `comb`) to be consumed later by [`hc_post_layer`](hc_post_layer.md).

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Function: `Block.hc_pre`, lines 674-682
- External helper: `kernel.hc_split_sinkhorn` (imported at line 12). Implements the
  Sinkhorn-normalized split into `pre / post / comb` from a `[..., (2 + hc_mult) * hc_mult]`
  mix vector. Iteration count = `args.hc_sinkhorn_iters` (=20 in V4-Flash-Base).

## Math

Let `H = hc_mult`, `D = dim`, `M = (2 + H) * H` (the "mix_hc" channel count).

```
x_flat = x.flatten(2)                                         # [B, S, H*D] in fp32
rsqrt  = (mean(x_flat^2, dim=-1) + norm_eps).rsqrt()          # [B, S, 1]
mixes  = (x_flat @ hc_fn.T) * rsqrt                           # [B, S, M]   (after RMS-style scale)
pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, # Sinkhorn-iter solve
                                    H, hc_sinkhorn_iters, hc_eps)
y      = sum(pre.unsqueeze(-1) * x_flat.view(B,S,H,D), dim=2) # [B, S, D]
```

`pre`, `post`, `comb` are produced jointly by `hc_split_sinkhorn`:
- `pre`  : `[B, S, H]`     — weights for the H→1 reduction
- `post` : `[B, S, H]`     — per-copy gain used in `hc_post`
- `comb` : `[B, S, H, H]`  — combination matrix used in `hc_post`

`hc_split_sinkhorn` interprets the `M = (2 + H) * H` channels as `[H, H, H, H, ...]` blocks
and runs `hc_sinkhorn_iters` rows-then-columns normalizations to produce doubly-stochastic
`pre` and `comb`, plus `post = sigmoid(scale[2] * mixes_post + base_post) + eps`.

The compute is fp32 throughout for numerical stability, then `y` is cast back to `x.dtype`
(BF16 in MPK).

## Python API (proposed)

```python
pk.hc_pre_layer(
    input: DTensor,                 # x_hc, BF16 [B, hc_mult, D]
    hc_fn:    DTensor,              # FP32 [M, hc_mult * D]
    hc_scale: DTensor,              # FP32 [3]
    hc_base:  DTensor,              # FP32 [M]
    output_y:    DTensor,           # BF16 [B, D]
    output_post: DTensor,           # FP32 [B, hc_mult]
    output_comb: DTensor,           # FP32 [B, hc_mult, hc_mult]
    grid_dim: tuple,
    block_dim: tuple,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    norm_eps: float = 1e-6,
)
```

Notes:
- The full `[B, S, ...]` shape collapses to `[B, ...]` in MPK because each step's batch is one
  token per sequence (S is folded into B at runtime, matching V3's MLA convention).
- `hc_fn`, `hc_scale`, `hc_base` are weight tensors, attached via `pk.attach_input`
  (use `_safe_attach` from `python/mirage/mpk/models/deepseek_v3/builder.py:459`).

## Inputs

| name | dtype | shape | source |
| --- | --- | --- | --- |
| `input` | BF16 | `[B, hc_mult, D]` | residual stream from previous block (or initial embedding broadcast to `hc_mult`) |
| `hc_fn` | FP32 | `[M, hc_mult * D]` where `M=(2+hc_mult)*hc_mult` | weight `block.hc_attn_fn` or `block.hc_ffn_fn` |
| `hc_scale` | FP32 | `[3]` | weight `block.hc_attn_scale` or `block.hc_ffn_scale` |
| `hc_base` | FP32 | `[M]` | weight `block.hc_attn_base` or `block.hc_ffn_base` |

For V4-Flash-Base: `hc_mult=4`, `D=4096`, `M=24`.

## Outputs

| name | dtype | shape | sink |
| --- | --- | --- | --- |
| `output_y` | BF16 | `[B, D]` | input to the following `rmsnorm_layer` (attn_norm / ffn_norm) |
| `output_post` | FP32 | `[B, hc_mult]` | held until `hc_post_layer` |
| `output_comb` | FP32 | `[B, hc_mult, hc_mult]` | held until `hc_post_layer` |

## Builder usage

Called twice per block — once before attention, once before FFN — with different weight sets:

```python
y, post_attn, comb_attn = pk.hc_pre_layer(
    input=x_hc, hc_fn=hc_attn_fn, hc_scale=hc_attn_scale, hc_base=hc_attn_base, ...)
# attn sub-block runs on y
x_hc = pk.hc_post_layer(o_proj_out, residual_hc=x_hc, post=post_attn, comb=comb_attn, ...)

y, post_ffn, comb_ffn = pk.hc_pre_layer(
    input=x_hc, hc_fn=hc_ffn_fn, hc_scale=hc_ffn_scale, hc_base=hc_ffn_base, ...)
# FFN sub-block runs on y
x_hc = pk.hc_post_layer(moe_out, residual_hc=x_hc, post=post_ffn, comb=comb_ffn, ...)
```

V3 analog: V3 does `x = residual + branch(x)`. There is no V3 layer that maps to this.

## Notes / risks

- The Sinkhorn loop is small (20 iters over an `H×H` matrix with `H=4`) but it is fp32 and
  branches between row/col normalizations — implement it as register-resident code, not a
  loop over global memory.
- `hc_fn` is **fp32** in the checkpoint (model.py:666-672 set fp32 default dtype). Loader
  must keep it fp32, not cast to BF16.
- The output of `hc_pre` is BF16; the subsequent `rmsnorm_layer` already accepts BF16.
- Verification: write a test-mode harness that compares `(y, post, comb)` against
  `Block.hc_pre` from the V4 reference. See `tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py`
  for the harness pattern.
