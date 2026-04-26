# `hc_head_layer`

Variant of [`hc_pre_layer`](hc_pre_layer.md) used **at the LM head only**. Reduces
`[B, hc_mult, D] → [B, D]` using a sigmoid-based weighting (no Sinkhorn iterations,
no `post` / `comb` outputs).

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Function: `ParallelHead.hc_head`, lines 729-736

## Math

With `H = hc_mult`, `D = dim`:

```
x_flat = x.flatten(2).float()                            # [B, H*D]
rsqrt  = (mean(x_flat^2, dim=-1) + norm_eps).rsqrt()     # [B, 1]
mixes  = (x_flat @ hc_head_fn.T) * rsqrt                 # [B, H]
pre    = sigmoid(mixes * hc_head_scale + hc_head_base) + hc_eps   # [B, H]
y      = sum(pre.unsqueeze(-1) * x.view(B, H, D), dim=1) # [B, D]
```

Note that here `hc_head_fn` has shape `[H, H*D]` (the `mix_hc` axis has size `H`, not
`(2+H)*H` like `hc_pre`), and `hc_head_scale` is a scalar (`[1]`). This is a strict
simplification of `hc_pre`: a sigmoid replaces the Sinkhorn solve.

## Python API (proposed)

```python
pk.hc_head_layer(
    input:    DTensor,                # BF16 [B, hc_mult, D]
    hc_fn:    DTensor,                # FP32 [hc_mult, hc_mult * D]
    hc_scale: DTensor,                # FP32 [1]
    hc_base:  DTensor,                # FP32 [hc_mult]
    output:   DTensor,                # FP32 [B, D]   (kept fp32 for the subsequent norm + lm_head)
    grid_dim: tuple,
    block_dim: tuple,
    hc_mult: int = 4,
    eps: float = 1e-6,
    norm_eps: float = 1e-6,
)
```

## Inputs

| name | dtype | shape | source |
| --- | --- | --- | --- |
| `input` | BF16 | `[B, hc_mult, D]` | `hc_post_layer`'s output of the final block |
| `hc_fn` | FP32 | `[hc_mult, hc_mult * D]` | weight `head.hc_head_fn` (or `mtp.hc_head_fn` if MTP added later) |
| `hc_scale` | FP32 | `[1]` | weight `head.hc_head_scale` |
| `hc_base` | FP32 | `[hc_mult]` | weight `head.hc_head_base` |

For V4-Flash-Base: `hc_mult=4`, `D=4096`.

## Outputs

| name | dtype | shape | sink |
| --- | --- | --- | --- |
| `output` | FP32 | `[B, D]` | input to the LM-head `rmsnorm_layer` (V4 stores `head.weight` in fp32; keeping `output` fp32 avoids a downcast/upcast pair) |

If kept BF16 to match V3 `embed_layer` → `rmsnorm_layer` → `lm_head` conventions, that is
acceptable; the upstream cast is tiny.

## Builder usage

Called once, at the very end of the model — between the last block's FFN-side `hc_post_layer`
and the final `rmsnorm_layer` + LM-head linear:

```python
last_x_hc = ...                                                 # [B, hc_mult, D]
x = pk.hc_head_layer(last_x_hc, head.hc_head_fn, head.hc_head_scale, head.hc_head_base, ...)
x = pk.rmsnorm_layer(x, final_norm.weight, ...)                 # reuse V3
logits = pk.linear_layer(x, lm_head.weight, ...)                # reuse V3
```

V3 analog: V3's LM head is `final_norm → lm_head` directly, with no HC reduction.

## Notes / risks

- The reference's `MTPBlock` (model.py:739-767) also uses `hc_head` (with its own
  `hc_head_fn`/`scale`/`base`). When MTP support is added later, this layer is reused with
  MTP weights.
- `hc_head_fn` shape differs from `hc_pre`'s `hc_fn`: `[H, H*D]` vs `[(2+H)*H, H*D]`. Do not
  share the same kernel signature naively.
- All compute is fp32; cast input to fp32 inside the kernel as in the reference.
- Verification: numeric equivalence with `ParallelHead.hc_head` at `world_size=1`.
