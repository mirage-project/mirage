# `q_normalize_layer`

Per-head **rsqrt-normalize** of Q after `wq_b`, applied across the head-dim axis. Inserted
between Q's low-rank-up projection and the RoPE step. Has no learned weight (unlike the
RMSNorm before `wq_b`), and is unique to V4 — V3 had no analog.

## V4 source

- File: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- Function: `Attention.forward`, line 498
- Snippet: `q *= rsqrt(q.square().mean(-1, keepdim=True) + eps)`

## Math

With `H = n_local_heads` (=64), `Dh = head_dim` (=512):

```
q  shape : [B, S, H, Dh]
rms_inv  : rsqrt(mean(q^2, dim=-1, keepdim=True) + norm_eps)   # [B, S, H, 1]
q_out    : q * rms_inv                                          # in-place
```

This is RMSNorm without a learned scale. Unlike `q_norm`/`kv_norm` (which act on the
`q_lora_rank`-dim or `head_dim`-dim of a 1D feature), this acts independently per head.

## Python API (proposed)

```python
pk.q_normalize_layer(
    input:  DTensor,   # BF16 [B, n_heads, head_dim]
    output: DTensor,   # BF16 [B, n_heads, head_dim]   (or in-place equivalent)
    grid_dim: tuple,
    block_dim: tuple,
    eps: float = 1e-6,
)
```

The MPK kernel can be implemented as either an out-of-place transform or fused into the
preceding `linear_fp8_layer` (the `wq_b` GEMM) — fusion is a follow-up optimization.

## Inputs

| name | dtype | shape | source |
| --- | --- | --- | --- |
| `input` | BF16 | `[B, n_heads, head_dim]` | output of `wq_b` (`linear_fp8_layer` reusing V3) |

## Outputs

| name | dtype | shape | sink |
| --- | --- | --- | --- |
| `output` | BF16 | `[B, n_heads, head_dim]` | input to RoPE inside the corresponding `sparse_attn_*_layer` |

## Builder usage

Called once per block, immediately after `wq_b`:

```python
qr  = pk.linear_fp8_layer(rms_out, wq_a, ...)             # reuse V3
qr  = pk.rmsnorm_layer(qr, q_norm.weight, ...)            # reuse V3
q   = pk.linear_fp8_layer(qr, wq_b, ...)                  # reuse V3 (output [B, H*Dh])
q   = pk.q_normalize_layer(q.view([B, H, Dh]), ...)       # NEW
# RoPE on q[..., -64:] is fused inside the chosen sparse_attn_*_layer
```

V3 analog: none. V3 does `q_a_proj → q_a_layernorm → q_b_proj`, then RoPE on the rope-dim
suffix; there is no per-head normalization between the up-projection and RoPE.

## Notes / risks

- `Dh = 512` with `n_heads = 64` per rank → reduction across 512 BF16 elements per head per
  token. This is small; consider implementing as part of a fused kernel with the
  preceding `wq_b` GEMM in a follow-up.
- BF16 → FP32 in the reduction; the cast back to BF16 happens at the multiply.
- Verification: compare against `q *= rsqrt(q.square().mean(-1) + eps)` with random Q
  inputs; numeric tolerance ~1e-3 due to BF16 round trip.
