# DeepSeek-V4-Flash-Base on MPK — what's new vs V3

This document captures the architectural diff between DeepSeek V3 (already supported in MPK
via `python/mirage/mpk/models/deepseek_v3/builder.py` + `demo/deepseek_v3/demo.py`) and
**DeepSeek-V4-Flash-Base**, and lists every layer that needs to be added or modified to
support V4 in MPK.

## References

- V4 modeling (Python reference): `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`
- V4-Flash-Base config: `deps/deepseek_v4/DeepSeek-V4-Flash-Base/config.json`
- V4-Flash-Base is a smaller / faster variant of DeepSeek-V4-Flash; the modeling code is shared
  (the two share `inference/model.py`). The only differences are hyperparameter values from the
  config (e.g., `n_layers=43`, `compress_ratios=[0,0,4,128,...]`).

## V4-Flash-Base hyperparameters at a glance

| Field | Value | Notes |
| --- | --- | --- |
| `vocab_size` | 129280 | same as V3 |
| `hidden_size` (`dim`) | 4096 | V3 was 7168 |
| `num_hidden_layers` | 43 | V3 was 61 |
| `num_attention_heads` | 64 | V3 was 128 |
| `num_key_value_heads` | 1 | MQA (V3 was 128 = MHA) |
| `head_dim` | 512 | V3 split: nope=128, rope=64, v=128 |
| `qk_rope_head_dim` | 64 | nope_head_dim = 512 - 64 = 448 |
| `q_lora_rank` | 1024 | V3 was 1536 |
| `o_lora_rank` | 1024 | NEW (V3 had no O low-rank) |
| `o_groups` | 8 | NEW (grouped low-rank O projection) |
| `sliding_window` | 128 | NEW |
| `compress_ratios` | `[0, 0, 4, 128, 4, 128, ..., 4, 0]` | per-layer (43 entries) |
| `compress_rope_theta` | 160000 | NEW (separate RoPE theta for compressed KV) |
| `rope_theta` | 10000 | base (used when compress_ratio==0) |
| `index_n_heads` | 64 | NEW (Indexer's heads) |
| `index_head_dim` | 128 | NEW |
| `index_topk` | 512 | NEW |
| `n_routed_experts` | 256 | same as V3 |
| `n_shared_experts` | 1 | same as V3 |
| `num_experts_per_tok` | 6 | V3 was 8 |
| `routed_scaling_factor` | 1.5 | V3 was 2.5 |
| `scoring_func` | `sqrtsoftplus` | V3 was `sigmoid` |
| `num_hash_layers` | 3 | NEW |
| `swiglu_limit` | 10.0 | NEW |
| `hc_mult` | 4 | NEW (Hyper-Connection copies) |
| `hc_sinkhorn_iters` | 20 | NEW |
| `hc_eps` | 1e-6 | NEW |
| `quantization_config` | FP8 e4m3, `[128,128]` block, ue8m0 scales | matches V3's FP8 path |
| `max_position_embeddings` | 1048576 | 1M context (V3: 163840) |

## What is new in V4 (architectural diff)

### 1. Hyper-Connections (mHC) replace simple residuals

Source: `Block` (lines 648-701), `ParallelHead.hc_head` (lines 729-736),
helper kernel `kernel.hc_split_sinkhorn` (imported at line 12).

Each block keeps `hc_mult=4` parallel hidden-state copies. Around the attention sub-block:

```
residual_hc = x                                           # [B, S, hc_mult, D]
x, post, comb = hc_pre(x, hc_attn_fn, hc_attn_scale, hc_attn_base)   # contracts hc->1
x = attn_norm(x); x = attn(x, start_pos)                  # standard branch on [B, S, D]
x = hc_post(x, residual_hc, post, comb)                   # expands 1->hc_mult
```

The same pattern wraps the FFN sub-block. The LM head uses a simpler `hc_head`
(sigmoid-based, no Sinkhorn) before the linear projection.

### 2. Hybrid sparse attention replaces V3's dense MLA

Source: `Attention` (lines 436-543). Three per-layer modes selected by `compress_ratios[layer_id]`:

| compress_ratio | Compressor | Indexer | top-k source |
| --- | --- | --- | --- |
| 0 | none | none | window only (`get_window_topk_idxs`) |
| 4 | yes (overlap) | yes (learned scorer) | window ∪ Indexer top-k |
| 128 | yes (no overlap) | none | window ∪ `get_compress_topk_idxs` (algorithmic) |

Per-layer attention forward (line 484-543):

1. `qr = q_norm(wq_a(x))` — Q low-rank projection + RMSNorm (analog of V3's `q_a_proj` + `q_a_layernorm`).
2. `q = wq_b(qr).unflatten(-1, (n_heads, head_dim))` — Q upscale.
3. **`q *= rsqrt(q.square().mean(-1) + eps)`** — per-head rsqrt normalization. **NEW** (no V3 analog).
4. `apply_rotary_emb(q[..., -rd:], freqs_cis)` — RoPE on the last `rope_head_dim` dims of Q.
5. `kv = kv_norm(wkv(x))` — single K=V vector per token (MQA, `head_dim=512`).
6. `apply_rotary_emb(kv[..., -rd:], freqs_cis)` — RoPE on KV.
7. `topk_idxs = window_topk_idxs(...)` — sliding-window indices.
8. If `compress_ratio>0`: append Indexer's (or strided helper's) compressed top-k indices.
9. If `start_pos==0`: write KV into ring buffer; if compress_ratio>0, run Compressor and append compressed KV.
   Else: append KV at `start_pos % win`; run Compressor incrementally.
10. `o = sparse_attn(q, kv_buf, attn_sink, topk_idxs, scale)` — sparse attention with sink tokens.
11. `apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)` — inverse RoPE on output.
12. `o = einsum("bsgd,grd->bsgr", o.view(..., n_groups, -1), wo_a)` then `wo_b(o.flatten(2))` — grouped low-rank O.

### 3. MoE Gate has new scoring + hash routing

Source: `Gate` (lines 546-584).

- Scoring function defaults to `sqrtsoftplus` (`sqrt(softplus(scores))`); V3 used `sigmoid`.
- For `layer_id < num_hash_layers (=3)`: indices come from a learned table
  `tid2eid[input_ids]` (one fixed expert assignment per token id, no per-token scoring).
  `weights` still come from `gather(original_scores, indices)` and are L1-normalized.
- For other layers: `indices = (scores + bias).topk(k)`. The bias only affects which experts
  are picked, not the routing weights (V3's `e_score_correction_bias` had similar semantics
  but combined with sigmoid scoring).
- `route_scale=1.5` (V3: 2.5).

### 4. Expert SwiGLU clamp

Source: `Expert` (lines 587-606). Before SiLU, gate is `min(gate, swiglu_limit)`; up is
`clamp(up, -swiglu_limit, swiglu_limit)`. This is the only change to the dense FFN compute
vs V3's pure `silu_mul`.

### 5. Per-layer compressed RoPE theta

Source: `Attention.__init__` (lines 475-482). Layers with `compress_ratio>0` build
`freqs_cis` with `compress_rope_theta=160000` and YaRN scaling. Layers with
`compress_ratio==0` build `freqs_cis` with `rope_theta=10000` and YaRN disabled
(`original_seq_len=0`).

### 6. Items deferred from this design pass (follow-ups)

- `ParallelEmbedding` / `ParallelHead` (TP sharding) — for the first design pass we use V3's
  existing `embed_layer` and the head pipeline, with `world_size=1`.
- `MTPBlock` (`Block` subclass with `e_proj`/`h_proj` + `hc_head`) — speculative decoding for
  V4 will be a follow-up; the current pass only covers the main 43 blocks.
- FP4 / `fp4_act_quant` paths used in `Indexer.compressor` (Hadamard rotation +
  `fp4_act_quant`) — kept as BF16 in this design.
- `rotate_activation` (Hadamard transform) used inside Indexer's internal Compressor.

## File map: new/modified layers ↔ V4 source

| Doc | V4 source | V3 analog | Status |
| --- | --- | --- | --- |
| [`hc_pre_layer.md`](hc_pre_layer.md) | `Block.hc_pre` (model.py:674-682) | none | NEW |
| [`hc_post_layer.md`](hc_post_layer.md) | `Block.hc_post` (model.py:684-687) | none | NEW |
| [`hc_head_layer.md`](hc_head_layer.md) | `ParallelHead.hc_head` (model.py:729-736) | none | NEW |
| [`compressor_prefill_layer.md`](compressor_prefill_layer.md) | `Compressor.forward` start_pos==0 branch (model.py:325-342, 360-377) | none | NEW |
| [`compressor_decode_layer.md`](compressor_decode_layer.md) | `Compressor.forward` start_pos>0 branch (model.py:343-377) | none | NEW |
| [`indexer_layer.md`](indexer_layer.md) | `Indexer.forward` (model.py:402-433) | none | NEW |
| [`q_normalize_layer.md`](q_normalize_layer.md) | `Attention.forward` line 498 | none | NEW |
| [`grouped_lowrank_o_proj_layer.md`](grouped_lowrank_o_proj_layer.md) | `Attention.forward` lines 537-542 (`wo_a` einsum + `wo_b`) | V3 single `o_proj` | NEW |
| [`sparse_attn_window_prefill_layer.md`](sparse_attn_window_prefill_layer.md) | `Attention.forward` start_pos==0 with compress_ratio==0 | `mla_prefill_layer` | NEW |
| [`sparse_attn_window_decode_layer.md`](sparse_attn_window_decode_layer.md) | `Attention.forward` start_pos>0 with compress_ratio==0 | `mla_mtp_decode_layer`+reduce | NEW |
| [`sparse_attn_topk_learned_prefill_layer.md`](sparse_attn_topk_learned_prefill_layer.md) | `Attention.forward` start_pos==0 with compress_ratio==4 | `mla_prefill_layer` | NEW |
| [`sparse_attn_topk_learned_decode_layer.md`](sparse_attn_topk_learned_decode_layer.md) | `Attention.forward` start_pos>0 with compress_ratio==4 | `mla_mtp_decode_layer`+reduce | NEW |
| [`sparse_attn_topk_strided_prefill_layer.md`](sparse_attn_topk_strided_prefill_layer.md) | `Attention.forward` start_pos==0 with compress_ratio==128 | `mla_prefill_layer` | NEW |
| [`sparse_attn_topk_strided_decode_layer.md`](sparse_attn_topk_strided_decode_layer.md) | `Attention.forward` start_pos>0 with compress_ratio==128 | `mla_mtp_decode_layer`+reduce | NEW |
| [`moe_topk_sqrtsoftplus_routing_layer.md`](moe_topk_sqrtsoftplus_routing_layer.md) | `Gate.forward` non-hash branch (model.py:565-583) | `moe_topk_sigmoid_routing_layer` | NEW |
| [`moe_topk_hash_routing_layer.md`](moe_topk_hash_routing_layer.md) | `Gate.forward` hash branch (model.py:565,572,576-583) | none | NEW |
| [`silu_mul_swiglu_limit_delta.md`](silu_mul_swiglu_limit_delta.md) | `Expert.forward` lines 596-606 | `silu_mul_layer` / `moe_silu_mul_layer` | DELTA |
| [`compress_rope_theta_delta.md`](compress_rope_theta_delta.md) | `Attention.__init__` lines 475-482 | RoPE constants in `mla_*_sm100.cuh` | DELTA |

Reused as-is (no doc): `embed_layer`, `rmsnorm_layer`, `linear_layer` /
`linear_with_residual_layer` / `linear_fp8_layer` / `linear_fp8_with_residual_layer`,
`quantize_fp8_layer`, `moe_w13_fp8_layer`, `moe_w2_fp8_layer`, `moe_mul_sum_add_layer`,
`elementwise_add_layer`, `argmax_partial_layer`, `argmax_reduce_layer`,
`sampling_sm100_layer`, `allreduce_layer`.

## V4 per-block forward flow (mapped to MPK layers)

For one V4 block with `compress_ratio = r ∈ {0, 4, 128}`, layer index `L`, and
`hash_layer = (L < num_hash_layers)`:

```
# input: x_hc [B, hc_mult, D] (D=4096)

# ---------- Attention sub-block ----------
residual_hc = x_hc
x, post, comb = hc_pre_layer(x_hc, hc_attn_fn, hc_attn_scale, hc_attn_base)   # NEW
x = rmsnorm_layer(x, attn_norm.weight)                                        # reuse V3

# Q
qr = linear_fp8_layer(x, wq_a) → rmsnorm_layer(_, q_norm.weight) → qr         # reuse V3
q  = linear_fp8_layer(qr, wq_b)                                               # reuse V3
q  = q_normalize_layer(q)                                                     # NEW
# RoPE on q[..., -64:] is fused inside sparse_attn_*_layer (see compress_rope_theta_delta.md)

# KV
kv = linear_fp8_layer(x, wkv) → rmsnorm_layer(_, kv_norm.weight) → kv         # reuse V3
# RoPE on kv[..., -64:] is fused inside sparse_attn_*_layer

# Compressor (only if r > 0)
if r > 0:
    if start_pos == 0:
        compressor_prefill_layer(...)                                         # NEW
    else:
        compressor_decode_layer(...)                                          # NEW

# Indexer (only if r == 4)
if r == 4:
    compress_topk_idxs = indexer_layer(x, qr, ...)                            # NEW

# Sparse attention
if r == 0 and start_pos == 0: o = sparse_attn_window_prefill_layer(...)
if r == 0 and start_pos > 0:  o = sparse_attn_window_decode_layer(...)
if r == 4 and start_pos == 0: o = sparse_attn_topk_learned_prefill_layer(...)
if r == 4 and start_pos > 0:  o = sparse_attn_topk_learned_decode_layer(...)
if r ==128 and start_pos==0:  o = sparse_attn_topk_strided_prefill_layer(...)
if r ==128 and start_pos>0:   o = sparse_attn_topk_strided_decode_layer(...)
# Each sparse_attn_*_layer applies inverse RoPE on the rope dims of o internally.

# Output projection (NEW: grouped low-rank)
x = grouped_lowrank_o_proj_layer(o, wo_a, wo_b)                               # NEW

x_hc = hc_post_layer(x, residual_hc, post, comb)                              # NEW

# ---------- FFN sub-block ----------
residual_hc = x_hc
x, post, comb = hc_pre_layer(x_hc, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)      # NEW
x = rmsnorm_layer(x, ffn_norm.weight)                                         # reuse V3

# Gate
router_logits = linear_layer(x, gate.weight)                                  # reuse V3 (BF16)
if hash_layer:
    weights, indices, mask = moe_topk_hash_routing_layer(router_logits, x_input_ids, tid2eid, ...)
                                                                              # NEW
else:
    weights, indices, mask = moe_topk_sqrtsoftplus_routing_layer(router_logits, gate.bias, ...)
                                                                              # NEW

# Routed experts (FP8 group GEMM, reuse V3 kernels)
x_fp8, x_scale = quantize_fp8_layer(x, ...)                                   # reuse V3
gate_up = moe_w13_fp8_layer(x_fp8, x_scale, w13_fp8, w13_scale, indices, mask, ...)
                                                                              # reuse V3
silu = moe_silu_mul_layer(gate_up, swiglu_limit=10.0)                         # MODIFIED via silu_mul_swiglu_limit_delta.md
down = moe_w2_fp8_layer(silu_fp8, silu_scale, w2_fp8, w2_scale, indices, mask, ...)
                                                                              # reuse V3

# Shared expert
shared_gate_up = linear_fp8_layer(x, shared.gate_up)                          # reuse V3
shared_silu = silu_mul_layer(shared_gate_up, swiglu_limit=10.0)               # MODIFIED via silu_mul_swiglu_limit_delta.md
shared_out = linear_fp8_layer(shared_silu, shared.down)                       # reuse V3

# Combine
y = moe_mul_sum_add_layer(down, weights, shared_out)                          # reuse V3

x_hc = hc_post_layer(y, residual_hc, post, comb)                              # NEW
```

After the last block: an `hc_head_layer` (NEW) reduces `[B, hc_mult, D] → [B, D]`, followed by
`rmsnorm_layer` (reuse) and the LM head linear (reuse).

## Per-file content template

Each per-layer doc follows this structure: V4 source citation, math summary, proposed Python
API on `PersistentKernel`, input/output tensor tables, builder usage (where called and the
dispatch condition), and notes/risks (FP8 quant boundaries, RoPE dtype, attention-sink
handling, etc.).
