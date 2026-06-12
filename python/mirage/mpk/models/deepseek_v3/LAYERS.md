# DeepSeek V3 MPK Layers

Layers registered by `DeepSeekV3Builder` in `builder.py`.

Architecture: 61 decoder layers (0–2 dense MLP, 3–60 MoE), MLA attention, optional MTP.

---

## 1. Embedding

| Layer | Purpose |
|---|---|
| `embed_layer` | Token embedding lookup: `input_tokens → embed_out` |

---

## 2. Normalization

| Layer | Purpose |
|---|---|
| `rmsnorm_layer` | RMSNorm (pre/post-attn, pre-MLP, final model norm) |
| `fused_rmsnorm_quantize_fp8_layer` | Fused RMSNorm + FP8 quantize in one pass |

---

## 3. MLA Attention

### RoPE / Q assembly
| Layer | Purpose |
|---|---|
| `deepseek_mla_rope_q_fused_layer` | RoPE for fused Q (decode path, no LoRA split) |
| `deepseek_mla_rope_q_split_layer` | RoPE for split Q (prefill path) |
| `deepseek_mla_rope_k_layer` | RoPE for K |
| `assemble_q_decode_sm100_layer` | Assemble full Q from nope+rope parts (decode) |

### KV cache gather
| Layer | Purpose |
|---|---|
| `mla_kv_gather_layer` | Gather paged KV cache entries (legacy decode) |
| `mla_kv_gather_unified_layer` | Unified paged KV gather (prefill + decode) |

### Decode attention (TP variants)
| Layer | Purpose |
|---|---|
| `mla_mtp_decode_layer` | MLA decode attention, TP=1 |
| `mla_mtp_reduce_layer` | Reduce split-KV partials, TP=1 |
| `mla_mtp_decode_tp2_layer` | MLA decode, TP=2 |
| `mla_mtp_decode_tp2_reduce_layer` | Reduce partials, TP=2 |
| `mla_mtp_decode_tp4_layer` | MLA decode, TP=4 |
| `mla_mtp_decode_tp4_reduce_layer` | Reduce partials, TP=4 |
| `mla_mtp_decode_tp8_layer` | MLA decode, TP=8 |
| `mla_mtp_decode_tp8_reduce_layer` | Reduce partials, TP=8 |

### Prefill attention
| Layer | Purpose |
|---|---|
| `mla_prefill_absorbed_layer` | Prefill MLA with absorbed KV weights |
| `mla_prefill_tp8_chunked_layer` | Chunked prefill attention, TP=8 |

---

## 4. Dense Linear (BF16 + FP8)

| Layer | Purpose |
|---|---|
| `linear_layer` | Standard GEMM (BF16) |
| `linear_with_residual_layer` | GEMM + residual add (BF16) |
| `splitk_linear_layer` | Split-K GEMM for small-M decode (BF16) |
| `linear_fp8_bmm_sm100_layer` | FP8 BMM (batched matmul), Blackwell |
| `linear_fp8_bmm_dense_sm100_layer` | FP8 BMM dense variant, Blackwell |
| `linear_splitk_swapAB_fp8_layer` | FP8 split-K with transposed operands |
| `fp8_gemm_dense_decode_splitk_layer` | FP8 dense GEMM with split-K for decode |

---

## 5. FP8 Quantization

| Layer | Purpose |
|---|---|
| `quantize_fp8_layer` | Quantize BF16 activations to FP8 |
| `fused_rmsnorm_quantize_fp8_layer` | Fused RMSNorm + FP8 quantize *(also in §2)* |

---

## 6. Activation

| Layer | Purpose |
|---|---|
| `silu_mul_layer` | SiLU(gate) × up elementwise (dense MLP) |

---

## 7. MoE MLP

### Routing & dispatch
| Layer | Purpose |
|---|---|
| `tensor_init_layer` | Initialize MoE metadata / dispatch buffers |
| `moe_topk_sigmoid_routing_layer` | Top-K expert selection via sigmoid gating |
| `moe_permute_sm100_layer` | Permute tokens into expert-contiguous order |

### Expert compute
| Layer | Purpose |
|---|---|
| `moe_w13_fp8_layer` | Fused gate+up projection for all experts (FP8 group GEMM) |
| `moe_silu_mul_layer` | SiLU(gate) × up for MoE experts |
| `moe_w2_fp8_layer` | Down projection for all experts (FP8 group GEMM) |
| `fp8_group_gemm_layer` | Generic FP8 group GEMM (shared expert path) |

### Output aggregation
| Layer | Purpose |
|---|---|
| `moe_unpermute_sm100_layer` | Unpermute expert outputs back to token order |
| `moe_mul_sum_add_layer` | Weighted sum of expert outputs + residual add |

---

## 8. Tensor-Parallel / Multi-GPU

| Layer | Purpose |
|---|---|
| `allreduce_layer` | Ring/reduce-scatter allreduce across TP ranks |
| `nvshmem_global_argmax_layer` | NVSHMEM-based global argmax across TP ranks |

---

## 9. Elementwise / Utility

| Layer | Purpose |
|---|---|
| `elementwise_add_layer` | Element-wise addition (residual or diagnostic) |
| `identity_layer` | No-op passthrough (phantom bridge for DAG shaping) |

---

## 10. ArgMax / Token Selection

| Layer | Purpose |
|---|---|
| `argmax_partial_layer` | Per-worker partial argmax over vocab dim |
| `argmax_reduce_layer` | Reduce partial results to final output token |
| `softmax_gather_layer` | Softmax over logits (for sampling / MTP verify) |
| `prob_scatter_layer` | Scatter token probabilities into buffer |
| `prob_extract_layer` | Extract per-token probabilities from buffer |

---

## 11. MTP (Multi-Token Prediction)

| Layer | Purpose |
|---|---|
| `mtp_build_embed_input_layer` | Construct draft token embedding input for next iter |
| `mtp_prepare_verify_layer` | Prepare verification inputs (query indptr etc.) |
| `mtp_verify_strict_layer` | Strict greedy verification: accept prefix matching target |
| `mtp_verify_probabilistic_layer` | Probabilistic (speculative sampling) verification |
| `mtp_accept_commit_layer` | Commit accepted tokens and advance step counter |
| `mtp_token_scatter_layer` | Scatter accepted token IDs into token buffer |
| `mtp_float_scatter_layer` | Scatter accepted token probabilities into buffer |

---

## Data Flow Summary

```
input_tokens
  → embed_layer                                    # token embedding

  → [×num_layers]
      ┌── Pre-attn norm ──────────────────────────
      │   rmsnorm / fused_rmsnorm_quantize_fp8
      │   linear_* (QKV-A fused GEMM)
      ├── MLA Attention ──────────────────────────
      │   deepseek_mla_rope_q_* + rope_k
      │   mla_kv_gather_unified
      │   mla_mtp_decode[_tpN] + reduce[_tpN]      # decode
      │   mla_prefill_absorbed                      # prefill
      ├── Post-attn projection ───────────────────
      │   linear_with_residual / linear_fp8_*
      │   [allreduce]                               # TP sync
      ├── Pre-MLP norm ───────────────────────────
      │   rmsnorm / fused_rmsnorm_quantize_fp8
      ├── MLP (layers 0-2: dense, 3-60: MoE) ────
      │   Dense: linear → silu_mul → linear_with_residual
      │   MoE:   moe_topk_sigmoid_routing
      │           → moe_permute
      │           → moe_w13_fp8 → moe_silu_mul
      │           → moe_w2_fp8
      │           → moe_unpermute → moe_mul_sum_add
      │   [allreduce]                               # TP sync
      └────────────────────────────────────────────

  → rmsnorm (final norm)
  → linear (lm_head)                               # vocab logits
  → argmax_partial + argmax_reduce                 # greedy decode
     OR softmax_gather + prob_scatter/extract       # sampling
  → output_tokens

  [MTP path, optional]
  → mtp_build_embed_input
  → [draft forward: embed → MLA → MoE → norm → lm_head → argmax]
  → mtp_prepare_verify → mtp_verify_strict/probabilistic
  → mtp_accept_commit → mtp_token_scatter / mtp_float_scatter
```
