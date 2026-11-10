# SGLang v0.5.22 (ROCm) GLM-5.3-Flash decoder-layer contract for our per-sublayer kernels

(2026-11-07, from an Explore agent reading the v0.5.22 ROCm image source; path:line under the sglang package root. To be
confirmed on the box: SGLANG_USE_AITER (source default False, the ROCm image normally 1), the checkpoint's gate_lower_bound.)

## Decoder layer (glm5_next.py:954-1010), mhc = True
- State: forward_batch.residual_stream (ResidualStream .residual / .pending, layer_boundary/residual/stream.py:106-175);
  h_res / h_post on a per-layer MHCState (residual/mhc.py:25-48), reached as layer.ffn_boundary.declaration.update.state.
- model start: embed -> residual_batch.start (fresh stream); layer 0 attn prepare: init residual = x.repeat(1, 4).
- attn_boundary.prepare: hc_attn_pre(R, input_layernorm) -> x_attn, h_res, h_post; stream.write(R) (mhc.py:54-64).
- self_attn: o_proj reduce_results=False (partial [T, 4096]).
- attn_boundary.finish: records the partial (no comm).
- ffn_boundary.prepare: TP all-reduce, then hc_ffn_post_pre (fused, T <= 16) or hc_post + hc_ffn_pre(post_attention_layernorm);
  stream.write(R') (prepare.py:121-180, mhc.py:66-94).
- mlp: MoE or dense, reduce_results=False.
- ffn_boundary.finish: all-reduce (sum_output), hc_post, clear_coefficients, last layer: hc_contract (mean over streams);
  stream.write(R''). FFN post + next attention pre never fused.
- after the last layer: hc_contract [T, H] bf16 -> final_norm RMSNorm -> lm_head.
Tensors: streams bf16 [T, 4 x 4096] stream-major; h_post fp32 [T, 4] (2 sigmoid); h_res fp32 [T, 16] [i_in, j_out] after
Sinkhorn; hc_post out_j = h_post_j x + sum_i h_res[i, j] R_i; hc params fp32 (fn [24, 16384], scale [3], base [24]);
x_attn / x_ffn bf16 [T, 4096].
Kernels on HIP: hc_pre aiter mhc_pre (norm fused) when SGLANG_USE_AITER + gfx95, else torch; hc_post aiter mhc_post;
hc_ffn_post_pre fused at T <= 16 (aiter mhc_fused_post_pre / Triton / TileLang).

## All-reduce
attention: ffn_boundary.prepare -> attn_tp_all_reduce (custom / quick AR); FFN: sum_output (MoE: sum_post_experts_output;
dense: TP all-reduce). Not fused with the mHC post.

## KDA layer (Glm5NextLinearAttention, glm5_next.py:341-682)
projections: fused bf16 fused_qkvbfg_a_proj -> [qkv, beta, fg_a] + fused_fg_b_proj (unquantized), else separate.
decode: HybridLinearAttnBackend.forward_decode -> KDAAttnBackend.forward_decode (attention/linear/kda_backend.py:751-999):
Triton causal_conv1d_update (conv_state_indices = cache_indices), then fused_recurrent_kda_packed_decode (lower_bound None)
or fused_sigmoid_gating_delta_rule_update (is_kda, lower_bound); o_norm FusedRMSNormGated (sigmoid g); o_proj.
state: req_to_token_pool.mamba2_layer_cache(layer_id): .conv[0] [slots + 1, 3, 3 proj / tp] bf16 (kernel-major), .temporal
[slots + 1, H_loc, 128 V, 128 K] fp32; slot 0 padding; strided views (envelope layout, memory_pool.py:690-718).
slots: linear_attn_backend.forward_metadata.mamba_cache_indices int32 [bs] (graph buffer state_indices_list[bs - 1],
padding rows -1). _track_mamba_state_decode copies states for the prefix cache (hybrid:975-990).

## MoE (DeepseekV2MoE, deepseek_v2.py:628-1430)
gate bf16 (aiter_dsv3_router_gemm); correction bias bf16 under aiter + fp8; aiter biased_grouped_topk (sigmoid, bias,
renorm, x2.5 folded); shared expert fused as expert 288 weight 1.0 (aiter, gfx950, TP4/8, block FP8, GLM geometry), else
a separate TP MLP; experts aiter fused_moe (128 x 128 FP8, swiglu_limit); output reduced in ffn_boundary.finish.

## DSA per-step getters
forward_batch .seq_lens .req_pool_indices .out_cache_loc .positions; get_attn_backend() (forward_context.py:66)
.token_to_kv_pool (HybridLinearKVPool) / .req_to_token_pool.req_to_token; pool get_key_buffer(layer_id) (latent fp8),
get_index_k_with_scale_buffer, get_compress_tail_buffers; metadata full_attn_backend.forward_metadata
(.cache_seqlens_int32, .page_table_1, .real_page_table) or get_indexer_metadata(layer_id, fb). Internals:
sglang_v0522_glm_dsa_contract.md.

## Hook points
(a) attention sublayer: replace glm5_next.py:970-985 after attn_boundary.prepare: kernel A(x_attn, R, st.h_post, st.h_res)
-> stream.write(R'); st.h_post, st.h_res = FFN coefficients; hidden = x_ffn; mlp + ffn_boundary.finish unchanged.
(b) FFN sublayer: in the model loop (glm5_next.py:1179-1208): skip mlp, ffn_boundary.finish and layer i+1's
attn_boundary.prepare; stream.write(R''); st_i.clear_coefficients(); next layer's h_post / h_res on its state; start layer
i+1 at self_attn; stream.check needs the hidden handle passed on to be stream.residual; last layer: write the contracted
[T, H].
Keep consistent: KDA states at cache_indices (+ track copy if the radix extra buffer is on); DSA latent at out_cache_loc,
index-K pages, tail rings; never write req_to_token / seq_lens.
Graph capture: FullCudaGraphBackend, one graph per bs (sizes 1, 2, 4, 8, 12, 16..256 step 8); static buffers for input_ids,
req_pool_indices, seq_lens, out_cache_loc; kernels launched on the current stream inside forward are captured; no Python at
replay: stable pointers, no host sync.
