# SGLang v0.5.22 (ROCm, MI355X) per-layer decode contract for GLM-5.3 (zai-org/GLM-5.3, GlmMoeDsaForCausalLM, block-FP8)

(2026-11-10, from reading the SGLang v0.5.22 source; no run. Citations are `path:line` under the sglang package root
(`.../sgl0522/sglang` = `python/sglang`). Server: `--tp-size 8 --ep-size 1 --page-size 64 --kv-cache-dtype fp8_e4m3
--dsa-prefill-backend tilelang --dsa-decode-backend tilelang --moe-runner-backend aiter`, `SGLANG_USE_AITER=1`, plain decode.
Model config: amd/loop/glm53_config.json. Same format as sglang_v0522_glm_dsa_contract.md / sglang_v0522_glm_layer_contract.md
(GLM-5.3-Flash). T = decode rows; every GPU holds all T rows (no DP attention, no input scatter on HIP).
aiter is not installed on this machine: every aiter kernel's internals are UNVERIFIED and described from the SGLang call site.)

## 0. One decode step, per GPU (8 GPUs, T rows, H 6144, 8 local heads)
```
h0  = all_reduce(embed_shard(ids))                         # bf16 [T, 6144]; vocab 19360 rows per GPU
for L in 0..77:
  x, res = AR_add_rmsnorm(ffn_partial[L-1], res, input_layernorm[L])   # L=0: rmsnorm(h0), res = h0
  # attention (MLA, 8 heads)
  qkv  = fp8_gemm(x, q_a|kv_a [2624, 6144])                # replicated
  q, kv = rmsnorm(q[:, :2048]), rmsnorm(qkv[:, 2048:2560]) # k_pe = qkv[:, 2560:2624]
  q    = fp8_gemm(q, q_b [2048, 2048]) -> [T, 8, 256]      # q_nope 192 | q_pe 64
  topk = indexer(x, q_lora=q_normed)  if indexer_types[L] == "full"   # int32 [T, 2048] physical KV slots
         topk of layer L-1            if "shared"           # 3,4,5 <- 2; 7,8,9 <- 6; ...; 75,76,77 <- 74
  q_cat = fused(q_nope x w_kc[8,192,512], rope(q_pe), rope(k_pe), write [kv 512 | k_pe 64] fp8 to cache)  # fp8 [T, 8, 576]
  o    = sparse_mla(q_cat, kv_cache[L], topk, scale 1/16)  # bf16 [T, 8, 512]
  o    = fp8_bmm(o, w_vc[8,512,256]) -> [T, 2048]; p = fp8_gemm(o, o_proj [6144, 2048])  # partial [T, 6144]
  h, res = AR_add_rmsnorm(p, res, post_attention_layernorm[L])
  ffn_partial[L] = dense_mlp(h)   if L < 3                 # gate_up [3072, 6144], silu*up, down [6144, 1536]
                   fused_moe(h)   else                     # 257 experts (256 = shared, weight 1.0), top-9, inter 256 per GPU
h = rmsnorm_add(all_reduce(ffn_partial[77]), res, model.norm); logits = all_gather(h @ lm_head_shard^T)  # fp32 [T, 154880]
```
Three layer kinds asked for (each section below gives the ops in order):
| Layer kind | Example layers | Attention | Indexer | FFN | All-reduces |
|---|---|---|---|---|---|
| dense, full top-k | 0, 1, 2 | MLA (section 2) | runs (section 3, D3) | `DeepseekV2MLP` (section 4, F.2) | 2, bf16 [T, 6144] |
| MoE, "full" indexer | 6, 10, ..., 74 | MLA | runs; returns topk (layer L+1 is shared) | `DeepseekV2MoE` (section 4, F.1/F.3/F.4) | 2, bf16 [T, 6144] |
| MoE, "shared" indexer | 3-5, 7-9, ..., 75-77 | MLA | no module; uses carried topk | `DeepseekV2MoE` | 2, bf16 [T, 6144] |
- `indexer_types` "full" = layers 0, 1, 2, 6, 10, ..., 74 (21 layers); the rest "shared" (config; `srt/configs/model_config.py:410-418`).
- Dense = layers 0-2 (`first_k_dense_replace 3`), MoE = 3-77 (`srt/models/deepseek_v2.py:2758-2766`). Layer 78 (MTP) is not built
  without speculative decoding (`num_hidden_layers 78`, `srt/models/deepseek_v2.py:2936-2945`).

## 1. Layer boundaries, all-reduce, model start/end (GLM-5.3, HIP gfx950, SGLANG_USE_AITER=1, TP8 EP1, no DP attention, plain decode)

Paths are relative to the sglang package root (`.../sgl0522/sglang`). T = decode rows per GPU (all 8 GPUs hold the same T rows; no input scatter on HIP).

### L.0 Which residual model applies
| Fact | Value | Cite |
|---|---|---|
| Layer class | `DeepseekV2DecoderLayer` for all 78 layers (GlmMoeDsaForCausalLM = DeepseekV2ForCausalLM) | `srt/models/glm4_moe.py:1289-1290`; `srt/models/deepseek_v2.py:2936-2945` |
| mHC vs plain residual | No config attribute is read: DeepseekV2 always declares the plain residual. `stage_facts` calls `declare_attn()` / `declare_ffn()` with defaults (read NORM_QUANT_READOUT / NORM_READOUT, update PLAIN_ADD). The mHC path belongs to the separate Glm5Next model class. `init_context(q_lora_rank, is_dsa)` is called with `is_mhc` left False | `srt/models/deepseek_v2.py:2668-2690, 3286`; `srt/layers/layer_boundary/factories.py:433-445, 492-503`; `srt/layers/layer_boundary/adapters/attention.py:78` |
| Residual ops | attention read = `NormQuantReadout` (norm + optional quant), FFN read = `NormReadout`, both updates = `PlainAdd` (`hidden_states += residual`) | `srt/layers/layer_boundary/residual/add_norm.py:273-287, 328-421, 450-462` |
| Input scatter | Off on HIP (`allow_input_scattered` needs `_is_cuda or _is_npu`; DSA without mHC also excluded) | `srt/layers/layer_boundary/adapters/attention.py:84-91` |
| Backend fusions object | None (`CuteDSLFusion` only when `_is_cuda`) | `srt/models/deepseek_v2.py:2559-2560, 2703-2712` |
| Alt stream | None on ROCm unless `SGLANG_ROCM_USE_MULTI_STREAM` (default False) | `srt/models/deepseek_v2.py:2925-2934`; `srt/environ.py:972` |
| Norms | `input_layernorm`, `post_attention_layernorm` = `RMSNorm(6144, eps=1e-5)`; `model.norm` = `RMSNorm(6144, eps=1e-5)` | `srt/models/deepseek_v2.py:2652-2655, 2978-2979` |
| RMSNorm kernel | With SGLANG_USE_AITER, `_forward_method = forward_aiter`: residual given -> aiter `rmsnorm2d_fwd_with_add` (fused add + RMSNorm, new `residual_out` tensor); no residual -> aiter `rmsnorm2d_fwd`. Falls back to `forward_native` only if weight dtype != x dtype (weights are built under default dtype = model dtype bf16) | `srt/layers/layernorm.py:111-126, 488-501, 650-755` (fused add at `:734-750`, plain at `:752`); `srt/model_loader/loader.py:836` |

### L.1 Attention-input quant format (`quant_format` passed to attn_boundary.prepare)
| Step | Result | Cite |
|---|---|---|
| `_detect_gfx95_quant_format` at init | weight dtype is fp8 e4m3fn, `weight_scale` attribute absent -> `"fp8_pending"` | `srt/models/deepseek_v2.py:2726-2745` |
| `_resolve_gfx95_quant_format` on first forward | still no `weight_scale` -> `""` | `srt/models/deepseek_v2.py:2747-2756` |
| Why no `weight_scale` | block-FP8 `Fp8LinearMethod` registers only `weight_scale_inv`; `weight_scale` is registered on the per-tensor branch only; block post-load keeps `weight_scale_inv` | `srt/layers/quantization/fp8.py:660-688, 800-832` |
| Consequence | `_norm_quant_kernel("")` = `_update_and_read_residual_plain` = `norm(hidden, residual)`: bf16 [T, 6144] attention input, no fp8 group quant here. The `"fp8"` branch (`fused_rms_fp8_group_quant`, group 128, plus bf16 copy under DSA) is not taken | `srt/layers/layer_boundary/residual/add_norm.py:187-192, 211-241, 259-270` |
| `qkv_latent_func` | `_attn_input_default` stores `AttentionInputs(hidden_states, fb, self_attn.prepare_qkv_latent)`; the attention calls `fetch_qkv_latent()` -> `fused_qkv_a_proj_with_mqa(x)` once per layer (cached); cleared after the attention | `srt/layers/layer_boundary/prepare.py:254-264`; `srt/layers/layer_boundary/adapters/attention.py:61-70`; `srt/models/deepseek_v2.py:2506-2531, 2660-2665, 2802` |

### L.2 One decoder layer, in order (per GPU)
| # | Call | What runs at TP8 decode | Tensors | Cite |
|---|---|---|---|---|
| 1 | `attn_boundary.prepare` (layer 0) | Stack entry: residual = embedding output; `read` -> aiter `rmsnorm2d_fwd` (input_layernorm) | in bf16 [T, 6144] (complete, after embedding all-reduce); out x bf16 [T, 6144]; residual = in | `srt/layers/layer_boundary/prepare.py:414-450`; `add_norm.py:348-354, 187-192` |
| 1' | `attn_boundary.prepare` (layers 1..77) | Previous FFN output arrives as `UnreducedOutput` (partial, sum over TP owed). Tries `fused_attn_input`: aiter fused AR + residual add + RMSNorm (input_layernorm) when `aiter_ar_fusion_applies`; fallback: `group.all_reduce(partial)` then `norm(partial_summed, residual)` (aiter fused add+RMSNorm) | in partial bf16 [T, 6144]; out x bf16 [T, 6144], residual' bf16 [T, 6144] | `srt/layers/layer_boundary/prepare.py:291-411` (carried fusions `:363-408`); `fusions/allreduce.py:41-111`; `construction.py:158-162` |
| 2 | `self_attn(...)` | MLA + DSA (other section); `o_proj` RowParallel with `reduce_results=False` -> partial | out partial bf16 [T, 6144] | `srt/models/deepseek_v2.py:2150-2158, 2613, 2788-2797` |
| 3 | `maybe_prefetch_next_full_attention_kv` | no-op (only DSA prefill-CP with layer-split pool) | - | `srt/models/deepseek_v2.py:2804-2806`; `srt/layers/attention/dsa/utils.py:307-323` |
| 4 | `attn_boundary.finish` | Attention is `ALWAYS_PARTIAL` -> `finishes_directly`: records the partial with `DeclaredSum(ATTN_TP)`, no communication | - | `srt/layers/layer_boundary/stage.py:198-223`; `construction.py:361-362`; `factories.py:433-441` |
| 5 | `ffn_boundary.prepare` | `_reduce_update_read(group=ATTN_TP)`: first `fused_ffn_input` (aiter fused AR + add + RMSNorm with post_attention_layernorm, `use_attn_tp_group=True`) when `aiter_ar_fusion_applies`; fallback `attn_tp_all_reduce` (full precision in decode) + `norm(h, residual)` | in partial bf16 [T, 6144]; out FFN input bf16 [T, 6144], residual'' bf16 [T, 6144] | `srt/layers/layer_boundary/prepare.py:121-180`; `boundary.py:656-674`; `fusions/allreduce.py:114-165`; `ops.py:209-218` |
| 6 | `mlp(...)` inside `ffn_boundary.exit` | Dense (layers 0-2) `DeepseekV2MLP(reduce_results=False)` or `DeepseekV2MoE(reduce_results=False)`; output partial over TP | out partial bf16 [T, 6144] | `srt/models/deepseek_v2.py:2628-2650, 2827-2832` |
| 6a | MoE output buffer | If the MoE runner is not inplace, `moe_output_buffer_ctx(hidden_states_orig)` lends the layer input's buffer to the MoE output | - | `srt/models/deepseek_v2.py:2780, 2814-2825` |
| 7 | `ffn_exit.finish` | Non-terminal layers: sum deferred (wrapped as `UnreducedOutput(group=TP group)`), no communication here; it is completed by step 1' of the next layer | - | `srt/layers/layer_boundary/exit.py:98-147, 267-353, 457-483`; `srt/models/deepseek_v2.py:2833` |

Deferral conditions that hold here: `tp_size > 1`, no DP attention, output group set, `post_experts_sum_is_one_all_reduce()` true (no a2a, no quant comm, no `SGLANG_SHARED_EXPERT_TP1`, no LoRA, not EP>1 and MoE-TP>1, reduction group is the TP group) -> `single_sum` -> `_batch_allows_deferred_sum` true (`srt/layers/layer_boundary/exit.py:118-129, 457-472`; `srt/layers/moe/utils.py:935-953`).

Group identities (why one fused kernel covers both FFN kinds): `_MOE_TP = _TP` when moe_tp_size == tp_size (8), `_ATTN_TP = _TP` when attn_tp_size == tp_size; `post_experts_reduction_group()` = moe_tp_group (EP 1); dense FFN group `SumGroup.TP` -> tp_group, MoE `SumGroup.MOE_OUTPUT` -> the same `_TP` object, so `fused_attn_input`'s check `owed.group is post_experts_reduction_group()` passes for dense and MoE layers (`srt/distributed/parallel_state.py:2787-2792, 2916-2919`; `srt/layers/moe/utils.py:923-932`; `srt/layers/layer_boundary/layout.py:175-185`; `factories.py:197-202`; `fusions/allreduce.py:84-91`).

### L.3 Aiter fused AR + add + RMSNorm: when it is taken
| Condition | Value at TP8 decode | Cite |
|---|---|---|
| `enable_aiter_allreduce_fusion` | False by default; set True for `GlmMoeDsaForCausalLM` on HIP, single node, no DP attention; `disable_aiter_allreduce_fusion_in_prefill=True` | `srt/arg_groups/fields/exec_.py:753-775`; `srt/arg_groups/model_hook.py:267-278, 435-456` |
| `aiter_ar_fusion_applies` | `_use_aiter`, hidden 6144 <= 16384, bytes = T*6144*2 <= 64 MiB (T <= 5461), tp != 6, no DP attention, decode not disabled | `srt/layers/layer_boundary/residual/add_norm.py:163-184` |
| Kernel call | `_forward_with_allreduce_fusion` -> `tensor_model_parallel_fused_allreduce_rmsnorm(x, residual, weight, eps)` -> `GroupCoordinator.fused_allreduce_rmsnorm`: `ca_comm.fused_allreduce_rmsnorm` if the communicator has it, else `ca_comm.custom_fused_ar_rms(..., use_1stage)`; 1-stage when bytes <= 128 KiB (T <= 10 at 6144 bf16) unless `SGLANG_USE_1STAGE_ALLREDUCE` set | `srt/layers/layernorm.py:196-260, 921-931`; `srt/distributed/communication_op.py:31-42`; `srt/distributed/parallel_state.py:819-878` |
| Fallback inside the fused helper | communicator returns None -> `tensor_model_parallel_all_reduce(x)` + `norm.forward(x, residual)` | `srt/layers/layernorm.py:262-265` |
| fp8 output from the fused kernel | Not used: `fuses_quant` needs `read.fp8_input`, which is None for `NORM_QUANT_READOUT` (dataclass default) | `srt/layers/layer_boundary/fusions/allreduce.py:51-67`; `add_norm.py:346, 452` |

### L.4 Communicator (ROCm)
| Item | Value | Cite |
|---|---|---|
| Custom AR class | `AiterCustomAllreduce` (`aiter.dist.device_communicators.custom_all_reduce`) when `SGLANG_USE_AITER_AR` (default true) and not the AMD deterministic impl (`SGLANG_USE_1STAGE_ALLREDUCE` unset, deterministic off) | `srt/distributed/device_communicators/custom_all_reduce.py:344-421` |
| Quick AR | `QuickAllReduce` also initialized on HIP when `qr_rocm_arch_available()` | `srt/distributed/parallel_state.py:515-543` |
| Plain all-reduce choice order | ca (`should_custom_ar`) -> pcie_ipc -> qr (`should_quick_allreduce`) -> pymscclpp -> torch_symm_mem -> in-place pynccl | `srt/distributed/parallel_state.py:767-806, 957-1000` |
| Dtype | bf16 (activations are bf16 [T, 6144]); quantized AR only for non-decode with `enable_quant_communications` | `srt/layers/layer_boundary/ops.py:209-246` |

### L.5 All-reduce count per layer (decode, TP8)
| Layer | Collectives | Size each |
|---|---|---|
| Embedding (once per step) | 1 TP all-reduce of the embedding output | bf16 [T, 6144] |
| Dense layer 0-2 | 2: (a) attention partial, at ffn_boundary.prepare (fused with post_attention_layernorm add+norm when aiter fusion applies); (b) dense MLP partial, at the next layer's attn_boundary.prepare (fused with that layer's input_layernorm) | bf16 [T, 6144] |
| MoE layer 3-76 | 2: same placement; the MoE all-reduce carries routed + shared (shared added into the partial before the sum, see below) | bf16 [T, 6144] |
| Layer 77 (last) | 2: attention one as above; FFN one completed by `residual_batch.complete_output` as a plain TP all-reduce, not fused with `model.norm` | bf16 [T, 6144] |
| lm_head (once) | 1 TP all-gather of logits along vocab | bf16 [T, 19360] per GPU -> [T, 154880] |

Total per step at T rows: 1 + 78 x 2 = 157 all-reduces + 1 all-gather. Cites: `srt/models/deepseek_v2.py:2808-2833, 3211-3215`; `srt/layers/layer_boundary/residual/batch.py:109-111`; `srt/layers/layer_boundary/residual/stream.py:63-71, 205-207`; `srt/layers/layer_boundary/output.py:90-96`; `srt/layers/vocab_parallel_embedding.py:515-536`.

Shared expert vs all-reduce: in `forward_normal` the shared output (when not fused into the routed call) is added by `maybe_fuse_routed_scale_and_shared_add` before the (skipped, `reduce_results=False`) block all-reduce; with `num_fused_shared_experts=1` it is inside `self.experts`. Either way the boundary sums routed + shared together. A TP1 replicated shared expert (`SGLANG_SHARED_EXPERT_TP1`) is the only case added after (`srt/models/deepseek_v2.py:1413-1430, 2636`). Whether fusion is on: other section.

### L.6 Model start
| Step | Detail | Cite |
|---|---|---|
| Embedding module | `VocabParallelEmbedding(154880, 6144, enable_tp=True, use_attn_tp_group=False)` (`SGLANG_ENABLE_EMBED_REPLICATION` default False) | `srt/models/deepseek_v2.py:2916-2921`; `srt/layers/vocab_parallel_embedding.py:166-185`; `srt/environ.py:763` |
| Sharding | vocab padded to multiple of 64 (x tp if needed): 154880 is a multiple of 64 and of 8 -> 19360 rows per GPU; weight `model.embed_tokens.weight` bf16 (not quantized: Fp8Config returns no method for embeddings -> `UnquantizedEmbeddingMethod`; it is also in `modules_to_not_convert`) | `srt/layers/vocab_parallel_embedding.py:48-57, 272-290, 302-306, 324-327`; `srt/layers/quantization/fp8.py:386-405` |
| Forward | masked local gather (ids outside this shard -> 0 rows), then `tensor_model_parallel_all_reduce` | `srt/layers/vocab_parallel_embedding.py:500-536` |
| Residual stream | `residual_batch.start(fb)` -> fresh `ResidualStream`; layer 0's prepare makes the embedding the residual | `srt/models/deepseek_v2.py:3078-3083`; `srt/layers/layer_boundary/residual/batch.py:24-30` |

### L.7 Model end, logits, sampling
| Step | Detail | Cite |
|---|---|---|
| Complete last FFN sum | `residual_batch.complete_output` -> plain TP all-reduce of layer 77's FFN partial | `srt/models/deepseek_v2.py:3211`; `batch.py:109-111` |
| Final norm | `final_norm` -> `final_norm_pair` -> `model.norm(hidden, residual)` = aiter `rmsnorm2d_fwd_with_add` (fused add + RMSNorm, eps 1e-5) -> bf16 [T, 6144] | `srt/models/deepseek_v2.py:3212-3215`; `batch.py:135-190`; `residual/access.py:40-54` |
| lm_head | `ParallelLMHead(154880, 6144)`, vocab-sharded 19360 x 6144 per GPU, bf16 (`lm_head` unquantized, `UnquantizedEmbeddingMethod` -> not applied as quant method); `torch.matmul(h.to(bf16), W.T)` -> bf16 [T, 19360]; fp32 lm_head only with `--enable-fp32-lm-head` (default False) | `srt/models/deepseek_v2.py:3260-3270, 3395-3398`; `srt/layers/logits_processor.py:82-86, 142-152, 451-466, 1059-1107`; `srt/arg_groups/fields/exec_.py:41-44` |
| Gather | `do_tensor_parallel_all_gather` (tp 8); `MultimemAllGatherer` uses its kernel only with a symmetric-memory multicast state, else `tensor_model_parallel_all_gather(x, dim=-1)`; on HIP `_all_gather_into_tensor` uses aiter custom all-gather when `SGLANG_USE_AITER_AG` (default True) and `should_custom_ag` | `srt/layers/logits_processor.py:474-479, 496-502, 939-955`; `srt/distributed/device_communicators/triton_symm_mem_ag.py:495-560`; `srt/distributed/parallel_state.py:1296-1331`; `srt/environ.py:916` |
| Logits dtype | slice to 154880, copy into the fp32 `next_token_logits_buffer` (or `.float()`) -> fp32 [T, 154880] | `srt/layers/logits_processor.py:1243-1271` |
| Sampling backend | `sampling_backend` default = "flashinfer" if available else "pytorch"; flashinfer availability requires CUDA -> "pytorch" on HIP | `srt/arg_groups/overrides.py:1018-1026`; `srt/utils/common.py:382-389` |
| All greedy batch | aiter `greedy_sample` (int32 ids) unless `SGLANG_DISABLE_AITER_GREEDY_SAMPLE` | `srt/layers/sampler.py:62-65, 72-74, 208-216` |
| Non-greedy | `logits.div_(temperatures)`; softmax via `sampler_softmax` -> `torch.softmax` on HIP (cake softmax needs `torch.version.cuda`); no top-k/top-p/min-p -> `sampling_from_probs_torch` = Gumbel-max (`exponential_` + argmax, `SGLANG_OPT_USE_GUMBEL_SAMPLE` default True); with top-k/top-p/min-p -> `top_k_top_p_min_p_sampling_from_probs_torch` | `srt/layers/sampler.py:225-229, 274-297, 362-386, 474-485, 977-1002`; `kernels/ops/sampling/__init__.py:38-57`; `kernels/cake_kernels/sampling.py:23-33`; `srt/environ.py:1777` |

### L.8 Graph capture for decode (ROCm)
| Item | Value | Cite |
|---|---|---|
| Decode max bs | GPU memory >= 160 GB branch -> 512 (capped by `max_running_requests` when that is smaller) | `srt/arg_groups/memory_hook.py:160-188` |
| Captured bs (no spec) | 1, 2, 4, 8, 12, 16..256 step 8, 272..496 step 16, 512 | `srt/arg_groups/cuda_graph_hook.py:500-531`; `srt/arg_groups/memory_hook.py:200-205` |
| Replay padding | live bs padded up to the smallest captured bs >= bs | `srt/model_executor/runner/base_cuda_graph_runner.py:165-180` |
| Collectives in graphs | capture runs under `ca_comm.capture()`; custom AR / quick AR / pynccl usable in graph mode | `srt/distributed/parallel_state.py:640-690` |
| Aiter fused AR in graphs | the `fused_ar_rms(registered=False)` branch is only for piecewise (prefill) graphs; full decode graph capture calls `custom_fused_ar_rms` | `srt/distributed/parallel_state.py:851-878` |

### UNVERIFIED
- AiterCustomAllreduce internals (whether it defines `fused_allreduce_rmsnorm` or only `custom_fused_ar_rms`, its `should_custom_ar` size limits, kernel numerics such as fp32 accumulation of add + norm): the `aiter` package is not installed on this machine.
- Which plain all-reduce wins per size (ca vs qr): depends on aiter `should_custom_ar` and `QuickAllReduce.should_quick_allreduce` thresholds at runtime; the fused path above bypasses it when it applies.
- `MultimemAllGatherer` on HIP: code falls back to `tensor_model_parallel_all_gather` when `create_state` fails or has no multicast; whether `create_state` succeeds on ROCm was not determined (torch symmetric-memory behaviour on ROCm not in this source).
- Whether the sampler call is inside the captured decode graph: not traced (the logits processor is inside `DeepseekV2ForCausalLM.forward`, `srt/models/deepseek_v2.py:3395-3398`; the sampler call site in the model runner was not read).
- Default temperature / top-p for requests that set none: depends on the checkpoint's generation_config and the request; not in the model code.
- Layer 77 FFN: whether the exit defers (then `complete_output` all-reduces) or completes in `_complete_now` (`sum_output`) was not resolved; both run one plain TP all-reduce before `model.norm`.

## 2. MLA attention, decode (DeepseekV2AttentionMLA, forward_absorb_rocm_prepare / forward_absorb_rocm_core)

### Which path runs
| Step | Branch and why | Cite |
|---|---|---|
| attention backend | `--attention-backend` unset + DSA arch -> `attention_backend = "dsa"` | `srt/arg_groups/model_overrides/deepseek_v2.py:99-104` |
| page size | ROCm + aiter preshuffle paged-MQA available -> `page_size = 64 * index_kpool = 64`; otherwise forced to 1 | `srt/arg_groups/model_overrides/deepseek_v2.py:152-175` |
| forward method | `handle_attention_dsa` -> MLA (decode: `use_mha` False), HIP remap -> `MLA_ROCM` | `srt/models/deepseek_common/attention_backend_handler.py:206-217, 32-43`; `srt/models/deepseek_v2.py:2286-2313, 2421-2429, 2491-2492` |
| single stream | `alt_stream` None on ROCm unless `SGLANG_ROCM_USE_MULTI_STREAM` (default False) -> all `alt_stream is not None` branches skipped | `srt/models/deepseek_v2.py:2925-2934`; `srt/environ.py:972` |
| RoPE module | `qk_rope_head_dim = 64`, `skip_rope = (qk_rope_head_dim == 0) = False` -> `rotary_emb = get_rope_wrapper(64, rotary_dim=64, max_position=1048576, base=8e6, rope_scaling=None, is_neox_style = not rope_interleave = False)`; with SGLANG_USE_AITER on HIP the wrapper is `aiter.rotary_embedding.get_rope` | `srt/models/deepseek_v2.py:2944, 2579-2583, 2161-2171`; `srt/layers/rotary_embedding/factory.py:73-75, 527-539` |
| softmax scale | `scaling = qk_head_dim^-0.5 = (192+64)^-0.5 = 1/16`; no yarn mscale (rope_scaling None because rope_type "default") | `srt/models/deepseek_v2.py:2041, 2055, 2173-2174, 2583` |
| q/kv norm kernel | `_is_block_scale_fp8(q_b_proj)` needs a 2-D `weight_scale` attr; Fp8 block layers register only `weight_scale_inv` -> False -> `elif _use_aiter:` aiter `fused_qk_rmsnorm` (quant_type No, bf16 out) | `forward_mla_rocm.py:531-592, 104-124`; `srt/models/deepseek_common/utils.py:79-93`; `srt/layers/quantization/fp8.py:660-688` |
| fused absorb + RoPE + KV write | decode, `_use_aiter_gfx95`, `dcp_size 1`, `w_kc` fp8 e4m3fn, `rotary_emb` set, `_skip_rope_for_dsa_tilelang_fused()` True (backend "dsa", dsa decode backend tilelang), "dsa" in FORWARD_ABSORB_CORE_ATTENTION_BACKENDS -> `fuse_bmm_rope_cache = True` | `forward_mla_rocm.py:424-451, 673-679, 1099-1110`; `srt/models/deepseek_common/utils.py:64-75` |
| standalone RoPE | skipped (`_skip_rope_for_dsa_tilelang_fused()` True) | `forward_mla_rocm.py:751-769` |
| min-latency fused_a / q_b GEMM | CUDA-only (`_IS_CUDA and SM >= 90`, bf16 weight) -> plain `ReplicatedLinear` / `ColumnParallelLinear` calls | `kernels/ops/gemm/fused_a_gemm.py:35-42`; `srt/models/deepseek_v2.py:2518-2531, 2533-2545` |
| input-scattered attention | not on HIP (`allow_input_scattered` needs `_is_cuda or _is_npu`) -> `fetch_qkv_latent` runs `qkv_latent_func(hidden_states_local)` without gather | `srt/layers/layer_boundary/adapters/attention.py:61-70, 78-91` |

### Ops in order, per GPU (TP 8 -> `num_local_heads = 64/8 = 8`; T = decode rows)
| # | Op | In -> out (per GPU) | dtype | Weights | Cite |
|---|---|---|---|---|---|
| A1 | `fused_qkv_a_proj_with_mqa` (`ReplicatedLinear(6144 -> 2048+512+64 = 2624)`) via `prepare_qkv_latent` | x [T, 6144] -> [T, 2624] | bf16 in/out; block-FP8 GEMM (activation quant inside, see Linear GEMMs) | `q_a_proj` + `kv_a_proj_with_mqa` stacked along N, replicated on all 8 GPUs | `srt/models/deepseek_v2.py:2066-2072, 3236-3243, 2506-2531`; `srt/layers/layer_boundary/adapters/attention.py:61-70` |
| A2 | split | q [T, 2048], latent_cache [T, 576]; k_nope = latent_cache[..., :512]; k_pe = latent_cache[..., 512:].unsqueeze(1) [T, 1, 64] | bf16 views | - | `forward_mla_rocm.py:513-521`; `forward_mla.py:196-203` |
| A3 | q_a_layernorm + kv_a_layernorm, one aiter `fused_qk_rmsnorm` launch | q [T, 2048] -> q (normed); k_nope [T, 512] -> k_nope (normed) | bf16 out; eps = rms_norm_eps 1e-5 | `q_a_layernorm.weight` [2048], `kv_a_layernorm.weight` [512] (replicated, not quantized) | `srt/models/deepseek_v2.py:2073, 2159`; `forward_mla_rocm.py:584-592, 110-124` |
| A4 | `q_lora = q` (normed, bf16) kept for the indexer | [T, 2048] | bf16 | - | `forward_mla_rocm.py:597-600` |
| A5 | `k_nope.unsqueeze(1)`; `q_b_proj` (`ColumnParallelLinear(2048 -> 64*256)`, attn_tp split) | q [T, 2048] -> [T, 2048] viewed [T, 8, 256] | bf16 out; block-FP8 GEMM | `q_b_proj` rows split 8 ways: 2048 of 16384 output rows per GPU (8 heads x 256) | `srt/models/deepseek_v2.py:2074-2081, 2533-2545`; `forward_mla_rocm.py:630-639` |
| A6 | indexer ("full" layers only): `self.indexer(x=hidden_states, q_lora=q_lora, positions, ...)` -> `topk_indices`; "shared" layers: `maybe_capture_indexer_topk(layer_id, prev_topk_indices)` returns the carried indices | x = attention input (input_layernorm output) [T, 6144], q_lora [T, 2048] -> topk [T, 2048] | int32 | see DSA indexer section | `forward_mla_rocm.py:641-653`; `forward_mla.py:146-165` |
| A7 | split q | q_nope [T, 8, 192], q_pe [T, 8, 64] | bf16 views | - | `forward_mla_rocm.py:671`; `forward_mla.py:196-203` |
| A8 | absorb skipped in prepare (`q_nope_out = None`), un-absorbed q_nope handed to core | - | - | - | `forward_mla_rocm.py:718-721, 812` |
| A9 | one aiter kernel `fused_fp8_bmm_rope_cat_and_cache_mla(q_nope^T [8, T, 192], w_kc^T, w_scale, q_pe, k_nope, k_pe, kv_buffer(layer), out_cache_loc, positions, cos_cache, sin_cache, group_size=128, k_scale, is_neox=False, q_out_dtype=fp8 e4m3fn)`: (a) W_UK absorb q_nope x w_kc -> [T, 8, 512] with per-token-group-128 fp8 activation quant and per-tensor weight scale `w_scale`, (b) RoPE (interleaved, 64 dims, `positions`) on q_pe and k_pe, (c) concat q -> `q_cat` [T, 8, 576], (d) write [k_nope 512 | roped k_pe 64] into the fp8 MLA cache at `out_cache_loc` | -> q_cat [T, 8, 576] fp8 e4m3fn; k_pe_fused | fp8 (q_out_dtype = fp8_dtype because kv_cache_dtype == "fp8_e4m3") | `w_kc` [8, 192, 512] from `kv_b_proj` (see weights) | `forward_mla_rocm.py:454-489, 830-855` |
| A10 | `attn_mqa(q_cat, None, None, forward_batch, q_rope=None, k_rope=k_pe_fused, save_kv_cache=False, topk_indices=topk_indices)` (cache already written in A9) | q_cat [T, 8, 576] -> attn_output viewed [T, 8, 512] | see DSA backend section | MLA latent cache (read) | `forward_mla_rocm.py:862-883, 1039`; `srt/models/deepseek_v2.py:2179-2188` |
| A11 | W_UV absorb `rocm_absorb_v_bmm`: fp8 branch (`w_kc` fp8 e4m3fn on gfx95) -> aiter Triton `batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant(X=attn_output, WQ=w_vc^T, w_scale, group_size=128, transpose_bm=True)` into `_bmm_buf` [T, 8, 256]; `_is_block_scale_fp8(o_proj)` False -> `_bmm_buf.flatten(1, 2)` | [T, 8, 512] -> [T, 2048] | bf16 out (activation quantized to fp8 per 128-group inside the kernel) | `w_vc` [8, 512, 256] from `kv_b_proj` | `forward_mla_rocm.py:249-300, 323-350, 1074`; `forward_mla_rocm.py:156-182` (decode tile config, M <= 256) |
| A12 | `o_proj` (`RowParallelLinear(64*256 -> 6144)`, `reduce_results=False`) | [T, 2048] -> partial sum [T, 6144] | bf16; block-FP8 GEMM | `o_proj` columns split 8 ways: 2048 of 16384 input columns per GPU | `srt/models/deepseek_v2.py:2150-2158, 2613`; `forward_mla_rocm.py:1088` |
| A13 | return: `next_skip_topk` False -> `(output, None)`; True -> `(output, topk_indices)` | - | - | - | `forward_mla_rocm.py:1090-1097`; `srt/models/deepseek_v2.py:2110-2111` |

Notes
- The all-reduce of the o_proj partial is not in this module (`reduce_results=False`); it is done by the layer boundary (see Layer section).
- `k_scale` passed to the fused kernel = `attn_mqa.k_scale`: `Fp8KVCacheMethod` initialises it to -1 and, with no k/v scale in the checkpoint, sets 1.0 (`srt/layers/quantization/fp8.py:471-472`; `srt/layers/quantization/kv_cache.py:34-82`). UNVERIFIED: that the GLM-5.3 checkpoint carries no `k_scale` / `v_scale` tensors (checkpoint index not read here).
- UNVERIFIED: the internals of `fused_fp8_bmm_rope_cat_and_cache_mla` (aiter `aiter/ops/triton/fusions/fused_bmm_rope_kv_cache.py`, imported at `srt/layers/rocm_linear_utils.py:23-28`), of aiter `fused_qk_rmsnorm`, and of aiter `get_rope` (cos/sin cache dtype, interleaved pairing = (2i, 2i+1) for `is_neox=False`): aiter is not installed on this machine, so the exact KV write layout from this kernel, its fp8 rounding (scale `k_scale`=1.0) and the q fp8 cast scale were not read.
- UNVERIFIED: whether `aiter_can_use_preshuffle_paged_mqa()` returns True on the box (needs Triton >= 3.5 or AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1); if False the server forces page_size 1 (`srt/arg_groups/model_overrides/deepseek_v2.py:152-161`).

## 3. DSA part: indexer, shared top-k, sparse MLA decode, caches (GLM-5.3, TP8, gfx950, tilelang, fp8 KV, page 64)

Paths are relative to the sglang package root (`.../sgl0522/sglang`). T = decode rows on this GPU (= bs, plain decode).
Settled before this section: kpool = 1 -> plain `Indexer` (`srt/models/deepseek_v2.py:2115-2117`, `srt/configs/model_config.py:452-453`).

Tiny example (bs 2, seq_lens [3000, 1500], layers 6 and 7):
```
layer 6 ("full"):   topk6 = indexer(x6, q_lora6)          # int32 [2, 2048] physical KV slots
                    row 0: best 2048 of positions 0..2999 -> slot = pt64[req0][p>>6]*64 + (p&63)
                    row 1: 1500 <= 2048 -> slots of positions 0..1499, then 548 x -1
                    attn6 = sparse_mla(q6, kv_cache[6], topk6); returns (out6, topk6)   # next layer is shared
layer 7 ("shared"): no indexer module; attn7 = sparse_mla(q7, kv_cache[7], topk6)
```

### D0. Which code path runs

| Item | Value under the given flags | Cite |
|---|---|---|
| attention backend name | `"dsa"` (set when `--attention-backend` is not given) | `srt/arg_groups/model_overrides/deepseek_v2.py:99-104` |
| page size 64 | kept only if `aiter_can_use_preshuffle_paged_mqa()` (SGLANG_USE_AITER and Triton >= 3.5 or `AITER_ENABLE_AOT_GLUON_PA_MQA_LOGITS=1`); otherwise forced to 1 | `srt/arg_groups/model_overrides/deepseek_v2.py:152-172`; `srt/layers/attention/dsa/utils.py:34-67` |
| layer has an indexer | only if `not skip_topk` (`indexer_types[l] == "full"`): layers 0,1,2,6,10,...,74 | `srt/models/deepseek_v2.py:2110-2139`; `srt/configs/model_config.py:410-418` |
| Indexer forward on HIP | `forward_hip` not defined -> `forward_cuda` | `srt/layers/attention/dsa/dsa_indexer.py:1892`; `kernels/fused_op.py:150-157` |
| indexer rope style | `is_neox_style = not indexer_rope_interleave = False` (interleaved) | `srt/models/deepseek_v2.py:2114` |
| indexer quant args | `scale_fmt="ue8m0"`, `block_size=128` | `srt/models/deepseek_v2.py:2127-2128` |
| alt stream | None on ROCm (`SGLANG_ROCM_USE_MULTI_STREAM` default False) -> single-stream branches | `srt/models/deepseek_v2.py:2925-2934`; `srt/environ.py:972` |
| DSA decode impl | `dsa_decode_impl = "tilelang"` | `srt/layers/attention/dsa_backend.py:559-560` |
| fused top-k | `SGLANG_DSA_FUSE_TOPK=True`, no PD -> `use_fused_topk=True`; top-k backend default `sgl-kernel`; `SGLANG_OPT_USE_TOPK_V2`: env default True, but the HIP model hook sets it False for DSA models unless the env var is set explicitly (the `set(True)` at `:501` is the DeepseekV4 branch) -> top-k v2 off | `srt/environ.py:1831, 1382`; `srt/arg_groups/model_hook.py:434-437, 464-466, 497-501`; `srt/layers/attention/dsa/utils.py:186-198`; `srt/layers/attention/dsa_backend.py:651`; `srt/arg_groups/fields/exec_.py:348-355` |
| topk broadcast from rank 0 | off (`SGLANG_DSA_TOPK_BROADCAST=False`): each GPU computes its own top-k | `srt/layers/attention/dsa/dsa_indexer.py:213-225`; `srt/environ.py:1840` |
| decode CUDA graphs | two per bs on HIP: `dense` (max seq_len in batch <= 2048) and `sparse` | `srt/layers/attention/graph_variants.py:27-43, 46-67`; `srt/model_executor/runner/decode_cuda_graph_runner.py:316-322` |

### D1. Indexer weights (full layers only), per GPU

| Module (SGLang name under `model.layers.L.self_attn.indexer.`) | Shape [out, in] | Quant | Sharding | Cite |
|---|---|---|---|---|
| `wq_b` | [32*128=4096, 2048] | Fp8LinearMethod (block 128x128, `weight_scale_inv`), since not in modules_to_not_convert | ReplicatedLinear: full copy on all 8 GPUs | `srt/layers/attention/dsa/dsa_indexer.py:333-339`; `srt/layers/quantization/fp8.py:393-404` |
| `wk` | [128, 6144] | Fp8LinearMethod (block) | Replicated | `srt/layers/attention/dsa/dsa_indexer.py:350-356` |
| `weights_proj` | [32, 6144] | none (no quant_config passed), `params_dtype=bf16` | Replicated | `srt/layers/attention/dsa/dsa_indexer.py:357-363` |
| `k_norm` | LayerNorm(128) weight + bias, eps 1e-6 (class default, not config `rms_norm_eps`) | none | Replicated | `srt/layers/attention/dsa/dsa_indexer.py:364-374`; `srt/layers/layernorm.py:1012-1029` |
| `rotary_emb` | rope dim 64, base `rope_theta`=8e6, `rope_scaling=None` (rope_type "default"), interleaved | - | - | `srt/layers/attention/dsa/dsa_indexer.py:375-383`; `srt/models/deepseek_v2.py:2579-2583` |
- On HIP `use_dsa_indexer_fusion` is False (CUDA only), so `wk` and `weights_proj` stay separate modules; the loader's fused-wk helper returns False and they load per tensor (`srt/layers/attention/dsa/dsa_indexer.py:299-303`; `srt/models/deepseek_common/deepseek_weight_loader.py:89-107, 333-343`).
- Checkpoint `.indexer.` tensors of layers with no indexer module (shared layers) are skipped by the loader (`srt/models/deepseek_common/deepseek_weight_loader.py:328-331`).
- `softmax_scale = 128^-0.5` (`srt/layers/attention/dsa/dsa_indexer.py:403`).

### D2. Indexer inputs

| Input | What it is | Cite |
|---|---|---|
| `x` | `hidden_states` given to self_attn = output of `attn_boundary.prepare` (input_layernorm); bf16 [T, 6144], or the gfx95 tuple (fp8, scale, bf16) when `quant_format="fp8"`; every indexer GEMM/gate takes the bf16 member `x[2]` | `srt/models/deepseek_v2.py:2780-2797`; `srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py:643-649`; `srt/layers/attention/dsa/dsa_indexer.py:739, 880-883, 554-555` |
| `q_lora` | `q_a_layernorm(q_a)` bf16 [T, 2048]: the unquantized output of `fused_rms_fp8_group_quant(output_unquantized_inp1=True)` on the block-fp8 branch, else the bf16 normed q | `srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py:545-558, 583-600` |
| `positions` | int64 [T] = clamp(seq_lens - 1, 0) | `srt/model_executor/forward_batch_info.py:1184`; `kernels/ops/attention/clamp_position.py:29-37` |
| `out_cache_loc` | int64 [T]; decode graph buffer dtype int64 | `srt/model_executor/runner/decode_cuda_graph_runner.py:586-587`; `srt/model_executor/runner/base_runner.py:99` |
| metadata | `DSAIndexerMetadata`: `get_seqlens_int32` = `cache_seqlens_int32`, `get_page_table_64` = `real_page_table`, `get_seqlens_expanded` = `dsa_seqlens_expanded` (= seq_lens in decode) | `srt/layers/attention/dsa_backend.py:4064-4083`; `srt/layers/attention/dsa/dsa_indexer_metadata.py:102-112` |
- Call order in the attention prepare (single stream): `q_b_proj` first, then `self.indexer(...)` if `should_run_indexer` (`not skip_topk`) (`srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py:630-657`; `srt/models/deepseek_common/attention_forward_methods/forward_mla.py:146-163`).

### D3. Indexer: three ways a full layer runs (decided per call)

| Case | When | Cite |
|---|---|---|
| D3a k-only | `dense` graph variant (max seq_len in batch <= 2048) at capture; eager: HIP and max seq_len <= 2048 | `srt/layers/attention/dsa/dsa_indexer.py:835-868, 1948-1966` |
| D3b gfx950 fused (4 kernels) | `use_gfx950_fused_indexer` and the per-call gate passes, incl. 1 <= T <= 48 | `srt/layers/attention/dsa/dsa_indexer.py:413-485, 693-833, 1971-1985`; `kernels/ops/attention/dsa/hip_gfx950/fused_decode.py:22-31` |
| D3c standard chain | everything else in the `sparse` variant (e.g. T > 48 = bs 64) | `srt/layers/attention/dsa/dsa_indexer.py:2077-2208` |

gfx950 fused gate (static, per layer at init):
- shape: gfx950, head_dim 128, rope 64, 32 heads, topk 2048, q_lora 2048, hidden 6144, block 128, ue8m0, not neox, LayerNorm k_norm, no init/local tokens - all hold for GLM-5.3 (`kernels/ops/attention/dsa/hip_gfx950/fused_decode.py:43-81`).
- weights: `weights_proj` bf16 and `wk`/`wq_b` both fp8 or both bf16 (`srt/layers/attention/dsa/dsa_indexer.py:437-447`); code comment names the block-FP8 wq_b/wk case "GLM-5.3" (`kernels/ops/attention/dsa/hip_gfx950/fused_decode.py:191-194`).
- runtime: `--enable-dsa-fused-indexer` default None = on where supported, SGLANG_USE_AITER, preshuffle available, the 4 JIT modules build (`srt/arg_groups/fields/exec_.py:405-417`; `srt/layers/attention/dsa/utils.py:72-107`; `kernels/ops/attention/dsa/hip_gfx950/loader.py:41-93`).
- when on: `use_aiter_fused_fp8_writer=False` and `k_norm` rebuilt as bf16 LayerNorm, so every path (D3a/b/c, prefill too) applies the Hadamard to index K and Q (`srt/layers/attention/dsa/dsa_indexer.py:479-485, 686-691`; `srt/layers/attention/dsa/utils.py:118-147`).
- per call it declines (falls to D3c) if: in a piecewise/breakable graph, `force_unfused_topk`, CP, x without bf16 copy, LoRA, `index_page_size != 64`, layer-split pool, T outside [1, 48], non-int64/non-contiguous loc/positions, page-table width > workspace width (`srt/layers/attention/dsa/dsa_indexer.py:715-803`). Workspace width = ceil((max_position_embeddings + 4)/64)+4 pages (`srt/layers/attention/dsa/dsa_indexer.py:503-521`).

#### D3b. gfx950 fused path (T <= 48, sparse variant), per GPU, all 32 index heads on every GPU

| # | Op | In -> out | Kernel | Cite |
|---|---|---|---|---|
| 1 | `q_proj = wq_b(q_lora)` | bf16 [T,2048] -> bf16 [T,4096] | block-fp8 linear (layer's own GEMM) | `kernels/ops/attention/dsa/hip_gfx950/fused_decode.py:269-276` |
| 2 | `k = wk(x)` | [T,6144] -> bf16 [T,128] | block-fp8 linear | same |
| 3 | `weights = weights_proj(x_bf16)` | bf16 [T,6144] -> bf16 [T,32] (HIP returns bf16, no fp32 cast) | bf16 GEMM | `srt/layers/attention/dsa/dsa_indexer.py:548-563` |
| 4 | q side, per (token, head): interleaved rope on dims 0..63 (pairs 2i,2i+1, cos/sin[pos, i]) -> bf16; Hadamard-128 x 128^-0.5 -> bf16; `q_scale = 2^ceil(log2(max(amax,1e-10)/448))`; `q_fp8 = q / q_scale` (e4m3fn) | -> fp8 [T,32,128] | JIT `qk_rope_hadamard_quant_kernel`, grid (T, 33), 64 threads | `kernels/jit/csrc/dsa_gfx950/qk_rope_hadamard_quant.cuh:50-60, 63-78, 130-151, 303` |
| 4g | head gate: `w[t,h] = bf16(weights[t,h] * 32^-0.5) * q_scale[t,h] * 128^-0.5` | -> fp32 [T,32] | same kernel | `kernels/jit/csrc/dsa_gfx950/qk_rope_hadamard_quant.cuh:152-158`; `kernels/ops/attention/dsa/hip_gfx950/fused_decode.py:161` |
| 5 | k side (block y = 32): LayerNorm(k) fp32 math, eps 1e-6, fp32 copies of weight/bias -> bf16; rope dims 0..63 -> Hadamard-128 -> bf16; `k_scale = 2^ceil(log2(max(amax,1e-4)/448))`; write fp8 + fp32 scale into index-K page at `out_cache_loc` (layout D6.2); skipped if loc < 0 | -> index-K cache | same kernel | `kernels/jit/csrc/dsa_gfx950/qk_rope_hadamard_quant.cuh:161-217`; `kernels/ops/attention/dsa/hip_gfx950/fused_decode.py:196-197, 281-296` |
| 6 | logits: `logit[t,p] = k_scale[p] * sum_h w[t,h] * relu(q_fp8[t,h] . k_fp8[p])`, fp32; p >= seq_len -> -inf; reads pages `pt64[t][p>>6]`; builds the top-k histogram when seq_len > 2048 | -> fp32 [T, W64*64] | JIT `logits_hist` (MFMA 16x16x128 f8f6f4) | `kernels/jit/csrc/dsa_gfx950/paged_mqa_logits.cuh:32-39, 131-132, 159-198`; `kernels/ops/attention/dsa/hip_gfx950/fused_decode.py:303-312` |
| 7 | top-2048 + page transform: seq_len <= 2048 -> all positions 0..len-1 in order; else top 2048 by logit; output `slot = pt64[t][p>>6]*64 + (p&63)`, -1 padded | -> int32 [T, 2048] physical page_size-1 slots | JIT `topk_transform` | `kernels/jit/csrc/dsa_gfx950/topk_transform.cuh:32, 102-108, 601-610`; `kernels/ops/attention/dsa/hip_gfx950/fused_decode.py:315-327` |
- Current token: its K is written in #5 before #6 reads the cache, and `seqlens = cache_seqlens_int32` counts it (`kernels/ops/attention/dsa/hip_gfx950/fused_decode.py:281-312`; `srt/layers/attention/dsa_backend.py:1093`).

#### D3c. Standard chain (sparse variant, T > 48 or fused declined)

| # | Op | In -> out | Kernel | Cite |
|---|---|---|---|---|
| 1 | `query = wq_b(q_lora)` -> view [T,32,128] | bf16 | block-fp8 linear | `srt/layers/attention/dsa/dsa_indexer.py:917-922` |
| 2 | `key = wk(x_bf16)` | bf16 [T,128] | block-fp8 linear | `srt/layers/attention/dsa/dsa_indexer.py:923-926` |
| 3 | `key = k_norm(key)` | bf16 LayerNorm eps 1e-6 (aiter `layer_norm` if present, else F.layer_norm) | aiter / torch | `srt/layers/attention/dsa/dsa_indexer.py:927, 483`; `srt/layers/layernorm.py:1060-1075` |
| 4 | `rotary_emb(positions, query[...,:64], key[...,:64])`, copy back | interleaved, first 64 dims | rope kernel | `srt/layers/attention/dsa/dsa_indexer.py:928-935` |
| 5 | Hadamard on query and key, scale 128^-0.5 | bf16 | `fast_hadamard_transform` | `srt/layers/attention/dsa/dsa_indexer.py:241-253, 959-961` |
| 6 | `q_fp8, q_scale = act_quant(query, 128, "ue8m0")` | one group = one head; amax floor 1e-4; pow2 scale; fp8 [T,32,128], fp32 [T,32,1] | TileLang `act_quant_kernel` | `srt/layers/attention/dsa/dsa_indexer.py:1916-1917, 2099-2100`; `kernels/ops/attention/dsa/tilelang_kernel.py:92-176` |
| 7 | index-K store: `indexer_k_quant_and_cache(key, kv_cache[pages,64,132] fp8, out_cache_loc, 128, "ue8m0", preshuffle=True)` | -> index-K cache | aiter | `srt/layers/attention/dsa/dsa_indexer.py:1843-1858` |
| 8 | head gate `_get_logits_head_gate`: `weights_proj(x_bf16)` (bf16) * 32^-0.5, then `* q_scale * 128^-0.5` | fp32 [T,32,1] | torch.compile | `srt/layers/attention/dsa/dsa_indexer.py:573-580, 2118-2124, 2178-2179` |
| 9 | logits `deepgemm_fp8_paged_mqa_logits(Preshuffle=True, KVBlockSize=64)` over `real_page_table`, lengths `cache_seqlens_int32` | fp32 [T, W64*64] | aiter Triton/Gluon | `srt/layers/attention/dsa/dsa_indexer.py:1235-1241, 1310-1318, 1366-1376`; `kernels/ops/attention/dsa/paged_mqa_logits.py:95-126` |
| 10 | `metadata.topk_transform` -> `should_use_topk_v2()` False (D0) -> legacy `sgl_kernel.fast_topk_transform_fused(score=logits, lengths, page_table_size_1=page_table_1, topk=2048)`: top-2048 of `logits[t,:len]` mapped through the page-size-1 table `page_table_1` (= `req_to_token` row) to physical slots, -1 pad | int32 [T,2048] physical slots | sgl_kernel AOT op; ROCm build compiles native `csrc/elementwise/topk.hip` | `srt/layers/attention/dsa/dsa_indexer.py:1436`; `srt/layers/attention/dsa/dsa_topk_backend.py:55-56, 96-133, 185-209`; `kernels/aot/python/sgl_kernel/top_k.py:45-81`; `kernels/aot/setup_rocm.py:51-52` |

#### D3a. k-only path (dense variant, max seq_len <= 2048)
- Steps D3c #2, #3, #4 (k only, dummy q), #5 (k only), #7; no wq_b, no weights_proj, no logits (`srt/layers/attention/dsa/dsa_indexer.py:968-1003, 1758-1769`).
- Top-k: zero logits [T,2048] -> `topk_transform` -> positions 0..len-1 mapped to physical slots, -1 pad (`srt/layers/attention/dsa/dsa_indexer.py:1775-1790`).

### D4. Shared layers ("shared" indexer type)

| Item | Value | Cite |
|---|---|---|
| modules | no `indexer` (no wq_b/wk/weights_proj/k_norm) | `srt/models/deepseek_v2.py:2113` |
| index-K cache | 0-row placeholder (`skip_topk_layers`), never written or read | `srt/mem_cache/index_key_cache.py:50-53`; `srt/mem_cache/kv_cache_configurator.py:2168-2174, 126-140` |
| top-k used | `prev_topk_indices` as is (`should_run_indexer` False) | `srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py:641-657`; `srt/models/deepseek_common/attention_forward_methods/forward_mla.py:146-163` |
| carry | model loop: `index_topk_share.update(topk_indices)` after every layer; layer returns `(out, topk)` iff `next_skip_topk`, else `(out, None)` | `srt/models/deepseek_v2.py:3091, 3159-3174`; `srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py:1090-1097`; `srt/layers/attention/index_topk_share.py:53-60` |
| source layer | nearest preceding full layer: 3,4,5 <- 2; 7,8,9 <- 6; ...; 75,76,77 <- 74 (from `indexer_types` + the carry rule) | config `indexer_types`; rows above |
| MLA KV | the shared layer still writes and reads its own MLA latent cache (D5); the indices are physical slots, valid for every layer's buffer of the same pool | `srt/layers/attention/dsa_backend.py:2600-2601` |

### D5. Sparse MLA decode in the DSA backend (all 78 layers)

| # | Op | Shapes per GPU | Cite |
|---|---|---|---|
| 1 | fused rope + cat + KV write (gfx95, tilelang): `fused_fp8_bmm_rope_cat_and_cache_mla` (q absorb + rope + write, when `w_kc` fp8) or `fused_qk_rope_cat_and_cache_mla`; writes `[k_nope(512) ; rope(k_pe)(64)]` as fp8 at `out_cache_loc` with `k_scale = attn_mqa.k_scale`; returns `q_cat` = `[q_nope_absorbed(512) ; rope(q_pe)(64)]` in fp8 (`q_out_dtype` = kv dtype since backend is "dsa"); `save_kv_cache=False` | q_cat fp8 [T, 8, 576] | `srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py:381-421, 424-491, 829-882, 1099-1110` |
| 1' | fallback KV write if the fused branch is not taken: `set_mla_kv_buffer` -> Triton `set_mla_kv_buffer_fp8_quant_kernel`, plain cast bf16 -> fp8, no scale, writes to loc 0 skipped | [T,1,576] | `srt/layers/attention/dsa_backend.py:2554-2567`; `srt/mem_cache/memory_pool.py:4996-5006`; `kernels/ops/kvcache/mla_buffer.py:330-435` |
| 2 | `q_all = q.view(-1, 8, 576)` (q_rope None -> zero-copy on HIP) | [T,8,576] | `srt/layers/attention/dsa_backend.py:2581-2587, 2645-2651` |
| 3 | `topk_indices` used as `page_table_1` unchanged (fused top-k, sgl-kernel backend) | int32 [T,2048] | `srt/layers/attention/dsa_backend.py:2590-2601, 992-1000` |
| 4 | `_forward_tilelang`: pad indices to multiple of 64 (2048 -> no pad) | | `srt/layers/attention/dsa_backend.py:3406-3435` |
| 5 | `tilelang_sparse_fwd`: q cast to kv dtype if different; `sparse_mla_fwd_decode_partial_fp8(num_heads=8, d_v=512, d_tail=64, topk=2048, block_I=64, threads=256, inner_iter=_pick_inner_iter(T,32,256,2))`; 16-head tile (8 heads used); per 64-index block: QK^T over 4x128 + 64 tail in fp32, idx < 0 -> -inf, online softmax `exp2(s*sm_scale*log2e)`, P x 448 -> fp8, P.V over the 512 latent dims of the same rows, x 1/448; partial O bf16, LSE fp32 | | `kernels/ops/attention/dsa/tilelang_kernel.py:1277-1354, 1001-1274` (sm_scale 1038, heads 1040-1046, mask 1142, P scale 1216-1234) |
| 6 | `sparse_mla_fwd_decode_combine(head_per_block=4)` | out bf16 [1, T, 8, 512] | `kernels/ops/attention/dsa/tilelang_kernel.py:921-997, 1336-1345` |
- Softmax scale: `layer.scaling = qk_head_dim^-0.5 = (192+64)^-0.5 = 1/16`; no yarn mscale because `rope_scaling` is None (`srt/models/deepseek_v2.py:2055, 2173-2174, 2179-2188, 2583`).
- MLA rope: dims are the 64 `k_pe` / `q_pe` dims, base 8e6, `is_neox_style = not rope_interleave = False` (`srt/models/deepseek_v2.py:2161-2171`).
- Local heads: 64 / 8 = 8 per GPU; KV is one shared latent head (`num_kv_heads=1`) (`srt/models/deepseek_v2.py:2052-2054, 2179-2188`; `srt/layers/attention/dsa_backend.py:534-536`).

### D6. Caches

#### D6.1 MLA latent cache (per layer, all 78 layers)
- `kv_cache_dim = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576` (HIP + tilelang: raw layout, no scale bytes) -> `dsa_kv_cache_store_fp8=False` (`srt/mem_cache/kv_cache_configurator.py:3202-3238`; `srt/mem_cache/memory_pool.py:4791-4801, 5311-5313`).
- dtype `torch.float8_e4m3fn` (gfx950 is not fnuz), stored as uint8, viewed fp8 by `get_key_buffer` (`srt/mem_cache/kv_cache_dtype.py:56-60`; `kernels/ops/quantization/fp8_kernel.py:95-106`; `srt/mem_cache/memory_pool.py:2054-2057, 4898-4905`).
- `kv_buffer[l]`: [size + 64, 1, 576]; 576 B/token; row 0 = dummy pad slot (`srt/mem_cache/memory_pool.py:4859-4876`).
- Scale: `attn_mqa.k_scale` = 1.0 unless the checkpoint has k/v scales (Fp8KVCacheMethod) (`srt/layers/quantization/fp8.py:472`; `srt/layers/quantization/kv_cache.py:31-75`).

#### D6.2 Index-K cache (per full layer)
- `index_page_size = page_size / kpool = 64`; `IndexKeyCache.buffer[l]`: uint8 [num_pages, 64 * (128 + 4) = 8448], `num_pages = (index_buf_size + 64 + 1)//64`, `index_buf_size = size` (no DCP) (`srt/mem_cache/memory_pool.py:5339-5349`; `srt/mem_cache/index_key_cache.py:16-41`).
- Page = the same 64-token KV page id as the MLA cache (`real_page_table` entry); bytes [0, 8192): fp8 values, preshuffled 16x16 tiles: offset `(slot//16)*2048 + (dim//16)*256 + (slot%16)*16 + dim%16`; bytes [8192, 8448): 64 fp32 scales, one per token (`kernels/jit/csrc/dsa_gfx950/qk_rope_hadamard_quant.cuh:195-217`; `kernels/jit/csrc/dsa_gfx950/paged_mqa_logits.cuh:32-39`; `srt/layers/attention/dsa/utils.py:150`).
- Contents: Hadamard-rotated, roped, LayerNormed K, pow2 (ue8m0) scale per token (when the gfx950 fused indexer is on, D3).

#### D6.3 Positions, slots, metadata
- `req_to_token` int32 [size+1, context_len + extra]; decode writes `req_to_token[req, seq_len_old] = out_cache_loc` (page-64 allocator) (`srt/mem_cache/memory_pool.py:398-406`; `srt/mem_cache/kv_cache_configurator.py:1597`; `srt/mem_cache/allocation.py:625-665`).
- `cache_seqlens_int32 = seq_lens` (includes the current token); `page_table_1 = req_to_token[req_pool_indices, :W]`; `real_page_table = page_table_1[:, ::64] // 64` (`srt/layers/attention/dsa_backend.py:1093, 1105-1107, 1054-1062, 1355-1388`).
- Graph (HIP): wide `page_table_1` [bs, req_to_token width] kept (`dsa_drop_wide_page_table` is CUDA-only); replay refreshes via `_fused_decode_metadata`; `topk_v2_plan` is None because top-k v2 is off (D0), so nothing to refresh (`srt/layers/attention/dsa_backend.py:963-990, 1496-1508, 1611-1625, 1809-1835`).
- `dsa_cache_seqlens_int32 = min(seq_lens, 2048)` is built but not read by the tilelang decode (`srt/layers/attention/dsa/utils.py:157-159`; `srt/layers/attention/dsa_backend.py:2645-2658`).

### D7. Numeric differences between the indexer paths
- q scale amax floor: 1e-10 in the fused kernel vs 1e-4 in `act_quant` (`kernels/jit/csrc/dsa_gfx950/qk_rope_hadamard_quant.cuh:147`; `kernels/ops/attention/dsa/tilelang_kernel.py:125`).
- k_norm: fp32 weight/bias copies in the fused kernel vs bf16 LayerNorm module in D3a/D3c (`kernels/ops/attention/dsa/hip_gfx950/fused_decode.py:196-197`; `srt/layers/attention/dsa/dsa_indexer.py:483`).
- Same index-K byte layout on all paths (`kernels/ops/attention/dsa/hip_gfx950/fused_decode.py:1-3`).

### D-UNVERIFIED
1. Whether the gfx950 fused indexer is actually on in the image: needs the 4 JIT modules to build against the installed aiter; not checkable without the box. (Page 64 itself implies preshuffle is available, `srt/arg_groups/model_overrides/deepseek_v2.py:152-172`.)
2. aiter internals not in this tree: `indexer_k_quant_and_cache`, `deepgemm_fp8_paged_mqa_logits`, `fused_qk_rope_cat_and_cache_mla`, `fused_fp8_bmm_rope_cat_and_cache_mla`, `layer_norm`; their math above comes from the SGLang callers and the fused kernel that states byte compatibility.
3. Which of D5 #1 kernels runs (`fused_fp8_bmm_...` vs `fused_qk_rope_...`) depends on `w_kc` being fp8 and on the attention prepare path (MLA section).
4. Checkpoint tensor names/dtypes of the indexer weights: no safetensors index here. Config `modules_to_not_convert` lists `self_attn.indexers_proj` (only full layers 0,1,2,6,...,78), which matches no SGLang module name; `weights_proj` is unquantized bf16 in SGLang regardless.
5. MLA `k_scale`: 1.0 unless the checkpoint carries `k_scale`/`v_scale` (not checked).
6. Fused-path workspace vs decode-graph page-table width: workspace = ceil((1048576+4)/64)+4 pages; page table width = `context_len + extra_max_context_len` (value of `extra_max_context_len` not traced); if wider, every fused call declines to D3c.
7. `in_piecewise_or_breakable_cuda_graph` taken as False during plain-decode graph capture (decode uses `decode_cuda_graph_runner`; the flag is the PCG/BCG context, `srt/layers/attention/dsa/dsa_prefill_cuda_graph.py:34-35`); not traced through the runner.

## 4. FFN (dense MLP layers 0-2, MoE layers 3-77) and weight sharding, GLM-5.3 TP8 EP1, ROCm gfx950, SGLANG_USE_AITER=1, --moe-runner-backend aiter

Paths are relative to the sglang package root (`python/sglang`). T = decode rows of the step; every GPU holds all T rows
(no DP attention). H = 6144.

### F.0 Code-path decisions

| Item | Branch taken and why | Cite |
|---|---|---|
| Dense vs MoE | `_is_layer_sparse`: `layer_id >= first_k_dense_replace (3)` and `layer_id % moe_layer_freq (1) == 0` -> layers 0-2 `DeepseekV2MLP`, 3-77 `DeepseekV2MoE` | `srt/models/deepseek_v2.py:2758-2766, 2628-2650` |
| Dense MLP group | `parallel_group = "replicated" if is_dense_ffn_fully_dp() else "tp"`; `is_dense_ffn_fully_dp()` is `moe_dense_tp_size == 1`; default `moe_dense_tp_size=None`, forced to 1 only under prefill CP -> **"tp"** | `srt/models/deepseek_v2.py:2639`; `srt/layers/layer_boundary/layout.py:83-84`; `srt/arg_groups/fields/parallel.py:289-295`; `srt/arg_groups/model_overrides/deepseek_v2.py:105-117, 190-201` |
| FFN input dtype | `ffn_boundary.prepare(hidden_states, forward_batch)` is called with no `quant_format` (the attention prepare gets one) -> FFN input is a plain bf16 tensor `[T, 6144]` | `srt/models/deepseek_v2.py:2809` vs `:2781-2786`; `srt/layers/layer_boundary/prepare.py:135, 427-450` |
| FFN output | both FFN modules are built `reduce_results=False`; the sum is completed by the FFN boundary exit (`ffn_exit.finish`) | `srt/models/deepseek_v2.py:2636, 2648, 2827-2833` |
| MoE forward path | `should_use_mega_moe` False (a2a backend none, `SGLANG_AMD_USE_FLYDSL_MEGA_MOE=False`); `_enable_a2a_moe` False; `_can_dual_stream_graph` False (CUDA-only env, flashinfer only); `alt_stream` is None on ROCm unless `SGLANG_ROCM_USE_MULTI_STREAM` (default False) -> **`forward_normal`** | `srt/models/deepseek_v2.py:999-1067, 967-981, 2925-2934`; `srt/layers/moe/mega_moe.py:50-53, 170-179`; `srt/environ.py:972, 985` |
| MoE impl class | a2a backend none -> `FusedMoE` | `srt/layers/moe/ep_moe/layer.py:354-367` |
| Expert quant method | `Fp8Config.get_quant_method` -> `Fp8MoEMethod` (experts not in `modules_to_not_convert`) | `srt/layers/quantization/fp8.py:405-470` |
| MoE runner | `--moe-runner-backend aiter` -> `MoeRunner(AITER)`; no fused func is registered for ("none","aiter"), so pre_permute -> `AiterRunnerCore.run` -> post_permute | `srt/layers/quantization/fp8.py:2851-2906`; `srt/layers/moe/moe_runner/runner.py:166-199`; `srt/layers/moe/moe_runner/aiter.py:330-354, 536-545` |
| fp8 dtype | gfx950 is not "gfx94" -> `is_fp8_fnuz()` False -> no e4m3fn->e4m3fnuz normalization anywhere; fp8 = `torch.float8_e4m3fn` | `kernels/ops/quantization/fp8_kernel.py:95-107` |
| Quant-once / mxfp8 prequant | `SGLANG_OPT_MOE_QUANT_ONCE` default False and CUDA-only; routed mxfp8 prequant CUDA-only -> both **off** | `srt/environ.py:1362`; `srt/models/deepseek_v2.py:1776-1786, 1862-1863` |
| Small-M MoE kernel | `SGLANG_ROCM_SMALLM_MOE` default True, but `smallm_moe_supported` requires `hidden_size == DIM = 4096` -> **not taken** for H 6144 | `srt/environ.py:974`; `srt/layers/moe/moe_runner/aiter.py:259-292`; `kernels/ops/moe/smallm_moe_gfx950/__init__.py:33, 283-288` |

### F.1 Shared-expert fusion decision (`num_fused_shared_experts`)

Decided once in the loader before layers are built: `install_shared_experts_fusion_decision(model_class, hf_config, quant_config)`
(`srt/model_loader/loader.py:285-294`; `srt/layers/moe/utils.py:618-647`). Starts from `--disable-shared-experts-fusion`
(default False, `srt/arg_groups/fields/exec_.py:946-952`); runner `aiter` is not in the auto-disable list
(`srt/arg_groups/overrides.py:1533-1555`), a2a none is not either (`:1558-1577`). Then
`DeepseekV2ForCausalLM.shared_experts_fusion_disable_reason` (`srt/models/deepseek_v2.py:3298-3367`):

| Check | Value here | Result | Cite |
|---|---|---|---|
| `quant_blocks_shared_experts_fusion(quant_config)` | `Fp8Config` has no `can_fuse_shared_expert` attr | pass | `srt/models/deepseek_common/utils.py:160-167` (only modelopt / quark define it: `grep can_fuse_shared_expert`) |
| `enforce_shared_experts_fusion` | default | - | `:3313-3314` |
| modelopt_fp4 + marlin | no | pass | `:3315-3325` |
| SBO / TBO | off | pass | `:3326-3327` |
| DeepEP-class a2a | none | pass | `:3328-3329` |
| `architectures[0] == fused_shared_experts_architecture` | `GlmMoeDsaForCausalLM` == `GlmMoeDsaForCausalLM.fused_shared_experts_architecture` | pass | `:3330-3331`; `srt/models/glm4_moe.py:1289-1290` |
| `n_routed_experts in (256, 384)`, `n_shared_experts == 1` | 256, 1 | pass | `:3340-3341` |
| HIP capability >= (9, 4) | gfx950 | pass | `:3348-3357` |
| `moe_ep_size > 1` branch | EP 1 | skipped | `:3358-3364` |
| W4AFP8 / W4A16 | no | pass | `:3365-3366` |

-> **fusion ON**: `num_fused_shared_experts = 1` (`srt/models/deepseek_v2.py:653-660, 3369-3373`). The checkpoint's
`modules_to_not_convert` does not list any `shared_experts` module, so it plays no part here.

Consequences:
- `FusedMoE(num_experts = 256 + 1 = 257, top_k = 8 + 1 = 9)` (`srt/models/deepseek_v2.py:674-678, 724-738`); no
  `self.shared_experts` module (`:804-812`).
- Loader renames `mlp.shared_experts.*` -> `mlp.experts.256.*` and loads it through the expert mapping with
  `num_experts = n_routed + 1` (`srt/models/deepseek_common/deepseek_weight_loader.py:239-244, 285-289`).
- `fused_shared_experts_scaling_factor` stays None (only set when `moe_ep_size > 1`) (`srt/models/deepseek_v2.py:713-722`).
- `_forward_shared_experts` returns None (`srt/models/deepseek_v2.py:1725-1747`); `maybe_fuse_routed_scale_and_shared_add`
  with `shared=None` is a pass-through for `Fp8MoEMethod` (`srt/layers/quantization/mxfp4_flashinfer_trtllm_moe.py:521-559`).

### F.2 Dense layer FFN (layers 0, 1, 2), `DeepseekV2MLP(6144 -> 12288)`, per GPU

| # | Op | In -> out (per GPU) | dtype | Weights (per GPU) | Kernel | Cite |
|---|---|---|---|---|---|---|
| D1 | `gate_up_proj` (`MergedColumnParallelLinear(6144, [12288, 12288])`, tp) | `[T, 6144]` -> `[T, 3072]` = `[gate 1536 | up 1536]` | in bf16, out bf16 | `w [3072, 6144]` fp8 e4m3fn, `weight_scale_inv [24, 48]` fp32 | `aiter_w8a8_block_fp8_linear` (F.6) | `srt/models/deepseek_v2.py:299-306, 386`; `srt/layers/quantization/fp8.py:1328-1363` |
| D2 | activation `SiluAndMul` | `[T, 3072]` -> `[T, 1536]`, `silu(gate) * up` | bf16 | - | `sgl_kernel.silu_and_mul` (HIP: `forward_cuda`; aiter variant only with `SGLANG_OPT_USE_AITER_SILU_MUL`, default False) | `srt/models/deepseek_v2.py:328, 485-486`; `srt/layers/activation.py:109-110, 130-150`; `srt/environ.py:1020`; `kernels/fused_op.py:156-163` |
| D3 | `down_proj` (`RowParallelLinear(12288 -> 6144)`, `reduce_results=False`) | `[T, 1536]` -> `[T, 6144]` **partial sum** | bf16 | `w [6144, 1536]` fp8, `weight_scale_inv [48, 12]` fp32 | `aiter_w8a8_block_fp8_linear` | `srt/models/deepseek_v2.py:307-315, 487`; `srt/layers/linear.py:1741-1790` |

- `swiglu_limit = getattr(config, "swiglu_limit", None)` is None for GLM-5.3 (config has no such key), so every clamp /
  fused-quant activation branch (`:390-484`) is skipped; `resolve_fused_clamp_route` still runs once (`:427-428`) and only
  sets flags (`srt/models/deepseek_common/amd/deepseek_v2_hip_act.py:17-52`).
- `gemm_output_zero_allocator` is forced None for dense layers (`srt/models/deepseek_v2.py:2811-2812`).
- Activation quant: inside D1 and D3, `aiter_per1x128_quant` = per-token, per-128-column group fp8 (one fp32 scale per
  row per 128 columns) (`srt/layers/quantization/fp8_utils.py:352, 1650-1664`).
- Reduction: one TP all-reduce of `[T, 6144]` bf16 over the 8 GPUs, owned by the FFN boundary exit
  (`srt/layers/layer_boundary/exit.py:262-265`, "the TP group a dense MLP reduces over").

### F.3 MoE layer FFN (layers 3-77), `DeepseekV2MoE.forward_normal`, per GPU

| # | Op | In -> out (per GPU) | dtype | Weights | Kernel | Cite |
|---|---|---|---|---|---|---|
| M1 | router logits `MoEGate.forward` (replicated: every GPU computes all 256 logits for all T rows) | `[T, 6144]` -> `[T, 256]` | in bf16, out **bf16** | `mlp.gate.weight [256, 6144]` **bf16** (param dtype = default dtype; `router_fp32` attr absent; config key `moe_router_dtype` is not read by deepseek_v2) | `aiter_dsv3_router_gemm` = `aiter.tuned_gemm.tgemm.mm(x, w, otype=x.dtype)` | `srt/models/deepseek_v2.py:503-512, 550-589, 1303-1309`; `srt/layers/rocm_linear_utils.py:30-35`; `grep moe_router_dtype` hits only mimo_v2 / iquest |
| M1a | (ROCm split-K router not used) | `rocm_router_max_tokens = -1` unless model_type `deepseek_v41` -> `rocm_dsv3_router_split_k` returns None -> plain `moe.gate(...)` | - | - | - | `srt/models/deepseek_v2.py:545-548`; `srt/models/deepseek_common/amd/deepseek_v2_hip_moe.py:17-32, 45-54`; `srt/layers/rocm_linear_utils.py:38-47` |
| M2 | bias dtype at the top-k boundary | `select_experts`: `_use_aiter and use_grouped_topk and correction_bias is not None and not SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK (default False)` -> `correction_bias = correction_bias_for_dtype(router_logits.dtype)` = **bf16 copy** of the fp32 parameter | bf16 | `mlp.gate.e_score_correction_bias [256]` fp32 param (GLM keeps fp32: `not is_glm_moe_dsa(config)` gate) | `TopKConfig.correction_bias_for_dtype` | `srt/layers/moe/topk.py:2616-2626, 280-299`; `srt/environ.py:1381`; `srt/models/deepseek_v2.py:513-529` |
| M3 | grouped top-k (`n_group=1`, `topk_group=1`, routed top-8) | logits `[T, 256]` + bias `[256]` -> `topk_weights [T, 8]` fp32, `topk_ids [T, 8]` int32 | inputs bf16/bf16 (since the bias arrives bf16, the fp32 upcast branch `correction_bias.dtype == float32` is not taken) | - | `aiter.biased_grouped_topk(gating, bias, w, ids, num_expert_group=1, topk_group=1, renormalize=True, routed_scaling_factor=2.5)` | `srt/layers/moe/topk.py:2661-2689 (topk = num_routed_topk = 9 - 1 on aiter), 2206, 1919-1965`; `srt/models/deepseek_v2.py:753-766` |
| M4 | append the shared expert | `[T, 8]` -> `ids [T, 9]` (col 8 = **256**), `weights [T, 9]` (col 8 = **1.0**) | fp32 / int32 | - | Triton `fused_append_shared_experts` (`N = router_logits.shape[1] = 256`, `scale_factor = 1.0` since `fused_shared_experts_scaling_factor` is None) | `srt/layers/moe/topk.py:2335-2339, 2488-2510`; `kernels/ops/moe/fused_moe_triton_kernels.py:1342-1372, 1300-1340` |
| M4a | HIP pad rows (graph padding, `num_token_non_padded`) | padded rows: ids -> 0 (not -1: aiter MoE cannot take -1), all weights -> 0 | - | - | `_mask_topk_ids_padded_region(fill_value=0)` or folded into M4; `_zero_topk_weights_padded_region` | `srt/layers/moe/topk.py:2386-2406, 2536-2540`; `kernels/ops/moe/fused_moe_triton_kernels.py:1327-1332` |
| M5 | dispatch (`StandardDispatcher`) | pass-through; EP 1 -> no `local_expert_mapping`, `expert_mask_gpu = None` | - | - | - | `srt/layers/moe/token_dispatcher/standard.py:143-256` |
| M6 | pre_permute standard->aiter | `topk_weights.to(fp32)`, `topk_ids.to(int32)`; `apply_router_weight_on_input` False | - | - | - | `srt/layers/moe/moe_runner/aiter.py:330-354` |
| M7 | experts: `aiter.fused_moe.fused_moe(x, w1=w13, w2=w2, topk_weight, topk_ids, quant_type=QuantType.per_128x128, activation=ActivationType.Silu, w1_scale, w2_scale, a1_scale=None, a2_scale=None, expert_mask=None, doweight_stage1=False, hidden_pad=0, intermediate_pad=0, moe_sorting_dispatch_policy=...)` | `[T, 6144]` bf16 -> `[T, 6144]` = `sum_{j<9} w[t,j] * down_e(silu(gate_e(x)) * up_e(x))`, e = ids[t,j], each GPU over its 256-column slice of every expert's intermediate -> **partial sum** | in bf16, out bf16 (UNVERIFIED, see list) | F.4 | aiter fused_moe (CK / asm inside aiter) | `srt/layers/quantization/fp8.py:3032-3042, 3305-3398`; `srt/layers/moe/moe_runner/aiter.py:105-137, 175-318` |
| M8 | routed scaling | **not** multiplied after the experts: the `*= routed_scaling_factor` line is skipped under `_use_aiter` ("fused in biased_grouped_topk"); 2.5 is inside the M3 weights; shared column weight 1.0 | - | - | - | `srt/models/deepseek_v2.py:1382-1390`; `srt/layers/moe/topk.py:2453-2466` (comment: aiter folds the factor, shared weight 1.0) |
| M9 | merge + reduce | `maybe_fuse_routed_scale_and_shared_add(experts, out, None, 2.5)` -> returns `out`; `self.reduce_results=False` -> no `post_experts_all_reduce` here; `FusedMoE.reduce_results` default False -> no all-reduce inside FusedMoE | bf16 | - | - | `srt/models/deepseek_v2.py:1413-1420`; `srt/layers/moe/fused_moe_triton/layer.py:336, 1585-1586` |

- MoE reduction: the FFN boundary exit owes one sum of `[T, 6144]` bf16 over the MoE-TP group; at EP 1, MoE-TP = TP 8
  (`srt/layers/moe/utils.py:891-913` `sum_post_experts_output` -> `reduce_tp = moe_tp_size > 1`). Whether it is a
  standalone all-reduce or deferred into the next layer's fused all-reduce + add + RMSNorm is the layer-boundary
  section's business (`srt/layers/layer_boundary/exit.py:150-176, 267-311`); note GLM on HIP sets
  `enable_aiter_allreduce_fusion=True` and `disable_aiter_allreduce_fusion_in_prefill=True` when no attention DP and one node
  (`srt/arg_groups/model_hook.py:438-455`).
- `moe_runner_config.inplace` stays True (aiter is not flashinfer) -> `defer_shared=False`, decoder `_mlp_ctx = nullcontext()`
  (`srt/layers/moe/fused_moe_triton/layer.py:342, 564-568`; `srt/models/deepseek_v2.py:1279, 2814-2825`).
- Math reference for M3 (the torch path SGLang ships, `biased_grouped_topk_impl`): `s = sigmoid(logit)`; choose by
  `s + bias`; weights `= s` gathered at the chosen ids; renormalize by the sum (+ `_RENORMALIZE_SUM_EPSILON`); times
  `routed_scaling_factor` (`srt/layers/moe/topk.py:1604-1680`). With `n_group = topk_group = 1` the group mask keeps all
  256 experts (`:1623-1641`).

### F.4 `FusedMoE` parameters per GPU (TP 8, EP 1; `moe_tp_size = 8`, `moe_intermediate_size 2048 / 8 = 256`)

| Param | Shape per GPU | dtype | Layout after load | Cite |
|---|---|---|---|---|
| `w13_weight` | `[257, 512, 6144]` (rows 0..255 = gate slice, 256..511 = up slice; expert 256 = shared) | fp8 e4m3fn | aiter `shuffle_weight(w, (16, 16))` in place, `is_shuffled = True`, `_aiter_gate_up_interleaved = False` | `srt/layers/quantization/fp8.py:1489-1500, 1567-1576, 2093-2103`; `srt/layers/moe/fused_moe_triton/layer.py:409-410, 742-763` |
| `w2_weight` | `[257, 6144, 256]` | fp8 e4m3fn | shuffled (16, 16) | `srt/layers/quantization/fp8.py:1577-1585, 2098-2100` |
| `w13_weight_scale_inv` | `[257, 2 * (256/128) = 4, 6144/128 = 48]` | fp32 | unchanged (scales not shuffled) | `srt/layers/quantization/fp8.py:1644-1670` |
| `w2_weight_scale_inv` | `[257, 48, 256/128 = 2]` | fp32 | unchanged | same |
| `w13_input_scale`, `w2_input_scale` | None (activation_scheme dynamic) | - | - | `srt/layers/quantization/fp8.py:1726-1748` |

- AITER padding: `AITER_PADDING_SIZE = 128`; 256 % 128 == 0 -> no padding, `weight_padded = False`
  (`srt/layers/moe/utils.py:1027, 1067-1107`).
- TP sharding of each expert: w1/w3 (`gate_proj` / `up_proj` `[2048, 6144]`) narrowed on dim 0 to rows
  `[256*r, 256*(r+1))`; w1 lands at `start = 0`, w3 at `start = shard_size`; w2 (`down_proj` `[6144, 2048]`) narrowed on
  dim 1 (`srt/layers/moe/fused_moe_triton/layer.py:734-806, 808-880`). Scales follow the same narrow at 128-block
  granularity (UNVERIFIED line-by-line; the BLOCK quant_method attr is set at `srt/layers/quantization/fp8.py:1704-1717`).
- Expert ids: `make_expert_params_mapping(num_experts=257)` maps `experts.{e}.{gate,up,down}_proj.` to
  `experts.w13_` / `experts.w2_` (`srt/layers/moe/fused_moe_triton/layer.py:1638-1664`).

### F.5 Weight sharding table, one layer (TP 8, per GPU)

`ignored_layers` = config `modules_to_not_convert` (+ "model."-stripped variants) -> `UnquantizedLinearMethod` for those
modules (`srt/layers/quantization/fp8.py:343-361, 393-397`). Of a decoder layer's linears, the list names only
`mlp.gate`, `indexers_proj`, norms and biases; `lm_head` and `model.embed_tokens` are also listed.

| Checkpoint tensor(s) | SGLang module | Parallel type (group) | Per-GPU shape | dtype after load | Cite |
|---|---|---|---|---|---|
| `self_attn.q_a_proj.weight [2048, 6144]` + `.weight_scale_inv [16, 48]`; `self_attn.kv_a_proj_with_mqa.weight [576, 6144]` + `.weight_scale_inv [5, 48]` | `fused_qkv_a_proj_with_mqa` = `ReplicatedLinear(6144 -> 2048 + 512 + 64 = 2624)`; loader caches both and `torch.cat(dim=0)` (weights and scales alike) | replicated | `w [2624, 6144]`, `scale_inv [21, 48]` | fp8 e4m3fn + fp32 | `srt/models/deepseek_v2.py:2065-2072`; `srt/models/deepseek_common/deepseek_weight_loader.py:412-477` |
| `self_attn.q_a_layernorm.weight [2048]` | `RMSNorm(2048, eps=1e-5)` | replicated | `[2048]` | bf16 (default dtype) | `srt/models/deepseek_v2.py:2073` |
| `self_attn.kv_a_layernorm.weight [512]` | `RMSNorm(512, eps=1e-5)` | replicated | `[512]` | bf16 | `srt/models/deepseek_v2.py:2159` |
| `self_attn.q_b_proj.weight [16384, 2048]` + scale `[128, 16]` | `ColumnParallelLinear(2048 -> 64*256)`, `parallel_group="attn_tp"` (attn_tp = tp / attn_dp / attn_cp = 8) | column | `w [2048, 2048]` (8 heads x 256), `scale_inv [16, 16]` | fp8 + fp32 | `srt/models/deepseek_v2.py:2074-2081`; `srt/runtime_context.py:73-76` |
| `self_attn.kv_b_proj.weight [28672, 512]` + scale `[224, 4]` | `ColumnParallelLinear(512 -> 64*(192+256))`, attn_tp | column | `w [3584, 512]` (8 heads x 448), `scale_inv [28, 4]`; post-load -> `w_kc`, `w_vc` (below) | fp8 + fp32 | `srt/models/deepseek_v2.py:2141-2148` |
| `self_attn.o_proj.weight [6144, 16384]` + scale `[48, 128]` | `RowParallelLinear(64*256 -> 6144)`, attn_tp, `reduce_results=False` | row | `w [6144, 2048]`, `scale_inv [48, 16]` | fp8 + fp32 | `srt/models/deepseek_v2.py:2150-2158, 2613` |
| `self_attn.indexer.wq_b.weight [4096, 2048]` + scale (full layers only) | `ReplicatedLinear(q_lora 2048 -> 32*128)`, `quant_config` passed | replicated | `w [4096, 2048]`, `scale_inv [32, 16]` | fp8 + fp32 (not in ignore list) | `srt/layers/attention/dsa/dsa_indexer.py:333-339` |
| `self_attn.indexer.wk.weight [128, 6144]` + scale (full layers only) | `ReplicatedLinear(6144 -> 128)`, `quant_config` passed (the fused bf16 `wk_weights_proj` is CUDA-only) | replicated | `w [128, 6144]`, `scale_inv [1, 48]` | fp8 + fp32 | `srt/layers/attention/dsa/dsa_indexer.py:299-303, 349-356`; `srt/models/deepseek_common/deepseek_weight_loader.py:89-107, 333-344` |
| `self_attn.indexer.weights_proj.weight [32, 6144]` (name UNVERIFIED, see list) | `ReplicatedLinear(6144 -> 32, params_dtype=bf16)`, **no** quant_config -> unquantized | replicated | `[32, 6144]` | bf16 | `srt/layers/attention/dsa/dsa_indexer.py:357-363` |
| `self_attn.indexer.k_norm.weight [128]`, `.bias [128]` | `LayerNorm(128)`; dtype bf16 under aiter, fp32 when the aiter fused fp8 writer is used | replicated | `[128]`, `[128]` | bf16 or fp32 | `srt/layers/attention/dsa/dsa_indexer.py:364-374, 384-389` |
| (shared layers) any `self_attn.indexer.*` | no indexer module is built (`skip_topk` True); loader skips `.indexer.` names whose prefix has no indexer params | - | - | - | `srt/models/deepseek_v2.py:2110-2113`; `srt/models/deepseek_common/deepseek_weight_loader.py:268-270, 328-331` |
| `input_layernorm.weight [6144]`, `post_attention_layernorm.weight [6144]` | `RMSNorm(6144, eps=1e-5)` | replicated | `[6144]` | bf16 | `srt/models/deepseek_v2.py:2652-2655` |
| dense: `mlp.gate_proj.weight [12288, 6144]`, `mlp.up_proj.weight [12288, 6144]` (+ scales `[96, 48]`) | `gate_up_proj` `MergedColumnParallelLinear`; stacked mapping gate -> shard 0, up -> shard 1 | column per shard | `w [3072, 6144]`, `scale_inv [24, 48]` | fp8 + fp32 | `srt/models/deepseek_v2.py:299-306`; `srt/models/deepseek_common/deepseek_weight_loader.py:231-235, 346-373` |
| dense: `mlp.down_proj.weight [6144, 12288]` + scale `[48, 96]` | `down_proj` `RowParallelLinear` | row | `w [6144, 1536]`, `scale_inv [48, 12]` | fp8 + fp32 | `srt/models/deepseek_v2.py:307-315` |
| MoE: `mlp.gate.weight [256, 6144]` | `MoEGate.weight` (plain `nn.Parameter`, default loader) | replicated | `[256, 6144]` | bf16 | `srt/models/deepseek_v2.py:503-512` |
| MoE: `mlp.gate.e_score_correction_bias [256]` | `MoEGate.e_score_correction_bias` | replicated | `[256]` | fp32 | `srt/models/deepseek_v2.py:513-529` |
| MoE: `mlp.experts.{0..255}.{gate,up,down}_proj.weight` + `.weight_scale_inv` | `experts.w13_weight / w2_weight` (+ scales) | MoE-TP column (w13) / row (w2), all 257 experts on every GPU | F.4 | fp8 + fp32 | F.4 |
| MoE: `mlp.shared_experts.{gate,up,down}_proj.weight` + scales | renamed to `mlp.experts.256.*` -> expert slot 256 of the same tensors | same as routed | intermediate 2048 / 8 = 256 per GPU | fp8 + fp32 | `srt/models/deepseek_common/deepseek_weight_loader.py:285-289` |

`kv_b_proj` -> `w_kc` / `w_vc` (runs inside `load_weights`, i.e. before `process_weights_after_loading`;
`srt/model_loader/loader.py:1026-1028`; `srt/models/deepseek_common/deepseek_weight_loader.py:511, 547-792`):

| Step | Result | Cite |
|---|---|---|
| weight dtype fp8 e4m3fn, block size `[128, 128]`, scale = `weight_scale_inv` | - | `deepseek_weight_loader.py:607-630` |
| not fnuz, not ue8m0 -> `weight = w` | - | `:634-641` |
| not CUDA/MUSA/XPU -> `block_quant_to_tensor_quant(weight, scale, [128,128])`: dequant every 128x128 block in fp32, then `input_to_float8(dtype=e4m3fn)` over the **local shard** | fp8 e4m3fn `[3584, 512]`; `self_attn.w_scale` = fp32 scalar `amax / 448` of this GPU's shard | `:662-685`; `srt/layers/quantization/fp8_utils.py:1878-1903, 1906-1946` |
| GLM bf16 requant branch (`_use_aiter_gfx95 and arch == GlmMoeDsaForCausalLM and w.dtype == bf16`) | **not taken** (w is fp8 already) | `:724-732` |
| split `w.unflatten(0, (-1, 192 + 256)).split([192, 256], dim=1)` | `w_kc [8, 192, 512]`, `w_vc [8, 256, 512]` | `:734-736` |
| store | `w_kc` = `w_kc.transpose(1,2).contiguous().transpose(1,2)` -> shape `[8, 192, 512]`, column-major per head; `w_vc` = `w_vc.contiguous().transpose(1,2)` -> shape `[8, 512, 256]` | `:751-758` |
| `w_scale` from `kv_b_proj.weight_scale` | skipped: block-FP8 layers register only `weight_scale_inv`, and `w_scale` is already set | `:759-767`; `srt/layers/quantization/fp8.py:668-681` |

### F.6 Block-FP8 linear kernel on gfx950 (every fp8 `Linear` above)

| Item | Value | Cite |
|---|---|---|
| Dispatch | `dispatch_w8a8_block_fp8_linear([128,128])` -> auto -> `_use_aiter` -> `aiter_w8a8_block_fp8_linear` | `srt/layers/quantization/fp8.py:535-544`; `srt/layers/quantization/fp8_utils.py:734-770, 1145-1163` |
| MXFP8 view | off: `can_serve_block_fp8_as_mxfp8` needs block K 32 + ue8m0 | `srt/layers/quantization/fp8.py:547-554`; `srt/layers/quantization/fp8_utils.py:912-921` |
| Activation quant (input bf16) | `aiter_per1x128_quant(x, quant_dtype=aiter.dtypes.fp8)` -> fp8 + per-token per-128-group fp32 scale | `srt/layers/quantization/fp8_utils.py:352, 1650-1664` |
| GEMM | ROCm >= 7.2 (`_use_aiter_bpreshuffle_gfx95`) and `(n, k)` not in the tuned-Triton list -> CK `gemm_a8w8_blockscale_bpreshuffle`; ROCm < 7.2 -> CK `gemm_a8w8_blockscale` (Triton only for listed shapes / above CK-safe M) | `srt/layers/quantization/fp8_utils.py:84, 320-337, 1617-1689` |
| Per-GPU (n, k) here | qkv_a (2624, 6144), q_b (2048, 2048), kv_b (3584, 512), o_proj (6144, 2048), wq_b (4096, 2048), wk (128, 6144), dense gate_up (3072, 6144), dense down (6144, 1536): none is in the tuned list (`(7168,…)`, `(16384,1536)`, … are DeepSeek shapes) | `srt/layers/quantization/fp8_utils.py:320-337` |
| Weight preshuffle | if bpreshuffle is on and the shape is not tuned-Triton: `shuffle_weight(w, (16,16))` in place, `aiter_bpreshuffled = True` | `srt/layers/quantization/fp8.py:884-908` |
| Output | dtype of the input (bf16) | `srt/layers/quantization/fp8_utils.py:1572, 1689` |

### F.7 Notes from the code

- Router bias precision: `MoEGate` keeps `e_score_correction_bias` fp32 for GLM on purpose (comment: bf16 "reorders top-k
  routing", `srt/models/deepseek_v2.py:515-517`), and `biased_grouped_topk_gpu` has an fp32 branch for it
  (`srt/layers/moe/topk.py:1944-1951`). But the router GEMM returns bf16 logits (`otype=hidden_states.dtype`,
  `srt/layers/rocm_linear_utils.py:35`) and `select_experts` first casts the bias to the logits dtype
  (`srt/layers/moe/topk.py:2619-2626`), so by code reading the aiter top-k receives **bf16 logits and a bf16 bias**.
  Check on the box (print `correction_bias.dtype` inside `biased_grouped_topk_gpu`).
- The shared expert is expert 256 of the same `fused_moe` call, with weight 1.0 while routed weights carry x2.5; the
  shared and routed experts use the same TP split (256 columns per GPU).

### F.8 UNVERIFIED

| Item | Why |
|---|---|
| aiter internals: `biased_grouped_topk` formula (sigmoid, renorm, x2.5 placement), `fused_moe` activation quant for `per_128x128` (per-token 1x128 fp8 inside?), its stage split, output dtype (bf16 assumed from the bf16 input), `tgemm.mm` kernel choice | aiter is not installed / its source is not in this tree; only the call sites are readable |
| ROCm version on the box (>= 7.2 decides bpreshuffle CK vs plain CK GEMM and the weight preshuffle) | runtime fact (`get_hip_version()`), not in the code |
| `aiter.dtypes.fp8` is e4m3fn on gfx950 | defined in aiter |
| Checkpoint name of the indexer head-weight projection: config `modules_to_not_convert` lists `model.layers.N.self_attn.indexers_proj` (21 full layers + 78), but SGLang only knows `indexer.weights_proj`; a tensor named `self_attn.indexers_proj.*` would hit "not found in params_dict" and be skipped (`srt/models/deepseek_common/deepseek_weight_loader.py:489-494`) | checkpoint index / safetensors headers not available here; check `model.safetensors.index.json` |
| Checkpoint dtypes: `mlp.gate.weight` (ignore-listed, so not fp8; stored bf16 or fp32 -> cast to bf16 param on load), `e_score_correction_bias` (fp32 param either way), `kv_b_proj`, `indexer.wq_b`, `indexer.wk` being block-fp8 (they are not ignore-listed, so SGLang allocates fp8 params; a bf16 tensor there would be mis-loaded) | safetensors headers not read |
| Exact narrow of the block scales in `_load_w13` / `_load_w2` and in `MergedColumnParallelLinear` / `RowParallelLinear` loaders for 128-block scales | not traced line by line; shapes above follow `create_fp8_moe_weight_` / `create_fp8_weight_` allocation |
| (resolved in section 1, L.2/L.3: the FFN sum is deferred and fused into the next layer's aiter AR + add + RMSNorm when `aiter_ar_fusion_applies`) | - |

## 5. UNVERIFIED, all sections (details and reasons in L/2/D-/F.8 lists above)
| # | Item | Reason | Decides |
|---|---|---|---|
| 1 | aiter kernel internals: `fused_fp8_bmm_rope_cat_and_cache_mla`, `fused_qk_rmsnorm`, `get_rope`, `biased_grouped_topk`, `fused_moe(per_128x128)`, `rmsnorm2d_fwd_with_add`, custom AR / fused AR+RMSNorm, `deepgemm_fp8_paged_mqa_logits`, `indexer_k_quant_and_cache`, `tgemm.mm`, `aiter_per1x128_quant` | aiter not installed here; only SGLang call sites read | exact rounding points, KV-write byte layout of the fused MLA kernel, MoE activation quant |
| 2 | gfx950 fused indexer actually on (4 JIT modules build against the image's aiter; workspace width vs graph page-table width) | needs the box / `extra_max_context_len` not traced | D3b vs D3c for T <= 48 |
| 3 | `aiter_can_use_preshuffle_paged_mqa()` True on the box (Triton >= 3.5 or AOT gluon) | runtime | page_size 64 vs forced 1 |
| 4 | ROCm >= 7.2 | runtime `get_hip_version()` | CK bpreshuffle GEMM + weight preshuffle vs plain CK |
| 5 | checkpoint tensor names / dtypes: `self_attn.indexers_proj` (in `modules_to_not_convert`, no SGLang module of that name) vs `indexer.weights_proj`; `mlp.gate.weight`, `kv_b_proj`, `indexer.wq_b`, `indexer.wk` dtypes; presence of `k_scale` / `v_scale` | safetensors index / headers not available here | whether the indexer head-weight projection loads at all; MLA cache `k_scale` (1.0 if absent) |
| 6 | router bias dtype seen by aiter top-k: code casts the fp32 `e_score_correction_bias` to the bf16 logits dtype before `aiter_biased_grouped_topk` (`srt/layers/moe/topk.py:2616-2626`) | by code reading; print `correction_bias.dtype` on the box | top-8 selection ties / order |
| 7 | sampler inside or outside the decode graph; default temperature / top-p | runner call site not read; generation_config / request | greedy (aiter `greedy_sample`) vs torch sampling |
| 8 | layer 77 FFN sum: deferred to `complete_output` or done at exit | not resolved; both are one plain TP all-reduce before `model.norm` | none for the math |

## 6. Checked against the checkpoint index (2026-11-10, model.safetensors.index.json of zai-org/GLM-5.3)
- The indexer's weight is `self_attn.indexer.weights_proj.weight` (22 tensors: the 21 "full" layers + the MTP layer 78); no
  tensor is named `indexers_proj`: SGLang loads it (UNVERIFIED item 5, first part, resolved).
- No `weight_scale_inv` (bf16): `mlp.gate.weight` (76 = 75 MoE layers + MTP), `indexer.weights_proj.weight`.
- With `weight_scale_inv` (fp8 128x128 blocks): `kv_b_proj`, `indexer.wq_b`, `indexer.wk`, every q_a / kv_a / q_b / o_proj,
  experts, shared experts, dense MLP.
- No `k_scale` / `v_scale` tensors: the MLA cache is written with scale 1.0.

## 7. aiter fused all-reduce + add + RMSNorm numerics (read 2026-11-10 in lmsysorg/sglang-rocm:v0.5.22-rocm10-mi35x-20261019,
`/sgl-workspace/aiter/csrc/include/custom_all_reduce.cuh`)
- 1-stage (`allreduce_fusion_kernel_1stage`, :1975-2090): acc = sum over ranks 0..7 in order of the bf16 partials (fp32);
  acc = fp32(bf16(acc)); acc += fp32(residual); residual_out = bf16(acc); RMSNorm on acc (fp32, unrounded):
  s = rsqrtf(sum(acc^2) / H + eps) (block reduce), out = bf16(acc * s * fp32(w)) (`ar_fusion_epilogue_rms_norm`).
- 2-stage (`reduce_scatter_cross_device_store` :1279 + `local_device_load_rmsnorm`): rms_inp = fp32(residual) + fp32(reduced bf16
  sum); the same norm on the fp32 rms_inp. (Which reduce order the 2-stage scatter uses: not read.)
