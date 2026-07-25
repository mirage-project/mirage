# MPK capability map and gap analysis for Qwen3.5-35B-A3B-FP8

**Scope.** What Mirage Persistent Kernel already has, and what is missing, for serving
`Qwen/Qwen3.5-35B-A3B-FP8` (text decode) on a single B200 (sm_100).

**Method.** Read-only inspection of `/home/catalyst/project` at branch `qwen3-5_support`,
HEAD `2c87a75` ("Support DFlash for Kimi-K2.6 (#728)"), which is byte-identical to `mpk`
at the time of writing (`git rev-parse qwen3-5_support mpk` → same sha). Every claim below
carries a `file:line` or a commit id. Question (d) was resolved by actually running the
cherry-pick in a throwaway `git worktree`. Where this document contradicts the earlier
scouting report (`design/scouting/mpk-internals.md` in the agent repo), the contradiction is
called out explicitly in §6.

**Target shapes** (from the vLLM/SGLang graph scouting docs, `config.json` of the checkpoint):

| | |
|---|---|
| layers | 40 = 30 × Gated-DeltaNet linear attention + 10 × full attention (3:1 pattern) |
| hidden | 2048; vocab 248320 |
| full attn | 16 Q / 2 KV heads, `head_dim` 256, `partial_rotary_factor` 0.25 → `rotary_dim` 64, `attn_output_gate=true` |
| linear attn | conv kernel 4; 16 K heads × 128, 32 V heads × 128 |
| MoE | 256 experts, top-8, `norm_topk_prob=true`, `moe_intermediate_size` 512; 1 shared expert (inter 512) with a sigmoid gate |
| FP8 | block-wise 128×128 weight scales, dynamic activation quant; `modules_to_not_convert` (lm_head, embeddings, conv1d, linear-attn a/b projections, router gate, shared_expert_gate) stay bf16 |

---

## 1. Existing-task inventory — what a Qwen3.5 graph could reuse

Only ops relevant to this model are listed. "Layer" = method in
`python/mirage/mpk/persistent_kernel.py`; "task name" = the string dispatched in
`src/kernel/graph.cc`. Enum values from
`include/mirage/persistent_kernel/runtime_header.h:84-201`.

| Op | Layer method (`persistent_kernel.py`) | Task name / TaskType | Kernel file | Instantiated by (shape) |
|---|---|---|---|---|
| Embedding | `embed_layer:627` | `embedding` / `TASK_EMBEDDING=101` | `tasks/ampere/embedding.cuh`, `hopper/embedding_hopper.cuh` — **no SM100 variant** | Qwen3 demo, DSV3 builder |
| RMSNorm | `rmsnorm_layer:650` | `rms_norm_hopper` for cc≥90 / `TASK_RMS_NORM_HOPPER=154` | `tasks/hopper/rmsnorm_hopper.cuh` | all models |
| Fused RMSNorm+Linear | `rmsnorm_linear_layer:668` | `rms_norm_linear*` | `tasks/ampere/norm_linear*.cuh`, `hopper/norm_linear_hopper.cuh` — **no SM100 fused version** | Qwen3 demo |
| Linear bf16 | `linear_layer:1861`, `linear_with_residual_layer:1892`, `splitk_linear_layer:1836` | `linear_sm100` / `TASK_LINEAR_SM100=253`, `TASK_LINEAR_WITH_RESIDUAL_SM100=252`, `TASK_SPLITK_LINEAR_SM100=251` | `tasks/blackwell/linear_sm100_mpk.cuh` | Qwen3-8B (hidden 4096), DSV3 bf16 fallback |
| **Quantize bf16→FP8** | `quantize_fp8_layer:1681` | `quantize_fp8_sm100` (UE8M0) / `quantize_fp8_f32scale_sm100` (fp32) — `TASK_QUANTIZE_FP8_SM100=275` | `tasks/blackwell/per_token_group_quantize_fp8.cuh` | DSV3 only: K ∈ {7168, 1536, 2048, …} |
| **Dense FP8 GEMM** | `linear_fp8_layer:1704`, `linear_fp8_with_residual_layer:1731` | `linear_fp8_sm100` / `TASK_LINEAR_FP8_SM100=276`, `..._WITH_RESIDUAL=277` | `tasks/blackwell/linear_fp8_sm100.cuh` (+8 support headers), `linear_fp8_1d2d_sm100.cuh` | DSV3 only (hidden 7168, q_lora 1536, kv_lora 512, dense inter 18432) |
| GQA paged decode + fused QK-norm + RoPE + KV store | `paged_attention_layer:896` | `paged_attention_sm100` / `TASK_ATTN_SM100=257` | `tasks/blackwell/attention_sm100.cuh` | Qwen3-8B (32Q/8KV, hd 128), Qwen3-30B-A3B |
| Split-KV attention + merge | `paged_attention_split_kv_layer:985`, `..._merge_layer:1073` | `TASK_PAGED_ATTENTION_SPLIT_KV_SM100=263`, `..._MERGE=264` | `tasks/blackwell/attention_sm100*_split_kv.cuh` | Qwen3 demo on B200 |
| MoE routing, softmax top-k, renormalized | `moe_topk_softmax_routing_layer:1505` | `moe_topk_softmax_sm100` / `TASK_MOE_TOPK_SOFTMAX_SM100=260` | `tasks/blackwell/topk_softmax_sm100.cuh` | Qwen3-30B-A3B (128 experts, top-8) |
| MoE routing, sigmoid + group/noaux_tc | `moe_topk_sigmoid_routing_layer:1528` | `moe_topk_sigmoid_sm100` / `=280` | `tasks/blackwell/topk_sigmoid_sm100.cuh` | DSV3 (256 experts, top-8, 8 groups) |
| MoE grouped GEMM bf16 | `moe_w13_linear_layer:1564`, `moe_w2_linear_layer:1772` | `moe_w13_linear_sm100`/`moe_w2_linear_sm100` / `=254`,`=255` | `tasks/blackwell/moe_linear_sm100.cuh` | Qwen3-30B-A3B (128e, inter 768) |
| **MoE grouped GEMM FP8** | `moe_w13_fp8_layer:1595`, `moe_w2_fp8_layer:1638` | `moe_w13_fp8_sm100`/`moe_w2_fp8_sm100` / `TASK_MOE_W13_FP8_SM100=248`, `..._W2_...=249` | `tasks/blackwell/fp8_group_gemm_sm100.cuh` (81 KB) | DSV3 only: `[256, 4096, 7168]` and `[256, 7168, 2048]` |
| MoE SiLU-mul | `moe_silu_mul_layer:1756` | `moe_silu_mul` variant of `TASK_SILU_MUL` | `tasks/ampere/silu_mul.cuh` | both MoE models |
| MoE weighted combine + residual | `moe_mul_sum_add_layer:1803` | `moe_mul_sum_add_sm100` / `=261` | `tasks/blackwell/mul_sum_add_sm100.cuh` | both MoE models |
| Dense SiLU-mul | `silu_mul_layer:1956` | `silu_mul*` / `TASK_SILU_MUL=118` | `tasks/ampere/silu_mul.cuh`, `hopper/silu_mul_hopper.cuh` — **no SM100 file** | Qwen3, DSV3 shared expert |
| Elementwise add | `elementwise_add_layer:1993` | `TASK_ELEMENTWISE_ADD_SM100=281` | `tasks/blackwell/elementwise_add_sm100.cuh` | DSV3 |
| argmax (greedy) | `argmax_partial_layer:2046`, `argmax_reduce_layer:2071` | `=259`, `=258` | `tasks/blackwell/argmax_sm100.cuh` | all |
| MLA prefill (chunked) | `mla_prefill_layer:1227` | `TASK_MLA_PREFILL_SM100=268` | `tasks/blackwell/mla_prefill_sm100.cuh` | DSV3 only |
| DFlash (Kimi-K2.6 draft) | `dflash_attention_layer:755`, `dflash_norm_rope_layer:782`, `dflash_kv_store_layer:805` | `=296`,`=297`,`=298` | `tasks/blackwell/dflash_*_sm100.cuh` | Kimi-K2.6 draft, **bf16 only** |

**Confirmed absent** (repo-wide grep, `deps/` excluded): `deltanet`, `delta_net`, `linear_attn`,
`linear_attention`, `gated_delta`, `causal_conv`, `conv1d`, `mamba`, `chunk_scan`, `recurrent`,
`gdn`, and any attention output gate, zero-centered-gamma RMSNorm, or partial-RoPE `rotary_dim`
parameter — all zero hits. There is also no FP8 KV cache anywhere.

---

## 2. Answer (a) — MPK's FP8 status

### 2.1 Which paths ship FP8

**DeepSeek-V3 only.** Kimi-K2.6 / DFlash has **zero** FP8:
`grep -rin "fp8\|e4m3\|quantiz" python/mirage/mpk/models/dflash/` returns nothing, and
`python/mirage/mpk/models/qwen3/builder.py` likewise has no FP8. The FP8 layers are hard-gated
to Blackwell — `assert self.target_cc == 100, "FP8 group GEMM requires SM100 (Blackwell)"`
(`persistent_kernel.py:1635`, `:1677`).

Three FP8 task families exist, all registered from `src/kernel/graph.cc:666-671, 793-809`:

| Task | Register fn | Kernel | Provenance |
|---|---|---|---|
| `TASK_QUANTIZE_FP8_SM100=275` | `task_register.cc:4153` | `per_token_group_quantize_fp8.cuh` | in-house |
| `TASK_LINEAR_FP8_SM100=276`, `..._WITH_RESIDUAL=277` | `task_register.cc:4216` | `linear_fp8_sm100.cuh` | DeepGEMM-derived (CUTLASS/CuTe, `kGranKA/kGranKB` naming) |
| `TASK_MOE_W13_FP8_SM100=248`, `TASK_MOE_W2_FP8_SM100=249` | `task_register.cc:2733` | `fp8_group_gemm_sm100.cuh` | in-house grouped GEMM |

### 2.2 Is it block-FP8 with 128×128 scales and dynamic activation quant? — Yes, via two different scale paths

**Activation quantization is dynamic, per-token, per-128-element group.**
`register_quantize_fp8_sm100_task` hardcodes `constexpr int GROUP_SIZE = 128;`
(`task_register.cc:4189`); the kernel amax-reduces each 128-wide group of a token row and emits
E4M3 plus either a packed UE8M0 `uint32` scale (dense GEMM path) or an fp32 scale (MoE path)
— `per_token_group_quantize_fp8.cuh:38-80`, `static_assert(GROUP_SIZE == 128, "Packed UE8M0
scale currently requires GROUP_SIZE == 128")`. Clamp range `[-448, 448]`, eps `1e-10`
(`task_register.cc:4207`). This is the standard DeepSeek/vLLM "1×128 dynamic activation" scheme.

**Weight scales: the checkpoint's 128×128 blocks are consumed, but the two paths differ.**

*Dense linear* re-quantizes. `DeepSeekV3Builder._requantize_fp8_for_ue8m0`
(`models/deepseek_v3/builder.py:475-542`) takes the checkpoint's
`scale_inv: [ceil(M/128), ceil(K/128)] float32`, dequantizes to fp32, then **re-quantizes with
per-row × per-128-K-group power-of-two (UE8M0) scales** and packs them 4-per-`uint32` in a
column-major `[M, packed_k]` layout. The docstring states the reason: "SM100 block-scaled UMMA
uses UE8M0 (8-bit exponent-only) scale factors. Checkpoint float32 scales are NOT powers of 2
… Fix (same as SGLang/vLLM): dequant → re-quantize with power-of-2 scales"
(`builder.py:477-483`). The GEMM is then instantiated with `kGranKA = kGranKB = 128`
(`task_register.cc:4272`), `BLOCK_M/N/K = 32/16/128` (`:4277`), `kNumStages=25` chosen to fit the
207 KB smem budget (`:4280`), A/B `float_e4m3_t`, **C/D `bfloat16_t`** (`:4291`).

*MoE grouped GEMM* does **not** re-quantize; it expands. `builder.py:920-931`:
"Checkpoint: scale_inv `[num_experts, out/128, K/128]` → Kernel expects scale
`[num_experts, out_rows, K/128]` (per-row, float32)", implemented as
`raw_scale_inv.repeat_interleave(128, dim=1)`. The kernel then does its own internal UE8M0
conversion (`builder.py:945`). So the MoE path preserves the checkpoint's exact 128×128 block
values (replicated across rows) while the dense path re-derives finer, power-of-2 scales.

**Net:** Qwen3.5's FP8 is **structurally expressible with the existing kernels** — they already
consume a DeepSeek-format `weight_scale_inv` checkpoint at 128×128 with dynamic 1×128 activation
quant, which is exactly our checkpoint's `quantization_config`
(`weight_block_size: [128,128]`, `activation_scheme: dynamic` —
`docs/qwen35/vllm-graph.md:701-705`). **Correctness and efficiency at our shapes remain
unproven** (§2.3, §2.5).

#### 2.2.1 The two scale paths differ, and our GEMMs split across both

This distinction governs every "MPK-FP8 ≠ HF-FP8" statement in this document, so state it once.

| | dense `linear_fp8_sm100` | MoE `moe_w13/w2_fp8_sm100` |
|---|---|---|
| the checkpoint's `[N/128, K/128]` block scale | **discarded after dequant** — weights are dequantized to fp32 then **re-quantized** with fresh **per-row (1×128) power-of-two UE8M0** scales (`builder.py:475-542`) | **preserved** — `repeat_interleave(128, dim=1)` expands the same block values to per-row fp32; the kernel does its own internal UE8M0 conversion (`builder.py:920-931, 945`) |
| effective weight-scale granularity at the MMA | 1×128, values snapped to powers of two | 128×128 values, replicated across rows |
| activation scale | per-token 1×128, **packed UE8M0** (`scale_ue8m0=True`) | per-token 1×128, **fp32** (`scale_ue8m0=False`, `builder.py:951`) |
| vs HF's block dequant | **finer scales, snapped to powers of two** — a real numeric delta, both directions | closer; activations and accumulation order still differ |

Which of *our* planned GEMMs inherits which (checkpoint scale shapes from
`docs/qwen35/vllm-graph.md:881-883, 1007-1008, 1475-1499`):

| Qwen3.5 GEMM | ckpt `weight_scale_inv` | MPK path | inherits |
|---|---|---|---|
| `linear_attn.in_proj_qkvz` `[12288,2048]` | `[96,16]` | dense | **UE8M0 requant** |
| `linear_attn.out_proj` `[2048,4096]` | `[16,32]` | dense | **UE8M0 requant** |
| `self_attn.qkv_proj` `[9216,2048]` | `[72,16]` | dense | **UE8M0 requant** |
| `self_attn.o_proj` `[2048,4096]` | `[16,32]` | dense | **UE8M0 requant** |
| `shared_expert.gate_up_proj` `[1024,2048]` | `[8,16]` | dense | **UE8M0 requant** |
| `shared_expert.down_proj` `[2048,512]` | `[16,4]` | dense | **UE8M0 requant** |
| routed `w13` `[256,1024,2048]` | `[256,8,16]` | MoE grouped | **scales preserved** |
| routed `w2` `[256,2048,512]` | `[256,16,4]` | MoE grouped | **scales preserved** |
| `in_proj_ba`, router `gate`, `shared_expert_gate`, `lm_head`, embeddings, conv1d | — | bf16 | neither (`modules_to_not_convert`) |

So **every dense projection in the model takes the requantized path, and every routed-expert
GEMM takes the preserved-scale path.** One further wrinkle from the checkpoint side: it stores
`weight_scale_inv` in **BF16**, not fp32 (`docs/qwen35/vllm-graph.md:752, 1672`) — 8 mantissa
bits. MPK's loader does `state_dict[scale_key].float()` (`builder.py:924, 1005`), which widens
but cannot recover precision; vLLM likewise holds them as fp32 at runtime. Any per-op comparison
against an HF or vLLM reference must start from the same bf16 source values.

### 2.3 Reusable at our shapes? — structurally yes; unproven at these sizes

Shape rules extracted from source:

* Dense FP8: `output_size >= 128` or the builder raises
  (`builder.py:134-139`: "FP8 linear: output_size=… < 128 (BLOCK_N). Must use BF16 linear for
  this dimension."); `K` must be a multiple of 128 (scale groups); `M < BLOCK_M` is handled by
  the kernel's `kAlignedShapeM` clamp (`linear_fp8_sm100.cuh:120-129`), so mbt ∈ {1..16} is fine.
* MoE FP8: weight TMA tile is `MMA_M=128` rows of N × `bK=128` of K
  (`task_register.cc:2809-2811`); the per-CTA N slice must be a multiple of 128, which
  `_moe_fp8_m_split()` (`builder.py:55-60`, `_MOE_FP8_MMA_M = 128` at `:52`) enforces by picking
  `grid_dim.y`.

Applied to Qwen3.5 (bf16 columns = `modules_to_not_convert`):

| GEMM | N × K | FP8-eligible? | Verdict |
|---|---|---|---|
| full-attn `qkv_proj` (q‖gate ‖ k ‖ v) | 9216 × 2048 | yes | N/128 = 72 ✓, K/128 = 16 ✓ |
| `o_proj` (both attn kinds) | 2048 × 4096 | yes | 16 ✓, 32 ✓ |
| GDN `in_proj_qkvz` | 12288 × 2048 | yes | 96 ✓, 16 ✓ |
| GDN `in_proj_ba` | 64 × 2048 | **bf16** (not_convert) | would fail `output_size >= 128` if it were FP8 |
| router `gate` | 256 × 2048 | **bf16** (not_convert) | — |
| shared expert `gate_up` | 1024 × 2048 | yes | 8 ✓, 16 ✓ |
| shared expert `down` | 2048 × 512 | yes | 16 ✓, K/128 = 4 ✓ |
| `shared_expert_gate` | 1 × 2048 | **bf16** (not_convert) | N=1, could never be FP8 |
| routed `w13` | `[256, 1024, 2048]` | yes | max `grid_dim.y` = 1024/128 = 8 |
| routed `w2` | `[256, 2048, 512]` | yes | max `grid_dim.y` = 16, K = 4 k-tiles |
| `lm_head` | 248320 × 2048 | **bf16** (not_convert) | — |

Every FP8 op we need is *expressible* with existing tasks and no new kernel. That is a shape
argument, not a correctness argument: the shapes are **smaller and shallower than anything
currently exercised** (`moe_intermediate` 512 vs DeepSeek-V3's 2048; w2 reduction 512 = only 4
k-tiles). The MoE pipeline depth in particular was tuned for a different regime —
`num_ab_stages = 8` was raised from 4 specifically because "at TP=4, moe_w2_fp8 has
`fp8_k_tile_count = 8` … with 4 stages the pipeline hung" (`task_register.cc:2812-2818`); our w2
has 4 k-tiles, i.e. below the count that motivated the current setting. Treat "no new kernel" as
the *expected* outcome, and hold the claim open until the shapes are actually run: a hang or a
numerics failure at K=512 would be a kernel change, not a tuning change.
The FP8 kernels' only in-tree tests are at DeepSeek-V3 shapes:
`tests/runtime_python/blackwell/sm100_fp8_moe/test_fp8_moe_gemm.py:35-36` (`N=4096, K=7168`) and
`:147-148` (w2 `N=7168, K=2048`); the pipeline test
`tests/runtime_python/test_mode/test_fp8_moe_pipeline_testmode.py:76-81` uses
`E=64, topk=8, K=7168, I=2048, B=16`.

### 2.4 Do attention / GDN inputs stay bf16 there? — Yes, unconditionally

* Dense FP8 GEMM output dtype is `cutlass::bfloat16_t` (`task_register.cc:4291`).
* MoE FP8 GEMM outputs are BF16 (`persistent_kernel.py:1613`, `:1656` docstrings).
* Paged attention is instantiated `multitoken_paged_attention_sm100_task_impl<bfloat16, …>`
  (`task_register.cc:2072`) and there is no dtype parameter exposed on `paged_attention_layer`.
* MLA prefill/decode take `__nv_bfloat16 const *` (`mla_prefill_sm100.cuh:206-210`).
* No FP8 KV cache exists (`kv_cache_dtype`/`fp8_kv` — zero hits repo-wide).

So the FP8 boundary in MPK is exactly "GEMM in FP8, everything between GEMMs in bf16", with an
explicit re-quantize task inserted before each FP8 GEMM. DeepSeek-V3 does this three times per
MoE block: quantize → w13 → silu → **quantize again** → w2 (`builder.py:946, 1025`). Qwen3.5's
GDN kernels would sit on the bf16 side, same as attention.

### 2.5 What is missing for our ops

No *known* kernel gap — the missing work is wiring and tuning. ("No known gap" is the honest
form: no FP8 kernel has been run at our shapes, so a hang or numerics failure at
`moe_intermediate = 512` / w2 K = 512 would reopen this, per §2.3.)

1. No builder consumes an FP8 checkpoint outside `models/deepseek_v3/builder.py`; the
   requantize + attach helpers (`_requantize_fp8_for_ue8m0`, `_attach_fp8_weight`,
   `_fp8_linear`) are private methods of that class and must be lifted or copied.
2. The FP8 dense path always allocates a shared per-`reduction_size` quantization buffer
   (`builder.py:156-176`). With Qwen3.5 having only two distinct K values (2048 and 4096) this
   caches well, but the buffer is shared *across layers*, which serialises quantize tasks that
   could otherwise overlap.
3. FP8 has never been run with `moe_intermediate = 512` or a w2 reduction of 512 — the
   `num_ab_stages = 8` pipeline depth (`task_register.cc:2818`) was tuned for
   `fp8_k_tile_count = 8`; ours is 4 for w2.
4. `use_cutlass_kernel=True` was required for the DFlash bring-up ("PTX path hangs at large
   reductions", commit `2c87a75`); assume the same for anything CUTLASS-backed.

---

## 3. Answer (b) — linear-attention prefill under MODE_OFFLINE

### 3.1 How MPK processes prompt tokens today

`MODE_OFFLINE`'s `prepare_next_batch` is
`include/mirage/persistent_kernel/persistent_kernel.cuh:222-417`. It runs on **one thread of one
scheduler warp** at each `EVENT_END_OF_TASK_GRAPH` (the source carries
`// TODO: parallelize this processing` at `:223`). The prefill mechanism is:

* A newly admitted request gets `num_new_tokens = min(prompt_length, mbt - num_tokens)`
  (`:368-369`); an already-running request in prefill gets
  `num_new_tokens = min(prompt_length - step, mbt - num_tokens)` (`:313-317`); a decoding
  request gets `min(1, …)` (`:326`).
* `step[request_id] += num_tokens` each iteration (`:268`).
* Prefill and decode requests coexist in one iteration, and one prompt is spread over
  `ceil(prompt_len / mbt)` iterations of the **same static task graph**.

**There is no separate prefill graph.** The only signal a task gets is
`qo_indptr_buffer[i+1] - qo_indptr_buffer[i]`. Every per-token layer already handles this
because the token dimension is either grid-parallel (`grid_dim=(mbt,1,1)`) or a compile-time
`BATCH_SIZE` template arg masked at runtime by
`qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]` (`task_register.cc:285` and friends).

For attention specifically, `attention_sm100.cuh` derives at runtime
`num_tokens` (chunk length, `:100`) and `seq_len` (history + chunk, `:111-112`), and applies the
causal mask against `seq_len - num_tokens` (`:114`, `:248`, `:345`). Its own header comment:
"this task implements the paged attention where a causal mask is applied. In each task, we
process one request with one or more tokens" (`:32-33`). So **the same task is MPK's prefill and
decode attention** — this is what `demo/qwen3/demo.py:587` uses, unconditionally, on the
non-spec-decode path. `single_batch_extend_attention_layer` is only reached under
`spec_decode_config` (`demo.py:549-550`).

Everything a task needs to distinguish prefill from decode is reachable: `RuntimeConfig` exposes
`step[]`, `prompt_length[]`, `request_ids[]`, `qo_indptr_buffer[]`, `paged_kv_*`
(`runtime_header.h:328-347`), and the code generator routinely injects them
(`task_register.cc:82, 285, 335, 403, 1238, 2090, 3493, …`).

### 3.2 Options for GDN prefill, with evidence of what is already expressible

**Option 1 — chunked scan inside the megakernel, same task as decode (recommended).**
A GDN task reads `Q_LEN = qo_indptr[bi+1] - qo_indptr[bi]`, loops over those ≤ mbt tokens
sequentially applying the gated delta rule, and leaves the recurrent state in a persistent
global tensor indexed by request slot. Cross-iteration recurrence is free because MPK serialises
task-graph iterations.

Evidence this is expressible, all from the shipped MLA prefill registration
(`task_register.cc:3738-3820`), which generates exactly this code:

```
int bi_    = task_desc->task_metadata.request_id;
int qo_fp_ = runtime_config.qo_indptr_buffer[bi_];
int S_     = (lp_ - fp_ - 1) * MPK_PAGE_SIZE
             + runtime_config.paged_kv_last_page_len_buffer[bi_];
int Q_LEN_ = runtime_config.qo_indptr_buffer[bi_ + 1]
             - runtime_config.qo_indptr_buffer[bi_];
auto *ckv_ptr_ = ... + bi_ * MPK_MAX_SEQ_LENGTH * D_CKV;   // per-request slice
```

That last line is the exact addressing a per-request recurrent state needs
(`state_base + request_id * state_stride`), and it is already in production. Comment at
`:3774-3777`: "Compute S (total KV length) and Q_LEN (this iteration's chunk length) at runtime
… Q_LEN = num_new_tokens this iteration (can be smaller than mbt)."

**Option 2 — separate prefill and decode tasks, dual-dispatched (also expressible today).**
Register *both* kernels for the same step; each early-exits on a runtime `Q_LEN` gate. This is
shipped for MLA. `models/deepseek_v3/builder.py:686-733`: "When `_use_prefill` is True
(mbt ≥ 32), BOTH the prefill kernel and the decode kernels are registered so that a single
compiled task graph handles both regimes at runtime … Both write `self.attn_out`. Builder order
is prefill → decode; the MPK event graph serialises the two writes, so whichever kernel really
runs produces the final value (the other becomes a no-op)." The gates are literal:
`if (Q_LEN < 16) { return; }` in `mla_prefill_sm100.cuh:228` and
`if (Q_LEN > 8) { return; }` in `mla_mtp_decode_sm100.cuh:110` and `:706`, with the
non-overlap explicitly documented at `mla_prefill_sm100.cuh:220-227`.
Cost: doubles the enum ids consumed and adds dead tasks to every iteration.

**Option 3 — recurrent loop (one task per token).** Would require `mbt` sequential task
instances with a serial dependency chain per layer. MPK's event graph can express the
dependency, but the task graph is **static**, so the chain length is fixed at `mbt` regardless
of how many tokens are live. Expected to be worse than Option 1 for v1, for three reasons, none
of which have been measured: (i) in decode each request supplies one token, so `mbt-1` of the
`mbt` instances early-return every iteration — 30 GDN layers × (mbt−1) no-op tasks per step,
dispatched and event-synchronised for nothing; (ii) each link in the chain is a separate
event round-trip through a scheduler warp rather than a register-resident loop iteration, so the
recurrent state round-trips to global memory `mbt` times per layer instead of once; (iii) it
consumes the same enum id and file set as Option 1 while doing strictly less per task. If the
in-kernel scan turns out to be register- or smem-bound at large `Q_LEN`, this ordering could
change — it is a cost model, not a proof.

**Option 4 — prefill outside the kernel in torch, hand the state in. NOT expressible as-is
under MODE_OFFLINE.** `init_kernel` unconditionally sets `step[i] = 0` for every request
(`persistent_kernel.cuh:151-153`), and the admission path always reads the prompt from token 0
(`:371-374`). There is no way for the host to seed `step = prompt_len` and have MPK go straight
to decode. Two escape hatches exist:
  * `MODE_ONLINE_NOTOKEN` — `prepare_next_batch` is a no-op that returns `true` on iteration 0
    and `false` afterwards (`persistent_kernel.cuh:691-702`), so the host owns `qo_indptr` /
    `paged_kv_*` entirely. This is the vLLM-compat mode; it also runs on a smaller smem budget
    (220 KB, `runtime_header.h:35-39`, vs the 207 KB base at `:55`) and gives up MPK's
    in-kernel batching.
  * Patch `MODE_OFFLINE`'s `prepare_next_batch` to honour a host-provided `step`. Small change,
    but it is runtime plumbing on the shared path — it would have to keep Qwen3-8B CI green.

**Recommendation:** Option 1 for v1 (one GDN task family serving both phases), keeping Option 2
in reserve if the chunked-scan kernel turns out to need a materially different tiling for
`Q_LEN ≥ 64` than for `Q_LEN = 1`. Note the decision is *not* free: whichever option is chosen,
the **recurrent-state lifecycle** (zero at admission, persist across decode, release at
completion) still has to be added — `prepare_next_batch` owns exactly this for KV pages today
(`:280-290` free, `:346-350` allocate) and knows nothing about a per-request state tensor. The
cheapest v1 form is a kernel-side `if (step == 0) zero_state()`, which needs no runtime change
because `runtime_config.step[]` is already injectable.

---

## 4. Answer (c) — free TaskType ids below `TASK_SM100_TASK_END`

**14 free ids**, not 2.

Enum file: `include/mirage/persistent_kernel/runtime_header.h`, `enum TaskType` at `:84-201`,
with `TASK_SM100_TASK_BEGIN = 230` (`:127`) and `TASK_SM100_TASK_END = 299` (`:190`).

Ids used in the open interval (230, 299): 231, 232, 233, 235, 236, 248, 249, 251–256, 257–278,
280–298 (54 values).

**Free ids: `234, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 250, 279`.**

Two structural facts qualify how freely they can be spent:

1. **The only mechanical range check in the SM100 window is the TMA one.** `runtime.cc:1170-1171`
   emits `if (task_type > TASK_SM100_TMA_START_TASK && task_type < TASK_SM100_TMA_END_TASK)
   { create_tma_desc_by_task(task_desc); }` — i.e. ids **232–255 get TMA descriptors built
   automatically**. Of the free ids, **13 fall inside that window** (234, 237–247, 250) and
   only **279 falls outside it**. Kernels that need TMA but sit outside the window are handled
   by explicit id lists (`runtime.cc:1174-1187` for MLA and FP8 linear), so either choice works
   — it just has to be a deliberate one.
2. **`TASK_SM100_TASK_BEGIN` / `TASK_SM100_TASK_END` are pure documentation.** Repo-wide,
   the only references are the enum itself and a stale entry in
   `python/mirage/mpk/profiler_persistent.py:54, 99`. No code range-checks them, so 299 is not
   a hard wall — but 300 is `TASK_MULTIGPU_TASK_BEGIN` (`runtime_header.h:197`), so growing past
   299 would collide with the multi-GPU family's documented range.

**Budget verdict:** a Gated-DeltaNet bring-up needs 3–4 ids (conv1d state update; gated
delta-rule state update; query read-out + gated output norm; optionally a separate prefill
variant) plus 1 for the attention output gate if it is not folded into the fused QKVG. Fourteen
free ids is comfortable — **this is not a constraint on the design**, contrary to what the
scouting report concluded.

Two ancillary requirements when adding an id, both verified:

* `src/kernel/runtime.cc` **must** be edited. `task_type_to_name` (`:1756-1868`) supplies the
  symbol emitted into `_execute_task`'s dispatch, guarded by
  `assert(task_type_to_name.find(task.first) != task_type_to_name.end());` at `:1878`. A missing
  entry aborts codegen. The DFlash commit `2c87a75` duly touched `runtime.cc` (+5 lines).
  **The `add-mpk-task` skill's 7-file list omits this — it is 8 files.**
* `python/mirage/mpk/profiler_persistent.py`'s `TASK_TYPE` map is *not* required for
  correctness, but it is already stale: it is missing 231/232/233/235/236 and 295–298, and its
  entry `298: "TASK_SM100_TASK_END"` (`:99`) is simply wrong — 298 is
  `TASK_DFLASH_KV_STORE_SM100` and `TASK_SM100_TASK_END` is 299. Any profiler-driven
  optimisation in M3/M4 will mislabel tasks until this is fixed.

---

## 5. Answer (d) — does `5715c6f` cherry-pick cleanly onto `qwen3-5_support`?

**Yes. Clean, zero conflicts.**

The commit is already in the local object store — no `upstream` remote was needed:

```
$ git -C /home/catalyst/project log --all --oneline | grep -i smem
5715c6f Fix SM100 attention smem overflow blocking GQA >= 8:1 (#702) (#739)
$ git branch -a --contains 5715c6f
  remotes/origin/pr-dsv3-v1
  remotes/origin/pr-skills
```

Full sha `5715c6f2a6cce5d0d18da4e6776332b6ad04d7e4`, Yizheng Jiao, 2026-07-12. It touches exactly
one file: `include/mirage/persistent_kernel/tasks/blackwell/attention_sm100.cuh`, +87 / −69.

Test (run in a throwaway detached worktree; the pick was **not** committed to the real branch):

```
$ git -C ~/project worktree add --detach $SCRATCH/pick-test qwen3-5_support
Preparing worktree (detached HEAD 2c87a75)
$ git -C $SCRATCH/pick-test cherry-pick 5715c6f2a6cce5d0d18da4e6776332b6ad04d7e4
[detached HEAD 6f0a8c3] Fix SM100 attention smem overflow blocking GQA >= 8:1 (#702) (#739)
 1 file changed, 87 insertions(+), 69 deletions(-)
$ git -C $SCRATCH/pick-test diff --name-only --diff-filter=U     # conflicts
(none)
$ git -C ~/project worktree remove --force $SCRATCH/pick-test
```

Conflict-free is unsurprising: `2c87a75` (DFlash) added three *new* `.cuh` files and never
touched `attention_sm100.cuh`.

### What the pick buys, quantified

The change is one line of arithmetic plus a chunked epilogue: `S_O_BUFFER_SIZE` goes from
`sizeof(float) * MMA_ITERS_M * NUM_THREADS * 64` to `sizeof(float) * NUM_THREADS * 64`
(post-pick `attention_sm100.cuh:186-192`), making total smem independent of
`MAX_TOKENS × NUM_QO_PER_KV`. Everything else in the arena is byte-identical.

Constants: `NUM_THREADS = 128` (`tasks/common/worker_config.h:18`), `KV_TILE_SIZE = 64`
(`attention_sm100.cuh:84`), `MMA_ITERS_M = ceil(MAX_TOKENS·NUM_QO_PER_KV / 16)` (`:86`),
budget `MAX_DYNAMIC_SHARED_MEMORY_SIZE = 207 KiB − 6 KiB = 201 KiB`
(`runtime_header.h:55` with `:30`), enforced by
`static_assert(S_TOTAL_OFFSET <= MAX_DYNAMIC_SHARED_MEMORY_SIZE)` (`:189`).

Re-deriving the arena for Qwen3.5's full attention (`HEAD_DIM = 256`, `NUM_QO_PER_KV = 16/2 = 8`,
bf16):

| `max_num_batched_tokens` | `MMA_ITERS_M` | before | fits 201 KiB? | after | fits? |
|---|---|---|---|---|---|
| 1 | 1 | 170.0 KiB | yes | 170.0 KiB | yes |
| 2 | 1 | 178.0 | yes | 178.0 | yes |
| 4 | 2 | 228.0 | **no** | 196.0 | **yes** |
| 8 | 4 | 328.0 | no | 232.0 | no |
| 16 | 8 | 528.0 | no | 304.0 | no |

The same model reproduces the commit message's own figures exactly for Qwen3-32B GQA 8:1
(`HEAD_DIM=128`, `MAX_TOKENS=8`): **232 KB before → 136 KB after**, which is the number the
author quotes. That is the validation that the arithmetic above is right.

**Conclusion: the pick is necessary but not sufficient.** It lifts Qwen3.5's full-attention
ceiling from `mbt ≤ 2` to `mbt ≤ 4`. It does **not** reach `mbt = 16`, because at
`HEAD_DIM = 256` the four KV tiles alone cost `4 × 2 × 64 × 256 = 128 KiB` and `S_Q + S_O` grow
linearly with `mbt`. Getting to batch 16 needs the second lever (§6, Gap 3): decouple the
attention kernel's `MAX_TOKENS` from `mbt`. Today `max_tokens = input_ops[0]->dtensor.dim[0]`
(`task_register.cc:2052`, and identically at `:384, :1097, :3468, :3543, :4000`), i.e. it is
just the activation tensor's leading dim, even though in pure decode each attention task only
ever handles one token of one request.

These are still paper numbers derived from the source. Confirm them the way `5715c6f` did —
a standalone `nvcc -arch=sm_100a` instantiation at `NUM_QO_PER_KV=8, HEAD_DIM=256` across
`MAX_TOKENS ∈ {1,2,4,8,16}` — before committing to a batching design.

---

## 6. Answer (e) — the correctness methodology M2 must keep green

### 6.1 What CI actually asserts

`.github/workflows/ci-tests-qwen3.yml` is the **only** workflow with a numeric correctness gate.
Runner `[self-hosted, gpu, b200]` (`:14`), 60-minute job (`:15`), triggered on push to `mpk` and
on **every pull request** (`:2-6`). Environment: `CUBLAS_WORKSPACE_CONFIG=":16:8"`,
`HF_HOME=/raid/catalyst/models` (`:20-23`), GPU chosen by
`scripts/find_idle_gpus.sh --num-gpus 1` (`:44`).

The gating step (`:47-53`, 10-minute timeout) is `bash tests/ci-tests/run_ci_tests_qwen3.sh`,
which is four commands (`run_ci_tests_qwen3.sh:9-18`):

```bash
python demo/qwen3/demo.py --save-tokens                 # torch baseline  → outputs/qwen3/torch_output.json
python demo/qwen3/demo.py --use-mirage --save-tokens    # MPK            → outputs/qwen3/mpk_output.json
pytest -q tests/ci-tests/test_inference_output.py       # THE gate
python tests/ci-tests/perf_comparison.py                # informational, never fails
```

**The threshold is exact integer equality on token ids**, not a tolerance:
`tests/ci-tests/test_inference_output.py:8` `NUM_TOKENS_TO_COMPARE = 50`; `:33`
`if torch_slice != mpk_slice: pytest.fail(...)`. Any single differing id fails the build.

The second CI step (`:55-79`, 30-minute timeout) sweeps
`python tests/ci-tests/run_batch_perf.py --max-num-batched-requests $r --ignore-eos` for
`r ∈ {1,2,4,8}` and prints a table. **It asserts nothing.**

### 6.2 Which model CI actually runs, and against which reference

* **Model: `Qwen/Qwen3-8B`** — `demo/qwen3/demo.py:109` (`--model` default) and
  `tests/ci-tests/run_batch_perf.py:45`. No other model is exercised by any workflow.
* **Config: mbt 8, mbr 1, page_size 4096, max_num_pages 16, max_seq_length 512, greedy**
  (`demo.py:72-76, 96, 122-124`), offline mode.
* **Prompt:** `"Give me a short introduction to large language model."` run through the chat
  template (`demo.py:137, 230-238`).
* **The reference is NOT HuggingFace `transformers`.** `demo/qwen3/demo.py:1` imports
  `from models.modeling_qwen3 import Qwen3ForCausalLM` — the repo's own torch model in
  `demo/qwen3/models/modeling_qwen3.py`. `transformers` supplies only `AutoTokenizer,
  AutoConfig` (`:2`). So CI is "MPK vs mirage's own eager torch model", self-consistent rather
  than externally anchored.
* **The graph is built inline, not through the model registry.** `demo.py` calls
  `mi.PersistentKernel(...)` directly at `:307` and never touches `MPK`/`MPKMetadata`/
  `get_builder`. Consequently **`python/mirage/mpk/models/qwen3/builder.py` is not covered by
  CI at all** — the CI-protected surface is `demo/qwen3/demo.py` plus everything under it
  (task registration, codegen, the SM100 kernels, the runtime).
* Layers CI exercises: `embed`, `rmsnorm`, `rmsnorm_linear`, `linear`, `linear_with_residual`,
  `splitk_linear`, `silu_mul`, `paged_attention`, `paged_attention_split_kv` + merge,
  `argmax_partial`, `argmax_reduce`, `allreduce`.

**Practical consequence for M2:** "don't break existing MPK models" means *do not change the
behaviour of the shared task/codegen/runtime layer for the Qwen3-8B shapes*. Any edit to
`attention_sm100.cuh`, `task_register.cc`, `graph.cc`, `runtime.cc`, or
`persistent_kernel.py`'s existing methods is inside the blast radius; a brand-new task file plus
a new builder is not.

### 6.3 The other workflows

| Workflow | Gate | Relevance |
|---|---|---|
| `code-format.yml` | clang-format-15 over `src`, `include`, `tests`, `python`; runs on every PR (`:2-6, 11-26`) | **Every new `.cuh`/`.cc` must be clang-format-15 clean.** Note `persistent_kernel.cuh:25-36` is wrapped in `// clang-format off` because "ORDER IS LOAD-BEARING" — do not remove that guard. |
| `build-test.yml` | ubuntu-22.04, no GPU, asserts only `import mirage` succeeds (`:53`); 600-minute timeout | cheap smoke test |
| `gpu-tests.yml` | **disabled** — header states the gpu-nvidia runner "is no longer functional as of April 2026"; also only triggers on `main` | ignore |
| `shell-check.yml`, `pypi-deploy.yml`, `release-wheels.yml` | not correctness gates | — |

Nothing in CI covers DeepSeek-V3, any FP8 path, any multi-GPU path, any
`tests/runtime_python/blackwell/**` kernel test, or the `tests/runtime_python/test_mode/**`
suite. Those are developer-run only.

### 6.4 Test-mode: the harness M2 will actually use per layer

`params["test_mode"] = True` → `-DMPK_TEST_MODE` → the task graph compiles and runs exactly
once. The mechanism is one guard at `persistent_kernel.cuh:271-278`: under
`MPK_ENABLE_PROFILING` or `MPK_TEST_MODE` the retire condition becomes `if (true)`, so every
request is retired after one iteration and `prepare_next_batch` returns false on its second
call. Only `mode="offline"` is supported.

Execution order (this is why meta-tensor-dependent layers are testable):

```
init_kernel                zero step / request_ids / qo_indptr / paged_kv_indptr; seed page_queue
1st END_OF_TASK_GRAPH      prepare_next_batch fills meta tensors for iter 0 → true
iter 0                     the layer under test runs with valid meta tensors
2nd END_OF_TASK_GRAPH      always-finalize → false → terminate
```

Ten meta tensors are auto-allocated if not supplied; override only what the scenario needs
(`.claude/skills/test-mode/SKILL.md:148-188`).

**The de-facto thresholds in the existing suite** (there is no golden-output file anywhere in
the repo):

| Level | Reference | Criterion |
|---|---|---|
| bf16 kernel unit test | hand-written torch eager | `torch.testing.assert_close(rtol=1e-2, atol=1e-2)` |
| FP8 quantize | torch eager | scales **bit-exact** (`rtol=0, atol=0`), values `rtol=1e-1, atol=16.0` — `blackwell/sm100_quantize_fp8/test_quantize_fp8.py:55-60` |
| FP8 dense linear | torch eager | `rtol=1e-2, atol=1e-2` — `blackwell/sm100_linear_fp8/test_runtime_quantize_linear_fp8.py:89` |
| FP8 MoE pipeline (test-mode) | torch eager | `max_rel < 0.1`, `sys.exit(1)` on fail — `test_mode/test_fp8_moe_pipeline_testmode.py:177, 244` |
| MLA attention | **FlashInfer** | `assert_close(rtol=1e-2, atol=1e-2)` — `blackwell/sm100_mla/test_mla_attention.py` |
| bf16 layer/pipeline test-mode | torch eager | hand-picked max-abs bounds scaled with GEMM depth: rmsnorm < 0.05, fork/join < 0.1, GEMM branch < 0.5, 2-GEMM diamond < 1.0, Qwen3 MLP < 1.0 |
| end-to-end (CI) | mirage's own torch model | **exact 50-token id match** |

Two things to know before writing tests:

* `tests/runtime_python/test_mode/*.py` are **`sys.exit(1)` scripts, not pytest** — there is no
  `conftest.py` and `pyproject.toml` declares only the `impure`/`slow` markers.
* **`pytorch_reference.py` does not exist anywhere.** `find tests -name pytorch_reference.py`
  returns nothing, yet both `add-mpk-task/SKILL.md:381` and `test-mode/SKILL.md:26-30` mandate
  it. The convention is aspirational; every in-tree test inlines its reference. M2 should
  either establish the convention for the new folders (preferred — it is a good convention and
  the user's dev-skill-maintenance rule points that way) or correct the two skills.

### 6.5 What AC-3 requires beyond this

**There is exactly one gate, and it is exact equality.** Quoting `.pm/goal.md` AC-3 verbatim:

> **AC-3 (correctness):** on the pinned prompt set `.pm/eval/prompts.jsonl` (10 prompts), greedy
> decode of 64 new tokens per prompt: mpk token ids must exactly match the HuggingFace
> `transformers` reference running the same FP8 checkpoint. A mismatch is a failure unless
> root-caused with per-position logit evidence as a numeric-precision tie flip (not an
> implementation error) and documented in the run report.

Read that as one rule, not two. **The pass/fail criterion the harness computes is
`mpk_token_ids == reference_token_ids`, with no tolerance anywhere in it.** The second sentence
is *not* a relaxed threshold and must never be implemented as one: it is a **documented,
per-position, human-adjudicated exception** to a gate that has already failed. A mismatch is
failing until someone writes the root-cause into the run report; an automated "close enough"
check would convert a hard gate into a soft one and is exactly the kind of target movement the
goal forbids.

**Mechanically, the M2 harness must therefore produce, for every one of the 10 × 64 positions:**

1. the reference's **top-2 logits and their identities** at that position, and the **margin**
   `logit[top1] − logit[top2]`;
2. **MPK's argmax identity** at the same position (and, when available, MPK's logit for both the
   reference top-1 and top-2 ids);
3. the boolean `mpk_argmax == reference_argmax`.

The gate is `all(3)`. Nothing else gates. Items 1–2 exist *only* so that a failure can be
adjudicated: a mismatch is a candidate tie-flip when the reference margin at that position is
within the run's observed FP8 noise floor **and** MPK's ranking of exactly those two ids is
inverted while everything upstream matches — and even then the waiver is a written entry in the
run report naming the position, the margin, and the mechanism, not a flag the harness sets.
A mismatch at a position with a wide margin, or one that propagates from an earlier divergence,
is an implementation bug by definition.

Two design consequences worth fixing now rather than after the first red run:

* Emit items 1–3 on **every** run, not just failing ones. A margin distribution measured while
  passing is what makes a later tie-flip claim credible; measured only after a failure, it is
  indistinguishable from motivated reasoning.
* Because greedy decode feeds its own output back in, the **first** divergent position is the
  only one that carries information — everything after it is a different conditioning sequence.
  The harness should report the first divergence per prompt explicitly and stop treating
  subsequent positions as independent evidence.

The reference itself does not exist in the repo — the closest
precedent is `demo/deepseek_v3/demo.py:895-901`, which does call
`AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)` but only as a *generation*
branch, never for comparison, and no CI consumes it. M2 therefore needs a **new** harness. The
methodology to copy is the one documented in the DFlash commit `2c87a75`:

1. Write the design spec first.
2. Build `ref_dump.py`: load the HF reference implementation with real weights and dump
   per-op / per-layer tensors as the numerical oracle.
3. Bring each kernel up standalone against that oracle (DFlash reported `relmax < 0.003`).
4. Re-validate in the real megakernel via test-mode (DFlash: `relmax 0.0011` attention core,
   `0.0037` full attention path vs `out::layers.0.self_attn`).
5. Check end-to-end against an external engine (DFlash: "draft tokens 7/7 == vLLM").

That commit's scaffolding was deliberately moved out of tree before merge — only shipped
kernels landed. Follow that for reviewability.

---

## 7. Gap list for Qwen3.5

Ordered by risk × effort. "Blast radius" is judged against §6.2 — whether the change is inside
the CI-protected shared surface or additive.

### Gap 1 — Gated DeltaNet linear attention: absent, zero code

*Missing:* causal depthwise conv1d (k=4) with a per-request conv state; the gated delta-rule
recurrence `S ← S·diag(a) + (v − S·k)·kᵀ` over a `[32 v-heads, 128, 128]` state; the query
read-out `o = S·q`; `RMSNormGated(128)` (norm × silu-gate, `norm_before_gate=True`); and the
whole **recurrent-state lifecycle** (reset on admission, persist across decode, release on
completion), which `prepare_next_batch` provides for KV pages only.

*Closest existing pattern:* structurally, `2c87a75` (DFlash) — three new SM100 attention-family
tasks brought up together, 12 files, +896 lines. For the per-request state addressing, the MLA
prefill registration's `bi_ * MPK_MAX_SEQ_LENGTH * D` slicing (`task_register.cc:3768-3772`).
For prefill/decode coexistence, the dual-dispatch gate (§3.2, Option 2). Nothing in the
paged-attention family can be re-parameterised into this — it is softmax flash attention over a
page table.

*SOP:* full `add-mpk-task` sequence, once per kernel: enum → `.cuh` → `task_header.cuh` →
`task_register.h` → `task_register.cc` → `graph.cc` → `persistent_kernel.py` → **plus
`runtime.cc`'s `task_type_to_name`** (§4). Then Steps A–C (pybind11 wrapper in
`runtime_kernel_wrapper.cu`, arch-specific `setup.py`, torch-reference comparison), Step 8
(test-mode), Step 9 (benchmark), in `tests/runtime_python/blackwell/sm100_<layer>/`.

*Blast radius:* **large but additive.** 3–4 new task ids (of 14 free), 3–4 new `.cuh` files,
~4 new layer methods. The only shared-surface edits are the enum, the three dispatch tables,
and `persistent_kernel.py` additions — all append-only. The one genuinely invasive piece is the
state lifecycle if it goes into `prepare_next_batch`; prefer the kernel-side
`if (step == 0) zero_state()` form, which touches nothing shared.

*Sizing input:* state per request = 32 × 128 × 128 × 4 B (fp32) = 2 MiB per layer, × 30 layers
= **60 MiB per request**; ~960 MiB at batch 16. Plus conv state `[·, 3, 8192]` per layer.
This is real memory planning, not a rounding error.

### Gap 2 — Attention output gate (`attn_output_gate=true`)

*Missing:* `out * sigmoid(gate)` before `o_proj`. Zero hits for `output_gate` / `attn_gate`.

*Closest existing pattern:* the fused QK-norm + RoPE epilogue already inside
`attention_sm100.cuh`. But `register_paged_attention_sm100_task` is at `num_inputs = 7`
(`task_register.cc:2039`) — exactly `MAX_INPUTS_PER_TASK` (`runtime_header.h:79`), so **a gate
input cannot be added**. It does not need to be: **the gate already rides inside the q
projection in the checkpoint**, so slicing it costs zero extra inputs and zero extra tasks.

This is verified on vLLM source by the sibling M1-I1 doc, `docs/qwen35/vllm-graph.md:409-441`
(cross-check those vLLM `file:line` refs there rather than re-deriving):

* `attn_output_gate = getattr(config, "attn_output_gate", True)` → `True`
  (`qwen3_next.py:265`), and `qkv_proj` is constructed with
  `total_num_heads = 16 * (1 + attn_output_gate) = 32` (`qwen3_next.py:267-275`) — i.e. the gate
  is allocated as extra Q head-slots, not a separate projection. Weight is
  `[32·256 + 2·256 + 2·256, 2048] = [9216, 2048]`, splitting `[q_gate: 8192 | k: 512 | v: 512]`
  (`qwen3_next.py:345-347, 368-370`).
* The 8192-wide block is **per-head interleaved**, not q-block-then-gate-block:
  `q_gate.view(T, 16, 512)` then `torch.chunk(q_gate, 2, dim=-1)`
  (`qwen3_next.py:372-373`), i.e. `[h0_q(256) | h0_gate(256) | h1_q(256) | h1_gate(256) | …]`.
  Independently confirmed by the fused kernel's addressing —
  `in_base = q_gate_ptr + token*stride + local_head*2*head_dim` (`fused_qk_norm_rope.py:54`),
  `gate_in_base = in_base + head_dim` (`:111`), and the launcher docstring
  (`:133`: *"q_gate: (n_tokens, num_q_heads * 2 * head_dim) -- per head: [q|gate]"*).
* In the checkpoint this lands as `q_proj.weight [8192, 2048]` F8_E4M3 = "32 head-slots of 256 =
  16 × `[q(256)|gate(256)]`" (`docs/qwen35/vllm-graph.md:1475`).
* vLLM applies the gate **outside** the attention kernel:
  `attn_output = attn_output * torch.sigmoid(gate)` after `self.attn(...)`
  (`qwen3_next.py:397-398`) — a full sigmoid, not SiLU, and not an attention sink
  (`sinks` is `None`).

Two consequences for MPK. (i) The per-head interleave means MPK's existing packed-QKV
convention needs the *same* interleave when the shard loader builds the fused tensor —
`attention_sm100.cuh` reads Q as `[num_tokens, head_dim * (num_qo_heads + 2*num_kv_heads)]`
(`:76-78`), so the q stride per head becomes `2*head_dim` and the gate sits at `+head_dim`.
(ii) Because vLLM applies the gate after attention, a standalone elementwise gate task is a
legitimate fallback that avoids touching `attention_sm100.cuh` at all — at the cost of one enum
id and an extra pass over `[B, 4096]` per full-attention layer.

*SOP:* kernel-only change plus a `params[]` flag; steps 1–2, 5, 7 of `add-mpk-task`
(no new enum id if it is a flag on the existing task; one new id if a standalone elementwise
task is preferred).

*Blast radius:* **medium-high for the fused route — it edits `attention_sm100.cuh`, which
Qwen3-8B CI depends on.** Gate it behind a `params[]` flag defaulting to off so the existing
variant's generated code string is byte-identical (variant dedup in `register_task_variant()`
keys on that string). The standalone-gate-task fallback above is **low** blast radius (additive
only) and mirrors what vLLM does; prefer it if the fused route destabilises CI.

### Gap 3 — `head_dim = 256` at GQA 8:1 blows the smem budget past `mbt = 4`

*Missing:* per §5, `5715c6f` buys `mbt ≤ 4`; AC-4 needs batch 16. The second lever is to stop
deriving the attention kernel's `MAX_TOKENS` from the activation tensor's leading dim
(`task_register.cc:2052`) when each decode task only ever handles one token of one request
(`prepare_next_batch` gives each decoding slot `num_new_tokens = min(1, …)`,
`persistent_kernel.cuh:326`), i.e. smem is over-provisioned by up to 16×.

*Closest existing pattern:* the plumbing is half-built — `Q_LEN_OVERRIDE` and `TAIL_OFFSET`
are already template parameters (`attention_sm100.cuh:43-44`) exposed as optional
`params[6]/params[7]` (`persistent_kernel.py:946-951`), added for Eagle3. A
`max_tokens_per_request` parameter is the same shape of change.

Cheaper partial levers: halve `KV_TILE_SIZE` 64 → 32 for large `head_dim` (saves 64 KiB, must
preserve `static_assert(PAGE_SIZE % KV_TILE_SIZE == 0)` at `attention_sm100.cuh:227`); or
`paged_attention_split_kv_layer` (reduces iteration count, **not** per-task S_Q/S_O footprint,
so it does not fix this on its own).

*SOP:* steps 1 (no new enum), 2, 5, 7 — plus a standalone `nvcc -arch=sm_100a` instantiation
check as the acceptance evidence, exactly as `5715c6f` did.

*Blast radius:* **high.** `MAX_TOKENS` derivation is shared by six registration sites
(`task_register.cc:384, 1097, 2052, 3468, 3543, 4000`). Any change must keep the Qwen3-8B
generated code identical for the default path.

### Gap 4 — Partial RoPE (`rotary_dim = 64` of `head_dim = 256`)

*Missing:* RoPE is fused into attention and `paged_attention_layer` asserts
`cos_pos_embed.dim(1) == head_dim` (`persistent_kernel.py:928-929`). More importantly the
rotation itself is **NeoX half-split over the full head_dim**: `rotary_embedding_sm100.cuh:61-71`
pairs column `i` with `i ± HEAD_DIM/2`, i.e. (0,128), (1,129), … Qwen3.5 needs (0,32), (1,33),
… over the first 64 columns only. Simply padding cos/sin with 1/0 does **not** work — the
pairing partner is wrong.

*Two routes:*

* **Kernel change (obvious):** add a `ROTARY_DIM` template parameter and pair `i` with
  `i ± ROTARY_DIM/2` for `i < ROTARY_DIM`. Precedent: DeepSeek-V3 does partial RoPE by keeping
  rope and nope as *separate tensors* (`models/deepseek_v3/builder.py:210-218`), which does not
  transfer to a packed 256-wide head.
* **Zero-kernel-change alternative (worth testing first):** permute the head_dim columns of
  `q_proj`, `k_proj`, `q_norm` and `k_norm` at load time so that Qwen's rotated pair `(j, j+32)`
  lands at MPK's `(j, j+128)`, and give cos = 1, sin = 0 at every non-rotated column. This is
  sound in principle — `q·k` is invariant under a permutation applied identically to q and k,
  and RMSNorm over head_dim is permutation-equivariant when its weight is permuted too — and it
  moves the whole problem into the weight loader. **Unverified; prove it numerically against the
  HF oracle before relying on it.**

*SOP:* the permutation route is builder/loader-only (no `add-mpk-task` steps). The kernel route
is steps 2, 5, 7.

*Blast radius:* permutation route ≈ zero (new builder only). Kernel route touches
`rotary_embedding_sm100.cuh` + `attention_sm100.cuh` + the assert, i.e. CI-protected.

### Gap 5 — Gemma-style `(1 + w)` RMSNorm

*Missing:* `zero_centered` — zero hits. The shipped kernels compute `x·rsqrt(mean(x²)+eps)·w`.
Qwen3.5 uses `(1+w)` everywhere except the GDN gated norm.

*Closest existing pattern:* none needed — **fold it into the weights at load time**
(`w_effective = 1.0 + w`), which is exact and costs nothing. The GDN gated norm is a new kernel
anyway (Gap 1), so it can implement its own convention.

*SOP:* builder only.

*Blast radius:* **zero.** Note eps differs across in-tree kernels (rmsnorm 1e-6, DFlash norm
1e-5) — pin eps from the HF config, never inherit a default.

### Gap 6 — `padded_vocab_size` hardcoded to 153600

*Missing:* `python/mirage/mpk/models/qwen3/builder.py:39` and `:76`:
`self.padded_vocab_size = 153600 #TODO: A better way to decide?`, used to size `argmax_in`
(`:156`, `:234`) and to zero-pad `lm_head` (`:528-534`). Qwen3.5's vocab is **248320 > 153600**,
so a Qwen3.5 builder must compute its own padding. The same constant is hardcoded in
`tests/ci-tests/run_batch_perf.py:100`.

*Closest existing pattern:* pick a padding that satisfies `grid_for_rmsnorm_linear_layer()`'s
divisibility branch. 248320 = 3880 × 64 is already a multiple of 256 (248320 / 256 = 970), so no
padding is strictly needed.

*SOP:* builder only.

*Blast radius:* **zero if a new builder is written** (the Qwen3 builder is not on the CI path,
§6.2). Do not "fix" the constant in place — that would change Qwen3-8B's argmax tensor shape.

### Gap 7 — MoE at `moe_intermediate = 512` (untested small)

*Missing:* nothing structural. 256 experts / top-8 is DeepSeek-V3's exact configuration;
`norm_topk_prob=True` is already the behaviour — `register_moe_topk_softmax_sm100_task` passes
`renormalize = true` unconditionally (`task_register.cc:2467`), and the kernel's
`static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS))` (`topk_softmax_sm100.cuh:106`) is
satisfied by 256. The routing kernel requires **8 warps → `block_dim = (256,1,1)`**
(`topk_softmax_sm100.cuh:49, 58`), unlike most MoE layers which use `(128,1,1)`.

*What is uncertain:* `moe_intermediate = 512` is 4× smaller than DeepSeek-V3's 2048.
`_moe_fp8_m_split(1024, …)` allows at most `grid_dim.y = 8` for w13 and 16 for w2, and w2's
reduction of 512 gives only 4 k-tiles against a pipeline whose depth was raised to
`num_ab_stages = 8` precisely because 4 stages **hung** at `fp8_k_tile_count = 8`
(`task_register.cc:2812-2818`). Our regime sits below the count that motivated the current
setting and has never been run. Expect grid/split re-derivation and poor efficiency until tuned
— and treat "builder only" as the *expected*, not guaranteed, scope (§2.3).

*Closest existing pattern:* `demo/qwen3/demo_30B_A3B.py:613-660` chains exactly the pipeline we
need in bf16 (`routing → w13 → silu → w2 → mul_sum_add`); `models/deepseek_v3/builder.py:832-1153`
is the FP8 + shared-expert version.

*SOP:* builder only, plus a `bench_fp8_moe_gemm.py`-style benchmark at our shapes.

*Blast radius:* **zero if it stays builder-only** (new builder; tuning knobs are all `grid_dim`
arguments). If the 4-k-tile regime needs a different `num_ab_stages`, that constant lives in
`task_register.cc:2818` on the shared FP8 MoE registration and would become a
DeepSeek-V3-affecting change — measure before touching it.

### Gap 8 — Shared expert with a sigmoid gate

*Missing:* DeepSeek-V3's shared expert has no gate; Qwen3.5 scales it by
`sigmoid(x @ shared_expert_gate.Wᵀ)` with `shared_expert_gate` a `[1, 2048]` unquantized GEMV.

*Closest existing pattern:* `models/deepseek_v3/builder.py:1064-1153` builds the shared expert as
a plain dense MLP and feeds the result as the `residual` argument of `moe_mul_sum_add_layer`
(`persistent_kernel.py:1803-1834`, which asserts `residual.num_dims == 2`), giving
`final = Σ(routed × weights) + (residual + shared_out)`. The gate multiply has to happen before
that. `linear_layer` at N=1 is degenerate; the cheapest v1 form is a small new elementwise task
(or reuse `silu_mul`-style machinery) — or fold the gate GEMV into the shared-expert down
projection's epilogue.

*SOP:* builder, plus possibly one small `add-mpk-task` for the gated multiply.

*Blast radius:* **low.**

### Gap 9 — SM100 gaps in the "boring" ops

`embed_layer` always registers `"embedding"` (no SM100 variant); `rmsnorm` on cc≥90 falls back to
the Hopper kernel; there is no SM100 fused RMSNorm+Linear and no SM100 `silu_mul` file. These
work (they compile for sm_100a) but are not Blackwell-tuned. Also `rmsnorm` is
**one token per task** — `static_assert(BATCH_SIZE == 1)` (`tasks/ampere/rmsnorm.cuh:24`),
enforced host-side at `task_register.cc:110-112` — so at batch 16 the graph issues 16× the
rmsnorm tasks per layer with no dead-row masking. Park these for M3.

*Blast radius:* zero for M2 (nothing to change); a batched rmsnorm task in M3 would be additive.

---

## 8. Risks, in priority order

1. **Gated DeltaNet is the whole project's critical path.** 30 of 40 layers, zero existing code,
   3–4 new kernels plus a state lifecycle the runtime has never had. Every downstream milestone
   is blocked on it. *Mitigation:* build the torch reference + `ref_dump.py` oracle **before any
   CUDA** (the DFlash methodology), and keep the state addressing on the proven
   `base + request_id * stride` pattern from `task_register.cc:3768`. Watch the `8b19538` class
   of bug: `blockIdx` is the worker id, not the data row — with a per-request recurrent state,
   row identity is load-bearing in a way it has never been before in this repo.

2. **FP8 numerics vs the AC-3 exact-token-match gate.** MPK's two FP8 paths diverge from HF's
   block dequant in *different* ways (§2.2.1): the **dense** path discards the checkpoint's
   128×128 block scales and re-quantizes to per-row power-of-two UE8M0
   (`builder.py:475-542`) — this is what **every dense projection in Qwen3.5** gets, including
   `qkv_proj`, `o_proj`, `in_proj_qkvz`, `out_proj` and both shared-expert GEMMs; the **MoE
   grouped** path preserves the block values and only replicates them
   (`builder.py:920-931`) — this is what **both routed-expert GEMMs** get. Neither is
   bit-identical to HF. The gate they face is exact token-id equality with **no tolerance**
   (§6.5); the "numeric-precision tie flip" clause is a documented per-position exception to a
   failed gate, not a threshold the harness may apply. *Mitigation:* build the per-position
   top-2 / margin / argmax instrumentation of §6.5 into the harness from day one and emit it on
   passing runs too, so a later tie-flip claim rests on a measured margin distribution; and
   settle early whether the reference should mirror the dense path's requantization (which would
   turn a per-GEMM delta into a known, quantified quantity rather than a surprise at position 37
   of prompt 6).

3. **Batch 16 at `head_dim = 256` is not reachable by cherry-picking alone.** §5 shows
   `5715c6f` lifts the ceiling only to `mbt ≤ 4`; AC-4 requires batch sizes up to 16. The fix
   (decoupling attention `MAX_TOKENS` from `mbt`) touches six registration sites on the
   CI-protected shared surface. *Mitigation:* do the standalone `nvcc -arch=sm_100a`
   instantiation sweep in M1/early M2 so the batching design is settled before the builder is
   written, rather than discovering the ceiling during M4 benchmarking.

4. **`prepare_next_batch` is serial and already the batch-8 knee.** Commit `92603ca` recorded
   4.40 / 4.41 / 4.44 / 7.49 ms/token at batch 1/2/4/8 on Qwen3-8B — +69 % latency for +18 %
   throughput at 8, with `// TODO: parallelize this processing` sitting at
   `persistent_kernel.cuh:223`. Qwen3.5 adds 30 GDN layers whose state lifecycle may want to
   live in that same serial section. *Mitigation:* keep state reset kernel-side; treat the
   scheduler critical path as an M3/M4 optimisation target with the MPK profiler CSV, not NCU.

5. **Silent-corruption footguns with no assert.** (i) No check that `mbt ≥ mbr`: surplus
   requests get 0 tokens and stall forever (`persistent_kernel.cuh:326`). (ii) No check that
   `max_num_pages ≥ mbr × ceil(max_seq_length / page_size)`: the page FIFO wraps
   (`page_queue[head % MPK_MAX_NUM_PAGES]`) and hands out pages already owned by another request
   → silent KV corruption. Only `run_batch_perf.py:62-63` computes it correctly;
   `demo/qwen3/demo.py:75` defaults to 16 regardless of `mbr`. *Mitigation:* assert both in the
   new builder.

6. **Enum/dispatch bookkeeping is easy to half-do.** A new TaskType that is missing from
   `runtime.cc`'s `task_type_to_name` aborts codegen at `runtime.cc:1878`; one missing from
   `profiler_persistent.py` silently mislabels profiler output (that map is *already* stale and
   wrong at id 298). *Mitigation:* treat `add-mpk-task` as an 8-file recipe and fix the skill.

7. **Editing shared kernels breaks the only correctness gate there is.** Gaps 2, 3, and 4's
   kernel route all edit `attention_sm100.cuh`, which Qwen3-8B CI depends on for an exact
   50-token match. *Mitigation:* gate every behavioural change behind a `params[]` flag that
   defaults to the current behaviour, so `register_task_variant()`'s code-string dedup yields a
   byte-identical variant for existing callers; run the CI script locally before pushing.

8. **Build/iteration friction.** Megakernel JIT is 1–10+ minutes per `compile()`
   (`tasks/blackwell/` alone is 748 KB across 45 files, in one translation unit with all of
   CUTLASS); the GPU must be exclusive or the runtime deadlocks; and CUDA 12.8 is the safe
   target (off-branch `80aa187`: CUDA 13.2 has "ptxas/synclog incompatibilities with CUTLASS
   v4.2.1"). *Mitigation:* use `demo/qwen3/demo_kernel_reuse.py`'s `load_mpk` to cache the `.so`
   during iteration.

---

## 9. Corrections to prior sources

Recorded so the next reader does not re-derive them.

**Versus the scouting report (`design/scouting/mpk-internals.md`):**

1. **"Only 2 free ids (`234`, `279`) inside the current SM100 window — budget this."**
   *Wrong.* There are **14** (§4). The report missed the 237–247 block and 250. Its downstream
   conclusion that "`TASK_SM100_TASK_END` and the TMA range checks will have to be adjusted" for
   a GDN bring-up does not follow. (It also mis-attributes a range check to
   `TASK_SM100_TASK_BEGIN/END`; those two are pure documentation — the only SM100 range check is
   the TMA one at `runtime.cc:1170-1171`.)

2. **"Fusing [the output gate] is preferable but the task already uses 7 inputs … Adding a gate
   input is not possible without removing one — e.g. by dropping the separate cos/sin tensors."**
   The 7-input constraint is right; the suggested workaround is not needed. Qwen3.5's own weight
   layout already packs `[q | gate]` per head inside `q_proj`, so slicing it inside the kernel
   costs zero inputs (§7, Gap 2).

3. **"If Qwen3.5 applies RoPE to only part of head_dim 256, the assert at
   `persistent_kernel.py:928` must be relaxed and the kernel taught a `ROPE_DIM < HEAD_DIM`
   template parameter."** Correct as far as it goes, but it understates the problem and misses a
   cheaper option. The blocker is not the assert — it is that the rotation pairs column `i` with
   `i ± HEAD_DIM/2` (`rotary_embedding_sm100.cuh:61-71`), so identity-padding cos/sin gives the
   wrong partner. A load-time column permutation may avoid the kernel change entirely
   (§7, Gap 4).

4. The report's smem estimate table for `HEAD_DIM=256` (170 / 178 / 196 / 232 / 304 KiB at
   mbt 1/2/4/8/16 after the pick) is **exactly right** — independently re-derived from source
   in §5, and cross-validated against `5715c6f`'s own quoted 232 → 136 KB for Qwen3-32B.

5. The report's ~70 `*_layer` line numbers were spot-checked against
   `grep -n "    def .*_layer(" python/mirage/mpk/persistent_kernel.py` and are accurate.

**Versus the in-repo dev skills** (candidates for the maintenance commits that `constraint.md`
§2b calls for):

* `add-mpk-task/SKILL.md` describes a **7-file** recipe. It is **8** —
  `src/kernel/runtime.cc`'s `task_type_to_name` is mandatory
  (hard `assert` at `runtime.cc:1878`), and `python/mirage/mpk/profiler_persistent.py` is a
  ninth file if you want usable profiles.
* `add-mpk-task/SKILL.md:381` and `test-mode/SKILL.md:26-30` mandate a per-folder
  `pytorch_reference.py`. **No such file exists in the repo** (`find tests -name
  pytorch_reference.py` → empty). Either establish it or downgrade the wording.
* `test-mode/SKILL.md:341, 347-348` names `test_prepare_next_batch_testmode.py` and
  `test_multigpu_rmsnorm_testmode.py` as canonical examples. Neither exists.
* `add-mpk-model/SKILL.md:18-20` states model code lives in `demo/<model>/` and "NOT in
  `python/mirage/mpk/models/`". That is stale — the registry path
  (`models/<model>/builder.py` + `@register_model_builder`, resolved by `model_registry.py`,
  invoked from `MPK.build()` at `mpk.py:458-467`) is the newer, preferred path and is what
  DeepSeek-V3, EAGLE3 and DFlash use. Both paths currently coexist, and (see §6.2) it is the
  *older* inline demo path that CI actually exercises.
* `add-mpk-task/SKILL.md:228-236` says `block_dim` must be `(256,1,1)` on Blackwell. In practice
  `models/deepseek_v3/builder.py` uses `(128,1,1)` for 84 of its 91 layer calls and `(256,1,1)`
  for 6 (the top-k routing layers, which require 8 warps —
  `topk_softmax_sm100.cuh:49, 58`). The rule is really "match what the kernel expects";
  `NUM_THREADS` is 128 (`worker_config.h:18`) while `WORKER_NUM_THREADS` is 256 (`:27`).
