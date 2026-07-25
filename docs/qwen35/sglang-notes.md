# SGLang notes — Qwen3.5-35B-A3B decode, secondary reference

**Scope:** this is the SECONDARY reference for the megakernel port. The primary reference is
`workspace/docs/qwen35/vllm-graph.md` (owned by M1-I1, full decode dataflow + FP8 section). This
doc does **not** re-derive that dataflow — it covers only what SGLang does **differently** and
that matters for small-batch (bs 1..16) single-GPU B200 (SM100) decode of
`Qwen/Qwen3.5-35B-A3B-FP8`, plus SGLang's FP8 kernel handling specifically (a gap neither scout
report closed — see §0).

## 0. Sources & provenance

- **SGLang source**, all `file:line` cites below: clone at
  `/tmp/claude-1006/-home-catalyst-agent/d305d170-45e6-4ad6-b21b-823f6f637deb/scratchpad/sglang`,
  commit **`6a046fad09d4ea4b377c5d98149edd1e55d52d5d`** (branch `main`, clean tree) — re-verified
  to be the same commit `design/scouting/sglang-qwen35-graph.md` recorded; no re-clone was
  needed. Re-clone (`--depth 1`) to refresh if this doc goes stale. Shallow clone means
  `git blame`/`git log -- <file>` only shows this one commit — cannot date how old any given
  line is.
- **vLLM facts** cited here for contrast are reused from `design/scouting/vllm-qwen35-graph.md`
  (vLLM commit `0ba2aa35a81dcc3246b26291368b53fa2389c7d7` as recorded there) — I did **not**
  independently re-verify vLLM source; that verification belongs to M1-I1. Where I could not find
  an equivalent claim in that report, the table says so rather than guessing.
- **Checkpoint config**, used to resolve every config-dependent kernel choice below:
  `/tmp/claude-1006/-home-catalyst-agent/d305d170-45e6-4ad6-b21b-823f6f637deb/scratchpad/qwen35fp8-config.json`.
  This is the single most load-bearing input in this document — it resolves two of the sglang
  scout's three open items (see §5). Key fields: `quantization_config.quant_method="fp8"`,
  `.activation_scheme="dynamic"`, `.weight_block_size=[128,128]`, `.modules_to_not_convert=[...]`
  (lines 115–123); `text_config.mamba_ssm_dtype="float32"` (line 81);
  `text_config.rope_parameters.partial_rotary_factor=0.25` (line 91). If a fresher/different
  `config.json` is later pulled from the real HF repo, re-check every row below that cites this
  file.

---

## 1. Differences table — sglang vs vLLM vs a plain HF-config reading

All "why it matters" entries are specific to bs 1..16, single B200, no spec-decode (AC-2).

| # | What | SGLang (file:line, this commit) | vLLM (per sibling scout, not re-verified here) | Plain HF-config reading | Why it matters at bs 1..16 / SM100 |
|---|---|---|---|---|---|
| 1 | **GDN decode kernel: Triton, not FlashInfer, under default launch args** | `GDNKernelDispatcher.__init__` (`gdn_backend.py:112-144`) picks the decode kernel from the resolved `linear_attn_decode_backend`. **Traced chain** (§5 logs a correction to how this doc first described it): `server_args.mamba_ssm_dtype` is a CLI-only dataclass field, default `None` (`server_args.py:2408-2415`); grepping every assignment site in the tree, the *only* place it's ever set besides that default is the unrelated, opt-in `--enable-gdn-replayssm-spec` path (`server_args.py:5935-5941`, forces `"float32"`, off by default) — nothing populates it from the checkpoint. So at `_handle_linear_attn_backend` (`server_args.py:5752-5771`) the SM100 upgrade check `is_sm100_supported() and self.mamba_ssm_dtype == "bfloat16"` compares against `None` → **false** under default args; `decode = self.linear_attn_decode_backend or self.linear_attn_backend` (`:5771`) falls to base `"triton"` (`server_args.py:2475-2482`). `initialize_linear_attn_config` (`layers/attention/linear/utils.py:44-56`) then carries that resolved value into the process-global `GDNKernelDispatcher` reads via `get_linear_attn_decode_backend()` (`:60-67`). The checkpoint's declared `mamba_ssm_dtype="float32"` **is** consumed — but by a separate consumer, `mamba2_state_dtype(config)` (`configs/mamba_utils.py:47-94`, called from `configs/qwen3_next.py:312`), which sizes the recurrent-state **cache buffer** dtype, not the kernel choice. | Not traced by sibling scout at this depth; vLLM's own GDN decode kernel is a vendored FLA-derived kernel (`third_party/flash_linear_attention`), not FlashInfer-based at all (no equivalent auto-upgrade mechanism reported). | No such concept — config just says "linear attention"; nothing there implies multiple selectable decode-kernel families gated on a *CLI serving flag* that isn't even auto-populated from the weight checkpoint. | This **overturns the naive "SM100 → use the fast FlashInfer kernel" assumption** — under default launch args (no explicit `--mamba-ssm-dtype`), which is what any straightforward sglang deployment of this checkpoint uses. A megakernel benchmarking "sglang's GDN decode" must target the Triton `packed_decode` algorithm (in-kernel split+L2-norm+gate+recurrence, one launch) as the actual baseline, not a hypothetical FlashInfer path sglang won't take unless explicitly configured to. |
| 2 | **Causal-conv1d decode-step update always Triton on CUDA (never rebound)** | See §2 — full re-verification. `gdn_backend.py:6-9` imports Triton version; `gdn_backend.py:37-42`'s `is_cuda()` branch rebinds only `causal_conv1d_fn`, not `causal_conv1d_update`; call site `gdn_backend.py:393-399`. | vLLM's decode-step update kernel (`vllm/.../causal_conv1d.py:749-1060`, `tl.store` at line 933) is **also Triton** — this is *not* a sglang-vs-vLLM difference in isolation (see §5 correction). | N/A — conv1d is a fused depthwise op invisible at the HF module-list level. | The citations establish that both engines run a Triton kernel here — they do **not** establish that a separate Triton launch actually costs more than the unused CUDA-native wrapper would under graph-captured decode (CUDA-graph replay bakes launches into one graph invocation, which changes the usual "per-launch CPU dispatch overhead" argument). Treat "a separate conv-update launch matters at small batch" as an **unmeasured hypothesis** worth a micro-benchmark, not an established fact. What *is* established: SGLang ships an unused CUDA-native alternative for exactly this op (§2) — a low-risk candidate to A/B before assuming it matters. |
| 3 | **Full-attention (GQA) decode: Triton, not the generic SM100 default `trtllm_mha`** | `_qwen3_5_hybrid_overrides` (`overrides.py:943-968`) explicitly downgrades to `"triton"` for the common case (radix cache on, no mamba extra buffer, no spec-decode) — `trtllm_mha` only re-activates under spec-decode or an extra mamba buffer. | Per sibling scout: `CudaPlatform._get_backend_priorities` for `device_capability.major==10` returns `[FLASHINFER, FLASH_ATTN, TRITON_ATTN, ...]`; FlashInfer passes every gate for this shape (head_dim 256, bf16) → **vLLM runs FlashInfer** for full-attn decode, with **no model-specific downgrade** reported. | No backend-selection concept at all. | Opposite defaults on the same op, same hardware: sglang's override takes this model family off the generic SM100 default (`trtllm_mha`) and onto Triton; vLLM takes the generic default. Only 10/40 layers, but they're the only layers with growing KV (higher per-step memory traffic than GDN's O(1) state). **Measurement hypothesis, not yet confirmed:** whether Triton or `trtllm_mha` is actually faster for this shape at bs 1..16 on B200 is unmeasured here — the override's own comment doesn't state a reason, so treat "profile both, don't assume the override implies Triton wins" as the M2 action item rather than inferring a performance conclusion from the code's existence. |
| 4 | **Routed-expert MoE GEMM forced to `flashinfer_trtllm`, confirmed to include our FP8 case** | `_qwen3_moe_family_overrides` (`overrides.py:994-1016`): forces server arg `moe_runner_backend="flashinfer_trtllm"` whenever SM100 and `quantization in ("fp8","modelopt_fp4",None)` and `moe_a2a_backend=="none"` (true at TP=1/EP=1) — **the condition literally includes `"fp8"`**, not just the unquantized case the scout traced. | Per sibling scout (bf16 case): `select_unquantized_moe_backend` priority `[FLASHINFER_TRTLLM, FLASHINFER_CUTLASS, TRITON, BATCHED_TRITON]`; FlashInfer TRT-LLM wins on B200 → same family of kernel, independently arrived at. vLLM's FP8-specific MoE path is not traced in the sibling report (see §5/§6). | Per-expert Python loop / dense reference implementation — no grouped-GEMM kernel concept. | Both engines converge on a FlashInfer TRT-LLM fused grouped-GEMM for MoE on B200 by default, i.e. both avoid the generic Triton grouped-GEMM path. At bs 1..16 with top-8-of-256 routing, average tokens/expert is ≤0.5. **Measurement hypothesis, not yet confirmed:** this shape *plausibly* makes the MoE GEMM the single most latency-dominated kernel choice in the model (256-way router × all 40 layers are MoE), but that's an inference from the shape, not a profiled result — treat it as the top candidate to profile first at M2, not an established conclusion. |
| 5 | **Shared-expert MLP: stream-overlap, not algebraic fusion, on CUDA** | `Qwen2MoeSparseMoeBlock.forward` (`qwen2_moe.py:540-564`): `elif self.alt_stream is not None and get_is_capture_mode():` → `forward_normal_dual_stream` (`qwen2_moe.py:498-538`) runs the shared-expert `Qwen2MoeMLP` (`qwen2_moe.py:171-217`, generic `quant_config`-parameterized) on `self.alt_stream` concurrently with routed-expert dispatch. Gated purely on CUDA-graph-capture mode (no token-count threshold). | Per sibling scout: `SharedExperts._determine_shared_experts_order` picks `MULTI_STREAM_OVERLAPPED` whenever `is_cuda() and aux_stream() is not None and M <= 256` — always true at decode. Functionally the **same strategy**, different threshold shape (M≤256 vs capture-mode flag). | Sequential dense add — no MoE-kernel-level scheduling concept. | Convergent design, not a difference — but worth confirming it survives FP8: I traced this and it does (§3) — `Qwen2MoeMLP` just takes `quant_config` generically, so FP8 quantization doesn't disturb the overlap. Shared-expert cost (inter=512, same as ~1 routed expert) is cheap to hide behind the routed grouped-GEMM at any batch, but the *win margin* shrinks as bs→1 because the thing it hides behind is itself shorter. |
| 6 | **GDN input-projection split across two streams** | `Qwen3_5GatedDeltaNet.forward` (`qwen3_5.py:555-571`): when `seq_len < DUAL_STREAM_TOKEN_THRESHOLD` (1024 on CUDA) and capture-mode and `_gdn_use_alt_stream`, the small `in_proj_ba` (2048→64) GEMM runs on `self.alt_stream` concurrently with the large `in_proj_qkvz` (2048→12288) GEMM on the main stream. Always true at decode (seq_len = batch size ≤16 ≪ 1024). | No equivalent claim found in the sibling scout report (not necessarily absent in vLLM — just not traced at this depth by that report). | No such split-stream concept. | A cheap ($) latency-hiding trick specific to sglang's GDN implementation: overlapping a tiny GEMM behind a much larger one is exactly the kind of thing a single persistent megakernel can do for free via warp-specialization, without any stream-launch machinery at all. |
| 7 | **CUDA-graph decode batch buckets exactly `{1,2,4,8,12,16,...}`** | `_generate_decode_cuda_graph_batch_sizes` (`server_args.py:4681`, list at `server_args.py:4692`): `[1,2,4,8,12] + list(range(16,257,8)) + ...` for the non-spec-decode case. | Not independently confirmed here (sibling scout mentions CUDA-graph padding/`NULL_BLOCK_ID` mechanics but I did not find its exact bucket list in that report). | No batching/padding concept. | Every batch size in our exact target set `{1,2,4,8,16}` (AC-4) lands **exactly** on an sglang capture bucket — no padding waste in sglang's own baseline numbers at the sizes we'll actually benchmark against (bucket 12 is simply unused by our target set). |
| 8 | **CPU/GPU overlap scheduling on by default** | `disable_overlap_schedule` defaults `False` (`server_args.py:908-915`) — the CPU scheduler and GPU model worker overlap by default; hybrid cache's mamba ping-pong buffering exists specifically to keep this safe for GDN state (per scout §3.3). | Not independently confirmed here. | No scheduling-overlap concept (single synchronous loop). | At bs≤16, GPU step time is small, so Python/scheduling overhead is proportionally large — sglang hides it structurally. A megakernel is a single persistent-kernel launch per step (or fewer), which sidesteps this entire overhead class **by construction** rather than needing to replicate the overlap machinery. |

---

## 2. Re-verification: the causal-conv1d decode-step quirk

**Scout's claim:** *"`gdn_backend.py:6-9` imports [`causal_conv1d_update`] directly and only
re-binds it to the CUDA `sgl_kernel` version for XPU/NPU/CPU (`gdn_backend.py:43-63`) — not for
CUDA. ... Worth double-checking on a fresher commit before assuming this is still true."*

**Verdict: CONFIRMED, at the recorded commit, with a stronger mechanism than "worth
double-checking" implied.**

1. `python/sglang/srt/layers/attention/linear/gdn_backend.py:6-9` imports both
   `causal_conv1d_fn` and `causal_conv1d_update` from
   `sglang.kernels.ops.mamba.causal_conv1d_triton` (the pure-Triton implementation).
2. `gdn_backend.py:37-42` (`if is_cuda():`) rebinds **only** `causal_conv1d_fn` to the
   CUDA-native version (`from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn as causal_conv1d_fn_cuda`).
   `causal_conv1d_update` is **not mentioned** in this branch.
3. The `elif is_xpu()` (`:43-48`), `elif is_npu()` (`:49-58`), and `elif is_cpu()` (`:59-63`)
   branches each rebind **both** functions to their platform-native versions. CUDA is the one
   platform where `causal_conv1d_update` is left on Triton.
4. The plain-decode call site — `forward_decode`, `gdn_backend.py:393-399` — calls this
   module-level `causal_conv1d_update` name, i.e. the Triton kernel, unconditionally on CUDA.
   (There is a second call site at `gdn_backend.py:526`, but it is inside the
   `is_target_verify` branch, i.e. MTP/speculative-decode verification — out of scope per AC-2's
   "no speculative decoding/MTP.")
5. **A CUDA-native wrapper for exactly this operation does exist**, just not wired up here:
   `python/sglang/srt/layers/attention/mamba/causal_conv1d.py:121-187` defines its own
   `causal_conv1d_update` that calls the real `sgl_kernel.causal_conv1d_update` CUDA kernel
   (imported at `:22-29` behind a `_HAS_SGL_KERNEL` try/except, invoked at `:175`) whenever
   `sgl_kernel` is importable, falling back to the same Triton kernel only when it isn't
   (`:153-164`). `gdn_backend.py`'s CUDA branch simply never imports this symbol.

Net: this reads as an **oversight, not a deliberate CUDA choice** — the CUDA-native decode-step
kernel exists one file away, is already wired up for XPU, and would be a one-line import change
to enable on CUDA. Git history can't help date this (shallow clone, one commit only). This is
directly on the hot path for our target: per §1 row 1's traced chain, the SM100 FlashInfer-GDN
auto-upgrade requires an explicit `--mamba-ssm-dtype bfloat16` CLI flag that no default sglang
launch of this checkpoint would pass (the field defaults to `None` and nothing populates it from
the checkpoint for this check), so the *entire* GDN decode stack (both the delta-rule recurrence
and this conv update) runs through Triton kernels by default for `Qwen3.5-35B-A3B-FP8` on B200 —
this isn't a hypothetical edge case.

**Correction to the scout's framing** (see also §5): don't read this as "sglang uses Triton where
vLLM uses CUDA" — vLLM's own decode-step conv-update kernel is *also* Triton (`tl.store`-based,
`vllm/model_executor/layers/mamba/ops/causal_conv1d.py:749-1060`, per the sibling scout report).
The real, more specific finding is that sglang built a CUDA-native alternative for this exact op
and left it unwired for CUDA — not that it picked Triton where a rival engine picked CUDA.

---

## 3. FP8 handling for this checkpoint class (block-wise 128×128, dynamic activation)

Neither scout report covers this in depth — both were explicitly scoped to a bf16 checkpoint
(`vllm-qwen35-graph.md:8`: *"TP=1, EP=1, bf16, no quantization"*; the sglang report's MoE kernel
matrix is captioned *"bf16 quant-type path included"*). This section is fresh source
investigation, grounded against the actual checkpoint's `quantization_config` (§0).

### 3.1 Config ingestion — `Fp8Config`

`python/sglang/srt/layers/quantization/fp8.py:220-273` (`__init__`) and `:298-334`
(`from_config`): reads `quant_method` (`"fp8"`), `activation_scheme` (`"dynamic"`),
`weight_block_size` (`[128,128]`), and `ignored_layers` from **either** `"ignored_layers"` **or**
`"modules_to_not_convert"` (`fp8.py:307-309`) — our checkpoint uses the latter key, and sglang
consumes it directly rather than hardcoding a skip-list. Block-wise quantization is asserted to
require the dynamic activation scheme (`fp8.py:264-267`; raises otherwise) — matches
`constraint.md`'s pinned recipe exactly, and the checkpoint's declared `modules_to_not_convert`
(lm_head, embeddings, per-linear-layer `conv1d`/`in_proj_a`/`in_proj_b`, per-MoE-layer
`mlp.gate`/`mlp.shared_expert_gate`, plus vision/MTP modules out of AC-2 scope) matches
`constraint.md`'s description 1:1.

**Per-module assignment, traced end to end (not just the config parse above):**
`Fp8Config.get_quant_method(layer, prefix)` (`fp8.py:336-399`) is the per-module loader hook —
for every `LinearBase` module it calls `is_layer_skipped(prefix, self.ignored_layers, ...)`
(`quantization/utils.py:70-110`) and returns `UnquantizedLinearMethod()` if skipped, else
`Fp8LinearMethod(self)` (`fp8.py:355-359` for the linear branch; the `FusedMoE` branch at
`:365-399` does the same skip check before returning `Fp8MoEMethod(self)`). The name match is
dotted-boundary, not substring (`_module_path_match`, `quantization/utils.py:48-59`: `mlp.gate`
matches only on a `.`-bounded prefix, so it does **not** false-match `mlp.gate_up_proj` — the
function's own comment cites this exact collision risk for a same-family checkpoint's
`modules_to_not_convert` list, i.e. this matching behavior was already hardened for names shaped
like ours). For every non-skipped linear, `Fp8LinearMethod.__init__` (`fp8.py:414-457`) sets
`self.w8a8_block_fp8_linear = dispatch_w8a8_block_fp8_linear()` (`:457`) once, and
`Fp8LinearMethod.apply()` (`fp8.py:929`, dispatch call at `:1004-1013`) calls
`self.w8a8_block_fp8_linear(...)` on every forward pass. So "everything not in
`modules_to_not_convert` gets the block-FP8 linear method, and its calls reach
`dispatch_w8a8_block_fp8_linear`" is traced, not asserted — module-by-module, from config parse
through the per-call dispatch.

### 3.2 Two different FP8 GEMM kernel families, running concurrently

This is the part worth calling out explicitly: **linear** FP8 GEMMs and **MoE** FP8 GEMMs go
through *different* dispatch logic and, on B200, end up on *different* kernel families —
concurrently, on separate streams (§1 row 5).

**Linear GEMMs** (attention `qkv_proj`/`o_proj`, GDN's `in_proj_qkvz`, shared-expert
`gate_up_proj`/`down_proj` — everything not in `modules_to_not_convert`) dispatch through
`dispatch_w8a8_block_fp8_linear()` (`fp8_utils.py:455-470`) → `_dispatch_auto_backend()`
(`fp8_utils.py:598-615`), priority order:

1. **DeepGEMM**, if `deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM` — which requires the `deep_gemm`
   package to be importable (`deep_gemm_wrapper/configurer.py:18-30`) **and**
   `SGLANG_ENABLE_JIT_DEEPGEMM` (env default **`True`**, `environ.py:690`). SM100 qualifies
   (`configurer.py:21-23`, rejects only SM<90 and exactly SM120), and
   `DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL` (`configurer.py:38`) unconditionally follows —
   i.e. once DeepGEMM is
   selected on B200, sglang requantizes weight scales to UE8M0
   (`requant_block_scale_ue8m0_for_deepgemm`, called from `fp8.py:684-692` whenever
   `self.w8a8_block_fp8_linear is deepgemm_w8a8_block_fp8_linear_with_fallback`). **Whether this
   actually fires is two separate conditionals, both worth stating explicitly:** (a) whether
   `deep_gemm` is pip-installed on the B200 box — open item, §6 — and (b) a model-specific
   accuracy gate that the sibling vLLM doc found and sglang does **not** appear to have: vLLM's
   `should_auto_disable_deep_gemm(model_type)` explicitly excludes `"qwen3_5_moe_text"` on SM100
   ("known to have accuracy degradation with DeepGemm's E8M0 scale format on Blackwell GPUs",
   `workspace/docs/qwen35/vllm-graph.md:939-965`). I grepped sglang's tree for an equivalent
   per-model exclusion (`EXCLUDED_MODEL_TYPES`, `disable_deep_gemm`, `qwen3_5`+`deep_gemm`) and
   found none for this model — the closest precedent is a **different** model
   (`overrides.py:540-544`: MiniMax-M3's `deep_gemm` **MoE runner** is force-disabled for a
   bf16-weights corruption bug, not an fp8/E8M0 accuracy issue, and not the dense-linear path
   this section is about). **So: if `deep_gemm` is installed on the B200 box, sglang's default
   dispatch will select it — and requantize to UE8M0 — for this model's dense FP8 linears on
   B200, with no equivalent of vLLM's guard against the accuracy issue vLLM's own team flagged
   for this exact model. This is a correctness risk to check at M2, not just an availability
   question.**
2. Else **FlashInfer TRT-LLM block-fp8 linear**
   (`flashinfer_gemm_w8a8_block_fp8_linear_with_fallback`, `fp8_utils.py:648+`) if
   `is_blackwell_supported()` (device capability major ∈ {10,11,12}, CUDA≥12.8 —
   `utils/common.py:274-279`; B200 = major 10, qualifies) and FlashInfer is importable.
3. Else CUTLASS (SM120 only — does not apply to B200/SM100), AITER (ROCm only), else **Triton**.

**Routed-expert MoE GEMMs** do *not* go through this priority list at all — the model-family
override (§1 row 4, `overrides.py:994-1016`) force-sets `moe_runner_backend="flashinfer_trtllm"`
as a server arg before `Fp8MoEMethod.create_moe_runner` (`fp8.py:2248-2277`) ever runs, so the
generic auto-dispatch (`Fp8MoEMethod.is_deepgemm_moe_runner_backend_enabled`, `fp8.py:1067-1080`
— which actually requires a **multi-GPU** all-to-all backend (`deepep`/`mooncake`/`nixl`) and so
can never fire at our single-GPU TP=1/EP=1 target regardless) is bypassed entirely. The runner
goes straight to the `flashinfer_trtllm` branch (`fp8.py:2467-2510`), builds a
`FlashInferTrtllmFp8MoeQuantInfo` (`moe_runner/flashinfer_trtllm.py:628-658`) with
`block_quant=True` and the raw `w13_weight_scale_inv`/`w2_weight_scale_inv` tensors (no
dequant-then-requant round trip), and calls FlashInfer's
`trtllm_fp8_block_scale_moe`/`_routed` kernel
(`fused_experts_none_to_flashinfer_trtllm_fp8`, `moe_runner/flashinfer_trtllm.py:661-790`; kernel
calls at `:738` and `:773`) with `Fp8QuantizationType.DeepSeekFp8` (`:695` — the
per-128×128-block format, as opposed to `MxFp8`) and **dynamic activation quantization, done once
per 128-wide K-group for each token**
(`per_token_group_quant_fp8(hidden_states, weight_block_k, column_major_scales=True)`, `:713-716`
— hidden_size/128 groups per token, not one quant op per token), freshly on every call.

**So:** *if and only if* `deep_gemm` is installed (§6) **and** sglang does not gain a
Qwen3.5-specific DeepGEMM exclusion on a fresher commit (neither is confirmed here, and §3.2's
DeepGEMM bullet above flags a real accuracy question vLLM's team already raised for this model),
the shared-expert MLP's GEMMs would run through **DeepGEMM** while the routed-expert grouped-GEMM
in the very same MoE block runs through **FlashInfer TRT-LLM** — two different FP8 kernel
providers, on two different CUDA streams, computing concurrently every decode step, every MoE
layer (all 40). The FlashInfer-TRT-LLM side of this split is unconditional (§1 row 4); the
DeepGEMM side is not.

### 3.3 What's worth stealing

- The **direct block-scale-in/block-scale-out kernel call** — no upcast-to-bf16-then-requantize
  detour. Activations are dynamically quantized, once per 128-wide K-group for each token, right
  before the kernel call, and fed straight in alongside the checkpoint's native
  `weight_scale_inv`; matches our checkpoint's storage format exactly, so a megakernel that also
  natively consumes block-fp8 scales avoids an entire extra pass that a naive "dequant to bf16,
  run a bf16 kernel" implementation would pay at every GEMM.
- The **routing-different-GEMM-shapes-to-different-kernel-families idea** (§3.2: dense linears
  vs. the MoE grouped-GEMM don't have to share one kernel) — worth replicating the *idea*
  regardless of which specific kernel wins each side.
- **Caution, now sharper than my first pass:** the sibling vLLM doc has since landed
  (`workspace/docs/qwen35/vllm-graph.md:1188`) and shows vLLM does **not** converge with sglang
  here for our checkpoint — `should_auto_disable_deep_gemm("qwen3_5_moe_text")` makes vLLM
  **always** reject DeepGEMM for this model's dense fp8 linears on SM100/SM120 and fall to
  CUTLASS instead, specifically because of the E8M0 accuracy issue (§3.2). So DeepGEMM-for-linears
  is *not* something to "steal" as a proven-safe convergent choice — it's a real accuracy question
  vLLM's team already answered "no" to for this exact model, that sglang (as of this commit) has
  no equivalent guard against. Treat the *dispatch-split idea* as worth stealing; treat
  *DeepGEMM specifically for this model on B200* as unresolved and worth validating against AC-3
  before relying on it for anything, including as a numeric baseline.

---

## 4. Ideas worth considering for a megakernel

1. [M2-design] Benchmark against the Triton `fused_recurrent_gated_delta_rule_packed_decode`
   algorithm as sglang's *actual* GDN decode reference for this checkpoint — under default launch
   args the SM100→FlashInfer auto-upgrade never fires (requires an explicit
   `--mamba-ssm-dtype bfloat16` that nothing populates from the checkpoint, §1 row 1), so don't
   design against a kernel sglang won't run unless specially configured.
2. [M3-optimization] Fuse the causal-conv1d ring-buffer update directly into the megakernel's GDN
   block (single warp-specialized op, no separate launch) — sglang's own unwired CUDA-native
   wrapper (§2) is a low-risk candidate to try; whether it's an actual win at bs≤16 under
   CUDA-graph-captured decode is an open measurement (§1 row 2), not yet established.
3. [M2-design] Don't assume `trtllm_mha` is the right full-attention decode target — sglang's
   model-family override takes this model off the generic SM100 default and onto Triton (§1 row
   3) while vLLM takes the generic FlashInfer default; profile both real kernels before picking
   which one to beat.
4. [M3-optimization] Replicate the small-GEMM/large-GEMM stream overlap (GDN `in_proj_ba` behind
   `in_proj_qkvz`; shared-expert MLP behind routed-expert dispatch) as warp-specialization inside
   one persistent kernel — same latency-hiding idea, zero stream-launch overhead.
5. [M2-design] Prioritize the MoE grouped-GEMM's small-M tiling early in the profiling plan — at
   top_k=8/256 experts and bs 1..16, both sglang and vLLM independently default to a FlashInfer
   TRT-LLM fused kernel instead of a generic per-expert loop. Treat that convergence as a signal
   worth profiling first at M2, not yet a measured conclusion about where the latency actually
   lives.
6. [M3-optimization] Steal the block-fp8 activation quantization pattern — done dynamically, once
   per 128-wide K-group for each token, feeding directly into a block-scaled GEMM, no bf16
   round-trip — matches this checkpoint's native storage; a naive dequant-then-run-bf16 path pays
   an avoidable extra pass at every quantized GEMM.
7. [M2-design] Branch each GEMM's dtype plan on the checkpoint's actual `modules_to_not_convert`
   list (GDN a/b-gate projections, router gates, shared-expert gate, conv1d, embeddings/lm_head
   stay bf16) rather than inferring it architecturally — sglang's `ignored_layers` mechanism reads
   this straight from the checkpoint, and the megakernel's weight-loading path should too.
8. [M3-optimization] Check whether `deep_gemm` is installed on the target B200 box (§6) — and if
   so, validate its dense-FP8-linear numerics against AC-3 before trusting them as a baseline: with
   no accuracy guard equivalent to vLLM's explicit exclusion of this exact model on SM100 (§3.2),
   sglang's default dispatch would use DeepGEMM's UE8M0 path unchecked. If it validates clean,
   those numbers are a ready-made bar for those ops; don't assume it validates clean first.

---

## 5. Corrections / contradictions vs the scout report (explicit)

- **Resolved, and this doc's own first pass over-asserted the mechanism (caught in review):** the
  scout's open item *"whether the real checkpoint's config.json sets `mamba_ssm_dtype`"* is
  answered — **`float32`**. My first draft then claimed this checkpoint value is what fails the
  SM100 FlashInfer-upgrade check; tracing the actual propagation (§1 row 1) shows that check reads
  a separate, CLI-only `server_args.mamba_ssm_dtype` field that defaults to `None` and is never
  populated from the checkpoint for this purpose — only an unrelated opt-in flag
  (`--enable-gdn-replayssm-spec`) ever sets it. **The conclusion is unchanged** (Triton beats
  FlashInfer for GDN decode under default launch args) and is actually more robust than first
  stated — it holds regardless of checkpoint specifics, contingent only on the operator not
  passing `--mamba-ssm-dtype bfloat16`. The checkpoint's declared `float32` does matter, but for a
  different consumer: `mamba2_state_dtype()` sizing the recurrent-state cache buffer.
- **Resolved, no change:** the scout's second open item (whether `output_gate_type` /
  `partial_rotary_factor` are restated in the checkpoint) is also answered by the same config
  file: `partial_rotary_factor=0.25` **is** explicitly present (`rope_parameters`, matches the
  scout's assumed inherited default); `output_gate_type` is **not** present in the checkpoint's
  `text_config`, so it does inherit Qwen3Next's default as the scout assumed. No correction
  needed, just now confirmed rather than assumed.
- **Confirmed, framing refined:** the causal-conv1d decode-step quirk (scout's third open item)
  is real (§2), but the scout's implicit vLLM contrast doesn't hold — vLLM's own decode-step
  update kernel is also Triton, per the sibling report. The distinguishing fact is sglang's dead
  CUDA-native wrapper sitting unused, not an engine-vs-engine kernel-language choice.
- **Extended, not contradicted:** the scout captioned the `flashinfer_trtllm` MoE finding "bf16
  quant-type path included," which reads as "this is *one* path among others." Direct source
  reading of the override condition (`overrides.py:994-1016`) shows fp8 is treated identically to
  bf16/unset — there is no separate/different MoE-runner selection path for our FP8 checkpoint.
  Low risk: the scout's finding generalizes cleanly.

---

## 6. Open items

- **Is `deep_gemm` actually installed on the B200 box, and does sglang's DeepGEMM path validate
  against AC-3 for this model?** Two questions, not one — (a) installed-or-not determines whether
  FP8 dense-linear GEMMs (§3.2) get DeepGEMM or fall through to FlashInfer/Triton; (b) *if*
  installed, sglang has no confirmed guard against the E8M0 accuracy issue vLLM's team explicitly
  excludes this exact model for on SM100 (§3.2/§3.3) — so a DeepGEMM-enabled run needs an AC-3
  correctness check specifically, not just a "does it run" check. Not checkable from this machine
  (no GPU, per `constraint.md` §1); check with `python -c "import deep_gemm"` on `catalyst-B200`,
  and if present, run the pinned prompt set through it before trusting any DeepGEMM-path numbers
  or outputs. Effort: low-to-medium ; owner: M2 build step.
- **Does any sglang launch command we'd actually run pass `--mamba-ssm-dtype bfloat16`
  explicitly?** §1 row 1's traced chain shows the FlashInfer-GDN auto-upgrade requires this exact
  flag and nothing populates it from the checkpoint by default — so the Triton-wins conclusion
  holds for a *default* launch, but would flip if a baseline/comparison script we use passes it.
  Check whatever launch command M2/M4 actually uses (or any reference sglang serving command in
  the repo) for this flag before treating "sglang uses Triton for GDN decode" as settled for that
  specific run. Effort: low ; owner: whoever stands up an sglang comparison run.
- **vLLM's own FP8 linear/MoE kernel dispatch — partially checked this round, not exhaustively.**
  `workspace/docs/qwen35/vllm-graph.md` has landed; this round's revision cross-checked the
  DeepGEMM/E8M0 question against it (§3.2/§3.3, resolved: vLLM excludes DeepGEMM for this model,
  sglang doesn't). The MoE-runner "convergence" claim in §1 row 4 (both engines defaulting to
  `flashinfer_trtllm`) is consistent with what I glimpsed in `vllm-graph.md` while chasing the
  DeepGEMM question but was not independently re-verified end-to-end this round — a full diff of
  vllm-graph.md's FP8 section against this doc's §3 is still worth doing before leaning hard on
  the row 4/5 convergence claims.
- **Rows without a confirmed vLLM comparison** (§1 rows 1, 6, 7, 8): absence of a claim in the
  sibling scout report is not evidence of absence in vLLM — it means that report didn't trace it
  at this depth. Don't read the blank cells as "vLLM doesn't do this."
