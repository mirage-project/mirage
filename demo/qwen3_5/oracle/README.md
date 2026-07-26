# M2-I3 — HF numerical oracle for Qwen/Qwen3.5-35B-A3B-FP8 (probe P6)

Per-op tensor dumps from the real HF `transformers` FP8 model, for one GDN layer, one
full-attention layer, and the MoE block belonging to each — the ground-truth oracle every M2
kernel unit test (M2-I4, M2-I5, M2-I6, M2-I7, ...) compares against, per the DFlash bring-up
ladder (`docs/qwen35/v1-architecture.md` §12): spec → **`ref_dump.py` oracle (this issue)** →
standalone kernel vs oracle → test-mode in the megakernel → end-to-end.

## Layers hooked

| role | layer index | why |
|---|---|---|
| GDN (linear attention) | **0** | first GDN layer; `layer_types[0] == "linear_attention"` |
| Full attention | **3** | first full-attention layer; `layer_types[i] = full_attention` iff `(i+1) % 4 == 0` (`docs/qwen35/vllm-graph.md` §1.1) |
| MoE block (primary, "the one MoE block" per the issue contract) | **moe0** = `layers[0].mlp` | co-located with the GDN layer |
| MoE block (secondary cross-check, per probe P6's own spec — `v1-architecture.md` §14 P6: *"router probs/ids/weights, per-expert partials, shared-expert path (both layers' MoE)"*) | **moe3** = `layers[3].mlp` | co-located with the full-attention layer |

`Qwen3_5MoeSparseMoeBlock` is structurally identical regardless of which layer it sits in (every
layer has MoE — `vllm-graph.md` §2.3), so `moe0`/`moe3` are two independent instances of the same
op, not two different op types. Indices are recorded in every manifest's `meta.gdn_layer_idx` /
`meta.attn_layer_idx` and are `--gdn-layer`/`--attn-layer` CLI flags, not hardcoded.

## Regeneration

```bash
# GPU etiquette first (see resources.md / B200 rules): pick a free GPU, claim the lock.
ssh catalyst-B200 'nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv'
ssh catalyst-B200 'echo "<you> M2-I3 ref_dump.py $(date -Iseconds)" > ~/mpk-qwen35/.gpu-locks/M2-I3.lock'

scp ref_dump.py pytorch_reference.py validate_self_consistency.py \
    catalyst-B200:~/mpk-qwen35/oracle-work/

ssh catalyst-B200 'bash -s' <<EOF
export CUDA_VISIBLE_DEVICES=<free_gpu_id>
export HF_HOME=~/mpk-qwen35/hf
export PATH=/usr/local/cuda-12.8/bin:\$PATH
source ~/mpk-qwen35/venv-vllm/bin/activate
cd ~/mpk-qwen35/oracle-work
TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1 python3 ref_dump.py \
  --model-id Qwen/Qwen3.5-35B-A3B-FP8 \
  --revision 9d1823d2dee688a6b25e77009dc727688c44936e \
  --prompts-file ~/mpk-qwen35/prompts_readonly_copy.jsonl \
  --prompt-id p01-history \
  --gdn-layer 0 --attn-layer 3 \
  --prefill-tokens 8 --decode-steps 1 \
  --out ~/mpk-qwen35/oracle-work/dumps
python3 validate_self_consistency.py --dump-dir ~/mpk-qwen35/oracle-work/dumps --mode both
EOF

ssh catalyst-B200 'rm ~/mpk-qwen35/.gpu-locks/M2-I3.lock'  # release when done
```

`--prompts-file` may also point at the repo's `.pm/eval/prompts.jsonl` (read-only; same pinned
prompt set the AC-3 harness uses) — `~/mpk-qwen35/prompts_readonly_copy.jsonl` on B200 is a copy
of the same file, used here purely to avoid an extra scp round trip. `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1`
matches `accept/reference/generate_reference.py`'s default: DeepGEMM crashes on this checkpoint's
GDN `in_proj_qkv` on B200/SM100 (see that directory's README "FP8 execution path"); this oracle
must run the SAME Triton `finegrained-fp8` backend as the pinned AC-3 reference, not a different
one. Env pins (this run): `torch==2.11.0+cu130`, `transformers==5.14.1`, `accelerate==1.14.0`,
same venv (`~/mpk-qwen35/venv-vllm`) as `accept/reference/`.

Model loading takes ~30s (safetensors already cached under `HF_HOME`); the whole run (load +
2 forward passes + validation) completes in well under a minute of GPU time.

## What actually ran (empirical, not assumed — see `runtime_diagnostics.json`)

- **GDN fast path is OFF**: `fla` / `causal_conv1d` are not installed in `venv-vllm`, so
  `Qwen3_5MoeGatedDeltaNet` runs transformers' pure-torch fallback (`torch_causal_conv1d_update`,
  `torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`) — confirmed via
  `is_causal_conv1d_available()`/`is_flash_linear_attention_available()` returning `False` and by
  inspecting the actually-bound function objects. `forward_is_plain_class: true` for the GDN
  module (the `@use_kernel_forward_from_hub` decorator did NOT swap in a hub kernel here).
- **The dense FP8 Linears** (`in_proj_qkv`, `in_proj_z`, `out_proj`, `q/k/v/o_proj`,
  `shared_expert.{gate,up,down}_proj`) keep their weight in `torch.float8_e4m3fn` with a
  `weight_scale_inv` buffer (BF16 in the checkpoint, shape `[N/128, K/128]`) — confirmed by
  direct attribute probe, matching `docs/qwen35/vllm-graph.md` §3.4's documented layout exactly.
- **Routed MoE experts are NOT the plain eager loop** shown in
  `Qwen3_5MoeExperts.forward` (transformers 5.14.1 source) — `@use_experts_implementation` swaps
  in a specialized `FP8Experts` backend class for this checkpoint (`forward_is_plain_loop_class:
  false`). Its internal per-expert accumulate/cast points are therefore an opaque black box, the
  same situation `vllm-graph.md` §2.3.4 documents for vLLM's FlashInfer TRT-LLM MoE cubin. `ref_dump.py`
  does NOT reimplement `FP8Experts`'s internals (an earlier attempt to do so crashed with a
  dtype mismatch — see git history / this file's "lessons learned" below); it calls
  `self.experts(...)` as a black box and dumps only the module boundary (input, output, which
  experts fired, and the weight+scale tensors needed for an independent recompute).
- **Attention ran with `attention_mask=None`** — HF's mask-construction utilities did not
  materialize an explicit additive mask for either the prefill or the decode call (recorded per
  op in `meta.attn.attention_mask_used_is_none`). The real attention backend therefore enforces
  causality implicitly (e.g. an `is_causal=True`-style SDPA path), not via `eager_attention_forward`'s
  literal `attn_weights + attention_mask` step. See "Answers to the open P6 bit-parity questions"
  below for why this matters and how the validator handles it.

## Dump name → `docs/qwen35/vllm-graph.md` op mapping

Prefix `gdn.` = the GDN layer's ops (§2.1.7 op table); `attn.` = the full-attention layer's ops
(§2.2.6); `moe0.`/`moe3.` = the two MoE block instances (§2.3.5). HF's eager/torch-fallback code
decomposes some vLLM kernel-internal steps into separate explicit ops (e.g. the q/k/v split
happens as a literal `torch.split` AFTER the conv here, where vLLM's fused kernel does it
implicitly inside the recurrence kernel) — noted per-row where the decomposition differs.

### GDN layer (`gdn.*`) vs vllm-graph.md §2.1.7

| dump name | vllm-graph row | note |
|---|---|---|
| `gdn.input_layernorm.{input,output}` | 1 (`input_layernorm`) | plain forward hook |
| `gdn.qkv_proj_out` | 3 (`in_proj_qkvz`), q/k/v portion | HF keeps `in_proj_qkv`/`in_proj_z` as TWO separate GEMMs (`v1-architecture.md` §2.0 "GDN projections kept separate" — matches MPK's own v1 choice, not vLLM's fused layout) |
| `gdn.z_proj_out` | 3, z portion | see above |
| `gdn.b_proj_out`, `gdn.a_proj_out` | 4 (`in_proj_ba`) | HF also keeps `in_proj_b`/`in_proj_a` separate (both BF16, unquantized — `N=32 < block_n=128`) |
| `gdn.conv_in` | — (pre-op 6 input) | fused `[q\|k\|v]`, pre-conv, `[B,8192,T]` |
| `gdn.conv_out` | 6 (`causal_conv1d_update`) | **two distinct code paths**, both dumped under the same name: prefill uses a plain padded `nn.Conv1d` + slice (`torch_causal_conv1d_prefill` in `pytorch_reference.py`, not a literal HF function — see its docstring); decode (cached, single-token) uses the literal `torch_causal_conv1d_update` |
| `gdn.q_split`, `gdn.k_split`, `gdn.v_split` | 5 (`split`), but positioned AFTER the conv here (HF explicit split vs vLLM's kernel-internal unpack) | |
| `gdn.beta` | feeds row 7 | `sigmoid(b)` computed in `b`'s native (bf16) dtype, **no `.float()` before the sigmoid** — this is the answer to `vllm-graph.md` §2.1.4/§6 g.10's open question: HF's `β` round-trips through bf16 exactly like vLLM's packed decode kernel |
| `gdn.decay_g` | feeds row 7 | `g = -exp(A_log.float()) * softplus(a.float() + dt_bias)`, computed in fp32 (both operands upcast) |
| `gdn.core_state_before` / `gdn.core_state_after` | row 7 state R/W | fp32 `[B,32,128,128]`; only `core_state_before` exists when a cache is present (decode) |
| `gdn.core_attn_out` | 7 (`fused_recurrent_gated_delta_rule_packed_decode`) | prefill → `torch_chunk_gated_delta_rule`; decode → `torch_recurrent_gated_delta_rule` |
| `gdn.gated_norm_out` | 8 (`RMSNormGated(128)`) | dumped in its PRE-reshape flat form `[B*T*32, 128]`, not yet regrouped to `[B,T,4096]` |
| `gdn.out_proj_out` | 10 (`out_proj`) | |
| `gdn.post_attention_layernorm.{input,output}` | (epilogue norm, feeds the MoE block) | plain forward hook |
| `gdn.__weight.*` | (weights/scales, static) | `in_proj_qkv`, `in_proj_z`, `in_proj_b`, `in_proj_a`, `out_proj`, `conv1d_weight`, `A_log`, `dt_bias`, `norm_weight` + `*_scale_inv` where fp8 |

Rows 2 and 9 of vllm-graph.md ("quantize activation") are internal to HF's fp8 Linear dispatch —
not separately hookable ops in the eager path; they are implicitly bracketed by the GEMM
input/output dumps (see "Dtype and tolerance policy").

### Full-attention layer (`attn.*`) vs vllm-graph.md §2.2.6

| dump name | vllm-graph row | note |
|---|---|---|
| `attn.input_layernorm.{input,output}` | 1 | |
| `attn.q_proj_out` | 3 (`qkv_proj`), q\|gate portion | HF keeps `q_proj` (carries `[q\|gate]`, 8192-wide) separate from `k_proj`/`v_proj` (512-wide each) — three GEMMs, not one fused `[9216,2048]` GEMM |
| `attn.k_proj_out`, `attn.v_proj_out` | 3, k/v portion | |
| `attn.q_split`, `attn.gate_split` | 5 (`per-head chunk`) | `torch.chunk(q_proj_out.view(...), 2, dim=-1)` |
| `attn.q_norm_out`, `attn.k_norm_out` | 6, 7 (`q_norm`, `k_norm`) | Gemma `(1+w)` |
| `attn.rope_cos`, `attn.rope_sin` | (RoPE table, computed at the Model level) | needed to recompute RoPE independent of a live model |
| `attn.q_rope`, `attn.k_rope` | 8 (partial NeoX RoPE) | rotary_dim=64 of 256; dims `[64:256)` pass through |
| `attn.kv_cache_k_after_write`, `attn.kv_cache_v_after_write` | 9 (KV-cache write) | decode only (`past_key_values.update(...)`); HF's `DynamicCache` — no separate slot/paged-cache machinery to dump, unlike vLLM |
| `attn.attention_mask_used` (usually absent — see `attention_mask_used_is_none` in meta) | (mask input to row 10) | **empirically `None`** on every call this run made — see "What actually ran" above |
| `attn.core_attn_out` | 10 (paged attention decode) | real backend unknown/opaque (mask was `None`); see tolerance policy |
| `attn.gate_sigmoid_mul_out` | 11 (`out * sigmoid(gate)`) | |
| `attn.o_proj_out` | 13 (`o_proj`) | |
| `attn.post_attention_layernorm.{input,output}` | (epilogue norm) | |
| `attn.__weight.*` | (weights/scales) | `q_proj`, `k_proj`, `v_proj`, `o_proj` + scales, `q_norm_weight`, `k_norm_weight` |

### MoE block (`moe0.*` / `moe3.*`) vs vllm-graph.md §2.3.5

| dump name | vllm-graph row | note |
|---|---|---|
| `moeN.layer_input` | (input to the whole block) | |
| `moeN.shared_gate_proj_out`, `moeN.shared_up_proj_out` | 4 (`shared_expert.gate_up_proj`) | HF keeps `gate_proj`/`up_proj` as two GEMMs (`Qwen3_5MoeMLP`), not one fused `[1024,2048]` |
| `moeN.shared_silu_mul_out` | 5 (`SiluAndMul`) | |
| `moeN.shared_down_proj_out` | 6 (`shared_expert.down_proj`) | |
| `moeN.router_logits` | (router GEMM, feeds row 3) | BF16, `quant_config=None` — never quantized |
| `moeN.router_probs` | 3, softmax stage | full-256 fp32 softmax |
| `moeN.topk_weights_raw`, `moeN.topk_ids` | 3, top-k stage | `torch.topk` — see "tie-breaking" finding below |
| `moeN.topk_renorm_weights` | 3, renormalize stage | the literal example name from this issue's contract |
| `moeN.expert_{id}.*` (`token_idx` in `meta`, `weights_for_tokens`, `__weight.gate_up_proj[_scale_inv]`, `__weight.down_proj[_scale_inv]`) | 9–11 (routed grouped GEMMs), per hit expert | **module boundary only** — see "FP8Experts is a black box" above; only experts that actually fired for this probe are dumped (52/44/8/8 across the 4 dump sets, well under the 128-expert cap) |
| `moeN.routed_expert_output` | 9–11 combined (weighted sum of routed experts) | the REAL `FP8Experts` output |
| `moeN.shared_gate_logit`, `moeN.shared_gate_sigmoid` | (shared-expert gate, feeds row 12 analog) | BF16, `quant_config=None` |
| `moeN.shared_output_gated` | | `sigmoid(x @ W_sg^T) * shared_down_proj_out` |
| `moeN.combined_output` | 12 (`weighted reduce + add shared`) | `routed_expert_output + shared_output_gated` |
| `moeN.__weight.router_gate_weight`, `.shared_expert_gate_weight`, `.shared_expert.*` | (weights) | |

`meta.moeN.num_distinct_experts_hit` records how many DISTINCT experts fired across the probe's
tokens (out of 256) — this is the number the "reconstructed from N/N hit experts" note in the
validator report refers to.

## Self-consistency validator (`validate_self_consistency.py`) — the acceptance centerpiece

For every dumped op, the validator recomputes it in plain torch **from that op's own dumped
inputs** (via the formulas in `pytorch_reference.py`) and compares to the dumped output. A hook
placed at the wrong point (e.g. capturing pre-RoPE `q` mislabeled as post-RoPE, or a stale view
before a reshape) produces a recompute that is *structurally* wrong and fails even the loosest
tolerance below; this is exactly what happened during development (see "Lessons learned") and is
why this check is the acceptance centerpiece, not a formality.

### Dtype and tolerance policy

Tensors are saved in their **native runtime dtype** — bf16 stays bf16, the fp32 recurrent state
stays fp32, fp8 weights stay `float8_e4m3fn` — never blanket-upcast on save (that would both
bloat the artifact and hide the real dtype semantics probe P6 exists to pin down). All ops are
compared in fp32 (upcasting only at comparison time, matching the test-mode skill's "cast to a
higher precision for a trustworthy reference" convention), against one of three tolerances:

| tier | atol / rtol | when | rationale |
|---|---|---|---|
| `TIGHT` | 5e-3 / 5e-3 | norms, RoPE, splits/reshapes, softmax+topk+renorm, elementwise combine, GDN recurrence (torch-fallback path is literally the same code) | the recompute is the SAME formula on the SAME dumped input; near-machine-precision agreement is expected |
| `LOOSE_FP8` | 2.0 / 0.25 | every dense/expert FP8 GEMM | the real op is a genuine fp8×fp8 kernel with its OWN dynamic activation quantization; the recompute dequantizes only the WEIGHT and matmuls against the full-precision dumped activation, so it cannot reproduce activation-side quantization noise — a quantization-scale-sized diff is expected, not a bug |
| `LOOSE_KERNEL` | 0.05 / 0.10 | `attn.core_attn_out` only | real backend is opaque (mask was `None`); see below |

Observed on this run (see `dumps_sample/validator_transcript.txt` for the full table): every
`TIGHT` and `LOOSE_FP8` check passed with large margin (e.g. FP8 GEMMs measured ~0.02–0.2 max abs
error against a ~2.0 budget); `LOOSE_KERNEL`'s two instances measured 0.023 (prefill) / 0.017
(decode) max abs against a 0.05 budget.

### Two real findings this validator surfaced (not hook bugs, but genuine — documented, not hand-waved)

1. **`attn.core_attn_out` needs a reconstructed causal mask.** First attempt used a literal
   `eager_attention_forward` transcription with the dumped `attention_mask` (which was `None`),
   producing FULL bidirectional attention — max abs error 2.56 (prefill, 8 tokens) / 0.017
   (decode, 1 query — causality is a no-op for a single trailing query, so this case was
   unaffected). This was a real bug in the RECOMPUTE, not the dump: the dump is what the model
   actually computed (causally correct, via whatever its real backend is); the fix was
   reconstructing the causal mask explicitly in `pytorch_reference.eager_attention` rather than
   trusting a `None` mask to mean "no masking needed". After the fix, prefill's error dropped
   ~100x to 0.023, converging to the SAME small residual as decode's unaffected 0.017 — consistent
   with ordinary fused-kernel-vs-naive-matmul accumulation noise, not a second bug.
2. **`torch.topk` tie-breaking is backend/device-dependent.** `moe0`/`moe3.topk_ids` initially
   mismatched on 2–5 of 8 rows (out of 64), always at the LAST selected slot, always at an EXACT
   floating-point tie in `router_probs` between two expert ids (e.g. `0.07472312450408936`
   appearing identically at both id 25 and id 188 in one row) — confirmed by gathering
   `router_probs` at both index sets and finding them identical. Root cause: the real computation
   ran `torch.topk` on CUDA; the validator's first pass ran it on CPU after loading the dumped
   tensor — different backends can break an exact tie differently. Fix: run the validator's topk
   recompute on CUDA when available (matches the original device) AND make the id comparison
   tie-tolerant regardless (gather-and-compare values, not raw index equality) as a robust
   fallback. **This is a live instance of exactly the tie-breaking risk `.pm/goal.md`'s AC-3
   waiver clause and `vllm-graph.md` §2.3.2's "lower index wins" kernel convention are both
   about** — HF's plain `torch.topk` does NOT itself guarantee "lower index wins" the way vLLM's
   custom `topk_softmax` CUDA kernel explicitly does; this is worth carrying into M2-I7's probe P5
   and M2-I1's AC-3 tie-flip handling as a concrete, reproduced example rather than a theoretical
   concern. Filed to memory (see below).

### Running it

```bash
python3 validate_self_consistency.py --dump-dir <dumps-dir> --mode {prefill,decode,both}
```
Exit code 0 iff every non-skipped check passes. Requires `torch` (CPU-only is fine except that
the CUDA-preferring `topk` tie-break workaround above is skipped on CPU-only boxes — expect
occasional benign `topk_ids` tie-order mismatches there; the value-gather fallback still catches
real bugs).

## What's committed here vs what stays on B200

Full raw dumps are **482 MB** (361 MB prefill + 120 MB decode; 594 + 196 tensors), dominated by
per-expert FP8 weight slices (426 of 594 prefill tensors are weights). Per this issue's own
instruction ("large raw dumps may stay on B200... only the validator report + small dumps
committed"):

- **Committed here** (`workspace/demo/qwen3_5/oracle/`): `ref_dump.py`, `pytorch_reference.py`,
  `validate_self_consistency.py`, this README, and `dumps_sample/` — the FULL `manifest.json` for
  both modes (shape/dtype/mean/std/min/max/nan/inf for every one of the 594+196 tensors, ~340 KB
  total, no raw tensor data), `runtime_diagnostics.json`, `validator_transcript.txt` (the full
  PASS transcript), and a curated set of ~10 small (<70 KB) illustrative activation tensors
  (`gdn.beta`, `gdn.decay_g`, `attn.rope_cos/sin`, `moe0.topk_ids`, `moe0.topk_renorm_weights`,
  `moe0.router_logits`, `attn.core_attn_out`, `gdn.gated_norm_out`) so a reviewer can inspect real
  values without B200 access.
- **Stays on B200** (`~/mpk-qwen35/oracle-work/dumps/`): the full tensor payload (all 790
  `.pt` files, weights + activations), reproducible on demand via the regeneration command above.
  The kernel unit-test issues (M2-I4 onward) run on B200 anyway and can read this directly.

## Known limitations / open items

- **Probe input is small by design** (8-token prefill chunk of `p01-history` + 1 decode step) —
  sized for a fast, cheap, repeatable oracle, not a stress test. If a future kernel test needs a
  longer sequence or a different prompt, rerun with `--prefill-tokens`/`--prompt-id`; the script
  is parameterized for this.
- **`FP8Experts`'s internal per-expert accumulate/rounding is unverified** beyond the
  dequantized-weight LOOSE_FP8 check (same epistemic status as vLLM's FlashInfer MoE cubin per
  `vllm-graph.md` §2.3.4 — "not observable from this repo").
- **CONTRACT ADDENDUM NOT ACTIONED THIS SESSION**: `.pm/issues/M2/M2-I3.md` picked up a
  coordinator-added addendum mid-flight (extend `accept/reference/generate_reference.py` to
  persist per-step top-k ids+logits and regenerate `reference_outputs.json`, for AC-3's tie-flip
  waiver). That file is outside this issue's original `OWNS PATHS`
  (`workspace/demo/qwen3_5/oracle/` only) and touching+regenerating a load-bearing AC-3 artifact
  under a different issue's ownership without an explicit re-brief seemed like the wrong call to
  make unilaterally mid-task — flagged for the coordinator instead of silently expanding scope.
