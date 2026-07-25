# Qwen3.5-35B-A3B-FP8 — HF `transformers` correctness reference (AC-3)

This directory holds the load-bearing correctness reference for M1-I5 / AC-3: greedy
(`do_sample=False`) decode of 64 new tokens for each of the 10 pinned prompts in
`.pm/eval/prompts.jsonl`, produced by HF `transformers` running the FP8 checkpoint
as-shipped (no re-quantization, no dequant "fix"). MPK's own output must exactly match
`output_ids` in `reference_outputs.json` for every prompt (AC-3 waiver rule in
`.pm/goal.md` covers only a documented numeric-precision tie-flip, not implementation bugs).

## Checkpoint

- Model: `Qwen/Qwen3.5-35B-A3B-FP8`
- Revision (pinned sha, NOT `main`): `9d1823d2dee688a6b25e77009dc727688c44936e`
- Quantization recipe (from the checkpoint's own `config.json["quantization_config"]`):
  `quant_method=fp8`, `activation_scheme=dynamic`, `weight_block_size=[128,128]`,
  per-tensor flags both `false` (i.e. genuinely block-wise, not per-tensor). bf16
  `modules_to_not_convert` includes `lm_head`, `embed_tokens`, every layer's GDN
  `conv1d`/`in_proj_a`/`in_proj_b`, every layer's MoE router `gate` and
  `shared_expert_gate`, the entire vision tower (`model.visual.*`), and the MTP head
  (`mtp.*`). We never re-quantize or calibrate differently (constraint.md 2a).
- Architecture: the checkpoint's `config.json["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]`
  (a VLM wrapper; AC-2 scopes vision out — see "Text-only usage" below), but transformers
  5.14.1 *also* registers a plain-text `Qwen3_5MoeForCausalLM` under `AutoModelForCausalLM`,
  which is what actually loaded this reference (`reference_outputs.json["meta"]["load_notes"]
  ["loaded_via"] == "AutoModelForCausalLM"`) — no VLM wrapper or `trust_remote_code` needed;
  the checkpoint ships no `modeling_*.py`, confirming native library support. The checkpoint
  also ships MTP (`mtp.*`) weights; standard `generate()` does not exercise them, so this
  reference is a plain autoregressive base-model decode with no speculative/MTP path,
  matching AC-2.

## EOS handling

- `model.generation_config.eos_token_id` (the value `generate()` actually honors) =
  `[248046, 248044]` = `["<|im_end|>", "<|endoftext|>"]`. Note this differs from the bare
  `config.json["text_config"]["eos_token_id"] = 248044` alone — `generation_config.json`
  is what's authoritative at generate() time and includes both.
- The checkpoint's shipped `generation_config.json` defaults to **sampling**
  (`do_sample=true, temperature=1.0, top_k=20, top_p=0.95`). We explicitly pass
  `do_sample=False, temperature=None, top_p=None, top_k=None, num_beams=1` — do not rely
  on the checkpoint defaults.
- We do **not** force `min_new_tokens`. `max_new_tokens=64` is a cap; if EOS is hit
  earlier, generation stops there (standard decode semantics, matching what a real
  serving engine — including MPK's own harness, which also breaks on
  `eos_token_id` — would do). Each prompt's entry in `reference_outputs.json` records
  `num_generated`, `hit_eos`, and `eos_step` explicitly so a shorter-than-64 sequence is
  never mistaken for a truncation bug.
- Every returned token was double-checked to be the true greedy argmax of its own
  step's logits (`generate_reference.py` asserts `top1_ids == output_ids`); this catches
  any accidental logits-warper interference from the sampling-oriented default config.

**Reading the outputs:** several prompts share an identical opening few tokens (e.g. all of
p01/p06/p07/p09 start with the ids for `"Thinking Process:\n\n1.  **Analyze the
Request:**"`, and several others start with `"Here's a thinking process that leads
to..."`). This is the model's own fixed reasoning-preamble style under greedy decoding —
not a bug — every response is verified topically correct for its own prompt immediately
after the shared opener (checked by hand against `decoded_output` for this artifact).

## Text-only usage of a VLM-wrapper checkpoint (AC-2)

No prompt in the pinned set includes images. We call the model's standard `.generate()`
with `input_ids`/`attention_mask` only (no `pixel_values`) — the vision tower is loaded
(bf16, per `modules_to_not_convert`) but never exercised. `generate_reference.py` tries
`AutoModelForCausalLM` first and falls back to `AutoModelForImageTextToText` / the direct
`Qwen3_5MoeForConditionalGeneration` class; `reference_outputs.json["meta"]["load_notes"]`
records which one actually loaded the checkpoint on this transformers version.

## FP8 execution path

**Weights stay natively FP8 at rest, and compute uses real FP8 GEMM kernels — not a
dequant-to-bf16 fallback.** `reference_outputs.json["meta"]["fp8_execution_introspection"]
["parameter_dtype_histogram"]` for this run: `{"torch.bfloat16": 633, "torch.float32": 60,
"torch.float8_e4m3fn": 330}` — 330 parameters are still literally `torch.float8_e4m3fn`
after load (`any_native_float8_storage: true`); the 633 bf16 ones are exactly the
`modules_to_not_convert` set, and the 60 fp32 ones are norm/scale buffers.

Compute goes through transformers' `integrations.finegrained_fp8` dispatcher, which loads
real fp8 GEMM kernels **dynamically from the HF `kernels` hub** (pip package `kernels`,
pinned `0.15.2 <= v < 0.16.0` — not a default dependency of either transformers or vllm,
had to be installed explicitly: `uv pip install "kernels>=0.15.2,<0.16.0"`). Two backends
exist:
1. **DeepGEMM** (preferred; downloads ~1447 files from `kernels-community/deep-gemm`,
   "3-6x faster" per transformers' own docstring). **Crashes on this B200 (SM100)** on the
   GDN `in_proj_qkv` fp8 linear: `RuntimeError: Assertion error
   (.../layout.hpp:71): sfb_dtype == torch::kFloat or sfb_dtype == torch::kInt`.
   transformers' source (`integrations/finegrained_fp8.py`, around `fp8_linear`) already
   documents this class of problem: *"a still-unexplained DeepGEMM-vs-Triton interaction
   that degrades end-to-end generation on B200 (per-row kernel outputs still measure
   bit-perfect, but final tokens drift; not reproducible with the DeepGEMM linear off)"*.
2. **Triton `finegrained-fp8`** fallback (~12 files from `kernels-community/finegrained-fp8`).
   Forced via the env var transformers itself ships for exactly this situation:
   `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1`. **This is what produced
   `reference_outputs.json`** — `generate_reference.py` sets it as a default
   (`os.environ.setdefault`, so it's on out of the box on B200 but stays overridable).

Why this matters beyond this one artifact: MPK's own B200 kernel design (M2+) should not
assume DeepGEMM-style fp8 scale handling is safe on SM100 without checking against this
exact failure mode, and if DeepGEMM is ever fixed upstream and produces *different* tokens
than this Triton-backed reference, that's a numerics question to root-cause
(first-principles) before treating either side as wrong. See `generation_run.log` for the
full raw traceback from the DeepGEMM crash and the fp8 kernel hub-download output.

## Chat template

`tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)` —
the model's own template (ships as both `tokenizer_config.json["chat_template"]` and a
sibling `chat_template.jinja`; we use whichever `AutoTokenizer` resolves, i.e. the
checkpoint's template, not a hand-written one). `bos_token` is `None` for this
tokenizer (nothing is prepended); `pad_token` = `<|endoftext|>`.

## Files

- `generate_reference.py` — self-contained regeneration script (see docstring for the
  exact CLI). Reads `.pm/eval/prompts.jsonl` read-only; never modifies it.
- `reference_outputs.json` — the artifact: `meta` (versions, revision, EOS/template
  identity, FP8 introspection, load notes) + `results[<prompt id>]` = `input_ids`,
  `output_ids`, `num_generated`, `hit_eos`, `eos_step`, `top1_logit_per_step`,
  `decoded_output`, `elapsed_seconds`.
- `generation_run.log` — full stdout/stderr captured while producing the artifact above
  (includes the raw HF/torch load-time warnings referenced above).
- `vllm_smoke/` — step 8: `vllm_smoke.py` (the script that produced this evidence),
  `vllm_smoke_result.json` (quantization method, load/generate timings, tokens/s,
  output ids, match result), `vllm_smoke_run.log` (full raw stdout/stderr).

## Versions

| Component | At `reference_outputs.json` generation (step 7) | At vLLM smoke (step 8) |
|---|---|---|
| torch | `2.13.0+cu130` | `2.11.0+cu130` (see note below) |
| transformers | `5.14.1` | `5.14.1` |
| accelerate | `1.14.0` | `1.14.0` |
| vllm | n/a | `0.25.1` |
| torchvision | `0.28.0+cu130` | `0.26.0+cu130` |
| kernels | `0.15.2` | n/a (vLLM uses its own CUTLASS/Triton kernels, not the HF `kernels` hub) |
| Python | `3.12.3` | `3.12.3` |
| GPU driver / CUDA (nvidia-smi header) | `595.58.03` / CUDA 13.2 | same |
| nvcc (MPK build only, unrelated to this reference) | `12.8.93` (`/usr/local/cuda-12.8`) | — |
| cutlass submodule commit (MPK build only) | `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8` (matches resources.md's pin) | — |

**Why torch differs between the two steps, and why that's fine:** all of this lives in one
venv, `~/mpk-qwen35/venv-vllm`. `vllm==0.25.1`'s bundled flash-attention `.so` extensions
are compiled against torch `2.11.0`'s C++ ABI; between step 7 and step 8 a `transformers`/
`accelerate` install had (independently, via normal dependency resolution) bumped the
shared venv's torch to `2.13.0`, which left those already-built `.so` files with undefined
libtorch symbols (`OSError: undefined symbol: ...TensorBase14const_data_ptr...`) — the same
"a later install in the same venv silently staled an earlier compiled extension" pattern
that also hit `torchvision` in this venv and, separately, mirage's `core.so`/`z3-solver`
in `venv-mpk` (see the M1-I5 memory-inbox entry, not duplicated here). Fixed by
`uv pip install --reinstall-package vllm vllm`, which uv resolved back down to vllm's
natural pairing (torch `2.11.0`). This does **not** invalidate `reference_outputs.json` —
that artifact was already written to disk under torch `2.13.0` and is immutable; its own
`meta.versions` block is the authoritative record for step 7. It only means a *fresh*
re-run of `generate_reference.py` from a clean checkout today would pick up whatever torch
the venv resolves to at that time (currently `2.11.0`), which should still exercise the
same FP8 code path (dynamically fetched `kernels` build matching whatever torch/cuda is
active) — not expected to change the output tokens, but not re-verified under `2.11.0`
since the already-validated artifact didn't need regenerating.

## Exact commands run (on `catalyst-B200`, user `muhengl`)

```bash
# GPU etiquette: pick a GPU with ~0% util and <500MiB used, then pin it.
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

export CUDA_VISIBLE_DEVICES=<free_gpu_id>
export HF_HOME=~/mpk-qwen35/hf
source ~/mpk-qwen35/venv-vllm/bin/activate   # has transformers/accelerate installed
python ~/mpk-qwen35/reference_run/generate_reference.py \
  --model-id Qwen/Qwen3.5-35B-A3B-FP8 \
  --revision 9d1823d2dee688a6b25e77009dc727688c44936e \
  --prompts-file ~/mpk-qwen35/reference_run/prompts.jsonl \
  --output-dir ~/mpk-qwen35/reference_run \
  --max-new-tokens 64
```

```bash
# vLLM smoke (step 8) - same GPU-etiquette recheck as above.
export CUDA_VISIBLE_DEVICES=<free_gpu_id>
export HF_HOME=~/mpk-qwen35/hf
source ~/mpk-qwen35/venv-vllm/bin/activate
python ~/mpk-qwen35/vllm_smoke.py \
  --model-id Qwen/Qwen3.5-35B-A3B-FP8 \
  --revision 9d1823d2dee688a6b25e77009dc727688c44936e \
  --prompt-id p01-history \
  --prompts-file ~/mpk-qwen35/prompts_readonly_copy.jsonl \
  --reference-json ~/mpk-qwen35/reference_run/reference_outputs.json \
  --max-tokens 64 \
  --output-json ~/mpk-qwen35/vllm_smoke_run/vllm_smoke_result.json
```

Canonical (repo-relative) re-run of the HF reference once this directory is checked out
inside the repo:

```bash
python workspace/demo/qwen3_5/accept/reference/generate_reference.py
# (defaults resolve --prompts-file to .pm/eval/prompts.jsonl and --output-dir to
#  this directory)
```

## vLLM smoke (step 8)

**Result: vLLM's 64 output token ids for `p01-history` are byte-identical to the HF
reference** (`vllm_smoke/vllm_smoke_result.json["matches_hf_reference"] = true`) — two
independent inference engines, same FP8 checkpoint, exact greedy agreement on this prompt.
This is informational corroboration, not the AC-3 gate itself (AC-3 is specifically
MPK-vs-HF); still a strong sanity signal that the HF reference is not some FP8-loading
artifact idiosyncratic to `transformers`.

- vLLM version: `0.25.1`.
- Quantization confirmed active via the engine's own config object (not just log-scraping):
  `llm.llm_engine.model_config.quantization == "fp8"`. Startup logs also show
  `quantization=fp8` in the full engine config dump, and — independent corroboration of
  the step-7 DeepGEMM finding — vLLM itself logs, unprompted: *"Auto-disabled DeepGemm for
  model_type=qwen3_5_moe_text on Blackwell. DeepGemm E8M0 scale format causes accuracy
  degradation for this architecture. Falling back to CUTLASS."* No dequant-to-bf16
  fallback warning of any kind appears.
- **Startup cost dominated by one-time JIT compilation**, not a hang: `init engine
  (profile, create kv cache, warmup model) took 900.27s` — almost all of it
  FlashInfer's first-use `ptxas`/`nvcc` compile of its fused-MoE TRT-LLM kernel for
  SM100a (~713s alone, confirmed via direct process-tree inspection: sustained 100% CPU
  in `ptxas`, not idle) plus a smaller `sampling` kernel and CUDA graph capture (51
  sizes, 21s). This is a **one-time cache cost** — `~/.cache/flashinfer/` and
  `~/.cache/vllm/torch_compile_cache/` persist it for subsequent runs on this box.
- Generation itself, once warm: 64 tokens in 2.14s → **29.88 tokens/s** (single request,
  batch=1). This is a smoke number only, **not** the AC-4 throughput benchmark protocol
  (that needs the fixed workload/batch-size sweep with an otherwise-idle GPU, per M4).
- Full raw log: `vllm_smoke/vllm_smoke_run.log`. Result JSON:
  `vllm_smoke/vllm_smoke_result.json`.
