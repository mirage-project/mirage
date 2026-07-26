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
- `gpu_etiquette_evidence.md` — verbatim pre-run `nvidia-smi` rows + `CUDA_VISIBLE_DEVICES`
  pinning, quoted from the actual B200 log files, for every GPU-using step (HF reference,
  vLLM smoke, and the optional qwen3-8B demo smoke — including an honest note on the one
  step whose log lacks an embedded `nvidia-smi` call).

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

## Full environment rebuild (from nothing, on `catalyst-B200`, user `muhengl`)

Everything below reproduces `~/mpk-qwen35` from an empty home directory through to being
able to run the two command blocks in the next section. GPU etiquette
(`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv`, pick
`<500 MiB` + `~0%` util, pin `CUDA_VISIBLE_DEVICES`, recheck immediately before every
GPU-touching command) applies from step 6 onward; see `gpu_etiquette_evidence.md` in this
directory for the actual verbatim pre-run `nvidia-smi` rows this run used.

### 1. Layout + uv (user-local)

```bash
mkdir -p ~/mpk-qwen35/hf ~/mpk-qwen35/logs ~/mpk-qwen35/scripts
curl -LsSf https://astral.sh/uv/install.sh | sh        # installs to ~/.local/bin
export PATH=$HOME/.local/bin:$PATH
```

**`UV_CACHE_DIR` cross-filesystem caveat (discovered here, applies to any fresh rebuild):**
`uv`'s wheel cache is meant to let multiple venvs *hardlink* shared downloads (e.g. torch)
instead of each paying the full size again. On this box `~/.cache/uv` was already a
symlink to `/tmp/muhengl_cache_relocate/uv` (root fs `/dev/md0`) — separate from
`~/mpk-qwen35` on `/raid` (`/dev/md1`). Because a hardlink cannot cross filesystems, every
`uv pip install` here printed `Failed to hardlink ... Invalid cross-device link (os error
18); falling back to copy` and silently paid the full copy cost per venv instead of
deduping. It still works, just uses more disk. If a fresh rebuild wants the real
dedup benefit, set `export UV_CACHE_DIR=~/mpk-qwen35/.uv-cache` *before* creating the
venvs so the cache and the venvs share one filesystem. (If instead you want to keep large
downloads off the tight `/raid` pool the way this run ended up doing, leave the cache on
`/tmp`/root-fs and accept the no-hardlink cost — root fs had ~268G free vs. `/raid`'s
tighter shared pool; either choice is legitimate, just know which one you're making.)

### 2. Both venvs

```bash
cd ~/mpk-qwen35
uv venv venv-mpk --python 3.12
uv venv venv-vllm --python 3.12
```
(uv-created venvs ship **no `pip` binary** — use `uv pip install ...` for everything below,
not `pip install` / `python -m pip`, even inside an activated venv.)

### 3. Clone mirage + submodules, pinned revisions

```bash
cd ~/mpk-qwen35
git clone -b qwen3-5_support git@github.com:bill810975/mirage.git mirage   # https fallback if no ssh key:
# git clone -b qwen3-5_support https://github.com/bill810975/mirage.git mirage
cd mirage
git submodule update --init --recursive   # deps/cutlass, deps/z3, deps/json (all https)
```
Revisions this run actually built against — verify a fresh clone lands on the same commits
(they're what the pinned branch/submodule refs resolve to, not extra manual checkouts):
mirage `qwen3-5_support` HEAD = `2c87a75` ("Support DFlash for Kimi-K2.6 (#728)");
`deps/cutlass` = `f3fde58372d33e9a5650ba7b80fc48b3b49d40c8` (matches resources.md's pin).

### 4. MPK editable build (CUDA 12.8) + the z3-solver ABI-drift fix

```bash
export PATH=$HOME/.local/bin:/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
cd ~/mpk-qwen35/mirage
source ~/mpk-qwen35/venv-mpk/bin/activate
uv pip install -e . -v          # full log captured to logs/build_mpk.log on this run
```

`import mirage` will then fail: `ImportError: libz3.so.5.0: cannot open shared object
file`. Root cause: `pyproject.toml`'s `[build-system] requires` lists unpinned
`"z3-solver"`; the **isolated PEP517 build env** resolved `z3-solver 5.0.0.0` (so
`core.cpython-312-x86_64-linux-gnu.so` got linked against `libz3.so.5.0`), but the
**main venv install** independently resolved `z3-solver==4.16.0.0` (only ships
`libz3.so.4.16`) — a build-time-vs-runtime dependency drift, not a broken build. Fix:
confirm what the compiled extension actually needs, then install exactly that version and
re-verify with a real import (not just "uv says installed"):

```bash
ldd ~/mpk-qwen35/mirage/python/mirage/core.cpython-312-x86_64-linux-gnu.so | grep z3
# -> libz3.so.5.0 => not found
uv pip install "z3-solver==5.0.0.0"
python3 -c "import mirage; print('mirage import OK')"   # smoke-import verification
```

### 5. venv-vllm: pinned installs + the recurring ABI-drift reinstall rule

```bash
source ~/mpk-qwen35/venv-vllm/bin/activate
uv pip install -U "huggingface_hub[cli,hf_transfer]"   # -> huggingface_hub 1.24.0
                                                         # (extras warnings are harmless;
                                                         #  the `hf` CLI ships in the base
                                                         #  package on this version)
uv pip install -U vllm                                  # -> vllm 0.25.1
uv pip install -U transformers accelerate                # -> transformers 5.14.1, accelerate 1.14.0
uv pip install "kernels>=0.15.2,<0.16.0"                  # -> kernels 0.15.2 (transformers'
                                                         #  native fp8 GEMM kernels need this;
                                                         #  not pulled in by default)
```

**Rule, root-caused twice in this one venv:** installing packages into a venv across more
than one `uv pip install` call can leave an *already-installed* compiled/ABI-sensitive
package silently stale the moment a *later* call bumps a shared dependency (here: torch)
that the earlier package was built against — `uv` only re-resolves what the current
command's graph touches, not everything already installed. **After all installs into a
venv are done, smoke-import every heavy compiled dependency; if one fails with an
`undefined symbol` / `cannot open shared object` error, reinstall exactly that package
(not torch itself) and re-verify:**

```bash
# hit #1: torchvision was resolved against an earlier torch (~2.11.0); a later
# `transformers accelerate` install bumped torch to 2.13.0 without re-resolving it.
python3 -c "import torchvision"
# -> RuntimeError: operator torchvision::nms does not exist
uv pip install --reinstall-package torchvision torchvision   # -> torchvision 0.28.0 (now matches torch 2.13.0)
python3 -c "import torchvision; print('torchvision import OK')"   # smoke-import verification

# hit #2: vllm's OWN bundled flash-attention .so files were compiled against torch
# 2.11.0's C++ ABI; they broke the same way once torch moved to 2.13.0.
python3 -c "
import torch
torch.ops.load_library('$HOME/mpk-qwen35/venv-vllm/lib/python3.12/site-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so')
"
# -> OSError: undefined symbol: _ZNK2at10TensorBase14const_data_ptrIiLi0EEEPKT_v
uv pip install --reinstall-package vllm vllm
# -> uv resolves the WHOLE stack back down to vllm's natural pairing: torch 2.11.0,
#    torchvision 0.26.0, triton 3.6.0, etc. (not always "forward to the newer shared dep" -
#    sometimes the correct fix is "the newer package wants the older dependency back").
python3 -c "
import torch
torch.ops.load_library('$HOME/mpk-qwen35/venv-vllm/lib/python3.12/site-packages/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so')
print('vllm flash-attn .so load OK')
"   # smoke-import verification
```

Final venv-vllm state after both fixes: `vllm==0.25.1`, `transformers==5.14.1`,
`accelerate==1.14.0`, `kernels==0.15.2`, `torch==2.11.0+cu130`,
`torchvision==0.26.0+cu130` — this is the state that produced the step-8 vLLM smoke
result below (torch differs from the `2.13.0` that step 7's HF reference used; see the
Versions table above for why that's fine).

### 6. ferret clone (small, read-only reference for M3+ kernel-opt tooling)

```bash
cd ~/mpk-qwen35
git clone -b cc git@github.com:xinhaoc/ferret.git ferret
```

### 7. Checkpoint download (resume-capable, disk-headroom-checked)

```bash
export HF_HOME=~/mpk-qwen35/hf
source ~/mpk-qwen35/venv-vllm/bin/activate
df -h /raid                                              # headroom check BEFORE
hf download Qwen/Qwen3.5-35B-A3B-FP8 \
  --revision 9d1823d2dee688a6b25e77009dc727688c44936e    # ~37.5GB, 14 safetensors shards
df -h /raid                                              # headroom check AFTER — keep >=10G free;
                                                          # STOP and reassess if projected free
                                                          # would drop below that
```
`hf download` resumes partial blob downloads natively — safe to re-run the identical
command if interrupted. This run monitored `/raid` before/after every step above too
(build, both venv installs); it went from 283G to 228G free over the whole rebuild+run,
never approaching the 10G floor.

## Exact commands run for the reference artifacts (steps 7–8 of the issue)

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

## Regeneration / reproducibility check (provenance)

| Run | Date | GPU | Script sha256 | Result |
|---|---|---|---|---|
| Original | 2026-07-25 03:14–03:17 EDT | `catalyst-B200` CUDA_VISIBLE_DEVICES=1 (4 MiB/0%) | `852d74cc...` (script has since been edited, see below) | Produced the first `reference_outputs.json` — but that script version predated the `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR` default + `transformers_disable_deepgemm_linear` meta field added afterward, leaving the committed artifact's schema stale relative to the script. |
| **Regeneration** | **2026-07-25 04:06–04:09 EDT** | `catalyst-B200` `CUDA_VISIBLE_DEVICES=2` (4 MiB/0%) | **`852d74ccc6a294dd08d65bf4e60d95adc642ad18f7c8c2c20e2a609ca817f063`** — verified byte-identical to the committed `generate_reference.py` (sha256 checked on both the local repo copy and the B200 copy immediately before this run; the B200 copy had independently drifted by one dead-code line from an earlier sync and was re-synced first) | **Re-ran the exact current script**, same env (`venv-vllm`, `HF_HOME=~/mpk-qwen35/hf`, `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1`, `--revision 9d1823d2dee688a6b25e77009dc727688c44936e`), output to a scratch dir first. **Diffed all 10 prompts × 64 tokens against the existing committed artifact: 100% identical, input_ids and output_ids both, for every prompt.** Only then replaced `reference_outputs.json` and `generation_run.log` with this run's pair — now schema-synced (`transformers_disable_deepgemm_linear: "1"` present) and reproducibility-proven. See `gpu_etiquette_evidence.md` → "Step 7 regeneration" for the verbatim pre-run `nvidia-smi` rows + pinning. |
| **M2-I3 addendum regeneration** | **2026-07-25 ~20:22–20:28 EDT** | `catalyst-B200` `CUDA_VISIBLE_DEVICES=5` (4 MiB/0%), locked via `~/mpk-qwen35/.gpu-locks/M2-I3.lock` | **`aad2469871bbead72fcc0405a91b433c041b4090ce37330b7608cbb87310cfb5`** — verified byte-identical between the local repo copy and the B200 standalone copy (`~/mpk-qwen35/generate_reference.py`) immediately before this run | **Extended the script** to persist per-step top-k token ids + logits (`--topk-logits 4`, new keys below) — the AC-3 tie-flip waiver was structurally unsubstantiable without this (see `../harness/README.md` "Known gap", now updated). **First attempt used `torch.topk(...)[0]` for top1 and hit a real `AssertionError` on `p06-poem`**: `torch.max` and `torch.topk` broke an exact top-logit tie differently on the same tensor (same risk class M2-I3's oracle work independently found in the MoE router's `torch.topk`) — root-caused, not papered over: fixed by keeping top1 derivation on the ORIGINAL unchanged `torch.max` call and computing the additional top-k separately (with a defensive tie-reconciliation assertion), so this script's identity-critical behavior is unchanged from the prior regeneration. Re-ran clean (exit 0, all 10 prompts), output to a scratch dir (`~/mpk-qwen35/regen_addendum/`) first. **Diffed all 10 prompts' `input_ids` + `output_ids` + `num_generated` + `hit_eos`/`eos_step` against the previously-committed artifact programmatically: 100% identical for every prompt** — only `reference_outputs.json`/`generation_run.log` were then replaced. |
| **M2-I3 addendum, tie-reconciliation fix** | **2026-07-25 ~20:36–20:38 EDT** | `catalyst-B200` `CUDA_VISIBLE_DEVICES=5` (4–5 MiB/0%), locked via `~/mpk-qwen35/.gpu-locks/M2-I3.lock` | **`ec16e7650179e2dfef63e2a8137c3fbd902077ad3c45d8956d99ab355fd91daa`** — verified byte-identical between the local repo copy and the B200 standalone copy immediately before this run | **Review of the row above caught a real defect**: the OVERWRITE reconciliation silently duplicated an id whenever the torch.max-verified top1 was already present at a LATER slot of the topk window — concretely, `p06-poem` step 56 came out `[288, 288, 75635, 91491]` (slot 0 overwritten with 288, but 288 already occupied slot 1; the genuinely distinct tied alternative, 71439, was dropped). **Fixed by SWAP instead of overwrite**: find the verified top1's existing position in the topk window (`ids_row.index(top1_id)`, asserted to exist and to share the exact logit value), swap it into slot 0, and let the displaced id keep its (now slot-0-adjacent) position — plus a new blanket assertion `len(set(topk_ids)) == k` for every one of the 640 positions across all 10 prompts. Re-ran clean (exit 0). `p06-poem` step 56 now reads `topk_ids_per_step=[288, 71439, 75635, 91491]`, `topk_logits_per_step=[18.5, 18.5, 18.25, 18.0]` — verified top1 (288) at slot 0, the real distinct tied alternative (71439) at slot 1 with an EQUAL logit (margin 0.0, correctly reflecting a genuine tie rather than a fabricated distinct margin). **Diffed all 10 prompts' `input_ids`/`output_ids`/`num_generated`/`hit_eos`/`eos_step` against the prior committed artifact: 100% identical** (2 genuine exact top1/top2 ties found across the full 640 positions, both now correctly represented as two distinct ids with equal logits, not a duplicate). |

Notable side detail: this regeneration ran under `torch==2.11.0+cu130` (the venv's current
state after the step-8 vllm-flash-attn ABI fix), vs. the original's `torch==2.13.0+cu130` —
i.e. the token-for-token match also held across two different torch/fp8-kernel-build
combinations on this box, not just a literal re-run of identical binaries. The M2-I3 addendum
work above extends this further: token-for-token identity now holds across FOUR separate runs
of this script (four different processes across two GPUs, three different sha256 script
versions differing only in additive top-k persistence and its tie-reconciliation logic) —
input/output tokens never moved once across any of it; only the per-step top-k metadata (and
the correctness of ITS OWN internal reconciliation) changed and was fixed in place.

## Schema addendum (M2-I3): per-step top-k logits

As of the regeneration above, each prompt's entry in `reference_outputs.json["results"]` carries
four new keys alongside the original `top1_logit_per_step`:

- `top2_id_per_step` / `top2_logit_per_step` — the exact optional keys
  `../harness/reference_loader.py` checks for (`_OPTIONAL_TOP2_ID_KEYS`/`_OPTIONAL_TOP2_LOGIT_KEYS`);
  populated for every position of every prompt now, so `margin_evidence.available` is `true`
  against this artifact (previously always `false` — see the harness README's former "Known gap").
- `topk_ids_per_step` / `topk_logits_per_step` — the full top-`k` (`meta.topk_logits_k`, `k=4` by
  default) per step, one list of `k` values per position. `topk_ids_per_step[step][0]` is
  guaranteed to equal `output_ids[step]` (the actually-generated token) even in the rare case of
  an exact top-logit tie — see the provenance row above for why that guarantee needed an explicit
  fix rather than a bare `torch.topk` call.
- `meta.topk_logits_k` — the `k` used for this run (regenerate with `--topk-logits N` to change
  it; the harness only ever needs `k>=2`).
