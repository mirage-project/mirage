# Qwen3.8-27B: MPK vs stock SGLang (exploratory)

This follows the GLM experiment's separate-engine, fixed-batch design. It runs
B1, B4, B8 and B16 on **one B200 at a time** with 128 input IDs and 64 generated
IDs per request. Each backend/batch receives **one timed generation**, with no
warmup, repeat, average, or one-token subtraction. Model loading, MPK compilation
and SGLang startup are outside the timer; the whole synchronous generation
request (prefill plus decode) is inside it. First-call initialization may be
included. This is a quick diagnostic, not a stable throughput estimate.

The input-ID file is prepared once with the checkpoint tokenizer and passed to
both backends. B8 MPK now uses one `offline` runtime call for tokenwise prefill
and decoding, with request metadata and planner-owned KV pages. B1/B4/B16 still
use PyTorch block prefill and fixed-batch MPK decode. Stock SGLang uses TP1,
BF16 KV, greedy sampling, no speculative decoding,
radix cache disabled, and its normal CUDA-graph path. Both ignore EOS and must
return 64 IDs per row. The SGLang worker runs without Mirage's GLM hook variables.
The B8 offline graph is still gated pending broader context validation. A
same-B200 HF comparison with 128 input IDs and 16 generated IDs matched
128/128 tokens and all 64 final decoder hidden states. Set
`MPK_QWEN_EXPERIMENTAL_B8=1` when running this benchmark's B8 offline case;
its token check must pass before using the timing as a speed comparison.

Use the two already-installed Python environments on Catalyst. The script scans
all B200s and selects one with at most 10 GiB memory and 5% reported compute
utilization for each batch pair; `--gpu GPU-...` pins a specific B200. It checks
again before each backend and stops if that GPU becomes busy. Run only one copy
of this script at a time. During each backend process it samples the selected
GPU's compute and graphics process IDs through NVML about every 0.2 seconds in
`*-gpu-monitor.jsonl`. The summary requires samples before, during and after
the timed call, with no outside process and no gap over 0.5 seconds. Sampling
cannot exclude interference shorter than the sampling interval; a publishable
comparison needs several paired samples in a later experiment.

Before any model loads, a separate short-lived process checks that the chosen
GPU can initialize CUDA, allocate one tensor and synchronize. Auto-selection
tries other idle B200s if this fails (a low-memory/0%-utilization GPU can still
be unavailable). The probe is never part of the timed generation. An explicit
`--gpu` UUID or index fails early with a probe log if CUDA is unavailable.

```bash
/path/to/mpk-env/bin/python benchmark/qwen38_sglang/compare.py \
  --model /path/to/Qwen3.8-27B \
  --mpk-python /path/to/mpk-env/bin/python \
  --sglang-python /path/to/sglang-env/bin/python \
  --sglang-cuda-home /usr/local/cuda-13.0 \
  --output-dir /path/to/new-results-directory
```

Optional `--checkpoint-manifest` records the manifest digest. Optional
`--compiled-root` accepts validated fixed-step MPK compile runs. B8 offline
compiles into `compiled-offline-b8` when no compiled graph is supplied;
explicit B8 `--compiled-root` or `--mpk-compiled-dir` selects the earlier
fixed-step path. For a compiled
graph made on another B200, add `--allow-other-b200-compiled` only after checking
that its source and configuration match.
`--sglang-cuda-home` changes `CUDA_HOME` and prepends its `bin` directory to
`PATH` only for SGLang; MPK retains its own CUDA toolkit environment.

`summary.json` contains a single-run wall time and aggregate output tokens/s
for each backend, the raw SGLang/MPK latency ratio, full generated IDs, first
token mismatch, GPU identity, versions, source commit/status and log locations.
The top-level status distinguishes fully valid cases from completed runs with
invalid cases; the latter return a nonzero exit code while retaining all raw
reports.
`mpk_vs_sglang_speedup` is populated only when both runs use the same inputs,
checkpoint and GPU, generated IDs are exactly equal, and the timed GPU window
has monitor coverage without external GPU processes. A token mismatch or an
observed external GPU process retains the raw timings but is **not** a valid
speedup claim. One JSON and log file per
backend/batch preserve the evidence. Neither script sends data or pushes Git.

## Four-backend fixed-batch decode check

`compare_decode_three.py` compares MPK, native HF, stock SGLang and vLLM for
the Qwen3.8-27B BF16 checkpoint at B4/B8/B16/B32. The filename predates the
HF addition. Prepare one shared token-ID file:

```bash
/path/to/mpk-python benchmark/qwen38_sglang/worker.py --backend prepare \
  --model /path/to/Qwen3.8-27B --output /path/to/inputs.json \
  --batch-sizes 4 8 16 32 --prompt-tokens 128 --new-tokens 64
```

Run each backend in a separate Python environment on the **same single B200**,
passing the same `--model` and `--inputs` paths and a distinct `--output` path.
Ensure the backend environment's `bin` directory and CUDA's `bin` directory
are on `PATH` for JIT compilation. Repeat the following commands after setting
`B` to each value in `4 8 16 32`, starting the next backend only after the
previous process exits:

```bash
B=4
CUDA_VISIBLE_DEVICES=2 /path/to/mpk-python \
  benchmark/qwen38_sglang/compare_decode_three.py --backend mpk --batch "$B" \
  --model /path/to/Qwen3.8-27B --inputs /path/to/inputs.json \
  --output "/path/to/mpk-b${B}.json"
CUDA_VISIBLE_DEVICES=2 /path/to/hf-python \
  benchmark/qwen38_sglang/compare_decode_three.py --backend hf --batch "$B" \
  --model /path/to/Qwen3.8-27B --inputs /path/to/inputs.json \
  --output "/path/to/hf-b${B}.json"
CUDA_VISIBLE_DEVICES=2 /path/to/sglang-python \
  benchmark/qwen38_sglang/compare_decode_three.py --backend sglang --batch "$B" \
  --model /path/to/Qwen3.8-27B --inputs /path/to/inputs.json \
  --output "/path/to/sglang-b${B}.json" --memory-fraction 0.6
CUDA_VISIBLE_DEVICES=2 /path/to/vllm-python \
  benchmark/qwen38_sglang/compare_decode_three.py --backend vllm --batch "$B" \
  --model /path/to/Qwen3.8-27B --inputs /path/to/inputs.json \
  --output "/path/to/vllm-b${B}.json" --memory-fraction 0.6
```

The default is one warmup and three timed generations per backend. Each timing
covers the 63 decode intervals after the first token; model load, graph
compilation, engine startup and prefill are excluded. MPK times host-observed
fixed-batch steps and token reads. HF times host-observed eager BF16 cached
forwards and token reads. SGLang and vLLM time streamed token arrival at the
client, requiring one-token first events. These measurements have different
delivery overhead and are exploratory. Compare all output IDs with the
same-prompt HF output before calling a latency ratio a valid speedup. The
result file records individual trials, output IDs, versions, timing scope and
repeat-output mismatches. B4/B32 use the generic MPK fixed-batch graph; B8/B16
use the optimized row/column-partitioned graph.
