#!/usr/bin/env python3
"""M1-I6: vLLM decode-throughput / e2e-latency benchmark for Qwen/Qwen3.5-35B-A3B-FP8.

Implements the measurement contract pinned in `workspace/docs/qwen35/bench-protocol.md`.
Uses vLLM's offline Python `LLM` engine (not the HTTP server) so timing is not polluted by
HTTP/serialization overhead and so we get direct access to `vllm_config` for fairness
introspection and to each `RequestOutput.metrics` for the prefill/decode timestamp split.

Two modes:
  --mode sweep   The pinned protocol: one persistent engine, loops over --batch-sizes,
                 each with 1 warmup + >=3 measured reps of the fixed input/output-len
                 workload. Writes one JSON per batch size + a summary.json.
  --mode ruling  Tiny single-engine-instance run (bs in {1,4} x 64 output tokens, 1 rep,
                 no warmup) used TWICE (once with --language-model-only on, once off) to
                 decide whether the fused QK-norm+RoPE+gate kernel is safe to pin into the
                 sweep config. Compares output token ids externally (see bench-protocol.md
                 Sec 4) - this script only *produces* the artifact for one side of that A/B.

Regeneration (sweep, on catalyst-B200, GPU already picked free + CUDA_VISIBLE_DEVICES pinned):
    python bench_vllm.py --mode sweep \
        --output-dir ~/mpk-qwen35/bench_vllm_run/vllm-0.25.1-<date> \
        --log-file   ~/mpk-qwen35/logs/bench_vllm_sweep.log \
        --language-model-only on

See README/bench-protocol.md for the full rationale behind every knob below; this docstring
only covers *how to run it*, not *why*.
"""
import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

MODEL_ID_DEFAULT = "Qwen/Qwen3.5-35B-A3B-FP8"
REVISION_DEFAULT = "9d1823d2dee688a6b25e77009dc727688c44936e"

# Substrings that MUST appear in the captured combined stdout+stderr log for the run to be
# considered "vLLM's best standard FP8 config" per vllm-graph.md Sec 3.7.5. Matched with a
# plain substring search (not regex) against the whole log file content.
REQUIRED_LOG_MARKERS = {
    "deepgemm_autodisabled": "Auto-disabled DeepGemm",
    "deepgemm_fallback_cutlass": "Falling back to CUTLASS",
    "dense_kernel_cutlass": "Selected CutlassFp8BlockScaledMMKernel",
    "moe_backend_flashinfer_trtllm": "Using FLASHINFER_TRTLLM Fp8 MoE backend",
}

# Env vars that must NOT be set to a truthy value - they would silently change which fp8
# kernel runs (Marlin -> different precision entirely) or disable batching-dependent kernels
# (batch-invariant mode swaps out the fast MoE/attention paths). See vllm-graph.md Sec 3.7.4.
DANGEROUS_ENV_VARS = ["VLLM_TEST_FORCE_FP8_MARLIN", "VLLM_BATCH_INVARIANT"]

# Informational-only env vars: recorded verbatim in every artifact, never asserted, because
# their *defaults* are already what we want and a non-default value should be visible, not
# silently masked - see bench-protocol.md Sec 7 for why these are "record" not "assert".
INFORMATIONAL_ENV_VARS = [
    "VLLM_USE_DEEP_GEMM",
    "VLLM_MOE_USE_DEEP_GEMM",
    "VLLM_DISABLED_KERNELS",
    "VLLM_USE_DEEP_GEMM_E8M0",
]


def log(msg: str) -> None:
    print(f"[bench_vllm] {msg}", flush=True)


# --------------------------------------------------------------------------- GPU etiquette --

def get_gpu_id() -> int:
    """The single physical GPU index this process is pinned to, from CUDA_VISIBLE_DEVICES.
    Hard-fails if not exactly one GPU is visible - exclusivity is a precondition, not a
    best-effort (resources.md: 'measurements are only valid with ZERO co-tenants')."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not raw or "," in raw:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must name exactly one physical GPU for a valid benchmark "
            f"run, got {raw!r}. Pin one free GPU before invoking this script."
        )
    return int(raw.strip())


def _run(cmd: list) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if p.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(cmd)}\nstderr: {p.stderr}")
    return p.stdout


def nvidia_smi_csv(fields: list, gpu_id: int) -> dict:
    out = _run([
        "nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits",
        "-i", str(gpu_id),
    ])
    values = [v.strip() for v in out.strip().split(",")]
    return dict(zip(fields, values))


def nvidia_smi_compute_pids(gpu_id: int) -> dict:
    """pid -> {used_memory_mib, process_name} for compute processes on this GPU. Empty dict
    if the GPU is idle (not an error - nvidia-smi exits 0 with empty output in that case)."""
    out = _run([
        "nvidia-smi",
        "--query-compute-apps=pid,used_memory,process_name",
        "--format=csv,noheader,nounits",
        "-i", str(gpu_id),
    ])
    pids = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        pid, mem, name = parts[0], parts[1], ",".join(parts[2:])
        try:
            pids[int(pid)] = {"used_memory_mib": int(mem), "process_name": name}
        except ValueError:
            continue
    return pids


def nvidia_smi_clocks_raw(gpu_id: int) -> str:
    return _run(["nvidia-smi", "-q", "-d", "CLOCK", "-i", str(gpu_id)])


def parse_current_clocks(raw: str) -> dict:
    """Best-effort parse of the FIRST 'Clocks' block (current clocks, not 'Max Clocks' /
    'Applications Clocks' which reuse the same field names later in the same report)."""
    result = {"graphics_mhz": None, "sm_mhz": None, "memory_mhz": None, "video_mhz": None}
    lines = raw.splitlines()
    in_block = False
    for line in lines:
        stripped = line.strip()
        if stripped == "Clocks":
            in_block = True
            continue
        if in_block:
            if stripped.endswith("Clocks") or (stripped and ":" not in line and not line.startswith(" " * 8)):
                # left the first "Clocks" block (next section header)
                if stripped != "Clocks":
                    break
            m = re.match(r"(Graphics|SM|Memory|Video)\s*:\s*(\d+)\s*MHz", stripped)
            if m:
                key = {"Graphics": "graphics_mhz", "SM": "sm_mhz", "Memory": "memory_mhz",
                       "Video": "video_mhz"}[m.group(1)]
                result[key] = int(m.group(2))
    return result


def preflight_gpu_check(gpu_id: int, max_mem_mib: int = 500, max_util_pct: int = 1) -> dict:
    info = nvidia_smi_csv(["memory.used", "utilization.gpu"], gpu_id)
    mem_used = int(info["memory.used"])
    util = int(info["utilization.gpu"])
    pids = nvidia_smi_compute_pids(gpu_id)
    if mem_used > max_mem_mib or util > max_util_pct or pids:
        raise RuntimeError(
            f"GPU {gpu_id} is NOT exclusively free (resources.md etiquette): "
            f"memory.used={mem_used}MiB util={util}% compute_pids={pids}. Pick a different "
            f"GPU or wait - never run a benchmark rep on a shared GPU."
        )
    log(f"preflight OK: GPU {gpu_id} memory.used={mem_used}MiB util={util}% pids={pids}")
    return {"memory_used_mib": mem_used, "utilization_pct": util}


def cotenant_check(gpu_id: int, baseline_pids: set) -> dict:
    """Returns {'ok': bool, 'extra_pids': {...}} - 'extra' = any compute-app pid on this GPU
    that was NOT part of our own engine's baseline snapshot (taken right after engine boot)."""
    current = nvidia_smi_compute_pids(gpu_id)
    extra = {pid: v for pid, v in current.items() if pid not in baseline_pids}
    return {"ok": len(extra) == 0, "extra_pids": extra, "current_pids": current}


# ------------------------------------------------------------------------- synthetic prompts --

def build_synthetic_prompts(tokenizer, batch_size: int, input_len: int, seed: int):
    """`batch_size` DISTINCT prompts, each EXACTLY `input_len` real-vocabulary token ids,
    sampled uniformly at random (fixed seed => fully reproducible). Random content is
    deliberate and standard for a pure throughput benchmark (vLLM's own
    benchmark_throughput.py --dataset-name random does the same): FP8 GEMM and the
    FlashInfer TRT-LLM MoE kernel do fixed-shape work independent of token identity, so
    content carries no signal here, only length does. tokenizer.vocab_size (248044 for this
    checkpoint) already excludes every special-token id (all >= 248044), so no extra
    filtering is required - see bench-protocol.md Sec 3."""
    import random as _random
    from vllm import TokensPrompt

    rng = _random.Random(seed)
    vocab_n = tokenizer.vocab_size
    prompts = []
    for _ in range(batch_size):
        ids = [rng.randrange(0, vocab_n) for _ in range(input_len)]
        prompts.append(TokensPrompt(prompt_token_ids=ids))
    return prompts


# --------------------------------------------------------------------------- fp8 assertions --

def collect_env_snapshot() -> dict:
    return {name: os.environ.get(name) for name in DANGEROUS_ENV_VARS + INFORMATIONAL_ENV_VARS}


def assert_dangerous_env_vars_absent(env_snapshot: dict) -> None:
    for name in DANGEROUS_ENV_VARS:
        val = env_snapshot.get(name)
        truthy = val is not None and val.strip().lower() not in ("", "0", "false", "no")
        if truthy:
            raise RuntimeError(
                f"FAIRNESS VIOLATION: {name}={val!r} is set to a truthy value. This is a "
                f"documented dangerous knob (vllm-graph.md Sec 3.7.4) that changes which fp8 "
                f"kernel/precision runs. Unset it before benchmarking."
            )


def check_required_log_markers(log_text: str) -> dict:
    return {key: (substr in log_text) for key, substr in REQUIRED_LOG_MARKERS.items()}


def collect_engine_assertions(llm, log_text: str) -> dict:
    """Hard-fail fp8-path + fairness introspection, run ONCE per engine instance right after
    construction (kernel selection is static at create_weights() time - vllm-graph.md
    Sec 3.5 - so it cannot change across the batch-size sweep on one engine)."""
    from vllm.utils.flashinfer import has_flashinfer, has_flashinfer_trtllm_fused_moe

    vc = llm.llm_engine.vllm_config
    model_config = vc.model_config
    parallel_config = vc.parallel_config
    cache_config = vc.cache_config

    quantization = model_config.quantization
    has_fi = has_flashinfer()
    has_fi_trtllm_moe = has_flashinfer_trtllm_fused_moe()
    log_markers = check_required_log_markers(log_text)

    try:
        import flashinfer
        flashinfer_version = flashinfer.__version__
    except Exception as e:  # noqa: BLE001
        flashinfer_version = f"IMPORT FAILED: {e}"

    env_snapshot = collect_env_snapshot()

    assertions = {
        "quantization": quantization,
        "quantization_is_fp8": quantization == "fp8",
        "has_flashinfer": has_fi,
        "has_flashinfer_trtllm_fused_moe": has_fi_trtllm_moe,
        "flashinfer_version": flashinfer_version,
        "log_markers": log_markers,
        "tensor_parallel_size": parallel_config.tensor_parallel_size,
        "pipeline_parallel_size": parallel_config.pipeline_parallel_size,
        "data_parallel_size": parallel_config.data_parallel_size,
        "lora_enabled": vc.lora_config is not None,
        "kv_cache_dtype": cache_config.cache_dtype,
        "mamba_cache_dtype": getattr(cache_config, "mamba_cache_dtype", None),
        "mamba_ssm_cache_dtype": getattr(cache_config, "mamba_ssm_cache_dtype", None),
        "enable_prefix_caching": cache_config.enable_prefix_caching,
        "moe_backend": str(getattr(vc.kernel_config, "moe_backend", None)),
        "linear_backend": str(getattr(vc.kernel_config, "linear_backend", None)),
        "enforce_eager": model_config.enforce_eager,
        "max_num_batched_tokens": vc.scheduler_config.max_num_batched_tokens,
        "language_model_only": getattr(vc.model_config.multimodal_config, "language_model_only", None)
        if vc.model_config.multimodal_config is not None else None,
        "env_snapshot": env_snapshot,
    }

    hard_checks = {
        "quantization_is_fp8": assertions["quantization_is_fp8"],
        "has_flashinfer": has_fi,
        "has_flashinfer_trtllm_fused_moe": has_fi_trtllm_moe,
        "log_deepgemm_autodisabled": log_markers["deepgemm_autodisabled"] and log_markers["deepgemm_fallback_cutlass"],
        "log_dense_kernel_cutlass": log_markers["dense_kernel_cutlass"],
        "log_moe_backend_flashinfer_trtllm": log_markers["moe_backend_flashinfer_trtllm"],
        "tensor_parallel_size_is_1": parallel_config.tensor_parallel_size == 1,
        "pipeline_parallel_size_is_1": parallel_config.pipeline_parallel_size == 1,
        "data_parallel_size_is_1": parallel_config.data_parallel_size == 1,
        "no_lora": vc.lora_config is None,
    }
    assert_dangerous_env_vars_absent(env_snapshot)  # raises on violation

    failed = [k for k, v in hard_checks.items() if not v]
    assertions["hard_checks"] = hard_checks
    assertions["all_hard_checks_passed"] = len(failed) == 0
    if failed:
        raise RuntimeError(
            f"FAIRNESS/FP8-PATH ASSERTION FAILURE: {failed} did not pass. This run is NOT a "
            f"valid 'vLLM best standard FP8 config' baseline - see assertions dict for detail:\n"
            f"{json.dumps(assertions, indent=2, default=str)}"
        )
    log(f"engine assertions OK: {json.dumps(hard_checks)}")
    return assertions


# ---------------------------------------------------------------------------------- one rep --

def run_one_rep(llm, prompts, output_len: int, sampling_seed: int):
    from vllm import SamplingParams

    sp = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        seed=sampling_seed,
        max_tokens=output_len,
        min_tokens=output_len,
        ignore_eos=True,
    )
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
    t1 = time.perf_counter()
    e2e_wall_seconds = t1 - t0

    bs = len(prompts)
    per_request = []
    for out in outputs:
        m = out.metrics
        if m is None:
            raise RuntimeError(
                "RequestOutput.metrics is None - LLM(...) must be constructed with "
                "disable_log_stats=False for this script's timing methodology to work."
            )
        completion = out.outputs[0]
        per_request.append({
            "scheduled_ts": m.scheduled_ts,
            "queued_ts": m.queued_ts,
            "first_token_ts": m.first_token_ts,
            "last_token_ts": m.last_token_ts,
            "num_generation_tokens": m.num_generation_tokens,
            "num_cached_tokens": out.num_cached_tokens,
            "finish_reason": completion.finish_reason,
            "output_len_actual": len(completion.token_ids),
        })

    # --- validity checks: every request must have generated EXACTLY output_len tokens via
    # the max_tokens cap (finish_reason == "length"), and hit ZERO prefix-cache reuse (our
    # prompts are fresh-random per (bs, rep) by construction - see build_synthetic_prompts).
    bad_len = [r for r in per_request if r["output_len_actual"] != output_len]
    bad_finish = [r for r in per_request if r["finish_reason"] != "length"]
    bad_cache = [r for r in per_request if (r["num_cached_tokens"] or 0) != 0]
    if bad_len or bad_finish or bad_cache:
        raise RuntimeError(
            f"rep validity check failed: bad_len={len(bad_len)} bad_finish={len(bad_finish)} "
            f"bad_cache={len(bad_cache)} out of {bs} requests. First offender: "
            f"{(bad_len or bad_finish or bad_cache)[0]}"
        )

    decode_window_start = max(r["first_token_ts"] for r in per_request)
    decode_window_end = min(r["last_token_ts"] for r in per_request)
    decode_wall_seconds = decode_window_end - decode_window_start
    decode_tokens = bs * (output_len - 1)
    decode_tokens_per_second = decode_tokens / decode_wall_seconds if decode_wall_seconds > 0 else float("nan")

    inference_times = [r["last_token_ts"] - r["scheduled_ts"] for r in per_request]

    return {
        "e2e_wall_seconds": e2e_wall_seconds,
        "decode_window_start": decode_window_start,
        "decode_window_end": decode_window_end,
        "decode_wall_seconds": decode_wall_seconds,
        "decode_tokens": decode_tokens,
        "decode_tokens_per_second": decode_tokens_per_second,
        "per_request_inference_time_median_s": statistics.median(inference_times),
        "per_request_inference_time_min_s": min(inference_times),
        "per_request_inference_time_max_s": max(inference_times),
        "batch_size": bs,
        "output_ids_sha256": hashlib.sha256(
            json.dumps([list(o.outputs[0].token_ids) for o in outputs]).encode()
        ).hexdigest(),
        "sample_output_ids_request0": list(outputs[0].outputs[0].token_ids),
    }


def summarize(values: list) -> dict:
    med = statistics.median(values)
    lo, hi = min(values), max(values)
    dispersion_pct = ((hi - lo) / med * 100.0) if med else float("nan")
    return {
        "median": med, "min": lo, "max": hi,
        "spread_half_range": (hi - lo) / 2.0,
        "dispersion_pct": dispersion_pct,
        "n": len(values),
    }


# -------------------------------------------------------------------------------- sweep mode --

def run_batch_size(llm, tokenizer, gpu_id: int, engine_baseline_pids: set, bs: int,
                    input_len: int, output_len: int, reps: int, warmup_reps: int,
                    seed_base: int, sampling_seed: int, max_dispersion_pct: float,
                    max_extra_attempts: int = 3) -> dict:
    log(f"=== batch_size={bs} input_len={input_len} output_len={output_len} "
        f"warmup_reps={warmup_reps} reps={reps} ===")

    clocks_before = parse_current_clocks(nvidia_smi_clocks_raw(gpu_id))

    cot = cotenant_check(gpu_id, engine_baseline_pids)
    if not cot["ok"]:
        raise RuntimeError(f"co-tenant present on GPU {gpu_id} before batch_size={bs}: {cot['extra_pids']}")

    warmup_records = []
    for w in range(warmup_reps):
        seed = seed_base + bs * 1000 + w
        prompts = build_synthetic_prompts(tokenizer, bs, input_len, seed)
        rec = run_one_rep(llm, prompts, output_len, sampling_seed)
        rec["seed"] = seed
        rec["role"] = "warmup"
        warmup_records.append(rec)
        log(f"  warmup[{w}]: decode_tps={rec['decode_tokens_per_second']:.2f} "
            f"e2e={rec['e2e_wall_seconds']:.2f}s")

    valid_reps = []
    discarded_reps = []
    attempt = 0
    rep_idx = 0
    while len(valid_reps) < reps and attempt < reps + max_extra_attempts:
        attempt += 1
        pre_check = cotenant_check(gpu_id, engine_baseline_pids)
        seed = seed_base + bs * 1000 + warmup_reps + rep_idx
        prompts = build_synthetic_prompts(tokenizer, bs, input_len, seed)
        rec = run_one_rep(llm, prompts, output_len, sampling_seed)
        post_check = cotenant_check(gpu_id, engine_baseline_pids)
        rec["seed"] = seed
        rec["role"] = "measured"
        rec["co_tenant_pre"] = pre_check
        rec["co_tenant_post"] = post_check
        rec_ok = pre_check["ok"] and post_check["ok"]
        rec["valid"] = rec_ok
        rep_idx += 1
        if rec_ok:
            valid_reps.append(rec)
            log(f"  rep[{len(valid_reps) - 1}]: decode_tps={rec['decode_tokens_per_second']:.2f} "
                f"e2e={rec['e2e_wall_seconds']:.2f}s")
        else:
            discarded_reps.append(rec)
            log(f"  rep DISCARDED (co-tenant detected): pre={pre_check} post={post_check}")

    if len(valid_reps) < reps:
        raise RuntimeError(
            f"batch_size={bs}: only {len(valid_reps)}/{reps} valid (co-tenant-free) reps "
            f"after {attempt} attempts. Discarded: {discarded_reps}"
        )

    clocks_after = parse_current_clocks(nvidia_smi_clocks_raw(gpu_id))

    decode_tps_summary = summarize([r["decode_tokens_per_second"] for r in valid_reps])
    e2e_summary = summarize([r["e2e_wall_seconds"] for r in valid_reps])
    dispersion_ok = decode_tps_summary["dispersion_pct"] <= max_dispersion_pct
    if not dispersion_ok:
        log(f"  ** DISPERSION WARNING: decode_tps dispersion "
            f"{decode_tps_summary['dispersion_pct']:.2f}% exceeds bound {max_dispersion_pct}% - "
            f"do not treat this batch size's median as final without investigating "
            f"(clocks/co-tenant/thermal - see bench-protocol.md Sec 6).")

    return {
        "batch_size": bs,
        "input_len": input_len,
        "output_len": output_len,
        "gpu_clocks_before": clocks_before,
        "gpu_clocks_after": clocks_after,
        "warmup": warmup_records,
        "reps": valid_reps,
        "discarded_reps": discarded_reps,
        "summary": {
            "decode_tokens_per_second": decode_tps_summary,
            "e2e_latency_seconds": e2e_summary,
            "max_dispersion_pct_bound": max_dispersion_pct,
            "dispersion_ok": dispersion_ok,
        },
    }


def run_sweep(args) -> None:
    from vllm import LLM

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_id = get_gpu_id()
    preflight = preflight_gpu_check(gpu_id)

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    lmo = args.language_model_only == "on"

    log(f"constructing LLM engine: model={args.model_id} revision={args.revision} "
        f"language_model_only={lmo} gpu_memory_utilization={args.gpu_memory_utilization} "
        f"max_model_len={args.max_model_len}")
    t_load0 = time.time()
    llm = LLM(
        model=args.model_id,
        revision=args.revision,
        dtype="auto",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        disable_log_stats=False,       # REQUIRED: populates RequestOutput.metrics
        language_model_only=lmo,
        seed=0,
    )
    load_seconds = time.time() - t_load0
    log(f"engine constructed in {load_seconds:.1f}s")

    # Read back our own combined stdout+stderr log (redirected by the caller's wrapper shell
    # script to args.log_file - see scripts/run_bench_vllm.sh) to build the log-based fp8
    # assertions. Flush first so everything written so far is actually on disk.
    sys.stdout.flush()
    sys.stderr.flush()
    log_text = ""
    if args.log_file and Path(args.log_file).exists():
        log_text = Path(args.log_file).read_text(errors="replace")
    else:
        log(f"WARNING: --log-file {args.log_file!r} not found yet; log-based assertions "
            f"(DeepGemm/CUTLASS/FlashInfer-TRTLLM markers) will show as False. The wrapper "
            f"script must redirect combined stdout+stderr to this exact path.")

    engine_assertions = collect_engine_assertions(llm, log_text)  # raises on hard failure

    engine_baseline_pids = set(nvidia_smi_compute_pids(gpu_id).keys())
    log(f"engine baseline GPU pids (ours): {engine_baseline_pids}")

    tokenizer = llm.get_tokenizer()

    shared_meta = {
        "schema_version": "1.0",
        "run_tag": args.run_tag,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": args.model_id,
        "revision": args.revision,
        "gpu_id_physical": gpu_id,
        "gpu_preflight": preflight,
        "engine_load_seconds": round(load_seconds, 1),
        "engine_assertions": engine_assertions,
        "versions": collect_versions(),
        "cli_args": vars(args),
    }

    bs_results = {}
    for bs in batch_sizes:
        result = run_batch_size(
            llm, tokenizer, gpu_id, engine_baseline_pids, bs,
            args.input_len, args.output_len, args.reps, args.warmup_reps,
            args.prompt_seed_base, args.sampling_seed, args.max_dispersion_pct,
        )
        result["shared_meta"] = shared_meta
        out_path = out_dir / f"bs{bs}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log(f"wrote {out_path}")
        bs_results[bs] = result

    summary = {
        "shared_meta": shared_meta,
        "batch_sizes": batch_sizes,
        "table": {
            str(bs): {
                "decode_tokens_per_second": bs_results[bs]["summary"]["decode_tokens_per_second"],
                "e2e_latency_seconds": bs_results[bs]["summary"]["e2e_latency_seconds"],
                "dispersion_ok": bs_results[bs]["summary"]["dispersion_ok"],
            }
            for bs in batch_sizes
        },
    }
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"wrote {summary_path}")
    log("SWEEP COMPLETE")


# ------------------------------------------------------------------------------- ruling mode --

def run_ruling(args) -> None:
    """One engine instance, --language-model-only as given; tiny fixed-content generations at
    a couple of batch sizes. Run this twice (on/off) and diff the two output JSONs' token ids
    externally - see bench-protocol.md Sec 4."""
    from vllm import LLM

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_id = get_gpu_id()
    preflight_gpu_check(gpu_id)

    lmo = args.language_model_only == "on"
    log(f"[ruling] constructing engine with language_model_only={lmo}")
    llm = LLM(
        model=args.model_id,
        revision=args.revision,
        dtype="auto",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        disable_log_stats=False,
        language_model_only=lmo,
        seed=0,
    )
    tokenizer = llm.get_tokenizer()

    from vllm import SamplingParams

    ruling_batch_sizes = [int(x) for x in args.ruling_batch_sizes.split(",")]
    results = {}
    for bs in ruling_batch_sizes:
        prompts = build_synthetic_prompts(tokenizer, bs, args.input_len, args.prompt_seed_base + bs)
        sp = SamplingParams(
            temperature=0.0, top_p=1.0, seed=args.sampling_seed,
            max_tokens=args.ruling_output_len, min_tokens=args.ruling_output_len, ignore_eos=True,
        )
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params=sp, use_tqdm=False)
        elapsed = time.perf_counter() - t0
        bad = [o for o in outputs if len(o.outputs[0].token_ids) != args.ruling_output_len
               or o.outputs[0].finish_reason != "length"]
        if bad:
            raise RuntimeError(f"[ruling] bs={bs}: {len(bad)} request(s) did not generate exactly "
                                f"{args.ruling_output_len} tokens via max_tokens - {bad[0]}")
        results[str(bs)] = {
            "output_ids": [list(o.outputs[0].token_ids) for o in outputs],
            "approx_tokens_per_second": (bs * args.ruling_output_len) / elapsed if elapsed > 0 else None,
        }
        log(f"[ruling] bs={bs}: generated, approx_tok/s={results[str(bs)]['approx_tokens_per_second']}")

    out = {
        "language_model_only": lmo,
        "ruling_batch_sizes": ruling_batch_sizes,
        "ruling_output_len": args.ruling_output_len,
        "prompt_seed_base": args.prompt_seed_base,
        "results": results,
    }
    out_path = out_dir / f"ruling_lmo_{'on' if lmo else 'off'}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log(f"wrote {out_path}")


def collect_versions() -> dict:
    import torch
    import transformers
    import vllm
    versions = {
        "python": sys.version,
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "transformers": transformers.__version__,
    }
    try:
        import flashinfer
        versions["flashinfer"] = flashinfer.__version__
    except Exception as e:  # noqa: BLE001
        versions["flashinfer"] = f"IMPORT FAILED: {e}"
    return versions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["sweep", "ruling"], required=True)
    ap.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    ap.add_argument("--revision", default=REVISION_DEFAULT)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--log-file", default=None,
                     help="Path the CALLER's wrapper script redirects combined stdout+stderr "
                          "to (required for log-based fp8 assertions to see subprocess lines).")
    ap.add_argument("--run-tag", default="vllm-run")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--language-model-only", choices=["on", "off"], required=True)
    ap.add_argument("--sampling-seed", type=int, default=0)
    ap.add_argument("--prompt-seed-base", type=int, default=20260725)

    # sweep mode - input_len=256/output_len=1024 is the PINNED workload (bench-protocol.md
    # Sec 3): I <~ O/4 is required for MPK's chunked in-kernel prefill to keep AC-5 (e2e <=
    # 1.25x vLLM) closeable at B=16 (workspace/docs/qwen35/v1-architecture.md Sec 8.2 - the
    # cost model there derives prefill_iters = ceil(B*I/mbt) each costing ~1 decode-equivalent
    # step, so overhead = 4*I/O at B=16,mbt=16; 256/1024 keeps that at the same 25% bound as
    # the doc's own I=64/O=256 floor recommendation, scaled up 4x for a more realistic length
    # and a longer, lower-noise steady-state decode window).
    ap.add_argument("--batch-sizes", default="1,2,4,8,16")
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=None,
                     help="defaults to input-len + output-len if unset")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--warmup-reps", type=int, default=1)
    ap.add_argument("--max-dispersion-pct", type=float, default=5.0)

    # ruling mode
    ap.add_argument("--ruling-batch-sizes", default="1,4")
    ap.add_argument("--ruling-output-len", type=int, default=64)

    args = ap.parse_args()
    if args.max_model_len is None:
        args.max_model_len = args.input_len + args.output_len

    log(f"args: {vars(args)}")

    if args.mode == "sweep":
        run_sweep(args)
    else:
        run_ruling(args)


if __name__ == "__main__":
    main()
