#!/usr/bin/env python3
"""M3-I10: per-kernel decode profiling of vLLM 0.25.1 at the PINNED baseline config.

Reuses ~/mpk-qwen35/bench_vllm.py verbatim for engine construction semantics, prompt
construction, the fp8/fairness assertions and the decode-window timing definition, so the
profiled engine is byte-for-byte the binding baseline identity (CUTLASS block-scale dense +
FlashInfer TRT-LLM fp8 MoE, lmo=off, KV block 1056, GDN state fp32).

Difference vs bench_vllm.py: VLLM_ENABLE_V1_MULTIPROCESSING=0 so the EngineCore runs
IN-PROCESS.  That is required for a step-driven torch profiler schedule (the profiler must
live in the process that launches the kernels).  It changes no kernel selection and no shape;
the unprofiled decode tok/s reps in this same script are the check on that.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

# the REPO copy of bench_vllm.py (binding-contract version), rsynced next to this script.
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH_DIR)
import bench_vllm as BV  # noqa: E402


def log(m):
    print(f"[m3i10] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", default="1,16,8")
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--log-file", required=True)
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=1024)
    ap.add_argument("--skip-first", type=int, default=300)
    ap.add_argument("--wait", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--active", type=int, default=50)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--profiled-gens", type=int, default=2)
    ap.add_argument("--timed-reps", type=int, default=3)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--prompt-seed-base", type=int, default=20260725)
    ap.add_argument("--sampling-seed", type=int, default=0)
    ap.add_argument("--model-id", default=BV.MODEL_ID_DEFAULT)
    ap.add_argument("--revision", default=BV.REVISION_DEFAULT)
    ap.add_argument("--prefill-trace", action="store_true")
    args = ap.parse_args()

    max_model_len = args.input_len + args.output_len
    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.out_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from torch.profiler import ProfilerActivity, profile, schedule
    from vllm import LLM
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    gpu_id = BV.get_gpu_id()
    preflight = BV.preflight_gpu_check(gpu_id)

    log(f"constructing LLM engine (IN-PROCESS EngineCore): model={args.model_id} "
        f"revision={args.revision} lmo=False gmu={args.gpu_memory_utilization} "
        f"max_model_len={max_model_len}")
    t0 = time.time()
    llm = LLM(
        model=args.model_id,
        revision=args.revision,
        dtype="auto",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        disable_log_stats=False,
        language_model_only=False,
        seed=0,
    )
    load_seconds = time.time() - t0
    log(f"engine constructed in {load_seconds:.1f}s")

    sys.stdout.flush()
    sys.stderr.flush()
    log_text = Path(args.log_file).read_text(errors="replace") if Path(args.log_file).exists() else ""
    engine_assertions = BV.collect_engine_assertions(llm, log_text)  # raises on identity failure
    log("BINDING IDENTITY CHECK PASSED (same hard checks as the baseline capture)")

    engine_baseline_pids = set(BV.nvidia_smi_compute_pids(gpu_id).keys())
    tokenizer = llm.get_tokenizer()

    # ---- step-driven profiler hook -------------------------------------------------------
    STATE = {"prof": None, "count": 0}
    orig_exec = GPUModelRunner.execute_model

    def patched_execute_model(self, *a, **kw):
        out = orig_exec(self, *a, **kw)
        p = STATE["prof"]
        if p is not None:
            STATE["count"] += 1
            p.step()
        return out

    GPUModelRunner.execute_model = patched_execute_model
    log("patched GPUModelRunner.execute_model for step-driven profiling")

    def run_profiled(bs, prompts, tag, sched, handler):
        STATE["count"] = 0
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=sched,
            on_trace_ready=handler,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
        )
        STATE["prof"] = prof
        prof.start()
        try:
            rec = BV.run_one_rep(llm, prompts, args.output_len, args.sampling_seed)
        finally:
            prof.stop()
            STATE["prof"] = None
        log(f"  {tag}: engine steps seen={STATE['count']} decode_tps={rec['decode_tokens_per_second']:.2f}")
        return rec

    results = {}
    for bs in [int(x) for x in args.batch_sizes.split(",")]:
        log(f"================ batch_size={bs} ================")
        cot = BV.cotenant_check(gpu_id, engine_baseline_pids)
        if not cot["ok"]:
            raise RuntimeError(f"co-tenant on GPU {gpu_id}: {cot['extra_pids']}")
        clocks_before = BV.parse_current_clocks(BV.nvidia_smi_clocks_raw(gpu_id))

        # warmup (unprofiled)
        wseed = args.prompt_seed_base + bs * 1000 + 900
        wrec = BV.run_one_rep(llm, BV.build_synthetic_prompts(tokenizer, bs, args.input_len, wseed),
                              args.output_len, args.sampling_seed)
        log(f"  warmup: decode_tps={wrec['decode_tokens_per_second']:.2f}")

        # unprofiled timed reps -> identity/overhead check vs the binding baseline
        unprof = []
        for r in range(args.timed_reps):
            seed = args.prompt_seed_base + bs * 1000 + r
            rec = BV.run_one_rep(llm, BV.build_synthetic_prompts(tokenizer, bs, args.input_len, seed),
                                 args.output_len, args.sampling_seed)
            rec["seed"] = seed
            unprof.append(rec)
            log(f"  unprofiled rep{r}: decode_tps={rec['decode_tokens_per_second']:.2f} "
                f"e2e={rec['e2e_wall_seconds']:.2f}s")

        # profiled generates
        win_idx = {"k": 0}

        def handler(p, _bs=bs, _wi=win_idx):
            k = _wi["k"]
            _wi["k"] += 1
            path = trace_dir / f"decode_bs{_bs}_win{k}.json"
            p.export_chrome_trace(str(path))
            log(f"  exported {path.name} ({path.stat().st_size / 1e6:.1f} MB)")

        sched = schedule(skip_first=args.skip_first, wait=args.wait,
                         warmup=args.warmup, active=args.active, repeat=args.repeat)
        prof_recs = []
        for g in range(args.profiled_gens):
            seed = args.prompt_seed_base + bs * 1000 + 500 + g
            prompts = BV.build_synthetic_prompts(tokenizer, bs, args.input_len, seed)
            rec = run_profiled(bs, prompts, f"profiled gen{g}", sched, handler)
            rec["seed"] = seed
            prof_recs.append(rec)

        # bonus: prefill window (steps 0..5 of a fresh generate)
        if args.prefill_trace:
            def pf_handler(p, _bs=bs):
                path = trace_dir / f"prefill_bs{_bs}_win0.json"
                p.export_chrome_trace(str(path))
                log(f"  exported {path.name} ({path.stat().st_size / 1e6:.1f} MB)")

            pseed = args.prompt_seed_base + bs * 1000 + 700
            run_profiled(bs, BV.build_synthetic_prompts(tokenizer, bs, args.input_len, pseed),
                         "prefill", schedule(wait=0, warmup=0, active=8, repeat=1), pf_handler)

        clocks_after = BV.parse_current_clocks(BV.nvidia_smi_clocks_raw(gpu_id))
        tps = [r["decode_tokens_per_second"] for r in unprof]
        results[str(bs)] = {
            "batch_size": bs,
            "unprofiled_decode_tps": BV.summarize(tps),
            "unprofiled_reps": unprof,
            "profiled_reps": [{k: v for k, v in r.items() if k != "sample_output_ids_request0"}
                              for r in prof_recs],
            "warmup": {k: v for k, v in wrec.items() if k != "sample_output_ids_request0"},
            "n_trace_windows": win_idx["k"],
            "clocks_before": clocks_before,
            "clocks_after": clocks_after,
            "cotenant_after": BV.cotenant_check(gpu_id, engine_baseline_pids),
        }
        log(f"  bs{bs} unprofiled median decode_tps={statistics.median(tps):.2f}")

    meta = {
        "schema_version": "1.0",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": args.model_id,
        "revision": args.revision,
        "gpu_id_physical": gpu_id,
        "gpu_preflight": preflight,
        "engine_load_seconds": round(load_seconds, 1),
        "engine_assertions": engine_assertions,
        "versions": BV.collect_versions(),
        "cli_args": vars(args),
        "vllm_enable_v1_multiprocessing": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
        "results": results,
    }
    p = out_dir / "profile_meta.json"
    p.write_text(json.dumps(meta, indent=2, default=str))
    log(f"wrote {p}")
    log("PROFILE RUN COMPLETE")


if __name__ == "__main__":
    main()
