#!/usr/bin/env python3
"""M3-I10 part D: timeboxed SGLang feasibility probe for Qwen/Qwen3.5-35B-A3B-FP8 on one B200.

Success = boots, serves the FP8 checkpoint at the pinned 256/1024 greedy workload, and yields a
per-kernel decode table comparable to the vLLM one.  Failure = record the exact blocker.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
import traceback
from pathlib import Path

MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"
REV = "9d1823d2dee688a6b25e77009dc727688c44936e"


def log(m):
    print(f"[sglang-probe] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--mem-fraction-static", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile-dir", default=None)
    ap.add_argument("--profile-steps", type=int, default=50)
    args = ap.parse_args()

    rec = {"stage": "start", "model": MODEL, "revision": REV, "args": vars(args),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    def dump():
        outp.write_text(json.dumps(rec, indent=2, default=str))

    try:
        import sglang as sgl
        import torch
        rec["sglang_version"] = sgl.__version__
        rec["torch_version"] = torch.__version__
        log(f"sglang {sgl.__version__} torch {torch.__version__}")
        dump()

        if args.profile_dir:
            os.environ["SGLANG_TORCH_PROFILER_DIR"] = args.profile_dir
            Path(args.profile_dir).mkdir(parents=True, exist_ok=True)

        rec["stage"] = "engine_construct"
        dump()
        t0 = time.time()
        eng = sgl.Engine(
            model_path=MODEL,
            revision=REV,
            tp_size=1,
            mem_fraction_static=args.mem_fraction_static,
            context_length=args.input_len + args.output_len,
            random_seed=0,
            log_level="info",
        )
        rec["engine_load_seconds"] = round(time.time() - t0, 1)
        rec["stage"] = "engine_up"
        log(f"engine up in {rec['engine_load_seconds']}s")
        dump()

        # same prompt construction convention as bench_vllm.py: random real-vocab token ids
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
            vocab_n = tok.vocab_size
        except Exception as e:  # noqa: BLE001
            log(f"tokenizer unavailable ({e}); using 240000 as vocab bound")
            vocab_n = 240000
        rec["vocab_n"] = vocab_n

        sp = {"temperature": 0.0, "top_p": 1.0, "max_new_tokens": args.output_len,
              "min_new_tokens": args.output_len, "ignore_eos": True}

        def one_rep(seed):
            rng = random.Random(seed)
            ids = [[rng.randrange(0, vocab_n) for _ in range(args.input_len)]
                   for _ in range(args.batch_size)]
            t = time.perf_counter()
            outs = eng.generate(input_ids=ids, sampling_params=sp)
            dt = time.perf_counter() - t
            n = sum(len(o["output_ids"]) if isinstance(o, dict) and "output_ids" in o
                    else o["meta_info"]["completion_tokens"] for o in outs)
            return {"wall_s": dt, "tokens": n, "tok_per_s_e2e": n / dt}

        rec["stage"] = "warmup"
        dump()
        rec["warmup"] = one_rep(999)
        log(f"warmup: {rec['warmup']}")

        reps = []
        for r in range(args.reps):
            reps.append(one_rep(r))
            log(f"rep{r}: {reps[-1]}")
        rec["reps"] = reps
        tps = [x["tok_per_s_e2e"] for x in reps]
        rec["e2e_tok_per_s_median"] = statistics.median(tps)
        rec["e2e_tok_per_s_range_pct"] = (max(tps) - min(tps)) / statistics.median(tps) * 100
        rec["stage"] = "throughput_done"
        dump()

        if args.profile_dir:
            rec["stage"] = "profile"
            dump()
            try:
                try:
                    eng.start_profile(num_steps=args.profile_steps, activities=["GPU"])
                except TypeError:
                    eng.start_profile()
                one_rep(4242)
                try:
                    eng.stop_profile()
                except Exception:  # noqa: BLE001
                    pass
                files = sorted(str(p) for p in Path(args.profile_dir).rglob("*"))
                rec["profile_files"] = files[:40]
                log(f"profile files: {len(files)}")
            except Exception as e:  # noqa: BLE001
                rec["profile_error"] = f"{type(e).__name__}: {e}"
                rec["profile_traceback"] = traceback.format_exc()
                log(f"PROFILE FAILED: {e}")
        rec["stage"] = "done"
        dump()
        try:
            eng.shutdown()
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
        rec["stage_failed_at"] = rec.get("stage")
        dump()
        log(f"FAILED at stage={rec.get('stage')}: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
    log("SGLANG PROBE COMPLETE")


if __name__ == "__main__":
    main()
