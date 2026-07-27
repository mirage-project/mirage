#!/usr/bin/env python3
"""Recover PER-SITE cost for kernels that fire many times per decode step.

A trace only gives one name for e.g. all 160 CUTLASS fp8 GEMMs in a step, but those 160 calls
are 6 different GEMM shapes.  Within one step the launch ORDER is deterministic (layer 0..39),
and each CUDA stream is in-order, so the ordinal position of a call within (step, name, stream)
is a stable site identity.  Aggregating duration by ordinal across ~300 steps therefore gives
the per-site cost with tight error bars, without needing shape metadata in the trace.
"""
import argparse
import gc
import json
import statistics
from collections import defaultdict
from pathlib import Path

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--names", nargs="+", required=True,
                    help="kernel-name substrings to profile by ordinal")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    # ordinal -> list of durations, per (matched-name-key, stream)
    acc = defaultdict(lambda: defaultdict(list))
    nsteps_total = 0
    for tp in args.traces:
        with open(tp) as f:
            data = json.load(f)
        gpu = []
        for e in data.get("traceEvents", []):
            if e.get("cat") not in GPU_CATS:
                continue
            a = e.get("args") or {}
            gpu.append((e.get("ts", 0) or 0, e.get("dur", 0) or 0, e.get("name", ""), a.get("stream")))
        del data
        gc.collect()
        gpu.sort()
        ats = [ts for ts, _, n, _ in gpu if n == args.anchor]
        if len(ats) < 2:
            raise SystemExit(f"{tp}: anchor fired {len(ats)}x")
        bounds = list(zip(ats[:-1], ats[1:]))
        nsteps_total += len(bounds)
        # bucket events per step
        per_step = defaultdict(list)
        bi, nb = 0, len(bounds)
        for ts, d, n, st in gpu:
            while bi < nb and ts >= bounds[bi][1]:
                bi += 1
            if bi >= nb or ts < bounds[bi][0]:
                continue
            per_step[bi].append((ts, d, n, st))
        for _, evs in per_step.items():
            seq = defaultdict(list)
            for ts, d, n, st in evs:
                for key in args.names:
                    if key in n:
                        seq[(key, str(st))].append((ts, d))
                        break
            for k, lst in seq.items():
                lst.sort()
                for i, (_, d) in enumerate(lst):
                    acc[k][i].append(d)
        del gpu, per_step
        gc.collect()
        print(f"[ordinal] {Path(tp).name}: {len(bounds)} steps", flush=True)

    out = {"label": args.label, "anchor": args.anchor, "n_steps_total": nsteps_total, "sites": {}}
    for (key, st), ords in sorted(acc.items()):
        rows = []
        for i in sorted(ords):
            ds = ords[i]
            rows.append({"ordinal": i, "n": len(ds), "median_us": statistics.median(ds),
                         "min_us": min(ds), "max_us": max(ds),
                         "mean_us": sum(ds) / len(ds)})
        out["sites"][f"{key}@stream{st}"] = rows
        tot = sum(r["median_us"] for r in rows)
        print(f"\n### {key[:80]} @stream {st}: {len(rows)} ordinals, "
              f"sum of medians {tot:.1f} us/step")
        for r in rows:
            print(f"   ord {r['ordinal']:3d}  n={r['n']:5d}  {r['median_us']:8.3f} us "
                  f"[{r['min_us']:.3f}-{r['max_us']:.3f}]")
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
