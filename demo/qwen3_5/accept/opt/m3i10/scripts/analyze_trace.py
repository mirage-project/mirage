#!/usr/bin/env python3
"""Parse torch/kineto chrome traces into EXACT per-kernel per-decode-step tables.

Why an anchor, not the ProfilerStep count: kineto's ProfilerStep# annotations live on the CPU
timeline, but with CUDA graphs the GPU timeline lags the CPU by roughly one step, so the set of
GPU kernels captured in an N-step window is N +- 1 steps' worth (~9 % error at N=10).  Instead we
pick an ANCHOR kernel that fires exactly once per decode step (auto-detected as the highest-total-
time GPU event whose occurrence count is within +-2 of the ProfilerStep count), and integrate only
over [first_anchor_ts, last_anchor_ts) which contains exactly (n_anchor - 1) COMPLETE steps.

QC: with a correct anchor every per-step call count must come out an integer.  max_calls_int_dev
reports the worst deviation; anything above ~0.02 invalidates the window.
"""
import argparse
import gc
import json
import statistics
from collections import defaultdict
from pathlib import Path

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def load_gpu_events(path: Path):
    with open(path) as f:
        data = json.load(f)
    evs = data.get("traceEvents", [])
    steps = set()
    gpu = []
    for e in evs:
        n = e.get("name", "")
        if n.startswith("ProfilerStep#"):
            steps.add(n)
            continue
        if e.get("cat") in GPU_CATS:
            a = e.get("args") or {}
            gpu.append((e.get("ts", 0) or 0, e.get("dur", 0) or 0, n, a.get("stream")))
    del data, evs
    gc.collect()
    gpu.sort()
    return gpu, len(steps)


def pick_anchor(gpu, n_steps, override=None):
    agg = defaultdict(lambda: [0, 0.0])
    for _, d, n, _s in gpu:
        a = agg[n]
        a[0] += 1
        a[1] += d
    if override:
        return override, agg[override][0]
    cands = [(n, c, t) for n, (c, t) in agg.items() if abs(c - n_steps) <= 2 and c >= 3]
    if not cands:
        raise SystemExit(f"no once-per-step anchor candidate near n_steps={n_steps}")
    cands.sort(key=lambda x: -x[2])  # most total GPU time => unmistakable, e.g. the lm_head GEMM
    return cands[0][0], cands[0][1]


def parse_trace(path: Path, anchor_override=None) -> dict:
    gpu, n_steps_cpu = load_gpu_events(path)
    anchor, n_anchor = pick_anchor(gpu, n_steps_cpu, anchor_override)
    ats = [ts for ts, _, n, _s in gpu if n == anchor]
    t0, t1 = ats[0], ats[-1]
    n_steps = len(ats) - 1
    if n_steps < 1:
        raise SystemExit(f"{path}: anchor {anchor} fired {len(ats)}x - cannot window")

    kern = defaultdict(lambda: {"calls": 0, "us": 0.0, "durs": [], "streams": set()})
    ivs = []
    for ts, d, n, st in gpu:
        if ts < t0 or ts >= t1:
            continue
        k = kern[n]
        k["calls"] += 1
        k["us"] += d
        k["durs"].append(d)
        k["streams"].add(st)
        ivs.append((ts, ts + d))
    ivs.sort()
    union, cs, ce = 0.0, None, None
    for a, b in ivs:
        if cs is None:
            cs, ce = a, b
        elif a <= ce:
            ce = max(ce, b)
        else:
            union += ce - cs
            cs, ce = a, b
    if cs is not None:
        union += ce - cs

    total_us = sum(v["us"] for v in kern.values())
    rows = []
    max_dev = 0.0
    for name, v in kern.items():
        cps = v["calls"] / n_steps
        max_dev = max(max_dev, abs(cps - round(cps)))
        rows.append({
            "kernel": name,
            "calls": v["calls"],
            "calls_per_step": cps,
            "total_us": v["us"],
            "us_per_step": v["us"] / n_steps,
            "mean_us_per_call": v["us"] / v["calls"],
            "median_us_per_call": statistics.median(v["durs"]),
            "streams": sorted(str(x) for x in v["streams"]),
        })
    rows.sort(key=lambda r: -r["total_us"])
    return {
        "trace": path.name,
        "anchor_kernel": anchor,
        "n_steps_cpu_markers": n_steps_cpu,
        "n_steps_windowed": n_steps,
        "max_calls_int_dev": max_dev,
        "n_kernel_names": len(rows),
        "gpu_busy_us_per_step": total_us / n_steps,
        "gpu_union_busy_us_per_step": union / n_steps,
        "step_wall_us_from_anchor": (t1 - t0) / n_steps,
        "kernels": rows,
    }


def merge(per_trace: list) -> dict:
    names = set()
    for t in per_trace:
        names |= {r["kernel"] for r in t["kernels"]}
    idx = [{r["kernel"]: r for r in t["kernels"]} for t in per_trace]
    out = []
    for name in names:
        ups, cps, mus = [], [], []
        for d in idx:
            r = d.get(name)
            ups.append(r["us_per_step"] if r else 0.0)
            cps.append(r["calls_per_step"] if r else 0.0)
            if r:
                mus.append(r["mean_us_per_call"])
        med = statistics.median(ups)
        out.append({
            "kernel": name,
            "n_windows": len(ups),
            "calls_per_step": statistics.median(cps),
            "us_per_step_median": med,
            "us_per_step_min": min(ups),
            "us_per_step_max": max(ups),
            "us_per_step_range_pct": (max(ups) - min(ups)) / med * 100.0 if med else 0.0,
            "mean_us_per_call": statistics.median(mus) if mus else 0.0,
            "streams": sorted({s for d in idx if d.get(name) for s in d[name]["streams"]}),
        })
    tot = sum(r["us_per_step_median"] for r in out)
    for r in out:
        r["pct_of_gpu_busy"] = 100.0 * r["us_per_step_median"] / tot if tot else 0.0
    out.sort(key=lambda r: -r["us_per_step_median"])
    busy = [t["gpu_busy_us_per_step"] for t in per_trace]
    uni = [t["gpu_union_busy_us_per_step"] for t in per_trace]
    wall = [t["step_wall_us_from_anchor"] for t in per_trace]
    return {
        "n_windows": len(per_trace),
        "gpu_busy_us_per_step_median": statistics.median(busy),
        "gpu_busy_us_per_step_min": min(busy),
        "gpu_busy_us_per_step_max": max(busy),
        "gpu_busy_range_pct": (max(busy) - min(busy)) / statistics.median(busy) * 100.0,
        "gpu_union_busy_us_per_step_median": statistics.median(uni),
        "profiled_step_wall_us_median": statistics.median(wall),
        "sum_kernel_us_per_step": tot,
        "kernels": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--top", type=int, default=60)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_trace = []
    for p in args.traces:
        t = parse_trace(Path(p), args.anchor)
        per_trace.append(t)
        print(f"[trace] {t['trace']}: cpu_steps={t['n_steps_cpu_markers']} "
              f"windowed_steps={t['n_steps_windowed']} int_dev={t['max_calls_int_dev']:.4f} "
              f"busy={t['gpu_busy_us_per_step']:.1f} step_wall={t['step_wall_us_from_anchor']:.1f} "
              f"anchor={t['anchor_kernel'][:60]}", flush=True)
        gc.collect()

    m = merge(per_trace)
    m["per_trace"] = [{k: v for k, v in t.items() if k != "kernels"} for t in per_trace]
    (out_dir / f"{args.label}_kernels.json").write_text(json.dumps(m, indent=2))
    with open(out_dir / f"{args.label}_kernels.csv", "w") as f:
        f.write("kernel,calls_per_step,us_per_step_median,us_per_step_min,us_per_step_max,"
                "range_pct,mean_us_per_call,pct_of_gpu_busy,streams\n")
        for r in m["kernels"]:
            f.write(f'"{r["kernel"]}",{r["calls_per_step"]:.4f},{r["us_per_step_median"]:.3f},'
                    f'{r["us_per_step_min"]:.3f},{r["us_per_step_max"]:.3f},'
                    f'{r["us_per_step_range_pct"]:.2f},{r["mean_us_per_call"]:.4f},'
                    f'{r["pct_of_gpu_busy"]:.3f},{"|".join(r["streams"])}\n')

    print(f"\n=== {args.label}: gpu busy {m['gpu_busy_us_per_step_median']:.1f} us/step "
          f"(+-{m['gpu_busy_range_pct']:.2f}%), profiled step wall "
          f"{m['profiled_step_wall_us_median']:.1f} us, {len(m['kernels'])} names, "
          f"{m['n_windows']} windows ===")
    print(f"{'us/step':>9} {'calls':>8} {'us/call':>8} {'rng%':>6} {'%':>6}  kernel")
    for r in m["kernels"][:args.top]:
        print(f"{r['us_per_step_median']:9.2f} {r['calls_per_step']:8.2f} "
              f"{r['mean_us_per_call']:8.3f} {r['us_per_step_range_pct']:6.1f} "
              f"{r['pct_of_gpu_busy']:6.2f}  {'|'.join(r['streams']):>6}  {r['kernel'][:120]}")


if __name__ == "__main__":
    main()
