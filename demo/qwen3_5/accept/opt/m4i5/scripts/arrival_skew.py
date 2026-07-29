#!/usr/bin/env python3
"""M4-I5: measure the ARRIVAL SKEW of a task level, as a function of how many
tasks the level emits.

Why this is the decisive instrument.  The wave-depth model says a level costs
`max_worker(live*T_live + dead*T_dead)`.  Measured spans exceed that, and the
excess grows with the EMITTED task count -- which no worker-side cost can
explain, because a dead task is 0.5 us however many of them there are.  What is
proportional to the emitted count is the DISPATCH: MPK's
`EVENT_LAUNCH_DEPENDENT_TASKS` handler walks the event's task range and pushes
each id into a worker queue individually with an `atom_add_release_gpu_u64` per
push, and a worker's `PROFILER_EVENT_START` fires only after its dependency wait
-- so enqueue time appears as LATE ARRIVAL, not as task time.

So: for each level of a stage, measure `max(begin) - min(begin)` over its tasks.
That is the wall time between the first and last task of one level starting, i.e.
the skew the dispatch imposes.  If it grows linearly in the emitted count with a
stable ns-per-task slope, the limit is dispatch rate, not graph width.

Levels are recovered from the trace by clustering task start times inside the
step: a stage with L levels per step has L bursts, so the L-1 largest inter-start
gaps are the level boundaries.  The recovered cluster count and sizes are
reported so the segmentation is checkable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import trace_lib as TL  # noqa: E402

NW = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="label=raw.npz,meta.json,names.json,lo:hi")
    ap.add_argument("--types", default="241,242",
                    help="task types to segment (default the routed MoE GEMMs)")
    ap.add_argument("--levels", type=int, default=40,
                    help="levels per step for those types (40 MoE layers)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    types = [int(x) for x in a.types.split(",")]

    report = {}
    for spec in a.arm:
        label, rest = spec.split("=", 1)
        raw, meta_p, names_p, win = rest.split(",")
        names = json.load(open(names_p))
        z = np.load(raw)
        idx, val = z["idx"], z["val"]
        buf = np.zeros(int(idx.max()) + 1, dtype=np.uint64)
        buf[idx.astype(np.int64)] = val
        buf[:1] = z["header"].view(np.uint64)
        ev = TL.decode_events(buf)
        del buf
        p = TL.pair_events(ev)
        del ev
        bounds = TL.iteration_bounds(p)
        n_it = len(bounds) - 1
        it = np.searchsorted(bounds, p["begin"], side="right") - 1
        lo, hi = (int(x) for x in win.split(":"))
        nsteps = float(hi - lo)
        step_us = (bounds[hi] - bounds[lo]) / 1e3 / nsteps

        arm = dict(step_us=round(float(step_us), 1), window=[lo, hi], stages={})
        for t in types:
            m = (it >= lo) & (it < hi) & (p["task_type"] == t) & (p["block"] < NW)
            if not m.any():
                continue
            # one iteration at the window midpoint, so level clustering is not
            # confused by iteration boundaries
            mid = lo + int(nsteps // 2)
            mm = (it == mid) & (p["task_type"] == t) & (p["block"] < NW)
            b = np.sort(p["begin"][mm].astype(np.int64))
            n = len(b)
            if n < a.levels * 2:
                continue
            gaps = np.diff(b)
            cut = np.sort(np.argsort(gaps)[-(a.levels - 1):])
            edges = np.concatenate([[0], cut + 1, [n]])
            skew, sizes = [], []
            for i in range(len(edges) - 1):
                seg = b[edges[i]:edges[i + 1]]
                if len(seg) < 2:
                    continue
                skew.append((seg[-1] - seg[0]) / 1e3)
                sizes.append(len(seg))
            skew = np.array(skew)
            sizes = np.array(sizes)
            emitted_per_level = float(np.median(sizes))
            arm["stages"][names.get(str(t), str(t))] = dict(
                task_type=t, iteration=mid,
                tasks_in_iteration=n,
                levels_recovered=len(skew),
                emitted_per_level_median=emitted_per_level,
                emitted_per_level_min=int(sizes.min()),
                emitted_per_level_max=int(sizes.max()),
                arrival_skew_us_median=round(float(np.median(skew)), 2),
                arrival_skew_us_mean=round(float(skew.mean()), 2),
                arrival_skew_us_p90=round(float(np.percentile(skew, 90)), 2),
                ns_per_emitted_task=round(
                    1e3 * float(np.median(skew)) / max(emitted_per_level, 1), 2),
                skew_total_us_per_step=round(float(skew.sum()), 1))
        report[label] = arm
        del p

    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"{'arm':6s}{'stage':10s}{'lvls':>6s}{'emit/lvl':>10s}"
          f"{'skew_us':>9s}{'p90':>8s}{'ns/task':>9s}{'sum_us':>9s}")
    for k, v in report.items():
        for nm, s in v["stages"].items():
            print(f"{k:6s}{nm.replace('TASK_MOE_','')[:10]:10s}"
                  f"{s['levels_recovered']:6d}{s['emitted_per_level_median']:10.1f}"
                  f"{s['arrival_skew_us_median']:9.2f}{s['arrival_skew_us_p90']:8.2f}"
                  f"{s['ns_per_emitted_task']:9.2f}{s['skew_total_us_per_step']:9.1f}")


if __name__ == "__main__":
    main()
