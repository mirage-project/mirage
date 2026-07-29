#!/usr/bin/env python3
"""M4-I5: is the width limit the TASK GRAPH or the SCHEDULER?

M3-I1 exonerated the scheduler; this re-tests it at current HEAD and, more to the
point, tests it as a FUNCTION OF EMITTED TASK COUNT, because that is what a split
changes.  `width.py` filters to `block < 128` (the workers), so scheduler CTAs --
`nblocks` is 208, i.e. 128 workers plus 80 scheduler blocks -- are invisible to
it.  This script reports them.

MPK's `EVENT_LAUNCH_DEPENDENT_TASKS` handler walks the event's task range and
pushes each task id into a worker queue individually, with an
`atom_add_release_gpu_u64` per push (`persistent_kernel.cuh`).  So the enqueue
cost is LINEAR in the number of tasks EMITTED, whether or not they do work -- and
a grid split multiplies the emitted count.  If that cost is material, then past
some emitted-task count a split stops paying, and the limit has moved from graph
width to dispatch rate.

Per arm it reports, per step: scheduler-CTA busy time by task type, the emitted
task count, and the implied nanoseconds of scheduler time per emitted task.
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


def load(raw):
    z = np.load(raw)
    idx, val = z["idx"], z["val"]
    buf = np.zeros(int(idx.max()) + 1, dtype=np.uint64)
    buf[idx.astype(np.int64)] = val
    buf[:1] = z["header"].view(np.uint64)
    ev = TL.decode_events(buf)
    nb, ng = ev["nblocks"], ev["ngroups"]
    del buf
    p = TL.pair_events(ev)
    del ev
    return p, nb, ng


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="label=raw.npz,meta.json,names.json[,window_lo:hi]")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    report = {}
    for spec in a.arm:
        label, rest = spec.split("=", 1)
        parts = rest.split(",")
        raw, meta_p, names_p = parts[0], parts[1], parts[2]
        win = parts[3] if len(parts) > 3 else None
        meta = json.load(open(meta_p))
        names = json.load(open(names_p))
        p, nb, ng = load(raw)
        bounds = TL.iteration_bounds(p)
        n_it = len(bounds) - 1
        it = np.searchsorted(bounds, p["begin"], side="right") - 1
        lo, hi = (int(x) for x in win.split(":")) if win else (1, n_it - 1)
        nsteps = float(hi - lo)
        sel = (it >= lo) & (it < hi)
        step_us = (bounds[hi] - bounds[lo]) / 1e3 / nsteps

        wk = sel & (p["block"] < NW)
        sc = sel & (p["block"] >= NW)
        emitted = float(wk.sum()) / nsteps

        def by_type(mask):
            out = {}
            tt = p["task_type"][mask]
            d = p["dur"][mask].astype(np.float64)
            b, e = p["begin"][mask], p["end"][mask]
            for t in np.unique(tt):
                m = tt == t
                out[names.get(str(int(t)), str(int(t)))] = dict(
                    n=round(float(m.sum()) / nsteps, 1),
                    total_us=round(float(d[m].sum()) / 1e3 / nsteps, 1),
                    span_us=round(float(TL.union_length(b[m], e[m]))
                                  / 1e3 / nsteps, 1),
                    mean_us=round(float(d[m].mean()) / 1e3, 4))
            return dict(sorted(out.items(), key=lambda kv: -kv[1]["total_us"]))

        sched = by_type(sc)
        sched_total = round(sum(v["total_us"] for v in sched.values()), 1)
        # union span of scheduler activity: how much of the step has ANY
        # scheduler CTA busy
        sb, se = p["begin"][sc], p["end"][sc]
        sched_span = round(float(TL.union_length(sb, se)) / 1e3 / nsteps, 1)
        n_sched_blocks = int(len(np.unique(p["block"][sc])))

        report[label] = dict(
            raw=os.path.basename(raw), batch_size=meta["batch_size"],
            nblocks=int(nb), ngroups=int(ng), n_scheduler_blocks=n_sched_blocks,
            window=[lo, hi], n_iterations=n_it, step_us=round(float(step_us), 1),
            worker_tasks_per_step=round(emitted, 1),
            worker_task_us_per_step=round(float(p["dur"][wk].sum())
                                          / 1e3 / nsteps, 1),
            scheduler_us_per_step=sched_total,
            scheduler_span_us_per_step=sched_span,
            scheduler_span_pct_of_step=round(100 * sched_span / step_us, 1),
            ns_of_scheduler_time_per_emitted_task=round(
                1e3 * sched_total / max(emitted, 1), 1),
            scheduler_by_type=sched)
        del p

    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"{'arm':6s}{'step_us':>10s}{'tasks/step':>12s}{'sched_us':>10s}"
          f"{'sched_span':>11s}{'%step':>7s}{'ns/task':>9s}")
    for k, v in report.items():
        print(f"{k:6s}{v['step_us']:10.1f}{v['worker_tasks_per_step']:12.1f}"
              f"{v['scheduler_us_per_step']:10.1f}"
              f"{v['scheduler_span_us_per_step']:11.1f}"
              f"{v['scheduler_span_pct_of_step']:7.1f}"
              f"{v['ns_of_scheduler_time_per_emitted_task']:9.1f}")
    for k, v in report.items():
        print(f"\n{k}: scheduler-CTA busy by task type (us/step)")
        for n, s in list(v["scheduler_by_type"].items())[:6]:
            print(f"   {n[:34]:34s} n={s['n']:9.1f} total={s['total_us']:8.1f} "
                  f"mean={s['mean_us']:.4f} span={s['span_us']:8.1f}")


if __name__ == "__main__":
    main()
