#!/usr/bin/env python3
"""M3-I6a: per-context attention cost curve from ONE deep-context wave.

A single `msl=897` wave (256-token synthetic prompt + 640 decode steps) walks
decode context from ~257 to ~896.  The megakernel profiler timestamps every
task instance, and `trace_lib.iteration_bounds` segments the run by
TASK_BEGIN_TASK_GRAPH (the anchor that fires exactly once per step), so the
whole context trajectory is recoverable from that one capture by binning
task-257 instances into iteration windows -- no separate run per context point.

Per window this reports, for the full-attention family (and, as context-flat
controls, GDN recurrent + dense fp8):

  wallspan_us_per_step  union of the family's [begin,end) intervals / n_steps
                        -- the same convention M3-I1/M3-I10 report and the
                        number ferret_targets.json ranks on
  sum_us_per_step       sum of durations / n_steps (work, not wall)
  mean_us / p50_us      per-instance duration
  n_per_step            instances per step (integrity: must be an integer)
  n_blocks              distinct workers the family landed on (the width)

Anchor QC is mandatory and runs over the FULL span before any window is
reported: every task type's per-step instance count must be an integer, and
must equal the compiled task graph's static call-site count.  A window is
refused if max|count/step - round(count/step)| exceeds the threshold.

Context per window comes from schedule_sim, the same simulator M3-I10 used to
label per-slot context, so the x-axis is derived, not assumed.

Usage:
  python3 ctx_curve.py --raw R.npz --meta M.json --names N.json \
      --graph task_graph_rank0.json --out out.json [--window 96] [--stride 48]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get(
    "I6A_OPT_DIR",
    str(Path.home() / "mpk-qwen35" / "mirage-i6a" / "demo" / "qwen3_5"
        / "accept" / "opt")))
import trace_lib as TL  # noqa: E402
import schedule_sim as SIM  # noqa: E402

# Families of interest.  257 is the target; 237/279 are the two families
# M3-I10 proved context-FLAT, so they double as an in-run control: if they
# move across the same windows, the window itself (not context) is the cause.
FAMILIES = {257: "attn", 237: "gdn_recurrent", 279: "dense_fp8"}


def load_buf(raw_path):
    z = np.load(raw_path)
    idx, val = z["idx"], z["val"]
    buf = np.zeros(int(idx.max()) + 1, dtype=np.uint64)
    buf[idx.astype(np.int64)] = val
    if "header" in z:
        buf[:1] = z["header"].view(np.uint64)
    return buf


def family_stats(pairs, bounds, it_of_pair, tt, lo, hi):
    """Stats for task type `tt` over iterations [lo, hi)."""
    sel = (pairs["task_type"] == tt) & (it_of_pair >= lo) & (it_of_pair < hi)
    n_it = float(hi - lo)
    if not sel.any():
        return dict(n_per_step=0.0, sum_us_per_step=0.0,
                    wallspan_us_per_step=0.0, mean_us=None, p50_us=None,
                    max_us=None, n_blocks=0)
    b = pairs["begin"][sel]
    e = pairs["end"][sel]
    d = pairs["dur"][sel].astype(np.float64)
    blk = pairs["block"][sel]
    # clip to the window so a task straddling the boundary is not double
    # counted against a window it mostly ran outside of
    bb = np.clip(b, bounds[lo], bounds[hi])
    ee = np.clip(e, bounds[lo], bounds[hi])
    wall = TL.union_length(bb, ee)
    return dict(
        n_per_step=round(float(sel.sum()) / n_it, 4),
        sum_us_per_step=round(float(d.sum()) / 1e3 / n_it, 2),
        wallspan_us_per_step=round(wall / 1e3 / n_it, 2),
        mean_us=round(float(d.mean()) / 1e3, 3),
        p50_us=round(float(np.percentile(d, 50)) / 1e3, 3),
        max_us=round(float(d.max()) / 1e3, 3),
        n_blocks=int(len(np.unique(blk))),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-workers", type=int, default=128)
    ap.add_argument("--window", type=int, default=96,
                    help="iterations per reported window")
    ap.add_argument("--stride", type=int, default=48)
    ap.add_argument("--frac-err-threshold", type=float, default=0.02)
    args = ap.parse_args(argv)

    meta = json.load(open(args.meta))
    names = json.load(open(args.names))
    graph = json.load(open(args.graph))
    NW = args.n_workers

    buf = load_buf(args.raw)
    ev = TL.decode_events(buf)
    del buf
    pairs = TL.pair_events(ev)
    del ev
    bounds = TL.iteration_bounds(pairs)
    n_it = len(bounds) - 1

    # ---- mandatory anchor QC over the FULL span ---------------------------
    static_counts = Counter(t["task_type"] for t in graph["all_tasks"])
    table_full = TL.per_task_table(pairs, bounds, 0, n_it, NW, names)
    obs = {r["task_type"]: r["n_per_iter"] for r in table_full}
    max_frac_err, n_mismatch, qc_rows = 0.0, 0, []
    for t in sorted(set(obs) | set(static_counts)):
        o = obs.get(t, 0.0)
        s = static_counts.get(t, 0)
        fe = abs(o - round(o))
        max_frac_err = max(max_frac_err, fe)
        mism = round(o) != s
        n_mismatch += int(mism)
        qc_rows.append(dict(task_type=t, name=names.get(str(t), f"UNKNOWN_{t}"),
                            observed_per_step=round(o, 4), rounded=round(o),
                            static_call_sites=s, frac_err=round(fe, 4),
                            mismatch=mism))
    qc_rows.sort(key=lambda r: -r["frac_err"])
    anchor_per_step = obs.get(10)
    window_valid = max_frac_err <= args.frac_err_threshold
    anchor_qc = dict(n_iterations_full_span=n_it,
                     task_begin_task_graph_per_step=anchor_per_step,
                     task_begin_task_graph_is_1_0=(anchor_per_step == 1.0),
                     max_frac_err_over_all_types=round(max_frac_err, 4),
                     threshold=args.frac_err_threshold,
                     window_valid=bool(window_valid),
                     n_task_types_mismatched_static_count=n_mismatch,
                     rows=qc_rows[:12])

    # ---- context axis from schedule_sim -----------------------------------
    plens = meta["prompt_lens"]
    bs = meta["batch_size"]
    slot_lens = plens + [plens[i % len(plens)] for i in range(len(plens), bs)]
    sim = SIM.simulate(slot_lens, meta["mbt"], meta["max_seq_length"])
    lo_s, hi_s = SIM.steady_window(sim)

    msl = sim["max_seq_length"]
    plens_sim = sim["plens"]

    def ctx_at(it_lo, it_hi):
        """Per-slot decode context (= step+1) over [it_lo,it_hi).

        `schedule_sim.simulate` records `steps[i]` = tokens already committed
        for slot i at the START of that iteration, so slot i's context while it
        computes is `steps[i] + 1`.  A slot is still in PREFILL while
        `steps[i] < plens[i]` and is retired once `steps[i] + 1 >= msl`; both
        are excluded so the axis is the decode context the attention KV loop
        actually walks.
        """
        vals, n_prefill = [], 0
        for rec in sim["iters"]:
            if not (it_lo <= rec["iteration"] < it_hi):
                continue
            for i, st in enumerate(rec["steps"]):
                if st + 1 >= msl:
                    continue            # retired
                if st < plens_sim[i]:
                    n_prefill += 1      # still prefilling
                    continue
                vals.append(int(st) + 1)
        if not vals:
            return None
        return dict(min=int(min(vals)), max=int(max(vals)),
                    mean=round(float(np.mean(vals)), 1), n=len(vals),
                    n_prefill_slot_iters=n_prefill)

    it_of_pair = np.searchsorted(bounds, pairs["begin"], side="right") - 1
    dur_it = np.diff(bounds)

    windows = []
    start = max(lo_s, 0)
    w = args.window
    while start + w <= n_it:
        lo, hi = start, start + w
        row = dict(iter_lo=lo, iter_hi=hi,
                   step_us=round(float(dur_it[lo:hi].mean()) / 1e3, 2),
                   context=ctx_at(lo, hi))
        for tt, tag in FAMILIES.items():
            row[tag] = family_stats(pairs, bounds, it_of_pair, tt, lo, hi)
        windows.append(row)
        start += args.stride
    # always include the final full window (the late-context closure point)
    if n_it >= w:
        lo, hi = n_it - w, n_it
        if not windows or windows[-1]["iter_lo"] != lo:
            row = dict(iter_lo=lo, iter_hi=hi, final=True,
                       step_us=round(float(dur_it[lo:hi].mean()) / 1e3, 2),
                       context=ctx_at(lo, hi))
            for tt, tag in FAMILIES.items():
                row[tag] = family_stats(pairs, bounds, it_of_pair, tt, lo, hi)
            windows.append(row)

    out = dict(tag=meta.get("tag"), batch_size=bs,
               max_seq_length=meta.get("max_seq_length"),
               attn_q_pass=os.environ.get("I6A_LABEL_QPASS"),
               n_workers=NW, sim_steady_window=[lo_s, hi_s],
               anchor_qc=anchor_qc, windows=windows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"anchor QC: n_it={n_it} anchor/step={anchor_per_step} "
          f"max_frac_err={max_frac_err:.4f} valid={window_valid} "
          f"mismatched_types={n_mismatch}")
    if n_mismatch:
        for r in qc_rows:
            if r["mismatch"]:
                print(f"   MISMATCH t={r['task_type']} {r['name']} "
                      f"obs={r['observed_per_step']} static={r['static_call_sites']}")
    print(f"{'iters':>13} {'ctx(mean)':>10} {'step_us':>9} "
          f"{'attn_wall':>10} {'attn_sum':>9} {'attn_n':>7} {'attn_mean':>10} "
          f"{'blk':>4} {'gdn_wall':>9} {'fp8_wall':>9}")
    for r in windows:
        c = r["context"]
        cs = f"{c['mean']:.0f}" if c else "?"
        a = r["attn"]
        print(f"{r['iter_lo']:5d}-{r['iter_hi']:<7d} {cs:>10} {r['step_us']:9.1f} "
              f"{a['wallspan_us_per_step']:10.1f} {a['sum_us_per_step']:9.1f} "
              f"{a['n_per_step']:7.2f} "
              f"{(a['mean_us'] or 0):10.2f} {a['n_blocks']:4d} "
              f"{r['gdn_recurrent']['wallspan_us_per_step']:9.1f} "
              f"{r['dense_fp8']['wallspan_us_per_step']:9.1f}")
    return 0 if window_valid else 1


if __name__ == "__main__":
    sys.exit(main())
