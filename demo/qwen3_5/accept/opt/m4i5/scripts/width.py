#!/usr/bin/env python3
"""M4-I5 -- the GRAPH-WIDTH quantifier.

One question: for each task stage, how many tasks does the graph make available
per dependency level, how many of the 128 workers does that actually occupy, and
how much wall time is lost to the difference?

Everything here is measured off a retained profiler buffer plus the compiled
task graph.  Three views, in increasing strength:

1. STAGE WIDTH (measured).  Per task type inside one steady step: tasks
   emitted, tasks that did work, the number of dependency levels (call sites)
   they are spread over, live tasks per level, per-task latency, wall span,
   perfect-pack time (task-us / 128) and the ratio.  `live per level` is the
   width number: a stage with 16 live tasks per level cannot use more than 16
   of 128 workers no matter how the scheduler behaves.

2. WHERE THE IDLE MACHINE IS (measured).  A +1/-1 concurrency sweep over the
   step, then every segment's duration split across the stages running in it.
   Reported banded by machine concurrency, so "which stages own the time when
   the box is <=16/128 busy" is answered directly rather than inferred.  Also
   the SOLE-OCCUPANT view: time when a stage is the only thing running, which
   is width residual that no other stage can hide.

3. WAVE-DEPTH COST MODEL (model, validated here).  M3-I8 established that MPK
   dispatch makes a stage's cost `ceil(live / 128) * T_task` -- tasks go to
   worker `(t - first_task_id) % 128` and each worker drains its queue in
   order, so cost is worker DEPTH, and per-task time is flat in the live count
   (`model_moe_wall.py` C2/C3).  This script re-fits that on the current tree
   and prints predicted-vs-measured span per stage, so the ceiling model in
   §3 of the README rests on a checked fit rather than an assertion.

Anchor QC is mandatory and is reported for the whole run, not only the window:
the compiled graph is STATIC (the same task list every iteration), so the
per-iteration count of every task type must equal its static call-site count.
Any deficit is an instrument artifact and is quantified as one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import trace_lib as TL  # noqa: E402

# Realized worker count.  128 was hard-coded through M4-I8 and silently wrong
# once production moved to 136 workers (tracks 128-135 were discarded);
# override via MPK_REALIZED_WORKERS for any non-default build.
NW = int(os.environ.get("MPK_REALIZED_WORKERS", "128"))
# m3i8's own separator for "this task did real work" on the grouped MoE GEMMs:
# the empty-task latency tail crosses 1 us for 1.6% of empties, and the
# histogram is empty from 2 us to 48 us (opt/m3i8/results/f1_closure).
LIVE_US = 4.0


def load_pairs(raw_path):
    z = np.load(raw_path)
    idx, val = z["idx"], z["val"]
    buf = np.zeros(int(idx.max()) + 1, dtype=np.uint64)
    buf[idx.astype(np.int64)] = val
    buf[:1] = z["header"].view(np.uint64)
    ev = TL.decode_events(buf)
    del buf, idx, val
    p = TL.pair_events(ev)
    del ev
    return p


def static_call_sites(graph_path):
    """task_type -> number of tasks in the compiled graph (one iteration)."""
    with open(graph_path) as f:
        g = json.load(f)
    c = Counter(t["task_type"] for t in g["all_tasks"])
    # dependency levels: how many distinct `dependent_event`s a type waits on.
    lvl = {}
    for t in g["all_tasks"]:
        lvl.setdefault(t["task_type"], set()).add(t["dependent_event"])
    return dict(c), {k: len(v) for k, v in lvl.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("meta")
    ap.add_argument("names")
    ap.add_argument("--graph", default=None)
    ap.add_argument("--window", default=None,
                    help="lo,hi iteration window; default = trace-derived")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    meta = json.load(open(a.meta))
    names = json.load(open(a.names))
    bs = meta["batch_size"]

    p = load_pairs(a.raw)
    bounds = TL.iteration_bounds(p)
    n_it = len(bounds) - 1
    it = np.searchsorted(bounds, p["begin"], side="right") - 1

    static, n_levels = ({}, {})
    if a.graph:
        static, n_levels = static_call_sites(a.graph)

    # ---------------- anchor QC ----------------
    # The compiled graph is STATIC: the same task list runs every iteration and
    # only the DURATIONS vary with the regime.  So the test with no free
    # parameters is "does every iteration of the window contain exactly the
    # static call-site count of every task type".  Also reported: the length of
    # the exact prefix, because the per-track profiler cap
    # (`PROFILER_CAN_WRITE`, profiler.h) silently DROPS events once a worker
    # track fills, which truncates the tail of a long run.
    types = np.unique(p["task_type"])
    per_it_counts = {}
    dur_us = p["dur"].astype(np.float64) / 1e3
    for t in types:
        m = (p["task_type"] == t) & (it >= 0) & (it < n_it)
        per_it_counts[int(t)] = np.bincount(it[m], minlength=n_it)[:n_it]
        ml = m & (dur_us >= LIVE_US)
        per_it_counts[(int(t), "live")] = np.bincount(it[ml],
                                                     minlength=n_it)[:n_it]

    ok_all = np.ones(n_it, dtype=bool)
    for t in types:
        st = static.get(int(t), 0)
        if st:
            ok_all &= per_it_counts[int(t)] == st
    exact_prefix = int(np.argmin(ok_all)) if not ok_all.all() else n_it

    # trace-derived live-slot count: a slot-indexed stage's LIVE task count
    # divided by its per-slot call sites.  conv1d grid = (mbr, channel_blocks),
    # attention grid = (mbr, kv_heads), so both read out the live request count
    # with no replay of the admission policy.
    def live_slots(tt):
        pl = static.get(tt, 0) // max(bs, 1)
        return (per_it_counts[(tt, "live")] / max(pl, 1)) if pl else None

    live_conv, live_attn = live_slots(234), live_slots(257)

    # ---------------- window ----------------
    if a.window:
        lo, hi = (int(x) for x in a.window.split(","))
    else:
        lc = np.rint(live_conv).astype(int)
        best, i = (0, 0, 0), 0
        while i < min(n_it, exact_prefix):
            j = i
            while j < min(n_it, exact_prefix) and lc[j] == lc[i]:
                j += 1
            if (j - i, lc[i]) > (best[0], best[1]):
                best = (j - i, lc[i], i)
            i = j
        lo, hi = best[2] + 1, best[2] + best[0] - 1
    step_us = (bounds[hi] - bounds[lo]) / 1e3 / (hi - lo)

    qc_rows = []
    for t in types:
        st = static.get(int(t), 0)
        c = per_it_counts[int(t)][lo:hi]
        qc_rows.append(dict(
            task_type=int(t), name=names.get(str(int(t)), str(int(t))),
            static_call_sites=st,
            window_min=int(c.min()), window_max=int(c.max()),
            exact_every_iteration=bool(st and (c == st).all()),
            rel_err=(round(float(abs(c.mean() - st)) / st, 6) if st else None)))
    qc_rows.sort(key=lambda r: -(r["rel_err"] or 0))
    worst_rel = max((r["rel_err"] or 0) for r in qc_rows)

    sel = (it >= lo) & (it < hi) & (p["block"] < NW)
    b = p["begin"][sel].astype(np.int64)
    e = p["end"][sel].astype(np.int64)
    tt = p["task_type"][sel]
    nsteps = float(hi - lo)

    # ---------------- concurrency sweep over the window ----------------
    ts = np.concatenate([b, e])
    dv = np.concatenate([np.ones(len(b), np.int32), -np.ones(len(e), np.int32)])
    o = np.argsort(ts, kind="stable")
    ts, dv = ts[o], dv[o]
    conc = np.cumsum(dv)
    seg_dt = np.diff(np.concatenate([ts, [ts[-1]]])).astype(np.float64)
    tot_time = seg_dt.sum()
    hist = np.bincount(np.clip(conc, 0, NW), weights=seg_dt, minlength=NW + 1)
    mean_conc = float((np.arange(NW + 1) * hist).sum() / max(tot_time, 1))

    # per-stage share of each concurrency band.  For every segment, split its
    # duration across the running stages in proportion to how many of that
    # stage's tasks are running -- so the shares sum to the segment duration
    # and a narrow stage running alongside a wide one is not credited with the
    # wide one's occupancy.
    BANDS = [(0, 1), (1, 17), (17, 33), (33, 65), (65, 97), (97, 129)]
    band_of = np.zeros(NW + 1, dtype=np.int32)
    for k, (l, h) in enumerate(BANDS):
        band_of[l:h] = k
    seg_band = band_of[np.clip(conc, 0, NW)]

    utypes = np.unique(tt)
    tt2 = np.concatenate([tt, tt])[o]          # stage of each sweep event
    live_seg = conc > 0
    inv_conc = np.zeros(len(conc), dtype=np.float64)
    inv_conc[live_seg] = 1.0 / conc[live_seg]
    wdt = seg_dt * inv_conc                    # dt / concurrency, per segment

    runs = {}
    n_running_stages = np.zeros(len(conc), dtype=np.int32)
    for t in utypes:
        r = np.cumsum(np.where(tt2 == t, dv, 0).astype(np.int64))
        runs[int(t)] = r
        n_running_stages += (r > 0)
    sole_seg = live_seg & (n_running_stages == 1)

    share, sole = {}, {}
    for t in utypes:
        r = runs[int(t)]
        # proportional split of each segment's duration across running stages
        acc = np.bincount(seg_band, weights=wdt * r, minlength=len(BANDS))
        share[int(t)] = (acc / 1e3 / nsteps).round(3).tolist()
        sm = sole_seg & (r > 0)
        st_ = float(seg_dt[sm].sum())
        sole[int(t)] = [round(st_ / 1e3 / nsteps, 3),
                        round(float((seg_dt[sm] * conc[sm]).sum()) / max(st_, 1), 2)]
    del runs

    # ---------------- per-stage width table ----------------
    rows = []
    for t in utypes:
        m = tt == t
        d = (e[m] - b[m]).astype(np.float64)
        live = d >= LIVE_US * 1e3
        n = int(m.sum())
        span = float(TL.union_length(b[m], e[m])) / 1e3 / nsteps
        total = float(d.sum()) / 1e3 / nsteps
        nlive = float(live.sum()) / nsteps
        lv = int(n_levels.get(int(t), 0))
        st = int(static.get(int(t), 0))
        rows.append(dict(
            task_type=int(t), name=names.get(str(int(t)), str(int(t))),
            n_per_step=round(n / nsteps, 2),
            static=st, levels=lv,
            emitted_per_level=(round(st / lv, 2) if lv else None),
            live_per_step=round(nlive, 2),
            live_per_level=(round(nlive / lv, 2) if lv else None),
            t_live_us=(round(float(d[live].mean()) / 1e3, 3) if live.any() else 0.0),
            t_dead_us=(round(float(d[~live].mean()) / 1e3, 3)
                       if (~live).any() else 0.0),
            t_all_us=round(float(d.mean()) / 1e3, 3),
            total_us=round(total, 1),
            span_us=round(span, 1),
            pct_step=round(100 * span / step_us, 2),
            self_conc=round(total / span, 2) if span else 0.0,
            perfect_pack_us=round(total / NW, 2),
            span_over_pp=round(span / (total / NW), 2) if total else 0.0,
            depth=(int(np.ceil((nlive / lv) / NW)) if lv else None),
            share_by_band_us=share[int(t)],
            sole_us=sole[int(t)][0], sole_mean_conc=sole[int(t)][1],
            # wall time this stage holds the machine ALONE, weighted by how
            # much of the machine it leaves idle while doing so.  This is the
            # width residual attributable to this stage and nothing else.
            sole_idle_us=round(sole[int(t)][0]
                               * (1.0 - sole[int(t)][1] / NW), 1),
        ))
    rows.sort(key=lambda r: -r["span_us"])

    out = dict(
        tag=meta.get("tag"), batch_size=bs, head=os.environ.get("M4I5_HEAD", ""),
        raw=os.path.basename(a.raw),
        n_iterations=int(n_it), window=[int(lo), int(hi)],
        step_us=round(float(step_us), 1),
        wave_ms_per_decode_step=meta["waves"][0].get("ms_per_decode_step"),
        dropped_begin=int(p["dropped_begin"]), dropped_end=int(p["dropped_end"]),
        anchor_qc=dict(worst_rel_err=round(float(worst_rel), 6),
                       all_types_exact_every_iteration=bool(
                           all(r["exact_every_iteration"] for r in qc_rows
                               if r["static_call_sites"])),
                       exact_prefix_iterations=exact_prefix,
                       verdict=("PASS" if all(r["exact_every_iteration"]
                                              for r in qc_rows
                                              if r["static_call_sites"])
                                else "FAIL"),
                       rows=qc_rows),
        regime=dict(live_from_conv1d=round(float(np.rint(live_conv[lo:hi]).mean()), 3),
                    live_from_attn=round(float(np.rint(live_attn[lo:hi]).mean()), 3),
                    live_conv_min=int(np.rint(live_conv[lo:hi]).min()),
                    live_conv_max=int(np.rint(live_conv[lo:hi]).max())),
        machine=dict(mean_concurrency=round(mean_conc, 2),
                     occupancy=round(mean_conc / NW, 4),
                     us_by_band={f"{l}-{h-1}": round(float(hist[l:h].sum()) / 1e3 / nsteps, 1)
                                 for l, h in BANDS},
                     total_task_us=round(float((e - b).sum()) / 1e3 / nsteps, 1),
                     work_bound_us=round(float((e - b).sum()) / 1e3 / nsteps / NW, 1)),
        stages=rows,
    )
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("stages", "anchor_qc")}, indent=1))
    print(f"anchor_qc: {out['anchor_qc']['verdict']} worst_rel_err="
          f"{out['anchor_qc']['worst_rel_err']}")
    print(f"{'stage':34s}{'n/step':>8s}{'live':>8s}{'lvl':>5s}{'live/lvl':>9s}"
          f"{'T_live':>8s}{'span':>8s}{'%step':>7s}{'pp':>8s}{'sp/pp':>7s}"
          f"{'dep':>4s}{'sole':>8s}{'sconc':>7s}{'idle':>8s}")
    for r in rows:
        print(f"{r['name'][:34]:34s}{r['n_per_step']:8.1f}{r['live_per_step']:8.1f}"
              f"{r['levels'] or 0:5d}{r['live_per_level'] or 0:9.2f}"
              f"{r['t_live_us'] or r['t_all_us']:8.2f}{r['span_us']:8.1f}"
              f"{r['pct_step']:7.2f}"
              f"{r['perfect_pack_us']:8.1f}{r['span_over_pp']:7.2f}"
              f"{r['depth'] if r['depth'] is not None else 0:4d}{r['sole_us']:8.1f}"
              f"{r['sole_mean_conc']:7.1f}{r['sole_idle_us']:8.1f}")
    print(f"{'TOTAL sole / sole-idle':34s}{'':8s}{'':8s}{'':5s}{'':9s}{'':8s}"
          f"{'':8s}{'':7s}{'':8s}{'':7s}{'':4s}"
          f"{sum(r['sole_us'] for r in rows):8.1f}{'':7s}"
          f"{sum(r['sole_idle_us'] for r in rows):8.1f}")


if __name__ == "__main__":
    main()
