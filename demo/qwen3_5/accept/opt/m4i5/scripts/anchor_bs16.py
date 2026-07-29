#!/usr/bin/env python3
"""Resolve the bs16 anchor-QC failure M3-I7 flagged.

M3-I7 reported `max_frac_err = 0.4437` at bs16 and excluded the row from
ranking.  That statistic is the FRACTIONAL PART of the window's mean
tasks-per-step, so it answers "is the per-step count an integer" and not "is it
the right integer".  The compiled graph is static (15 360 GDN-recurrent tasks,
5 200 dense-fp8 tasks, ... every iteration regardless of regime), so the direct
test is: does each iteration of the trace contain exactly the static call-site
count?

This script prints that per iteration, so the deficit can be located rather
than averaged.  Also prints the trace-derived live-slot count per iteration
(from tasks that ran longer than the dead-task threshold), which is what the
window's regime label should rest on.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import trace_lib as TL  # noqa: E402

LIVE_US = 4.0


def main():
    raw, meta_p, graph_p, out_p = sys.argv[1:5]
    meta = json.load(open(meta_p))
    bs = meta["batch_size"]
    with open(graph_p) as f:
        g = json.load(f)
    from collections import Counter
    static = Counter(t["task_type"] for t in g["all_tasks"])
    del g

    z = np.load(raw)
    idx, val = z["idx"], z["val"]
    buf = np.zeros(int(idx.max()) + 1, dtype=np.uint64)
    buf[idx.astype(np.int64)] = val
    buf[:1] = z["header"].view(np.uint64)
    ev = TL.decode_events(buf)
    nblocks, ngroups = ev["nblocks"], ev["ngroups"]
    n_slots_used = int(idx.max()) + 1
    del buf
    p = TL.pair_events(ev)
    # per-track event counts vs the per-track allotment implied by the layout
    trk = ev["block"] * max(ngroups, 1) + ev["group"]
    per_track = np.bincount(trk)
    del ev
    bounds = TL.iteration_bounds(p)
    n_it = len(bounds) - 1
    it = np.searchsorted(bounds, p["begin"], side="right") - 1
    dur = p["dur"].astype(np.float64)

    watch = [279, 237, 241, 253, 257, 234]
    rows = {}
    for t in watch:
        m = (p["task_type"] == t) & (it >= 0) & (it < n_it)
        rows[t] = np.bincount(it[m], minlength=n_it)[:n_it]
        ml = m & (dur >= LIVE_US * 1e3)
        rows[(t, "live")] = np.bincount(it[ml], minlength=n_it)[:n_it]

    exact = {t: int((rows[t] == static[t]).sum()) for t in watch}
    bad = {t: np.flatnonzero(rows[t] != static[t]).tolist() for t in watch}
    out = dict(
        batch_size=bs, n_iterations=int(n_it),
        profiler=dict(nblocks=int(nblocks), ngroups=int(ngroups),
                      slots_written=n_slots_used,
                      slots_requested=meta.get("profiler_slots"),
                      per_track_allotment=int(meta.get("profiler_slots", 0)
                                              // max(nblocks * max(ngroups, 1), 1)),
                      per_track_events_max=int(per_track.max()),
                      per_track_events_p50=int(np.percentile(per_track, 50)),
                      n_tracks_at_or_over_allotment=int(
                          (per_track >= (meta.get("profiler_slots", 0)
                                         // max(nblocks * max(ngroups, 1), 1))).sum()),
                      dropped_begin=int(p["dropped_begin"]),
                      dropped_end=int(p["dropped_end"])),
        static={str(t): int(static[t]) for t in watch},
        n_iters_exact={str(t): exact[t] for t in watch},
        first_deficit_iter={str(t): (bad[t][0] if bad[t] else None) for t in watch},
        n_deficit_iters={str(t): len(bad[t]) for t in watch},
        deficit_iters_head={str(t): bad[t][:12] for t in watch},
        deficit_iters_tail={str(t): bad[t][-12:] for t in watch},
        live_slots_per_iter=dict(
            conv1d=(rows[(234, "live")] / (static[234] // bs)).round(2).tolist(),
            attn=(rows[(257, "live")] / (static[257] // bs)).round(2).tolist()),
    )
    with open(out_p, "w") as f:
        json.dump(out, f, indent=1)
    pr = dict(out)
    pr.pop("live_slots_per_iter")
    print(json.dumps(pr, indent=1))
    lc = np.array(out["live_slots_per_iter"]["conv1d"])
    print("live slots (conv1d) by iteration, sampled every 50:",
          np.rint(lc[::50]).astype(int).tolist())
    for lo, hi in ((720, 733),):
        print(f"window [{lo},{hi}) live slots:",
              np.rint(lc[lo:hi]).astype(int).tolist())
        for t in watch:
            print(f"   {t}: counts {sorted(set(rows[t][lo:hi].tolist()))} "
                  f"static {static[t]}")


if __name__ == "__main__":
    main()
