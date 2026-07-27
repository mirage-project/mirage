#!/usr/bin/env python3
"""Why are only ~1/3 of the 128 worker blocks busy?  Two discriminating views
of one steady step:

1. CONCURRENCY PROFILE -- how many worker blocks are inside a task as a
   function of time, and how the step's wall time splits across concurrency
   levels.  A step spent at low concurrency is a *graph-width* problem; a step
   spent at high concurrency with a long tail is an imbalance problem.
2. WORKER GAP STRUCTURE -- per worker, the idle interval between the END of
   one task and the START of the next.  `PROFILER_EVENT_START` fires after the
   dependency-wait loop, so these gaps contain both "queue was empty"
   (dispatch rate) and "waiting for a predecessor event" (dependency).  Many
   small gaps => dispatch/latency bound; few long gaps => dependency bound.

Also reports, per task type, the mean concurrency while that task type is
running -- which localises the low-width regions to actual layers.
"""
from __future__ import annotations

import json
import sys

import numpy as np

import trace_lib as TL
import schedule_sim as SIM


def main():
    raw, meta_p, names_p, out_p = sys.argv[1:5]
    meta = json.load(open(meta_p))
    names = json.load(open(names_p))
    z = np.load(raw)
    idx, val = z["idx"], z["val"]
    buf = np.zeros(int(idx.max()) + 1, dtype=np.uint64)
    buf[idx.astype(np.int64)] = val
    buf[:1] = z["header"].view(np.uint64)
    ev = TL.decode_events(buf)
    del buf
    p = TL.pair_events(ev)
    bounds = TL.iteration_bounds(p)
    n_it = len(bounds) - 1

    bs = meta["batch_size"]
    plens = meta["prompt_lens"]
    slots = plens + [plens[i % len(plens)] for i in range(len(plens), bs)]
    sim = SIM.simulate(slots, meta["mbt"], meta["max_seq_length"])
    lo, hi = SIM.steady_window(sim)
    hi = min(hi, n_it)
    mid = min(lo + (hi - lo) // 2, n_it - 1)
    NW = 128
    t0, t1 = int(bounds[mid]), int(bounds[mid + 1])
    step_ns = t1 - t0

    m = (p["block"] < NW) & (p["begin"] < t1) & (p["end"] > t0)
    b = np.clip(p["begin"][m], t0, t1)
    e = np.clip(p["end"][m], t0, t1)
    tt = p["task_type"][m]
    blk = p["block"][m]

    # --- 1. concurrency profile via a +1/-1 sweep ---
    ts = np.concatenate([b, e])
    dv = np.concatenate([np.ones(len(b), np.int32), -np.ones(len(e), np.int32)])
    o = np.argsort(ts, kind="stable")
    ts, dv = ts[o], dv[o]
    conc = np.cumsum(dv)
    dt = np.diff(np.concatenate([ts, [t1]]))
    lvl = conc
    hist = np.bincount(np.clip(lvl, 0, NW), weights=dt.astype(np.float64),
                       minlength=NW + 1)
    tot = hist.sum()
    mean_conc = float((np.arange(NW + 1) * hist).sum() / max(tot, 1))

    def band(a, bnd):
        return float(hist[a:bnd].sum()) / 1e3

    out = dict(
        batch_size=bs, iteration=int(mid), step_us=step_ns / 1e3,
        regime=list(SIM.regime_key(sim["iters"][mid])),
        mean_concurrency=mean_conc,
        p50_concurrency=float(np.searchsorted(np.cumsum(hist), tot * 0.5)),
        us_at_concurrency=dict(
            zero=band(0, 1), c1_4=band(1, 5), c5_16=band(5, 17),
            c17_32=band(17, 33), c33_64=band(33, 65), c65_96=band(65, 97),
            c97_127=band(97, 128), c128=band(128, 129)),
        concurrency_hist_us=[round(float(x) / 1e3, 2) for x in hist],
    )

    # --- 2. per-worker gap structure ---
    gaps, gap_after = [], []
    order = np.lexsort((b, blk))
    bb, ee, kk, tk = b[order], e[order], blk[order], tt[order]
    same = kk[1:] == kk[:-1]
    g = bb[1:] - ee[:-1]
    sel = same & (g > 0)
    gaps = g[sel].astype(np.float64)
    gap_after = tk[:-1][sel]
    # leading/trailing idle per worker
    first_idx = np.flatnonzero(np.concatenate([[True], ~same]))
    last_idx = np.flatnonzero(np.concatenate([~same, [True]]))
    lead = (bb[first_idx] - t0).astype(np.float64)
    trail = (t1 - ee[last_idx]).astype(np.float64)
    out["gaps"] = dict(
        n_gaps=int(len(gaps)),
        gaps_per_worker=float(len(gaps)) / NW,
        total_gap_us_per_worker=float(gaps.sum()) / 1e3 / NW,
        lead_us_per_worker=float(lead.sum()) / 1e3 / NW,
        trail_us_per_worker=float(trail.sum()) / 1e3 / NW,
        mean_gap_us=float(gaps.mean()) / 1e3 if len(gaps) else 0.0,
        p50_gap_us=float(np.percentile(gaps, 50)) / 1e3 if len(gaps) else 0.0,
        p90_gap_us=float(np.percentile(gaps, 90)) / 1e3 if len(gaps) else 0.0,
        p99_gap_us=float(np.percentile(gaps, 99)) / 1e3 if len(gaps) else 0.0,
        max_gap_us=float(gaps.max()) / 1e3 if len(gaps) else 0.0,
    )
    edges = np.array([0, .5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1e9]) * 1e3
    h, _ = np.histogram(gaps, bins=edges)
    out["gap_hist"] = [
        dict(lo_us=float(edges[i] / 1e3), hi_us=float(edges[i + 1] / 1e3),
             n=int(h[i]),
             total_us=float(gaps[(gaps >= edges[i]) & (gaps < edges[i + 1])].sum()) / 1e3)
        for i in range(len(h)) if h[i]]
    # which task type precedes the biggest total gap time
    agg = {}
    for t in np.unique(gap_after):
        s = gap_after == t
        agg[names.get(str(int(t)), str(int(t)))] = dict(
            n=int(s.sum()), total_us=float(gaps[s].sum()) / 1e3,
            mean_us=float(gaps[s].mean()) / 1e3)
    out["gap_after_task"] = dict(sorted(agg.items(),
                                        key=lambda kv: -kv[1]["total_us"])[:12])

    # --- 3. mean concurrency while each task type runs ---
    per = {}
    for t in np.unique(tt):
        s = tt == t
        # sample concurrency at each task's midpoint
        mids = ((b[s] + e[s]) // 2)
        ci = np.searchsorted(ts, mids, side="right") - 1
        ci = np.clip(ci, 0, len(conc) - 1)
        per[names.get(str(int(t)), str(int(t)))] = dict(
            n=int(s.sum()),
            total_us=float((e[s] - b[s]).sum()) / 1e3,
            wall_span_us=float(TL.union_length(b[s], e[s])) / 1e3,
            mean_concurrency_during=float(conc[ci].mean()))
    out["per_task_concurrency"] = dict(
        sorted(per.items(), key=lambda kv: -kv[1]["total_us"]))

    with open(out_p, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("concurrency_hist_us",)}, indent=2))


if __name__ == "__main__":
    main()
