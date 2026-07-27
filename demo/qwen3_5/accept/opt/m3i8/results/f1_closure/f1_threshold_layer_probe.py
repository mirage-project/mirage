#!/usr/bin/env python3
"""M3-I9x F1 root-cause probe (CPU-only, reruns over the SAME raw npz that
f1_per_iteration.py consumed).

Three questions the per-iteration JSON cannot answer, all answerable offline:

  Q1 (instrument)  is the >=1us "long" split manufacturing the excess?
                   -> full duration histogram for task 241/242, plus the
                      per-iteration activated count re-derived at 7 thresholds.
  Q2 (dispatch)    is the number of task-241 events per iteration DYNAMIC
                   (== n_active * splits * layers, i.e. the count comes from
                   mpk_active_expert_ids[NUM_EXPERTS]) or FIXED (a static grid
                   where "long" means "found rows")?  Dynamic => the excess is
                   in the ACTIVE-EXPERT COUNT itself.
  Q3 (per-layer)   is the excess spread evenly over the 40 layers or carried by
                   a subset?  Within an iteration the task graph is sequential
                   in layer order, so clustering task-241 begins by time gives
                   40 layer groups.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

OPT = os.environ.get("MPK_OPT_DIR")
if not OPT:
    raise SystemExit("set MPK_OPT_DIR to demo/qwen3_5/accept/opt")
sys.path.insert(0, OPT)

import trace_lib as TL  # noqa: E402
import schedule_sim as SIM  # noqa: E402

THRESH_NS = [0.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0, 32000.0]


def load_pairs(raw):
    z = np.load(raw)
    idx, val = z["idx"], z["val"]
    header = z["header"] if "header" in z else None
    n_slots = int(idx.max()) + 1
    buf = np.zeros(n_slots, dtype=np.uint64)
    buf[idx.astype(np.int64)] = val
    if header is not None:
        buf[:1] = header.view(np.uint64)
    ev = TL.decode_events(buf)
    del buf
    return TL.pair_events(ev)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--moe-n-splits", type=int, default=2)
    ap.add_argument("--last-n", type=int, default=32)
    args = ap.parse_args(argv)

    meta = json.load(open(args.meta))
    pairs = load_pairs(args.raw)
    bounds = TL.iteration_bounds(pairs)
    n_it = len(bounds) - 1

    plens = meta["prompt_lens"]
    bs = meta["batch_size"]
    slot_lens = plens + [plens[i % len(plens)] for i in range(len(plens), bs)]
    sim = SIM.simulate(slot_lens, meta["mbt"], meta["max_seq_length"])
    lab = SIM.label(sim)

    lo, hi = max(0, n_it - args.last_n), n_it
    # restrict to genuinely decode_full tail iterations
    dec = [i for i in range(lo, hi) if i < len(lab) and lab[i] == "decode_full"]

    out = dict(raw=os.path.abspath(args.raw), bs=bs, n_it=n_it,
               tail=[lo, hi], n_decode_full_in_tail=len(dec))

    denom = float(args.layers * args.moe_n_splits)
    for tt in (241, 242, 260):
        m = pairs["task_type"] == tt
        if not m.any():
            continue
        d = pairs["dur"][m].astype(np.float64)
        b = pairs["begin"][m]
        it = np.searchsorted(bounds, b, side="right") - 1
        rec = {}
        # ---- Q1 histogram (ns log-ish buckets) ----
        edges = [0, 100, 250, 500, 750, 1000, 1500, 2000, 4000, 8000, 16000,
                 24000, 32000, 48000, 64000, 128000, 1 << 30]
        h, _ = np.histogram(d, bins=edges)
        rec["hist_edges_ns"] = edges[:-1]
        rec["hist_counts_all"] = [int(x) for x in h]
        rec["dur_percentiles_ns"] = {p: float(np.percentile(d, p))
                                     for p in (1, 5, 25, 50, 75, 90, 95, 99,
                                               99.5, 99.9)}
        # ---- Q2 dispatch count ----
        sel_dec = np.isin(it, dec)
        n_per_it_all = np.bincount(it[sel_dec], minlength=n_it)[dec]
        rec["events_per_decode_iter_mean"] = float(n_per_it_all.mean())
        rec["events_per_decode_iter_min"] = int(n_per_it_all.min())
        rec["events_per_decode_iter_max"] = int(n_per_it_all.max())
        rec["events_per_decode_iter_over_denom"] = float(
            n_per_it_all.mean() / denom)
        # ---- Q1b activated at each threshold ----
        by_thr = {}
        for thr in THRESH_NS:
            s = sel_dec & (d >= thr)
            c = np.bincount(it[s], minlength=n_it)[dec].astype(np.float64)
            a = c / denom
            by_thr[str(int(thr))] = dict(
                mean=float(a.mean()), min=float(a.min()), max=float(a.max()),
                median=float(np.median(a)))
        rec["activated_by_threshold_ns"] = by_thr
        # ---- long-task duration stats (>=1us) ----
        lg = d >= 1000.0
        if lg.any():
            rec["long_dur_stats_ns"] = dict(
                n=int(lg.sum()), mean=float(d[lg].mean()),
                p1=float(np.percentile(d[lg], 1)),
                p5=float(np.percentile(d[lg], 5)),
                p50=float(np.percentile(d[lg], 50)),
                p95=float(np.percentile(d[lg], 95)))
        # count of events in the "gap" 1us..16us during decode iters
        gap = sel_dec & (d >= 1000.0) & (d < 16000.0)
        real = sel_dec & (d >= 16000.0)
        rec["decode_gap_1to16us_per_iter"] = float(gap.sum() / max(len(dec), 1))
        rec["decode_ge16us_per_iter"] = float(real.sum() / max(len(dec), 1))
        rec["decode_gap_over_denom"] = float(gap.sum() / max(len(dec), 1) / denom)
        rec["decode_ge16us_over_denom"] = float(real.sum() / max(len(dec), 1)
                                                / denom)
        out[f"task_{tt}"] = rec

    # ---- Q3 per-layer clustering on task 241, one representative iteration ----
    m241 = pairs["task_type"] == 241
    b241 = pairs["begin"][m241]
    d241 = pairs["dur"][m241].astype(np.float64)
    it241 = np.searchsorted(bounds, b241, side="right") - 1
    per_layer = []
    for i in dec[:8]:
        s = it241 == i
        bb = np.sort(b241[s])
        if len(bb) == 0:
            continue
        # cluster: split where the inter-begin gap exceeds the median gap * K
        gaps = np.diff(bb)
        # choose the (layers*splits - 1) largest gaps as cluster separators
        nsep = args.layers * args.moe_n_splits - 1
        if len(gaps) <= nsep:
            continue
        thr_gap = np.sort(gaps)[-nsep]
        cut = np.flatnonzero(gaps >= thr_gap) + 1
        groups = np.split(np.arange(len(bb)), cut)
        sizes = [int(len(g)) for g in groups]
        per_layer.append(dict(iteration=int(i), n_events=int(len(bb)),
                              n_groups=len(groups), group_sizes=sizes))
    out["per_layer_clustering_task241"] = per_layer

    # long-only clustering (the >=1us population) -- what the oracle counts
    per_layer_long = []
    for i in dec[:8]:
        s = (it241 == i) & (d241 >= 1000.0)
        bb = np.sort(b241[s])
        if len(bb) < 10:
            continue
        gaps = np.diff(bb)
        nsep = args.layers * args.moe_n_splits - 1
        if len(gaps) <= nsep:
            continue
        thr_gap = np.sort(gaps)[-nsep]
        cut = np.flatnonzero(gaps >= thr_gap) + 1
        groups = np.split(np.arange(len(bb)), cut)
        sizes = [int(len(g)) for g in groups]
        per_layer_long.append(dict(iteration=int(i), n_long=int(len(bb)),
                                   n_groups=len(groups), group_sizes=sizes))
    out["per_layer_clustering_task241_long"] = per_layer_long

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    printable = {k: v for k, v in out.items()
                 if not k.startswith("per_layer_clustering")}
    print(json.dumps(printable, indent=2)[:12000])
    print("\n-- per-layer group sizes (all task241 events) --")
    for r in out["per_layer_clustering_task241"][:3]:
        print(r["iteration"], r["n_events"], r["n_groups"], r["group_sizes"])
    print("\n-- per-layer group sizes (long >=1us only) --")
    for r in out["per_layer_clustering_task241_long"][:3]:
        print(r["iteration"], r["n_long"], r["n_groups"], r["group_sizes"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
