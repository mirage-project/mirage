#!/usr/bin/env python3
"""M3-I9x F1 ROOT-CAUSE probe (CPU-only, offline over the same raw npz).

Probe v1 (f1_threshold_layer_probe.py) established that task 241 dispatches a
STATIC 10240 launches/iteration (40 layers x 256 slots) and that the activated
count is threshold-sensitive: 10.07 at >=1us, 8.00 at >=2us.  This script
nails that down:

  A) exact per-LAYER decomposition -- task 241 emits exactly 256 events per
     layer and layers are strictly sequential in the task graph, so sorting an
     iteration's begins and chunking by 256 recovers layer identity WITHOUT a
     gap heuristic.  Reports per-layer long counts at both thresholds, so H5
     (per-layer heterogeneity) is decided rather than assumed.
  B) the excess population's identity -- duration/ block / layer distribution
     of the events in [1us, 16us), i.e. the ones the >=1us split counts and the
     >=16us split does not.
  C) the falsifier restated: min/max/mean of activated over EVERY decode-only
     iteration at the empty-vs-real separating threshold.
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
    ap.add_argument("--task-type", type=int, default=241)
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--slots-per-layer", type=int, default=256)
    ap.add_argument("--moe-n-splits", type=int, default=2)
    ap.add_argument("--last-n", type=int, default=32)
    ap.add_argument("--sep-threshold-ns", type=float, default=16000.0,
                    help="empty-vs-real separator (the bimodal gap).")
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
    lo = max(0, n_it - args.last_n)
    dec = [i for i in range(lo, n_it) if i < len(lab) and lab[i] == "decode_full"]
    n_live_of = {i: int(sim["iters"][i]["n_live"]) for i in dec}

    tt = args.task_type
    m = pairs["task_type"] == tt
    b, d, blk = pairs["begin"][m], pairs["dur"][m].astype(np.float64), pairs["block"][m]
    it = np.searchsorted(bounds, b, side="right") - 1

    denom = float(args.layers * args.moe_n_splits)
    spl = args.slots_per_layer
    per_layer_long1, per_layer_long_sep = [], []
    gap_layers, gap_blocks, gap_durs = [], [], []
    per_iter = []
    for i in dec:
        s = it == i
        bb, dd, kk = b[s], d[s], blk[s]
        o = np.argsort(bb, kind="stable")
        bb, dd, kk = bb[o], dd[o], kk[o]
        if len(bb) != args.layers * spl:
            per_iter.append(dict(iteration=int(i), malformed=int(len(bb))))
            continue
        layer = np.arange(len(bb)) // spl
        l1 = np.bincount(layer[dd >= 1000.0], minlength=args.layers)
        ls = np.bincount(layer[dd >= args.sep_threshold_ns],
                         minlength=args.layers)
        per_layer_long1.append(l1)
        per_layer_long_sep.append(ls)
        gm = (dd >= 1000.0) & (dd < args.sep_threshold_ns)
        gap_layers.append(np.bincount(layer[gm], minlength=args.layers))
        gap_blocks.append(kk[gm])
        gap_durs.append(dd[gm])
        per_iter.append(dict(
            iteration=int(i), n_live=n_live_of[i],
            activated_thr1us=float(l1.sum() / denom),
            activated_thr_sep=float(ls.sum() / denom),
            per_layer_sep_min=int(ls.min()), per_layer_sep_max=int(ls.max()),
            per_layer_1us_min=int(l1.min()), per_layer_1us_max=int(l1.max()),
        ))

    L1 = np.array(per_layer_long1)          # [n_dec, layers]
    LS = np.array(per_layer_long_sep)
    GL = np.array(gap_layers)
    gap_blocks_all = np.concatenate(gap_blocks) if gap_blocks else np.array([])
    gap_durs_all = np.concatenate(gap_durs) if gap_durs else np.array([])

    a1 = L1.sum(axis=1) / denom
    asep = LS.sum(axis=1) / denom
    out = dict(
        raw=os.path.abspath(args.raw), bs=bs, task_type=tt,
        n_iterations=n_it, n_decode_full=len(dec),
        sep_threshold_ns=args.sep_threshold_ns,
        moe_n_splits=args.moe_n_splits, layers=args.layers,
        slots_per_layer=spl,
        # --- C: the falsifier, both thresholds ---
        activated_thr1us=dict(mean=float(a1.mean()), min=float(a1.min()),
                              max=float(a1.max())),
        activated_thr_sep=dict(mean=float(asep.mean()), min=float(asep.min()),
                               max=float(asep.max())),
        n_live_set=sorted(set(n_live_of.values())),
        # --- A: per-layer, at the separating threshold ---
        per_layer_sep_mean=[float(x) for x in LS.mean(axis=0)],
        per_layer_sep_min=[int(x) for x in LS.min(axis=0)],
        per_layer_sep_max=[int(x) for x in LS.max(axis=0)],
        per_layer_sep_all_equal=bool(LS.min() == LS.max()),
        per_layer_sep_unique_values=sorted(int(x) for x in np.unique(LS)),
        # --- A': per-layer at the 1us threshold (where the excess lives) ---
        per_layer_1us_mean=[float(x) for x in L1.mean(axis=0)],
        per_layer_1us_min=[int(x) for x in L1.min(axis=0)],
        per_layer_1us_max=[int(x) for x in L1.max(axis=0)],
        # --- B: the [1us, sep) population ---
        gap_per_iter_mean=float(GL.sum(axis=1).mean()),
        gap_per_iter_over_denom=float(GL.sum(axis=1).mean() / denom),
        gap_per_layer_mean=[float(x) for x in GL.mean(axis=0)],
        gap_per_layer_cv=float(GL.mean(axis=0).std() / max(GL.mean(), 1e-9)),
        gap_dur_percentiles_ns={p: float(np.percentile(gap_durs_all, p))
                                for p in (0, 5, 25, 50, 75, 95, 100)}
        if len(gap_durs_all) else {},
        gap_n_distinct_blocks=int(len(np.unique(gap_blocks_all))),
        gap_block_hist_top10=[[int(v), int(c)] for v, c in
                              sorted(zip(*np.unique(gap_blocks_all,
                                                    return_counts=True)),
                                     key=lambda t: -t[1])[:10]]
        if len(gap_blocks_all) else [],
        per_iteration=per_iter,
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    p = {k: v for k, v in out.items()
         if k not in ("per_iteration", "per_layer_sep_mean", "per_layer_1us_mean",
                      "gap_per_layer_mean", "per_layer_sep_min",
                      "per_layer_sep_max", "per_layer_1us_min",
                      "per_layer_1us_max")}
    print(json.dumps(p, indent=2))
    print("per_layer_sep_mean :", [round(x, 3) for x in out["per_layer_sep_mean"]])
    print("per_layer_1us_mean :", [round(x, 2) for x in out["per_layer_1us_mean"]])
    print("gap_per_layer_mean :", [round(x, 2) for x in out["gap_per_layer_mean"]])
    return 0


if __name__ == "__main__":
    sys.exit(main())
