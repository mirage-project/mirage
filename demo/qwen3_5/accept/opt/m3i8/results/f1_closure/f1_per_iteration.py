#!/usr/bin/env python3
"""M3-I9x -- M3-I8 F1 per-iteration closure (VALIDATION.md post-verdict addendum #1/#2).

Extends opt/m3i8/analyze_m3i8.py's run-averaged mechanism inference
(``activated = nlong / (40 * moe_n_splits)``, nlong = count of TASK_MOE_W13_
FP8_BLOCKSCALE_SM100 "long" launches, i.e. duration >= 1us, a real tile rather
than a dispatched-but-empty task) to PER-ITERATION granularity, using the
BEGIN_TASK_GRAPH iteration boundaries opt/trace_lib.py already extracts for
the M3-I1 profiler buffer.

The existing method (opt/m3i8/analyze_m3i8.py -> opt/parse_profile.py's
per_task_table) only ever reports one number: n_long_per_iter AVERAGED over a
window.  That average is what produced the "9.6 / 16.9, over the strict
per-decode-row cap of 8 / 16" signal that the codex completion reviewer
refused to accept as closure (VALIDATION.md finding 1: the average cannot
distinguish "every decode iteration is over cap" from "prefill iterations
(live=chunk, ~87 activated) are mixed into an average with genuinely-capped
decode iterations").  This script computes the SAME formula per iteration
instead of once over a window, so the two hypotheses become distinguishable.

Usage:
    python3 f1_per_iteration.py --raw raw_bs1_rep0.npz --meta meta_bs1_rep0.json
        --out f1_bs1.json [--task-type 241] [--moe-n-splits 2] [--layers 40]
        [--last-n 32] [--long-threshold-ns 1000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.environ.get("MPK_OPT_DIR")
if not OPT:
    raise SystemExit("set MPK_OPT_DIR to demo/qwen3_5/accept/opt (trace_lib.py, "
                      "schedule_sim.py live there)")
sys.path.insert(0, OPT)

import schedule_sim as SIM  # noqa: E402
import trace_lib as TL  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--task-type", type=int, default=241,
                    help="TASK_MOE_W13_FP8_BLOCKSCALE_SM100, the mechanism "
                         "row analyze_m3i8.py reads (STAGES[241]).")
    ap.add_argument("--moe-n-splits", type=int, default=2,
                    help="v1/base arm value (analyze_m3i8.py SPLITS['v1']=2).")
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--last-n", type=int, default=32,
                    help="steady-decode tail window size to report the "
                         "per-iteration distribution over.")
    ap.add_argument("--long-threshold-ns", type=float, default=1000.0,
                    help="matches trace_lib.per_task_table's short/long split.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    meta = json.load(open(args.meta))
    z = np.load(args.raw)
    idx, val = z["idx"], z["val"]
    header = z["header"] if "header" in z else None

    n_slots = int(idx.max()) + 1
    buf = np.zeros(n_slots, dtype=np.uint64)
    buf[idx.astype(np.int64)] = val
    if header is not None:
        buf[:1] = header.view(np.uint64)

    ev = TL.decode_events(buf)
    del buf
    pairs = TL.pair_events(ev)
    bounds = TL.iteration_bounds(pairs)
    n_it = len(bounds) - 1
    if n_it <= 0:
        raise SystemExit(f"no complete iterations decoded from {args.raw}")

    plens = meta["prompt_lens"]
    bs = meta["batch_size"]
    slot_lens = plens + [plens[i % len(plens)] for i in range(len(plens), bs)]
    sim = SIM.simulate(slot_lens, meta["mbt"], meta["max_seq_length"])
    lab = SIM.label(sim)
    # Falsifiable cross-check, same one parse_profile.py uses: the schedule
    # replay's OWN predicted iteration count must match what BEGIN_TASK_GRAPH
    # actually produced, or the iteration boundaries below are not trustworthy.
    schedule_model_agrees = bool(sim["n_iterations"] == n_it + 1)

    # -- per-iteration long-task-241 count -> activated groups -------------
    m_tt = pairs["task_type"] == args.task_type
    m_long = pairs["dur"] >= args.long_threshold_ns
    sel = m_tt & m_long
    b_sel = pairs["begin"][sel]
    it_idx = np.searchsorted(bounds, b_sel, side="right") - 1
    in_win = (it_idx >= 0) & (it_idx < n_it)
    it_idx = it_idx[in_win]
    counts = np.bincount(it_idx, minlength=n_it).astype(np.float64)[:n_it]
    denom = float(args.layers * args.moe_n_splits)
    activated = counts / denom

    lo = max(0, n_it - args.last_n)
    hi = n_it
    tail = activated[lo:hi]
    tail_labels = [lab[i] if i < len(lab) else "?" for i in range(lo, hi)]
    tail_n_live = [int(sim["iters"][i]["n_live"]) if i < len(sim["iters"]) else None
                  for i in range(lo, hi)]
    tail_tokens = [int(sim["iters"][i]["tokens"]) if i < len(sim["iters"]) else None
                  for i in range(lo, hi)]

    full_run_avg = float(activated.mean())
    prefill_mixed_n = sum(1 for x in lab[:n_it] if x in ("prefill", "mixed"))
    decode_only_avg = (float(activated[prefill_mixed_n:].mean())
                      if prefill_mixed_n < n_it else None)

    out = dict(
        source_raw=os.path.abspath(args.raw),
        source_meta=os.path.abspath(args.meta),
        tag=meta.get("tag"), batch_size=bs, mbt=meta.get("mbt"),
        max_seq_length=meta.get("max_seq_length"),
        n_iterations_traced=n_it, sim_iterations=int(sim["n_iterations"]),
        schedule_model_agrees=schedule_model_agrees,
        task_type=args.task_type, moe_n_splits=args.moe_n_splits,
        layers=args.layers, long_threshold_ns=args.long_threshold_ns,
        # --- the closure: per-iteration distribution over the FINAL last_n ---
        tail_window=[int(lo), int(hi)],
        tail_labels=tail_labels, tail_n_live=tail_n_live,
        tail_tokens=tail_tokens,
        tail_activated=[round(float(x), 4) for x in tail],
        tail_min=float(tail.min()), tail_max=float(tail.max()),
        tail_mean=float(tail.mean()), tail_median=float(np.median(tail)),
        tail_all_decode_full=all(x == "decode_full" for x in tail_labels),
        # --- full-run reconciliation (prefill folded back in) ---
        full_run_average_activated=full_run_avg,
        full_run_iterations=n_it,
        n_prefill_or_mixed_iterations=prefill_mixed_n,
        decode_only_average_activated=decode_only_avg,
        # --- entire per-iteration series, for anyone re-deriving a plot ---
        per_iteration_activated_all=[round(float(x), 4) for x in activated],
        per_iteration_label_all=(lab[:n_it] if len(lab) >= n_it
                                 else lab + ["?"] * (n_it - len(lab))),
    )
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    printable = {k: v for k, v in out.items() if not k.startswith("per_iteration")}
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
