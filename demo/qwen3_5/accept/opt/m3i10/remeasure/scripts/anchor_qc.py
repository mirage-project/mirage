#!/usr/bin/env python3
"""M3-I10 remeasure_spec.md sec 5 QC + sec 2 per-call-site split, all CPU.

1. Anchor QC: TASK_BEGIN_TASK_GRAPH (type 10) fires once/step; integrate over
   [first ts, last ts) => count-1 complete steps (trace_lib.iteration_bounds
   already does exactly this). Assert every task type's per-step count is an
   integer equal to the compiled task graph's static call-site count.
   Report max |count/step - round(count/step)|; > ~0.02 invalidates the
   window.
2. Cross-check against parse_profile.py's warm/steady window mean step_us.
3. Per-call-site split of task 253 (in_proj_ba x30 / MoE router gate x40 /
   lm_head x1, distinguished by output base_ptr: *_gdn_ba / *_router_logits /
   argmax_in) and task 279 (6 GEMM shapes, distinguished by output base_ptr
   suffix). MPK dispatch is deterministic round-robin over the FULL graph
   (position i -> worker i % n_workers, mod a constant offset that cancels in
   a rotation search) and each worker executes its queue strictly in
   arrival/graph order, so a worker's observed task-<T> durations (time
   order, per iteration) line up positionally with that worker's static
   task-<T> entries (graph-index order). Validated empirically per run (see
   `rotation_validation`) before trusting the split, not assumed.

Usage:
    python3 anchor_qc.py --raw R.npz --meta M.json --names N.json \
        --graph task_graph_rank0.json --out out.json \
        [--warm-iters 8 --steady-iters 80] [--n-workers 128]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# trace_lib.py lives in the isolated mirage-rm clone's opt/ dir (single
# source of truth -- no duplicate copy here to drift from it).
sys.path.insert(0, os.environ.get(
    "M3I10RM_OPT_DIR",
    str(Path.home() / "mpk-qwen35" / "mirage-rm" / "demo" / "qwen3_5"
        / "accept" / "opt")))
import trace_lib as TL  # noqa: E402
import schedule_sim as SIM  # noqa: E402

SITE_253 = {
    "argmax_in": "lm_head",
    "_router_logits": "moe_router_gate",   # suffix match, layer_N_router_logits
    "_gdn_ba": "in_proj_ba",               # suffix match, layer_N_gdn_ba
}
# task 279 site kinds are whatever distinct output base_ptr suffixes the
# compiled graph actually has -- discovered empirically, not hardcoded, so a
# graph change can't silently mislabel a site.


def classify(base_ptr: str, table: dict) -> str:
    for suf, label in table.items():
        if base_ptr == suf or base_ptr.endswith(suf):
            return label
    return f"UNCLASSIFIED:{base_ptr}"


def site_suffix(base_ptr: str) -> str:
    import re
    return re.sub(r"^layer_[0-9]+_", "", base_ptr)


def load_buf(raw_path):
    z = np.load(raw_path)
    idx, val = z["idx"], z["val"]
    buf = np.zeros(int(idx.max()) + 1, dtype=np.uint64)
    buf[idx.astype(np.int64)] = val
    if "header" in z:
        buf[:1] = z["header"].view(np.uint64)
    return buf


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--graph", required=True, help="task_graph_rank0.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-workers", type=int, default=128)
    ap.add_argument("--warm-iters", type=int, default=8)
    ap.add_argument("--steady-iters", type=int, default=80)
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
    bounds = TL.iteration_bounds(pairs)
    n_it = len(bounds) - 1

    # ---------------- 1. anchor QC: full-span per-step counts vs static ----
    static_counts = Counter(t["task_type"] for t in graph["all_tasks"])
    table_full = TL.per_task_table(pairs, bounds, 0, n_it, NW, names)
    obs = {r["task_type"]: r["n_per_iter"] for r in table_full}

    all_types = sorted(set(obs) | set(static_counts))
    qc_rows = []
    max_frac_err = 0.0
    n_mismatch = 0
    for t in all_types:
        o = obs.get(t, 0.0)
        s = static_counts.get(t, 0)
        rounded = round(o)
        frac_err = abs(o - rounded)
        mismatch = (rounded != s)
        max_frac_err = max(max_frac_err, frac_err)
        if mismatch:
            n_mismatch += 1
        qc_rows.append(dict(
            task_type=t, name=names.get(str(t), f"UNKNOWN_{t}"),
            observed_per_step=round(o, 4), rounded=rounded,
            static_call_sites=s, frac_err=round(frac_err, 4),
            mismatch=mismatch))
    qc_rows.sort(key=lambda r: -r["frac_err"])

    anchor_n_per_step = obs.get(10, None)  # TASK_BEGIN_TASK_GRAPH
    anchor_qc = dict(
        n_iterations_full_span=n_it,
        task_begin_task_graph_per_step=anchor_n_per_step,
        task_begin_task_graph_is_1_0=(anchor_n_per_step == 1.0),
        max_frac_err_over_all_types=round(max_frac_err, 4),
        threshold=args.frac_err_threshold,
        window_valid=(max_frac_err <= args.frac_err_threshold),
        n_task_types_mismatched_static_count=n_mismatch,
        rows=qc_rows,
    )

    # ---------------- 2. cross-check vs parse_profile.py windowing ---------
    # EXACT replica of parse_profile.py's own steady-window derivation (same
    # schedule_sim call, same warm/steady clipping) -- a fixed [warm_iters,
    # warm_iters+steady_iters) offset from iteration 0 would still overlap
    # the PREFILL phase (16 iterations at bs1) and silently contaminate both
    # this crosscheck and the call-site split below.
    plens = meta["prompt_lens"]
    bs = meta["batch_size"]
    slot_lens = plens + [plens[i % len(plens)] for i in range(len(plens), bs)]
    sim = SIM.simulate(slot_lens, meta["mbt"], meta["max_seq_length"])
    lo, hi = SIM.steady_window(sim)
    lo = min(lo + args.warm_iters, hi)
    hi = min(hi, n_it)
    if hi - lo > args.steady_iters:
        hi = lo + args.steady_iters
    if hi <= lo:
        lo, hi = max(n_it - 8, 0), n_it
    dur_it = np.diff(bounds)
    step_us_full = float(dur_it.mean()) / 1e3
    step_us_steady = float(dur_it[lo:hi].mean()) / 1e3
    windowing_crosscheck = dict(
        full_span_step_us=round(step_us_full, 2),
        steady_window=[lo, hi],
        steady_window_step_us=round(step_us_steady, 2),
        pct_diff=round(100 * (step_us_steady - step_us_full) / step_us_full, 4),
    )

    # ---------------- 3. per-call-site split (253, 279) ---------------------
    site_split = {}
    rotation_validation = {}
    for task_type, table in ((253, SITE_253), (279, None)):
        # static per-worker ordered sequence of (global_index, base_ptr)
        static_seq = defaultdict(list)
        for i, t in enumerate(graph["all_tasks"]):
            if t["task_type"] != task_type:
                continue
            outs = t.get("outputs") or []
            bp = outs[0]["base_ptr"] if outs else "?"
            w = i % NW
            static_seq[w].append(bp)

        # empirical alignment: observed count on block==w (trace, SUMMED over
        # all iterations, /n_it) vs len(static_seq[w]) -- searched over every
        # rotation offset (not assumed 0), because MPK's first_task_id is not
        # guaranteed to be a multiple of n_workers (e.g. a couple of leading
        # non-dispatched sentinel entries in all_tasks shift it). A constant
        # rotation cannot change which entries share a worker or their
        # relative order (i1 = i2 mod W iff (i1-c) = (i2-c) mod W for any c),
        # so searching for the best-matching c is exact model calibration,
        # not overfitting.
        tt_mask = pairs["task_type"] == task_type
        blk = pairs["block"][tt_mask]
        obs_total_by_block = np.bincount(blk, minlength=NW).astype(np.int64)
        # round SUM/n_it once (matches per_task_table's own rounding order),
        # not per-worker-then-sum (that compounds rounding error).
        obs_by_block = np.round(obs_total_by_block / float(n_it)).astype(np.int64)
        static_count_by_offset = np.zeros(NW, dtype=np.int64)
        for w in range(NW):
            static_count_by_offset[w] = len(static_seq.get(w, []))

        best_off, best_n = 0, -1
        for off in range(NW):
            rot = np.roll(static_count_by_offset, off)
            n = int((rot == obs_by_block).sum())
            if n > best_n:
                best_off, best_n = off, n
        match_frac = best_n / NW
        rotation_validation[task_type] = dict(
            offset=best_off, n_blocks_matched=best_n, n_blocks_total=NW,
            match_frac=round(match_frac, 4),
            aligned=bool(match_frac >= 0.90),
        )
        if not rotation_validation[task_type]["aligned"]:
            site_split[task_type] = {"ERROR": f"best rotation offset={best_off} only "
                                     f"matched {best_n}/{NW} blocks; split skipped"}
            continue
        # re-key static_seq by the DISCOVERED offset. Derivation: we accepted
        # offset `off` because static_count_by_offset[(k-off) % NW] ==
        # obs_by_block[k] for (almost) every observed worker k -- i.e. the
        # naive group keyed `w = (k-off) % NW` is actually TRUE worker k.
        # Solving for k: k = (w + off) % NW.
        if best_off != 0:
            rotated = defaultdict(list)
            for w, seq in static_seq.items():
                rotated[(w + best_off) % NW] = seq
            static_seq = rotated

        # extraction: per iteration, per worker, zip trace (time-order) with
        # static (graph-index order) for this task type only.
        lo_w, hi_w = lo, hi  # steady window, same as sec 2
        begin_all = pairs["begin"][tt_mask]
        dur_all = pairs["dur"][tt_mask]
        blk_all = pairs["block"][tt_mask]
        it_all = np.searchsorted(bounds, begin_all, side="right") - 1

        durs_by_label = defaultdict(list)
        intervals_by_label = defaultdict(list)
        classify_fn = ((lambda bp: classify(bp, table)) if table is not None
                       else site_suffix)
        for w in range(NW):
            seq = static_seq.get(w, [])
            if not seq:
                continue
            labels_w = [classify_fn(bp) for bp in seq]
            sel = (blk_all == w) & (it_all >= lo_w) & (it_all < hi_w)
            if not sel.any():
                continue
            b_w, d_w, i_w = begin_all[sel], dur_all[sel], it_all[sel]
            for it_val in range(lo_w, hi_w):
                m = i_w == it_val
                if not m.any():
                    continue
                order = np.argsort(b_w[m], kind="stable")
                d_sorted = d_w[m][order]
                b_sorted = b_w[m][order]
                if len(d_sorted) != len(labels_w):
                    continue  # dropped/truncated iteration, skip rather than misalign
                for lab, dd, bb in zip(labels_w, d_sorted, b_sorted):
                    durs_by_label[lab].append(float(dd))
                    intervals_by_label[lab].append((int(bb), int(bb + dd)))

        n_win_iters = hi_w - lo_w
        rows = []
        for lab, ds in sorted(durs_by_label.items(), key=lambda kv: -sum(kv[1])):
            ds = np.asarray(ds, dtype=np.float64)
            ivs = intervals_by_label[lab]
            b_arr = np.array([x[0] for x in ivs])
            e_arr = np.array([x[1] for x in ivs])
            wall = TL.union_length(b_arr, e_arr) if len(b_arr) else 0
            rows.append(dict(
                site=lab, n_instances_total=int(len(ds)),
                n_per_iter=round(len(ds) / n_win_iters, 2),
                sum_us_per_iter=round(float(ds.sum()) / 1e3 / n_win_iters, 2),
                mean_us=round(float(ds.mean()) / 1e3, 3) if len(ds) else None,
                wallspan_us_per_iter=round(wall / 1e3 / n_win_iters, 2),
            ))
        site_split[task_type] = dict(steady_window=[lo_w, hi_w], sites=rows)

    out = dict(
        tag=meta.get("tag"), batch_size=meta.get("batch_size"),
        n_workers=NW,
        anchor_qc=anchor_qc,
        windowing_crosscheck=windowing_crosscheck,
        rotation_validation={str(k): v for k, v in rotation_validation.items()},
        call_site_split={str(k): v for k, v in site_split.items()},
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"anchor QC: n_it={n_it} max_frac_err={max_frac_err:.4f} "
          f"(threshold {args.frac_err_threshold}) "
          f"window_valid={anchor_qc['window_valid']} "
          f"n_mismatched_types={n_mismatch}")
    print(f"windowing crosscheck: full={step_us_full:.2f}us "
          f"steady={step_us_steady:.2f}us pct_diff={windowing_crosscheck['pct_diff']:.4f}%")
    for tt in (253, 279):
        rv = rotation_validation[tt]
        print(f"task {tt}: best rotation offset={rv['offset']} matched "
              f"{rv['n_blocks_matched']}/{rv['n_blocks_total']} blocks "
              f"({rv['match_frac']:.1%}), aligned={rv['aligned']}")
        if isinstance(site_split[tt], dict) and "sites" in site_split[tt]:
            for r in site_split[tt]["sites"]:
                print(f"    {r['site']:20s} n/it={r['n_per_iter']:7.2f} "
                      f"sum_us/it={r['sum_us_per_iter']:8.2f} "
                      f"wallspan_us/it={r['wallspan_us_per_iter']:8.2f} "
                      f"mean_us={r['mean_us']}")
    return 0 if anchor_qc["window_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
