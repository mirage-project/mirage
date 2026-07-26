#!/usr/bin/env python3
"""raw_bs<N>_rep<R>.npz -> per-iteration attribution, per-task tables, trace.

Run separately from the capture so re-analysis costs no GPU.  Everything is
derived from the saved buffer; nothing is recomputed from memory of a run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import schedule_sim as SIM
import trace_lib as TL


def dead_gaps(pairs: dict, t0: int, t1: int, n_workers: int):
    """Intervals inside [t0,t1) where NO worker block is inside a task.
    Returns (starts, ends, predecessor task_type) -- the predecessor is the
    task type whose END is the gap start, i.e. what the machine was waiting
    on when everything went quiet."""
    m = (pairs["block"] < n_workers) & (pairs["begin"] < t1) & (pairs["end"] > t0)
    b = np.clip(pairs["begin"][m], t0, t1)
    e = np.clip(pairs["end"][m], t0, t1)
    tt = pairs["task_type"][m]
    if len(b) == 0:
        return np.array([t0]), np.array([t1]), np.array([-1])
    o = np.argsort(b, kind="stable")
    b, e, tt = b[o], e[o], tt[o]
    emax = np.maximum.accumulate(e)
    new = np.empty(len(b), dtype=bool)
    new[0] = True
    new[1:] = b[1:] > emax[:-1]
    idx = np.flatnonzero(new)
    seg_b = b[idx]
    seg_e = np.maximum.reduceat(e, idx)
    gs = np.concatenate(([t0], seg_e))
    ge = np.concatenate((seg_b, [t1]))
    keep = ge > gs
    # predecessor: the task whose end equals the gap start
    pred = np.full(len(gs), -1, dtype=np.int64)
    for k in range(1, len(gs)):
        seg = slice(idx[k - 1], idx[k] if k < len(idx) else len(b))
        if seg.stop > seg.start:
            j = int(np.argmax(e[seg]))
            pred[k] = tt[seg][j]
    return gs[keep], ge[keep], pred[keep]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--n-workers", type=int, default=128)
    ap.add_argument("--warm-iters", type=int, default=4,
                    help="decode iterations skipped after prefill ends")
    ap.add_argument("--steady-iters", type=int, default=56,
                    help="decode iterations analysed (must stay inside the "
                         "first 64, where every request is still live)")
    ap.add_argument("--perfetto-iters", type=int, default=3)
    ap.add_argument("--no-perfetto", action="store_true")
    args = ap.parse_args(argv)

    meta = json.load(open(args.meta))
    names = json.load(open(args.names))
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

    plens = meta["prompt_lens"]
    bs = meta["batch_size"]
    slot_lens = plens + [plens[i % len(plens)] for i in range(len(plens), bs)]
    # Exact replay of prepare_next_batch (schedule_sim) labels every iteration;
    # its predicted iteration count is checked against the trace's own
    # BEGIN_TASK_GRAPH count below, so the labelling is falsifiable, not
    # assumed.
    sim = SIM.simulate(slot_lens, meta["mbt"], meta["max_seq_length"])
    lab = SIM.label(sim)
    n_prefill_mixed = sum(1 for x in lab if x in ("prefill", "mixed"))

    att = TL.attribute(pairs, bounds, args.n_workers)

    lo, hi = SIM.steady_window(sim)
    lo = min(lo + args.warm_iters, hi)
    hi = min(hi, n_it)
    if hi - lo > args.steady_iters:
        hi = lo + args.steady_iters
    if hi <= lo:
        lo, hi = max(n_it - 8, 0), n_it
    regime = SIM.regime_key(sim["iters"][lo]) if lo < len(sim["iters"]) else None

    us = 1e-3
    sel = slice(lo, hi)
    n = hi - lo
    step_us = float(att["iter_ns"][sel].mean()) * us
    summary = dict(
        tag=meta["tag"], batch_size=bs, rep=meta["rep"],
        n_events=int(ev["n_events"]), n_iterations_traced=n_it,
        header_nblocks=int(ev["nblocks"]), header_ngroups=int(ev["ngroups"]),
        n_worker_blocks=int((np.unique(pairs["block"]) < args.n_workers).sum()),
        n_sched_blocks=int((np.unique(pairs["block"]) >= args.n_workers).sum()),
        dropped_dangling_begin=int(pairs["dropped_begin"]),
        dropped_unmatched_end=int(pairs["dropped_end"]),
        sim_iterations=int(sim["n_iterations"]),
        trace_iterations=int(n_it + 1),
        schedule_model_agrees=bool(sim["n_iterations"] == n_it + 1),
        n_prefill_or_mixed=int(n_prefill_mixed),
        n_decode_full=int(sum(1 for x in lab if x == "decode_full")),
        n_decode_draining=int(sum(1 for x in lab if x == "decode_draining")),
        first_retirement_iter=next((r["iteration"] for r in sim["iters"]
                                    if r["n_live"] < bs), None),
        steady_window=[int(lo), int(hi)],
        steady_regime_live_prefill_decode_tokens=list(regime) if regime else None,
        # --- the attribution, per decode step, averaged over the window ---
        step_us=step_us,
        step_us_p50=float(np.percentile(att["iter_ns"][sel], 50)) * us,
        step_us_min=float(att["iter_ns"][sel].min()) * us,
        step_us_max=float(att["iter_ns"][sel].max()) * us,
        task_sum_us=float(att["task_ns"][sel].mean()) * us,
        sched_events_sum_us=float(att["sched_events_ns"][sel].mean()) * us,
        task_sum_per_worker_us=float(att["task_ns"][sel].mean()) * us / args.n_workers,
        perfect_pack_us=float(att["perfect_pack_ns"][sel].mean()) * us,
        dead_all_idle_us=float(att["dead_ns"][sel].mean()) * us,
        prepare_batch_us=float(att["prepare_batch_ns"][sel].mean()) * us,
        worker_idle_us=float(att["worker_idle_ns"][sel].mean()) * us,
        busy_any_us=float(att["busy_any_ns"][sel].mean()) * us,
        occupancy=float(att["occupancy"][sel].mean()),
        tasks_per_step=float(att["n_task"][sel].mean()),
        sched_events_per_step=float(att["n_sched_events"][sel].mean()),
        # --- what the steady regime means in tokens ---
        tokens_per_step=(regime[3] if regime else None),
        decode_tokens_per_step=(regime[2] if regime else None),
        decode_tokens_per_s=((regime[2] / (step_us * 1e-6))
                             if regime and step_us else None),
        # --- prefill / mixed phase, for the WY-UT + mbt levers ---
        prefill_step_us=(float(att["iter_ns"][:n_prefill_mixed].mean()) * us
                         if n_prefill_mixed > 0 else None),
        prefill_total_ms=(float(att["iter_ns"][:n_prefill_mixed].sum()) * 1e-6
                          if n_prefill_mixed > 0 else None),
        drain_total_ms=float(
            att["iter_ns"][[i for i in range(n_it)
                            if i < len(lab) and lab[i] == "decode_draining"]
                           ].sum()) * 1e-6,
        # --- closure vs the independent CUDA-event wall clock ---
        wall_ms_cuda_event=meta["waves"][0]["wall_ms"],
        trace_span_ms=float(bounds[-1] - bounds[0]) * 1e-6,
    )
    summary["closure_error_pct"] = (
        100.0 * (summary["trace_span_ms"] - summary["wall_ms_cuda_event"])
        / summary["wall_ms_cuda_event"])
    summary["residual_pct_of_step"] = 100.0 * (
        summary["step_us"] - summary["perfect_pack_us"]
        - summary["dead_all_idle_us"] - summary["worker_idle_us"]) / summary["step_us"]

    # per-task-type table over the steady window
    table = TL.per_task_table(pairs, bounds, lo, hi, args.n_workers, names)
    buckets = {}
    for r in table:
        b = buckets.setdefault(r["bucket"], dict(
            bucket=r["bucket"], n_per_iter=0.0, total_us_per_iter=0.0,
            per_worker_us_per_iter=0.0, task_types=[]))
        b["n_per_iter"] += r["n_per_iter"]
        b["total_us_per_iter"] += r["total_us_per_iter"]
        b["per_worker_us_per_iter"] += r["per_worker_us_per_iter"]
        b["task_types"].append(r["task_type"])
    bucket_rows = sorted(buckets.values(), key=lambda r: -r["total_us_per_iter"])

    # stall structure: the dead gaps inside one representative steady step
    mid = (lo + hi) // 2
    gs, ge, pred = dead_gaps(pairs, int(bounds[mid]), int(bounds[mid + 1]),
                             args.n_workers)
    gap_len = (ge - gs).astype(np.float64) * us
    order = np.argsort(-gap_len)
    stalls = [dict(start_us=float((gs[i] - bounds[mid]) * us),
                   len_us=float(gap_len[i]),
                   after_task=names.get(str(int(pred[i])), str(int(pred[i]))))
              for i in order[:15]]
    by_pred = {}
    for i in range(len(gap_len)):
        k = names.get(str(int(pred[i])), str(int(pred[i])))
        e = by_pred.setdefault(k, [0, 0.0])
        e[0] += 1
        e[1] += float(gap_len[i])
    stall_by_pred = sorted(
        ({"after_task": k, "n_gaps": v[0], "total_us": v[1]}
         for k, v in by_pred.items()), key=lambda r: -r["total_us"])[:15]

    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + "_attrib.json", "w") as f:
        json.dump(dict(summary=summary, buckets=bucket_rows, per_task=table,
                       stall_gaps_one_step=stalls,
                       stall_by_predecessor_one_step=stall_by_pred,
                       n_dead_gaps_one_step=int(len(gap_len)),
                       dead_gap_total_us_one_step=float(gap_len.sum())), f,
                  indent=2)

    with open(str(out) + "_iters.csv", "w") as f:
        f.write("iter,phase,n_live,n_prefill,n_decode,tokens,iter_us,"
                "task_sum_us,sched_events_us,busy_any_us,"
                "dead_us,prepare_batch_us,perfect_pack_us,worker_idle_us,"
                "occupancy,n_tasks\n")
        for i in range(n_it):
            phase = lab[i] if i < len(lab) else "?"
            r = sim["iters"][i] if i < len(sim["iters"]) else dict(
                n_live=0, n_prefill=0, n_decode_active=0, tokens=0)
            f.write(f"{i},{phase},{r['n_live']},{r['n_prefill']},"
                    f"{r['n_decode_active']},{r['tokens']},"
                    f"{att['iter_ns'][i]*us:.2f},"
                    f"{att['task_ns'][i]*us:.2f},"
                    f"{att['sched_events_ns'][i]*us:.2f},"
                    f"{att['busy_any_ns'][i]*us:.2f},"
                    f"{att['dead_ns'][i]*us:.2f},"
                    f"{att['prepare_batch_ns'][i]*us:.2f},"
                    f"{att['perfect_pack_ns'][i]*us:.2f},"
                    f"{att['worker_idle_ns'][i]*us:.2f},"
                    f"{att['occupancy'][i]:.4f},{att['n_task'][i]}\n")

    if not args.no_perfetto:
        k = args.perfetto_iters
        t0, t1 = int(bounds[mid]), int(bounds[min(mid + k, n_it)])
        ne = TL.export_window_perfetto(ev, t0, t1,
                                       str(out) + "_decode_window.perfetto-trace",
                                       names)
        summary["perfetto_window_events"] = ne
        summary["perfetto_window_iters"] = [mid, min(mid + k, n_it)]
        with open(str(out) + "_attrib.json") as f:
            doc = json.load(f)
        doc["summary"] = summary
        with open(str(out) + "_attrib.json", "w") as f:
            json.dump(doc, f, indent=2)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
