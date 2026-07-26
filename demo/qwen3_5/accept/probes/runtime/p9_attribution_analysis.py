"""P9 probe, analysis pass -- CPU-only, no GPU needed.

Reads the raw profiler CSVs produced by p9_profile_capture.py (r=4, r=8) plus
the step-1 (profiler-off) run_batch_perf.py JSON outputs (r=1,2,4,8), and
computes the inter-iteration gap vs summed-task-time attribution described in
v1-architecture.md S14 P9.

We read RAW NUMERIC task_type_id only (never the name string) because
mpk-gaps.md S4/S9 flags mirage.mpk.profiler_persistent.event_name_list as
stale (confirmed independently: it still lists TASK_SM100_TASK_END at 298,
but include/mirage/persistent_kernel/runtime_header.h pins it at 299) --
until M2-I10 lands the fix we trust only the numeric ids we cross-checked
directly against runtime_header.h:
    TASK_TERMINATE=0, TASK_BEGIN_TASK_GRAPH=10, TASK_SCHD_TASKS=200,
    TASK_SCHD_EVENTS=201, TASK_GET_EVENT=202, TASK_GET_NEXT_TASK=203,
    TASK_SCHD_PREPARE_BATCH=204.
TASK_SCHD_PREPARE_BATCH (204) is emitted exactly once per scheduler iteration
(persistent_kernel.cuh:1233/1242/1247, guarded by EVENT_END_OF_TASK_GRAPH) --
it is our per-iteration anchor. Everything else observed in the trace is
"compute" (a real dispatched task).
"""
import argparse
import bisect
import csv
import json
import os
import statistics

NON_COMPUTE_IDS = {0, 10, 200, 201, 202, 203, 204}
PREPARE_BATCH_ID = 204
WRAP = 1 << 32
WARMUP_ITERS_DROPPED = 2  # drop first N iteration buckets (JIT/cache warmup)


def load_rows(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["task_type_id"] = int(r["task_type_id"])
        r["block_idx"] = int(r["block_idx"])
        r["group_idx"] = int(r["group_idx"])
        r["event_no"] = int(r["event_no"])
        r["begin_ts"] = int(r["begin_ts"])
        r["end_ts"] = int(r["end_ts"])
        r["duration_ns"] = int(r["duration_ns"])
    return rows


def unwrap_group(rows_for_key):
    """Sort one (block,group,event_idx)-key's rows by event_no (which the
    runtime increments monotonically per dispatch) and correct any 32-bit
    %globaltimer_lo wraparound (see profiler_persistent.py export_to_csv
    docstring: raw ts wrap at ~4.3s) by adding WRAP whenever begin_ts goes
    backwards relative to event_no order."""
    rows_for_key = sorted(rows_for_key, key=lambda r: r["event_no"])
    offset = 0
    prev_begin = None
    out = []
    for r in rows_for_key:
        b = r["begin_ts"] + offset
        if prev_begin is not None and b < prev_begin - WRAP // 2:
            offset += WRAP
            b = r["begin_ts"] + offset
        e = r["end_ts"] + offset
        if e < b:  # this one event's own end wrapped past begin
            e += WRAP
        out.append({**r, "begin_ts": b, "end_ts": e})
        prev_begin = b
    return out


def unwrap_all(rows):
    """Apply unwrap_group per (block,group,event_idx) key, then return the
    globally-merged, absolute-ns list. All keys share one hardware
    %globaltimer_lo on a single GPU, so once each key's own wraps are
    resolved the absolute values are directly comparable across keys."""
    keyed = {}
    for r in rows:
        k = (r["block_idx"], r["group_idx"], r["task_type_id"])
        keyed.setdefault(k, []).append(r)
    out = []
    for k, group_rows in keyed.items():
        out.extend(unwrap_group(group_rows))
    return out


def cluster_iterations(all_rows):
    """Return (anchors, buckets) where anchors is the chronological list of
    TASK_SCHD_PREPARE_BATCH (begin,end) pairs and buckets[i] = dict(gap_before_ns,
    task_sum_ns, n_tasks) for the i-th inter-anchor compute span."""
    anchors = sorted(
        [r for r in all_rows if r["task_type_id"] == PREPARE_BATCH_ID],
        key=lambda r: r["begin_ts"],
    )
    compute = sorted(
        [r for r in all_rows if r["task_type_id"] not in NON_COMPUTE_IDS],
        key=lambda r: r["begin_ts"],
    )
    assert compute, "no compute-task rows found in trace"
    assert anchors, "no TASK_SCHD_PREPARE_BATCH (204) rows found -- profiler wiring changed?"

    # O(log n) lookups instead of an O(anchors x rows) nested scan: `compute`
    # is sorted by begin_ts; build a parallel view sorted by end_ts too.
    begin_sorted = compute  # already sorted by begin_ts
    begin_list = [c["begin_ts"] for c in begin_sorted]
    end_sorted = sorted(compute, key=lambda c: c["end_ts"])
    end_list = [c["end_ts"] for c in end_sorted]
    # prefix sums of duration_ns over begin_sorted, for O(log n) range-sum queries
    prefix_dur = [0]
    for c in begin_sorted:
        prefix_dur.append(prefix_dur[-1] + c["duration_ns"])

    gaps = []  # one per anchor: (last_compute_end_before, first_compute_begin_after, gap_ns)
    for a in anchors:
        i = bisect.bisect_right(end_list, a["begin_ts"])
        j = bisect.bisect_left(begin_list, a["end_ts"])
        if i == 0 or j == len(begin_list):
            continue  # anchor at the very start/end of the trace; skip
        last_end = end_list[i - 1]
        first_begin = begin_list[j]
        gaps.append({"last_compute_end_before": last_end,
                      "first_compute_begin_after": first_begin,
                      "gap_ns": first_begin - last_end})

    # Iteration buckets sandwiched between consecutive gap boundaries. Each
    # interior `hi` equals the NEXT bucket's `lo` (both are the same
    # first_compute_begin_after timestamp) -- half-open [lo, hi) on every
    # bucket except the last avoids double-crediting that boundary task to
    # both neighbors (caught by test_synthetic_trace.py: a naive closed
    # [lo, hi] on both ends double-counted and inflated task_sum ~2x).
    boundaries = [begin_list[0]] + [g["first_compute_begin_after"] for g in gaps] + [begin_sorted[-1]["end_ts"]]
    buckets = []
    n_buckets = len(boundaries) - 1
    for idx, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        lo_idx = bisect.bisect_left(begin_list, lo)
        is_last = (idx == n_buckets - 1)
        hi_idx = bisect.bisect_right(begin_list, hi) if is_last else bisect.bisect_left(begin_list, hi)
        task_sum_ns = prefix_dur[hi_idx] - prefix_dur[lo_idx]
        buckets.append({"task_sum_ns": task_sum_ns,
                         "n_tasks": hi_idx - lo_idx, "lo": lo, "hi": hi})

    return anchors, gaps, buckets


def summarize(csv_path):
    rows = load_rows(csv_path)
    rows = unwrap_all(rows)
    anchors, gaps, buckets = cluster_iterations(rows)

    n_drop_head = min(WARMUP_ITERS_DROPPED, max(0, len(buckets) - 1))
    kept_buckets = buckets[n_drop_head:-1] if len(buckets) > n_drop_head + 1 else buckets
    kept_gaps = gaps[n_drop_head:-1] if len(gaps) > n_drop_head + 1 else gaps

    task_sums_ms = [b["task_sum_ns"] / 1e6 for b in kept_buckets]
    gap_ms = [g["gap_ns"] / 1e6 for g in kept_gaps]

    return {
        "csv_path": csv_path,
        "n_anchors_total": len(anchors),
        "n_gaps_used": len(kept_gaps),
        "n_buckets_used": len(kept_buckets),
        "n_buckets_dropped_head": n_drop_head,
        "mean_task_sum_ms": statistics.fmean(task_sums_ms) if task_sums_ms else None,
        "median_task_sum_ms": statistics.median(task_sums_ms) if task_sums_ms else None,
        "mean_gap_ms": statistics.fmean(gap_ms) if gap_ms else None,
        "median_gap_ms": statistics.median(gap_ms) if gap_ms else None,
        "max_gap_ms": max(gap_ms) if gap_ms else None,
        "min_gap_ms": min(gap_ms) if gap_ms else None,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv-r4", required=True)
    p.add_argument("--csv-r8", required=True)
    p.add_argument("--step1-json-dir", required=True,
                   help="dir containing batch_perf_t*_r{1,2,4,8}.json from step 1")
    p.add_argument("--mbt", type=int, default=8)
    p.add_argument("--out-json", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    step1 = {}
    for r in (1, 2, 4, 8):
        path = os.path.join(args.step1_json_dir, f"batch_perf_t{args.mbt}_r{r}.json")
        with open(path) as f:
            step1[r] = json.load(f)

    knee_ms_per_tok = {r: step1[r]["latency_ms_per_token"] for r in (1, 2, 4, 8)}
    observed_jump_4_to_8 = knee_ms_per_tok[8] - knee_ms_per_tok[4]
    recorded_92603ca = {1: 4.40, 2: 4.41, 4: 4.44, 8: 7.49}
    knee_reproduced = (
        knee_ms_per_tok[8] > knee_ms_per_tok[4] > knee_ms_per_tok[2]
        and (knee_ms_per_tok[8] - knee_ms_per_tok[4]) > 2.0  # qualitative: a real jump, not noise
    )

    s4 = summarize(args.csv_r4)
    s8 = summarize(args.csv_r8)

    delta_gap_ms = s8["mean_gap_ms"] - s4["mean_gap_ms"]
    delta_task_ms = s8["mean_task_sum_ms"] - s4["mean_task_sum_ms"]

    # attribution: which delta actually explains the observed per-token jump?
    gap_explains = abs(delta_gap_ms - observed_jump_4_to_8) <= 0.5 * abs(observed_jump_4_to_8) or (
        delta_gap_ms > 0.5 * observed_jump_4_to_8 and abs(delta_task_ms) < 0.3 * observed_jump_4_to_8
    )
    task_explains = abs(delta_task_ms - observed_jump_4_to_8) <= 0.5 * abs(observed_jump_4_to_8) or (
        delta_task_ms > 0.5 * observed_jump_4_to_8 and abs(delta_gap_ms) < 0.3 * observed_jump_4_to_8
    )
    if gap_explains and not task_explains:
        attribution = "scheduler (serial prepare_next_batch + dispatch)"
    elif task_explains and not gap_explains:
        attribution = "task (kernel/occupancy side)"
    else:
        attribution = "ambiguous -- see delta_gap_ms vs delta_task_ms vs observed_jump_ms"

    findings = {
        "probe": "P9",
        "step1_ms_per_token": knee_ms_per_tok,
        "step1_recorded_92603ca_ms_per_token": recorded_92603ca,
        "knee_reproduced": knee_reproduced,
        "observed_jump_r4_to_r8_ms_per_token": observed_jump_4_to_8,
        "attribution": attribution,
        "evidence": {
            "r4": s4, "r8": s8,
            "delta_gap_ms_r4_to_r8": delta_gap_ms,
            "delta_task_sum_ms_r4_to_r8": delta_task_ms,
        },
    }

    with open(args.out_json, "w") as f:
        json.dump(findings, f, indent=2)
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
