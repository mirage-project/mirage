#!/usr/bin/env python3
"""Query MPK profiler CSV traces produced alongside the Perfetto trace.

The CSV is emitted automatically when MPK runs with profiling enabled — see
python/mirage/mpk/profiler_persistent.py:export_to_csv. This script aggregates
per-task-type stats (min/max/avg/median/wall) over all recorded events.

KERNEL-LATENCY METRIC: use `--stat wall`, NOT median/avg. The per-task
duration_ns is the per-CTA span, and at decode many task types are BIMODAL:
e.g. the mediumm dense-GEMM launches grid_dim=128 CTAs of which ~99 idle-exit
in <1us (active_rows < num_CTAs) and only ~17 do real work (~29us). The MEDIAN
duration_ns is therefore an *idle CTA* (~0.66us) and grossly understates kernel
latency. The faithful single-kernel latency is the WALL-SPAN of the task's
events: max(end_ts) - min(begin_ts), i.e. wallclock from the first CTA starting
to the last CTA finishing. Use `--stat wall` for any WIN/SLOWER perf decision;
median/avg only characterize the per-CTA work split.

Usage:
    # Enumerate task types observed in the run, with event counts
    python scripts/parse_profile.py mirage_0.csv --list

    # Wall-span (kernel latency) of one task type — the correct perf metric
    python scripts/parse_profile.py mirage_0.csv TASK_LINEAR_SM100 --stat wall

    # Average runtime of one task type
    python scripts/parse_profile.py mirage_0.csv TASK_LINEAR_SM100 --stat avg

    # All stats at once (now includes wall_ns)
    python scripts/parse_profile.py mirage_0.csv TASK_LINEAR_SM100 --stat all

    # Numeric task-type id also accepted
    python scripts/parse_profile.py mirage_0.csv 253 --stat min

Output is JSON. Errors print {"error": "..."} on stdout and exit 2.
"""

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

# `event_name_list` lives in mirage.mpk.profiler_persistent, but importing it
# pulls in mirage/__init__.py which does `import z3` — a heavy dep that is not
# installed in every env (e.g. a CPU box without the superoptimizer extras).
# Only the *name* resolution path (resolving a TASK_* string -> numeric id, or
# the all-stats `task_type` label) actually needs it. `--list` reads the
# task_type_name column straight from the CSV, and `--stat` with a NUMERIC id
# needs nothing. So we import it lazily and tolerate its absence: the name-only
# paths get a clean error instead of an import traceback.
_EVENT_NAMES_CACHE = None


def _event_name_list():
    """Lazily import event_name_list; return {} if its deps (z3) are missing."""
    global _EVENT_NAMES_CACHE
    if _EVENT_NAMES_CACHE is None:
        try:
            from mirage.mpk.profiler_persistent import event_name_list
            _EVENT_NAMES_CACHE = dict(event_name_list)
        except Exception:  # noqa: BLE001 (e.g. ModuleNotFoundError: z3)
            _EVENT_NAMES_CACHE = {}
    return _EVENT_NAMES_CACHE


def fail(msg, code=2):
    print(json.dumps({"error": msg}))
    sys.exit(code)


def load_rows(csv_path):
    if not os.path.isfile(csv_path):
        fail(f"csv file not found: {csv_path}")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _csv_name_to_id(rows):
    """Build a name->id map from the CSV's own columns (no mirage import)."""
    m = {}
    for r in rows:
        nm = r.get("task_type_name")
        tid = r.get("task_type_id")
        if nm and tid is not None:
            m[nm] = int(tid)
    return m


def resolve_task_type(token, rows=None):
    """Return (task_type_id, task_type_name). Token may be numeric or a name.

    Name resolution prefers mirage's event_name_list, but falls back to the
    CSV's own (task_type_name, task_type_id) columns so the tool still works
    when mirage's import deps (z3) are unavailable.
    """
    if token.isdigit():
        tid = int(token)
        return tid, _event_name_list().get(tid, f"UNKNOWN_{tid}")
    name_to_id = {n: t for t, n in _event_name_list().items()}
    if token in name_to_id:
        return name_to_id[token], token
    if rows is not None:
        csv_map = _csv_name_to_id(rows)
        if token in csv_map:
            return csv_map[token], token
    fail(f"unknown task type: {token!r} (not in event_name_list or CSV)")


def cmd_list(rows):
    counts = Counter(r["task_type_name"] for r in rows)
    out = [{"task_type": n, "count": c}
           for n, c in counts.most_common()]
    print(json.dumps(out))


_U32 = 1 << 32   # 2^32, the %globaltimer 32-bit wrap modulus
_HALF = 1 << 31  # 2^31, the wrap-detection threshold


def wall_span_ns(begins, ends):
    """WALL-SPAN = max(end_ts) - min(begin_ts) over a task's events, with
    32-bit wrap correction.

    begin_ts/end_ts in the CSV are raw 32-bit %globaltimer reads (they wrap
    every ~4.3s = 2^32 ns). If the task's events straddle a wrap, a raw
    end_ts can be numerically SMALLER than the run's min begin_ts even though
    it occurred LATER in real time (it rolled past 2^32 back near zero). We
    anchor on the task's global min begin_ts and detect a straddle by the
    NAIVE span exceeding 2^31: a real single-task span is microseconds, so a
    naive max(end)-min(begin) of ~gigaseconds can only be a wrap artifact.
    When that fires, any raw end below the anchor is a post-wrap value and
    gets one +2^32 period added before taking the max; without a straddle the
    raw max(end) - min(begin) IS the span.

    NOTE on scope: the canonical traces have NO wrap (both spans ~20-30us, no
    end < min_begin), so this returns the exact raw span there. The correction
    is a guard for runs whose events land across a 2^32 boundary by a wide
    (>2^31) margin. A microsecond-scale task whose wrap falls *mid-task* (start
    raw near 2^32, finish raw near 0) is fundamentally ambiguous from raw
    32-bit values alone — there is no unwrapped reference to anchor on — so it
    is out of scope here; profiles that need it should widen the timestamp to
    64-bit upstream rather than guess.
    """
    min_begin = min(begins)
    naive_span = max(ends) - min_begin
    if naive_span <= _HALF:
        # Common case (incl. every canonical trace): no wrap straddle. Raw
        # arithmetic is the true span.
        return naive_span
    # Wide straddle: lift every below-anchor (post-wrap) end by one 2^32 period
    # so the true latest finish wins the max. Above-anchor ends are pre-wrap and
    # already correctly ordered relative to the anchor.
    corrected_ends = [(e + _U32) if e < min_begin else e for e in ends]
    return max(corrected_ends) - min_begin


def cmd_stat(rows, task_type_token, stat):
    tid, name = resolve_task_type(task_type_token, rows)
    matched = [r for r in rows if int(r["task_type_id"]) == tid]
    durations = [int(r["duration_ns"]) for r in matched]
    if not durations:
        fail(f"task type {name!r} not found in trace")
    begins = [int(r["begin_ts"]) for r in matched]
    ends = [int(r["end_ts"]) for r in matched]

    if stat == "all":
        wall = wall_span_ns(begins, ends)
        out = {
            "task_type": name,
            "count": len(durations),
            "min_ns": min(durations),
            "max_ns": max(durations),
            "avg_ns": statistics.fmean(durations),
            "median_ns": statistics.median(durations),
            "wall_ns": wall,
            "wall_us": wall / 1000.0,
        }
    elif stat == "wall":
        wall = wall_span_ns(begins, ends)
        out = {"task_type": name, "count": len(durations),
               "wall_ns": wall, "wall_us": wall / 1000.0}
    elif stat == "avg":
        out = {"task_type": name, "stat": "avg",
               "value_ns": statistics.fmean(durations),
               "count": len(durations)}
    elif stat == "min":
        out = {"task_type": name, "stat": "min",
               "value_ns": min(durations),
               "count": len(durations)}
    elif stat == "max":
        out = {"task_type": name, "stat": "max",
               "value_ns": max(durations),
               "count": len(durations)}
    else:
        fail(f"unknown stat: {stat!r}")

    print(json.dumps(out))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv_path")
    p.add_argument("task_type", nargs="?",
                   help="Task type name (e.g. TASK_LINEAR_SM100) or numeric id")
    p.add_argument("--stat", choices=["avg", "min", "max", "all", "wall"],
                   default="avg",
                   help="wall = max(end_ts)-min(begin_ts) (kernel latency; "
                        "use this for perf decisions, NOT median/avg)")
    p.add_argument("--list", action="store_true", dest="list_mode",
                   help="List all task types observed, with event counts")
    args = p.parse_args()

    rows = load_rows(args.csv_path)

    if args.list_mode:
        if args.task_type is not None:
            fail("--list is mutually exclusive with positional task_type")
        cmd_list(rows)
        return

    if args.task_type is None:
        fail("task_type is required unless --list is given")

    cmd_stat(rows, args.task_type, args.stat)


if __name__ == "__main__":
    main()
