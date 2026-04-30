#!/usr/bin/env python3
"""Query MPK profiler CSV traces produced alongside the Perfetto trace.

The CSV is emitted automatically when MPK runs with profiling enabled — see
python/mirage/mpk/profiler_persistent.py:export_to_csv. This script aggregates
per-task-type stats (min/max/avg/median) over all recorded events.

Usage:
    # Enumerate task types observed in the run, with event counts
    python scripts/parse_profile.py mirage_0.csv --list

    # Average runtime of one task type
    python scripts/parse_profile.py mirage_0.csv TASK_LINEAR_SM100 --stat avg

    # All stats at once
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

from mirage.mpk.profiler_persistent import event_name_list  # noqa: E402

NAME_TO_ID = {name: tid for tid, name in event_name_list.items()}


def fail(msg, code=2):
    print(json.dumps({"error": msg}))
    sys.exit(code)


def load_rows(csv_path):
    if not os.path.isfile(csv_path):
        fail(f"csv file not found: {csv_path}")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def resolve_task_type(token):
    """Return (task_type_id, task_type_name). Token may be numeric or a name."""
    if token.isdigit():
        tid = int(token)
        return tid, event_name_list.get(tid, f"UNKNOWN_{tid}")
    if token in NAME_TO_ID:
        return NAME_TO_ID[token], token
    fail(f"unknown task type: {token!r} (not in event_name_list)")


def cmd_list(rows):
    counts = Counter(r["task_type_name"] for r in rows)
    out = [{"task_type": n, "count": c}
           for n, c in counts.most_common()]
    print(json.dumps(out))


def cmd_stat(rows, task_type_token, stat):
    tid, name = resolve_task_type(task_type_token)
    durations = [int(r["duration_ns"]) for r in rows
                 if int(r["task_type_id"]) == tid]
    if not durations:
        fail(f"task type {name!r} not found in trace")

    if stat == "all":
        out = {
            "task_type": name,
            "count": len(durations),
            "min_ns": min(durations),
            "max_ns": max(durations),
            "avg_ns": statistics.fmean(durations),
            "median_ns": statistics.median(durations),
        }
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
    p.add_argument("--stat", choices=["avg", "min", "max", "all"],
                   default="avg")
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
