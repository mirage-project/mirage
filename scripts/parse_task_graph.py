#!/usr/bin/env python3
"""Parse and inspect MPK task graph JSON files.

Usage:
    # Print summary (event/task counts, task type breakdown)
    python scripts/parse_task_graph.py task_graph_rank0.json

    # Inspect a specific task by index
    python scripts/parse_task_graph.py task_graph_rank0.json --task 2

    # Inspect a specific event by index
    python scripts/parse_task_graph.py task_graph_rank0.json --event 1

    # Print all tasks
    python scripts/parse_task_graph.py task_graph_rank0.json --task all

    # Print all events
    python scripts/parse_task_graph.py task_graph_rank0.json --event all
"""

import argparse
import json
import os
import sys
from collections import Counter


def parse_enum(prefix):
    """Parse TaskType or EventType enum from runtime_header.h."""
    mirage_home = os.environ.get("MIRAGE_HOME", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    header = os.path.join(mirage_home, "include/mirage/persistent_kernel/runtime_header.h")
    mapping = {}
    inside_enum = False
    with open(header) as f:
        for line in f:
            stripped = line.strip()
            if f"enum {prefix}" in stripped:
                inside_enum = True
                continue
            if inside_enum:
                if "}" in stripped:
                    break
                if "=" in stripped and stripped and not stripped.startswith("//"):
                    # e.g. "TASK_TERMINATE = 0,"  or  "TASK_TERMINATE = 0, // comment"
                    parts = stripped.split("//")[0].strip().rstrip(",").split("=")
                    name = parts[0].strip()
                    value = int(parts[1].strip())
                    mapping[value] = name
    return mapping


# Data type enum from include/mirage/type.h
DATA_TYPE_NAMES = {
    920: "float4",
    925: "int4",
    926: "uint4",
    930: "float8",
    935: "int8",
    936: "uint8",
    940: "float16",
    941: "bfloat16",
    945: "int16",
    946: "uint16",
    950: "float32",
    955: "int32",
    956: "uint32",
    960: "double",
    965: "int64",
    966: "uint64",
}

DATA_TYPE_SIZES = {
    920: 1,   # float4 (packed, treat as 1 byte)
    925: 1,   # int4
    926: 1,   # uint4
    930: 1,   # float8
    935: 1,   # int8
    936: 1,   # uint8
    940: 2,   # float16
    941: 2,   # bfloat16
    945: 2,   # int16
    946: 2,   # uint16
    950: 4,   # float32
    955: 4,   # int32
    956: 4,   # uint32
    960: 8,   # double
    965: 8,   # int64
    966: 8,   # uint64
}


def format_tensor(t):
    """Format a tensor descriptor as a compact string."""
    dtype = DATA_TYPE_NAMES.get(t["data_type"], f"dtype={t['data_type']}")
    dims = "x".join(str(d) for d in t["dims"])
    elem_size = DATA_TYPE_SIZES.get(t["data_type"], 1)
    offset_elems = t["offset"] // elem_size if elem_size else t["offset"]
    offset_str = f"+{offset_elems}" if offset_elems else ""
    return f"{t['base_ptr']}{offset_str} [{dims}] {dtype}"


def print_summary(graph, task_names, event_names):
    """Print summary statistics."""
    events = graph["all_events"]
    tasks = graph["all_tasks"]
    first_tasks = graph.get("first_tasks", [])

    print("=" * 60)
    print("Task Graph Summary")
    print("=" * 60)
    print(f"  Events:      {len(events)}")
    print(f"  Tasks:       {len(tasks)}")
    print(f"  First tasks: {len(first_tasks)}")
    print()

    # Event type breakdown
    event_counts = Counter(e["event_type"] for e in events)
    print("Event Types:")
    print(f"  {'Type':<40s} {'Count':>5s}")
    print(f"  {'-'*40} {'-'*5}")
    for etype, count in sorted(event_counts.items()):
        name = event_names.get(etype, f"UNKNOWN({etype})")
        print(f"  {name:<40s} {count:>5d}")
    print()

    # Task type breakdown
    task_counts = Counter(t["task_type"] for t in tasks)
    print("Task Types:")
    print(f"  {'Type':<45s} {'Count':>5s}")
    print(f"  {'-'*45} {'-'*5}")
    for ttype, count in sorted(task_counts.items()):
        name = task_names.get(ttype, f"UNKNOWN({ttype})")
        print(f"  {name:<45s} {count:>5d}")
    print()


def print_event(events, idx, event_names):
    """Print details of a single event."""
    if idx < 0 or idx >= len(events):
        print(f"Error: event index {idx} out of range [0, {len(events)-1}]", file=sys.stderr)
        sys.exit(1)
    e = events[idx]
    name = event_names.get(e["event_type"], f"UNKNOWN({e['event_type']})")
    print(f"Event {idx}:")
    print(f"  event_type:    {name} ({e['event_type']})")
    print(f"  num_triggers:  {e['num_triggers']}")
    print(f"  first_task_id: {e['first_task_id']}")
    print(f"  last_task_id:  {e['last_task_id']}")


def print_task(tasks, idx, task_names):
    """Print details of a single task."""
    if idx < 0 or idx >= len(tasks):
        print(f"Error: task index {idx} out of range [0, {len(tasks)-1}]", file=sys.stderr)
        sys.exit(1)
    t = tasks[idx]
    name = task_names.get(t["task_type"], f"UNKNOWN({t['task_type']})")
    print(f"Task {idx}:")
    print(f"  task_type:        {name} ({t['task_type']})")
    print(f"  variant_id:       {t['variant_id']}")
    print(f"  trigger_event:    {t['trigger_event']}")
    print(f"  dependent_event:  {t['dependent_event']}")
    print(f"  request_id:       {t['request_id']}")
    print(f"  task_offset:      {t['task_offset']}")
    print(f"  expert_offset:    {t['expert_offset']}")
    print(f"  kv_idx:           {t['kv_idx']}")
    print(f"  merge_task_offset:{t['merge_task_offset']}")

    if t["inputs"] is not None:
        print(f"  inputs ({len(t['inputs'])}):")
        for i, inp in enumerate(t["inputs"]):
            print(f"    [{i}] {format_tensor(inp)}")
    else:
        print(f"  inputs: None")

    if t["outputs"] is not None:
        print(f"  outputs ({len(t['outputs'])}):")
        for i, out in enumerate(t["outputs"]):
            print(f"    [{i}] {format_tensor(out)}")
    else:
        print(f"  outputs: None")


def main():
    parser = argparse.ArgumentParser(description="Parse and inspect MPK task graph JSON files.")
    parser.add_argument("json_file", help="Path to the task graph JSON file")
    parser.add_argument("--task", "-t", default=None,
                        help="Print details of a specific task by index, or 'all' for all tasks")
    parser.add_argument("--event", "-e", default=None,
                        help="Print details of a specific event by index, or 'all' for all events")
    args = parser.parse_args()

    with open(args.json_file) as f:
        graph = json.load(f)

    task_names = parse_enum("TaskType")
    event_names = parse_enum("EventType")

    if args.task is None and args.event is None:
        print_summary(graph, task_names, event_names)
        return

    if args.event is not None:
        events = graph["all_events"]
        if args.event == "all":
            for i in range(len(events)):
                print_event(events, i, event_names)
                print()
        else:
            print_event(events, int(args.event), event_names)

    if args.task is not None:
        tasks = graph["all_tasks"]
        if args.task == "all":
            for i in range(len(tasks)):
                print_task(tasks, i, task_names)
                print()
        else:
            print_task(tasks, int(args.task), task_names)


if __name__ == "__main__":
    main()
