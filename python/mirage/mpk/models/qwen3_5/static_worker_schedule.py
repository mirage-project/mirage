"""Build a fixed worker schedule for the complete Qwen3.8 B8 decode graph.

The persistent task kernels and their event dependencies are unchanged.  The
schedule replaces per-token task discovery in the GPU scheduler with a
parallel copy of precomputed, topologically ordered worker task lists.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


_MASK = (1 << 64) - 1
_INVALID_EVENT = (1 << 63) - 2


def graph_signature(tasks: list[dict]) -> int:
    value = 1469598103934665603
    for task in tasks:
        for field in ("task_type", "dependent_event", "trigger_event"):
            value = ((value ^ int(task[field])) * 1099511628211) & _MASK
    return value


def costs_from_profiler(csv_path: Path) -> dict[int, float]:
    """Return median microseconds per task type from MPK's paired GPU stamps."""
    groups: dict[int, list[float]] = {}
    with csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            task_type = int(row["task_type_id"])
            if task_type in (0, 10) or task_type >= 200 and task_type < 205:
                continue
            duration = int(row["duration_ns"]) / 1000
            if duration > 0:
                groups.setdefault(task_type, []).append(duration)
    if not groups:
        raise ValueError("profiler CSV contains no timed model tasks")
    return {kind: statistics.median(values) for kind, values in groups.items()}


def _validate_graph(graph: dict, num_workers: int) -> tuple[list[dict], list[dict]]:
    tasks, events = graph["all_tasks"], graph["all_events"]
    if num_workers < 1 or len(tasks) < 3:
        raise ValueError("invalid graph or worker count")
    launch = events[1]
    if (launch["event_type"], launch["first_task_id"], launch["last_task_id"]) != (
            903, 2, len(tasks)):
        raise ValueError("static schedule requires one full-graph dependent-task launch")
    if len(graph["first_tasks"]) != 8 or sorted(graph["first_tasks"]) != list(range(2, 10)):
        raise ValueError("static schedule supports the complete fixed B8 graph only")
    producers: dict[int, list[int]] = {}
    for task_id, task in enumerate(tasks):
        event = int(task["trigger_event"])
        if event < len(events):
            producers.setdefault(event, []).append(task_id)
    for task_id, task in enumerate(tasks[2:], 2):
        event = int(task["dependent_event"])
        if event < len(events) and any(p >= task_id for p in producers.get(event, ())):
            raise ValueError(f"task {task_id} has a non-topological dependency")
    for event, ids in producers.items():
        if len(ids) != events[event]["num_triggers"]:
            raise ValueError(f"event {event} has an unexpected trigger count")
    return tasks, events


def build_schedule(graph: dict, num_workers: int, costs: dict[int, float] | None = None,
                   *, strategy: str = "balanced", num_schedulers: int | None = None
                   ) -> list[list[int]]:
    tasks, events = _validate_graph(graph, num_workers)
    if strategy not in ("round_robin", "runtime_layout", "balanced"):
        raise ValueError("strategy must be round_robin, runtime_layout or balanced")
    if strategy == "runtime_layout" and (num_schedulers is None or
                                         not 1 <= num_schedulers <= num_workers):
        raise ValueError("runtime_layout requires the original local scheduler count")
    if costs is None:
        costs = {}
    unknown = {int(t["task_type"]) for t in tasks[2:]} - costs.keys()
    if strategy == "balanced" and unknown:
        raise ValueError(f"missing measured task costs for types {sorted(unknown)}")
    lists: list[list[int]] = [[] for _ in range(num_workers)]
    ready = [0.0] * num_workers
    event_time = [0.0] * len(events)
    for task_id in range(2, len(tasks)):
        task = tasks[task_id]
        dep = int(task["dependent_event"])
        dependency_time = event_time[dep] if dep < len(events) else 0.0
        cost = costs.get(int(task["task_type"]), 1.0)
        if strategy == "round_robin":
            worker = (task_id - 2) % num_workers
        elif strategy == "runtime_layout":
            slot = (task_id - 2) % num_workers
            # Scheduler 0 dispatched the begin task to its first worker, so
            # its next_worker cursor is advanced by one for the full graph.
            first_group = (num_workers + num_schedulers - 1) // num_schedulers
            worker = (slot + 1) % first_group if slot < first_group else slot
        else:
            worker = min(range(num_workers), key=lambda w: (max(ready[w], dependency_time),
                                                             ready[w], w))
        finish = max(ready[worker], dependency_time) + cost
        ready[worker] = finish
        event = int(task["trigger_event"])
        if event < len(events):
            event_time[event] = max(event_time[event], finish)
        lists[worker].append(task_id)
    if max(map(len, lists)) + 1 > 8192:
        raise ValueError("a worker queue exceeds MPK's 8192-entry capacity")
    return lists


def write_schedule(path: Path, graph: dict, lists: list[list[int]]) -> None:
    tasks = graph["all_tasks"]
    if sorted(task_id for ids in lists for task_id in ids) != list(range(2, len(tasks))):
        raise ValueError("schedule must cover every model task exactly once")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(f"MPK_STATIC_WORKERS_V1 {len(lists)} {len(tasks)} "
                     f"{graph_signature(tasks)}\n")
        for ids in lists:
            handle.write(str(len(ids)))
            handle.write(" ")
            handle.write(" ".join(map(str, ids)))
            handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-csv", type=Path)
    parser.add_argument("--strategy", choices=("round_robin", "runtime_layout", "balanced"),
                        default="runtime_layout")
    parser.add_argument("--schedulers", type=int, default=80,
                        help="Original local scheduler count for runtime_layout")
    args = parser.parse_args()
    graph = json.loads(args.graph.read_text())
    costs = costs_from_profiler(args.profile_csv) if args.profile_csv else None
    lists = build_schedule(graph, args.workers, costs, strategy=args.strategy,
                           num_schedulers=args.schedulers)
    write_schedule(args.output, graph, lists)
    print(json.dumps({"output": str(args.output), "tasks": len(graph["all_tasks"]) - 2,
                      "workers": args.workers, "per_worker_min": min(map(len, lists)),
                      "per_worker_max": max(map(len, lists)), "costs_us": costs}, sort_keys=True))


if __name__ == "__main__":
    main()
