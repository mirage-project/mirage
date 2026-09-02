"""Predict a partition's performance instead of building it.

Compile once, measure what each task SHAPE cost (CostTable), then price any
candidate partition by replaying the runtime's scheduling over its dataflow
graph: one task per grid point, a fired event's tasks dealt round-robin across
workers, each worker FIFO with no stealing, plus a per-task switch cost. Rank
by predicted makespan and spend real builds only on the top few.

It is a RANKING device: ~5% low in absolute terms, and a candidate containing
a shape this build never compiled is an estimate, not a measurement -- which
is what `Candidate.unpriced` reports. See docs/superoptimizer_ir.md.
"""
from __future__ import annotations

import csv
import dataclasses
import math
import re
import statistics
from collections import defaultdict
from typing import Callable, Iterable, Optional

from .group import Group
from .node import ModelGraph, is_opaque

@dataclasses.dataclass
class Trace:
    """What the simulation predicts."""
    makespan: float
    per_group: list[tuple[str, float, int, float]]  # tag, start, tasks, end
    busy: float                                     # summed worker-seconds
    num_workers: int

    @property
    def utilization(self) -> float:
        return self.busy / (self.makespan * self.num_workers)

    def report(self, top: int = 10) -> str:
        rows = sorted(self.per_group, key=lambda r: r[3] - r[1], reverse=True)
        w = [f"makespan {self.makespan * 1e3:.3f} ms   "
             f"utilization {self.utilization * 100:.1f}%   "
             f"{self.num_workers} workers"]
        w.append(f"{'group':16} {'tasks':>6} {'span_us':>9} {'start_us':>9}")
        for tag, st, n, en in rows[:top]:
            w.append(f"{tag:16} {n:6d} {(en - st) * 1e6:9.1f} {st * 1e6:9.1f}")
        return "\n".join(w)


def producers(graph: ModelGraph,
              partition: list[Group]) -> list[list[int]]:
    """For each group, the indices of the groups producing its inputs."""
    owner = {}
    for gi, grp in enumerate(partition):
        for n in grp.nodes:
            owner[graph.nodes[n].output.name] = gi
        for v in [grp.output, *grp.extra_outputs]:
            owner[v.name] = gi

    preds = []
    for gi, grp in enumerate(partition):
        p = set()
        for v in grp.external_inputs:
            j = owner.get(v.name)
            if j is not None and j != gi:
                p.add(j)
        preds.append(sorted(p))
    return preds


# Measured floor of the inter-task gap: a worker whose next task is already
# queued still takes this long to start it. Charged once per task.
SWITCH_COST = 1.344e-6


def simulate(graph: ModelGraph,
             partition: list[Group],
             duration: Callable[[Group], float],
             *,
             num_workers: int,
             grid_for: Optional[Callable[[Group], tuple]] = None,
             tasks_for: Optional[Callable[[Group], int]] = None,
             switch_cost: float = SWITCH_COST) -> Trace:
    """Makespan of `partition` on `num_workers` persistent workers.

    `duration` gives one task's runtime in seconds; `switch_cost` is added to
    each. Task count comes from `tasks_for`, else the product of `grid_for`.
    """
    if tasks_for is None:
        if grid_for is None:
            raise ValueError("pass grid_for or tasks_for")
        tasks_for = lambda g: math.prod(grid_for(g))

    preds = producers(graph, partition)
    order = _toposort(preds)

    free = [0.0] * num_workers      # when each worker's queue drains
    next_worker = 0
    done = [0.0] * len(partition)
    per_group, busy = [], 0.0

    for gi in order:
        grp = partition[gi]
        release = max((done[p] for p in preds[gi]), default=0.0)
        d, n = duration(grp) + switch_cost, tasks_for(grp)

        end = release
        for _ in range(n):
            w = next_worker
            next_worker = (next_worker + 1) % num_workers
            free[w] = max(free[w], release) + d
            busy += d
            end = max(end, free[w])
        done[gi] = end
        per_group.append((grp.tag or grp.output.name, release, n, end))

    return Trace(max(done, default=0.0), per_group, busy, num_workers)


def _toposort(preds: list[list[int]]) -> list[int]:
    seen, out = set(), []

    def visit(i):
        if i in seen:
            return
        seen.add(i)
        for p in preds[i]:
            visit(p)
        out.append(i)

    for i in range(len(preds)):
        visit(i)
    return out


_LINE = re.compile(r"\[shape-index\] (\d+) in=(\S*) out=(\S*) ops=(\S*)")
# Older builds printed the dims run together with the tb op ids:
#   [shape-index] 0 i8x1024xi1x1024xi8x1024xo8x1024x2001,2001,...
_LEGACY = re.compile(r"\[shape-index\] (\d+) ([io][0-9x]+.*)")


def _dims(text: str) -> tuple:
    return tuple(tuple(int(d) for d in part.split("x"))
                 for part in text.split(",") if part)


@dataclasses.dataclass
class CostTable:
    """duration(group) -> seconds per task, keyed by shape."""
    by_shape: dict          # (inputs, outputs) -> seconds
    by_task: dict           # hand-written task name -> seconds
    floor: float
    misses: int = 0

    @classmethod
    def from_profile(cls, csv_path: str, index_log: str) -> "CostTable":
        shape_of = {}
        with open(index_log, errors="ignore") as f:
            for line in f:
                m = _LINE.search(line)
                if m:
                    shape_of[int(m.group(1))] = (_dims(m.group(2)),
                                                 _dims(m.group(3)))
                    continue
                m = _LEGACY.search(line)
                if m:
                    # cut the op-id tail, then the id glued to the last dim
                    sig = m.group(2).split(",")[0].rsplit("x", 1)[0] + "x"
                    ins = tuple(tuple(int(d) for d in p.strip("x").split("x")
                                      if d)
                                for p in re.findall(r"i([0-9x]+)", sig))
                    outs = tuple(tuple(int(d) for d in p.strip("x").split("x")
                                       if d)
                                 for p in re.findall(r"o([0-9x]+)", sig))
                    shape_of[int(m.group(1))] = (ins, outs)
        gen, other = defaultdict(list), defaultdict(list)
        with open(csv_path) as f:
            for r in csv.DictReader(f):
                name, ns = r["task_type_name"], int(r["duration_ns"])
                if name.startswith("TASK_GENERATED"):
                    i = (0 if name == "TASK_GENERATED"
                         else int(name.split("shape")[1].rstrip("]")))
                    gen[i].append(ns)
                else:
                    other[name].append(ns)
        by_shape = {shape_of[i]: statistics.median(v) * 1e-9
                    for i, v in gen.items() if i in shape_of}
        by_task = {k: statistics.median(v) * 1e-9 for k, v in other.items()}
        if not by_shape:
            raise ValueError(
                f"no generated-task shapes in {csv_path}; was the run profiled "
                f"with MPK_DUMP_SHAPE_INDEX=1 so {index_log} has the ids?")
        return cls(by_shape, by_task, min(by_shape.values()))

    # The customized op reads its output buffer as a trailing input, so the
    # key a group presents must match: external inputs, then the output.
    def _key(self, group: Group) -> tuple:
        ins = tuple(v.dims for v in group.external_inputs)
        out = group.output.dims
        return (ins + (out,), (out,))

    def _node_key(self, graph: ModelGraph, i: int) -> tuple:
        n = graph.nodes[i]
        ins = tuple(v.dims for v in n.inputs)
        return (ins + (n.output.dims,), (n.output.dims,))

    def duration(self, graph: ModelGraph, group: Group,
                 opaque_task: Optional[dict] = None) -> float:
        op = graph.nodes[group.nodes[0]].op
        if is_opaque(op):
            name = (opaque_task or {}).get(op.split(":", 1)[1])
            return self.by_task.get(name, self.floor)
        hit = self.by_shape.get(self._key(group))
        if hit is not None:
            return hit
        # This build never compiled this group, so its cost is an estimate.
        # Use the SUM of its nodes measured on their own -- an upper bound,
        # since fusing saves at least the loads between them. Falling back to
        # a floor instead would price every unseen fusion at the cheapest
        # thing ever measured, which biases the ranking toward fusion exactly
        # where the evidence is missing.
        self.misses += 1
        return sum(self.by_shape.get(self._node_key(graph, i), self.floor)
                   for i in group.nodes)


def task_counter(grid_for: Callable, opaque_tasks: Optional[dict] = None):
    """tasks_for(group): one task per grid point (runtime.cc:1442)."""
    opaque_tasks = opaque_tasks or {}

    def tasks_for(group: Group) -> int:
        n = opaque_tasks.get(group.tag)
        if n is not None:
            return n
        try:
            return max(1, math.prod(grid_for(group)))
        except Exception:
            return 1

    return tasks_for


@dataclasses.dataclass
class Candidate:
    partition: list
    makespan: float
    tasks: int
    utilization: float
    unpriced: int          # groups whose shape this build never measured

    @property
    def tags(self) -> str:
        return " | ".join(g.tag or str(g.nodes) for g in self.partition)


def rank(graph: ModelGraph,
         candidates: Iterable[list],
         cost: CostTable,
         *,
         num_workers: int,
         grid_for: Callable,
         opaque_task: Optional[dict] = None,
         opaque_tasks: Optional[dict] = None,
         switch_cost: float = SWITCH_COST,
         limit: Optional[int] = None) -> list:
    """Every candidate, cheapest predicted makespan first."""
    tasks_for = task_counter(grid_for, opaque_tasks)
    out = []
    for part in candidates:
        before = cost.misses
        tr = simulate(graph, part,
                      lambda g: cost.duration(graph, g, opaque_task),
                      num_workers=num_workers, tasks_for=tasks_for,
                      switch_cost=switch_cost)
        out.append(Candidate(part, tr.makespan,
                             sum(tasks_for(g) for g in part),
                             tr.utilization, cost.misses - before))
        if limit and len(out) >= limit:
            break
    return sorted(out, key=lambda c: c.makespan)


def report(ranked: list, top: int = 10) -> str:
    w = [f"{len(ranked)} candidates ranked",
         f"{'#':>3} {'makespan_us':>12} {'tasks':>7} {'util':>6} {'unpriced':>9}"]
    for i, c in enumerate(ranked[:top]):
        w.append(f"{i:3d} {c.makespan * 1e6:12.1f} {c.tasks:7d} "
                 f"{c.utilization * 100:5.1f}% {c.unpriced:9d}")
    if len(ranked) > top:
        c = ranked[-1]
        w.append(f"{'...':>3} {c.makespan * 1e6:12.1f} {c.tasks:7d} "
                 f"{c.utilization * 100:5.1f}% {c.unpriced:9d}   (worst)")
        w.append(f"spread best->worst: "
                 f"{ranked[-1].makespan / ranked[0].makespan:.2f}x")
    return "\n".join(w)
