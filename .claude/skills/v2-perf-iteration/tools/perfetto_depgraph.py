#!/usr/bin/env python3
"""MPK true-dependency-graph analyzer.

Where ``scripts/perfetto_analyze.py`` infers a critical path from *timing
overlap* (which slice ended just before the next began), this tool reconstructs
the **real task-dependency DAG** from ``build/task_graph_rank{N}.json`` and
overlays the *measured* per-task durations from ``trace_rank{N}.csv``. That lets
us separate true data-dependencies from accidental scheduler serialization and
find concrete overlap / rebalance levers.

Four analyses, focused on one representative middle MoE layer:

  1. True critical path  — longest dependency-respecting path through the layer,
     each task weighted by its measured wallclock. Sanity-checked against the
     layer wallclock.
  2. Per-task slack       — ASAP/ALAP per task; slack==0 => on the critical path,
     large slack => overlappable.
  3. False-serialization  — task pairs whose execution windows are DISJOINT on
     the trace but have NO dependency path between them (the scheduler serialized
     them even though the data allowed overlap). Ranked by wasted gap.
  4. Per-worker idle attribution — for each gap>2us on a worker, was the next
     task's dependency already satisfied (=> scheduler/dispatch idle, a lever) or
     not (=> dependency-forced idle, not our lever)?

JOIN SEMANTICS (verified empirically, see module docstring in repo history):
  The CSV ``event_no`` column is a PER-WORKER MONOTONIC COUNTER (the i-th task
  that worker ran), NOT a DAG event_id or task_id -- it does NOT join to the DAG
  directly. Instead the join is STRUCTURAL: every DAG task is exactly one CTA,
  and the per-task_type instance counts match the CSV row counts exactly (verified
  for all 25 task types in the test trace). So the k-th DAG task of a given
  task_type maps to the k-th CSV slice of that task_type (ordered by begin_ts).
  ``TASK_SCHD_EVENTS`` (id 201) rows are the scheduler's own event bookkeeping and
  have no DAG task -- they are dropped from the join.

  An event ``e`` fires when ALL tasks with ``trigger_event == e`` have completed
  (verified: num_triggers == count of such producers for every EVENT_EMPTY). A
  task with ``dependent_event == e`` cannot start until ``e`` fires. The sentinel
  9223372036854775806 means "no dependency".

Usage:
  python .claude/skills/v2-perf-iteration/tools/perfetto_depgraph.py <trace_dir>
      [--rank N] [--layer L]
      # expects <trace_dir>/trace_rank{N}.csv + <trace_dir>/build/task_graph_rank{N}.json

Outputs (under <trace_dir>): depgraph.md, depgraph.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

SENTINEL_NO_DEP = 9223372036854775806
SCHD_EVENTS_TYPE_ID = 201  # scheduler bookkeeping rows in the CSV; no DAG task
NOOP_TYPE_IDS = (0, 10)    # type 0 = padding no-op, type 10 = BEGIN_TASK_GRAPH marker
TOPK_SIGMOID_TYPE_ID = 280
GAP_IDLE_NS = 2_000        # gaps > 2us get idle-cause attribution


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def load_csv_slices(csv_path: str):
    """Return (rows, id2name) where rows is a list of dicts with int-cast fields."""
    rows = []
    id2name: dict[int, str] = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            tt = int(r["task_type_id"])
            id2name.setdefault(tt, r["task_type_name"])
            rows.append({
                "task_type_id": tt,
                "task_type_name": r["task_type_name"],
                "block_idx": int(r["block_idx"]),
                "begin_ts": int(r["begin_ts"]),
                "end_ts": int(r["end_ts"]),
            })
    return rows, id2name


def load_dag(task_graph_path: str):
    with open(task_graph_path) as f:
        raw = json.load(f)
    return raw["all_tasks"], raw["all_events"]


# --------------------------------------------------------------------------- #
# Structural join: DAG task_id -> measured (begin_ts, end_ts, block_idx)
# --------------------------------------------------------------------------- #

def build_join(tasks: list[dict], csv_rows: list[dict]):
    """Map each DAG task_id to its single CTA's measured slice.

    Returns (ts_of, join_report):
      ts_of[task_id] = {"begin": int, "end": int, "block": int}
      join_report    = dict with per-type count check + n_joined.
    """
    # CSV slices grouped by type, sorted by begin_ts (== dispatch order per type).
    csv_by_type: dict[int, list[dict]] = defaultdict(list)
    for r in csv_rows:
        if r["task_type_id"] == SCHD_EVENTS_TYPE_ID:
            continue
        csv_by_type[r["task_type_id"]].append(r)
    for tt in csv_by_type:
        csv_by_type[tt].sort(key=lambda r: r["begin_ts"])

    # DAG task_ids grouped by type, in task_id order (== dispatch order).
    dag_by_type: dict[int, list[int]] = defaultdict(list)
    for tid, t in enumerate(tasks):
        if t["task_type"] in NOOP_TYPE_IDS:
            continue
        dag_by_type[t["task_type"]].append(tid)

    ts_of: dict[int, dict] = {}
    mismatches = []
    for tt, ids in dag_by_type.items():
        sl = csv_by_type.get(tt, [])
        if len(ids) != len(sl):
            mismatches.append({"type": tt, "n_dag": len(ids), "n_csv": len(sl)})
            n = min(len(ids), len(sl))
        else:
            n = len(ids)
        for k in range(n):
            ts_of[ids[k]] = {
                "begin": sl[k]["begin_ts"],
                "end": sl[k]["end_ts"],
                "block": sl[k]["block_idx"],
            }
    report = {
        "n_dag_tasks_typed": sum(len(v) for v in dag_by_type.values()),
        "n_csv_rows_typed": sum(len(v) for v in csv_by_type.values()),
        "n_joined": len(ts_of),
        "type_count_mismatches": mismatches,
    }
    return ts_of, report


# --------------------------------------------------------------------------- #
# Layer segmentation (by TOPK_SIGMOID task_ids in the DAG)
# --------------------------------------------------------------------------- #

def find_layers(tasks: list[dict]) -> list[tuple[int, int]]:
    """Return list of (lo_task_id, hi_task_id) half-open ranges, one per MoE layer.

    Boundaries are TOPK_SIGMOID task starts (matches perfetto_analyze's TOPK->TOPK
    convention). Each range covers [topk_i, topk_{i+1}); the trailing partial layer
    after the last TOPK is dropped (no clean end boundary)."""
    topk_ids = sorted(i for i, t in enumerate(tasks)
                      if t["task_type"] == TOPK_SIGMOID_TYPE_ID)
    # collapse to first task_id of each contiguous TOPK group (a layer dispatches
    # 128 TOPK CTAs as consecutive task_ids).
    starts: list[int] = []
    prev = None
    for tid in topk_ids:
        if prev is None or tid - prev > 1:
            starts.append(tid)
        prev = tid
    return [(starts[i], starts[i + 1]) for i in range(len(starts) - 1)]


# --------------------------------------------------------------------------- #
# Dependency DAG restricted to a task_id window
# --------------------------------------------------------------------------- #

def build_layer_dag(tasks, events, ts_of, lo, hi):
    """Build predecessor/successor maps for tasks in [lo, hi) that have a measured
    slice. Edges follow the event mechanism:
        producer task (trigger_event=e) -> event e -> consumer task (dependent_event=e)
    Only edges between in-window, joined tasks are kept (predecessors outside the
    window are treated as already-satisfied / "boundary" -- their finish time sets
    the consumer's earliest-possible start via the event-fire time computed over
    ALL producers, in-window or not).

    Returns dict with: layer_tasks (sorted list of task_id in window with a slice),
    preds, succs (task->set), event_producers (event->list of producer task_ids,
    all of them, not just in-window), event_consumers (event->list of in-window consumers)."""
    layer_tasks = [tid for tid in range(lo, hi) if tid in ts_of]
    in_win = set(layer_tasks)

    # producers of each event (global -- an event may be fed by tasks outside the
    # window, e.g. the layer's first task depends on the previous layer's output).
    event_producers: dict[int, list[int]] = defaultdict(list)
    for tid, t in enumerate(tasks):
        te = t["trigger_event"]
        if te != SENTINEL_NO_DEP:
            event_producers[te].append(tid)

    preds: dict[int, set[int]] = {tid: set() for tid in layer_tasks}
    succs: dict[int, set[int]] = {tid: set() for tid in layer_tasks}
    event_consumers: dict[int, list[int]] = defaultdict(list)
    for tid in layer_tasks:
        de = tasks[tid]["dependent_event"]
        if de == SENTINEL_NO_DEP:
            continue
        event_consumers[de].append(tid)
        for p in event_producers.get(de, ()):  # producers that feed this event
            if p in in_win and p != tid:
                preds[tid].add(p)
                succs[p].add(tid)
    return {
        "layer_tasks": layer_tasks,
        "in_win": in_win,
        "preds": preds,
        "succs": succs,
        "event_producers": event_producers,
        "event_consumers": event_consumers,
    }


# --------------------------------------------------------------------------- #
# ASAP / ALAP / critical path
# --------------------------------------------------------------------------- #

def topo_order(layer_tasks, preds):
    """Kahn topological sort restricted to the window (task_ids already roughly
    topo by construction, but be safe)."""
    indeg = {t: len(preds[t]) for t in layer_tasks}
    ready = sorted(t for t in layer_tasks if indeg[t] == 0)
    succs = defaultdict(list)
    for t in layer_tasks:
        for p in preds[t]:
            succs[p].append(t)
    order = []
    while ready:
        t = ready.pop(0)
        order.append(t)
        for s in succs[t]:
            indeg[s] -= 1
            if indeg[s] == 0:
                # insert keeping ascending task_id for determinism
                ready.append(s)
        ready.sort()
    if len(order) != len(layer_tasks):
        # cycle (shouldn't happen) -- fall back to id order
        return sorted(layer_tasks)
    return order


def compute_asap_alap(g, ts_of, tasks, t0):
    """ASAP = earliest finish respecting deps; ALAP = latest finish without
    delaying layer end. All times are relative to t0 (layer start), in ns.

    We work in 'finish-time' space using each task's measured duration. A task's
    earliest start = max over preds of (pred earliest finish); its earliest finish
    = that + its own duration. Boundary preds (outside window) are accounted for by
    clamping the in-window roots to their measured begin (they could not have
    started before their real begin, which already encodes upstream waits)."""
    layer_tasks = g["layer_tasks"]
    preds, succs = g["preds"], g["succs"]
    order = topo_order(layer_tasks, preds)

    dur = {t: ts_of[t]["end"] - ts_of[t]["begin"] for t in layer_tasks}
    meas_begin = {t: ts_of[t]["begin"] - t0 for t in layer_tasks}

    # ASAP finish.
    asap_fin: dict[int, int] = {}
    asap_start: dict[int, int] = {}
    for t in order:
        if preds[t]:
            est = max(asap_fin[p] for p in preds[t])
        else:
            # root in-window: earliest it could run is bounded by its measured
            # begin (which captures cross-layer / external dependency latency).
            est = meas_begin[t]
        # never earlier than 0
        est = max(est, 0)
        asap_start[t] = est
        asap_fin[t] = est + dur[t]

    layer_end = max(asap_fin.values()) if asap_fin else 0

    # ALAP finish (backward).
    alap_fin: dict[int, int] = {}
    alap_start: dict[int, int] = {}
    for t in reversed(order):
        if succs[t]:
            lf = min(alap_start[s] for s in succs[t])
        else:
            lf = layer_end
        alap_fin[t] = lf
        alap_start[t] = lf - dur[t]

    slack = {t: alap_start[t] - asap_start[t] for t in layer_tasks}
    return {
        "order": order,
        "dur": dur,
        "asap_start": asap_start,
        "asap_fin": asap_fin,
        "alap_start": alap_start,
        "alap_fin": alap_fin,
        "slack": slack,
        "layer_end": layer_end,
    }


def critical_path(g, sched, ts_of, tasks, id2name, t0):
    """Recover the longest dependency-respecting path (slack==0 chain) using the
    ASAP finish times. Returns list of step dicts."""
    layer_tasks = g["layer_tasks"]
    preds = g["preds"]
    asap_fin = sched["asap_fin"]
    if not layer_tasks:
        return []
    # end = task with max asap_fin
    end = max(layer_tasks, key=lambda t: asap_fin[t])
    chain = [end]
    cur = end
    while preds[cur]:
        # critical predecessor = the one whose asap_fin == cur's asap_start
        cur_start = sched["asap_start"][cur]
        crit = None
        for p in preds[cur]:
            if asap_fin[p] == cur_start:
                if crit is None or asap_fin[p] > asap_fin[crit]:
                    crit = p
        if crit is None:
            # numeric edge case: take the latest-finishing pred
            crit = max(preds[cur], key=lambda p: asap_fin[p])
        chain.append(crit)
        cur = crit
    chain.reverse()
    out = []
    for t in chain:
        out.append({
            "task_id": t,
            "task": id2name.get(tasks[t]["task_type"], f"TYPE_{tasks[t]['task_type']}"),
            "begin_us": round((ts_of[t]["begin"] - t0) / 1e3, 2),
            "end_us": round((ts_of[t]["end"] - t0) / 1e3, 2),
            "dur_us": round(sched["dur"][t] / 1e3, 2),
        })
    return out


# --------------------------------------------------------------------------- #
# Global event-fire times (used by ready-time / idle analyses)
# --------------------------------------------------------------------------- #

def compute_event_fire(tasks, ts_of):
    """event_fire[e] = max end_ts over ALL producers (trigger_event==e). An event
    fires only after every one of its producers completes (num_triggers == producer
    count, verified). Producers may live outside any one layer window, so this is
    computed globally."""
    prod_by_event: dict[int, list[int]] = defaultdict(list)
    for tid, t in enumerate(tasks):
        te = t["trigger_event"]
        if te != SENTINEL_NO_DEP:
            prod_by_event[te].append(tid)
    event_fire: dict[int, int] = {}
    for e, prods in prod_by_event.items():
        ends = [ts_of[p]["end"] for p in prods if p in ts_of]
        if ends:
            event_fire[e] = max(ends)
    return event_fire


# --------------------------------------------------------------------------- #
# Transitive reachability (for false-serialization detection)
# --------------------------------------------------------------------------- #

def reachable_sets(g):
    """For each in-window task, the set of tasks reachable from it (its transitive
    successors). Computed in reverse-topo order. O(V*E) in the worst case but the
    layer is small (~4.5k tasks)."""
    layer_tasks = g["layer_tasks"]
    succs = g["succs"]
    preds = g["preds"]
    order = topo_order(layer_tasks, preds)
    reach: dict[int, set[int]] = {}
    for t in reversed(order):
        s = set()
        for u in succs[t]:
            s.add(u)
            s |= reach[u]
        reach[t] = s
    return reach


# --------------------------------------------------------------------------- #
# Analysis 3: false serialization
# --------------------------------------------------------------------------- #

def collapse_to_instances(g, ts_of, tasks, id2name):
    """Collapse the per-CTA tasks into logical task-instances: one instance per
    (task_type, trigger_event) group within the layer (all CTAs that fan out from
    the same wave). Instance window = [min begin, max end] across its CTAs.

    Returns list of instance dicts and a map task_id -> instance_idx."""
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for t in g["layer_tasks"]:
        key = (tasks[t]["task_type"], tasks[t]["trigger_event"])
        groups[key].append(t)
    instances = []
    tid2inst = {}
    for (tt, trig), tids in groups.items():
        begins = [ts_of[t]["begin"] for t in tids]
        ends = [ts_of[t]["end"] for t in tids]
        idx = len(instances)
        instances.append({
            "idx": idx,
            "task_type": tt,
            "name": id2name.get(tt, f"TYPE_{tt}"),
            "trigger_event": trig,
            "task_ids": tids,
            "begin": min(begins),
            "end": max(ends),
            "n_ctas": len(tids),
        })
        for t in tids:
            tid2inst[t] = idx
    return instances, tid2inst


def instance_ready_time(inst, tasks, event_fire):
    """Earliest time ALL of an instance's CTAs had their dependency satisfied =
    max over its tasks' dependent_event fire-times. None if any dep never fired."""
    rt = None
    for tid in inst["task_ids"]:
        de = tasks[tid]["dependent_event"]
        if de == SENTINEL_NO_DEP:
            continue
        f = event_fire.get(de)
        if f is None:
            continue
        rt = f if rt is None else max(rt, f)
    return rt


def false_serializations(g, instances, tid2inst, reach, ts_of, tasks, event_fire,
                         worker_busy_intervals, t0, top_n=15):
    """A *false serialization* = an instance B that was dependency-READY before an
    INDEPENDENT instance A finished, yet B did not start until after A finished --
    i.e. the schedule serialized A then B even though the data allowed them to run
    concurrently. We further classify whether the overlap was *physically possible*
    (a worker was idle during B's ready-but-waiting window) vs *saturated* (all
    workers busy -> not a scheduler lever, the GPU just had no free SM).

    Metric `wasted_overlap_us` = (A.end - B.ready) clamped to >=0: how much of B's
    window could have moved earlier into A's, had the scheduler co-scheduled them.

    Independence: no transitive dependency path between A and B in either direction.
    Adjacency: B.ready <= A.end (so overlap with A specifically was the realistic
    alternative -- avoids pairing tasks that are merely far apart in the pipeline)."""
    inst_reach: dict[int, set[int]] = defaultdict(set)
    for tid, succs_set in reach.items():
        a = tid2inst[tid]
        for s in succs_set:
            inst_reach[a].add(tid2inst[s])

    # ready time per instance
    for inst in instances:
        inst["ready"] = instance_ready_time(inst, tasks, event_fire)

    # sorted worker-busy intervals per block for the "was a worker idle?" check
    def any_worker_idle(t_lo, t_hi):
        """True if at least one worker had NO task running for some sub-interval of
        [t_lo, t_hi]. Conservative: returns True if total covered worker-time over
        [t_lo,t_hi] across all workers < n_workers * span (i.e. some SM was free)."""
        span = t_hi - t_lo
        if span <= 0:
            return False, 0
        nW = len(worker_busy_intervals)
        covered = 0
        for ivs in worker_busy_intervals.values():
            for (b, e) in ivs:
                lo2 = max(b, t_lo)
                hi2 = min(e, t_hi)
                if hi2 > lo2:
                    covered += (hi2 - lo2)
        capacity = nW * span
        idle_sm_ns = capacity - covered
        return (idle_sm_ns > 0.05 * capacity), idle_sm_ns

    n = len(instances)
    pairs = []
    for ai in range(n):
        A = instances[ai]
        for bi in range(n):
            if ai == bi:
                continue
            B = instances[bi]
            if B["ready"] is None:
                continue
            # B must start strictly after A ends (serialized), windows disjoint
            if not (A["end"] <= B["begin"]):
                continue
            # B was ready at/before A finished -> overlap with A was possible
            if not (B["ready"] <= A["end"]):
                continue
            # genuine independence (no dep path either way)
            if bi in inst_reach.get(ai, ()) or ai in inst_reach.get(bi, ()):
                continue
            wasted = A["end"] - B["ready"]
            if wasted <= 0:
                continue
            # was a worker physically free during B's ready->start wait?
            idle_possible, idle_sm_ns = any_worker_idle(B["ready"], B["begin"])
            pairs.append({
                "ready_but_waited": B["name"], "later_idx": bi,
                "occupied_by": A["name"], "earlier_idx": ai,
                "wasted_overlap_us": round(wasted / 1e3, 2),
                "B_ready_us": round((B["ready"] - t0) / 1e3, 2),
                "B_begin_us": round((B["begin"] - t0) / 1e3, 2),
                "A_window_us": [round((A["begin"] - t0) / 1e3, 2),
                                round((A["end"] - t0) / 1e3, 2)],
                "B_dur_us": round((B["end"] - B["begin"]) / 1e3, 2),
                "worker_idle_during_wait": bool(idle_possible),
                "idle_sm_us_in_window": round(idle_sm_ns / 1e3, 2),
            })
    # rank by wasted overlap, prefer ones where a worker was actually idle
    pairs.sort(key=lambda p: (-(p["worker_idle_during_wait"]), -p["wasted_overlap_us"]))
    return pairs[:top_n]


# --------------------------------------------------------------------------- #
# Analysis 4: per-worker timeline + idle attribution
# --------------------------------------------------------------------------- #

def worker_idle_attribution(g, tasks, events, ts_of, id2name, event_fire, t0,
                            layer_end_ns):
    """For each worker (block_idx), order its in-window tasks by begin and find
    gaps > GAP_IDLE_NS. Attribute each gap: at the moment the worker went idle
    (prev task end), had the NEXT task's dependent_event already fired?

    event_fire (passed in) = max end_ts over ALL producers (trigger_event==e),
    computed globally (producers may be outside the layer window)."""
    # group layer tasks by worker
    by_worker: dict[int, list[int]] = defaultdict(list)
    for tid in g["layer_tasks"]:
        by_worker[ts_of[tid]["block"]].append(tid)

    workers = []
    total_dep_idle = 0
    total_sched_idle = 0
    for blk, tids in by_worker.items():
        tids.sort(key=lambda t: ts_of[t]["begin"])
        busy = sum(ts_of[t]["end"] - ts_of[t]["begin"] for t in tids)
        gaps = []
        dep_idle = 0
        sched_idle = 0
        for i in range(1, len(tids)):
            prev, nxt = tids[i - 1], tids[i]
            gap = ts_of[nxt]["begin"] - ts_of[prev]["end"]
            if gap <= GAP_IDLE_NS:
                continue
            idle_at = ts_of[prev]["end"]  # moment worker became free
            de = tasks[nxt]["dependent_event"]
            fire = event_fire.get(de, None) if de != SENTINEL_NO_DEP else None
            if fire is not None and fire > idle_at:
                cause = "dependency-forced"
                dep_idle += gap
            else:
                cause = "scheduler/dispatch"
                sched_idle += gap
            gaps.append({
                "after": id2name.get(tasks[prev]["task_type"], "?"),
                "before": id2name.get(tasks[nxt]["task_type"], "?"),
                "gap_us": round(gap / 1e3, 2),
                "cause": cause,
                "dep_event": de if de != SENTINEL_NO_DEP else None,
                "fired_after_idle_us": (round((fire - idle_at) / 1e3, 2)
                                        if (fire is not None and fire > idle_at) else None),
            })
        workers.append({
            "block": blk,
            "n_tasks": len(tids),
            "busy_us": round(busy / 1e3, 2),
            "dep_idle_us": round(dep_idle / 1e3, 2),
            "sched_idle_us": round(sched_idle / 1e3, 2),
            "n_gaps": len(gaps),
            "gaps": gaps,
        })
        total_dep_idle += dep_idle
        total_sched_idle += sched_idle

    workers.sort(key=lambda w: -w["busy_us"])
    return {
        "workers": workers,
        "total_dep_idle_us": round(total_dep_idle / 1e3, 2),
        "total_sched_idle_us": round(total_sched_idle / 1e3, 2),
        "n_workers": len(workers),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def analyze(trace_dir, rank=0, focus_layer=None):
    csv_path = os.path.join(trace_dir, f"trace_rank{rank}.csv")
    tg_path = os.path.join(trace_dir, "build", f"task_graph_rank{rank}.json")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    if not os.path.exists(tg_path):
        raise FileNotFoundError(tg_path)

    csv_rows, id2name = load_csv_slices(csv_path)
    tasks, events = load_dag(tg_path)
    ts_of, join_report = build_join(tasks, csv_rows)

    layers = find_layers(tasks)
    n_layers = len(layers)
    if focus_layer is None:
        focus_layer = n_layers // 2 if n_layers >= 3 else 0
    focus_layer = max(0, min(focus_layer, n_layers - 1))
    lo, hi = layers[focus_layer]

    g = build_layer_dag(tasks, events, ts_of, lo, hi)
    layer_tasks = g["layer_tasks"]

    # layer wallclock from measured slices
    if layer_tasks:
        t0 = min(ts_of[t]["begin"] for t in layer_tasks)
        t_end = max(ts_of[t]["end"] for t in layer_tasks)
        layer_wall_us = (t_end - t0) / 1e3
    else:
        t0 = 0
        layer_wall_us = 0.0

    event_fire = compute_event_fire(tasks, ts_of)

    sched = compute_asap_alap(g, ts_of, tasks, t0)
    cpath = critical_path(g, sched, ts_of, tasks, id2name, t0)
    cpath_total_us = round(sum(c["dur_us"] for c in cpath), 2)
    # span = wallclock occupied by the chain end-to-end (includes inter-step
    # dispatch/straggler gaps). This is the honest "how much of the layer is the
    # critical chain" number and is the join sanity check (must be <= wallclock).
    cpath_span_us = (round(cpath[-1]["end_us"] - cpath[0]["begin_us"], 2)
                     if cpath else 0.0)
    cpath_gap_us = round(cpath_span_us - cpath_total_us, 2)

    # slack table (collapse to task-type for readability, keep max-slack tasks too)
    slack = sched["slack"]
    slack_rows = []
    for t in layer_tasks:
        slack_rows.append({
            "task_id": t,
            "task": id2name.get(tasks[t]["task_type"], "?"),
            "dur_us": round(sched["dur"][t] / 1e3, 2),
            "asap_start_us": round(sched["asap_start"][t] / 1e3, 2),
            "alap_start_us": round(sched["alap_start"][t] / 1e3, 2),
            "slack_us": round(slack[t] / 1e3, 2),
        })
    # aggregate slack by task_type: min slack (criticality) + total dur movable
    by_type_slack: dict[str, list] = defaultdict(list)
    for r in slack_rows:
        by_type_slack[r["task"]].append(r)
    type_slack_summary = []
    for name, rs in by_type_slack.items():
        slacks = [r["slack_us"] for r in rs]
        type_slack_summary.append({
            "task": name,
            "n_inst": len(rs),
            "min_slack_us": round(min(slacks), 2),
            "max_slack_us": round(max(slacks), 2),
            "mean_slack_us": round(statistics.mean(slacks), 2),
            "sum_dur_us": round(sum(r["dur_us"] for r in rs), 2),
        })
    # critical types (min slack ~ 0) vs overlappable (largest min-slack)
    critical_types = sorted([r for r in type_slack_summary if r["min_slack_us"] <= 1.0],
                            key=lambda r: -r["sum_dur_us"])
    overlappable_types = sorted(type_slack_summary, key=lambda r: -r["min_slack_us"])

    # per-worker busy intervals within the layer window (for the "was a worker
    # physically idle?" classification in false-serialization).
    worker_busy_intervals: dict[int, list] = defaultdict(list)
    for t in layer_tasks:
        worker_busy_intervals[ts_of[t]["block"]].append((ts_of[t]["begin"], ts_of[t]["end"]))

    # false serialization (on collapsed instances)
    reach = reachable_sets(g)
    instances, tid2inst = collapse_to_instances(g, ts_of, tasks, id2name)
    false_ser = false_serializations(g, instances, tid2inst, reach, ts_of, tasks,
                                     event_fire, worker_busy_intervals, t0, top_n=15)

    # worker idle attribution
    idle = worker_idle_attribution(g, tasks, events, ts_of, id2name, event_fire, t0,
                                   layer_end_ns=int(layer_wall_us * 1e3))

    summary = {
        "trace_dir": trace_dir,
        "rank": rank,
        "join": join_report,
        "n_dag_tasks": len(tasks),
        "n_events": len(events),
        "n_moe_layers": n_layers,
        "focus_layer": focus_layer,
        "focus_layer_task_range": [lo, hi],
        "focus_layer_n_tasks": len(layer_tasks),
        "focus_layer_n_instances": len(instances),
        "layer_wallclock_us": round(layer_wall_us, 2),
        "critical_path_total_us": cpath_total_us,
        "critical_path_span_us": cpath_span_us,
        "critical_path_interstep_gap_us": cpath_gap_us,
        "critical_path_span_vs_wall_ratio": (round(cpath_span_us / layer_wall_us, 3)
                                             if layer_wall_us else None),
        "critical_path_vs_wall_ratio": (round(cpath_total_us / layer_wall_us, 3)
                                        if layer_wall_us else None),
        "critical_path": cpath,
        "type_slack_summary": sorted(type_slack_summary, key=lambda r: -r["sum_dur_us"]),
        "critical_types": critical_types,
        "overlappable_types_top": overlappable_types[:8],
        "false_serializations": false_ser,
        "false_ser_summary": {
            "n_pairs": len(false_ser),
            "n_pairs_worker_was_idle": sum(1 for p in false_ser
                                           if p["worker_idle_during_wait"]),
            "max_wasted_overlap_us": (max((p["wasted_overlap_us"] for p in false_ser),
                                          default=0.0)),
        },
        "idle": idle,
    }
    return summary


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #

def fmt_md(s):
    L = []
    L.append("# MPK True-Dependency-Graph Analysis")
    L.append("")
    L.append(f"- trace_dir: `{s['trace_dir']}` (rank {s['rank']})")
    L.append(f"- DAG: {s['n_dag_tasks']} tasks, {s['n_events']} events; "
             f"{s['n_moe_layers']} MoE layers")
    j = s["join"]
    L.append(f"- join: {j['n_joined']} DAG tasks matched to CSV slices "
             f"(DAG typed={j['n_dag_tasks_typed']}, CSV typed={j['n_csv_rows_typed']}, "
             f"mismatches={len(j['type_count_mismatches'])})")
    L.append(f"- **focus layer: L{s['focus_layer']}** (DAG task_ids "
             f"{s['focus_layer_task_range'][0]}..{s['focus_layer_task_range'][1]}, "
             f"{s['focus_layer_n_tasks']} CTAs / {s['focus_layer_n_instances']} logical instances)")
    L.append("")
    L.append("## Sanity check")
    L.append(f"- layer wallclock (measured): **{s['layer_wallclock_us']:.1f} us**")
    L.append(f"- critical-path SPAN (first begin -> last end, the chain end-to-end): "
             f"**{s['critical_path_span_us']:.1f} us**")
    L.append(f"- critical-path compute SUM (Σ task durations on chain): "
             f"**{s['critical_path_total_us']:.1f} us** "
             f"(+ {s['critical_path_interstep_gap_us']:.1f} us inter-step dispatch/straggler gaps)")
    sratio = s["critical_path_span_vs_wall_ratio"]
    verdict = ("OK (span <= wallclock; chain dominates the layer)"
               if sratio is not None and sratio <= 1.02
               else "SUSPECT -- join may be wrong (span should be <= wallclock)")
    L.append(f"- **critical-path span / wallclock = {sratio}** -> {verdict}")
    L.append(f"- (compute-sum / wallclock = {s['critical_path_vs_wall_ratio']}; the "
             f"gap to span is dispatch latency between dependency-linked tasks)")
    L.append("")

    L.append("## 1. True critical path")
    L.append("```")
    prev_end = None
    for c in s["critical_path"]:
        gap = "" if prev_end is None else f"  [+{c['begin_us']-prev_end:.1f}us gap]"
        L.append(f"  {c['task']:<42} @ {c['begin_us']:>8.1f} -> {c['end_us']:>8.1f}  "
                 f"({c['dur_us']:>6.2f} us){gap}")
        prev_end = c["end_us"]
    L.append("")
    L.append(f"  steps: {len(s['critical_path'])}   "
             f"span: {s['critical_path_span_us']:.1f} us   "
             f"compute-sum: {s['critical_path_total_us']:.1f} us")
    L.append("```")
    L.append("")

    L.append("## 2. Per-task slack (by task_type)")
    L.append("min_slack ~ 0 => on the critical path; large min_slack => overlappable.")
    L.append("")
    L.append("| task_type | n | min_slack_us | mean_slack_us | max_slack_us | sum_dur_us |")
    L.append("|---|---|---|---|---|---|")
    for r in s["type_slack_summary"]:
        L.append(f"| {r['task']} | {r['n_inst']} | {r['min_slack_us']} | "
                 f"{r['mean_slack_us']} | {r['max_slack_us']} | {r['sum_dur_us']} |")
    L.append("")
    L.append("### Top overlappable task_types (largest min-slack)")
    L.append("| task_type | min_slack_us | sum_dur_us |")
    L.append("|---|---|---|")
    for r in s["overlappable_types_top"]:
        L.append(f"| {r['task']} | {r['min_slack_us']} | {r['sum_dur_us']} |")
    L.append("")

    L.append("## 3. False serializations (independent work the schedule serialized)")
    L.append("B = an instance that was dependency-READY before independent instance A "
             "finished, yet only started AFTER A. `wasted_overlap_us` = A.end - B.ready "
             "(how much of B could have moved earlier). `worker_idle?` = was an SM "
             "physically free during B's ready->start wait (Y = real scheduler lever; "
             "N = all SMs busy = saturated, not a lever).")
    fss = s["false_ser_summary"]
    L.append("")
    L.append(f"- candidate pairs: {fss['n_pairs']}; with a worker physically idle "
             f"during the wait: **{fss['n_pairs_worker_was_idle']}**; "
             f"max wasted_overlap = {fss['max_wasted_overlap_us']:.2f} us")
    L.append("")
    if s["false_serializations"]:
        L.append("| ready-but-waited (B) | occupied-by (A) | wasted_overlap_us | B_dur_us | worker_idle? | A_window_us | B_ready→begin_us |")
        L.append("|---|---|---|---|---|---|---|")
        for p in s["false_serializations"][:10]:
            L.append(f"| {p['ready_but_waited']} | {p['occupied_by']} | "
                     f"**{p['wasted_overlap_us']}** | {p['B_dur_us']} | "
                     f"{'Y' if p['worker_idle_during_wait'] else 'N'} | "
                     f"{p['A_window_us']} | {p['B_ready_us']}→{p['B_begin_us']} |")
    else:
        L.append("_None found — every independent instance pair was already "
                 "co-scheduled or had a dependency edge. The layer is genuinely serial._")
    L.append("")

    L.append("## 4. Per-worker idle attribution")
    idle = s["idle"]
    L.append(f"- workers active in layer: {idle['n_workers']}")
    L.append(f"- total **dependency-forced** idle (next task's dep not yet fired "
             f"when worker freed): **{idle['total_dep_idle_us']:.1f} us** "
             f"(not our lever)")
    L.append(f"- total **scheduler/dispatch** idle (deps satisfied but task not "
             f"dispatched): **{idle['total_sched_idle_us']:.1f} us** (a lever)")
    tot = idle["total_dep_idle_us"] + idle["total_sched_idle_us"]
    if tot > 0:
        L.append(f"- split: {100*idle['total_dep_idle_us']/tot:.0f}% dependency-forced "
                 f"/ {100*idle['total_sched_idle_us']/tot:.0f}% scheduler")
    L.append("")
    L.append("### Straggler workers (longest busy)")
    L.append("| block | n_tasks | busy_us | dep_idle_us | sched_idle_us |")
    L.append("|---|---|---|---|---|")
    for w in idle["workers"][:8]:
        L.append(f"| {w['block']} | {w['n_tasks']} | {w['busy_us']} | "
                 f"{w['dep_idle_us']} | {w['sched_idle_us']} |")
    L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace_dir")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--layer", type=int, default=None,
                    help="focus MoE layer index (default: middle)")
    ap.add_argument("--no-write", action="store_true",
                    help="print only, do not write depgraph.md/.json")
    args = ap.parse_args()

    s = analyze(args.trace_dir, rank=args.rank, focus_layer=args.layer)
    md = fmt_md(s)
    print(md)
    if not args.no_write:
        md_path = os.path.join(args.trace_dir, "depgraph.md")
        json_path = os.path.join(args.trace_dir, "depgraph.json")
        with open(md_path, "w") as f:
            f.write(md)
        with open(json_path, "w") as f:
            json.dump(s, f, indent=2)
        print(f"\nWrote {md_path} and {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
