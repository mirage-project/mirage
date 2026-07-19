#!/usr/bin/env python3
"""MPK perfetto trace analysis pipeline — drives:

  1. Parse perfetto protobuf + matching task_graph_rank0.json (build dir)
  2. Cross-reference perfetto BEGIN/END events ↔ task graph TaskDesc entries
  3. Per-task wallclock breakdown
  4. Per-layer breakdown (MoE layers segmented by TOPK_SIGMOID events)
  5. Per-trigger-event fan-out analysis (catches "many CTAs launched in
     one wave" — user-flagged FP8 over-dispatch)
  6. Critical-path chain estimator (greedy backward walk from layer-end)
  7. Anomaly detection (intra-task variance, producer/consumer mismatches,
     unmapped task types)
  8. Output: markdown report + JSON summary + ASCII timeline

Usage:
  python scripts/perfetto_analyze.py <trace_dir>
    # expects trace_dir/trace_rank0.perfetto-trace + trace_dir/build/task_graph_rank0.json

  python scripts/perfetto_analyze.py <trace_dir> --rank N
    # analyze rank N instead of 0

  python scripts/perfetto_analyze.py <trace_dir> --layer L
    # focus per-layer Gantt on layer index L (default: middle MoE layer)

Outputs (under <trace_dir>):
  analysis.md   — human + agent-readable markdown report
  analysis.json — structured data for downstream agent ingestion
"""
from __future__ import annotations
import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, field
from typing import Optional

try:
    from tg4perfetto import perfetto_trace_pb2 as perfetto_pb2
except ImportError:
    raise SystemExit(
        "tg4perfetto not installed. Run inside the project venv:\n"
        "  python .claude/skills/v2-perf-iteration/tools/perfetto_analyze.py ..."
    )


TYPE_SLICE_BEGIN = 1
TYPE_SLICE_END = 2


# -- type-enum loader -------------------------------------------------------

def load_runtime_enums(mirage_root: str):
    """Parse runtime_header.h to map TaskType + EventType ints -> names."""
    path = os.path.join(mirage_root, "include/mirage/persistent_kernel/runtime_header.h")
    task_names: dict[int, str] = {}
    event_names: dict[int, str] = {}
    cur = None
    for ln in open(path):
        s = ln.strip()
        if s.startswith("enum TaskType") or s.startswith("enum class TaskType"):
            cur = task_names
            continue
        if s.startswith("enum EventType") or s.startswith("enum class EventType"):
            cur = event_names
            continue
        if cur is not None:
            if s.startswith("}"):
                cur = None
                continue
            m = re.match(r"^([A-Z_][A-Z0-9_]*)\s*=\s*(\d+)", s)
            if m:
                cur[int(m.group(2))] = m.group(1)
    return task_names, event_names


# -- perfetto loader --------------------------------------------------------

@dataclass
class Slice:
    """One perfetto BEGIN/END pair = one CTA execution."""
    task_type_name: str
    task_inst_id: int  # the _N suffix on the perfetto label
    track_uuid: int
    begin_ns: int
    end_ns: int

    @property
    def dur_ns(self) -> int:
        return self.end_ns - self.begin_ns


def load_trace(path: str) -> list[Slice]:
    with open(path, "rb") as f:
        trace = perfetto_pb2.Trace.FromString(f.read())
    iid_to_name: dict[int, str] = {}
    for pkt in trace.packet:
        if pkt.HasField("interned_data"):
            for s in pkt.interned_data.event_names:
                iid_to_name[s.iid] = s.name
    events_per_track: dict[int, list] = defaultdict(list)
    for pkt in trace.packet:
        if pkt.HasField("track_event"):
            te = pkt.track_event
            if te.type not in (TYPE_SLICE_BEGIN, TYPE_SLICE_END):
                continue
            name = iid_to_name.get(te.name_iid, "")
            events_per_track[te.track_uuid].append((pkt.timestamp, te.type, name))
    slices: list[Slice] = []
    name_re = re.compile(r"^(TASK_[A-Z0-9_]+?)_(\d+)$")
    for tid, events in events_per_track.items():
        events.sort()
        stack: list[tuple[int, str]] = []
        for ts, etype, name in events:
            if etype == TYPE_SLICE_BEGIN:
                stack.append((ts, name))
            elif etype == TYPE_SLICE_END and stack:
                bts, bname = stack.pop()
                m = name_re.match(bname)
                if m:
                    slices.append(Slice(
                        task_type_name=m.group(1),
                        task_inst_id=int(m.group(2)),
                        track_uuid=tid,
                        begin_ns=bts,
                        end_ns=ts,
                    ))
    return slices


# -- task graph loader ------------------------------------------------------

@dataclass
class TaskNode:
    inst_id: int
    task_type_id: int
    task_type_name: str
    variant_id: int
    trigger_event: int
    dependent_event: int
    request_id: int
    inputs: list[dict]
    outputs: list[dict]


def load_task_graph(path: str, task_enum: dict[int, str]) -> tuple[list[TaskNode], list[dict]]:
    with open(path) as f:
        raw = json.load(f)
    tasks: list[TaskNode] = []
    for i, t in enumerate(raw["all_tasks"]):
        tasks.append(TaskNode(
            inst_id=i,
            task_type_id=t["task_type"],
            task_type_name=task_enum.get(t["task_type"], f"TYPE_{t['task_type']}"),
            variant_id=t["variant_id"],
            trigger_event=t["trigger_event"],
            dependent_event=t["dependent_event"],
            request_id=t.get("request_id", -1),
            inputs=t.get("inputs") or [],
            outputs=t.get("outputs") or [],
        ))
    events = raw["all_events"]
    return tasks, events


# -- analysis primitives ----------------------------------------------------

@dataclass
class TaskTypeStats:
    name: str
    n_events: int
    sum_dur_us: float
    avg_dur_us: float
    max_dur_us: float
    p50_dur_us: float
    p90_dur_us: float


def per_task_breakdown(slices: list[Slice]) -> list[TaskTypeStats]:
    by_type: dict[str, list[int]] = defaultdict(list)
    for s in slices:
        by_type[s.task_type_name].append(s.dur_ns)
    out: list[TaskTypeStats] = []
    for name, durs in by_type.items():
        durs_sorted = sorted(durs)
        out.append(TaskTypeStats(
            name=name,
            n_events=len(durs),
            sum_dur_us=sum(durs) / 1e3,
            avg_dur_us=statistics.mean(durs) / 1e3,
            max_dur_us=max(durs) / 1e3,
            p50_dur_us=durs_sorted[len(durs)//2] / 1e3,
            p90_dur_us=durs_sorted[min(int(len(durs)*0.9), len(durs)-1)] / 1e3,
        ))
    out.sort(key=lambda r: -r.sum_dur_us)
    return out


def per_inst_walls(slices: list[Slice]) -> dict[tuple[str, int], dict]:
    """For each (task_type_name, inst_id) compute call wallclock (max(end)-min(begin))."""
    by_inst: dict[tuple[str, int], list[Slice]] = defaultdict(list)
    for s in slices:
        by_inst[(s.task_type_name, s.task_inst_id)].append(s)
    out: dict[tuple[str, int], dict] = {}
    for key, ss in by_inst.items():
        starts = [s.begin_ns for s in ss]
        ends = [s.end_ns for s in ss]
        out[key] = dict(
            n_ctas=len(ss),
            min_start=min(starts),
            max_end=max(ends),
            wall_us=(max(ends) - min(starts)) / 1e3,
            sum_dur_us=sum(s.dur_ns for s in ss) / 1e3,
            avg_dur_us=statistics.mean(s.dur_ns for s in ss) / 1e3,
        )
    return out


def layer_starts(slices: list[Slice]) -> list[int]:
    """Use TOPK_SIGMOID BEGIN events to mark per-MoE-layer starts."""
    topk = [s for s in slices if "TOPK_SIGMOID" in s.task_type_name]
    topk.sort(key=lambda s: s.begin_ns)
    starts: list[int] = []
    prev = None
    LAYER_GAP_NS = 100_000  # cluster within 100 us
    for s in topk:
        if prev is None or s.begin_ns - prev > LAYER_GAP_NS:
            starts.append(s.begin_ns)
        prev = s.begin_ns
    return starts


def per_layer_walls(layers: list[int]) -> list[dict]:
    """Per-MoE-layer wallclock from topk_start[i] -> topk_start[i+1]."""
    out: list[dict] = []
    for i, ts in enumerate(layers):
        if i + 1 < len(layers):
            out.append(dict(layer=i, start_us=ts/1e3, dur_us=(layers[i+1]-ts)/1e3))
        else:
            out.append(dict(layer=i, start_us=ts/1e3, dur_us=None))  # last
    return out


# -- event-chain analysis ---------------------------------------------------

@dataclass
class EventFanout:
    trigger_event: int
    n_tasks: int
    task_type_breakdown: dict[str, int]
    total_cta_count: int  # sum over tasks in fan-out (= total CTAs in this wave)
    over_worker_factor: float  # CTA_count / num_workers (default 136 = current MPK default)


def trigger_fanouts(tasks: list[TaskNode], num_workers: int = 136) -> list[EventFanout]:
    by_trig: dict[int, list[TaskNode]] = defaultdict(list)
    for t in tasks:
        by_trig[t.trigger_event].append(t)
    out: list[EventFanout] = []
    for ev, ts in by_trig.items():
        tc = Counter(t.task_type_name for t in ts)
        n_cta = len(ts)  # each TaskNode is 1 CTA in MPK runtime
        out.append(EventFanout(
            trigger_event=ev,
            n_tasks=len(ts),
            task_type_breakdown=dict(tc),
            total_cta_count=n_cta,
            over_worker_factor=round(n_cta / num_workers, 2),
        ))
    out.sort(key=lambda e: -e.total_cta_count)
    return out


@dataclass
class DepMismatch:
    event: int
    declared_triggers: int
    actual_producers: int
    actual_consumers: int
    note: str


def dep_consistency_check(tasks: list[TaskNode], events: list[dict]) -> list[DepMismatch]:
    out: list[DepMismatch] = []
    cons_count: dict[int, int] = defaultdict(int)
    prod_count: dict[int, int] = defaultdict(int)
    for t in tasks:
        cons_count[t.trigger_event] += 1
        prod_count[t.dependent_event] += 1
    INVALID = 9223372036854775806
    for i, e in enumerate(events):
        if e.get("event_type") != 900:
            continue  # only EVENT_EMPTY have counter semantics
        nt = e["num_triggers"]
        ac = cons_count.get(i, 0)
        ap = prod_count.get(i, 0)
        # heuristic: num_triggers usually equals consumer count.
        if nt != ac:
            out.append(DepMismatch(
                event=i,
                declared_triggers=nt,
                actual_producers=ap,
                actual_consumers=ac,
                note=f"num_triggers={nt} but actual consumers={ac}; producers={ap}",
            ))
    return out


# -- buffer producer/consumer cross-reference -------------------------------

def buffer_flow(tasks: list[TaskNode], buffer_name_filter: Optional[str] = None) -> dict:
    """Map base_ptr -> {producers: [task_inst_ids], consumers: [task_inst_ids]}."""
    flow: dict[str, dict[str, list[int]]] = {}
    for t in tasks:
        for o in t.outputs:
            bp = o.get("base_ptr")
            if not bp: continue
            if buffer_name_filter and buffer_name_filter not in bp: continue
            flow.setdefault(bp, {"producers": [], "consumers": []})
            flow[bp]["producers"].append(t.inst_id)
        for o in t.inputs:
            bp = o.get("base_ptr")
            if not bp: continue
            if buffer_name_filter and buffer_name_filter not in bp: continue
            flow.setdefault(bp, {"producers": [], "consumers": []})
            flow[bp]["consumers"].append(t.inst_id)
    return flow


# -- critical-path chain estimator ------------------------------------------

def ascii_gantt(slices: list[Slice], t_start: int, t_end: int,
                width: int = 120, max_rows: int = 30) -> str:
    """Render a fixed-width ASCII Gantt of the focus layer window.
    One row per task type. Bars use '#'. Rows ordered by earliest start
    so the dependency chain reads top-to-bottom.
    """
    in_win = [s for s in slices if s.begin_ns >= t_start and s.end_ns <= t_end]
    if not in_win or t_end <= t_start:
        return "(no slices in window)"
    span = t_end - t_start
    ns_per_col = span / width
    by_type: dict[str, list[Slice]] = defaultdict(list)
    for s in in_win:
        by_type[s.task_type_name].append(s)
    # order by earliest begin_ns
    ordered = sorted(by_type.items(), key=lambda kv: min(s.begin_ns for s in kv[1]))
    lines: list[str] = []
    # scale header
    scale_us = span / 1e3
    scale_marks = "0" + "".join(
        (f"{int(round((c+1) * scale_us / 6, 0))}μs").rjust((width - 1) // 6)
        for c in range((width - 1) // ((width - 1) // 6 + 1) if False else 5)
    )
    # simpler header
    lines.append(f"  {'task':<36} | 0{' ' * (width - 2)}{scale_us:>6.0f}μs")
    lines.append(f"  {'-'*36} | {'-'*width}")
    for name, ss in ordered[:max_rows]:
        bar = ['.'] * width
        for s in ss:
            i0 = int((s.begin_ns - t_start) / ns_per_col)
            i1 = int((s.end_ns - t_start) / ns_per_col)
            i0 = max(0, min(width - 1, i0))
            i1 = max(0, min(width - 1, i1))
            for k in range(i0, i1 + 1):
                bar[k] = '#'
        lines.append(f"  {name:<36} | {''.join(bar)}")
    if len(ordered) > max_rows:
        lines.append(f"  ... ({len(ordered) - max_rows} more task types)")
    return "\n".join(lines)


def critical_path_in_window(slices: list[Slice], t_start: int, t_end: int) -> list[dict]:
    """Greedy backward walk: start from latest-ending cluster in window,
    walk backward by finding the latest-ending task strictly before current
    cluster start. Approximate, ignores overlapping fan-outs."""
    in_win = [s for s in slices if s.begin_ns >= t_start and s.end_ns <= t_end]
    if not in_win:
        return []
    # cluster by (task_type, contiguous time)
    clusters: dict[str, list[Slice]] = defaultdict(list)
    for s in in_win:
        clusters[s.task_type_name].append(s)
    cluster_walls: list[tuple[str, int, int]] = []
    for name, ss in clusters.items():
        starts = sorted(s.begin_ns for s in ss)
        ends = sorted(s.end_ns for s in ss)
        # split clusters with > 5 us internal gap
        cur_start = starts[0]
        cur_end = ends[0]
        for i in range(1, len(starts)):
            if starts[i] - cur_end > 5_000:
                cluster_walls.append((name, cur_start, cur_end))
                cur_start = starts[i]
            cur_end = max(cur_end, ends[i])
        cluster_walls.append((name, cur_start, cur_end))
    cluster_walls.sort(key=lambda c: c[2])  # by end_ts ascending
    if not cluster_walls:
        return []
    # walk backward
    chain: list[tuple[str, int, int]] = [cluster_walls[-1]]
    while True:
        cur_start = chain[-1][1]
        cand = [c for c in cluster_walls if c[2] <= cur_start]
        if not cand:
            break
        prev = max(cand, key=lambda c: c[2])
        if prev == chain[-1]:
            break
        chain.append(prev)
        if len(chain) > 40:
            break  # safety
    chain.reverse()
    return [dict(task=name, begin_us=(b - t_start)/1e3, end_us=(e - t_start)/1e3, dur_us=(e-b)/1e3) for name, b, e in chain]


# -- main analysis driver ---------------------------------------------------

def run_analysis(trace_dir: str, rank: int = 0, focus_layer: Optional[int] = None,
                 mirage_root: str = ".",  # repo root (archived copy defaults to cwd)
                 num_workers: int = 136) -> dict:
    trace_path = os.path.join(trace_dir, f"trace_rank{rank}.perfetto-trace")
    task_graph_path = os.path.join(trace_dir, "build", f"task_graph_rank{rank}.json")
    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"missing trace: {trace_path}")
    if not os.path.exists(task_graph_path):
        raise FileNotFoundError(f"missing task graph: {task_graph_path}")
    task_enum, event_enum = load_runtime_enums(mirage_root)
    slices = load_trace(trace_path)
    tasks, events = load_task_graph(task_graph_path, task_enum)

    # Shape
    if not slices:
        return {"error": "no slices"}
    min_t = min(s.begin_ns for s in slices)
    max_t = max(s.end_ns for s in slices)
    trace_span_us = (max_t - min_t) / 1e3
    n_iters = sum(1 for s in slices if s.task_type_name == "TASK_BEGIN_TASK_GRAPH")

    # Per-task breakdown
    per_task = per_task_breakdown(slices)

    # Per-instance wallclock (groups of CTAs)
    walls = per_inst_walls(slices)

    # Layer segmentation
    layer_t = layer_starts(slices)
    per_layer = per_layer_walls(layer_t)
    n_moe_layers = len(layer_t)

    # Focus layer Gantt
    if focus_layer is None and n_moe_layers >= 3:
        focus_layer = n_moe_layers // 2
    elif focus_layer is None:
        focus_layer = 0
    focus_window: Optional[tuple[int, int]] = None
    if 0 <= focus_layer < len(layer_t) - 1:
        focus_window = (layer_t[focus_layer], layer_t[focus_layer + 1])

    # Critical path on focus layer
    crit_path: list[dict] = []
    gantt: str = ""
    if focus_window:
        crit_path = critical_path_in_window(slices, *focus_window)
        gantt = ascii_gantt(slices, *focus_window)

    # Event fan-out
    fanouts = trigger_fanouts(tasks, num_workers=num_workers)

    # Dep consistency
    dep_mismatches = dep_consistency_check(tasks, events)

    # Per-task wallclock variance (catches the user-flagged "same task different time")
    task_inst_stats: list[dict] = []
    by_type_walls: dict[str, list[float]] = defaultdict(list)
    for (tp, _), w in walls.items():
        by_type_walls[tp].append(w["wall_us"])
    for tp, ws in by_type_walls.items():
        if len(ws) < 4 or max(ws) < 5.0: continue
        std = statistics.pstdev(ws) if len(ws) > 1 else 0.0
        cv = std / statistics.mean(ws) if statistics.mean(ws) > 0 else 0.0
        task_inst_stats.append(dict(
            task=tp, n_instances=len(ws),
            min_us=round(min(ws), 2),
            mean_us=round(statistics.mean(ws), 2),
            max_us=round(max(ws), 2),
            cv=round(cv, 3),
        ))
    task_inst_stats.sort(key=lambda r: -r["cv"])

    summary = dict(
        trace_path=trace_path,
        task_graph_path=task_graph_path,
        rank=rank,
        focus_layer=focus_layer,
        trace_span_us=round(trace_span_us, 1),
        n_iters=n_iters,
        n_total_slices=len(slices),
        n_task_instances=len(tasks),
        n_events=len(events),
        n_moe_layer_starts=n_moe_layers,
        per_layer=per_layer,
        per_task_breakdown=[asdict(r) for r in per_task[:30]],
        critical_path=crit_path,
        top_fanouts=[asdict(f) for f in fanouts[:15]],
        dep_mismatches=[asdict(m) for m in dep_mismatches[:20]],
        n_dep_mismatches_total=len(dep_mismatches),
        task_variance=task_inst_stats[:20],
        gantt_ascii=gantt,
    )
    return summary


# -- output formatters ------------------------------------------------------

def fmt_markdown(s: dict) -> str:
    L: list[str] = []
    L.append(f"# Perfetto Analysis Report")
    L.append(f"")
    L.append(f"- trace: `{s['trace_path']}`")
    L.append(f"- task_graph: `{s['task_graph_path']}`")
    L.append(f"- trace_span: **{s['trace_span_us']/1000:.2f} ms** ({s['n_iters']} iter(s), {s['n_total_slices']} slices, {s['n_task_instances']} TaskDescs)")
    L.append(f"- MoE layers detected: {s['n_moe_layer_starts']}")
    if s['per_layer']:
        deltas = [r['dur_us'] for r in s['per_layer'] if r['dur_us'] is not None]
        if deltas:
            L.append(f"- per-MoE-layer wallclock: min={min(deltas):.1f} μs, mean={statistics.mean(deltas):.1f} μs, max={max(deltas):.1f} μs (target ~100 μs)")
    L.append("")
    L.append(f"## Per-task wallclock (top 20 by Σ duration)")
    L.append(f"| task | n | sum_ms | avg_us | p90_us | max_us |")
    L.append(f"|---|---|---|---|---|---|")
    for r in s['per_task_breakdown'][:20]:
        L.append(f"| {r['name']} | {r['n_events']} | {r['sum_dur_us']/1000:.2f} | {r['avg_dur_us']:.2f} | {r['p90_dur_us']:.2f} | {r['max_dur_us']:.2f} |")
    L.append("")
    L.append(f"## Task wallclock variance (sorted by CV — catches \"same task different time\")")
    L.append(f"| task | n | min_us | mean_us | max_us | CV |")
    L.append(f"|---|---|---|---|---|---|")
    for r in s['task_variance'][:12]:
        L.append(f"| {r['task']} | {r['n_instances']} | {r['min_us']} | {r['mean_us']} | {r['max_us']} | {r['cv']} |")
    L.append("")
    L.append(f"## Trigger-event fan-out (catches \"too many CTAs in one wave\")")
    L.append(f"With 136 workers, total_cta_count > 136 means > 1 wave needed.")
    L.append(f"| trigger_event | total_CTAs | over_worker_factor | task_type_breakdown |")
    L.append(f"|---|---|---|---|")
    for f in s['top_fanouts'][:12]:
        L.append(f"| ev{f['trigger_event']} | {f['total_cta_count']} | {f['over_worker_factor']}× | {f['task_type_breakdown']} |")
    L.append("")
    L.append(f"## Focus layer {s['focus_layer']} Gantt (ASCII)")
    if s.get('gantt_ascii'):
        L.append("```")
        L.append(s['gantt_ascii'])
        L.append("```")
    L.append("")
    L.append(f"## Critical path estimate (focus layer {s['focus_layer']})")
    if s['critical_path']:
        L.append(f"```")
        for c in s['critical_path']:
            L.append(f"  {c['task']:<42} @ {c['begin_us']:>8.1f} → {c['end_us']:>8.1f}  ({c['dur_us']:>6.1f} μs)")
        total = sum(c['dur_us'] for c in s['critical_path'])
        L.append(f"")
        L.append(f"  total chain: {total:.1f} μs")
        L.append(f"```")
    else:
        L.append(f"(no chain — only {s['n_moe_layer_starts']} MoE-layer starts)")
    L.append("")
    L.append(f"## Dep-graph consistency")
    if s['dep_mismatches']:
        L.append(f"⚠ {s['n_dep_mismatches_total']} EVENT_EMPTY events declare num_triggers != actual consumer count. Top 10:")
        L.append(f"| event | declared_triggers | actual_consumers | actual_producers | note |")
        L.append(f"|---|---|---|---|---|")
        for d in s['dep_mismatches'][:10]:
            L.append(f"| ev{d['event']} | {d['declared_triggers']} | {d['actual_consumers']} | {d['actual_producers']} | {d['note']} |")
    else:
        L.append(f"✓ all EVENT_EMPTY counters match consumer count (no obvious dep bug)")
    L.append("")
    L.append(f"## Per-MoE-layer wallclock (TOPK_SIGMOID → TOPK_SIGMOID)")
    L.append(f"| layer | start_us | dur_us |")
    L.append(f"|---|---|---|")
    for r in s['per_layer']:
        d = f"{r['dur_us']:.1f}" if r['dur_us'] is not None else "(last)"
        L.append(f"| L{r['layer']} | {r['start_us']:.1f} | {d} |")
    return "\n".join(L)


# -- CLI --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir", help="output dir containing trace_rank0.perfetto-trace and build/task_graph_rank0.json")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--layer", type=int, default=None, help="focus MoE layer index (default: middle)")
    ap.add_argument("--write", action="store_true", help="write analysis.md + analysis.json into trace_dir")
    ap.add_argument("--mirage-root", default=".", help="mirage repo root (run from repo root or pass explicitly)")
    ap.add_argument("--num-workers", type=int, default=136,
                    help="num_workers (default 136 = current MPK default per commit 73394982)")
    args = ap.parse_args()

    summary = run_analysis(args.trace_dir, rank=args.rank, focus_layer=args.layer,
                           mirage_root=args.mirage_root, num_workers=args.num_workers)
    md = fmt_markdown(summary)
    print(md)
    if args.write:
        md_path = os.path.join(args.trace_dir, "analysis.md")
        json_path = os.path.join(args.trace_dir, "analysis.json")
        with open(md_path, "w") as f: f.write(md)
        with open(json_path, "w") as f: json.dump(summary, f, indent=2)
        print(f"\nWrote {md_path} and {json_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
