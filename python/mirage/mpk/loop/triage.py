"""Triage (docs/megakernel_dev_loop.md section 9): the critical path of a timed launch and where its time goes.

  critical_path(schedules, times, timing)   the chain of tasks that sets the launch's span on the latest-ending GPU
  attribute(path)                           the span split into classes, summed per class
  python -m mirage.mpk.loop.triage RUN_DIR [--launch 0] [--json OUT] [--exchange-nodes 30,37] [--pool-node 38]
                                            RUN_DIR = a run.py --record directory (schedules/, task_times.npy, timing.npy)

The walk. Times are the timing build's stamps (cost_records.stamps_of_gpu): a task's start stamp is taken when its SM reaches the
list entry, and the task then waits for its inputs inside its body, so (rule D1) the task's own time begins at
  begin = max(end of the task before it on the SM (its own start if it is first), latest end of the tasks it depends on).
The path starts at the latest-ending task of the GPU whose (latest end - start barrier) is largest. From a task it goes back to
the task before it on the SM or to its latest-ending dependency, whichever ended later (that one ended exactly at `begin`). It stops
at a task that is first on its SM and began at its own start stamp: the time from the start barrier to that start is "start". The
segments [begin, end] of the path's tasks and that start time add up to the GPU's span.

The classes of a path task's segment (seg = end - begin):
  work:<task type>  the node's median segment over its warm tasks (not the first task of the node on their SM) on all GPUs in this
                    launch; no bound formula exists yet, so the whole of it counts as work ("no bound" in the notes)
  cold              a cold task (the first task of its node on its SM): up to (the node's median cold segment - its median warm
                    segment), out of the time above the warm median
  cross_gpu         a task that reads a node in exchange_nodes (its output is in the cross-GPU exchange region; the schedule's
                    dependencies are this GPU's tasks only): up to (the latest end of that node on any GPU - begin), out of the time
                    left above the warm median. Without exchange_nodes this class is not computed.
  imbalance         a pool task (one task per SM, sharing work at run time): its end - the mean end of the pool's tasks on the GPU
  handoff           the rest of the time above the warm median: time inside the task that its warm median, cold code and other GPUs
                    do not explain (waiting for a counter or a polled value to arrive)
  start             start barrier -> the path's first task's start
Each path task also reports waited_us = max(0, latest dependency end - max(own start, previous task end)): the time the SM sat in
the task before its inputs were ready. That time is not on the path (the path goes through the dependency there), so it is not a
class; it shows hand-offs whose producers are late.
"""
import argparse
import glob
import json
import os
import re
import statistics
from typing import Dict, Iterable, List, Optional


def _durations(schedule: dict, st: Dict[int, dict]) -> Dict[int, dict]:
    """Per stamped task: begin (rule D1), ready (latest dependency end or None), seg = end - begin (ns)."""
    out = {}
    for t, s in st.items():
        deps = [d for d in schedule["all_tasks"][t]["deps"] if d in st]
        ready = max(st[d]["end"] for d in deps) if deps else None
        begin = s["prev_end"] if ready is None else max(s["prev_end"], ready)
        out[t] = {"begin": begin, "ready": ready, "seg": s["end"] - begin,
                  "dep": max(deps, key=lambda d: st[d]["end"]) if deps else None}
    return out


def find_pool_node(schedules: List[dict]) -> Optional[int]:
    """A node with exactly one task on every SM, task x on SM x (one_task_per_sm; MoE: moe_experts), or None."""
    s = schedules[0]
    sms = len(s["worker_task_queues"])
    for idx, node in s["nodes"].items():
        if node["grid"][0] != sms:
            continue
        ok = True
        for sm, sm_list in enumerate(s["worker_task_queues"]):
            mine = [t for t in sm_list if s["all_tasks"][t]["node"] == int(idx)]
            if len(mine) != 1 or s["all_tasks"][mine[0]]["pos"][0] != sm:
                ok = False
                break
        if ok:
            return int(idx)
    return None


def critical_path(schedules: List[dict], times: list, timing: list, exchange_nodes: Iterable[int] = (),
                  pool_node: Optional[int] = None) -> dict:
    """One launch: times[gpu][sm][entry] = (start, end, stage 0, stage 1) ns; timing[gpu] = (start barrier passed, end) ns.
    pool_node: the pool's graph_idx; None: find_pool_node; -1: no pool.
    Returns {"gpu", "span_us", "start_us", "path": [{gpu, sm, task, node, name, pos, start, begin, end, seg, classes,
    waited_us}, ...] (times in us after the GPU's start barrier, the path's first task first)}."""
    from ..cost_records import stamps_of_gpu, task_context
    exchange_nodes = set(exchange_nodes)
    if pool_node is None:
        pool_node = find_pool_node(schedules)
    stamps = [stamps_of_gpu(s, times[g]) for g, s in enumerate(schedules)]
    bars = [tim[0] for tim in timing]
    durs = [_durations(s, stamps[g]) for g, s in enumerate(schedules)]
    colds = [task_context(stamps[g], s["all_tasks"])[0] for g, s in enumerate(schedules)]
    warm: Dict[int, List[float]] = {}
    cold: Dict[int, List[float]] = {}
    last_end: Dict[int, float] = {}             # node -> its latest end on any GPU, ns after that GPU's start barrier
    pool_ends: List[List[float]] = [[] for _ in schedules]
    for g, s in enumerate(schedules):
        for t, d in durs[g].items():
            node = s["all_tasks"][t]["node"]
            (cold if colds[g][t] else warm).setdefault(node, []).append(d["seg"])
            last_end[node] = max(last_end.get(node, float("-inf")), stamps[g][t]["end"] - bars[g])
            if node == pool_node:
                pool_ends[g].append(stamps[g][t]["end"])
    warm_med = {n: statistics.median(v) for n, v in warm.items()}
    cold_med = {n: statistics.median(v) for n, v in cold.items()}

    g = max(range(len(schedules)), key=lambda k: max((st["end"] for st in stamps[k].values()), default=0) - bars[k])
    s, st, du, bar = schedules[g], stamps[g], durs[g], bars[g]
    cur = max(st, key=lambda t: st[t]["end"])
    span_ns = st[cur]["end"] - bar
    path = []
    while True:
        task, d, node = st[cur], du[cur], s["all_tasks"][cur]["node"]
        seg = d["seg"]
        classes: Dict[str, float] = {}
        name = s["nodes"][str(node)]["name"]
        if node == pool_node and pool_ends[g]:
            imb = min(seg, max(0.0, task["end"] - statistics.mean(pool_ends[g])))
            classes["imbalance"] = imb
            classes[f"work:{name}"] = seg - imb
        else:
            expected = min(seg, warm_med.get(node, seg))
            excess = seg - expected
            if colds[g][cur] and excess > 0 and node in cold_med and node in warm_med:
                classes["cold"] = min(excess, max(0.0, cold_med[node] - warm_med[node]))
                excess -= classes["cold"]
            reads_exchange = {s["all_tasks"][p]["node"] for p in s["all_tasks"][cur]["deps"]} & exchange_nodes
            if reads_exchange and excess > 0:
                remote = max(last_end[n] for n in reads_exchange if n in last_end) + bar if any(
                    n in last_end for n in reads_exchange) else d["begin"]
                classes["cross_gpu"] = min(excess, max(0.0, remote - d["begin"]))
                excess -= classes["cross_gpu"]
            if excess > 0:
                classes["handoff"] = excess
            classes[f"work:{name}"] = expected
        free = task["prev_end"]
        waited = 0.0 if d["ready"] is None else max(0.0, d["ready"] - max(task["start"], free))
        us = lambda v: round((v - bar) / 1e3, 3)
        path.append({"gpu": g, "sm": task["sm"], "task": cur, "node": node, "name": name, "pos": s["all_tasks"][cur]["pos"],
                     "start": us(task["start"]), "begin": us(d["begin"]), "end": us(task["end"]), "seg": round(seg / 1e3, 3),
                     "classes": {k: round(v / 1e3, 3) for k, v in classes.items() if v}, "waited_us": round(waited / 1e3, 3)})
        has_prev = task["index"] > 0
        if d["ready"] is not None and d["ready"] >= free and (not has_prev or d["ready"] > free):
            cur = d["dep"]                          # the dependency ended last (or at the same time as an SM start)
        elif has_prev:
            cur = task["prev"]
        else:
            break                                   # first on its SM, began at its own start: the rest is "start"
    path.reverse()
    start_ns = st[path[0]["task"]]["start"] - bar if path else 0
    return {"gpu": g, "span_us": round(span_ns / 1e3, 3), "start_us": round(start_ns / 1e3, 3), "path": path,
            "pool_node": pool_node}


def attribute(cp: dict, model_error: Optional[Dict[str, float]] = None) -> dict:
    """The class totals of a critical path (critical_path's result), largest first; they add up to the span. model_error: the
    model's error per phase (model.phase_error's phase_err), reported as "model" when a phase's error is over 1 us (class D: explore),
    not part of the span's sum."""
    totals: Dict[str, float] = {"start": cp["start_us"]}
    for p in cp["path"]:
        for k, v in p["classes"].items():
            totals[k] = totals.get(k, 0.0) + v
    totals = {k: round(v, 3) for k, v in sorted(totals.items(), key=lambda kv: -kv[1]) if v}
    out = {"span_us": cp["span_us"], "classes": totals, "sum_us": round(sum(totals.values()), 3),
           "notes": ["no bound: work:<task type> counts the whole warm median (no per-task bound formula yet)"]}
    if model_error:
        out["model"] = {ph: e for ph, e in model_error.items() if abs(e) > 1.0}
    return out


def stage_split(schedules: List[dict], task_times: list, timing: list, name: str) -> List[dict]:
    """The stage stamps of every task of the nodes named `name` (a timing build: the task type writes L.stage_stamps[0] and [1]),
    in us after its GPU's start barrier, the median over launches: start, inputs (the latest end of the tasks it depends on),
    stage 0, stage 1, end. [{gpu, sm, node, pos, start, inputs, s0, s1, end}] sorted by GPU and SM."""
    per: Dict[tuple, List[tuple]] = {}
    for times, tim in zip(task_times, timing):
        for g, s in enumerate(schedules):
            bar = tim[g][0]
            ends = {}
            for sm, sm_list in enumerate(s["worker_task_queues"]):
                for i, t in enumerate(sm_list):
                    if times[g][sm][i][1]:
                        ends[t] = times[g][sm][i][1]
            for sm, sm_list in enumerate(s["worker_task_queues"]):
                for i, t in enumerate(sm_list):
                    task = s["all_tasks"][t]
                    if s["nodes"][str(task["node"])]["name"] != name:
                        continue
                    st, en, s0, s1 = times[g][sm][i][:4]
                    if not st or not en:
                        continue
                    dep = max([ends[d] for d in task["deps"] if d in ends] + [bar])
                    rel = lambda v: (v - bar) / 1e3 if v else float("nan")
                    per.setdefault((g, sm, task["node"], tuple(task["pos"])), []).append(
                        (rel(st), rel(dep), rel(s0), rel(s1), rel(en)))
    out = []
    for (g, sm, node, pos), vs in sorted(per.items()):
        med = [round(statistics.median(v[k] for v in vs), 3) for k in range(5)]
        out.append(dict(gpu=g, sm=sm, node=node, pos=list(pos), start=med[0], inputs=med[1], s0=med[2], s1=med[3], end=med[4]))
    return out


def load_run(run_dir: str):
    """(schedules in GPU order, task_times [launch][gpu][sm][entry][4], timing [launch][gpu][2]) of a run.py --record directory."""
    import numpy as np
    paths = sorted(glob.glob(os.path.join(run_dir, "schedules", "schedule_gpu*.json")),
                   key=lambda p: int(re.search(r"schedule_gpu(\d+)\.json$", p).group(1)))
    schedules = []
    for p in paths:
        with open(p) as f:
            schedules.append(json.load(f))
    return schedules, np.load(os.path.join(run_dir, "task_times.npy")).tolist(), np.load(os.path.join(run_dir, "timing.npy")).tolist()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--launch", type=int, default=0, help="the launch whose path is printed")
    p.add_argument("--exchange-nodes", default="", help="graph_idx list (comma): nodes whose output is in the exchange region")
    p.add_argument("--pool-node", type=int, default=None, help="the one-task-per-SM node (default: found from the lists; -1: none)")
    p.add_argument("--json", default=None, help="write the paths and totals here")
    p.add_argument("--stages", default=None, metavar="NAME", help="instead: the stage stamps of every task of the nodes NAME")
    p.add_argument("--per-gpu", action="store_true", help="instead: per GPU, every node's latest task end (us after the barrier)")
    args = p.parse_args(argv)
    if args.per_gpu:
        schedules, task_times, timing = load_run(args.run_dir)
        names = sorted({n["name"] for n in schedules[0]["nodes"].values()})
        per = {name: stage_split(schedules, task_times, timing, name) for name in names}
        print("per GPU, each node's latest task end (us after the GPU's start barrier, median over launches):")
        for g in range(len(schedules)):
            print(f"  GPU {g}: " + " | ".join(f"{name} {max(x['end'] for x in per[name] if x['gpu'] == g):6.2f}"
                                             for name in names if any(x["gpu"] == g for x in per[name])))
        return
    if args.stages:
        schedules, task_times, timing = load_run(args.run_dir)
        rows = stage_split(schedules, task_times, timing, args.stages)
        print(f"{args.stages}: per task, median over {len(task_times)} launches, us after the start barrier "
              f"(inputs = the latest end of the tasks it depends on)")
        for r in rows:
            print(f"  GPU {r['gpu']} SM {r['sm']:>3} pos {tuple(r['pos'])}: start {r['start']:7.3f} inputs {r['inputs']:7.3f} "
                  f"s0 {r['s0']:7.3f} s1 {r['s1']:7.3f} end {r['end']:7.3f} | inputs->s1 {r['s1'] - max(r['inputs'], r['s0']):6.3f} "
                  f"s1->end {r['end'] - r['s1']:6.3f}")
        return
    ex = [int(v) for v in args.exchange_nodes.split(",") if v]
    schedules, task_times, timing = load_run(args.run_dir)
    results = []
    for i in range(len(task_times)):
        cp = critical_path(schedules, task_times[i], timing[i], ex, args.pool_node)
        results.append({"path": cp, "totals": attribute(cp)})
    cp = results[args.launch]["path"]
    print(f"launch {args.launch}: GPU {cp['gpu']}, span {cp['span_us']} us; critical path (us after the start barrier):")
    print(f"  start barrier -> first task: {cp['start_us']}")
    for e in cp["path"]:
        print(f"  {e['name']:<16} node {e['node']:>3} pos {tuple(e['pos'])} SM {e['sm']:>3}  begin {e['begin']:8.3f} end "
              f"{e['end']:8.3f}  {e['classes']}" + (f"  waited {e['waited_us']}" if e["waited_us"] else ""))
    keys = sorted({k for r in results for k in r["totals"]["classes"]})
    med = {k: round(statistics.median(r["totals"]["classes"].get(k, 0.0) for r in results), 3) for k in keys}
    med = dict(sorted(med.items(), key=lambda kv: -kv[1]))
    print(f"class totals, median over {len(results)} launches (us): {med}")
    print(f"span median {statistics.median(r['path']['span_us'] for r in results):.3f} us; " + results[0]["totals"]["notes"][0])
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"launches": results, "median_classes": med}, f, indent=1)


if __name__ == "__main__":
    main()
