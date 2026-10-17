"""Cost records from the timing build's stamps (static_schedule.compile_static(profile=True), static_harness.Harness.task_times).

Duration of a placed task i on its SM, in one launch:
  d_i = end_i - max(end of the task before it on the SM, latest end of the tasks it depends on)
(the first task on an SM: its own start instead of the previous end). This is the solver's rule read backwards (start = max(previous
end, inputs ready), end = start + d), so the durations and the waits add up to each SM's measured time.

The pool (the MoE expert queue, one task per SM): its work in one launch on one GPU = the sum over SMs of
  end - max(end of the task before it on the SM, the pool's opening)
where the opening = the latest end, over ALL GPUs (times after each GPU's start barrier), of the tasks of its gating nodes (MoE:
topk_route and sum_quant_send; the expert queue needs z_q from every GPU). One record per launch and GPU; the solver's pool rule uses the total.

Records (one per task type, cost key and context, over all tasks, GPUs and launches of a run):
  {"task_type", "key", "context": {"first_on_sm", "prev"}, "median_us", "p90_us", "n", "run", "build": "profile", "machine"}
machine = {"gpu": the device name, "cuda": the nvcc version}: costs of different GPU variants and CUDA versions are kept apart
(one file per machine, e.g. b300_sxm6pc_cuda13.0.json)."""
import json
import os
import statistics
from typing import Callable, Dict, List, Optional, Tuple


def stamps_of_gpu(schedule: dict, gpu_times: List[List[Tuple[int, int]]]) -> Dict[int, dict]:
    """Per task id: {"start", "end", "prev_end", "index"} (ns) from gpu_times[sm][i] = (start, end) of SM sm's i-th list entry;
    a list stops at the first entry without stamps."""
    out = {}
    for sm, sm_list in enumerate(schedule["worker_task_queues"]):
        prev_end = None
        for i, t in enumerate(sm_list):
            start, end = gpu_times[sm][i][:2]
            if not start or not end:
                break
            out[t] = {"start": start, "end": end, "prev_end": start if prev_end is None else prev_end, "index": i,
                      "prev": None if i == 0 else sm_list[i - 1]}
            prev_end = end
    return out


def launch_durations(schedules: List[dict], times: list, bars: List[int], pool_node: int, gating_nodes: set):
    """One launch: (per GPU {task id: duration us} of the placed tasks, per GPU the pool's work in us)."""
    stamps = [stamps_of_gpu(s, times[g]) for g, s in enumerate(schedules)]
    opening = max([st["end"] - bars[g] for g, s in enumerate(schedules) for t, st in stamps[g].items()
                   if s["all_tasks"][t]["node"] in gating_nodes] + [0])
    placed, pool_work = [], []
    for g, s in enumerate(schedules):
        tasks, d, work = s["all_tasks"], {}, 0.0
        for t, st in stamps[g].items():
            if tasks[t]["node"] == pool_node:
                work += (st["end"] - max(st["prev_end"], bars[g] + opening)) / 1e3
            else:
                ready = max([stamps[g][p]["end"] for p in tasks[t]["deps"] if p in stamps[g]] + [st["prev_end"]])
                d[t] = (st["end"] - max(st["prev_end"], ready)) / 1e3
        placed.append(d)
        pool_work.append(work)
    return placed, pool_work, stamps


from .search import cost_key   # noqa: E402  the key of a record (the pool's: without the producers' grids)


def records_from_run(schedules: List[dict], launches: List[Tuple[list, list]], nodes: Dict[int, object],
                     key_of: Callable[[object], Optional[dict]], pool_node: int, gating_nodes: set, run: str,
                     machine: dict) -> List[dict]:
    """launches = [(times[g][sm][i], timing[g] = (start barrier passed, end))]; key_of(node) = the node's cost key (None: no
    record); nodes = {graph_idx: Node} as compiled."""
    groups: Dict[str, List[float]] = {}
    pool = []
    for times, timing in launches:
        bars = [b for b, _ in timing]
        placed, work, stamps = launch_durations(schedules, times, bars, pool_node, gating_nodes)
        pool += work
        for g, s in enumerate(schedules):
            tasks = s["all_tasks"]
            for t, us in placed[g].items():
                n = nodes[tasks[t]["node"]]
                key = key_of(n)
                if key is None:
                    continue
                prev = stamps[g][t]["prev"]
                context = {"first_on_sm": stamps[g][t]["index"] == 0, "prev": None if prev is None else nodes[tasks[prev]["node"]].name}
                groups.setdefault(json.dumps([n.name, key, context], sort_keys=True), []).append(us)
    records = []
    for k, values in sorted(groups.items()):
        task_type, key, context = json.loads(k)
        values.sort()
        records.append({"task_type": task_type, "key": key, "context": context, "median_us": round(statistics.median(values), 4),
                        "p90_us": round(values[int(0.9 * (len(values) - 1))], 4), "n": len(values), "run": run, "build": "profile",
                        "machine": machine})
    if pool_node is not None and pool:
        records.append({"task_type": nodes[pool_node].name, "key": cost_key(nodes[pool_node], nodes[pool_node].grid), "context": {},
                        "median_us": round(statistics.median(pool), 2), "p90_us": round(sorted(pool)[int(0.9 * (len(pool) - 1))], 2),
                        "n": len(pool), "run": run, "build": "profile", "machine": machine})
    return records


def append_records(path: str, records: List[dict]) -> None:
    """Add records to the cost file at path (created if missing); records of the same run are replaced."""
    old = []
    if os.path.exists(path):
        with open(path) as f:
            old = json.load(f)["records"]
    runs = {r["run"] for r in records}
    with open(path, "w") as f:
        json.dump({"records": [r for r in old if r["run"] not in runs] + records}, f, indent=0)


def records_of_run(pk, schedule_paths: List[str], launches: List[Tuple[list, list]], run: str, machine: dict) -> List[dict]:
    """The cost records of a run of pk's layer in the timing build: launches = per timed launch (static_harness.Harness.task_times(),
    Harness.timing()). The pool: the node with one task per SM (moe_experts), opened by the nodes that write its inputs (topk_route,
    sum_quant_send); every record gets the current kernel templates' code tag (static_schedule.code_tag)."""
    from .compiler import graph_nodes, producer_of
    from .search import producer_grids
    from .static_schedule import code_tag
    schedules = []
    for path in schedule_paths:
        with open(path) as f:
            schedules.append(json.load(f))
    nodes = {n.graph_idx: n for n in graph_nodes(pk)}
    for k, n in nodes.items():   # the params compile_plan gave (sum_rmsnorm on cluster pairs: [2, eps])
        n.params = schedules[0]["nodes"][str(k)]["params"]
    pools = [k for k, n in nodes.items() if n.info["one_task_per_sm"]]
    if len(pools) > 1:
        raise ValueError(f"cost records: one pool node supported, the graph has {pools}")
    pool = pools[0] if pools else None
    gating = set() if pool is None else {p.graph_idx for p in (producer_of(t, list(nodes.values())) for t in nodes[pool].inputs) if p}
    key_of = lambda n: cost_key(n, n.grid, producer_grids(n, list(nodes.values())))
    records = records_from_run(schedules, launches, nodes, key_of, pool, gating, run, machine)
    tag = code_tag()
    for r in records:
        r["code"] = tag   # the kernel templates these times were measured with
    return records
