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
  {"task_type", "key", "context": {"first_on_sm", "prev"}, "median_us", "p75_us", "p90_us", "n", "ctx_stats", "run",
   "build": "profile", "machine"}
p75_us: the 0.75 quantile (rule D4: the span follows the slowest tasks, so the loop's model reads p75, not the median).
ctx_stats (rule D2), over the record's values, without splitting the record (search.Costs sees the same groups as before):
  cold_frac          the share of values whose task was the first task of its node on its SM (its code not yet run there:
                     cold instruction cache, no dry-pass benefit). A gemm_tile task behind a topk_route task on SM 3 with no
                     gemm_tile before it on SM 3: cold; the next gemm_tile on SM 3: warm.
  in_flight_median   the median, over the values, of how many other tasks were running on the same GPU (any SM) when the task
                     started: tasks whose [start, end) holds this task's start. 140 GEMM tiles started together: 139 each.
KEEP_VALUES = True: each record also keeps "values": [[us, cold (0 / 1), in flight], ...] (for fitting context tables; large
files, so off by default).
machine = {"gpu": the device name, "cuda": the nvcc version}: costs of different GPU variants and CUDA versions are kept apart
(one file per machine, e.g. b300_sxm6pc_cuda13.0.json)."""
import bisect
import json
import os
import statistics
from typing import Callable, Dict, List, Optional, Tuple

KEEP_VALUES = False   # True: every record keeps its values with their context (see the module comment)


def quantile(values: List[float], q: float) -> float:
    """values sorted; the value at index int(q (n - 1)) (p90_us has always been read this way). [1, 2, 3, 4, 5], 0.75 -> 4."""
    return values[int(q * (len(values) - 1))]


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
            s0 = gpu_times[sm][i][2] if len(gpu_times[sm][i]) > 2 else 0
            out[t] = {"start": start, "end": end, "s0": s0, "prev_end": start if prev_end is None else prev_end, "index": i,
                      "prev": None if i == 0 else sm_list[i - 1], "sm": sm}
            prev_end = end
    return out


def launch_durations(schedules: List[dict], times: list, bars: List[int], pool_node: int, gating_nodes: set,
                     arrival_nodes: frozenset = frozenset()):
    """One launch: (per GPU {task id: duration us} of the placed tasks, per GPU the pool's work in us). arrival_nodes: nodes whose
    inputs come from every GPU (their layer's stage0_inputs, e.g. attn_residual: every GPU's o_proj rows): their inputs are
    ready when the tasks they depend on have ended on every GPU (times after each GPU's start barrier, as the pool's opening;
    the GPUs' task ids are the same) and not before their stage 0 stamp (thread 0's first vector landed from every GPU), so
    waiting for other GPUs is not counted as their own time. (10-14: with the stage 0 stamp alone, KDA random plans, whose
    GPUs run different lists, recorded attn_residual at 8..24 us vs 2.85 in the start plan: the rest of the rows landed after
    the first vector.)
    A concurrent group (a node's concurrent_along_axis: the tasks that wait for each other, e.g. mla_attn_part's splits of one
    head group, sum_rmsnorm's halves of one token): every member is ready when the latest member is (the solver starts a group
    together), so waiting for the group's other members is not counted as a member's own time. (10-14: MLA random plans
    recorded mla_attn_part at 9..80 us vs 8.5 in the start plan, whose splits start together.)"""
    stamps = [stamps_of_gpu(s, times[g]) for g, s in enumerate(schedules)]
    opening = max([st["end"] - bars[g] for g, s in enumerate(schedules) for t, st in stamps[g].items()
                   if s["all_tasks"][t]["node"] in gating_nodes] + [0])
    placed, pool_work = [], []
    for g, s in enumerate(schedules):
        tasks, d, work, ready = s["all_tasks"], {}, 0.0, {}
        for t, st in stamps[g].items():
            if tasks[t]["node"] == pool_node:
                work += (st["end"] - max(st["prev_end"], bars[g] + opening)) / 1e3
            else:
                r = max([stamps[g][p]["end"] for p in tasks[t]["deps"] if p in stamps[g]] + [st["prev_end"]])
                if tasks[t]["node"] in arrival_nodes:
                    r = max([r, st["s0"]] + [bars[g] + stamps[h][p]["end"] - bars[h] for h in range(len(schedules))
                                             for p in tasks[t]["deps"] if p in stamps[h]])
                ready[t] = r
        group_of, group_ready = {}, {}
        for t in ready:
            a = (s["nodes"].get(str(tasks[t]["node"])) or {}).get("concurrent_along_axis")
            if a is not None:
                group_of[t] = (tasks[t]["node"],) + tuple(p for i, p in enumerate(tasks[t]["pos"]) if i != a)
                group_ready[group_of[t]] = max(group_ready.get(group_of[t], 0), ready[t])
        for t, r in ready.items():
            st = stamps[g][t]
            d[t] = (st["end"] - max(st["prev_end"], group_ready.get(group_of.get(t), r))) / 1e3
        placed.append(d)
        pool_work.append(work)
    return placed, pool_work, stamps


from .search import cost_key   # noqa: E402  the key of a record (the pool's: without the producers' grids)


def records_from_run(schedules: List[dict], launches: List[Tuple[list, list]], nodes: Dict[int, object],
                     key_of: Callable[[object], Optional[dict]], pool_node: int, gating_nodes: set, run: str,
                     machine: dict, arrival_nodes: frozenset = frozenset()) -> List[dict]:
    """launches = [(times[g][sm][i], timing[g] = (start barrier passed, end))]; key_of(node) = the node's cost key (None: no
    record); nodes = {graph_idx: Node} as compiled."""
    groups: Dict[str, List[Tuple[float, bool, int]]] = {}   # record -> its values (us, cold, in flight)
    pool = []
    for times, timing in launches:
        bars = [b for b, _ in timing]
        placed, work, stamps = launch_durations(schedules, times, bars, pool_node, gating_nodes, arrival_nodes)
        pool += work
        for g, s in enumerate(schedules):
            tasks = s["all_tasks"]
            cold, in_flight = task_context(stamps[g], tasks)
            for t, us in placed[g].items():
                n = nodes[tasks[t]["node"]]
                key = key_of(n)
                if key is None:
                    continue
                prev = stamps[g][t]["prev"]
                context = {"first_on_sm": stamps[g][t]["index"] == 0, "prev": None if prev is None else nodes[tasks[prev]["node"]].name}
                groups.setdefault(json.dumps([n.name, key, context], sort_keys=True), []).append((us, cold[t], in_flight[t]))
    records = []
    for k, entries in sorted(groups.items()):
        task_type, key, context = json.loads(k)
        values = sorted(us for us, _, _ in entries)
        r = {"task_type": task_type, "key": key, "context": context, "median_us": round(statistics.median(values), 4),
             "p75_us": round(quantile(values, 0.75), 4), "p90_us": round(quantile(values, 0.9), 4), "n": len(values),
             "ctx_stats": {"cold_frac": round(sum(c for _, c, _ in entries) / len(entries), 4),
                           "in_flight_median": statistics.median(f for _, _, f in entries)},
             "run": run, "build": "profile", "machine": machine}
        if KEEP_VALUES:
            r["values"] = [[round(us, 4), int(c), f] for us, c, f in entries]
        records.append(r)
    if pool_node is not None and pool:
        pool = sorted(pool)
        records.append({"task_type": nodes[pool_node].name, "key": cost_key(nodes[pool_node], nodes[pool_node].grid), "context": {},
                        "median_us": round(statistics.median(pool), 2), "p75_us": round(quantile(pool, 0.75), 2),
                        "p90_us": round(quantile(pool, 0.9), 2), "n": len(pool), "run": run, "build": "profile", "machine": machine})
    return records


def task_context(stamps: Dict[int, dict], tasks: List[dict]) -> Tuple[Dict[int, bool], Dict[int, int]]:
    """For one GPU's stamped tasks (stamps_of_gpu): cold[t] = no task of t's node ran earlier on t's SM; in_flight[t] = how many
    other tasks of the GPU were running at t's start (start <= t's start < end). SM 0 runs [router 0, router 7, route 0]:
    router 0 and route 0 are cold, router 7 warm."""
    cold, seen = {}, set()
    for t, st in sorted(stamps.items(), key=lambda kv: (kv[1]["sm"], kv[1]["index"])):
        k = (st["sm"], tasks[t]["node"])
        cold[t] = k not in seen
        seen.add(k)
    starts = sorted(st["start"] for st in stamps.values())
    ends = sorted(st["end"] for st in stamps.values())
    # tasks started at or before s minus tasks ended at or before s = tasks running at s (this task included: - 1)
    in_flight = {t: bisect.bisect_right(starts, st["start"]) - bisect.bisect_right(ends, st["start"]) - (st["end"] > st["start"])
                 for t, st in stamps.items()}
    return cold, in_flight


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
    from .static_schedule import code_tag, task_code_tag
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
    arrival = frozenset(k for k, n in nodes.items() if n.info.get("stage0_inputs"))
    records = records_from_run(schedules, launches, nodes, key_of, pool, gating, run, machine, arrival)
    tag, task_tags = code_tag(), {}
    for r in records:
        r["code"] = tag   # the kernel templates these times were measured with
        if r["task_type"] not in task_tags:
            task_tags[r["task_type"]] = task_code_tag(r["task_type"])
        r["task_code"] = task_tags[r["task_type"]]   # this task type's own code (static_schedule.task_code_tag)
    return records
