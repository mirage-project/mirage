"""Build the solver's input (solver.py) from the graph and the cost file, run the solver, and turn its answer into the plan that
compiler.compile_plan takes ({"grids", "lists"}). Nothing here is specific to a model; `cost` supplies the durations.

search_plan does, with the MoE layer's numbers:
  1. options    every node's possible grids (compiler.candidate_grids): a node with changeable grid dims (in the MoE layer the 3
                gemm_tile nodes) has several, router (7, k, 1) for k = 1, 2, 4, 7, 8, 14, 28, 56; every other node one, its
                own grid (topk_route (8, 1, 1)). Each node decides alone: topk_route does not choose with the router; its tasks wait for
                the router's tasks whatever the router's grid (the dependencies come from the slices).
  2. durations  every task's measured time from `cost`. A node whose input is written by a node with several options (topk_route
                reads the router's output) has one time per option of that node: topk_route adds as many partial sums as the router
                has K parts, so its time for router 7x14x1 is the time measured for adding 14. An option with a missing time
                (its own, or one of a node reading it) is left out (1, 2, 28, 56 K parts in the current cost file).
  3. GPU sets   GPUs whose tasks are the same once each GPU's own part of a split node is shifted to 0: GPUs 0, 2, 4, 6 compute 3
                latent_down row blocks, GPUs 1, 3, 5, 7 compute 4 -> "gpus_0_2_4_6" and "gpus_1_3_5_7". The solver places the
                tasks of the first GPU of each set; the other GPUs of the set get the same placement.
  4. tasks      per GPU set, every task of every option (router 7x14x1: 98 tasks, 7x4x1: 28, ...) with its duration(s); the
                dependencies (a task that reads a slice some other node's task writes waits for it: compiler.box / overlaps);
                the one-per-SM nodes are not placed: moe_experts becomes one Pool per GPU set (its measured total work),
                and every other task is placed, also the ones after it
  5. solve      solver.solve: one option per node, an SM and start time per task (at most 62 tasks per SM: the 64 entries of
                static_schedule.MAX_TASKS_PER_SM minus the end entry and the pool's task), the layer's end as early as possible
  6. plan       "grids": the chosen grid of each node with changeable grid dims; "lists": for every GPU, the solver's per-SM
                lists of its GPU set, each task moved from the set's first GPU's positions to this GPU's own positions
cost: task_us(node, grid, producer_grids) -> us or None (not measured); optional first_us(node, grid, producer_grids) -> its time
when it is the first task on its SM, or None (the same as task_us; solver rule 6); pool_us(node) -> a pool's total work (us, summed over
the SMs) or None if the node is not a pool: Costs below.
"""
import itertools
import json
import os
import statistics
from typing import Dict, List, Optional, Tuple

from . import static_schedule
from .compiler import (Node, box, can_pair, candidate_grids, dims, graph_nodes, overlaps, owned_positions, producer_of, tname)
from .solver import Pool, SolverTask, solve


def option_id(grid: tuple) -> str:
    """The name of an option in the solver: the grid written with x. (7, 14, 1) -> "7x14x1"."""
    return "x".join(str(v) for v in grid)


def as_grid(option: str) -> tuple:
    """"7x14x1" -> (7, 14, 1)."""
    return tuple(int(v) for v in option.split("x"))


def choosing_producer(n: Node, nodes: List[Node], options: Dict[int, List[str]]) -> Optional[Node]:
    """The node with several options that writes one of n's inputs, or None. Route: the router. The router: None (its inputs
    are graph inputs). A node reading the outputs of two such nodes is not handled (raises)."""
    found = {p.graph_idx: p for p in (producer_of(t, nodes) for t in n.inputs) if p is not None and len(options[p.graph_idx]) > 1}
    if len(found) > 1:
        raise NotImplementedError(f"node {n.graph_idx} ({n.name}) reads {len(found)} nodes that have several options")
    return next(iter(found.values()), None)


def producer_grids(n: Node, nodes: List[Node], override: Optional[Tuple[int, tuple]] = None) -> List[Optional[tuple]]:
    """Per input of n, the grid of the node that writes it (None: a graph input); override = (graph_idx, grid) replaces one
    producer's grid. Route with override (26, (7, 14, 1)): [(7, 14, 1), None]."""
    grids = []
    for t in n.inputs:
        p = producer_of(t, nodes)
        if p is None:
            grids.append(None)
        elif override and p.graph_idx == override[0]:
            grids.append(override[1])
        else:
            grids.append(p.grid)
    return grids


def tasks_of(pk, n: Node, grid: tuple, gpu: int, num_gpus: int):
    """The node's tasks on one GPU under `grid`: (position, relative position, input slices, output slices) per task.
    relative position = the position with this GPU's first owned row block as 0 (only differs for rows_split_over_gpus nodes):
    latent_down (28, 8, 1) on GPU 1 owns blocks 3..6: task (4, 2, 0) has relative position (1, 2, 0)."""
    lo, hi = owned_positions(n, grid[0], gpu, num_gpus)
    tensors = n.inputs + n.outputs
    for pos in itertools.product(*(range(s) for s in grid)):
        if lo <= pos[0] < hi:
            slices = [(tname(pk, t), box(t, m, grid, pos)) for t, m in zip(tensors, n.info["input_maps"])]
            yield pos, (pos[0] - lo,) + tuple(pos[1:]), slices[:len(n.inputs)], slices[len(n.inputs):]


def search_plan(pk, cost, time_limit_s: float = 60.0, workers: int = 16, given: Optional[dict] = None,
                start_from: Optional[dict] = None) -> Tuple[dict, dict]:
    """Returns (plan, summary): plan = {"grids", "lists"} for compiler.compile_plan; summary = the solver's status, predicted
    time, lower bound, solve time, chosen grids. given: a plan (e.g. the K3 demo's hand plan) whose grids and lists are fixed, so
    the solver only computes its time with the same durations (to compare with the searched plan). start_from: a plan the
    free search starts from (its times computed as for `given`, then handed to the solver as a hint): the search returns it or
    a plan the model predicts faster."""
    num_gpus, num_sms = pk.num_gpus, pk._num_sms()             # StaticMegakernel.num_gpus, num_sms
    nodes = graph_nodes(pk)
    by_idx = {n.graph_idx: n for n in nodes}                    # graph_idx -> Node, e.g. by_idx[29] = the topk_route node
    # a pool: a one_task_per_sm node whose work the SMs share at run time (cost.pool_us gives its total; MoE: moe_experts). It is
    # not placed task by task; every other node's tasks are placed
    is_pool = lambda n: n.info["one_task_per_sm"] and cost.pool_us(n) is not None

    # step 1: every node's options (its possible grids, as names)
    options = {n.graph_idx: [option_id(g) for g in candidate_grids(n)] for n in nodes}
    all_options = {k: list(v) for k, v in options.items()}     # before the options without measured times are left out

    # step 2: durations. own[(node, option)] = us for a node whose time depends only on its own grid; by_producer[node] =
    # {own option: {(producer, producer's option): us}} for a node whose time also depends on the option of the node writing
    # its input (situ_and_mul: its own features per task and shared gate_up's K parts).
    own: Dict[Tuple[int, str], float] = {}
    first_extra: Dict[Tuple[int, str], float] = {}              # (node, option) -> its first-on-SM time minus own (solver rule 6)
    by_producer: Dict[int, Dict[str, Dict[Tuple[int, str], float]]] = {}
    for n in nodes:
        if is_pool(n):
            continue                                            # not placed: a pool, e.g. moe_experts (see step 4)
        p = choosing_producer(n, nodes, options)
        if p is None:
            for opt in options[n.graph_idx]:
                us = cost.task_us(n, as_grid(opt), producer_grids(n, nodes))
                if us is not None:
                    own[(n.graph_idx, opt)] = us
                    first = cost.first_us(n, as_grid(opt), producer_grids(n, nodes)) if hasattr(cost, "first_us") else None
                    if first is not None and first > us:
                        first_extra[(n.graph_idx, opt)] = first - us
        else:
            by_producer[n.graph_idx] = {}
            for opt in options[n.graph_idx]:
                for popt in options[p.graph_idx]:
                    us = cost.task_us(n, as_grid(opt), producer_grids(n, nodes, (p.graph_idx, as_grid(popt))))
                    if us is not None:
                        by_producer[n.graph_idx].setdefault(opt, {})[(p.graph_idx, popt)] = us
    # keep an option only if it has a time and every node reading its output has a time for it
    for n in nodes:
        if is_pool(n):
            continue
        readers = [c for c, per in by_producer.items() if any(k[0] == n.graph_idx for d in per.values() for k in d)]
        ok = [opt for opt in options[n.graph_idx]
              if (opt in by_producer.get(n.graph_idx, {}) or (n.graph_idx, opt) in own)
              and all(any((n.graph_idx, opt) in d for d in by_producer[c].values()) for c in readers)]
        if not ok:
            raise ValueError(f"node {n.graph_idx} ({n.name}): no option has measured durations")
        if len(ok) < len(options[n.graph_idx]):
            print(f"search: node {n.graph_idx} ({n.name}): options not measured, left out: "
                  f"{sorted(set(options[n.graph_idx]) - set(ok))}", flush=True)
        options[n.graph_idx] = ok
    groups = {str(n.graph_idx): [{n.graph_idx: opt} for opt in options[n.graph_idx]] for n in nodes}   # one entry per node

    # step 3: GPU sets. key(gpu) = every (node, option, relative position) of the GPU's tasks; GPUs with the same key are one
    # GPU set, named by its GPUs.
    def key(gpu):
        return tuple(sorted((n.graph_idx, opt, rel) for n in nodes for opt in options[n.graph_idx]
                            for _, rel, _, _ in tasks_of(pk, n, as_grid(opt), gpu, num_gpus)))
    gpu_sets: Dict[tuple, List[int]] = {}
    for gpu in range(num_gpus):
        gpu_sets.setdefault(key(gpu), []).append(gpu)
    gpu_sets = {"gpus_" + "_".join(map(str, gpus)): gpus for gpus in gpu_sets.values()}     # e.g. "gpus_0_2_4_6": [0, 2, 4, 6]

    # step 4: per GPU set, the tasks of every option of every node (on the set's first GPU), their dependencies, the pool (the
    # tasks it waits for, and the tasks that run after it), the concurrent groups
    tasks, deps, pools, where, together = [], [], [], {}, []   # where[task id] = (node, option, relative position)
    for gpu_set, gpus in gpu_sets.items():
        writers, readers = {}, []                       # tensor name -> [(slice, task id, node)]; [(task id, node, its reads)]
        for n in nodes:
            for opt in options[n.graph_idx]:
                for pos, rel, reads, writes in tasks_of(pk, n, as_grid(opt), gpus[0], num_gpus):
                    tid = f"{gpu_set}:{n.graph_idx}:{opt}:{'.'.join(map(str, rel))}"   # e.g. "gpus_0_2_4_6:26:7x14x1:2.5.0"
                    where[tid] = (n.graph_idx, opt, rel)
                    for name, b in writes:
                        writers.setdefault(name, []).append((b, tid, n.graph_idx))
                    readers.append((tid, n.graph_idx, reads))
                    if is_pool(n):
                        continue
                    if n.graph_idx in by_producer:      # topk_route: one duration per router option that is still kept
                        d = {k: us for k, us in by_producer[n.graph_idx][opt].items() if k[1] in options[k[0]]}
                        tasks.append(SolverTask(tid, n.graph_idx, opt, gpu_set, max(d.values()), duration_by=d))
                    else:
                        tasks.append(SolverTask(tid, n.graph_idx, opt, gpu_set, own[(n.graph_idx, opt)],
                                                first_extra_us=first_extra.get((n.graph_idx, opt), 0.0)))
        placed = {t.id for t in tasks if t.gpu_set == gpu_set}
        pool_inputs, pool_readers = {}, {}              # pool node -> the placed tasks it waits for / that read its output
        set_deps = []
        for tid, node, reads in readers:
            # the tasks of other nodes (any of their options) that write a slice this task reads; the solver applies each
            # dependency only when both tasks' options are used
            srcs = {w for name, b in reads for wb, w, wn in writers.get(name, []) if wn != node and overlaps(b, wb) and w in placed}
            if tid in placed:
                set_deps += [(s, tid) for s in srcs]
                for name, b in reads:                   # reads the output of a pool (allreduce_send reads moe_experts' R)
                    for wb, w, wn in writers.get(name, []):
                        if wn != node and is_pool(by_idx[wn]) and overlaps(b, wb):
                            pool_readers.setdefault(wn, set()).add(tid)
            elif is_pool(by_idx[node]):
                pool_inputs.setdefault(node, set()).update(srcs)
        deps += set_deps
        succ = {}
        for a, b in set_deps:
            succ.setdefault(a, []).append(b)
        for n in nodes:
            if is_pool(n):
                after, stack = set(), list(pool_readers.get(n.graph_idx, ()))   # the pool's readers and everything after them
                while stack:
                    t = stack.pop()
                    if t not in after:
                        after.add(t)
                        stack += succ.get(t, [])
                pools.append(Pool(f"{n.graph_idx}@{gpu_set}", gpu_set, cost.pool_us(n), sorted(pool_inputs.get(n.graph_idx, ())),
                                  sorted(after)))
        # concurrent groups: the tasks of one option of a node with concurrent_along_axis that differ only in that axis
        for n in nodes:
            a = n.info["concurrent_along_axis"]
            if a is None or is_pool(n):
                continue
            for opt in options[n.graph_idx]:
                by_key = {}
                for pos, rel, _, _ in tasks_of(pk, n, as_grid(opt), gpus[0], num_gpus):
                    by_key.setdefault(tuple(v for i, v in enumerate(rel) if i != a), []).append(
                        f"{gpu_set}:{n.graph_idx}:{opt}:{'.'.join(map(str, rel))}")
                together += list(by_key.values())

    # a given plan: fix every node's option (its grid in the plan, or its own grid) and every task's SM and order (its
    # positions moved to the GPU set's first GPU's relative positions)
    fixed_options = fixed_lists = fixed_post = None
    given = given or start_from                         # start_from: fixed first (its times), then the free search from it
    if given:
        fixed_post = set()                              # the tasks listed after the pool entry on their SM
        fixed_options = {n.graph_idx: option_id(tuple(given["grids"].get(n.graph_idx, n.grid))) for n in nodes}
        fixed_lists = {}
        for gpu_set, gpus in gpu_sets.items():
            g0 = gpus[0]
            fixed_lists[gpu_set] = []
            for sm_list in given["lists"][g0]:
                ids, past_pool = [], False
                for node, pos in sm_list:
                    n = by_idx[node]
                    if is_pool(n):
                        past_pool = True
                        continue                        # the pool is not a placed task
                    lo = owned_positions(n, as_grid(fixed_options[node])[0], g0, num_gpus)[0]
                    rel = (pos[0] - lo,) + tuple(pos[1:])
                    ids.append(f"{gpu_set}:{node}:{fixed_options[node]}:{'.'.join(map(str, rel))}")
                    if past_pool:
                        fixed_post.add(ids[-1])
                fixed_lists[gpu_set].append(ids)

    # step 5. An SM's list holds at most MAX_TASKS_PER_SM - 1 tasks (then the end entry), one of them each pool's task
    max_placed_per_sm = static_schedule.MAX_TASKS_PER_SM - 1 - sum(is_pool(n) for n in nodes)
    print(f"search: {len(nodes)} nodes, options per node {{{', '.join(f'{k}: {len(v)}' for k, v in options.items())}}}, "
          f"gpu_sets {list(gpu_sets.values())}, {len(tasks)} tasks over all options ({num_sms} SMs each)", flush=True)
    result = solve(groups, tasks, deps, pools, list(gpu_sets), num_sms, max_placed_per_sm,
                   min(time_limit_s, 300) if start_from else time_limit_s, workers,
                   fixed_options=fixed_options, fixed_lists=fixed_lists, concurrent=together, fixed_post=fixed_post)
    if start_from and result.status in ("OPTIMAL", "FEASIBLE"):
        print(f"search: start plan predicted {result.predicted_us} us; free search from it", flush=True)
        free = solve(groups, tasks, deps, pools, list(gpu_sets), num_sms, max_placed_per_sm, time_limit_s, workers,
                     concurrent=together, hint=result)
        if free.status in ("OPTIMAL", "FEASIBLE") and free.predicted_us <= result.predicted_us:
            result = free
        else:
            print(f"search: the free search found no faster plan ({free.status}); the start plan", flush=True)
    if result.status not in ("OPTIMAL", "FEASIBLE"):
        raise RuntimeError(f"solver: {result.status}")

    # step 6: the plan. grids: each node with changeable grid dims: its chosen option as a grid. lists: per GPU, per SM, the
    # solver's task ids of the GPU's set turned back into (node, position on this GPU): relative position + this GPU's first
    # owned row block; the SM's pool task (task x = SM x) goes between the tasks before the pool and the tasks after it (the
    # solver's post: the tasks reading its output, and the tasks it chose to run after it).
    grids = {n.graph_idx: as_grid(result.chosen[n.graph_idx]) for n in nodes if n.info["changeable_grid_dims"]}
    pool_nodes = [n.graph_idx for n in nodes if is_pool(n)]
    after_pool = result.post
    lists = [None] * num_gpus
    for gpu_set, gpus in gpu_sets.items():
        for gpu in gpus:
            per_sm = []
            for sm, sm_list in enumerate(result.lists[gpu_set]):
                before, after = [], []
                for tid in sm_list:
                    node, opt, rel = where[tid]
                    n = by_idx[node]
                    lo = owned_positions(n, as_grid(opt)[0], gpu, num_gpus)[0]
                    (after if tid in after_pool else before).append((node, (rel[0] + lo,) + tuple(rel[1:])))
                per_sm.append(before + [(k, (sm, 0, 0)) for k in pool_nodes] + after)
            lists[gpu] = per_sm
    summary = {"status": result.status, "predicted_us": result.predicted_us, "bound_us": result.bound_us,
               "solve_s": result.solve_s, "grids": {f"{by_idx[k].name}#{k}": list(v) for k, v in grids.items()}}
    print(f"search: {summary}", flush=True)
    summary["candidates"], summary["timeline"] = prediction(nodes, all_options, options, own, by_producer, is_pool, result, where,
                                                            gpu_sets, pools)
    return {"grids": grids, "lists": lists}, summary


def prediction(nodes, all_options, options, own, by_producer, is_pool, result, where, gpu_sets, pools) -> Tuple[list, dict]:
    """The solver's prediction, for a viewer (plan_viewer.py): candidates = per node its options (all, those with measured times,
    the chosen one) and the time of one task per option (a node timed per producer option: {option: {"<producer>#<idx>
    <producer option>": us}}); timeline = per GPU set, per SM the placed tasks' predicted start and end (us), and each pool's
    open time, every SM's join and end."""
    by_idx = {n.graph_idx: n for n in nodes}
    candidates = []
    for n in nodes:
        k, placed = n.graph_idx, not is_pool(n)
        if k in by_producer:
            task_us = {opt: {f"{by_idx[p].name}#{p} {popt}": us for (p, popt), us in d.items()} for opt, d in by_producer[k].items()}
        else:
            task_us = {opt: us for (node, opt), us in own.items() if node == k}
        candidates.append({"node": k, "name": n.name, "params": list(n.params), "all": all_options[k], "measured": options[k],
                           "chosen": result.chosen.get(k), "placed": placed, "task_us": task_us if placed else {},
                           "tasks_per_gpu_set": {g: sum(where[t][0] == k for sm in per_sm for t in sm)
                                                 for g, per_sm in result.lists.items()}})
    timeline = {}
    for gpu_set, gpus in gpu_sets.items():
        sms = [[{"node": where[t][0], "name": by_idx[where[t][0]].name, "option": where[t][1], "pos": list(where[t][2]),
                 "start": result.times[t][0], "end": result.times[t][1]} for t in sm] for sm in result.lists[gpu_set]]
        timeline[gpu_set] = {"gpus": gpus, "sms": sms,
                             "pools": [dict(result.pools[p.id], id=p.id, work=p.work_us, after=0.0) for p in pools
                                       if p.gpu_set == gpu_set]}
    return candidates, timeline


# ---------------- durations from cost records ----------------
def cost_key(node: Node, grid: tuple, producer_grids: Optional[list] = None) -> dict:
    """The key of a task's cost record: what decides its time, read from the node as its layer declared it: its params, its tensors'
    shapes, its grid, and the grids of the nodes that write the inputs its layer lists in cost_inputs (the inputs whose split
    changes its time; none: no "inputs" field; producer_grids None: none either, as for a pool). The record's task type is the
    node's name. topk_route after the router in 14 K parts: {"params": [896, 16], "shapes": [[8, 896], [896], [8, 16]],
    "grid": [8, 1, 1], "inputs": {"0": [7, 14, 1]}}."""
    key = {"params": list(node.params), "shapes": [list(dims(t)) for t in node.inputs + node.outputs], "grid": list(grid)}
    which = node.info["cost_inputs"]
    if producer_grids is not None and which:
        key["inputs"] = {str(i): list(producer_grids[i]) if producer_grids[i] is not None else None for i in which}
    return key


class Costs:
    """The durations search_plan needs, from cost record files (the median of the records of a task type and key, cost_key), of
    one kernel version (code: static_schedule.code_tag; None: any) on one GPU type (gpu: the device name, e.g. "NVIDIA B300 SXM6
    PC"; None: any). A record (cost_records.py): {"task_type", "key", "context": {"first_on_sm"}, "median_us", "machine": {"gpu"},
    "code", ...}. A key with no record: None (search_plan leaves that choice out)."""

    def __init__(self, paths, code: Optional[str] = None, gpu: Optional[str] = None):
        self.records = []
        for path in ([paths] if isinstance(paths, str) else list(paths or [])):
            if path and os.path.exists(path):
                with open(path) as f:
                    self.records += [r for r in json.load(f)["records"] if (code is None or r.get("code") == code)
                                     and (gpu is None or (r.get("machine") or {}).get("gpu") == gpu)]

    def median(self, task_type: str, key: dict, first: Optional[bool] = None) -> Optional[float]:
        """The median of the records of this task type and key; first True / False: only the records of tasks first / not first
        on their SM (context first_on_sm; records without it count as neither)."""
        values = [r["median_us"] for r in self.records if r["task_type"] == task_type and r["key"] == key
                  and (first is None or (r.get("context") or {}).get("first_on_sm") is first)]
        return statistics.median(values) if values else None

    def task_us(self, node: Node, grid: tuple, producer_grids: list) -> Optional[float]:
        """A task's time behind another task on its SM (first_us: first on it), else of all its records. A node that may run its
        concurrent pairs on a cluster (cluster_pair_params) and has no record with its own params: the cluster variant's (the
        launch compile_plan picks when the pairs fit). A median below 0 (the task ended before a task the graph says it waits
        for, as when its input maps cover more than it reads): 0."""
        key = cost_key(node, grid, producer_grids)
        us = self.median(node.name, key, first=False)
        if us is None:
            us = self.median(node.name, key)
        if us is None and can_pair(node, grid):
            us = self.median(node.name, dict(key, params=list(node.info["cluster_pair_params"])))
        return None if us is None else max(us, 0.0)

    def first_us(self, node: Node, grid: tuple, producer_grids: list) -> Optional[float]:
        """A task's time when it is the first on its SM (nothing before it hides its first loads); None: no such record."""
        return self.median(node.name, cost_key(node, grid, producer_grids), first=True)

    def pool_us(self, node: Node) -> Optional[float]:
        """A one-task-per-SM node's total work (its records' key: cost_key without the producers' grids); None for other nodes."""
        return self.median(node.name, cost_key(node, node.grid)) if node.info["one_task_per_sm"] else None

