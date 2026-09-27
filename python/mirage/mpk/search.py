"""Build the solver's input (solver.py) from the graph and the cost file, run the solver, and turn its answer into the plan that
compiler.compile_plan takes ({"grids", "lists"}). Nothing here is specific to a model; `cost` supplies the durations.

search_plan does, with the MoE layer's numbers:
  1. options    every node's possible grids (compiler.candidate_grids): a node with changeable grid dims (in the MoE layer the 3
                gemm_tile nodes) has several, router (7, k, 1) for k = 1, 2, 4, 7, 8, 14, 28, 56; every other node one, its
                own grid (route (8, 1, 1)). Each node decides alone: route does not choose with the router; its tasks wait for
                the router's tasks whatever the router's grid (the dependencies come from the slices).
  2. durations  every task's measured time from `cost`. A node whose input is written by a node with several options (route
                reads the router's output) has one time per option of that node: route adds as many partial sums as the router
                has K parts, so its time for router 7x14x1 is the time measured for adding 14. An option with a missing time
                (its own, or one of a node reading it) is left out (1, 2, 28, 56 K parts in the current cost file).
  3. GPU sets   GPUs whose tasks are the same once each GPU's own part of a split node is shifted to 0: GPUs 0, 2, 4, 6 compute 3
                latent_down row blocks, GPUs 1, 3, 5, 7 compute 4 -> "gpus_0_2_4_6" and "gpus_1_3_5_7". The solver places the
                tasks of the first GPU of each set; the other GPUs of the set get the same placement.
  4. tasks      per GPU set, every task of every option (router 7x14x1: 98 tasks, 7x4x1: 28, ...) with its duration(s); the
                dependencies (a task that reads a slice some other node's task writes waits for it: compiler.box / overlaps);
                the one-per-SM nodes are not placed: the expert queue becomes one Pool per GPU set (its measured total work),
                and the tail's measured time is added after it (Pool.after_us)
  5. solve      solver.solve: one option per node, an SM and start time per task, the layer's end as early as possible
  6. plan       "grids": the chosen grid of each node with changeable grid dims; "lists": for every GPU, the solver's per-SM
                lists of its GPU set, each task moved from the set's first GPU's positions to this GPU's own positions
cost: task_us(node, grid, producer_grids) -> us or None (not measured), pool_us(node) -> the queue's total work (us, summed over
the SMs) or None if the node is not a queue, after_us(node) -> us added after the queue (the tail). MoE: moe.MoeCosts.
"""
import itertools
from typing import Dict, List, Optional, Tuple

from .compiler import (Node, box, candidate_grids, graph_nodes, overlaps, owned_row_blocks, producer_of, tname)
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
    producer's grid. Route with override (30, (7, 14, 1)): [(7, 14, 1), None, None]."""
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
    lo, hi = owned_row_blocks(grid[0], gpu, num_gpus) if n.info["rows_split_over_gpus"] else (0, grid[0])
    tensors = n.inputs + n.outputs
    for pos in itertools.product(*(range(s) for s in grid)):
        if lo <= pos[0] < hi:
            slices = [(tname(pk, t), box(t, m, grid, pos)) for t, m in zip(tensors, n.info["maps"])]
            yield pos, (pos[0] - lo,) + tuple(pos[1:]), slices[:len(n.inputs)], slices[len(n.inputs):]


def search_plan(pk, cost, num_gpus: int, num_sms: int, max_placed_per_sm: int, time_limit_s: float = 60.0,
                workers: int = 16, given: Optional[dict] = None) -> Tuple[dict, dict]:
    """Returns (plan, summary): plan = {"grids", "lists"} for compiler.compile_plan; summary = the solver's status, predicted
    time, lower bound, solve time, chosen grids. given: a plan (e.g. moe.hand_plan) whose grids and lists are fixed, so
    the solver only computes its time with the same durations (to compare with the searched plan)."""
    nodes = graph_nodes(pk)
    by_idx = {n.graph_idx: n for n in nodes}                    # graph_idx -> Node, e.g. by_idx[31] = the route node

    # step 1: every node's options (its possible grids, as names)
    options = {n.graph_idx: [option_id(g) for g in candidate_grids(n)] for n in nodes}

    # step 2: durations. own[(node, option)] = us for a node whose time depends only on its own grid; by_producer[node] =
    # {(producer, producer's option): us} for a node whose time depends on the option of the node writing its input.
    own: Dict[Tuple[int, str], float] = {}
    by_producer: Dict[int, Dict[Tuple[int, str], float]] = {}
    for n in nodes:
        if n.info["one_task_per_sm_at_end"]:
            continue                                            # not placed: the expert queue and the tail (see step 4)
        p = choosing_producer(n, nodes, options)
        if p is None:
            for opt in options[n.graph_idx]:
                us = cost.task_us(n, as_grid(opt), producer_grids(n, nodes))
                if us is not None:
                    own[(n.graph_idx, opt)] = us
        else:
            by_producer[n.graph_idx] = {}
            for popt in options[p.graph_idx]:
                us = cost.task_us(n, n.grid, producer_grids(n, nodes, (p.graph_idx, as_grid(popt))))
                if us is not None:
                    by_producer[n.graph_idx][(p.graph_idx, popt)] = us
    # keep an option only if it has a time and every node reading its output has a time for it
    for n in nodes:
        if n.info["one_task_per_sm_at_end"]:
            continue
        readers = [c for c, d in by_producer.items() if any(k[0] == n.graph_idx for k in d)]
        ok = [opt for opt in options[n.graph_idx]
              if (n.graph_idx in by_producer or (n.graph_idx, opt) in own)
              and all((n.graph_idx, opt) in by_producer[c] for c in readers)]
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

    # step 4: per GPU set, the tasks of every option of every node (on the set's first GPU), their dependencies, the queue
    tasks, deps, pools, where = [], [], [], {}          # where[task id] = (node, option, relative position), to read the answer
    for gpu_set, gpus in gpu_sets.items():
        writers, readers = {}, []                       # tensor name -> [(slice, task id, node)]; [(task id, node, its reads)]
        for n in nodes:
            for opt in options[n.graph_idx]:
                for pos, rel, reads, writes in tasks_of(pk, n, as_grid(opt), gpus[0], num_gpus):
                    tid = f"{gpu_set}:{n.graph_idx}:{opt}:{'.'.join(map(str, rel))}"   # e.g. "gpus_0_2_4_6:30:7x14x1:2.5.0"
                    where[tid] = (n.graph_idx, opt, rel)
                    for name, b in writes:
                        writers.setdefault(name, []).append((b, tid, n.graph_idx))
                    readers.append((tid, n.graph_idx, reads))
                    if n.info["one_task_per_sm_at_end"]:
                        continue
                    if n.graph_idx in by_producer:      # route: one duration per router option that is still kept
                        d = {k: us for k, us in by_producer[n.graph_idx].items() if k[1] in options[k[0]]}
                        tasks.append(SolverTask(tid, n.graph_idx, opt, gpu_set, max(d.values()), duration_by=d))
                    else:
                        tasks.append(SolverTask(tid, n.graph_idx, opt, gpu_set, own[(n.graph_idx, opt)]))
        placed = {t.id for t in tasks}
        pool_inputs = {}                                # one-per-SM node -> the placed tasks it waits for
        for tid, node, reads in readers:
            # the tasks of other nodes (any of their options) that write a slice this task reads; the solver applies each
            # dependency only when both tasks' options are used
            srcs = {w for name, b in reads for wb, w, wn in writers.get(name, []) if wn != node and overlaps(b, wb) and w in placed}
            if tid in placed:
                deps += [(s, tid) for s in srcs]
            elif by_idx[node].info["one_task_per_sm_at_end"]:
                pool_inputs.setdefault(node, set()).update(srcs)
        # the expert queue (cost.pool_us gives a work) becomes a Pool: it waits for its inputs (route, quant, sact tasks); after
        # it, the time of the one-per-SM nodes that are not queues (the tail: cost.after_us)
        for n in nodes:
            if n.info["one_task_per_sm_at_end"] and cost.pool_us(n) is not None:
                pools.append(Pool(f"{n.graph_idx}@{gpu_set}", gpu_set, cost.pool_us(n), sorted(pool_inputs.get(n.graph_idx, ())),
                                  sum(cost.after_us(m) for m in nodes
                                      if m.info["one_task_per_sm_at_end"] and cost.pool_us(m) is None)))

    # a given plan: fix every node's option (its grid in the plan, or its own grid) and every task's SM and order (its
    # positions moved to the GPU set's first GPU's relative positions)
    fixed_options = fixed_lists = None
    if given:
        fixed_options = {n.graph_idx: option_id(tuple(given["grids"].get(n.graph_idx, n.grid))) for n in nodes}
        fixed_lists = {}
        for gpu_set, gpus in gpu_sets.items():
            g0 = gpus[0]
            fixed_lists[gpu_set] = []
            for sm_list in given["lists"][g0]:
                ids = []
                for node, pos in sm_list:
                    n = by_idx[node]
                    lo = owned_row_blocks(as_grid(fixed_options[node])[0], g0, num_gpus)[0] if n.info["rows_split_over_gpus"] else 0
                    rel = (pos[0] - lo,) + tuple(pos[1:])
                    ids.append(f"{gpu_set}:{node}:{fixed_options[node]}:{'.'.join(map(str, rel))}")
                fixed_lists[gpu_set].append(ids)

    # step 5
    print(f"search: {len(nodes)} nodes, options per node {{{', '.join(f'{k}: {len(v)}' for k, v in options.items())}}}, "
          f"gpu_sets {list(gpu_sets.values())}, {len(tasks)} tasks over all options ({num_sms} SMs each)", flush=True)
    result = solve(groups, tasks, deps, pools, list(gpu_sets), num_sms, max_placed_per_sm, time_limit_s, workers,
                   fixed_options=fixed_options, fixed_lists=fixed_lists)
    if result.status not in ("OPTIMAL", "FEASIBLE"):
        raise RuntimeError(f"solver: {result.status}")

    # step 6: the plan. grids: each node with changeable grid dims: its chosen option as a grid. lists: per GPU, per SM, the
    # solver's task ids of the GPU's set turned back into (node, position on this GPU): relative position + this GPU's first
    # owned row block.
    grids = {n.graph_idx: as_grid(result.chosen[n.graph_idx]) for n in nodes if n.info["changeable_grid_dims"]}
    lists = [None] * num_gpus
    for gpu_set, gpus in gpu_sets.items():
        for gpu in gpus:
            per_sm = []
            for sm_list in result.lists[gpu_set]:
                entries = []
                for tid in sm_list:
                    node, opt, rel = where[tid]
                    n = by_idx[node]
                    lo = owned_row_blocks(as_grid(opt)[0], gpu, num_gpus)[0] if n.info["rows_split_over_gpus"] else 0
                    entries.append((node, (rel[0] + lo,) + tuple(rel[1:])))
                per_sm.append(entries)
            lists[gpu] = per_sm
    summary = {"status": result.status, "predicted_us": result.predicted_us, "bound_us": result.bound_us,
               "solve_s": result.solve_s, "grids": {f"{by_idx[k].name}#{k}": list(v) for k, v in grids.items()}}
    print(f"search: {summary}", flush=True)
    return {"grids": grids, "lists": lists}, summary
