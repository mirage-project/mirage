"""Plans without cost records: compiler.compile_plan's {"grids", "lists"} for any graph, so a layer can be built and timed
(cost_records.py) before search.search_plan has measured durations to work from, and the plans a cost collection measures.

round_robin_plan: every GPU gets the same placement. The nodes are taken in graph order and each node's tasks are dealt to the
SMs in turn, continuing where the previous node stopped; a one_task_per_sm node puts task x on SM x. Every SM's list is then in
graph order, and a task only waits for tasks of earlier nodes, which sit earlier in every list: the lists cannot deadlock
(static_schedule.check_deadlock_free still checks). The tasks of a concurrent group (concurrent_along_axis: they wait for each
other while they run) are dealt one after the other, so they land on different SMs.

random_plan: the same lists in graph order, but each node's tasks in a random order over a random order of the SMs, so that a
task is measured in many placements (its prefetch window, cost_records.py). A concurrent group takes consecutive SMs of that
order (different SMs), and the groups of a node go into the lists in one order on every SM, so no two groups wait for each other
across SMs (check_deadlock_free counts a group as one task).

collection_plans: the plans a cost collection measures (the demo's --collect): round robin with the declared grids, round robin
with every candidate grid of every node (one node changed at a time: samples of tasks queued behind tasks of their own node), and
balanced random plans whose grids are drawn so that every candidate grid of every node is in at least `rounds` of them (samples
with every prefetch window).
"""
import itertools
import random
from typing import Dict, List, Optional, Tuple

from .compiler import candidate_grids, graph_nodes, owned_positions


def _groups(grid: tuple, axis: Optional[int]) -> List[List[tuple]]:
    """The grid's positions as concurrent groups: with `axis` a, the positions that differ only in axis a form one group (in
    axis order); without, every position is a group of its own. Groups in position order (x slowest)."""
    others = [i for i in range(3) if i != axis]
    groups = []
    for rest in itertools.product(*(range(grid[i]) for i in others)):
        group = []
        for v in (range(grid[axis]) if axis is not None else [None]):
            pos = [0, 0, 0]
            for i, r in zip(others, rest):
                pos[i] = r
            if axis is not None:
                pos[axis] = v
            group.append(tuple(pos))
        groups.append(group)
    return groups


def _check_grids(nodes, grids: Dict[int, tuple]) -> None:
    for n in nodes:
        if n.graph_idx in grids and grids[n.graph_idx] not in candidate_grids(n):
            raise ValueError(f"node {n.graph_idx} ({n.name}): grid {grids[n.graph_idx]} is not one of {candidate_grids(n)}")


def round_robin_plan(pk, grids: Optional[Dict[int, tuple]] = None) -> dict:
    """grids: {graph_idx: grid} for the nodes whose grid the plan changes (each must be one of compiler.candidate_grids);
    the other nodes keep their grid. Returns {"grids", "lists"} for compiler.compile_plan."""
    grids = {int(k): tuple(v) for k, v in (grids or {}).items()}
    nodes = graph_nodes(pk)
    _check_grids(nodes, grids)
    num_gpus, num_sms = pk.num_gpus, pk._num_sms()
    lists = []
    for gpu in range(num_gpus):
        per_sm = [[] for _ in range(num_sms)]
        cursor = 0
        for n in nodes:
            grid = grids.get(n.graph_idx, n.grid)
            if n.info["one_task_per_sm"]:
                for sm in range(num_sms):
                    per_sm[sm].append((n.graph_idx, (sm, 0, 0)))
                continue
            lo, hi = owned_positions(n, grid[0], gpu, num_gpus)
            for group in _groups(grid, n.info["concurrent_along_axis"]):
                for pos in group:
                    if lo <= pos[0] < hi:
                        per_sm[cursor % num_sms].append((n.graph_idx, pos))
                        cursor += 1
        lists.append(per_sm)
    return {"grids": grids, "lists": lists}


def random_plan(pk, grids: Optional[Dict[int, tuple]], rng: random.Random) -> dict:
    """A random balanced placement: every SM's list in graph order, as round_robin_plan (no deadlock), each node's groups in a
    random order dealt over the node's own random order of the SMs (an SM holds a second task of a node only once every SM
    holds one, so the tasks of a node do not queue behind each other); a one_task_per_sm node: task x on SM x. The same lists on
    every GPU's own positions."""
    grids = {int(k): tuple(v) for k, v in (grids or {}).items()}
    nodes = graph_nodes(pk)
    _check_grids(nodes, grids)
    num_gpus, num_sms = pk.num_gpus, pk._num_sms()
    order: Dict[int, List[Tuple[tuple, int]]] = {}   # graph_idx -> (position, SM) in list order
    for n in nodes:
        grid = grids.get(n.graph_idx, n.grid)
        if n.info["one_task_per_sm"]:
            order[n.graph_idx] = [((sm, 0, 0), sm) for sm in range(num_sms)]
            continue
        groups = _groups(grid, n.info["concurrent_along_axis"])
        if max(len(g) for g in groups) > num_sms:
            raise ValueError(f"node {n.graph_idx} ({n.name}): a concurrent group of {max(len(g) for g in groups)} tasks, "
                             f"the GPU has {num_sms} SMs")
        rng.shuffle(groups)
        sms = list(range(num_sms))
        rng.shuffle(sms)
        placed, cursor = [], 0
        for group in groups:
            placed += [(pos, sms[(cursor + j) % num_sms]) for j, pos in enumerate(group)]
            cursor += len(group)
        order[n.graph_idx] = placed
    lists = []
    for gpu in range(num_gpus):
        per_sm = [[] for _ in range(num_sms)]
        for n in nodes:
            grid = grids.get(n.graph_idx, n.grid)
            lo, hi = owned_positions(n, grid[0], gpu, num_gpus)
            for pos, sm in order[n.graph_idx]:
                if lo <= pos[0] < hi:
                    per_sm[sm].append((n.graph_idx, pos))
        lists.append(per_sm)
    return {"grids": grids, "lists": lists}


def collection_plans(pk, rounds: int = 3, seed: int = 0) -> List[Tuple[str, dict]]:
    """[(name, plan)] of a cost collection (the module docstring): "rr" (the declared grids), "rr_<node>_<grid>" for every
    candidate grid of every node with several, and rounds x (the most candidates a node has) plans "rand_<k>" (random_plan) with
    each node's grids drawn in rounds of a random order of its candidates."""
    nodes = graph_nodes(pk)
    cands = {n.graph_idx: candidate_grids(n) for n in nodes if len(candidate_grids(n)) > 1}
    plans = [("rr", round_robin_plan(pk))]
    for n in nodes:
        for g in cands.get(n.graph_idx, []):
            if g != n.grid:
                plans.append((f"rr_{n.graph_idx}_{'x'.join(map(str, g))}", round_robin_plan(pk, {n.graph_idx: g})))
    rng = random.Random(seed)
    count = rounds * max((len(gs) for gs in cands.values()), default=0)
    draws: Dict[int, list] = {idx: [] for idx in cands}
    for idx, gs in cands.items():
        while len(draws[idx]) < count:
            cycle = list(gs)
            rng.shuffle(cycle)
            draws[idx] += cycle
    for k in range(count):
        plans.append((f"rand_{k:02d}", random_plan(pk, {idx: draws[idx][k] for idx in cands}, rng)))
    return plans
