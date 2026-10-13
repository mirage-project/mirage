"""A starting plan without cost records: compiler.compile_plan's {"grids", "lists"} for any graph, so a layer can be built and
timed (cost_records.py) before search.search_plan has measured durations to work from.

round_robin_plan: every GPU gets the same placement. The nodes are taken in graph order and each node's tasks are dealt to the
SMs in turn, continuing where the previous node stopped; a one_task_per_sm node puts task x on SM x. Every SM's list is then in
graph order, and a task only waits for tasks of earlier nodes, which sit earlier in every list: the lists cannot deadlock
(static_schedule.check_deadlock_free still checks). A node with concurrent_along_axis a has its positions ordered with axis a
fastest, so the tasks of a group land on consecutive (different) SMs.
"""
import itertools
from typing import Dict, Optional

from .compiler import candidate_grids, graph_nodes, owned_positions


def _positions(grid: tuple, fast_axis: Optional[int]):
    """The grid's positions, axis `fast_axis` varying fastest (None: x slowest, z fastest)."""
    order = [0, 1, 2]
    if fast_axis is not None:
        order.remove(fast_axis)
        order.append(fast_axis)
    for p in itertools.product(*(range(grid[a]) for a in order)):
        pos = [0, 0, 0]
        for a, v in zip(order, p):
            pos[a] = v
        yield tuple(pos)


def round_robin_plan(pk, grids: Optional[Dict[int, tuple]] = None) -> dict:
    """grids: {graph_idx: grid} for the nodes whose grid the plan changes (each must be one of compiler.candidate_grids);
    the other nodes keep their grid. Returns {"grids", "lists"} for compiler.compile_plan."""
    grids = {int(k): tuple(v) for k, v in (grids or {}).items()}
    nodes = graph_nodes(pk)
    for n in nodes:
        if n.graph_idx in grids and grids[n.graph_idx] not in candidate_grids(n):
            raise ValueError(f"node {n.graph_idx} ({n.name}): grid {grids[n.graph_idx]} is not one of {candidate_grids(n)}")
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
            for pos in _positions(grid, n.info["concurrent_along_axis"]):
                if lo <= pos[0] < hi:
                    per_sm[cursor % num_sms].append((n.graph_idx, pos))
                    cursor += 1
        lists.append(per_sm)
    return {"grids": grids, "lists": lists}


def node_index(pk, name: str, nth: int = 0) -> int:
    """The graph index of the nth node (0 = first) with task type `name`, in graph order."""
    found = [n.graph_idx for n in graph_nodes(pk) if n.name == name]
    if nth >= len(found):
        raise ValueError(f"the graph has {len(found)} {name} nodes, asked for #{nth}")
    return found[nth]
