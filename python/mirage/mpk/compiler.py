"""Plan -> one schedule file per GPU (schedule_gpu<g>.json), which static_schedule.py turns into layer.cu.

A plan:  {"grids": {graph_idx: grid},                     e.g. {26: (7, 14, 1), 28: (28, 8, 1), 30: (12, 8, 1)}
          "lists": per GPU, per SM, the ordered (graph_idx, grid position) of the SM's tasks}
Nodes not in "grids" keep the grid their layer method gave them. A one_task_per_sm node's task x must be in SM x's list.

compile_plan:
  1. apply_grids     each node in the plan: its grid must be one of candidate_grids; kn_graph.regrid
  2. per GPU:
       gpu_tasks        every grid position of every node (split-over-GPU nodes: only this GPU's part), with its slices (box)
       add_dependencies a task waits for the tasks of other nodes whose output slices overlap its input slices
       lists_of_gpu     the plan's lists as task ids (every task exactly once; one_task_per_sm task x on SM x; the tasks of a
                        concurrent group on different SMs)
       write_schedule   deadlock check; the file: the nodes (grid, params, the grids of their inputs' producers), the tasks
                        ({node, position, deps}), the per-SM lists
"""
import itertools
import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from . import static_schedule

Box = Tuple[Tuple[int, int], ...]        # per tensor dimension, [lo, hi)


@dataclass
class Node:
    graph_idx: int
    name: str                     # the registered task name, e.g. "gemm_tile"
    params: List[int]
    grid: Tuple[int, int, int]
    inputs: list                  # DTensors
    outputs: list
    info: dict                    # what the layer method declared (static_megakernel._add_node): input_maps, changeable_grid_dims,
                                  # rows_split_over_gpus, one_task_per_sm, concurrent_along_axis, ...


def graph_nodes(pk) -> List[Node]:
    """The graph's static-schedule nodes (the ones pk._static_nodes has), in graph order, with their grid and params now."""
    nodes = []
    for graph_idx, info in sorted(pk._static_nodes.items()):
        name, params, grid, num_inputs, num_outputs, tensors = pk.kn_graph.get_task_info(graph_idx)
        assert name == info["name"], f"graph node {graph_idx} is {name}, its record says {info['name']}"
        nodes.append(Node(graph_idx, name, list(params), tuple(grid), tensors[:num_inputs],
                          tensors[num_inputs:num_inputs + num_outputs], info))
    return nodes


def tname(pk, tensor) -> str:
    return pk._tensor_names[tensor.guid]


def dims(tensor) -> List[int]:
    return [tensor.dim(i) for i in range(tensor.num_dims)]


def producer_of(tensor, nodes: List[Node]) -> Optional[Node]:
    """The node with `tensor` among its outputs (None: a graph input)."""
    return next((n for n in nodes if any(t.guid == tensor.guid for t in n.outputs)), None)


def candidate_grids(node: Node) -> List[Tuple[int, int, int]]:
    """The grids the compiler may give the node: on each changeable grid dim a (unit u, optionally between min and max units per
    task), every size s such that every tensor dimension that axis cuts splits into s blocks of a whole number of units; the other
    axes as they are.
    GEMM, K = 7168, changeable grid dim y with unit 128: 7168 / 128 = 56 units -> y in 1, 2, 4, 7, 8, 14, 28, 56.
    latent_up, K = 3584 = 28 units of 128, 2 to 5 units per task -> y in 7 (4 units), 14 (2 units)."""
    tensors = node.inputs + node.outputs
    per_axis = []
    for axis in range(3):
        if axis not in node.info["changeable_grid_dims"]:
            per_axis.append([node.grid[axis]])
            continue
        spec = node.info["changeable_grid_dims"][axis]
        unit, lo, hi = (spec, 1, None) if isinstance(spec, int) else tuple(spec)
        cut = [dims(t)[m[axis]] for t, m in zip(tensors, node.info["input_maps"]) if m[axis] >= 0]
        if not cut:
            raise ValueError(f"node {node.graph_idx} ({node.name}): grid axis {axis} is changeable but no tensor map uses it")
        units = min(d // unit for d in cut)
        per_axis.append([s for s in range(1, units + 1) if all(d % (s * unit) == 0 for d in cut)
                         and all(lo <= d // (s * unit) and (hi is None or d // (s * unit) <= hi) for d in cut)])
    return list(itertools.product(*per_axis))


def apply_grids(pk, grids: Dict[int, tuple]) -> List[Node]:
    """Regrid every node the plan gives a grid (kn_graph.regrid, params unchanged); returns the nodes as they are then."""
    for n in graph_nodes(pk):
        if n.graph_idx not in grids:
            continue
        grid = tuple(grids[n.graph_idx])
        if grid not in candidate_grids(n):
            raise ValueError(f"node {n.graph_idx} ({n.name}): grid {grid} is not one of {candidate_grids(n)}")
        pk.kn_graph.regrid(n.graph_idx, grid, n.params)
    return graph_nodes(pk)


def owned_row_blocks(num_blocks: int, gpu: int, num_gpus: int) -> Tuple[int, int]:
    """The blocks [lo, hi) GPU `gpu` computes when num_blocks are split over num_gpus GPUs.
    28 blocks, 8 GPUs: GPU 0 -> [0, 3), GPU 1 -> [3, 7), GPU 2 -> [7, 10), ... (3 or 4 blocks)."""
    return (num_blocks * gpu) // num_gpus, (num_blocks * (gpu + 1)) // num_gpus


def owned_positions(n: "Node", grid_x: int, gpu: int, num_gpus: int) -> Tuple[int, int]:
    """The grid-x positions [lo, hi) GPU `gpu` computes of node n with grid_x positions. A rows_split_over_gpus node is split over the
    GPUs in the blocks of its declared grid (info gpu_split_blocks), whatever its grid now, so that every node split this way gives a
    GPU the same rows: latent_down declared (28, 1, 1); with grid x 56 (64-row tasks) GPU 1 owns blocks 3..6 = positions 6..13,
    the rows sum_quant_send task 3..6 of GPU 1 reads. Other nodes: all positions."""
    if not n.info["rows_split_over_gpus"]:
        return 0, grid_x
    blocks = n.info["gpu_split_blocks"]
    if grid_x % blocks != 0:
        raise ValueError(f"node {n.graph_idx} ({n.name}): grid x {grid_x} is not a multiple of its GPU split blocks {blocks}")
    lo, hi = owned_row_blocks(blocks, gpu, num_gpus)
    return lo * (grid_x // blocks), hi * (grid_x // blocks)


def box(tensor, tensor_map: tuple, grid: tuple, pos: tuple) -> Box:
    """The slice of `tensor` task `pos` touches (MPK's rule): for each grid axis a with tensor_map[a] = d >= 0, dimension d is
    cut in grid[a] equal blocks and the task gets block pos[a]; dimensions no axis cuts are whole.
    Router output [8, 896], map (1, -1, -1), grid (7, 14, 1), task (2, 5): ((0, 8), (256, 384)); grid y cuts nothing, so all
    14 K parts write the same columns (each its own partial sum)."""
    ranges = [(0, d) for d in dims(tensor)]
    for axis, d in enumerate(tensor_map):
        if d >= 0:
            if dims(tensor)[d] % grid[axis] != 0:
                raise ValueError(f"tensor dimension {d} ({dims(tensor)[d]}) does not divide into grid[{axis}] = {grid[axis]} blocks")
            block = dims(tensor)[d] // grid[axis]
            ranges[d] = (pos[axis] * block, (pos[axis] + 1) * block)
    return tuple(ranges)


@dataclass
class Task:
    id: int                       # index in this GPU's task list
    node: int                     # graph_idx
    pos: Tuple[int, int, int]
    reads: List[Tuple[str, Box]]  # (tensor name, slice)
    writes: List[Tuple[str, Box]]
    deps: List[int] = field(default_factory=list)


def gpu_tasks(pk, nodes: List[Node], gpu: int, num_gpus: int) -> List[Task]:
    """Every task GPU `gpu` computes: every grid position of every node (graph order, then position order); for a rows_split_over_gpus
    node only its part of grid x (latent_down grid (28, 8, 1): GPU 0 gets positions (0..2, 0..7, 0); owned_positions)."""
    tasks = []
    for n in nodes:
        lo, hi = owned_positions(n, n.grid[0], gpu, num_gpus)
        tensors = n.inputs + n.outputs
        for pos in itertools.product(*(range(size) for size in n.grid)):
            if not lo <= pos[0] < hi:
                continue
            slices = [(tname(pk, t), box(t, m, n.grid, pos)) for t, m in zip(tensors, n.info["input_maps"])]
            tasks.append(Task(len(tasks), n.graph_idx, tuple(pos), slices[:len(n.inputs)], slices[len(n.inputs):]))
    return tasks


def overlaps(a: Box, b: Box) -> bool:
    return all(a0 < b1 and b0 < a1 for (a0, a1), (b0, b1) in zip(a, b))


def add_dependencies(tasks: List[Task]) -> None:
    """task.deps = the tasks of other nodes that write a slice overlapping a slice the task reads.
    Route task t reads router_logits rows [t, t + 1), all columns; router task (x, y) writes all rows, columns
    [128 x, 128 x + 128): they overlap, so topk_route t waits for all 7 x 14 router tasks."""
    writers: Dict[str, List[Tuple[Box, Task]]] = {}
    for t in tasks:
        for name, b in t.writes:
            writers.setdefault(name, []).append((b, t))
    for t in tasks:
        t.deps = sorted({w.id for name, b in t.reads for wb, w in writers.get(name, []) if w.node != t.node and overlaps(b, wb)})


def concurrent_groups(tasks: List[Task], nodes: List[Node]) -> List[List[int]]:
    """The tasks that must run at the same time: for a node with concurrent_along_axis a, the tasks whose positions differ only
    in axis a. sum_rmsnorm (2, 8, 1), axis 0: the 2 half tasks of each token -> 8 groups of 2."""
    groups: Dict[tuple, List[int]] = {}
    axis_of = {n.graph_idx: n.info["concurrent_along_axis"] for n in nodes}
    for t in tasks:
        a = axis_of[t.node]
        if a is not None:
            groups.setdefault((t.node,) + tuple(p for i, p in enumerate(t.pos) if i != a), []).append(t.id)
    return list(groups.values())


def lists_of_gpu(tasks: List[Task], nodes: List[Node], plan_lists: List[list], num_sms: int, gpu: int) -> List[List[int]]:
    """Per SM, the task ids in the order the SM runs them: the plan's tasks ((graph_idx, position) -> task id). Checks: every
    task in exactly one list; a one_task_per_sm node's task x in SM x's list; the tasks of a concurrent group on different SMs."""
    by_pos = {(t.node, t.pos): t.id for t in tasks}
    if len(plan_lists) != num_sms:
        raise ValueError(f"GPU {gpu}: the plan has {len(plan_lists)} SM lists, the GPU has {num_sms} SMs")
    per_sm_node = {n.graph_idx for n in nodes if n.info["one_task_per_sm"]}
    for n in nodes:
        if n.graph_idx in per_sm_node and n.grid[0] != num_sms:
            raise ValueError(f"node {n.graph_idx} ({n.name}): one task per SM needs grid x = {num_sms}, it has {n.grid[0]}")
    lists, sm_of = [], {}
    for sm in range(num_sms):
        sm_list = []
        for node, pos in plan_lists[sm]:
            if (node, tuple(pos)) not in by_pos:
                raise ValueError(f"GPU {gpu} SM {sm}: the plan lists task {tuple(pos)} of node {node}, "
                                 f"which is not a task of this GPU")
            if node in per_sm_node and pos[0] != sm:
                raise ValueError(f"GPU {gpu} SM {sm}: the plan lists task {tuple(pos)} of node {node}; "
                                 f"one task per SM: task x runs on SM x")
            sm_list.append(by_pos[(node, tuple(pos))])
            sm_of[sm_list[-1]] = sm
        lists.append(sm_list)
    counts = Counter(i for sm_list in lists for i in sm_list)
    missing = sorted(set(t.id for t in tasks) - set(counts))
    twice = sorted(i for i, c in counts.items() if c > 1)
    if missing or twice:
        first_missing = [(tasks[i].node, tasks[i].pos) for i in missing[:5]]
        raise ValueError(f"GPU {gpu}: plan lists miss {len(missing)} tasks (first {first_missing}) "
                         f"and list {len(twice)} twice (first {twice[:5]})")
    pairs_of = {n.graph_idx for n in nodes if in_cluster_pairs(n)}
    for group in concurrent_groups(tasks, nodes):
        if len({sm_of[t] for t in group}) != len(group):
            raise ValueError(f"GPU {gpu}: tasks {[(tasks[t].node, tasks[t].pos) for t in group]} must run at the same time, "
                             f"on different SMs; the plan puts some on the same SM")
        if tasks[group[0]].node in pairs_of and (len(group) != 2 or {sm_of[t] // 2 for t in group} != {sm_of[group[0]] // 2}):
            raise ValueError(f"GPU {gpu}: tasks {[(tasks[t].node, tasks[t].pos) for t in group]} must be on the 2 CTAs of one "
                             f"cluster (CTAs 2k, 2k + 1); the plan puts them on {[sm_of[t] for t in group]}")
    return lists


def in_cluster_pairs(node: Node) -> bool:
    """The node's concurrent groups run on cluster pairs: compile_plan gave it its cluster_pair_params."""
    return node.info["cluster_pair_params"] is not None and list(node.params) == list(node.info["cluster_pair_params"])


def can_pair(node: Node, grid: Optional[tuple] = None) -> bool:
    """The node may run its concurrent groups on cluster pairs: it has cluster_pair_params and groups of 2 tasks (with `grid`, or
    its own grid)."""
    a = node.info["concurrent_along_axis"]
    return node.info["cluster_pair_params"] is not None and a is not None and (grid or node.grid)[a] == 2


def pair_clusters(nodes: List[Node], paired: set, plan_lists: List[list], num_sms: int, gpu: int) -> List[list]:
    """The cluster pairing pass (the plan places tasks without a pair rule): renumber the SM lists so that the 2 tasks of every
    concurrent group of the nodes in `paired` (sum_rmsnorm: a token's 2 halves) are on the 2 CTAs of one cluster, CTAs 2k, 2k + 1. Raises
    ValueError when they do not fit (an SM holds tasks of two pairs, an odd number of CTAs). Every CTA
    runs the same kernel, so a list can move to any CTA index; a one_task_per_sm task moves with its list and takes the new index
    (its body uses the CTA's own index, not the task's). Pairs already on one cluster stay; the others take the free clusters,
    lowest first; every other list keeps its index when that is free, else takes the lowest free one. Returns the lists (the
    plan's own when nothing moves).
    Example: token 0's halves on SMs 5 and 9 -> clusters 0..1 are free -> SM 5's list becomes CTA 0's, SM 9's CTA 1's, and the
    lists of SMs 0 and 1 move to the lowest indices left free (5 and 9)."""
    by_idx = {n.graph_idx: n for n in nodes}
    if not paired:
        return plan_lists
    if num_sms % 2:
        raise ValueError(f"GPU {gpu}: cluster pairs need an even number of CTAs, the GPU has {num_sms}")
    group_sms: Dict[tuple, list] = {}
    group_of_sm: Dict[int, tuple] = {}
    for sm in range(num_sms):
        for node, pos in plan_lists[sm]:
            if node not in paired:
                continue
            a = by_idx[node].info["concurrent_along_axis"]
            key = (node,) + tuple(p for i, p in enumerate(pos) if i != a)
            if group_of_sm.setdefault(sm, key) != key:
                raise ValueError(f"GPU {gpu} SM {sm}: holds tasks of two cluster pairs ({group_of_sm[sm]} and {key}); "
                                 f"each CTA of a cluster can hold one task of one pair")
            group_sms.setdefault(key, []).append(sm)
    new_of: Dict[int, int] = {}
    for key, sms in group_sms.items():
        if len(sms) != 2 or sms[0] == sms[1]:
            raise ValueError(f"GPU {gpu}: the pair {key} must be 2 tasks on 2 SMs; the plan puts its tasks on {sms}")
        if sms[0] // 2 == sms[1] // 2:
            new_of[sms[0]], new_of[sms[1]] = sms[0], sms[1]
    taken = set(new_of.values())
    free_clusters = [k for k in range(num_sms // 2) if 2 * k not in taken and 2 * k + 1 not in taken]
    for key, sms in sorted(group_sms.items(), key=lambda kv: min(kv[1])):
        if sms[0] in new_of:
            continue
        k = free_clusters.pop(0)
        new_of[sms[0]], new_of[sms[1]] = 2 * k, 2 * k + 1
        taken |= {2 * k, 2 * k + 1}
    rest = [sm for sm in range(num_sms) if sm not in new_of]
    for sm in rest:
        if sm not in taken:
            new_of[sm] = sm
            taken.add(sm)
    free = iter(i for i in range(num_sms) if i not in taken)
    for sm in rest:
        if sm not in new_of:
            new_of[sm] = next(free)
    if all(new_of[sm] == sm for sm in range(num_sms)):
        return plan_lists
    lists: List[list] = [None] * num_sms
    for sm in range(num_sms):
        new = new_of[sm]
        # a one_task_per_sm task takes its list's new index (task x runs on SM x); the other tasks keep their positions
        moved = lambda node, pos: (new,) + tuple(pos[1:]) if by_idx[node].info["one_task_per_sm"] else tuple(pos)
        lists[new] = [(node, moved(node, pos)) for node, pos in plan_lists[sm]]
    print(f"GPU {gpu}: cluster pairing moved {sum(new_of[sm] != sm for sm in range(num_sms))} SM lists", flush=True)
    return lists


def node_table(nodes: List[Node]) -> Dict[str, dict]:
    """What the kernel's case for each node needs: its name, grid, params, its first counter and its output buffer slot (-1:
    none; static_megakernel._add_node), its own buffer and map slots and whether it has a dry pass, and per input the node that writes it (grid,
    counter, buffer slot, params; None for a graph input). topk_route after the router in 14 K parts: {"name": "topk_route",
    "grid": [8, 1, 1], "params": [896, 16], "counter": -1, "buf": 2, "inputs": [{"grid": [7, 14, 1], "counter": -1, "buf": 0,
    "params": [0, 7168, 896, 0]}, None], "buf_slots": [1], "map_slots": [], "dry": True}."""
    table = {}
    for n in nodes:
        producers = [producer_of(t, nodes) for t in n.inputs]
        table[str(n.graph_idx)] = {"name": n.name, "grid": list(n.grid), "params": list(n.params),
                                   "counter": n.info["counter"], "buf": n.info["buf"],
                                   "inputs": [{"grid": list(p.grid), "counter": p.info["counter"],
                                               "buf": p.info["buf"], "params": list(p.params)} if p else None
                                              for p in producers],
                                   "buf_slots": list(n.info["buf_slots"]), "map_slots": list(n.info["map_slots"]),
                                   "dry": n.info["dry"]}
    return table


def compile_plan(pk, plan: dict, out_dir: str) -> Tuple[List[str], dict]:
    """Returns (the schedule file paths in GPU order, {"grids": {name#graph_idx: grid}}). See the module comment. For pk's GPUs
    and SMs (StaticMegakernel.num_gpus, num_sms); an SM's list holds at most static_schedule.MAX_TASKS_PER_SM - 1 tasks (then
    the end entry)."""
    num_gpus, num_sms, max_tasks = pk.num_gpus, pk._num_sms(), static_schedule.MAX_TASKS_PER_SM - 1
    if len(plan["lists"]) != num_gpus:
        raise ValueError(f"the plan has lists for {len(plan['lists'])} GPUs, compiling for {num_gpus}")
    os.makedirs(out_dir, exist_ok=True)
    nodes = apply_grids(pk, {int(k): v for k, v in plan["grids"].items()})
    # the launch mode: clusters of 2 when every pair of a node that can pair (can_pair: sum_rmsnorm with 2 tasks per token) fits on
    # one cluster after renumbering the lists (pair_clusters); those nodes then take their cluster_pair_params. Otherwise (or
    # plan "cluster_launch": false) the plain launch, and the nodes keep their params (sum_rmsnorm: the swap through G).
    paired = {n.graph_idx for n in nodes if can_pair(n)}
    plan_lists, cluster_size = plan["lists"], 1
    if paired and plan.get("cluster_launch", True):
        try:
            plan_lists = [pair_clusters(nodes, paired, plan["lists"][gpu], num_sms, gpu) for gpu in range(num_gpus)]
            cluster_size = 2
            for n in nodes:
                if n.graph_idx in paired:
                    n.params = list(n.info["cluster_pair_params"])
        except ValueError as e:
            print(f"plain launch: the cluster pairs do not fit ({e})", flush=True)
    table = node_table(nodes)
    info = {"grids": {f"{n.name}#{n.graph_idx}": list(n.grid) for n in nodes}, "cluster_size": cluster_size}
    paths = []
    for gpu in range(num_gpus):
        tasks = gpu_tasks(pk, nodes, gpu, num_gpus)
        add_dependencies(tasks)
        lists = lists_of_gpu(tasks, nodes, plan_lists[gpu], num_sms, gpu)
        longest = max(len(sm_list) for sm_list in lists)
        if longest > max_tasks:
            raise ValueError(f"GPU {gpu}: an SM has {longest} tasks, at most {max_tasks}")
        all_tasks = [{"node": t.node, "pos": list(t.pos), "deps": t.deps} for t in tasks]
        paths.append(os.path.join(out_dir, f"schedule_gpu{gpu}.json"))
        static_schedule.write_schedule(paths[-1], table, all_tasks, lists, {"cluster_size": cluster_size, "gpu": gpu},
                                       concurrent_groups(tasks, nodes))
    print(f"plan: grids {info['grids']} | {'cluster launch (2 CTAs)' if cluster_size == 2 else 'plain launch'}", flush=True)
    return paths, info


def write_plan_file(path: str, plan: dict) -> None:
    doc = {"grids": {str(k): list(v) for k, v in plan["grids"].items()},
           "lists": [[[[node, list(pos)] for node, pos in sm_list] for sm_list in gpu_lists] for gpu_lists in plan["lists"]]}
    if "cluster_launch" in plan:   # optional: false = the plain launch even when the cluster pairs fit (compile_plan)
        doc["cluster_launch"] = bool(plan["cluster_launch"])
    with open(path, "w") as f:
        json.dump(doc, f)


def plan_from_file(path: str) -> dict:
    with open(path) as f:
        doc = json.load(f)
    plan = {"grids": {int(k): tuple(v) for k, v in doc["grids"].items()},
            "lists": [[[(node, tuple(pos)) for node, pos in sm_list] for sm_list in gpu_lists] for gpu_lists in doc["lists"]]}
    if "cluster_launch" in doc:
        plan["cluster_launch"] = bool(doc["cluster_launch"])
    return plan
