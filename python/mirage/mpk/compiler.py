"""Plan -> one schedule file per GPU (schedule_gpu<g>.json), which static_schedule.py turns into layer.cu.

A plan:  {"grids": {graph_idx: grid},                     e.g. {30: (7, 14, 1), 32: (28, 8, 1), 34: (12, 8, 1)}
          "lists": per GPU, per SM, the ordered (graph_idx, grid position) of the SM's tasks}
Nodes not in "grids" keep the grid their layer method gave them. One-per-SM nodes (one_task_per_sm_at_end) are appended to every SM's list.

compile_plan:
  1. apply_grids     each node in the plan: its grid must be one of candidate_grids; kn_graph.regrid
  2. per GPU:
       gpu_tasks        every grid position of every node (split-over-GPU nodes: only this GPU's part), with its slices (box)
       add_dependencies a task waits for the tasks of other nodes whose output slices overlap its input slices
       lists_of_gpu     the plan's lists as task ids (every task exactly once), then the one-per-SM tasks
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
    info: dict                    # kept by the layer method: maps, changeable_grid_dims, rows_split_over_gpus, one_task_per_sm_at_end (static_megakernel._add_node)


def graph_nodes(pk) -> List[Node]:
    """The graph's static-schedule nodes (the ones pk._static_nodes has), in graph order, with their grid and params now."""
    nodes = []
    for graph_idx, info in sorted(pk._static_nodes.items()):
        name, params, grid, num_inputs, num_outputs, tensors, _, _ = pk.kn_graph.get_task_info(graph_idx)
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
    """The grids the compiler may give the node: on each changeable grid dim a (unit u), every size s such that every tensor dimension
    that axis cuts splits into s blocks of a whole number of units; the other axes as they are.
    GEMM, K = 7168, changeable grid dim y with unit 128: 7168 / 128 = 56 units -> y in 1, 2, 4, 7, 8, 14, 28, 56."""
    tensors = node.inputs + node.outputs
    per_axis = []
    for axis in range(3):
        if axis not in node.info["changeable_grid_dims"]:
            per_axis.append([node.grid[axis]])
            continue
        unit = node.info["changeable_grid_dims"][axis]
        cut = [dims(t)[m[axis]] for t, m in zip(tensors, node.info["maps"]) if m[axis] >= 0]
        if not cut:
            raise ValueError(f"node {node.graph_idx} ({node.name}): grid axis {axis} is changeable but no tensor map uses it")
        units = min(d // unit for d in cut)
        per_axis.append([s for s in range(1, units + 1) if all(d % (s * unit) == 0 for d in cut)])
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
    one_task_per_sm_at_end: bool
    deps: List[int] = field(default_factory=list)


def gpu_tasks(pk, nodes: List[Node], gpu: int, num_gpus: int) -> List[Task]:
    """Every task GPU `gpu` computes: every grid position of every node (graph order, then position order); for a rows_split_over_gpus
    node only its part of grid x (latent_down grid (28, 8, 1): GPU 0 gets positions (0..2, 0..7, 0))."""
    tasks = []
    for n in nodes:
        lo, hi = owned_row_blocks(n.grid[0], gpu, num_gpus) if n.info["rows_split_over_gpus"] else (0, n.grid[0])
        tensors = n.inputs + n.outputs
        for pos in itertools.product(*(range(size) for size in n.grid)):
            if not lo <= pos[0] < hi:
                continue
            slices = [(tname(pk, t), box(t, m, n.grid, pos)) for t, m in zip(tensors, n.info["maps"])]
            tasks.append(Task(len(tasks), n.graph_idx, tuple(pos), slices[:len(n.inputs)], slices[len(n.inputs):],
                              n.info["one_task_per_sm_at_end"]))
    return tasks


def overlaps(a: Box, b: Box) -> bool:
    return all(a0 < b1 and b0 < a1 for (a0, a1), (b0, b1) in zip(a, b))


def add_dependencies(tasks: List[Task]) -> None:
    """task.deps = the tasks of other nodes that write a slice overlapping a slice the task reads.
    Route task t reads router_logits rows [t, t + 1), all columns; router task (x, y) writes all rows, columns
    [128 x, 128 x + 128): they overlap, so route t waits for all 7 x 14 router tasks."""
    writers: Dict[str, List[Tuple[Box, Task]]] = {}
    for t in tasks:
        for name, b in t.writes:
            writers.setdefault(name, []).append((b, t))
    for t in tasks:
        t.deps = sorted({w.id for name, b in t.reads for wb, w in writers.get(name, []) if w.node != t.node and overlaps(b, wb)})


def lists_of_gpu(tasks: List[Task], nodes: List[Node], plan_lists: List[list], num_sms: int, gpu: int) -> List[List[int]]:
    """Per SM, the task ids in the order the SM runs them: the plan's tasks ((graph_idx, position) -> task id), then the SM's
    one_task_per_sm_at_end tasks in graph order (the task at grid position x = the SM). Every other task must be in exactly one list."""
    by_pos = {(t.node, t.pos): t.id for t in tasks}
    if len(plan_lists) != num_sms:
        raise ValueError(f"GPU {gpu}: the plan has {len(plan_lists)} SM lists, the GPU has {num_sms} SMs")
    lists = []
    for sm in range(num_sms):
        sm_list = []
        for node, pos in plan_lists[sm]:
            if (node, tuple(pos)) not in by_pos:
                raise ValueError(f"GPU {gpu} SM {sm}: the plan lists task {tuple(pos)} of node {node}, which is not a task of this GPU")
            task = tasks[by_pos[(node, tuple(pos))]]
            if task.one_task_per_sm_at_end:
                raise ValueError(f"GPU {gpu} SM {sm}: the plan lists task {tuple(pos)} of node {node}, which runs once per SM "
                                 f"at the end and is added by the compiler")
            sm_list.append(task.id)
        lists.append(sm_list)
    placed = [t.id for t in tasks if not t.one_task_per_sm_at_end]
    counts = Counter(i for sm_list in lists for i in sm_list)
    missing = sorted(set(placed) - set(counts))
    twice = sorted(i for i, c in counts.items() if c > 1)
    if missing or twice:
        raise ValueError(f"GPU {gpu}: plan lists miss {len(missing)} tasks (first {missing[:5]}) and list "
                         f"{len(twice)} twice (first {twice[:5]})")
    for n in nodes:
        if n.info["one_task_per_sm_at_end"] and n.grid[0] != num_sms:
            raise ValueError(f"node {n.graph_idx} ({n.name}): one task per SM needs grid x = {num_sms}, it has {n.grid[0]}")
    for sm in range(num_sms):
        lists[sm] += [t.id for n in nodes if n.info["one_task_per_sm_at_end"] for t in tasks if t.node == n.graph_idx and t.pos[0] == sm]
    return lists


def node_table(nodes: List[Node]) -> Dict[str, dict]:
    """What the kernel's case for each node needs: its name, grid, params, and per input the grid of the node that writes it
    (None for a graph input). Route after router K14: {"name": "route", "grid": [8, 1, 1], "params": [],
    "input_grids": [[7, 14, 1], None, None]}."""
    table = {}
    for n in nodes:
        producers = [producer_of(t, nodes) for t in n.inputs]
        table[str(n.graph_idx)] = {"name": n.name, "grid": list(n.grid), "params": list(n.params),
                                   "input_grids": [list(p.grid) if p else None for p in producers]}
    return table


def compile_plan(pk, plan: dict, out_dir: str, num_gpus: int, num_sms: int,
                 max_tasks_per_sm: Optional[int] = None) -> Tuple[List[str], dict]:
    """Returns (the schedule file paths in GPU order, {"grids": {name#graph_idx: grid}}). See the module comment.
    max_tasks_per_sm: the most tasks one SM's list may hold (MoE: 63 = the 64 entries of static_mk::MAX_TASKS_PER_SM minus the end
    marker); None: not checked here (the generated host code checks it when the layer is loaded)."""
    if len(plan["lists"]) != num_gpus:
        raise ValueError(f"the plan has lists for {len(plan['lists'])} GPUs, compiling for {num_gpus}")
    os.makedirs(out_dir, exist_ok=True)
    nodes = apply_grids(pk, {int(k): v for k, v in plan["grids"].items()})
    table = node_table(nodes)
    info = {"grids": {f"{n.name}#{n.graph_idx}": list(n.grid) for n in nodes}}
    paths = []
    for gpu in range(num_gpus):
        tasks = gpu_tasks(pk, nodes, gpu, num_gpus)
        add_dependencies(tasks)
        lists = lists_of_gpu(tasks, nodes, plan["lists"][gpu], num_sms, gpu)
        longest = max(len(sm_list) for sm_list in lists)
        if max_tasks_per_sm is not None and longest > max_tasks_per_sm:
            raise ValueError(f"GPU {gpu}: an SM has {longest} tasks, at most {max_tasks_per_sm}")
        all_tasks = [{"node": t.node, "pos": list(t.pos), "deps": t.deps} for t in tasks]
        paths.append(os.path.join(out_dir, f"schedule_gpu{gpu}.json"))
        static_schedule.write_schedule(paths[-1], table, all_tasks, lists, dict(info, gpu=gpu))
    print(f"plan: grids {info['grids']}", flush=True)
    return paths, info


def write_plan_file(path: str, plan: dict) -> None:
    doc = {"grids": {str(k): list(v) for k, v in plan["grids"].items()},
           "lists": [[[[node, list(pos)] for node, pos in sm_list] for sm_list in gpu_lists] for gpu_lists in plan["lists"]]}
    with open(path, "w") as f:
        json.dump(doc, f)


def plan_from_file(path: str) -> dict:
    with open(path) as f:
        doc = json.load(f)
    return {"grids": {int(k): tuple(v) for k, v in doc["grids"].items()},
            "lists": [[[(node, tuple(pos)) for node, pos in sm_list] for sm_list in gpu_lists] for gpu_lists in doc["lists"]]}
