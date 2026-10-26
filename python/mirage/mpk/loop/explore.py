"""Plans for measuring (docs/megakernel_dev_loop.md section 5): they fill the cost records with the contexts the model has
the fewest records of. They are measured with the timing build only (run.py --record) and never compared with the incumbent.

  random_plan(pk, gpus, sms, seed)        a random split and placement (every option the layers declare gets records)
  greedy_plan(pk, cost, gpus, sms)        the per-operator greedy baseline: greedy_grids placed by round_robin_plan
  round_robin_plan(pk, grids, gpus, sms)  given grids, every node's tasks round robin over all SMs in graph order
  perturb_plan(pk, plan, seed, kind)      the plan with one change: one node's tasks moved to other SMs ("move_node"), or one
                                          node's grid changed ("option"); checked by compiling it
  exploration_targets(records, pk, k)     the (node, grid, first on SM or not) combinations with the fewest measured values
"""
import itertools
import random
import tempfile
from typing import List, Optional

from ..compiler import (apply_grids, can_pair, candidate_grids, compile_plan, concurrent_groups, gpu_tasks, graph_nodes,
                        owned_positions, producer_of)


def random_plan(pk, num_gpus: int, num_sms: int, seed: int, max_tasks_per_sm: int = 62, tries: int = 100) -> dict:
    """A plan for measuring costs (demo --plan random:SEED --record): every node with changeable grid dims gets one of its
    candidate_grids at random, every task a random SM: a one_task_per_sm node's task x SM x, the tasks of a concurrent group
    different SMs. On every SM the tasks run in graph order, so a task never waits for one behind it on its SM (dependencies go
    from earlier to later graph nodes); within a node in position order, except that a concurrent node's tasks are ordered by
    their group first (sum_rmsnorm (x, token): by token, then x), so every SM meets the groups in the same order (otherwise group A
    can wait on an SM that is busy with group B, which waits on an SM busy with A). Seed s gives the same plan every time; a draw with more than
    max_tasks_per_sm tasks on an SM is drawn again (up to `tries` times)."""
    rng = random.Random(seed)
    nodes = graph_nodes(pk)
    before = {n.graph_idx: (tuple(n.grid), list(n.params)) for n in nodes}
    grids = {n.graph_idx: rng.choice(candidate_grids(n)) for n in nodes if n.info["changeable_grid_dims"]}
    nodes = apply_grids(pk, grids)
    by_idx = {n.graph_idx: n for n in nodes}
    lists = []
    for gpu in range(num_gpus):
        tasks = gpu_tasks(pk, nodes, gpu, num_gpus)
        for _ in range(tries):
            sm_of, pair_sms = {}, set()
            for group in concurrent_groups(tasks, nodes):          # members on different SMs; a pair's SMs hold no other pair, so
                if can_pair(by_idx[tasks[group[0]].node]):          # compile_plan's pair_clusters can put every pair on a cluster
                    sms = rng.sample([s for s in range(num_sms) if s not in pair_sms], len(group)); pair_sms |= set(sms)
                else:
                    sms = rng.sample(range(num_sms), len(group))
                for t, sm in zip(group, sms):
                    sm_of[t] = sm
            for t in tasks:
                if by_idx[t.node].info["one_task_per_sm"]:
                    sm_of[t.id] = t.pos[0]
                elif t.id not in sm_of:
                    sm_of[t.id] = rng.randrange(num_sms)
            per_sm = [[] for _ in range(num_sms)]
            for t in tasks:
                per_sm[sm_of[t.id]].append((t.node, t.pos))
            def order(entry):
                node, pos = entry
                a = by_idx[node].info.get("concurrent_along_axis")
                return (node,) + ((tuple(p for i, p in enumerate(pos) if i != a) + (pos[a],)) if a is not None else tuple(pos))
            if max(len(l) for l in per_sm) <= max_tasks_per_sm:
                break
        else:
            raise ValueError(f"random_plan seed {seed}: GPU {gpu}: no draw with at most {max_tasks_per_sm} tasks per SM")
        lists.append([sorted(l, key=order) for l in per_sm])
    for idx, (grid, params) in before.items():   # the graph's grids put back (the plan names its own)
        pk.kn_graph.regrid(idx, grid, params)
    return {"grids": grids, "lists": lists}


def greedy_grids(pk, cost, num_gpus: int, num_sms: int) -> dict:
    """The per-operator greedy split (the rule a person writes): every node with changeable grid dims, in graph order, gets the
    candidate grid that makes the node alone fastest on all SMs, ceil(tasks of a GPU / num_sms) x its task time (cost.task_us
    with the grids its producers got; a tie: more tasks). A grid with no recorded time is skipped; a node with none keeps its
    grid. Returns {graph_idx: grid}."""
    nodes = graph_nodes(pk)
    grids = {}
    for n in nodes:
        if not n.info["changeable_grid_dims"]:
            continue
        best = None
        for grid in candidate_grids(n):
            pgrids = [None if p is None else tuple(grids.get(p.graph_idx, p.grid))
                      for p in (producer_of(t, nodes) for t in n.inputs)]
            us = cost.task_us(n, grid, pgrids)
            if us is None:
                continue
            lo, hi = owned_positions(n, grid[0], 0, num_gpus)
            count = (hi - lo) * grid[1] * grid[2]
            alone = -(-count // num_sms) * us
            if best is None or (alone, -count) < best[0]:
                best = ((alone, -count), tuple(grid))
        if best is not None:
            grids[n.graph_idx] = best[1]
    return grids


def round_robin_plan(pk, grids: dict, num_gpus: int, num_sms: int, max_tasks_per_sm: int = 62) -> dict:
    """The rule placement: the nodes regridded to `grids`, every node's tasks round robin over all SMs in graph order,
    continuing from the SM after the previous node's last task; a one_task_per_sm node's task x on SM x. The lists are in graph
    order (random_plan's order), so no task waits for one behind it on its SM, and a concurrent group's members land on
    consecutive SMs."""
    nodes = graph_nodes(pk)
    before = {n.graph_idx: (tuple(n.grid), list(n.params)) for n in nodes}
    nodes = apply_grids(pk, grids)
    by_idx = {n.graph_idx: n for n in nodes}

    def order(t):
        a = by_idx[t.node].info.get("concurrent_along_axis")
        return (t.node,) + ((tuple(p for i, p in enumerate(t.pos) if i != a) + (t.pos[a],)) if a is not None else tuple(t.pos))
    lists = []
    for gpu in range(num_gpus):
        per_sm, nxt = [[] for _ in range(num_sms)], 0
        for t in sorted(gpu_tasks(pk, nodes, gpu, num_gpus), key=order):
            if by_idx[t.node].info["one_task_per_sm"]:
                per_sm[t.pos[0]].append((t.node, t.pos))
            else:
                per_sm[nxt].append((t.node, t.pos))
                nxt = (nxt + 1) % num_sms
        if max(len(l) for l in per_sm) > max_tasks_per_sm:
            raise ValueError(f"round_robin_plan: GPU {gpu}: an SM has more than {max_tasks_per_sm} tasks")
        lists.append(per_sm)
    for idx, (grid, params) in before.items():   # the graph's grids put back (the plan names its own)
        pk.kn_graph.regrid(idx, grid, params)
    return {"grids": dict(grids), "lists": lists}


def greedy_plan(pk, cost, num_gpus: int, num_sms: int) -> dict:
    """The per-operator greedy baseline: greedy_grids, placed by round_robin_plan."""
    return round_robin_plan(pk, greedy_grids(pk, cost, num_gpus, num_sms), num_gpus, num_sms)


def _insert(sm_list: list, entry: tuple) -> None:
    """Put (node, pos) into an SM's list before the first entry of a later graph node, after the entries of its own node with
    smaller positions (random_plan's graph order): [(26, a), (31, b)] + (29, c) -> [(26, a), (29, c), (31, b)]."""
    node, pos = entry
    i = 0
    while i < len(sm_list) and (sm_list[i][0] < node or (sm_list[i][0] == node and tuple(sm_list[i][1]) < tuple(pos))):
        i += 1
    sm_list.insert(i, (node, tuple(pos)))


def _copy(plan: dict) -> dict:
    new = dict(plan)
    new["grids"] = dict(plan["grids"])
    new["lists"] = [[list(sm_list) for sm_list in gpu_lists] for gpu_lists in plan["lists"]]
    return new


def _draw(pk, plan: dict, rng: random.Random, kind: str) -> dict:
    nodes = graph_nodes(pk)
    num_gpus, num_sms = len(plan["lists"]), len(plan["lists"][0])
    new = _copy(plan)
    if kind == "move_node":
        placed = sorted({node for gpu_lists in plan["lists"] for sm_list in gpu_lists for node, _ in sm_list})
        movable = [n for n in nodes if n.graph_idx in placed and not n.info["one_task_per_sm"]
                   and n.info["concurrent_along_axis"] is None]
        if not movable:
            raise ValueError("no node can move (all are one per SM or in concurrent groups)")
        n = rng.choice(movable)
        sm_seq = rng.getstate()
        for gpu in range(num_gpus):
            rng.setstate(sm_seq)                     # the same SM draws on every GPU
            lists = new["lists"][gpu]
            moved = []
            for sm, sm_list in enumerate(lists):
                moved += [(sm, e) for e in sm_list if e[0] == n.graph_idx]
                lists[sm] = [e for e in sm_list if e[0] != n.graph_idx]
            for old_sm, e in moved:
                sm = rng.choice([s for s in range(num_sms) if s != old_sm] or [old_sm])
                _insert(lists[sm], e)
        new["why"] = f"move_node {n.graph_idx} ({n.name})"
        return new
    if kind == "option":
        choosable = [n for n in nodes if len(candidate_grids(n)) > 1]
        if not choosable:
            raise ValueError("no node has several grids")
        n = rng.choice(choosable)
        now = tuple(plan["grids"].get(n.graph_idx, n.grid))
        grid = rng.choice([g for g in candidate_grids(n) if tuple(g) != now])
        new["grids"][n.graph_idx] = tuple(grid)
        for gpu in range(num_gpus):
            lists = new["lists"][gpu]
            sms = []
            for sm, sm_list in enumerate(lists):
                if any(e[0] == n.graph_idx for e in sm_list):
                    sms.append(sm)
                    lists[sm] = [e for e in sm_list if e[0] != n.graph_idx]
            sms = sms or list(range(num_sms))
            lo, hi = owned_positions(n, grid[0], gpu, num_gpus)
            positions = [pos for pos in itertools.product(*(range(v) for v in grid)) if lo <= pos[0] < hi]
            for i, pos in enumerate(positions):
                _insert(lists[sms[i % len(sms)]], (n.graph_idx, pos))
        new["why"] = f"option {n.graph_idx} ({n.name}) {now} -> {tuple(grid)}"
        return new
    raise ValueError(f"kind {kind}: move_node or option")


def perturb_plan(pk, plan: dict, seed: int, kind: str, draws: int = 20) -> dict:
    """A plan that differs from `plan` in one thing (docs/megakernel_dev_loop.md section 5, exploration):
      move_node  one placed node (not one task per SM, not in concurrent groups) has every task moved to another random SM (the
                 same draws on every GPU), inserted in graph order (_insert)
      option     one node with several candidate grids gets another one; its tasks go round robin over the SMs its old tasks
                 used, in graph order
    Each draw is compiled (compiler.compile_plan into a temporary directory: every task once, the per-SM cap, no deadlock); a
    draw that fails is drawn again, up to `draws` times, then ValueError. The returned plan has "why" (what changed). The nodes'
    grids in pk are put back afterwards (compile_plan regrids them)."""
    before = {n.graph_idx: (tuple(n.grid), list(n.params)) for n in graph_nodes(pk)}
    rng = random.Random(seed)
    last = None
    try:
        for _ in range(draws):
            try:
                new = _draw(pk, plan, rng, kind)
                with tempfile.TemporaryDirectory() as d:
                    compile_plan(pk, {k: v for k, v in new.items() if k != "why"}, d)
                return new
            except ValueError as e:
                last = e
                if "no node" in str(e) or str(e).startswith("kind "):
                    raise
    finally:
        for idx, (grid, params) in before.items():
            pk.kn_graph.regrid(idx, grid, params)
    raise ValueError(f"perturb_plan {kind} seed {seed}: no valid draw in {draws} ({last})")


def option_plans(pk, plan: dict, per_node: int = 3, nodes_only: Optional[List[int]] = None) -> List[dict]:
    """Candidates that change one node's grid (its split): for each node with several candidate grids (or the nodes in
    nodes_only), up to per_node other grids, the nearest first by task count. The node's tasks are re-placed round robin over
    every SM, the least loaded first (tasks in the plan per SM, then SM number): a split into more tasks spreads over more SMs
    (re-using only the node's old SMs would put the new tasks behind each other). Inserted in graph order (_insert). Each plan
    is compiled (compile_plan into a temporary directory); one that fails is left out. Returns plans with "why".
    KDA's projection (99, 1, 1) on SMs 0..98: (99, 2, 1) -> 198 tasks, the first 49 on the least loaded SMs 99.., then the rest."""
    before = {n.graph_idx: (tuple(n.grid), list(n.params)) for n in graph_nodes(pk)}
    out = []
    try:
        for n in graph_nodes(pk):
            grids = candidate_grids(n)
            if len(grids) < 2 or (nodes_only is not None and n.graph_idx not in nodes_only):
                continue
            now = tuple(plan["grids"].get(n.graph_idx, n.grid))
            count = lambda g: g[0] * g[1] * g[2]
            others = sorted((tuple(g) for g in grids if tuple(g) != now), key=lambda g: (abs(count(g) - count(now)), count(g)))
            for grid in others[:per_node]:
                new = _copy(plan)
                new["grids"][n.graph_idx] = grid
                for gpu, lists in enumerate(new["lists"]):
                    for sm in range(len(lists)):
                        lists[sm] = [e for e in lists[sm] if e[0] != n.graph_idx]
                    order = sorted(range(len(lists)), key=lambda sm: (len(lists[sm]), sm))
                    lo, hi = owned_positions(n, grid[0], gpu, len(new["lists"]))
                    positions = [pos for pos in itertools.product(*(range(v) for v in grid)) if lo <= pos[0] < hi]
                    for i, pos in enumerate(positions):
                        _insert(lists[order[i % len(order)]], (n.graph_idx, pos))
                new["why"] = f"option {n.graph_idx} ({n.name}) {now} -> {grid}"
                try:
                    with tempfile.TemporaryDirectory() as d:
                        compile_plan(pk, {k: v for k, v in new.items() if k != "why"}, d)
                except ValueError:
                    continue
                out.append(new)
    finally:
        for idx, (grid, params) in before.items():
            pk.kn_graph.regrid(idx, grid, params)
    return out


def exploration_targets(records, pk, k: int = 5) -> List[dict]:
    """The k (node, grid, first on SM; a pool: None) combinations with the fewest measured values (the sum of the records' n over the records of
    that task type, params, shapes and grid, any producer grids), fewest first: where the next exploration plans should put tasks.
    records: a list of records, or an object with .records (model.Model, search.Costs).
    [{"node": 26, "name": "gemm_tile", "grid": (7, 2, 1), "first_on_sm": True, "records": 0, "values": 0}, ...]"""
    recs = getattr(records, "records", records)
    out = []
    for n in graph_nodes(pk):
        shapes = [[t.dim(i) for i in range(t.num_dims)] for t in n.inputs + n.outputs]
        for grid in candidate_grids(n):
            for first in ((None,) if n.info["one_task_per_sm"] else (True, False)):   # a pool: one record per run, no context
                match = [r for r in recs if r["task_type"] == n.name and r["key"].get("params") == list(n.params)
                         and r["key"].get("shapes") == shapes and tuple(r["key"].get("grid", ())) == tuple(grid)
                         and (first is None or (r.get("context") or {}).get("first_on_sm") is first)]
                out.append({"node": n.graph_idx, "name": n.name, "grid": tuple(grid), "first_on_sm": first,
                            "records": len(match), "values": sum(int(r.get("n", 0)) for r in match)})
    out.sort(key=lambda d: (d["values"], d["node"], d["grid"], d["first_on_sm"]))
    return out[:k]
