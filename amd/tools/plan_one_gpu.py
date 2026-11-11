"""A plan for the layer on FEWER GPUs (1, 2 or 4) from a plan for more (the first n GPUs' lists): for checking every task on a box
where only a few GPUs are free.

With n GPUs, the nodes whose rows are split over the GPUs (latent_down, sum_quant_send, latent_up, sum_send) give GPU g the row
blocks [B g / n, B (g + 1) / n) instead of its share of the larger count (compiler.owned_row_blocks); every other task stays on its
GPU. For each GPU g < n its lists stay as they are, less the tasks GPU g no longer owns; each task they lack goes onto the
shortest list (fewest tasks; ties: the lowest list), before that list's first task of any node that depends on the task's node
(directly or through other nodes), else at its end. So an SM never meets a task it waits for later in its own list. The grids
stay.

  plan = fewer_gpus_plan(pk, plan)   pk: the StaticMegakernel of the graph, built with num_gpus=n; plan: compiler.plan_from_file's
  one_gpu_plan(pk, plan)             the same, n = 1
  plan = simple_plan(pk, grids)      any small graph: the nodes' tasks in graph order, dealt round-robin over the SMs (a node with
                                     one task per SM: task x on SM x); grids: {graph_idx: grid} for the nodes whose layer default
                                     is not wanted
  plan = phase_plan(pk, grids, phases)  a hand plan: phases = [[graph_idx, ...], ...] in an order every node's producers come
                                     first; each phase's tasks (independent nodes interleaved, one task of each in turn;
                                     otherwise node after node) dealt
                                     round-robin over the SMs, continuing after the SM the previous phase ended on (a node with one
                                     task per SM: task x on SM x). Every SM's list is then in phase order, so no SM meets a task
                                     before one it waits for; nodes not named go in a last phase in graph order.
"""
from typing import Dict, List, Set

from mirage.mpk import compiler


def depends_on(nodes) -> Dict[int, Set[int]]:
    """graph_idx -> the graph_idx of every node it depends on, directly or through other nodes."""
    direct = {n.graph_idx: {p.graph_idx for t in n.inputs for p in [compiler.producer_of(t, nodes)] if p is not None}
              for n in nodes}
    out: Dict[int, Set[int]] = {}
    for n in nodes:   # graph order: a node's producers come first
        out[n.graph_idx] = set().union(*([direct[n.graph_idx]] + [out[p] for p in direct[n.graph_idx]]))
    return out


def fewer_gpus_plan(pk, plan: dict) -> dict:
    n = pk.num_gpus
    if len(plan["lists"]) < n:
        raise ValueError(f"the plan has lists for {len(plan['lists'])} GPUs, the graph is built for {n}")
    nodes = compiler.apply_grids(pk, {int(k): v for k, v in plan["grids"].items()})
    deps = depends_on(nodes)
    out_lists = []
    for gpu in range(n):
        need_list = [(t.node, tuple(t.pos)) for t in compiler.gpu_tasks(pk, nodes, gpu, n)]
        need = set(need_list)
        lists: List[list] = [[(node, tuple(pos)) for node, pos in sm_list if (node, tuple(pos)) in need]
                             for sm_list in plan["lists"][gpu]]
        if len(lists) != pk._num_sms():
            raise ValueError(f"the plan has {len(lists)} lists, the GPU {pk._num_sms()} SMs (widen it first: plan_for_cus.py)")
        removed = sum(len(a) for a in plan["lists"][gpu]) - sum(len(l) for l in lists)
        have = {task for sm_list in lists for task in sm_list}
        added = 0
        for node, pos in need_list:
            if (node, pos) in have:
                continue
            sm = min(range(len(lists)), key=lambda i: (len(lists[i]), i))
            at = next((i for i, (n2, _) in enumerate(lists[sm]) if node in deps[n2]), len(lists[sm]))
            lists[sm].insert(at, (node, pos))
            added += 1
        print(f"GPU {gpu} of {n}: {removed} tasks dropped (rows another GPU owns now), {added} added; longest list "
              f"{max(len(l) for l in lists)} tasks", flush=True)
        out_lists.append(lists)
    out = {"grids": plan["grids"], "lists": out_lists}
    if "cluster_launch" in plan:
        out["cluster_launch"] = plan["cluster_launch"]
    return out


def one_gpu_plan(pk, plan: dict) -> dict:
    assert pk.num_gpus == 1, "the graph is built for one GPU"
    return fewer_gpus_plan(pk, plan)


def simple_plan(pk, grids: dict = None) -> dict:
    grids = {int(k): tuple(v) for k, v in (grids or {}).items()}
    nodes = compiler.apply_grids(pk, grids)
    nsm = pk._num_sms()
    out = []
    for gpu in range(pk.num_gpus):
        lists = [[] for _ in range(nsm)]
        nxt = 0
        for t in compiler.gpu_tasks(pk, nodes, gpu, pk.num_gpus):
            n = next(m for m in nodes if m.graph_idx == t.node)
            if n.info["one_task_per_sm"]:
                lists[t.pos[0]].append((t.node, tuple(t.pos)))
            else:
                lists[nxt].append((t.node, tuple(t.pos)))
                nxt = (nxt + 1) % nsm
        out.append(lists)
    return {"grids": {n.graph_idx: tuple(n.grid) for n in nodes}, "lists": out}


def dedicate_plan(plan: dict, nodes: Set[int]) -> dict:
    """The plan with the tasks of `nodes` alone on CUs of their own, one task each, first in those lists (they start at
    the launch: e.g. MLA's per-head attention / value tasks copy their weights into LDS while the GEMMs before them run), the
    last CUs of the GPU. The tasks those CUs had move to the other CUs, round-robin, each inserted after the list's last task
    of a node <= its own (the lists stay in node order, which is a topological order: no list waits on a later task)."""
    out = []
    for lists in plan["lists"]:
        nsm = len(lists)
        mine = [t for cu in lists for t in cu if t[0] in nodes]
        k = len(mine)
        assert 0 < k < nsm, f"dedicate_plan: {k} tasks of nodes {sorted(nodes)}"
        rest = [[t for t in cu if t[0] not in nodes] for cu in lists]
        moved = [t for cu in rest[nsm - k:] for t in cu]
        new = [list(cu) for cu in rest[:nsm - k]] + [[t] for t in mine]
        for i, task in enumerate(sorted(moved, key=lambda t: t[0])):
            cu = new[i % (nsm - k)]
            at = max((j + 1 for j, t in enumerate(cu) if t[0] <= task[0]), default=0)
            cu.insert(at, task)
        out.append(new)
    return {**plan, "lists": out}


def first_plan(plan: dict, nodes: Set[int]) -> dict:
    """The plan with the tasks of `nodes` first on CUs of their own (one task each, the last CUs of the GPU), which keep their
    tasks of later nodes (after it) and give every task of an earlier node to the other CUs (inserted in node order, as
    dedicate_plan): the node's task starts at the launch (e.g. MLA's per-head attention / value tasks copy their weights into
    LDS while the GEMMs run elsewhere), nothing on its CU waits for it before its turn, and the GEMMs keep most CUs."""
    lo = min(nodes)
    out = []
    for lists in plan["lists"]:
        nsm = len(lists)
        mine = [t for cu in lists for t in cu if t[0] in nodes]
        k = len(mine)
        assert 0 < k < nsm, f"first_plan: {k} tasks of nodes {sorted(nodes)}"
        rest = [[t for t in cu if t[0] not in nodes] for cu in lists]
        moved = [t for cu in rest[nsm - k:] for t in cu if t[0] < lo]
        new = [list(cu) for cu in rest[:nsm - k]] + \
              [[mine[i]] + [t for t in rest[nsm - k + i] if t[0] >= lo] for i in range(k)]
        for i, task in enumerate(sorted(moved, key=lambda t: t[0])):
            cu = new[i % (nsm - k)]
            at = max((j + 1 for j, t in enumerate(cu) if t[0] <= task[0]), default=0)
            cu.insert(at, task)
        out.append(new)
    return {**plan, "lists": out}


def light_plan(plan: dict, node: int, heavy: int) -> dict:
    """The plan with `node`'s tasks moved off the CUs holding the most tasks of node `heavy` (they are free last) onto the
    CUs holding the fewest, round-robin over those (the one with the fewest `node` tasks first), each inserted after the
    list's last task of a node <= its own (node order kept: no list waits on a later task). E.g. KDA's short_conv_silu off
    the CUs with 3 projection tasks (their conv started ~5 us after its input, kda58)."""
    out = []
    for lists in plan["lists"]:
        nh = [sum(t[0] == heavy for t in cu) for cu in lists]
        lo, hi = min(nh), max(nh)
        if lo == hi:
            out.append(lists)
            continue
        new = [[t for t in cu if not (t[0] == node and nh[i] == hi)] for i, cu in enumerate(lists)]
        moved = [t for i, cu in enumerate(lists) if nh[i] == hi for t in cu if t[0] == node]
        light = [i for i in range(len(lists)) if nh[i] == lo]
        for task in moved:
            i = min(light, key=lambda c: (sum(t[0] == node for t in new[c]), c))
            at = max((j + 1 for j, t in enumerate(new[i]) if t[0] <= task[0]), default=0)
            new[i].insert(at, task)
        out.append(new)
    return {**plan, "lists": out}


def off_plan(plan: dict, node: int, avoid: Set[int], most: int = 0, prefer: int = None, pmin: int = 1) -> dict:
    """The plan with `node`'s tasks moved off the CUs that hold a task of a node in `avoid`, onto the other CUs with the
    fewest `node` tasks (one at a time, the fewest first), each inserted after the list's last task of a node <= its own.
    E.g. the fused MoE + KDA kernel's projection off moe_agg1's CU (MI355X dec48, T = 1: that CU's projection task started
    after aggregation 1 with no weights issued before it, ended ~11 us after the others and held up the conv, the
    recurrence, o_proj and attn_residual). most > 0: off the CUs holding more than `most` tasks of the `avoid` nodes (e.g. the
    KDA conv off the CUs with two projection tasks: there it started after the second one, MI355X dec114 T = 8). prefer: the
    CUs holding at least `pmin` tasks of node `prefer` first (e.g. T 8 KDA o_proj onto the CUs with two projection tasks, free
    by ~92 us, weights in before y lands, instead of after a recurrence task; MI355X dec165)."""
    out = []
    for lists in plan["lists"]:
        bad = [sum(t[0] in avoid for t in cu) > most for cu in lists]
        new = [[t for t in cu if not (t[0] == node and bad[i])] for i, cu in enumerate(lists)]
        moved = [t for i, cu in enumerate(lists) if bad[i] for t in cu if t[0] == node]
        ok = [i for i in range(len(lists)) if not bad[i]]
        for task in moved:
            i = min(ok, key=lambda c: (prefer is not None and sum(t[0] == prefer for t in new[c]) < pmin,
                                       sum(t[0] == node for t in new[c]), c))
            at = max((j + 1 for j, t in enumerate(new[i]) if t[0] <= task[0]), default=0)
            new[i].insert(at, task)
        out.append(new)
    return {**plan, "lists": out}


def late_plan(plan: dict, node: int, late) -> dict:
    """The plan with `node`'s tasks for which late(pos) is true (or by late(pos) as a key, 0 first) in its last slots: the node's slots (its places in the lists, kept)
    in order of their rank on their CU (every CU's first task of the node, then every second, ...), filled with the other tasks
    first, then the late ones. E.g. the KDA projection's g / f_a / beta rows (MPK_KDA_SPLIT_PROJ): the conv waits only for the
    q / k / v rows, which then fit in the first wave."""
    out = []
    for lists in plan["lists"]:
        new = [list(cu) for cu in lists]
        slots = sorted(((r, i, j) for i, cu in enumerate(new)
                        for r, j in enumerate(j for j, t in enumerate(cu) if t[0] == node)), key=lambda s: (s[0], s[1]))
        tasks = [new[i][j] for _, i, j in slots]
        tasks = sorted(tasks, key=lambda t: int(late(tuple(t[1]))))   # (stable: a bool key = the late ones last; an int key: by it)
        for (_, i, j), t in zip(slots, tasks):
            new[i][j] = t
        out.append(new)
    return {**plan, "lists": out}


def apart_plan(plan: dict, node: int, avoid: Set[int]) -> dict:
    """The plan with every task of `node` on a CU of its own that holds no task of the nodes in `avoid` (the CUs with the
    fewest tasks first; a task already on such a CU stays), inserted after the list's last task of a node <= its own. E.g.
    KDA's attn_residual off the o_proj CUs: behind an o_proj task it started 1-3 us after it (its pushes' stores share
    vmcnt) and its token's partials landed 2-4 us later (kda66)."""
    out = []
    for lists in plan["lists"]:
        bad = [any(t[0] in avoid for t in cu) for cu in lists]
        new = [[t for t in cu if not (t[0] == node and bad[i])] for i, cu in enumerate(lists)]
        moved = [t for i, cu in enumerate(lists) if bad[i] for t in cu if t[0] == node]
        free = sorted((i for i in range(len(lists)) if not bad[i] and not any(t[0] == node for t in new[i])),
                      key=lambda c: (len(new[c]), c))
        assert len(free) >= len(moved), f"apart_plan: {len(moved)} tasks of node {node}, {len(free)} CUs free"
        for task, i in zip(moved, free):
            at = max((j + 1 for j, t in enumerate(new[i]) if t[0] <= task[0]), default=0)
            new[i].insert(at, task)
        out.append(new)
    return {**plan, "lists": out}


def rank_plan(plan: dict, node: int, axis: int = 1, free: Set[int] = frozenset(), dies: int = 8) -> dict:
    """The plan with `node`'s tasks (one per CU) permuted within each die (CU % dies, as die_plan): the lowest indices along
    `axis` on the CUs with the fewest tasks before it (tasks of the nodes in `free` not counted). E.g. MLA's attention: the
    low KV splits hold rows at any context length; split 0 sat behind q_b and the gate GEMM (start 22.8 vs 18.7 us, kda73)
    while the empty high splits had lighter CUs."""
    out = []
    for lists in plan["lists"]:
        new = [list(cu) for cu in lists]
        for d in range(dies):
            cus = [i for i in range(d, len(new), dies) if any(t[0] == node for t in new[i])]
            assert all(sum(t[0] == node for t in new[i]) == 1 for i in cus), "rank_plan: one task of the node per CU"
            at = {i: next(j for j, t in enumerate(new[i]) if t[0] == node) for i in cus}
            pre = {i: sum(1 for t in new[i][:at[i]] if t[0] not in free) for i in cus}
            tasks = sorted((new[i][at[i]] for i in cus), key=lambda t: (t[1][axis], t[1]))
            for i, task in zip(sorted(cus, key=lambda i: (pre[i], i)), tasks):
                new[i][at[i]] = task
        out.append(new)
    return {**plan, "lists": out}


def die_plan(plan: dict, node: int, axis: int = 1, dies: int = 8) -> dict:
    """The plan with node `node`'s tasks moved among the places it has (each CU keeps its list; only which task of the node sits
    at a place changes) so that the tasks with the same grid position on `axis` run on one die: CU c's die is c mod `dies` up to
    a fixed rotation (MI355X sync_latency: workgroup w on XCC (w + 2) mod 8, the same every launch). Key k goes to die k mod
    `dies`, while that die has places left; the rest fill the remaining places. Grouped MLA (axis 1 = the KV split): the head
    groups of a split read the same cache rows, through one die's L2 (one HBM read instead of one per group)."""
    out = []
    for gpu in plan["lists"]:
        lists = [list(cu) for cu in gpu]
        places = [(i, j) for i, cu in enumerate(lists) for j, t in enumerate(cu) if t[0] == node]
        tasks = sorted((lists[i][j] for i, j in places), key=lambda t: (t[1][axis], t[1]))
        free = {d: [pl for pl in places if pl[0] % dies == d] for d in range(dies)}
        rest = []
        for t in tasks:
            d = t[1][axis] % dies
            if free[d]:
                i, j = free[d].pop(0)
                lists[i][j] = t
            else:
                rest.append(t)
        left = [pl for d in range(dies) for pl in free[d]]
        assert len(left) == len(rest)
        for (i, j), t in zip(left, rest):
            lists[i][j] = t
        assert sorted(t for cu in lists for t in cu if t[0] == node) == sorted(tasks)
        if rest:
            print(f"die_plan: node {node}: {len(rest)} of {len(tasks)} tasks not on their key's die", flush=True)
        out.append(lists)
    return {**plan, "lists": out}


def regrid_plan(pk, plan: dict, node: int, grid: tuple, after: Set[int]) -> dict:
    """The plan with node `node` at grid `grid` and every other task where it was: the node's old tasks leave their lists; its
    new tasks go round-robin to the CUs that had it, then to the CUs with the fewest tasks, each right after the CU's last task
    of a node in `after` (at the list's start when it has none). `after` must hold every node the CU runs before that the new
    tasks wait for (e.g. latent_up: moe_experts, allreduce_send, sum_rmsnorm, sum_gpus), so no list waits on a later task."""
    grids = {int(k): tuple(v) for k, v in plan["grids"].items()}
    grids[node] = tuple(grid)
    nodes = compiler.apply_grids(pk, grids)
    nsm = pk._num_sms()
    out = []
    for gpu, old in enumerate(plan["lists"]):
        lists = [[t for t in cu if t[0] != node] for cu in old]
        had = [i for i, cu in enumerate(old) if any(t[0] == node for t in cu)]
        order = had + sorted((i for i in range(nsm) if i not in had), key=lambda i: (len(lists[i]), i))
        new = [(t.node, tuple(t.pos)) for t in compiler.gpu_tasks(pk, nodes, gpu, pk.num_gpus) if t.node == node]
        for k, task in enumerate(new):
            cu = lists[order[k % nsm]]
            at = max((j + 1 for j, t in enumerate(cu) if t[0] in after or t[0] == node), default=0)
            cu.insert(at, task)
        out.append(lists)
    plan = {**plan, "grids": grids, "lists": out}
    return plan


def follow_plan(plan: dict, node: int, lead: int, after: Set[int] = frozenset(), avoid: Set[int] = frozenset()) -> dict:
    """The plan with node `node`'s tasks moved onto the CUs that run node `lead`, dealt round-robin over them and each placed
    right after the CU's last task of `lead`, of a node in `after` or of `node` itself: a consumer that waits only for those
    (and earlier nodes) starts as soon as they end on its CU, not behind the CU's later tasks; and a node a re-deal by token
    count (tokens_plan) stacked on its few old CUs spreads over many (MI355X dec30, T = 64: topk_route's 64 tasks 8 per CU on
    8 CUs, the last ending at ~64 us, the router's K parts landed by ~13). `avoid`: not on a CU that holds a task of those
    nodes (e.g. sum_gpus over the experts' CUs but not latent_up's, which pre-issue their weights and wait)."""
    out = []
    for gpu in plan["lists"]:
        lists = [[t for t in cu if t[0] != node] for cu in gpu]
        tasks = [t for cu in gpu for t in cu if t[0] == node]
        leads = [i for i, cu in enumerate(lists) if any(t[0] == lead for t in cu) and not any(t[0] in avoid for t in cu)]
        assert leads, f"follow_plan: no CU runs node {lead}"
        for k, task in enumerate(tasks):
            cu = lists[leads[k % len(leads)]]
            at = max(j + 1 for j, t in enumerate(cu) if t[0] in (lead, node) or t[0] in after)
            cu.insert(at, task)
        out.append(lists)
    return {**plan, "lists": out}


def pack_plan(plan: dict, node: int, per_cu: int) -> dict:
    """The plan with node `node`'s tasks on fewer CUs, `per_cu` of them in a row on each: on every GPU, the CUs that had the
    node (in list order) keep it while tasks are left, each taking the next `per_cu` tasks at the place its one task was; the
    other CUs lose theirs (their other tasks keep their places). The tasks wait for what the one task there waited for, and
    nothing on a CU that loses its task waited for that task there (MI355X 10-12, TP8: the shared expert's down GEMM, 56
    tasks of ~98 KB, held 56 CUs until ~33 us, each waiting from ~9 us for situ_and_mul; its output is needed only at
    the residual, ~95 us)."""
    out = []
    for gpu in plan["lists"]:
        lists = [list(cu) for cu in gpu]
        had = [i for i, cu in enumerate(lists) if any(t[0] == node for t in cu)]
        tasks = [t for i in had for t in lists[i] if t[0] == node]
        for k, i in enumerate(had):
            at = next(j for j, t in enumerate(lists[i]) if t[0] == node)
            mine = tasks[k * per_cu:(k + 1) * per_cu]
            lists[i] = [t for t in lists[i] if t[0] != node]
            lists[i][at:at] = mine
        assert sorted(t for cu in lists for t in cu if t[0] == node) == sorted(tasks)
        out.append(lists)
    return {**plan, "lists": out}


def drop_plan(plan: dict, drop: set) -> dict:
    """The plan without the tasks of the nodes `drop` (graph indices of the full graph), the later nodes renumbered as the graph
    without them numbers them; every other task's place as it was."""
    drop = set(drop)
    def nid(n):
        return n - sum(1 for d in drop if n > d)
    grids = {nid(int(k)): tuple(v) for k, v in plan["grids"].items() if int(k) not in drop}
    lists = [[[(nid(t[0]), tuple(t[1])) for t in cu if t[0] not in drop] for cu in gpu] for gpu in plan["lists"]]
    return {**plan, "grids": grids, "lists": lists}


def insert_plan(plan: dict, at: int, k: int, new: dict) -> dict:
    """The plan for the graph with k operators inserted at index `at` (demo MPK_MOE_INPUT_WAIT: two input_wait nodes and their
    output tensors before the router): the nodes from `at` on renumbered + k; new {node: (grid, [(gpu or None, cu, pos), ...])}
    gives each new task node's grid and its tasks (at the start of the CU's list; gpu None: on every GPU)."""
    def nid(n):
        return n + k if n >= at else n
    grids = {nid(int(n)): tuple(v) for n, v in plan["grids"].items()}
    lists = [[[(nid(t[0]), tuple(t[1])) for t in cu] for cu in gpu] for gpu in plan["lists"]]
    for node, (grid, places) in new.items():
        grids[node] = tuple(grid)
        for gpu, cu, pos in reversed(places):
            for gi, gl in enumerate(lists):
                if gpu is None or gpu == gi:
                    gl[cu].insert(0, (node, tuple(pos)))
    return {**plan, "grids": grids, "lists": lists}


def fold_plan(plan: dict, norm: int = 38, send: int = 41) -> dict:
    """A plan of the graph with sum_rmsnorm (`norm`) and sum_send (`send`) for the graph without them (demo MPK_RMSNORM_FOLD:
    latent_up adds R itself and its last K part per row block finishes): their tasks dropped, the later nodes renumbered
    (39 -> 38, 40 -> 39, 42 -> 40); the other tasks' places as they were."""
    def nid(n):
        return n - (n > norm) - (n > send)
    grids = {nid(int(k)): tuple(v) for k, v in plan["grids"].items() if int(k) not in (norm, send)}
    lists = [[[(nid(t[0]), tuple(t[1])) for t in cu if t[0] not in (norm, send)] for cu in gpu] for gpu in plan["lists"]]
    return {**plan, "grids": grids, "lists": lists}


def lup_finish_plan(plan: dict, send: int = 41, residual: int = 42) -> dict:
    """A plan of the graph with sum_send (`send`) for the graph without it (demo MPK_LATENT_UP_FINISH: latent_up's last K part
    per row block adds the parts and pushes them): sum_send's tasks dropped, residual_add renumbered `residual` -> `send`;
    latent_up's grid and places as they were."""
    grids = {int(k): tuple(v) for k, v in plan["grids"].items() if int(k) != send}
    if residual in grids:
        grids[send] = grids.pop(residual)
    lists = [[[(send if t[0] == residual else t[0], tuple(t[1])) for t in cu if t[0] != send] for cu in gpu]
             for gpu in plan["lists"]]
    return {**plan, "grids": grids, "lists": lists}


def lup_push_plan(pk, plan: dict, grid: tuple, lup: int = 40, send: int = 41, residual: int = 42) -> dict:
    """A plan of the graph with sum_send (`send`) for the graph without it (demo MPK_LATENT_UP_PUSH: latent_up `lup` pushes its
    rows itself): sum_send's tasks dropped, residual_add renumbered `residual` -> `send`, latent_up's tasks placed again at
    `grid` (regrid_plan, right after moe_experts .. sum_gpus)."""
    grids = {int(k): tuple(v) for k, v in plan["grids"].items() if int(k) not in (lup, send)}
    if residual in grids:
        grids[send] = grids.pop(residual)
    lists = [[[(send if t[0] == residual else t[0], tuple(t[1])) for t in cu if t[0] not in (lup, send)] for cu in gpu]
             for gpu in plan["lists"]]
    return regrid_plan(pk, {**plan, "grids": grids, "lists": lists}, lup, tuple(grid), {36, 37, 38, 39})


def front_plan(pk, plan: dict, router: int, router_grid: tuple, down: int, gate_up: int) -> dict:
    """The plan with the router first everywhere it runs: the router (node `router`) at `router_grid`, one task at the head of
    each of the first CUs; latent_down (`down`) at the head of the CUs after those; the shared gate_up (`gate_up`) only right
    after a router task (so it reads its weights once the router's are read, not at the same time), and not on the CUs of
    topk_route / sum_quant_send (`router` + 1, `down` + 1: the K3 graph's order). The other tasks stay where
    they were (plan_move.DEPS checks no task comes before one it waits for on its CU)."""
    import plan_move
    grids = {int(k): tuple(v) for k, v in plan["grids"].items()}
    grids[router] = tuple(router_grid)
    nodes = compiler.apply_grids(pk, grids)
    nsm = pk._num_sms()
    out = []
    for gpu, old in enumerate(plan["lists"]):
        lists = [[t for t in cu if t[0] not in (router, down, gate_up)] for cu in old]
        tasks = compiler.gpu_tasks(pk, nodes, gpu, pk.num_gpus)
        rt = [(t.node, tuple(t.pos)) for t in tasks if t.node == router]
        dt = [(t.node, tuple(t.pos)) for t in tasks if t.node == down]
        gt = [(t.node, tuple(t.pos)) for t in tasks if t.node == gate_up]
        assert len(rt) < nsm, "front_plan: fewer router tasks than CUs"
        for i, t in enumerate(rt):
            lists[i].insert(0, t)
        rest = list(range(len(rt), nsm))
        for k, t in enumerate(dt):
            cu = lists[rest[k % len(rest)]]
            at = max((j + 1 for j, u in enumerate(cu) if u[0] == down), default=0)
            cu.insert(at, t)
        # gate_up behind a router task, not on the CUs of the small tasks the experts wait for (topk_route, sum_quant_send)
        hosts = [i for i in range(len(rt)) if not any(u[0] in (router + 1, down + 1) for u in lists[i])]
        for k, t in enumerate(gt):
            lists[hosts[k % len(hosts)]].insert(1, t)
        for cu in lists:
            ids = [t[0] for t in cu]
            for j, n in enumerate(ids):
                assert not (plan_move.waits_for(n) & set(ids[j + 1:])), (n, ids)
        out.append(lists)
    return {**plan, "grids": grids, "lists": out}


def shared_early_plan(pk, plan: dict, gate_up: int = 30, act: int = 31, down: int = 34, moe: int = 36) -> dict:
    """The plan with the shared expert (gate_up `gate_up`, situ_and_mul `act`, down `down`) on the CUs whose list starts
    with moe_experts (`moe`), ahead of it: gate_up first on the first of them, situ_and_mul first on the next, down first
    on the rest and then behind those (after the CU's shared tasks); the other tasks stay where they were. (MI355X 10-12,
    TP8 trace: with the shared expert behind the router, down ends at 33 us and its 56 CUs take the last W13 units, while
    ~108 CUs wait in moe_experts from 2 to ~22 us for z_q.)"""
    nodes = compiler.apply_grids(pk, {int(k): tuple(v) for k, v in plan["grids"].items()})
    deps = depends_on(nodes)
    mine = (gate_up, act, down)
    out = []
    for old in plan["lists"]:
        tasks = {n: [t for cu in old for t in cu if t[0] == n] for n in mine}
        lists = [[t for t in cu if t[0] not in mine] for cu in old]
        idle = [i for i, cu in enumerate(lists) if cu and cu[0][0] == moe]
        assert len(idle) >= len(tasks[gate_up]) + len(tasks[act]), "shared_early_plan: too few CUs start with moe_experts"
        for k, t in enumerate(tasks[gate_up]):
            lists[idle[k]].insert(0, t)
        ng = len(tasks[gate_up])
        for k, t in enumerate(tasks[act]):
            lists[idle[ng + k]].insert(0, t)
        hosts = idle[ng + len(tasks[act]):] + idle[ng:ng + len(tasks[act])] + idle[:ng]
        for k, t in enumerate(tasks[down]):
            cu = lists[hosts[k % len(hosts)]]
            cu.insert(max((j + 1 for j, u in enumerate(cu) if u[0] in mine), default=0), t)
        for cu in lists:   # no task before one it waits for
            ids = [t[0] for t in cu]
            for j, n in enumerate(ids):
                assert not (deps.get(n, set()) & set(ids[j + 1:])), (n, ids)
        out.append(lists)
    return {**plan, "lists": out}


def shared_late_plan(pk, plan: dict, act: int = 31, down: int = 34, sar: int = 35, after: int = 37,
                     avoid: tuple = (38, 39)) -> dict:
    """The plan with the shared expert's situ_and_mul (`act`) and down GEMM (`down`), and the all-reduce send of its output
    (`sar`), after moe_experts' R send (`after`) on CUs with none of the `avoid` nodes (the R path after the send: they
    would wait behind the shared tasks): situ_and_mul on the first, down + its send on the next ones; the shared gate_up
    stays where it was. (MI355X 10-12, TP8 trace: the CUs with situ_and_mul and down ahead of moe_experts join it at
    25-33 us, ~8-15 us after z_q; per-CU read speed in W13 is at the bench's best, so moe's speed is the number of CUs
    on it. The shared output is needed only by residual_add, at the end.)"""
    nodes = compiler.apply_grids(pk, {int(k): tuple(v) for k, v in plan["grids"].items()})
    deps = depends_on(nodes)
    mine = (act, down, sar)
    out = []
    for old in plan["lists"]:
        tasks = {n: [t for cu in old for t in cu if t[0] == n] for n in mine}
        lists = [[t for t in cu if t[0] not in mine] for cu in old]
        hosts = [i for i, cu in enumerate(lists)
                 if any(t[0] == after for t in cu) and not any(t[0] in avoid for t in cu)]
        need = len(tasks[act]) + len(tasks[down])
        assert len(hosts) >= need, f"shared_late_plan: {len(hosts)} CUs without {avoid}, need {need}"

        def put(cu, t, behind):
            ids = [u[0] for u in lists[cu]]
            at = max(j + 1 for j, n in enumerate(ids) if n in behind)
            lists[cu].insert(at, t)

        na = len(tasks[act])
        for k, t in enumerate(tasks[act]):
            put(hosts[k], t, (after,))
        dh = hosts[na:need]
        for k, t in enumerate(tasks[down]):
            put(dh[k], t, (after,))
        for k, t in enumerate(tasks[sar]):
            put(dh[k % len(dh)], t, (after, down))
        for cu in lists:   # no task before one it waits for
            ids = [t[0] for t in cu]
            for j, n in enumerate(ids):
                assert not (deps.get(n, set()) & set(ids[j + 1:])), (n, ids)
        out.append(lists)
    return {**plan, "lists": out}


def phase_plan(pk, grids: dict, phases: List[List[int]]) -> dict:
    grids = {int(k): tuple(v) for k, v in (grids or {}).items()}
    nodes = compiler.apply_grids(pk, grids)
    deps = depends_on(nodes)
    named = [i for ph in phases for i in ph]
    rest = [n.graph_idx for n in nodes if n.graph_idx not in named]
    phases = [list(ph) for ph in phases] + ([rest] if rest else [])
    seen: Set[int] = set()
    for ph in phases:   # every producer in an earlier phase, or earlier in the same phase
        for i in ph:
            missing = deps[i] - seen - set(ph[:ph.index(i)])
            if missing:
                raise ValueError(f"phase_plan: node {i} comes before its producers {sorted(missing)}")
        seen |= set(ph)
    by_idx = {n.graph_idx: n for n in nodes}
    nsm = pk._num_sms()
    out = []
    for gpu in range(pk.num_gpus):
        tasks: Dict[int, list] = {}
        for t in compiler.gpu_tasks(pk, nodes, gpu, pk.num_gpus):
            tasks.setdefault(t.node, []).append((t.node, tuple(t.pos)))
        lists = [[] for _ in range(nsm)]
        nxt = 0
        for ph in phases:
            queues = [list(tasks.get(i, [])) for i in ph if not by_idx[i].info["one_task_per_sm"]]
            for i in ph:
                if by_idx[i].info["one_task_per_sm"]:
                    for node, pos in tasks.get(i, []):
                        lists[pos[0]].append((node, pos))
            free = [i for i in ph if not by_idx[i].info["one_task_per_sm"]]
            independent = all(not (deps[i] & set(free)) for i in free)
            # independent nodes: one task of each in turn; otherwise node after node (a consumer dealt among its producer's
            # tasks could sit before one of them on an SM: that SM would wait for itself)
            for group in ([queues] if independent else [[q] for q in queues]):
                while any(group):
                    for q in group:
                        if q:
                            lists[nxt].append(q.pop(0))
                            nxt = (nxt + 1) % nsm
        out.append(lists)
    return {"grids": {n.graph_idx: tuple(n.grid) for n in nodes}, "lists": out}


def spread_plan(pk, plan: dict, grids: dict, spread: List[int]) -> dict:
    """The plan with the nodes in `spread` regridded (grids) and their tasks moved to the front of the SM lists, one per SM: first
    on the SMs whose list starts with a one_task_per_sm node (moe_experts: they only wait at the start), then on the others; every
    other task keeps its place. The spread nodes must depend on no other node (they run first)."""
    grids = {int(k): tuple(v) for k, v in (grids or {}).items()}
    base = {int(k): tuple(v) for k, v in plan["grids"].items()}
    base.update(grids)
    nodes = compiler.apply_grids(pk, base)
    deps = depends_on(nodes)
    for i in spread:
        if deps[i]:
            raise ValueError(f"spread_plan: node {i} depends on {sorted(deps[i])}; only nodes with no producer can go first")
    by_idx = {n.graph_idx: n for n in nodes}
    per_sm = {n.graph_idx for n in nodes if n.info["one_task_per_sm"]}
    out = []
    for gpu in range(pk.num_gpus):
        lists = [[(node, tuple(pos)) for node, pos in sm_list if node not in spread] for sm_list in plan["lists"][gpu]]
        idle = [sm for sm, l in enumerate(lists) if l and l[0][0] in per_sm]
        busy = [sm for sm, l in enumerate(lists) if not (l and l[0][0] in per_sm)]
        order = idle + busy
        tasks = [(t.node, tuple(t.pos)) for t in compiler.gpu_tasks(pk, nodes, gpu, pk.num_gpus) if t.node in spread]
        # interleave the spread nodes' tasks (one of each in turn), then deal them over `order`
        queues = [[t for t in tasks if t[0] == i] for i in spread]
        dealt = []
        while any(queues):
            for q in queues:
                if q:
                    dealt.append(q.pop(0))
        heads = [[] for _ in lists]
        for k, t in enumerate(dealt):
            heads[order[k % len(order)]].append(t)
        out.append([h + l for h, l in zip(heads, lists)])
    res = {"grids": {n.graph_idx: tuple(n.grid) for n in nodes}, "lists": out}
    if "cluster_launch" in plan:
        res["cluster_launch"] = plan["cluster_launch"]
    return res


def token_axes(n) -> set:
    """the grid axes of node n that cut the token dimension (dimension 0) of one of its tensors, per its input maps, where
    its grid has the token count (topk_route x, sum_gpus y, attn_residual x; not a GEMM's K split, which may equal it)"""
    t = n.grid
    maps = n.info["input_maps"]
    if all(m[a] < 0 for m in maps for a in range(3)):   # (a node that reads and writes by pointer, e.g. moe_agg1: task x =
        return {a for a in range(3) if n.inputs and t[a] == compiler.dims(n.inputs[0])[0]}   # token x)
    return {a for tensor, m in zip(n.inputs + n.outputs, maps) for a in range(3)
            if m[a] == 0 and compiler.dims(tensor)[0] == t[a]}


def tokens_grids(pk, grids: dict, t_old: int) -> dict:
    """a plan's grids made for t_old tokens, for the graph pk now has: t_old on a token axis (token_axes) -> pk.tokens"""
    base = {n.graph_idx: n for n in compiler.graph_nodes(pk)}
    return {int(k): tuple(pk.tokens if a in token_axes(base[int(k)]) and d == t_old else d for a, d in enumerate(tuple(v)))
            for k, v in grids.items() if int(k) in base}


def tokens_plan(pk, plan: dict, t_old: int) -> dict:
    """A plan made for t_old tokens per step, for the graph pk now has (pk.tokens). A planned grid with t_old on an axis
    where the node's own grid has the new count gets the new count. Every node whose tasks differ (a grid axis that is the
    token count: topk_route, sum_gpus, moe_agg1, attn_residual, ...): its tasks of tokens >= the new count leave the
    lists; a task of token t >= t_old goes right after the same node's task of token t mod t_old, on that task's list (the
    list context the plan made deadlock-free: a placement by the data dependencies alone can put a task before one that
    waits, through other lists' order, for it)."""
    t_new = pk.tokens
    if t_new == t_old:
        return plan
    base = {n.graph_idx: n for n in compiler.graph_nodes(pk)}   # (the layers' own grids)
    grids = {}
    for k, v in plan["grids"].items():
        ta = token_axes(base[int(k)])
        grids[int(k)] = tuple(t_new if a in ta and d == t_old else d for a, d in enumerate(tuple(v)))
    nodes = compiler.apply_grids(pk, grids)
    out = []
    for gpu, old in enumerate(plan["lists"]):
        lists = [list(cu) for cu in old]
        for n in nodes:
            want = [(t.node, tuple(t.pos)) for t in compiler.gpu_tasks(pk, nodes, gpu, pk.num_gpus) if t.node == n.graph_idx]
            have = {(node, tuple(pos)) for cu in lists for node, pos in cu if node == n.graph_idx}
            if set(want) == have:
                continue
            axes = [a for a in token_axes(n) if n.grid[a] == t_new and have and max(p[a] for _, p in have) + 1 == t_old]
            if len(axes) != 1:   # (e.g. the fused harness's moe_agg1 in residual_add's planned place: a later regrid_plan)
                if gpu == 0:
                    print(f"tokens_plan: node {n.graph_idx} ({n.name}) left as planned (no single token axis {axes}): "
                          f"a later pass must place it", flush=True)
                continue
            ax = axes[0]
            if gpu == 0:
                print(f"tokens_plan: node {n.graph_idx} ({n.name}) re-dealt: {len(have)} -> {len(want)} tasks per GPU", flush=True)
            wset = set(want)
            lists = [[t for t in cu if t[0] != n.graph_idx or (t[0], tuple(t[1])) in wset] for cu in lists]
            for node, pos in want:
                if (node, pos) in have:
                    continue
                src = tuple(p % t_old if a == ax else p for a, p in enumerate(pos))
                ci, j = next((ci, j) for ci, cu in enumerate(lists) for j, t in enumerate(cu) if t[0] == node and tuple(t[1]) == src)
                lists[ci].insert(j + 1, (node, pos))
        out.append(lists)
    return {**plan, "grids": grids, "lists": out}
