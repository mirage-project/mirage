"""The solver of solver.py without per-SM variables: same input, rules and result, but the SMs of a GPU set are one resource of
num_sms units. The SMs are identical, so instead of choosing an SM for every task (on[task, SM]: tasks x SMs yes/no variables,
one AddNoOverlap per SM, and as many equivalent plans as there are ways to number the SMs), the model only requires that at
no time more than num_sms tasks of a GPU set hold an SM (one AddCumulative); every other rule is solver.py's.

The SMs are given after the solve: the used tasks of a GPU set sorted by start time, each onto a free SM (one whose last task
has ended), the free SM with the fewest tasks first. With at most num_sms intervals at any time a free SM always exists (an
interval graph needs no more colours than its largest overlap), so this is exact; each SM's list is its tasks in start order,
so a task starts on its SM no earlier than the model's start and the plan runs no later than predicted.
max_tasks_per_sm is not a model rule here: the assignment checks it.

A pool (solver.py rule 4) is num_sms participation intervals [join_k, end] in the same resource, all ending at the pool's end,
their lengths adding up to its work (sum_k (end - join_k) >= work: solver.py's num_sms x end >= work + sum of joins), each
join_k >= its opening. A task before the pool ends by the pool's end, a task after it starts then. The assignment puts the
participations with a length on different SMs (they all overlap just before the end) and the empty ones on the other SMs, so
every SM gets exactly one, after its own earlier tasks and before its later ones: solver.py's per-SM rule. The joins are
sorted (join_0 <= join_1 <= ...): any order of them is the same plan. One pool per GPU set.

The placement rules of solver.py (8-11):
  8 same_sm (a, b), b right after a on a's SM: when both are used one interval [start a, end b] holds an SM instead of theirs
    (start b >= end a; the pool on the same side of both); the assignment puts it on one SM, a then b. A task in two pairs (a
    chain) raises NeedsPerSM
  9 cluster_pairs: not in the model (the search gives their tasks as concurrent groups); the assignment puts each pair task on
    a free SM that has held none, the other tasks first on the free SMs that have (keeping the others free for later pairs),
    then numbers the SMs so that the k-th pair is on an SM pair 2k, 2k + 1. When a pair task finds no such SM it raises
    NeedsPerSM
  10, 11 same_group, near: each task in one SM group (in[task, group]), and per group one more resource of its SMs (the pool's
    participation k in SM k's group); same_group: the same group, near: the same group or b ends us later. The assignment is
    per group, so it stays exact. sm_group None: solver.default_groups (two dies)
A plan this model returns keeps every rule of solver.py, so its end is solver.py's optimum when it is this model's; NeedsPerSM
means the plan needs solver.py.
"""
import heapq
import time
from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

from .solver import TICK, OptionKey, Pool, Result, SolverTask, default_groups, ticks


class NeedsPerSM(RuntimeError):
    """The placement rules need solver.py: a cluster pair task found no free SM that has held no pair task, or a task is in two
    same_sm pairs (module docstring)."""


def assign_sms(intervals: List[Tuple[tuple, int, int]], num_sms: int, max_tasks_per_sm: int,
               joins: Optional[List[int]] = None, paired: frozenset = frozenset()) -> Tuple[List[List[str]], List[int]]:
    """intervals = [(task ids, start, end)] (ticks; several ids: tasks that run one right after the other on one SM), at most
    num_sms overlapping at any time (with the pool's participations) -> (per SM its task ids in start order, per SM its join
    time). Greedy by start time onto a free SM, the free SM with the
    fewest tasks first. joins: a pool's participations [join, pool end] (all ending at the same time); the ones with a length
    take part in the greedy assignment (not counted in max_tasks_per_sm, not in the lists), the empty ones go to the SMs left
    without one. paired: the cluster pairs' tasks, each onto a free SM that has held none of them (else NeedsPerSM); the
    other items first onto the free SMs that have."""
    lists = [[] for _ in range(num_sms)]
    sm_join = [None] * num_sms
    busy = []                                        # heap of (end, sm) of the SMs running a task
    held = [False] * num_sms                         # the SM has held a pair task
    free = {False: [(0, sm) for sm in range(num_sms)], True: []}   # held -> heap of (tasks so far, sm) of the free SMs
    pool_end = max(joins) if joins else None
    items = [(s, e, 0, tid) for tid, s, e in intervals]
    if joins:
        items += [(j, 0, 1, k) for k, j in enumerate(joins) if j < pool_end]   # kind 1: a participation [j, pool_end]
    for s, e, kind, item in sorted(items, key=lambda x: (x[0], x[1] if x[2] == 0 else pool_end, x[2])):
        if kind == 1:
            e = pool_end
        while busy and busy[0][0] <= s:
            _, sm = heapq.heappop(busy)
            heapq.heappush(free[held[sm]], (len(lists[sm]), sm))
        pair = kind == 0 and any(tid in paired for tid in item)
        heap = free[False] if pair or not free[True] else free[True]
        if not heap:
            if pair and free[True]:
                raise NeedsPerSM(f"solver_cumulative: at time {s} every free SM has held a cluster pair task ({item})")
            raise RuntimeError(f"solver_cumulative: more than {num_sms} tasks at time {s} ({item})")
        n, sm = heapq.heappop(heap)
        held[sm] = held[sm] or pair
        if kind == 1:
            if sm_join[sm] is not None:
                raise RuntimeError(f"solver_cumulative: SM {sm} would join the pool twice")
            sm_join[sm] = s
        else:
            if n + len(item) > max_tasks_per_sm:
                raise RuntimeError(f"solver_cumulative: SM {sm} would get more than {max_tasks_per_sm} tasks")
            lists[sm].extend(item)
        heapq.heappush(busy, (e, sm))
    if joins:   # the empty participations: the SMs without one join at the pool's end
        sm_join = [pool_end if j is None else j for j in sm_join]
    return lists, sm_join


def pair_sms(lists: List[List[str]], sm_join: Optional[List[int]], cluster_pairs: List[Tuple[str, str]],
             slots: List[Tuple[int, int]]):
    """The SM lists (and joins) renumbered so that the k-th cluster pair (a, b) placed in them is on slots[k] (the k-th SM pair
    2j, 2j + 1 among them: a on the first), the other lists after them in their order. Every SM holds at most one pair task
    (assign_sms)."""
    sm_of = {tid: sm for sm, sm_list in enumerate(lists) for tid in sm_list}
    new_of = {}
    for a, b in cluster_pairs:
        if a in sm_of and b in sm_of:
            k = len(new_of) // 2
            if k >= len(slots):
                raise NeedsPerSM(f"solver_cumulative: more cluster pairs than SM pairs ({len(slots)})")
            new_of[sm_of[a]], new_of[sm_of[b]] = slots[k]
    taken = set(new_of.values())
    rest = iter(i for i in range(len(lists)) if i not in taken)
    for sm in range(len(lists)):
        if sm not in new_of:
            new_of[sm] = next(rest)
    out = [None] * len(lists)
    joins = None if sm_join is None else [None] * len(lists)
    for sm, new in new_of.items():
        out[new] = lists[sm]
        if joins is not None:
            joins[new] = sm_join[sm]
    return out, joins


def solve(groups: Dict[str, List[Dict[int, str]]], tasks: List[SolverTask], deps: List[Tuple[str, str]], pools: List[Pool],
          gpu_sets: List[str], num_sms: int, max_tasks_per_sm: int, time_limit_s: float = 60.0, workers: int = 16,
          fixed_options: Optional[Dict[int, str]] = None, fixed_lists: Optional[Dict[str, List[List[str]]]] = None,
          concurrent: List[List[str]] = (), fixed_post: Optional[set] = None, hint: Optional[Result] = None,
          same_sm: List[Tuple[str, str]] = (), cluster_pairs: List[Tuple[str, str]] = (),
          same_group: List[Tuple[str, str]] = (), near: List[Tuple[str, str, float]] = (),
          sm_group: Optional[List[int]] = None) -> Result:
    """solver.solve's interface and result. Rules 1, 3, 5, 6, 7 as there; rule 2 is one cumulative resource of num_sms per
    GPU set (plus, for a given plan, its SM order); rule 4: a pool's participations in that resource; rules 8-11 as in the
    module docstring (NeedsPerSM when the plan needs solver.py)."""
    if len({p.gpu_set for p in pools}) < len(pools):
        raise NotImplementedError("solver_cumulative: more than one pool in a GPU set")
    model = cp_model.CpModel()
    fixed_options = fixed_options or {}
    horizon = ticks(sum(t.duration_us + t.first_extra_us + t.prefetch_us for t in tasks) + sum(p.work_us for p in pools)
                    + sum(us for _, _, us in near) + 300)

    # ---- rule 1: exactly one choice per entry; use[node, option]
    x: Dict[Tuple[str, int], cp_model.IntVar] = {}
    containing: Dict[OptionKey, list] = {}
    for name, choices in groups.items():
        for i, choice in enumerate(choices):
            x[(name, i)] = model.NewBoolVar(f"x_{name}_{i}")
            for node, opt in choice.items():
                containing.setdefault((node, opt), []).append(x[(name, i)])
        model.AddExactlyOne([x[(name, i)] for i in range(len(choices))])
    use: Dict[OptionKey, cp_model.IntVar] = {}
    for (node, opt), xs in containing.items():
        use[(node, opt)] = model.NewBoolVar(f"use_{node}_{opt}")
        model.Add(use[(node, opt)] == sum(xs))
        if node in fixed_options:
            model.Add(use[(node, opt)] == (1 if fixed_options[node] == opt else 0))
    tasks = [t for t in tasks if t.node not in fixed_options or fixed_options[t.node] == t.option]
    by_id = {t.id: t for t in tasks}
    used = {t.id: use[(t.node, t.option)] for t in tasks}

    # ---- rule 2: per GPU set, at most num_sms tasks hold an SM at any time; lengths as in solver.py (rules 6, 7)
    start, end, length, span = {}, {}, {}, {}
    ivs: Dict[str, list] = {g: [] for g in gpu_sets}
    for t in tasks:
        start[t.id] = model.NewIntVar(0, horizon, f"s_{t.id}")
        end[t.id] = model.NewIntVar(0, horizon, f"e_{t.id}")
        if t.duration_by:
            times = {key: ticks(us) for key, us in t.duration_by.items()}
            length[t.id] = model.NewIntVar(min(times.values()), max(times.values()), f"len_{t.id}")
            model.Add(length[t.id] == sum(tm * use[key] for key, tm in times.items())).OnlyEnforceIf(used[t.id])
        elif t.first_extra_us > 0:
            base, extra = ticks(t.duration_us), ticks(t.first_extra_us)
            first = model.NewBoolVar(f"first_{t.id}")
            length[t.id] = model.NewIntVar(base, base + extra, f"len_{t.id}")
            model.Add(length[t.id] == base + extra * first)
            model.Add(start[t.id] >= extra).OnlyEnforceIf([used[t.id], first.Not()])
        else:
            length[t.id] = ticks(t.duration_us)
        if t.prefetch_us > 0:
            span[t.id] = model.NewIntVar(0, horizon, f"span_{t.id}")
            model.Add(span[t.id] >= length[t.id] + ticks(t.prefetch_us))
        else:
            span[t.id] = length[t.id]
        model.Add(end[t.id] == start[t.id] + span[t.id])
        model.Add(start[t.id] == 0).OnlyEnforceIf(used[t.id].Not())

    # ---- rule 8 (same_sm, b right after a): when both are used one interval [start a, end b] holds the SM instead of theirs
    nxt, prv, both_of = {}, {}, {}
    for a, b in same_sm:
        if a in by_id and b in by_id:
            if by_id[a].gpu_set != by_id[b].gpu_set:
                raise ValueError(f"solve: {a} and {b} are in different GPU sets")
            if a in nxt or a in prv or b in nxt or b in prv:
                raise NeedsPerSM(f"solver_cumulative: {a} or {b} is in two same_sm pairs")
            nxt[a], prv[b] = b, a
    holders = []                                     # (task ids, start, span, end, present, GPU set): what holds an SM
    for a, b in nxt.items():
        bo = model.NewBoolVar(f"both_{a}_{b}")
        model.AddBoolAnd([used[a], used[b]]).OnlyEnforceIf(bo)
        model.AddBoolOr([used[a].Not(), used[b].Not()]).OnlyEnforceIf(bo.Not())
        model.Add(start[b] >= end[a]).OnlyEnforceIf(bo)
        holders.append(((a, b), start[a], model.NewIntVar(0, horizon, f"hold_{a}_{b}"), end[b], bo, by_id[a].gpu_set))
        both_of[a] = both_of[b] = bo
    for t in tasks:
        present = used[t.id]
        if t.id in both_of:                          # its own interval only when its pair is not both used
            present = model.NewBoolVar(f"own_{t.id}")
            model.AddBoolAnd([used[t.id], both_of[t.id].Not()]).OnlyEnforceIf(present)
            model.AddBoolOr([used[t.id].Not(), both_of[t.id]]).OnlyEnforceIf(present.Not())
        holders.append(((t.id,), start[t.id], span[t.id], end[t.id], present, t.gpu_set))
    for ids, s, sz, e, present, g in holders:
        ivs[g].append(model.NewOptionalIntervalVar(s, sz, e, present, f"iv_{'_'.join(ids)}"))

    # ---- rules 10, 11 (same_group, near): every task in one SM group, and per group one more resource of its SMs
    grouped = bool(same_group or near)
    names, sms_of, ing = [], {}, {}
    givs: Dict[Tuple[str, int], list] = {}
    if grouped:
        sm_group = sm_group or default_groups(num_sms)
        names = sorted(set(sm_group))
        sms_of = {gr: [sm for sm in range(num_sms) if sm_group[sm] == gr] for gr in names}
        for t in tasks:
            ing[t.id] = {gr: model.NewBoolVar(f"in_{t.id}_{gr}") for gr in names}
            model.Add(sum(ing[t.id].values()) == used[t.id])
        givs = {(g, gr): [] for g in gpu_sets for gr in names}
        for ids, s, sz, e, present, g in holders:
            for gr in names:
                lit = ing[ids[0]][gr]
                if len(ids) > 1 or present is not used[ids[0]]:
                    lit = model.NewBoolVar(f"hold_{'_'.join(ids)}_{gr}")
                    model.AddBoolAnd([present, ing[ids[0]][gr]]).OnlyEnforceIf(lit)
                    model.AddBoolOr([present.Not(), ing[ids[0]][gr].Not()]).OnlyEnforceIf(lit.Not())
                givs[(g, gr)].append(model.NewOptionalIntervalVar(s, sz, e, lit, f"giv_{'_'.join(ids)}_{gr}"))

        for g, per_sm in (fixed_lists or {}).items():   # a given plan: a task is in its SM's group
            for sm, sm_list in enumerate(per_sm):
                for tid in sm_list:
                    if tid in by_id:
                        model.Add(ing[tid][sm_group[sm]] == used[tid])

        def same_group_if(a, b, cond):
            for gr in names:
                model.Add(ing[a][gr] == ing[b][gr]).OnlyEnforceIf(cond)

        for a, b in list(same_group) + list(cluster_pairs):    # a cluster pair is on one SM pair: one group
            if a in by_id and b in by_id:
                same_group_if(a, b, [used[a], used[b]])
        for a, b in nxt.items():
            same_group_if(a, b, [both_of[a]])
        for a, b, us in near:                        # far = 0 only when they share a group
            if a in by_id and b in by_id:
                far = model.NewBoolVar(f"far_{a}_{b}")
                same_group_if(a, b, [used[a], used[b], far.Not()])
                model.Add(end[b] >= end[a] + length[b] + ticks(us)).OnlyEnforceIf([used[a], used[b], far])
    # ---- rule 4: a pool = num_sms participations [join_k, end] in its GPU set's resource
    post = {}                                        # task id -> 1, 0 or a yes/no: runs after the pool on its SM
    pool_vars = {}                                   # pool id -> (opens, joins, ends)
    for p in pools:
        opens = model.NewIntVar(0, horizon, f"open_{p.id}")
        for tid in p.inputs:
            if tid in by_id:
                model.Add(opens >= end[tid]).OnlyEnforceIf(used[tid])
        after, inputs = set(p.after or ()), set(p.inputs)
        for t in tasks:
            if t.gpu_set != p.gpu_set:
                continue
            if t.id in after:
                post[t.id] = 1
            elif t.id in inputs:
                post[t.id] = 0
            elif fixed_post is not None:
                post[t.id] = 1 if t.id in fixed_post else 0
            else:
                post[t.id] = model.NewBoolVar(f"post_{t.id}")
        ends = model.NewIntVar(0, horizon, f"end_{p.id}")
        joins, lens = [], []
        for k in range(num_sms):
            join = model.NewIntVar(0, horizon, f"join_{p.id}_{k}")
            ln = model.NewIntVar(0, horizon, f"plen_{p.id}_{k}")
            model.Add(join >= opens)
            model.Add(ln == ends - join)              # >= 0: every SM joins before the end
            piv = model.NewIntervalVar(join, ln, ends, f"piv_{p.id}_{k}")
            ivs[p.gpu_set].append(piv)
            if grouped:                              # participation k is SM k's, in SM k's group
                givs[(p.gpu_set, sm_group[k])].append(piv)
            joins.append(join)
            lens.append(ln)
        model.Add(sum(lens) >= ticks(p.work_us))
        if fixed_lists is None:                      # the participations are interchangeable (within a group)
            for ks in (sms_of.values() if grouped else [range(num_sms)]):
                for a, b in zip(list(ks), list(ks)[1:]):
                    model.Add(joins[a] <= joins[b])
        else:                                        # a given plan: SM k joins after its own tasks before the pool
            for k, sm_list in enumerate(fixed_lists.get(p.gpu_set, [])):
                for tid in sm_list:
                    v = post.get(tid)
                    if tid in by_id and v is not None and not (isinstance(v, int) and v == 1):
                        model.Add(joins[k] >= end[tid]).OnlyEnforceIf([used[tid]] + ([] if isinstance(v, int) else [v.Not()]))
        for tid, v in post.items():
            if isinstance(v, int):
                if v == 1:
                    model.Add(start[tid] >= ends).OnlyEnforceIf(used[tid])
                else:
                    model.Add(end[tid] <= ends).OnlyEnforceIf(used[tid])
            else:
                model.Add(start[tid] >= ends).OnlyEnforceIf([used[tid], v])
                model.Add(end[tid] <= ends).OnlyEnforceIf([used[tid], v.Not()])
        pool_vars[p.id] = (opens, joins, ends)
    for g, iv in ivs.items():
        if iv:
            model.AddCumulative(iv, [1] * len(iv), num_sms)
    for (g, gr), iv in givs.items():
        if len(iv) > len(sms_of[gr]):
            model.AddCumulative(iv, [1] * len(iv), len(sms_of[gr]))
    for a, b in nxt.items():                         # rule 8: the pool is not between a and b
        pa, pb = post.get(a), post.get(b)
        if pa is not None and pb is not None:
            if not (isinstance(pa, int) and isinstance(pb, int)):
                model.Add(pa == pb).OnlyEnforceIf(both_of[a])
            elif pa != pb:
                model.Add(both_of[a] == 0)
    for per_sm in (fixed_lists or {}).values():      # a given plan: each SM runs its tasks in the given order
        for sm_list in per_sm:
            for a, b in zip(sm_list, sm_list[1:]):
                if a in by_id and b in by_id:
                    model.Add(start[b] >= end[a])

    # ---- rule 3: a consumer ends at least its length after the producer ends
    for src, dst in deps:
        if src in by_id and dst in by_id:
            model.Add(end[dst] >= end[src] + length[dst]).OnlyEnforceIf([used[src], used[dst]])

    # ---- rule 5: a concurrent group starts together (different SMs: they overlap, so the assignment separates them)
    for group in concurrent:
        members = [tid for tid in group if tid in by_id]
        for a, b in zip(members, members[1:]):
            model.Add(start[b] == start[a]).OnlyEnforceIf([used[a], used[b]])

    C = model.NewIntVar(0, horizon, "C")
    for t in tasks:
        model.Add(C >= end[t.id]).OnlyEnforceIf(used[t.id])
    for _, _, ends in pool_vars.values():
        model.Add(C >= ends)
    model.Minimize(C)

    if hint is not None:
        for (name, i), v in x.items():
            model.AddHint(v, int(all(hint.chosen.get(node) == opt for node, opt in groups[name][i].items())))
        for t in tasks:
            model.AddHint(start[t.id], ticks(hint.times[t.id][0]) if t.id in hint.times else 0)
        for pid, (_, joins, _) in pool_vars.items():
            if hint.pools and pid in hint.pools:
                for v, j in zip(joins, sorted(hint.pools[pid]["joins"]) if fixed_lists is None else hint.pools[pid]["joins"]):
                    model.AddHint(v, ticks(j))
        model.AddHint(C, ticks(hint.predicted_us))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    t0 = time.time()
    status = solver.Solve(model)
    solve_s = round(time.time() - t0, 1)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Result(solver.StatusName(status), float("nan"), float("nan"), solve_s, {}, {})
    chosen = {node: opt for (node, opt), v in use.items() if solver.Value(v)}
    live = [t for t in tasks if solver.Value(used[t.id])]
    us = lambda v: round(v * TICK, 3)
    pool_of = {p.gpu_set: p.id for p in pools}
    lists, pool_times = {}, {}
    for g in gpu_sets:
        pid = pool_of.get(g)
        joins = [solver.Value(j) for j in pool_vars[pid][1]] if pid else None
        if fixed_lists:
            lists[g] = [[tid for tid in sm_list if tid in by_id] for sm_list in fixed_lists.get(g, [])]
            sm_join = joins
        else:   # per group (all SMs: one group), the holders onto its SMs, then the cluster pairs onto its SM pairs
            items = [(ids, solver.Value(s), solver.Value(e)) for ids, s, _, e, present, hg in holders
                     if hg == g and solver.Value(present)]
            paired = frozenset(t for ab in cluster_pairs for t in ab if t in by_id and by_id[t].gpu_set == g)
            lists[g], sm_join = [None] * num_sms, ([None] * num_sms if joins else None)
            for gr, sms in (sms_of.items() if grouped else [(None, list(range(num_sms)))]):
                mine = [it for it in items if gr is None or solver.Value(ing[it[0][0]][gr])]
                local, local_join = assign_sms(mine, len(sms), max_tasks_per_sm, [joins[k] for k in sms] if joins else None,
                                               paired)
                if paired:
                    slots = [(i, i + 1) for i in range(len(sms) - 1) if sms[i] % 2 == 0 and sms[i + 1] == sms[i] + 1]
                    local, local_join = pair_sms(local, local_join, [ab for ab in cluster_pairs if ab[0] in paired], slots)
                for i, sm in enumerate(sms):
                    lists[g][sm] = local[i]
                    if joins:
                        sm_join[sm] = local_join[i]
        if pid:
            o, _, e = pool_vars[pid]
            pool_times[pid] = {"open": us(solver.Value(o)), "joins": [us(j) for j in sm_join], "end": us(solver.Value(e))}
    times = {t.id: (us(solver.Value(start[t.id])), us(solver.Value(end[t.id]))) for t in live}
    post_ids = {tid for tid, v in post.items() if solver.Value(used[tid]) and (v == 1 if isinstance(v, int) else solver.Value(v))}
    return Result(solver.StatusName(status), us(solver.Value(C)), us(solver.BestObjectiveBound()), solve_s, chosen, lists, times,
                  post_ids, pool_times)
