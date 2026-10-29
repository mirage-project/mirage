"""The solver: in one problem, choose the grid of every node that has several (MoE: the K parts of each gemm_tile node), and
give every task of the chosen grids an SM and a start time, so that the layer ends as early as possible (split and placement
chosen jointly).

It uses CP-SAT (Google OR-Tools): we describe variables (yes/no or integers) and rules between them, and one number to make as
small as possible; CP-SAT returns values for all variables that obey every rule. Nothing here is specific to a model: search.py
builds the input from the graph.

The input (built by search.search_plan), with the MoE numbers:
  groups     {entry name: its choices}; one choice = {node: option}; exactly one choice per entry is used. search.py gives
             each node its own entry: "26": [{26: "7x4x1"}, {26: "7x7x1"}, {26: "7x8x1"}, {26: "7x14x1"}] = the router (node 26)
             in 4, 7, 8 or 14 K parts; a node with nothing to choose has one: "27": [{27: "8x1x1"}] (topk_route)
  tasks      one SolverTask per task of every option: id "gpus_0_2_4_6:26:7x14x1:2.5.0" = GPUs 0, 2, 4, 6; node 26; option
             7x14x1; task (2, 5, 0). A task exists in the plan only if its option is chosen.
  deps       (producer task id, consumer task id): the consumer reads data the producer writes
  pools      a one-task-per-SM node (MoE: moe_experts), one per GPU set: its work is shared by the SMs at run time, so it is not placed task by task
  gpu_sets   ["gpus_0_2_4_6", "gpus_1_3_5_7"]: each set of GPUs with the same work gets its own copy of the 148 SMs
  num_sms    148;  max_tasks_per_sm  62 (static_schedule.MAX_TASKS_PER_SM = 64 entries, minus the end entry and the pool's task)
  fixed_options, fixed_lists   a given plan (e.g. the hand plan): its options and its per-SM lists; the solver then only
             computes its start times and its end (to compare it with a searched plan)
  same_sm, cluster_pairs, same_group, near, sm_group   where some tasks must run relative to others (rules 8-11); every other
             task may go on any SM, and the SMs nothing tells apart are interchangeable (aid B)

The variables (time in units of TICK = 10 ns, so all times are integers):
  x[entry, i]              yes/no: the entry uses its i-th choice
  use[node, option]        yes/no: the node uses this option (= the x of the choice that contains it)
  on[task, gpu set, SM]    yes/no: the task runs on this SM
  start[task], end[task]   when the task starts and ends on its SM
  C                        when the layer ends; made as small as possible
The rules (numbered as in the code), then the goal:
  1. each entry uses exactly one of its choices
  2. a task of a used option runs on exactly one SM of its GPU set; a task of an unused option on none; two tasks on the same
     SM never overlap in time; at most max_tasks_per_sm tasks on an SM
  3. a task ends at least its duration after every task it depends on has ended (applied only when both are used)
  4. a pool (moe_experts) opens when its input tasks have ended; each SM joins it when its own tasks before it are done (and it is
     open); the SMs share its work, so it ends when 148 x end >= work + the sum of the join times; the tasks after it start when
     it has ended. A task runs after it when it reads its output (or reads a task that does: allreduce_send and the layer's last
     steps); any other task (e.g. sum_gpus, which reads S from the shared-down node) may run before or after it on its SM, and
     the solver chooses (post[task]); a given plan fixes it by the task's list position
  5. the tasks of a concurrent group (they wait for each other while they run) start at the same time, on different SMs
  6. a task with a first-on-SM time (MoE: a GEMM tile; nothing before it on its SM hides its first loads) takes first_extra_us
     more when it starts before first_extra_us: first[task] = 1 adds it to the length; first[task] = 0 needs start >=
     first_extra_us. A task can start that early only as the first task of its SM, as long as the first tasks of the SMs are
     such tasks themselves (MoE: only the GEMM tiles have no inputs from other tasks); starting late on purpose gains nothing
  7. a task with prefetch_us holds its SM from its start: it ends at least prefetch_us + its duration after its start, and
     (rule 3) at least its duration after its inputs: what it loads while it waits is off its time once the inputs are there
  8. same_sm (a, b): b runs right after a on a's SM (b reads what a leaves in the SM, e.g. its shared memory): no task, and
     not the pool, runs on that SM between a's end and b's start
  9. cluster_pairs (a, b): a on SM 2k, b on SM 2k + 1: the 2 CTAs of one cluster (a cluster launch keeps them in one GPC);
     an SM holds at most one task of all the pairs (compile_plan.pair_clusters: a CTA holds one task of one pair)
 10. same_group (a, b): a and b on SMs of the same group, sm_group[SM] (e.g. its die; default: the two dies, SMs
     0 .. num_sms / 2 - 1 and the rest, default_groups)
 11. near (a, b, us): b reads a; when they are on SMs of different groups, b ends us later (the hand-off's extra time)
The goal: C >= the end of every used task and of every pool; minimise C.
Two aids that change no plan's end, only how fast the solver finds and proves the best one:
  A. at no time more than num_sms tasks (and pool shares) of a GPU set hold an SM: implied by the SMs' AddNoOverlap, stated
     once more as one AddCumulative, which gives the solver a much better lower bound
  B. the SMs nothing tells apart are interchangeable: any plan stays a plan with its SMs renumbered. Units (an SM, or an SM pair
     when there are cluster pairs) of the same groups form a class; in each class the units are numbered in the order the
     tasks (in the input's order) first use them: task i's unit index is at most 1 + the largest among tasks 0 .. i - 1. One
     plan per renumbering class is left, instead of up to 148! copies. With cluster pairs and no groups, each pair takes an SM
     pair of its own (rule 9) and the SM pairs are interchangeable, so the pairs go in the first SM pairs (as many as the
     most pairs a plan can use) and the other SMs are units one by one. A hint is renumbered the same way first
"""
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ortools.sat.python import cp_model

TICK = 0.01                      # us per time unit of the model (10 ns)
OptionKey = Tuple[int, str]      # (node graph_idx, option name), e.g. (30, "7x14x1")


@dataclass
class SolverTask:
    id: str                      # e.g. "gpus_0_2_4_6:28:7x14x1:2.5.0"
    node: int                    # graph_idx of its node
    option: str                  # the option this task belongs to
    gpu_set: str                 # "gpus_0_2_4_6" or "gpus_1_3_5_7"
    duration_us: float           # its measured time on an SM (with duration_by: the longest of those times)
    # when its time depends on another node's option: {(that node, its option): us}. Route task t: {(30, "7x4x1"): ..., ...,
    # (30, "7x14x1"): ...} = topk_route's time when the router uses 4 ... 14 K parts (it adds that many partial sums).
    duration_by: Optional[Dict[OptionKey, float]] = None
    first_extra_us: float = 0.0  # its time first on its SM minus duration_us (rule 6); 0: the same wherever it is
    prefetch_us: float = 0.0     # its loads that need no input (a GEMM's first weight stages), done from when it starts on its
                                 # SM, also while its inputs are on their way (rule 7); 0: none


@dataclass
class Pool:
    """A pool (MoE: moe_experts) of one GPU set: work the SMs share at run time."""
    id: str                      # e.g. "38@gpus_0_2_4_6"
    gpu_set: str
    work_us: float               # its total work, summed over all SMs (measured: about 7500 us)
    inputs: List[str]            # tasks that must end before any SM can start on it (topk_route, sum_quant_send, situ_and_mul tasks)
    after: List[str] = None      # placed tasks that run after it (allreduce_send reads its output, and everything after allreduce_send)


@dataclass
class Result:
    status: str                  # "OPTIMAL" (proven best), "FEASIBLE" (best found before the time limit), or a failure
    predicted_us: float          # the plan's end time C
    bound_us: float              # no plan can end earlier than this
    solve_s: float
    chosen: Dict[int, str]       # node graph_idx -> its chosen option
    lists: Dict[str, List[List[str]]]      # GPU set -> per SM, the task ids in start order
    times: Dict[str, Tuple[float, float]] = None    # used task id -> (start, end) in us (a hint for a later solve)
    post: set = None                                 # used task ids that run after the pool on their SM
    pools: Dict[str, dict] = None                    # pool id -> {"open", "joins" (per SM), "end"} in us


def ticks(us: float) -> int:
    """us -> model time units (10 ns), rounded: 1.7 us -> 170."""
    return int(round(us / TICK))


def renumber(per_sm: List[List[str]], order: List[str], units: List[tuple], classes: Dict[tuple, List[int]]) -> List[List[str]]:
    """One GPU set's per-SM lists with the units of each class renumbered as aid B wants them: in the order the tasks (`order`)
    first use them, the units no task uses after those. An SM keeps its place in its unit. The same plan, other SM numbers."""
    unit_of = {sm: (u, i) for u, sms in enumerate(units) for i, sm in enumerate(sms)}
    sm_of = {tid: sm for sm, sm_list in enumerate(per_sm) for tid in sm_list}
    new_unit = {}
    for cls in classes.values():
        first = []
        for tid in order:
            if tid in sm_of and sm_of[tid] in unit_of:
                u = unit_of[sm_of[tid]][0]
                if u in cls and u not in first:
                    first.append(u)
        for old, new in zip(first + [u for u in cls if u not in first], cls):
            new_unit[old] = new
    out = [[] for _ in per_sm]
    for sm, sm_list in enumerate(per_sm):
        if sm in unit_of and unit_of[sm][0] in new_unit:
            u, i = unit_of[sm]
            out[units[new_unit[u]][i]] = list(sm_list)
        else:
            out[sm] = list(sm_list)
    return out


def default_groups(num_sms: int) -> List[int]:
    """The SM groups when none are given: the two dies of a B200, the first half of the SMs and the second. The plan's SM k is CTA
    k: these groups are dies only when the runtime gives CTA k the task list of physical SM k (its %smid)."""
    return [sm * 2 // num_sms for sm in range(num_sms)]


def solve(groups: Dict[str, List[Dict[int, str]]], tasks: List[SolverTask], deps: List[Tuple[str, str]], pools: List[Pool],
          gpu_sets: List[str], num_sms: int, max_tasks_per_sm: int, time_limit_s: float = 60.0, workers: int = 16,
          fixed_options: Optional[Dict[int, str]] = None, fixed_lists: Optional[Dict[str, List[List[str]]]] = None,
          concurrent: List[List[str]] = (), fixed_post: Optional[set] = None, hint: Optional["Result"] = None,
          same_sm: List[Tuple[str, str]] = (), cluster_pairs: List[Tuple[str, str]] = (),
          same_group: List[Tuple[str, str]] = (), near: List[Tuple[str, str, float]] = (),
          sm_group: Optional[List[int]] = None, capacity: bool = True, break_symmetry: bool = True) -> Result:
    """Build the model (rules 1-11, the goal and the aids above), solve it for at most time_limit_s seconds with `workers` threads,
    read the plan. hint: a solved plan (e.g. a given plan's Result) the solver starts from: its options, SMs, start times and end.
    capacity / break_symmetry: aids A / B (off: the same plans, for comparing)."""
    model = cp_model.CpModel()
    fixed_options = fixed_options or {}
    if (same_group or near) and sm_group is None:
        sm_group = default_groups(num_sms)
    # the latest time any variable may take: all durations one after another + the queue's work spread over the SMs + 3 us
    horizon = ticks(sum(t.duration_us + t.first_extra_us + t.prefetch_us for t in tasks) + sum(p.work_us for p in pools) / num_sms
                    + sum(us for _, _, us in near) + 300)

    # ---- rule 1: exactly one choice per entry; use[node, option] = 1 when the chosen choice contains that option.
    # Entry "30" with 4 choices: x[("30", 0..3)], exactly one of them is 1; use[(30, "7x14x1")] = x[("30", 3)], and so on.
    x: Dict[Tuple[str, int], cp_model.IntVar] = {}
    containing: Dict[OptionKey, list] = {}           # (node, option) -> the x of the choices that contain it
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
        if node in fixed_options:                    # a given plan: this node's option is fixed
            model.Add(use[(node, opt)] == (1 if fixed_options[node] == opt else 0))

    # with fixed options, the tasks of the other options can never run: leave them out
    tasks = [t for t in tasks if t.node not in fixed_options or fixed_options[t.node] == t.option]
    by_id = {t.id: t for t in tasks}
    used = {t.id: use[(t.node, t.option)] for t in tasks}    # task id -> "its option is used"

    # a given plan: the SM of each task
    fixed_sm = {tid: sm for per_sm in (fixed_lists or {}).values() for sm, sm_list in enumerate(per_sm) for tid in sm_list}

    # ---- rule 2: each used task on exactly one SM of its GPU set; tasks on one SM do not overlap; at most max_tasks_per_sm.
    # Per task and SM a yes/no on[...] and an interval [start, end) that exists only when on[...] = 1; AddNoOverlap over the
    # intervals of one SM = no two tasks on that SM at the same time.
    start, end, length = {}, {}, {}
    span = {}                                        # task id -> its time on its SM: length, or (rule 7) more
    on: Dict[Tuple[str, str, int], cp_model.IntVar] = {}             # (task id, GPU set, SM) -> yes/no
    intervals: Dict[Tuple[str, int], list] = {(g, sm): [] for g in gpu_sets for sm in range(num_sms)}
    for t in tasks:
        start[t.id] = model.NewIntVar(0, horizon, f"s_{t.id}")
        end[t.id] = model.NewIntVar(0, horizon, f"e_{t.id}")
        if t.duration_by:
            # the length depends on another node's option: the sum over its options of use[option] x the time for it
            # (exactly one use is 1, so it is the time for the chosen option). Only for a used task: a combination with no
            # measured time (the other node's chosen option not in duration_by) then rules out using this task's option
            times = {key: ticks(us) for key, us in t.duration_by.items()}
            length[t.id] = model.NewIntVar(min(times.values()), max(times.values()), f"len_{t.id}")
            model.Add(length[t.id] == sum(tm * use[key] for key, tm in times.items())).OnlyEnforceIf(used[t.id])
        elif t.first_extra_us > 0:
            # rule 6: first[task] = 1 adds the extra; without it the task must start after the extra (so not first on its SM)
            base, extra = ticks(t.duration_us), ticks(t.first_extra_us)
            first = model.NewBoolVar(f"first_{t.id}")
            length[t.id] = model.NewIntVar(base, base + extra, f"len_{t.id}")
            model.Add(length[t.id] == base + extra * first)
            model.Add(start[t.id] >= extra).OnlyEnforceIf([used[t.id], first.Not()])
        else:
            length[t.id] = ticks(t.duration_us)
        if t.prefetch_us > 0:   # rule 7
            span[t.id] = model.NewIntVar(0, horizon, f"span_{t.id}")
            model.Add(span[t.id] >= length[t.id] + ticks(t.prefetch_us))
        else:
            span[t.id] = length[t.id]
        model.Add(end[t.id] == start[t.id] + span[t.id])
        sms = [fixed_sm[t.id]] if t.id in fixed_sm else range(num_sms)
        for sm in sms:
            b = model.NewBoolVar(f"on_{t.id}_{sm}")
            on[(t.id, t.gpu_set, sm)] = b
            intervals[(t.gpu_set, sm)].append(
                model.NewOptionalIntervalVar(start[t.id], span[t.id], end[t.id], b, f"iv_{t.id}_{sm}"))
        model.Add(sum(on[(t.id, t.gpu_set, sm)] for sm in sms) == used[t.id])   # used: exactly one SM; unused: none
        model.Add(start[t.id] == 0).OnlyEnforceIf(used[t.id].Not())             # unused tasks: start 0, out of the way
    on_sm: Dict[Tuple[str, int], list] = {}          # (GPU set, SM) -> [(task id, on[...])]
    for (tid, g, sm), b in on.items():
        on_sm.setdefault((g, sm), []).append((tid, b))
    for key in intervals:
        if len(on_sm.get(key, [])) > max_tasks_per_sm:
            model.Add(sum(b for _, b in on_sm[key]) <= max_tasks_per_sm)
    for per_sm in (fixed_lists or {}).values():      # a given plan: each SM runs its tasks in the given order
        for sm_list in per_sm:
            for a, b in zip(sm_list, sm_list[1:]):
                if a in by_id and b in by_id:
                    model.Add(start[b] >= end[a])

    # ---- rule 3: a consumer ends at least its length after the producer ends (when both are used).
    # Route task t with the router in 14 parts: after all 98 router tasks of option 7x14x1 (the other options' tasks are unused).
    for src, dst in deps:
        if src in by_id and dst in by_id:
            model.Add(end[dst] >= end[src] + length[dst]).OnlyEnforceIf([used[src], used[dst]])

    # ---- rule 4: a pool (its work is shared by the SMs at run time; MoE: moe_experts); the goal: the layer's end C
    C = model.NewIntVar(0, horizon, "C")
    post = {}                                        # task id -> 1, 0 or a yes/no: runs after the pool on its SM
    pool_vars = {}                                   # pool id -> (opens, joins, ends)
    for p in pools:
        opens = model.NewIntVar(0, horizon, f"open_{p.id}")           # the pool can start once its inputs have ended
        for tid in p.inputs:
            if tid in by_id:
                model.Add(opens >= end[tid]).OnlyEnforceIf(used[tid])
        after = set(p.after or ())
        # post[tid]: the task runs after the pool on its SM. 1 for the tasks in after (they read its output); 0 for its inputs;
        # a given plan: 1 for the tasks listed after the pool entry; otherwise the solver chooses
        inputs = set(p.inputs)
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
        joins = []
        for sm in range(num_sms):                    # SM k joins once the pool can start and SM k's own earlier tasks have ended
            join = model.NewIntVar(0, horizon, f"join_{p.id}_{sm}")
            model.Add(join >= opens)
            for tid, b in on_sm.get((p.gpu_set, sm), []):
                v = post[tid]
                if not isinstance(v, int):
                    model.Add(join >= end[tid]).OnlyEnforceIf([b, v.Not()])
                elif v == 0:
                    model.Add(join >= end[tid]).OnlyEnforceIf(b)
            joins.append(join)
        ends = model.NewIntVar(0, horizon, f"end_{p.id}")
        work = ticks(p.work_us)
        # SM k works on the pool from join_k to the end: the work done = sum over SMs of (end - join_k)
        # = 148 x end - sum of joins, which must be at least the pool's work W. Example, 2 SMs, W = 10 us, joins 0 and 4:
        # (end - 0) + (end - 4) = 10 -> end = 7
        model.Add(num_sms * ends >= work + sum(joins))
        model.Add(ends >= opens + work // num_sms)
        # and it ends only after every SM has joined: the tasks after it wait for every SM's pool task to finish (MoE:
        # allreduce_send waits until C_EXPDONE = 148), and an SM that joins late still runs its pool task (it finds no items left)
        for join in joins:
            model.Add(ends >= join)
        model.Add(C >= ends)
        pool_vars[p.id] = (opens, joins, ends)
        for tid, v in post.items():                  # the tasks after it start when it has ended
            if by_id[tid].gpu_set != p.gpu_set or (isinstance(v, int) and v == 0):
                continue
            model.Add(start[tid] >= ends).OnlyEnforceIf([used[tid]] if isinstance(v, int) else [used[tid], v])
    for t in tasks:
        model.Add(C >= end[t.id]).OnlyEnforceIf(used[t.id])

    # ---- rule 5: a concurrent group starts together, and at most one of its tasks is on an SM
    for group in concurrent:
        members = [tid for tid in group if tid in by_id]
        for a, b in zip(members, members[1:]):
            model.Add(start[b] == start[a]).OnlyEnforceIf([used[a], used[b]])
        for g in gpu_sets:
            for sm in range(num_sms):
                bs = [on[(tid, g, sm)] for tid in members if (tid, g, sm) in on]
                if len(bs) > 1:
                    model.Add(sum(bs) <= 1)

    # ---- rules 8-11: where some tasks run relative to others
    def sm_var(tid, sm):   # on[...] of the task on this SM of its GPU set; 0 when it cannot be there (a given plan's other SMs)
        return on.get((tid, by_id[tid].gpu_set, sm), 0)

    def both(a, b):
        if by_id[a].gpu_set != by_id[b].gpu_set:
            raise ValueError(f"solve: {a} and {b} are in different GPU sets")
        return [used[a], used[b]]

    def same(x, y, cond):  # x == y when cond; two constants (a given plan's other SMs: both 0) need no rule
        if not (isinstance(x, int) and isinstance(y, int)):
            model.Add(x == y).OnlyEnforceIf(cond)

    for a, b in same_sm:                             # rule 8: the same SM, and on it the gap [end a, start b] holds the SM
        if a in by_id and b in by_id:
            g = by_id[a].gpu_set
            for sm in range(num_sms):
                same(sm_var(a, sm), sm_var(b, sm), both(a, b))
            gap = model.NewIntVar(0, horizon, f"gap_{a}_{b}")
            for sm in range(num_sms):
                if (a, g, sm) in on:
                    held = model.NewBoolVar(f"held_{a}_{b}_{sm}")   # on[a, sm] and b used
                    model.AddBoolAnd([on[(a, g, sm)], used[b]]).OnlyEnforceIf(held)
                    model.AddBoolOr([on[(a, g, sm)].Not(), used[b].Not()]).OnlyEnforceIf(held.Not())
                    intervals[(g, sm)].append(model.NewOptionalIntervalVar(end[a], gap, start[b], held, f"gap_{a}_{b}_{sm}"))
            pa, pb = post.get(a), post.get(b)        # the pool is not between them either
            if pa is not None and pb is not None:
                if not (isinstance(pa, int) and isinstance(pb, int)):
                    model.Add(pa == pb).OnlyEnforceIf(both(a, b))
                elif pa != pb:
                    model.AddBoolOr([used[a].Not(), used[b].Not()])
    for a, b in cluster_pairs:                       # rule 9
        if a in by_id and b in by_id:
            for k in range(num_sms // 2):
                same(sm_var(a, 2 * k), sm_var(b, 2 * k + 1), both(a, b))
            wrong = [v for v in [sm_var(a, sm) for sm in range(1, num_sms, 2)] + [sm_var(b, sm) for sm in range(0, num_sms, 2)]
                     if not isinstance(v, int)]
            if wrong:
                model.Add(sum(wrong) == 0)
    paired = sorted({t for ab in cluster_pairs for t in ab if t in by_id})
    for g in gpu_sets:
        for sm in range(num_sms):
            bs = [v for v in (sm_var(t, sm) for t in paired if by_id[t].gpu_set == g) if not isinstance(v, int)]
            if len(bs) > 1:
                model.Add(sum(bs) <= 1)
    for key, ivs in intervals.items():               # rule 2 (with rule 8's gaps): no two on one SM at a time
        if ivs:
            model.AddNoOverlap(ivs)
    group = {}                                       # task id -> the group of its SM (rules 10, 11)

    def group_of(tid):
        if tid not in group:
            group[tid] = model.NewIntVar(0, max(sm_group), f"grp_{tid}")
            model.Add(group[tid] == sum(sm_group[sm] * sm_var(tid, sm) for sm in range(num_sms)))
        return group[tid]

    for a, b in same_group:                          # rule 10
        if a in by_id and b in by_id:
            model.Add(group_of(a) == group_of(b)).OnlyEnforceIf(both(a, b))
    for a, b, us in near:                            # rule 11: far = 0 only when they share a group
        if a in by_id and b in by_id:
            far = model.NewBoolVar(f"far_{a}_{b}")
            model.Add(group_of(a) == group_of(b)).OnlyEnforceIf(both(a, b) + [far.Not()])
            model.Add(end[b] >= end[a] + length[b] + ticks(us)).OnlyEnforceIf(both(a, b) + [far])

    # ---- aid A: at most num_sms tasks and pool shares of a GPU set at a time (one AddCumulative per GPU set)
    if capacity:
        held = {g: [] for g in gpu_sets}
        for t in tasks:
            held[t.gpu_set].append(model.NewOptionalIntervalVar(start[t.id], span[t.id], end[t.id], used[t.id], f"cap_{t.id}"))
        for p in pools:                              # SM k's share of the pool: from its join to the pool's end
            _, joins, ends = pool_vars[p.id]
            for k, j in enumerate(joins):
                share = model.NewIntVar(0, horizon, f"share_{p.id}_{k}")
                model.Add(share == ends - j)
                held[p.gpu_set].append(model.NewIntervalVar(j, share, ends, f"cap_{p.id}_{k}"))
        for g, ivs in held.items():
            if len(ivs) > num_sms:
                model.AddCumulative(ivs, [1] * len(ivs), num_sms)

    # ---- aid B: the units of a class numbered in the order the tasks first use them
    sm_pairs = [(2 * k, 2 * k + 1) for k in range(num_sms // 2)]
    pairs_first = bool(cluster_pairs) and not sm_group and break_symmetry and not fixed_sm
    units: Dict[str, List[tuple]] = {}               # GPU set -> its units
    classes: Dict[str, Dict[tuple, List[int]]] = {}  # GPU set -> (unit size, the groups of its SMs) -> its units (indices)
    for g in gpu_sets:
        if pairs_first:
            most: Dict[int, int] = {}                # node -> the most pairs one of its options has
            per_option: Dict[Tuple[int, str], int] = {}
            for a, _ in cluster_pairs:
                if a in by_id and by_id[a].gpu_set == g:
                    k = (by_id[a].node, by_id[a].option)
                    per_option[k] = per_option.get(k, 0) + 1
                    most[k[0]] = max(most.get(k[0], 0), per_option[k])
            n = min(len(sm_pairs), sum(most.values()))
            units[g] = sm_pairs[:n] + [(sm,) for sm in range(2 * n, num_sms)]
            for t in paired:                         # the pairs in the first n SM pairs
                if by_id[t].gpu_set == g:
                    rest = [v for v in (sm_var(t, sm) for sm in range(2 * n, num_sms)) if not isinstance(v, int)]
                    if rest:
                        model.Add(sum(rest) == 0)
        else:
            units[g] = sm_pairs if cluster_pairs else [(sm,) for sm in range(num_sms)]
        classes[g] = {}
        for u, sms in enumerate(units[g]):
            classes[g].setdefault((len(sms),) + (tuple(sm_group[sm] for sm in sms) if sm_group else ()), []).append(u)
    order = {g: [t.id for t in tasks if t.gpu_set == g] for g in gpu_sets}   # the input's order
    if break_symmetry and not fixed_sm:
        for g in gpu_sets:
            for key, cls in classes[g].items():
                if len(cls) < 2:
                    continue
                prev = model.NewConstant(0)
                for tid in order[g]:
                    # its unit's index in the class, from 1; 0: not in this class (or not used)
                    idx = model.NewIntVar(0, len(cls), f"unit_{tid}_{key}")
                    model.Add(idx == sum((i + 1) * on[(tid, g, sm)] for i, u in enumerate(cls) for sm in units[g][u]))
                    model.Add(idx <= prev + 1)
                    top = model.NewIntVar(0, len(cls), f"top_{tid}_{key}")
                    model.AddMaxEquality(top, [prev, idx])
                    prev = top
    model.Minimize(C)

    # ---- a start: the hinted plan's choices, SMs, start times and end (CP-SAT completes the variables not hinted)
    if hint is not None:
        for (name, i), v in x.items():
            model.AddHint(v, int(all(hint.chosen.get(node) == opt for node, opt in groups[name][i].items())))
        lists = hint.lists
        if break_symmetry and not fixed_sm:          # aid B: the hint's units renumbered in the order its tasks first use them
            def renumbered(g, per_sm):
                if pairs_first:                      # first the SM pairs holding pairs to the front
                    first = [t for t in paired if by_id[t].gpu_set == g]
                    seen = set(first)
                    per_sm = renumber(per_sm, first + [t for t in order[g] if t not in seen], sm_pairs,
                                      {(): list(range(len(sm_pairs)))})
                return renumber(per_sm, order[g], units[g], classes[g])
            lists = {g: renumbered(g, per_sm) for g, per_sm in hint.lists.items() if g in units}
        where = {tid: (g, sm) for g, per_sm in lists.items() for sm, sm_list in enumerate(per_sm) for tid in sm_list}
        for (tid, g, sm), b in on.items():
            model.AddHint(b, int(where.get(tid) == (g, sm)))
        for t in tasks:
            model.AddHint(start[t.id], ticks(hint.times[t.id][0]) if t.id in hint.times else 0)
        for tid, v in post.items():
            if not isinstance(v, int):
                model.AddHint(v, int(tid in hint.post))
        model.AddHint(C, ticks(hint.predicted_us))

    # ---- solve, then read the plan: the used options, and per SM the tasks placed on it in start order
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    t0 = time.time()
    status = solver.Solve(model)
    solve_s = round(time.time() - t0, 1)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return Result(solver.StatusName(status), float("nan"), float("nan"), solve_s, {}, {})
    chosen = {node: opt for (node, opt), v in use.items() if solver.Value(v)}
    lists = {g: [[] for _ in range(num_sms)] for g in gpu_sets}
    for (tid, g, sm), b in on.items():
        if solver.Value(b):
            lists[g][sm].append(tid)
    for per_sm in lists.values():
        for sm_list in per_sm:
            sm_list.sort(key=lambda tid: solver.Value(start[tid]))
    times = {t.id: (round(solver.Value(start[t.id]) * TICK, 3), round(solver.Value(end[t.id]) * TICK, 3))
             for t in tasks if solver.Value(used[t.id])}
    post_ids = {tid for tid, v in post.items() if solver.Value(used[tid]) and (v == 1 if isinstance(v, int) else solver.Value(v))}
    us = lambda v: round(solver.Value(v) * TICK, 3)
    pool_times = {pid: {"open": us(o), "joins": [us(j) for j in js], "end": us(e)} for pid, (o, js, e) in pool_vars.items()}
    return Result(solver.StatusName(status), round(solver.Value(C) * TICK, 3), round(solver.BestObjectiveBound() * TICK, 3),
                  solve_s, chosen, lists, times, post_ids, pool_times)
