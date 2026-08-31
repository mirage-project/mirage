"""A Group: the set of nodes lowered together as one MPK task.

Where node.py holds the model with the task boundary still open, this is the
boundary itself -- and it is DERIVED, never asserted. make_group takes a set of
node indices and computes what the group reads, what it produces, and whether
that is a legal task at all; group_to_taskspec_build turns it back into a
muGraph for search.

Deriving rather than declaring is the point. A partition proposes node sets; if
external inputs and the live output had to be spelled out alongside, every
proposal would be a chance to spell them out wrongly, and the failure would
surface deep inside MPK's lowering with nothing pointing back.

lower_group() is the other half: one group in, one registered MPK task out. It
is the only place that knows the three ways a group can become a task -- an
opaque handler, a hand-written override, or a searched schedule.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

from . import task_search
from .node import ModelGraph, Value, is_opaque
from .task_search import TaskSpec, TensorSpec, default_forloop

@dataclasses.dataclass
class Group:
    """A set of node indices lowered together as one task.

    `external_inputs` are values the group reads but does not produce;
    `output` is the single value it produces that anything outside reads.
    task_search requires exactly one output per task, so a group producing two
    live values is not lowerable and the partitioner must not emit one.
    """
    nodes: tuple[int, ...]
    external_inputs: tuple[Value, ...]
    output: Value
    tag: str = ""

    def __repr__(self) -> str:
        return f"Group({self.tag or list(self.nodes)}, {len(self.external_inputs)} in)"


def make_group(graph: ModelGraph, node_ids, tag: str = "") -> Group:
    """Derive a group's boundary from the node set. Raises if it is not a
    legal task: no live output, or more than one."""
    ids = tuple(sorted(node_ids))
    opaque = [i for i in ids if is_opaque(graph.nodes[i].op)]
    if opaque and len(ids) > 1:
        raise ValueError(
            f"group {tag or ids} mixes opaque node(s) "
            f"{[graph.nodes[i].op for i in opaque]} with muGraph ops; an "
            f"opaque task cannot be fused with anything")
    inside = set(ids)
    produced = {graph.nodes[i].output for i in ids}

    external, seen = [], set()
    for i in ids:
        for v in graph.nodes[i].inputs:
            if v in produced or v.name in seen:
                continue
            seen.add(v.name)
            external.append(v)

    # Live = read by a node outside the group, or a graph output.
    live = [v for v in produced
            if any(c not in inside for c in graph.consumers(v))
            or v in graph.outputs]
    if len(live) != 1:
        raise ValueError(
            f"group {tag or ids} has {len(live)} live outputs, need exactly 1: "
            f"{[v.name for v in live]}")
    return Group(nodes=ids, external_inputs=tuple(external), output=live[0],
                 tag=tag)


def group_to_taskspec_build(graph: ModelGraph, group: Group) -> Callable:
    """The `build` lambda for a TaskSpec: replay the group on a fresh KNGraph.

    Signature matches task_search.TaskSpec -- (kn, t) where t is the list of
    input tensors, in the same order as group.external_inputs.
    """
    def build(kn, t):
        env = {v.name: t[i] for i, v in enumerate(group.external_inputs)}
        out = None
        for i in group.nodes:
            n = graph.nodes[i]
            args = [env[v.name] for v in n.inputs]
            out = getattr(kn, n.op)(*args, **n.attrs)
            env[n.output.name] = out
        return env[group.output.name]
    return build


# search() explores randomly and returns as soon as one candidate verifies, so
# a single draw can come back with nothing usable even where schedules exist --
# typically it yields 3-4 candidates, but a draw that only produced the trivial
# plain-op graph raises "a task is one fused op; candidate has 0". Retry rather
# than fail the whole lowering on one unlucky draw.
SEARCH_ATTEMPTS = 3

# Some groups search will never fuse, however many draws you give it. Measured:
# a matmul whose K is not a power of two never produces a customized op --
# K=1024 and K=2048 fuse, K=1280/1536/2560/3072 do not, at any grid and at
# every forloop_range tried (K tile 64, 128, 256, 512). Qwen3's down projection
# is (8,3072)@(3072,1024), which is exactly that shape, and is why it has always
# carried a hand-written schedule.
#
# So a group may name a fallback: a PersistentKernel layer method that
# registers the same computation with a hand-written schedule. lower() uses it
# only after search has genuinely failed, and says so.


def _search_with_retry(spec, grid, tag: str, forloop=None):
    """Ask for `forloop` first, then let search choose.

    Constraining the forloop is a perf decision, not a correctness one, so a
    spec that has no schedule at the asked-for K tile should still lower --
    just more slowly -- rather than fall all the way through to a fallback."""
    last = None
    for fl in ([forloop] if forloop and forloop > 1 else []) + [None]:
        for _ in range(SEARCH_ATTEMPTS):
            try:
                return task_search.search_task_schedules(
                    spec, grid_dim=grid, forloop_range=fl)[0]
            except task_search.TaskSearchError as e:
                last = e
    raise task_search.TaskSearchError(
        f"group {tag!r}: no usable schedule in {SEARCH_ATTEMPTS} draws; "
        f"last reason: {last}")


def to_taskspec(graph: ModelGraph, group: Group) -> TaskSpec:
    """The group as a spec search can schedule: what it computes, and the
    shapes it reads. The name is the tag, which is also the schedule cache's
    key, so two groups computing the same thing share a search."""
    return TaskSpec(group.tag or group.output.name,
                    group_to_taskspec_build(graph, group),
                    [TensorSpec(v.dims) for v in group.external_inputs])


def lower_group(pk, graph: ModelGraph, group: Group, ins, out, *,
                grid_for, forloop_for=None, stages_for=None, overrides=None,
                fallbacks=None, opaque=None, memo=None, verbose=False):
    """Register ONE group as an MPK task, writing its result into `out`.

    Three ways a group becomes a task, in the order they are tried:

      opaque    the graph cannot model it at all -- embedding, a KV-cache
                append, attention, argmax -- so a registered handler supplies
                the hand-written task.
      override  the graph CAN model it and search CAN schedule it, but the
                hand-written task is known to win. Search succeeding is not
                search winning: the lm_head searches fine and lowers to a
                2374-block task where linear_sm100 uses 148.
      searched  the normal path -- cache, then memo, then search, then (only
                if search found nothing at all) a fallback schedule.

    `memo` is shared across the whole partition: Qwen3 has 28 identical layers,
    so without it a 312-group model runs 312 searches for about seven distinct
    questions. A remembered None is a remembered FAILURE, and takes the
    fallback without searching again.

    grid_for/forloop_for/stages_for are CALLABLES, not values, because an
    opaque group has no schedule to ask for: argmax's output last dim is 1 and
    default_grid rejects anything not a multiple of 64. Resolving them before
    the opaque branch would fail on a group that never needed them.
    """
    opaque = opaque or {}
    memo = {} if memo is None else memo

    def pick(table):
        """fallbacks/overrides are either {tag: fn} or a callable. The callable
        form gets the whole group, which is what you need when several groups
        share a tag and only one of them needs the treatment."""
        if callable(table):
            return table(group)
        return (table or {}).get(group.tag)

    op0 = graph.nodes[group.nodes[0]].op
    if is_opaque(op0):
        name = op0.split(":", 1)[1]
        fn = opaque.get(name)
        if fn is None:
            raise ValueError(
                f"no handler for opaque task {name!r}; the graph cannot "
                f"model it, so lowering needs one")
        if verbose:
            print(f"[lower] {name}: hand-written task", flush=True)
        fn(pk, group, ins, out)
        return

    grid = tuple(grid_for(group))
    ov = pick(overrides)
    if ov is not None:
        if verbose:
            print(f"[lower] {group.tag or group.output.name}: hand-written "
                  f"(override)", flush=True)
        ov(pk, ins, out, grid)
        return

    spec = to_taskspec(graph, group)
    in_dims = [tuple(v.dims) for v in group.external_inputs]
    forloop = (forloop_for(group) if forloop_for is not None
               else default_forloop(graph, group))
    stages = stages_for(group) if stages_for is not None else None
    sig = (spec.name, tuple(in_dims), grid, forloop, stages,
           tuple(graph.nodes[i].op for i in group.nodes))

    sched = task_search.lookup_schedule(spec.name, in_dims, grid)
    source = "cache"
    if sched is None and sig in memo:
        sched, source = memo[sig], "memo"
    elif sched is None:
        try:
            sched = _search_with_retry(spec, grid, group.tag, forloop)
            memo[sig] = sched
            source = "search"
        except task_search.TaskSearchError:
            memo[sig] = None

    if sched is None:
        fb = pick(fallbacks)
        if fb is None:
            raise task_search.TaskSearchError(
                f"group {group.tag!r}: search found nothing, no fallback")
        if verbose:
            print(f"[lower] {group.tag}: search found nothing, using the "
                  f"hand-written schedule", flush=True)
        fb(pk, ins, out, grid)
        return

    if verbose:
        print(f"[lower] {group.tag or group.output.name}: {source} "
              f"{sched.describe()}", flush=True)
    task_search.register_searched_task(pk, sched, inputs=ins, output=out,
                                       pipeline_stages=stages)
