"""A Group: the set of nodes lowered together as one MPK task."""
from __future__ import annotations

import dataclasses
from typing import Callable

from . import task_search
from .node import ModelGraph, Value, is_opaque
from .task_search import TaskSpec, TensorSpec, default_forloop

@dataclasses.dataclass
class Group:
    """A set of node indices lowered together as one task."""
    nodes: tuple[int, ...]
    external_inputs: tuple[Value, ...]
    output: Value
    tag: str = ""
    # An opaque task may write several tensors; output stays outputs[0] so
    # every existing caller is unchanged.
    extra_outputs: tuple[Value, ...] = ()
    # An opaque node's attrs: the parameters its hand-written task needs and
    # the graph cannot otherwise supply, e.g. attention's kv-head count.
    attrs: dict = dataclasses.field(default_factory=dict)

    @property
    def outputs(self) -> tuple:
        return (self.output,) + self.extra_outputs

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
    # A muGraph task has exactly one output. An OPAQUE task may have several:
    # attention prep stages q/k^T/v/mask for the generated core, and each has
    # to be a real Value or MPK cannot see that the core depends on the prep.
    if opaque:
        # An opaque task's outputs are whatever it DECLARES, not whatever
        # happens to be read: attention prep stages four tensors and the task
        # writes all four whether or not every one has a consumer yet.
        n = graph.nodes[ids[0]]
        return Group(nodes=ids, external_inputs=tuple(external),
                     output=n.output, tag=tag,
                     extra_outputs=tuple(n.attrs.get("extra_outputs", ())),
                     attrs={k: v for k, v in n.attrs.items()
                            if k != "extra_outputs"})
    if len(live) != 1:
        raise ValueError(
            f"group {tag or ids} has {len(live)} live outputs, need exactly 1: "
            f"{[v.name for v in live]}")
    return Group(nodes=ids, external_inputs=tuple(external), output=live[0],
                 tag=tag)


def group_to_taskspec_build(graph: ModelGraph, group: Group) -> Callable:
    """The `build` lambda for a TaskSpec: replay the group on a fresh KNGraph."""
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


SEARCH_ATTEMPTS = 3



def _search_with_retry(spec, grid, tag: str, forloop=None, max_ops=None):
    """Ask for `forloop` first, then let search choose."""
    last = None
    for fl in ([forloop] if forloop and forloop > 1 else []) + [None]:
        for _ in range(SEARCH_ATTEMPTS):
            try:
                return task_search.search_task_schedules(
                    spec, grid_dim=grid, forloop_range=fl, max_ops=max_ops,
                    wide_inputs=len(spec.inputs) > 3)[0]
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
                opaque=None, memo=None, verbose=False, outs=None):
    """Register ONE group as an MPK task, writing its result into `out`."""
    opaque = opaque or {}
    memo = {} if memo is None else memo

    def pick(table):
        """overrides is either {tag: fn} or a callable taking the group."""
        if callable(table):
            return table(group)
        return (table or {}).get(group.tag)

    op0 = graph.nodes[group.nodes[0]].op
    if is_opaque(op0):
        name = op0.split(":", 1)[1]
        fn = opaque.get(name)
        if fn is None:
            raise ValueError(
                f"no handler for opaque task {name!r}; the graph cannot model "
                f"it, so lowering needs one. lowering.standard_handlers("
                f"shapes, meta) supplies the four every decoder has "
                f"(embedding, attention, attn_prep/attn_finalize, argmax); "
                f"pass its result as lower(..., opaque=...). Got: "
                f"{sorted(opaque) or 'nothing'}")
        if verbose:
            print(f"[lower] {name}: hand-written task", flush=True)
        fn(pk, group, ins, outs if outs is not None else [out])
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
            # A fused group needs room for its own ops: 4 inputs + an
            # accumulator + an output on top of the body itself.
            n_ops = len(group.nodes) + len(group.external_inputs) + 4
            sched = _search_with_retry(spec, grid, group.tag, forloop,
                                       max_ops=max(9, n_ops))
            memo[sig] = sched
            source = "search"
        except task_search.TaskSearchError:
            memo[sig] = None

    if sched is None:
        shapes = " @ ".join("x".join(str(d) for d in dims) for dims in in_dims)
        raise task_search.TaskSearchError(
            f"group {group.tag!r} ({shapes}, grid={grid}, "
            f"forloop_range={forloop}): search found no usable schedule.")

    if verbose:
        print(f"[lower] {group.tag or group.output.name}: {source} "
              f"{sched.describe()}", flush=True)
    task_search.register_searched_task(pk, sched, inputs=ins, output=out,
                                       pipeline_stages=stages)
