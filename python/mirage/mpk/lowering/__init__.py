"""Lowering: a model held as a graph, to registered MPK tasks.

    node          the model as SSA nodes over named values
    group         what a group is, and how one becomes a task
    partition     which groupings are legal; coverage and buffer liveness
    task_search   the superoptimizer, tile candidates, and the guards

A partition must COVER the graph and every group boundary must be a
pre-attached tensor: MPK has no intermediate DTensors, and a leftover plain op
aborts rather than degrading.
"""
from __future__ import annotations

from typing import Callable, Optional

from .group import (Group, group_to_taskspec_build, lower_group, make_group,
                    to_taskspec)
from .node import (ModelGraph, Node, OPAQUE, OPAQUE_OPS, OPS, Value,
                   is_opaque)
from .opaque import standard_handlers
from .partition import (MAX_GROUP_INPUTS, MAX_GROUP_OPS, Rejection,
                        Schedulable, assign_buffers, check_covers,
                        check_fork_join, check_shapes, default_partition,
                        enumerate_partitions,
                        feasible_partitions, group_signature)
from .task_search import (MATMUL_K_TILES, MATMUL_N_TILES, MMA_K_ATOM,
                          MPK_BLOCK_DIM, Schedule, TaskSearchError, TaskSpec,
                          TensorSpec, cache_key, default_forloop, default_grid,
                          batched_grid_candidates, forloop_candidates, grid_candidates,
                          has_rms_norm, knobs_from_env,
                          lookup_schedule, register_searched_task,
                          rows_grid_candidates, search_task_schedule,
                          search_task_schedules, store_schedule)


def lower(
    pk,
    graph: ModelGraph,
    partition: list[Group],
    bindings: dict,
    *,
    outputs: Optional[dict] = None,
    overrides=None,
    opaque: Optional[dict] = None,
    knobs=None,
    dtype=None,
    verbose: bool = False,
) -> dict:
    """Register every group as an MPK task."""
    import mirage as mi

    check_covers(graph, partition)
    dtype = dtype if dtype is not None else mi.bfloat16
    outputs = outputs or {}
    # The graph already says which nodes are opaque; standard_handlers covers
    # every name OPAQUE_OPS declares, so a caller overrides this only to
    # substitute a task of its own.
    opaque = standard_handlers() if opaque is None else opaque
    # The schedule knobs are always the same triple; a caller overrides them
    # only to sweep one. Buffers are always reused: one tensor per boundary
    # grows with depth, which nothing model-sized can afford.
    grid_for, forloop_for, stages_for = knobs or knobs_from_env(graph)
    env = dict(bindings)
    buf_of = assign_buffers(graph, partition, outputs)
    memo: dict[tuple, object] = {}
    pool: dict[str, object] = {}

    for g in partition:
        missing = [v.name for v in g.external_inputs if v.name not in env]
        if missing:
            raise ValueError(
                f"group {g.tag!r} reads {missing} before anything produced "
                f"them; is the partition in topological order?")
        ins = [env[v.name] for v in g.external_inputs]

        # One tensor per declared output. Only an opaque task has more than
        # one -- attention prep stages q/k^T/v/mask for the generated core.
        resolved = []
        for v in g.outputs:
            out = outputs.get(v.name)
            if out is None:
                buf = buf_of.get(v.name)
                if buf is not None and buf in pool:
                    out = pool[buf]
                else:
                    nm = buf or f"mg_{v.name.replace('.', '_')}"
                    out = pk.new_tensor(
                        dims=v.dims,
                        dtype=getattr(mi, v.dtype) if v.dtype else dtype,
                        name=nm, io_category="cuda_tensor")
                    if buf is not None:
                        pool[buf] = out
            resolved.append(out)
        out = resolved[0]

        lower_group(
            pk, graph, g, ins, out,
            grid_for=grid_for, forloop_for=forloop_for,
            stages_for=stages_for, overrides=overrides, opaque=opaque,
            memo=memo, verbose=verbose, outs=resolved)
        for v, t in zip(g.outputs, resolved):
            env[v.name] = t

    return env
