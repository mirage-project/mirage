"""Lowering: from a model held as a graph to registered MPK tasks.

MPK's imperative builder decides the task boundaries in Python -- whoever wrote
linear_layer chose what one task computes. This package holds the same model
with that decision still open, one module per question:

    node          the model as plain SSA nodes over named values -- no grid,
                  no tiling, no task boundaries.
    group         what a group IS (its boundary derived from a node set) and
                  how ONE group becomes a registered MPK task.
    partition     WHICH groupings of a graph are legal, and what a whole
                  partition implies -- coverage, buffer liveness.
    task_search   HOW a group is scheduled: the superoptimizer, the tile
                  candidates, and the guards that reject schedules the
                  Blackwell backend cannot express.

`lower()` below is the join, and the package's whole public surface.

Two lowering rules come from MPK and are not negotiable:

  Every op must land inside a customized op. build_annotated_graph skips
  anything that is not KN_CUSTOMIZED_OP (annotated_graph.cc:300) and
  print_task_graph then asserts on it (runtime.cc:1423), so a leftover plain op
  produces zero layers and aborts rather than degrading. A partition must
  therefore COVER the graph -- check_covers enforces that.

  Every group boundary must be a pre-attached tensor. print_task_graph looks
  tensors up in io_config by guid and asserts owner_op is a KN_INPUT_OP
  (runtime.cc:1515-1518); MPK has no intermediate DTensors. So each group
  output is materialised with pk.new_tensor before the group is registered.

Nothing here ranks anything. search() verifies equivalence and never measures,
and a task that is faster in isolation can leave the model slower (measured: a
silu_mul schedule 1.20x faster per task left Qwen3-0.6B 2.5% slower). Ranking
is whole-model throughput and lives in experiments/searched_tasks/.
"""
from __future__ import annotations

from typing import Callable, Optional

from .group import (Group, group_to_taskspec_build, lower_group, make_group,
                    to_taskspec)
from .node import ModelGraph, Node, OPAQUE, OPS, Value, is_opaque
from .partition import (MAX_GROUP_INPUTS, MAX_GROUP_OPS, Rejection,
                        Schedulable, assign_buffers, check_covers,
                        check_fork_join, check_shapes, enumerate_partitions,
                        feasible_partitions, group_signature)
from .task_search import (MATMUL_K_TILES, MATMUL_N_TILES, MMA_K_ATOM,
                          MPK_BLOCK_DIM, Schedule, TaskSearchError, TaskSpec,
                          TensorSpec, cache_key, default_forloop, default_grid,
                          forloop_candidates, grid_candidates, has_rms_norm,
                          lookup_schedule, register_searched_task,
                          rows_grid_candidates, search_task_schedule,
                          search_task_schedules, store_schedule)


def lower(
    pk,
    graph: ModelGraph,
    partition: list[Group],
    bindings: dict,
    *,
    grid_for: Callable[[Group], tuple] = default_grid,
    forloop_for=None,
    stages_for=None,
    dtype=None,
    outputs: Optional[dict] = None,
    fallbacks: Optional[dict] = None,
    overrides=None,
    alias=None,
    opaque: Optional[dict] = None,
    reuse_buffers: bool = False,
    verbose: bool = False,
) -> dict:
    """Register every group as an MPK task.

    This walks the partition in order and resolves each group's OUTPUT TENSOR;
    lower_group decides what task computes it.

    bindings  value name -> already-attached DTensor, for graph inputs
              (weights and fed activations).
    outputs   optional value name -> DTensor, to write a group's result into a
              caller-owned buffer instead of a fresh one. This is how a
              residual target or the model's own output tensor is threaded in.
    forloop_for callable group -> int, the K steps a matmul group takes.
              Defaults to a 64-wide K tile; forloop_candidates enumerates the
              alternatives.
    stages_for callable group -> int|None, this task's A/B pipeline depth.
              Reaches register_generated_task as params[0].
    alias     callable group -> int|None. When it returns i, the group writes
              IN PLACE into external_inputs[i] instead of a fresh buffer.
    overrides / fallbacks / opaque   see lower_group.
    reuse_buffers assign group outputs to a reused pool by liveness rather
              than one tensor per boundary; needed for anything model-sized.

    Returns the environment: every value name that now has a DTensor.
    """
    import mirage as mi

    check_covers(graph, partition)
    dtype = dtype if dtype is not None else mi.bfloat16
    outputs = outputs or {}
    env = dict(bindings)
    buf_of = (assign_buffers(graph, partition, outputs, alias) if reuse_buffers
              else {})
    memo: dict[tuple, object] = {}
    pool: dict[str, object] = {}

    for g in partition:
        missing = [v.name for v in g.external_inputs if v.name not in env]
        if missing:
            raise ValueError(
                f"group {g.tag!r} reads {missing} before anything produced "
                f"them; is the partition in topological order?")
        ins = [env[v.name] for v in g.external_inputs]

        out = outputs.get(g.output.name)
        ali = alias(g) if (alias is not None and out is None) else None
        if ali is not None:
            out = ins[ali]
        elif out is None:
            buf = buf_of.get(g.output.name)
            if buf is not None and buf in pool:
                out = pool[buf]
            else:
                nm = buf or f"mg_{g.output.name.replace('.', '_')}"
                out = pk.new_tensor(dims=g.output.dims, dtype=dtype, name=nm,
                                    io_category="cuda_tensor")
                if buf is not None:
                    pool[buf] = out

        lower_group(
            pk, graph, g, ins, out,
            grid_for=grid_for, forloop_for=forloop_for,
            stages_for=stages_for, overrides=overrides, fallbacks=fallbacks,
            opaque=opaque, memo=memo, verbose=verbose)
        env[g.output.name] = out

    return env
