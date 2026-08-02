"""Segmented muGraph compilation prototype.

Compiles selected model regions through Mirage's ordinary KNGraph muGraph
compiler (``KNGraph.superoptimize`` / ``KNGraph.compile``) instead of lowering
them to the MPK task/event graph.

Nothing in this package calls ``generate_task_graph()``, ``register_task()``,
``PersistentKernel.compile()``, or any MPK task registration.  The MPK
task-graph implementation is used *only* as an unmodified benchmark baseline,
and only from the benchmark drivers.
"""

from .runner import (  # noqa: F401
    CompiledRegion,
    RegionKey,
    RegionKind,
    SegmentedMuGraphRunner,
    TensorSpec,
    assert_no_task_graph_artifacts,
    no_task_graph_guard,
)

__all__ = [
    "CompiledRegion",
    "RegionKey",
    "RegionKind",
    "SegmentedMuGraphRunner",
    "TensorSpec",
    "assert_no_task_graph_artifacts",
    "no_task_graph_guard",
]
