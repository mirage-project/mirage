"""Zero-fill a tensor with a dependency-chained task.

Wraps :meth:`PersistentKernel.tensor_init_layer` — task ``tensor_init``.

The pk method zeroes ``target`` and uses ``dummy`` as a dep-only edge
(it appears as both an input and an output of the task so the MPK
dep-tracker chains ``tensor_init`` between the producer of ``dummy``
and any downstream consumer of ``dummy``). The kernel never reads or
writes ``dummy``'s data.

Typical use is inside the DSv3 / qwen3 builder before a split-K linear
that uses ``tma_reduce_add_async`` and therefore needs its output
buffer pre-zeroed. See ``persistent_kernel.py:splitk_linear_layer``
where the accumulate=False branch invokes ``tensor_init_layer``
internally.

Tensor contract
---------------

* ``target``           : the buffer to zero (any shape / dtype the kernel
                         can write bytes to).
* ``dummy``            : dependency carrier — typically the producer of
                         ``target`` in the same iteration (e.g. the
                         input of the consumer split-K linear).
* ``dummy_input_map``  : MPK partition map ``(x, y, z)`` for ``dummy``
                         (passes through to ``TBGraph.new_input``).
* ``target_input_map`` : MPK partition map for ``target``.

The auto grid heuristic just mirrors whatever the caller threads
through; ``tensor_init`` has no kernel-specific alignment requirement.

Forward
-------

``forward()`` returns a zeroed tensor with the same shape / dtype as
``target``. The PyTorch reference doesn't involve ``dummy`` (the
dependency is a graph artifact, not algebra).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from ._base import BlockDim, GridDim, MPKModule


__all__ = ["TensorInit"]


class TensorInit(MPKModule):
    """Zero-fill ``target`` and chain a dep edge through ``dummy``.

    Args:
        prefix: Reserved. No parameters live in this module.
    """

    def __init__(self, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        """Return ``torch.zeros_like(target)``.

        The pk method's ``dummy`` argument is a dep-only graph edge —
        the algebra is a pure zero-fill.
        """
        return torch.zeros_like(target)

    def auto_grid_dim(
        self,
        target: Any = None,
        dummy: Any = None,
    ) -> GridDim:
        """No kernel-mandated grid; the legacy callers pass an explicit
        ``grid_dim`` (typically the consumer linear's grid). We default
        to ``(num_workers, 1, 1)`` as a saturate-the-pool fallback.
        """
        from .. import context as _ctx

        pk = _ctx.current_pk()
        return (int(pk.num_workers), 1, 1)

    def compile(
        self,
        target: Any,
        dummy: Any,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
        dummy_input_map: Tuple[int, int, int] = (-1, 1, -1),
        target_input_map: Tuple[int, int, int] = (1, -1, -1),
    ) -> Any:
        """Register a ``tensor_init`` task.

        Args:
            target: DTensor to zero.
            dummy:  DTensor carrying the dep edge (often the consumer's
                input).
            grid_dim / block_dim: explicit overrides; ``None`` falls
                back to :meth:`auto_grid_dim` / :meth:`default_block_dim`.
            dummy_input_map / target_input_map: MPK partition maps for
                ``dummy`` and ``target`` respectively. Defaults match
                the SplitK linear usage in
                ``persistent_kernel.py:splitk_linear_layer``.

        Returns:
            ``target`` (now scheduled to be zeroed at graph runtime).
        """
        from .. import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(target, dummy)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (the body that used to live on
        # ``PersistentKernel.tensor_init_layer``). Each catalog module
        # owns its own task wiring so adding a new layer doesn't require
        # editing ``persistent_kernel.py``.
        #
        # The bgraph order is [dummy, target, dummy] -> arity (1, 2):
        #   input_ops[0]  = dummy   (read dep)
        #   output_ops[0] = target  (the buffer the kernel zeroes)
        #   output_ops[1] = dummy   (dep-only write)
        # ``dummy`` carries a dependency edge only — the kernel never
        # reads or writes its data.
        from ...core import CyTBGraph
        from ...kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(dummy, dummy_input_map, -1, True)
        tb_graph.new_input(target, target_input_map, -1, True)
        tb_graph.new_input(dummy, dummy_input_map, -1, True)
        pk.kn_graph.customized([dummy, target, dummy], tb_graph)
        pk.kn_graph.register_task(tb_graph, "tensor_init")
        return target
