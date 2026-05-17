"""Zero-fill a tensor with a chained dependency edge.

Backed by ``tasks/blackwell/tensor_init.cuh``
(``tensor_init_zero_sm100_task_impl``). Blackwell/sm100-only — there
is no Ampere or Hopper variant. ``OUTPUT_SIZE`` must be a multiple of
8 (16B vector store, per the kernel ``static_assert``).

``dummy`` is a dep-only edge: it appears as both an input and an output
of the task so the MPK scheduler chains ``tensor_init`` between the
producer of ``dummy`` and any downstream consumer; the kernel never
reads or writes ``dummy``'s data. Typical use: pre-zero the output of a
split-K linear that uses ``tma_reduce_add_async``.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from ._base import BlockDim, GridDim, MPKModule


__all__ = ["TensorInit"]


class TensorInit(MPKModule):
    """Zero-fill ``target`` and chain a dep edge through ``dummy``."""

    def __init__(self, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        """Return ``torch.zeros_like(target)`` (algebra is a pure zero-fill)."""
        return torch.zeros_like(target)

    def auto_grid_dim(
        self,
        target: Any = None,
        dummy: Any = None,
    ) -> GridDim:
        """``(num_workers, 1, 1)`` — saturate the pool; the kernel has no
        kernel-side alignment constraint other than ``OUTPUT_SIZE % 8 == 0``.
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
        """Register a ``tensor_init`` task (Blackwell/SM100-only zero-fill).

        Tensor contract:
          target: (*shape) any dtype, zeroed by the kernel at runtime.
          dummy:  any DTensor, dep-only edge (appears as both input and output;
                  data is never read or written). Used to chain the task
                  between ``dummy``'s producer and downstream consumers.

        Notes: requires ``output_size % 8 == 0`` (16B vector store
        ``static_assert``). ``dummy_input_map`` / ``target_input_map`` default
        to the SplitK-linear partitioning. Returns ``target``.
        """
        from .. import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(target, dummy)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # bgraph arity (1 input, 2 outputs):
        #   input_ops[0]  = dummy  (read dep)
        #   output_ops[0] = target (zeroed by the kernel)
        #   output_ops[1] = dummy  (dep-only write)
        from ...core import CyTBGraph
        from ...kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(dummy, dummy_input_map, -1, True)
        tb_graph.new_input(target, target_input_map, -1, True)
        tb_graph.new_input(dummy, dummy_input_map, -1, True)
        pk.kn_graph.customized([dummy, target, dummy], tb_graph)
        pk.kn_graph.register_task(tb_graph, "tensor_init")
        return target
