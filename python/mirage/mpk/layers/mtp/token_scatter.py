"""MTP slot scatter primitives (int64 token / float32 prob).

Two single-purpose wrappers around tasks ``mtp_token_scatter`` and
``mtp_float_scatter`` defined in
``include/mirage/persistent_kernel/tasks/speculative_decoding/mtp_token_ops.cuh``.
Each writes one column of a ``(batch_size, num_slots)`` accumulator at
the compile-time-baked ``slot_idx``; the kernel iterates batch with
``threadIdx.x``.
"""
from __future__ import annotations

from typing import Optional

import torch

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

from ....core import DTensor


__all__ = ["MTPTokenScatter", "MTPFloatScatter"]


class MTPTokenScatter(MPKModule):
    """Scatter ``src: (batch_size, 1) int64`` into ``dst[:, slot_idx]``.

    Wraps task ``mtp_token_scatter``; ``slot_idx`` is a compile-time
    constant so MTP unrolls one task per draft step.
    """

    def __init__(
        self,
        batch_size: int,
        num_slots: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if batch_size <= 0:
            raise ValueError(
                f"MTPTokenScatter batch_size must be positive; got {batch_size}"
            )
        if num_slots <= 0:
            raise ValueError(
                f"MTPTokenScatter num_slots must be positive; got {num_slots}"
            )
        self.batch_size = batch_size
        self.num_slots = num_slots

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        slot_idx: int,
    ) -> torch.Tensor:
        """Reference: ``dst[:, slot_idx] = src[:, 0]`` (int64, in-place)."""
        if src.dim() != 2 or src.shape[1] != 1:
            raise ValueError(
                f"MTPTokenScatter.forward expects src of shape "
                f"(batch_size, 1); got {tuple(src.shape)}"
            )
        if dst.dim() != 2:
            raise ValueError(
                f"MTPTokenScatter.forward expects 2-D dst; "
                f"got shape {tuple(dst.shape)}"
            )
        if not (0 <= slot_idx < dst.shape[1]):
            raise ValueError(
                f"MTPTokenScatter.forward slot_idx {slot_idx} out of "
                f"range [0, {dst.shape[1]})"
            )
        dst[:, slot_idx] = src[:, 0]
        return dst

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """``(1, 1, 1)`` — kernel walks ``batch_size`` via ``threadIdx.x``."""
        return (1, 1, 1)

    def compile(
        self,
        src: DTensor,
        dst: DTensor,
        slot_idx: int,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``mtp_token_scatter`` task.

        Tensor contract:
          src: (batch_size, 1) int64, dense. Per-batch draft token.
          dst: (batch_size, num_slots) int64, dense. In-place output;
               kernel writes ``dst[b, slot_idx] = src[b]`` only.
        Params (compile-time): ``[batch_size, num_slots, slot_idx]``.

        Notes: ``0 <= slot_idx < num_slots``. No runtime meta-tensor
        dependency. Returned ``dst`` is the mutated buffer.
        """
        pk = current_pk()

        if not (0 <= slot_idx < self.num_slots):
            raise ValueError(
                f"MTPTokenScatter.compile slot_idx {slot_idx} out of "
                f"range [0, {self.num_slots})"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(src, dst)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.batch_size, self.num_slots, slot_idx]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(src, (-1, -1, -1), -1, True)
        tb_graph.new_input(dst, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([src, dst], tb_graph)
        pk.kn_graph.register_task(tb_graph, "mtp_token_scatter", params)
        return dst


class MTPFloatScatter(MPKModule):
    """Scatter ``src: (batch_size, 1) float32`` into ``dst[:, slot_idx]``.

    Wraps task ``mtp_float_scatter``; same shape contract as
    :class:`MTPTokenScatter` but float32, used to stash per-step draft
    probabilities for the probabilistic verifier.
    """

    def __init__(
        self,
        batch_size: int,
        num_slots: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if batch_size <= 0:
            raise ValueError(
                f"MTPFloatScatter batch_size must be positive; got {batch_size}"
            )
        if num_slots <= 0:
            raise ValueError(
                f"MTPFloatScatter num_slots must be positive; got {num_slots}"
            )
        self.batch_size = batch_size
        self.num_slots = num_slots

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        slot_idx: int,
    ) -> torch.Tensor:
        """Reference: ``dst[:, slot_idx] = src[:, 0]`` (float32)."""
        if src.dim() != 2 or src.shape[1] != 1:
            raise ValueError(
                f"MTPFloatScatter.forward expects src of shape "
                f"(batch_size, 1); got {tuple(src.shape)}"
            )
        if dst.dim() != 2:
            raise ValueError(
                f"MTPFloatScatter.forward expects 2-D dst; "
                f"got shape {tuple(dst.shape)}"
            )
        if not (0 <= slot_idx < dst.shape[1]):
            raise ValueError(
                f"MTPFloatScatter.forward slot_idx {slot_idx} out of "
                f"range [0, {dst.shape[1]})"
            )
        dst[:, slot_idx] = src[:, 0]
        return dst

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """``(1, 1, 1)`` — same single-CTA pattern as :class:`MTPTokenScatter`."""
        return (1, 1, 1)

    def compile(
        self,
        src: DTensor,
        dst: DTensor,
        slot_idx: int,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``mtp_float_scatter`` task.

        Tensor contract:
          src: (batch_size, 1) float32, dense. Per-batch draft prob.
          dst: (batch_size, num_slots) float32, dense. In-place output;
               kernel writes ``dst[b, slot_idx] = src[b]`` only.
        Params (compile-time): ``[batch_size, num_slots, slot_idx]``.

        Notes: ``0 <= slot_idx < num_slots``. No runtime meta-tensor
        dependency. Returned ``dst`` is the mutated buffer.
        """
        pk = current_pk()

        if not (0 <= slot_idx < self.num_slots):
            raise ValueError(
                f"MTPFloatScatter.compile slot_idx {slot_idx} out of "
                f"range [0, {self.num_slots})"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(src, dst)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.batch_size, self.num_slots, slot_idx]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(src, (-1, -1, -1), -1, True)
        tb_graph.new_input(dst, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([src, dst], tb_graph)
        pk.kn_graph.register_task(tb_graph, "mtp_float_scatter", params)
        return dst
