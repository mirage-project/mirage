"""Transpose UE8M0-packed scale buffer ``(M, K_PACKED) → (K_PACKED, M)``.

Backed by ``tasks/blackwell/transpose_scale_sm100.cuh``
(``transpose_scale_sm100_task_impl``). Bridges :class:`QuantizeFP8UE8M0`'s
M-outermost output to the K-outermost layout that ``fp8_group_gemm_*``
SFA/SFB TMA descriptors expect. Single-CTA, uint32 elementwise copy.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

import mirage as mi

from ._base import BlockDim, GridDim, MPKModule


__all__ = ["TransposeScale"]


class TransposeScale(MPKModule):
    """``(M, K_PACKED) uint32 → (K_PACKED, M) uint32``.

    Single CTA, fixed grid ``(1, 1, 1)`` and block ``(128, 1, 1)`` —
    overrides must equal the auto values or be ``None``.
    """

    def __init__(self, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)

    def forward(self, scale_in: torch.Tensor) -> torch.Tensor:
        """Plain transpose of the packed-uint32 scale buffer."""
        return scale_in.transpose(0, 1).contiguous()

    def auto_grid_dim(self, scale_in: Any = None) -> GridDim:
        """Single CTA: ``(1, 1, 1)`` — kernel is one-CTA only."""
        return (1, 1, 1)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

    def compile(
        self,
        scale_in: Any,
        *,
        scale_out: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Register a ``transpose_scale_sm100`` task.

        Tensor contract:
          scale_in:  (M, K_PACKED) uint32, M-outermost UE8M0-packed scales.
          scale_out: (K_PACKED, M) uint32, K-outermost transposed scales.

        Notes: SM100-only; single-CTA — ``grid_dim`` is fixed at ``(1, 1, 1)``
        and ``block_dim`` at ``(128, 1, 1)``; overrides must match or be
        ``None``. Bridges :class:`QuantizeFP8UE8M0` output to the layout the
        FP8 group GEMM SFA/SFB TMA descriptors expect.
        """
        import torch as _torch
        from .. import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is not None and grid_dim != self.auto_grid_dim():
            raise ValueError(
                f"TransposeScale: grid_dim is fixed at "
                f"{self.auto_grid_dim()}; got {grid_dim}"
            )
        if block_dim is not None and block_dim != self.default_block_dim():
            raise ValueError(
                f"TransposeScale: block_dim is fixed at "
                f"{self.default_block_dim()}; got {block_dim}"
            )

        M = scale_in.dim(0)
        K_PACKED = scale_in.dim(1)
        if scale_out is None:
            out_dt = pk.new_tensor(
                dims=(K_PACKED, M),
                dtype=mi.uint32,
                name=f"{self.prefix}scale_transposed",
            )
        elif isinstance(scale_out, _torch.Tensor):
            out_dt = pk.attach_input(
                scale_out, name=f"{self.prefix}scale_transposed"
            )
        else:
            out_dt = scale_out

        from ...core import CyTBGraph
        from ...kernel import TBGraph

        assert scale_in.num_dims == 2
        assert out_dt.num_dims == 2
        assert out_dt.dim(0) == K_PACKED
        assert out_dt.dim(1) == M
        params = [M, K_PACKED]
        tb_graph = TBGraph(CyTBGraph((1, 1, 1), (128, 1, 1), 1, 64))
        tb_graph.new_input(scale_in, (-1, -1, -1), -1, True)
        tb_graph.new_input(out_dt, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([scale_in, out_dt], tb_graph)
        pk.kn_graph.register_task(tb_graph, "transpose_scale_sm100", params)
        return out_dt
