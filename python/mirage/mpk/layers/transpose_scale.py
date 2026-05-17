"""Transpose ``(M, K_PACKED) → (K_PACKED, M)`` for the FP8 group-GEMM SFA layout.

Wraps :meth:`PersistentKernel.transpose_scale_sm100_layer` — task
``transpose_scale_sm100``. The kernel is a single-CTA copy that bridges
:meth:`QuantizeFP8`'s M-outermost packed-uint32 scale output to the
K-outermost layout the ``fp8_group_gemm_*`` TMA descriptors require.

Both shapes carry the same UE8M0-packed bytes; only the dim order
changes. See ``transpose_scale_sm100.cuh`` for the actual copy loop.

Forward reference
-----------------

``forward()`` is a plain ``scale_in.transpose(0, 1)`` of the
``uint32``-packed buffer — the bytes are the same after transpose
because each ``uint32`` is treated atomically along the K-block axis.

Grid / block
------------

Fixed at ``(1, 1, 1)`` / ``(128, 1, 1)`` by the pk method (single CTA).
``compile()`` accepts ``grid_dim`` / ``block_dim`` overrides for API
parity but rejects non-default values.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

import mirage as mi

from ._base import BlockDim, GridDim, MPKModule


__all__ = ["TransposeScale"]


class TransposeScale(MPKModule):
    """``(M, K_PACKED) uint32 → (K_PACKED, M) uint32`` transpose.

    Args:
        prefix: Reserved. No parameters live here.
    """

    def __init__(self, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)

    def forward(self, scale_in: torch.Tensor) -> torch.Tensor:
        """Plain transpose of the packed-uint32 scale buffer."""
        return scale_in.transpose(0, 1).contiguous()

    def auto_grid_dim(self, scale_in: Any = None) -> GridDim:
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

        Args:
            scale_in:  ``(M, K_PACKED)`` uint32 DTensor.
            scale_out: ``None`` allocates a fresh ``(K_PACKED, M)``
                uint32 DTensor; ``torch.Tensor`` attaches a host buffer;
                ``DTensor`` is used as-is.
            grid_dim / block_dim: Must equal the auto values
                (``(1, 1, 1)`` / ``(128, 1, 1)``) or be ``None``.

        Returns:
            ``scale_out``.
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

        # Inlined task registration (was pk.transpose_scale_sm100_layer).
        from ...core import CyTBGraph
        from ...kernel import TBGraph

        assert scale_in.num_dims == 2
        assert out_dt.num_dims == 2
        assert out_dt.dim(0) == K_PACKED
        assert out_dt.dim(1) == M
        params = [M, K_PACKED]
        grid_dim_local = (1, 1, 1)
        block_dim_local = (128, 1, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim_local, block_dim_local, 1, 64))
        tb_graph.new_input(scale_in, (-1, -1, -1), -1, True)
        tb_graph.new_input(out_dt,   (-1, -1, -1), -1, True)
        pk.kn_graph.customized([scale_in, out_dt], tb_graph)
        pk.kn_graph.register_task(tb_graph, "transpose_scale_sm100", params)
        return out_dt
