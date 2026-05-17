"""Interleave per-head ``q_nope`` with ``q_pe`` for MLA decode.

Wraps :meth:`PersistentKernel.assemble_q_decode_sm100_layer` — task
``assemble_q_decode_sm100``. Builds the per-head ``[nope | pe]``
layout the MLA attention kernel expects:

    q_nope_pe[n, h, :D_nope] = q_nope_abs[n, h, :]
    q_nope_pe[n, h, D_nope:] = q_pe[n, h, :]

Used by the DSv3 ``MPK_DSV3_BMM=1`` decode Q path:

    rmsnorm_linear(q_a, q_b_nope) → q_nope          (N, H, 128)
    quantize_fp8(q_nope)         → q_nope_fp8       (N, H, 128)
    linear_fp8_bmm_sm100(...)    → q_nope_abs       (N, H, 512)
    rmsnorm_linear(q_a, q_b_pe)  → q_pe             (N, H, 64)
    assemble_q_decode_sm100(...) → q_nope_pe        (N, H, 576)

The PE-only mode (``pe_only=True``) writes only the trailing ``D_pe``
slice and leaves ``q_nope_abs`` untouched — used when ``q_nope_abs``
has already been written into the output buffer via a prior task
(saves one TMA roundtrip).

Forward reference
-----------------

``forward()`` concatenates ``q_nope_abs`` and ``q_pe`` along the
trailing axis. ``pe_only=True`` returns just ``q_pe`` (the caller is
expected to have populated the nope half itself).
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from ._base import BlockDim, GridDim, MPKModule


__all__ = ["AssembleQDecode"]


class AssembleQDecode(MPKModule):
    """Interleave per-head ``q_nope`` with ``q_pe`` into ``[nope | pe]``.

    Args:
        pe_only: If ``True``, only write the PE half of ``q_nope_pe``.
            The caller must have populated the nope half. Default
            ``False`` (full assemble).
        prefix: Reserved. No parameters live here.
    """

    def __init__(self, *, pe_only: bool = False, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        self.pe_only = pe_only

    def forward(
        self,
        q_nope_abs: torch.Tensor,
        q_pe: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate ``[q_nope_abs | q_pe]`` along the last axis."""
        if self.pe_only:
            return q_pe
        return torch.cat([q_nope_abs, q_pe], dim=-1)

    def auto_grid_dim(self, q_nope_abs: Any) -> GridDim:
        """``(N, 1, 1)`` — one CTA per token."""
        return (q_nope_abs.dim(0), 1, 1)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

    def compile(
        self,
        q_nope_abs: Any,
        q_pe: Any,
        q_nope_pe: Any,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Register a ``assemble_q_decode_sm100`` task.

        Args:
            q_nope_abs: ``(N, H, D_nope)`` bf16 DTensor.
            q_pe:       ``(N, H, D_pe)`` bf16 DTensor.
            q_nope_pe:  ``(N, H, D_nope + D_pe)`` or
                ``(N, H * (D_nope + D_pe))`` bf16 DTensor — the
                kernel handles either layout.
            grid_dim / block_dim: explicit overrides.

        Returns:
            ``q_nope_pe``.
        """
        from .. import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(q_nope_abs)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (was pk.assemble_q_decode_sm100_layer).
        # q_nope_pe may be 3D (N, H, D_TOTAL) or 2D (N, H*D_TOTAL) — same
        # byte layout. The kernel handles either via its register codegen.
        from ...core import CyTBGraph
        from ...kernel import TBGraph

        assert q_nope_abs.num_dims == 3
        assert q_pe.num_dims == 3
        assert q_nope_pe.num_dims in (2, 3)
        assert q_nope_abs.dim(0) == q_pe.dim(0) == q_nope_pe.dim(0)
        H = q_nope_abs.dim(1)
        assert q_pe.dim(1) == H
        D_TOTAL = q_nope_abs.dim(2) + q_pe.dim(2)
        if q_nope_pe.num_dims == 3:
            assert q_nope_pe.dim(1) == H
            assert q_nope_pe.dim(2) == D_TOTAL
        else:
            assert q_nope_pe.dim(1) == H * D_TOTAL
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_nope_abs, (0, -1, -1), -1, True)
        tb_graph.new_input(q_pe,       (0, -1, -1), -1, True)
        tb_graph.new_input(q_nope_pe,  (0, -1, -1), -1, True)
        pk.kn_graph.customized([q_nope_abs, q_pe, q_nope_pe], tb_graph)
        params = [1] if self.pe_only else []
        pk.kn_graph.register_task(tb_graph, "assemble_q_decode_sm100", params)
        return q_nope_pe
