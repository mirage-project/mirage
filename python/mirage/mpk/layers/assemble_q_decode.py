"""Interleave per-head ``q_nope`` with ``q_pe`` for MLA decode.

Backed by ``tasks/blackwell/assemble_q_decode_sm100.cuh``
(``assemble_q_decode_sm100_task_impl``). Builds the per-head
``[nope | pe]`` layout the MLA attention kernel expects. Single-CTA per
token (kernel partitions over ``blockDim.x`` internally). Used in the
DSv3 ``MPK_DSV3_BMM=1`` decode Q path.

``pe_only=True`` writes only the trailing ``D_pe`` slice and leaves
``q_nope_abs`` untouched — used when the nope half was already written
into the destination buffer by a prior task (saves one TMA roundtrip).
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from ._base import BlockDim, GridDim, MPKModule


__all__ = ["AssembleQDecode"]


class AssembleQDecode(MPKModule):
    """Interleave per-head ``q_nope`` with ``q_pe`` into ``[nope | pe]``."""

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
        """Register an ``assemble_q_decode_sm100`` task — interleave ``[nope | pe]``.

        Tensor contract:
          q_nope_abs: (N, H, D_nope)             bf16, nope half of per-head Q.
          q_pe:       (N, H, D_pe)               bf16, RoPE-applied half of Q.
          q_nope_pe:  (N, H, D_nope + D_pe) OR
                      (N, H * (D_nope + D_pe))   bf16, destination ``[nope | pe]``.

        Notes: SM100-only; one CTA per token. ``pe_only=True`` writes only the
        trailing ``D_pe`` slice and leaves ``q_nope_abs`` untouched (used when
        the nope half was already written by a prior task — saves one TMA RTT).
        """
        from .. import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(q_nope_abs)
        if block_dim is None:
            block_dim = self.default_block_dim()

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
        tb_graph.new_input(q_pe, (0, -1, -1), -1, True)
        tb_graph.new_input(q_nope_pe, (0, -1, -1), -1, True)
        pk.kn_graph.customized([q_nope_abs, q_pe, q_nope_pe], tb_graph)
        params = [1] if self.pe_only else []
        pk.kn_graph.register_task(tb_graph, "assemble_q_decode_sm100", params)
        return q_nope_pe
