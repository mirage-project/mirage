"""MoE SiLU-Mul activation (2-D or 3-D input).

Wraps the ``moe_silu_mul`` task (kernel:
``include/mirage/persistent_kernel/tasks/blackwell/silu_mul.cuh`` via
the moe task variant). One kernel serves both 3-D ``(B, K, 2*inter)``
and 2-D ``(M, 2*inter)`` input layouts; the trailing axis is
``[gate | up]``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["MoESiluMul"]


class MoESiluMul(MPKModule):
    """SiLU-Mul over the trailing ``2*intermediate_size`` axis.

    Accepts 3-D ``(B, K, 2*inter)`` (per-token, per-slot — feeds W2)
    and 2-D ``(M, 2*inter)`` (post-permute grouped path) layouts.
    Computes ``SiLU(gate) * up`` in fp32, returns the input dtype.
    """

    def __init__(self, intermediate_size: int, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        self.intermediate_size = intermediate_size

    def forward(self, gateup: torch.Tensor) -> torch.Tensor:
        """``SiLU(gate) * up`` in fp32 over the trailing axis."""
        if gateup.dim() not in (2, 3):
            raise ValueError(
                f"MoESiluMul.forward expects 2-D or 3-D input; got {gateup.dim()}-D"
            )
        if gateup.size(-1) != 2 * self.intermediate_size:
            raise ValueError(
                f"trailing dim must equal 2*intermediate_size="
                f"{2*self.intermediate_size}; got {gateup.size(-1)}"
            )
        gate = gateup[..., : self.intermediate_size]
        up = gateup[..., self.intermediate_size :]
        return (F.silu(gate.float()) * up.float()).to(gateup.dtype)

    def auto_grid_dim(self, input_dt: DTensor) -> GridDim:
        """3-D: ``(B, K, 1)`` — one CTA per (token, slot).
        2-D: ``(min(num_workers, M), 1, 1)`` — stripe over M to saturate workers.
        """
        from ... import context as _ctx
        pk = _ctx.current_pk()
        if input_dt.num_dims == 3:
            return (input_dt.dim(0), input_dt.dim(1), 1)
        m_total = input_dt.dim(0)
        gx = max(1, min(int(getattr(pk, "num_workers", 1)), m_total))
        return (gx, 1, 1)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

    def compile(
        self,
        gateup: DTensor,
        output: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``moe_silu_mul`` — SiLU(gate) * up over the trailing axis.

        Tensor contract:
          gateup: (B, num_experts_per_tok, 2*intermediate_size) bf16 (3-D, per-token-per-slot)
            OR (M_total, 2*intermediate_size) bf16 (2-D, post-permute grouped path).
            Trailing axis layout = [gate(intermediate) | up(intermediate)].
          output: (B, num_experts_per_tok, intermediate_size) bf16 (3-D)
            OR (M_total, intermediate_size) bf16 (2-D). Rank MUST match gateup.

        Notes: 3-D grid is (B, K, 1) one CTA per token-slot; 2-D grid stripes M over
        num_workers. ``num_experts_per_tok`` is implicit (=1 for the 2-D layout).
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        if gateup.num_dims not in (2, 3):
            raise ValueError(
                f"MoESiluMul expects 2-D or 3-D gateup; got num_dims={gateup.num_dims}"
            )
        if gateup.num_dims != output.num_dims:
            raise ValueError(
                "MoESiluMul: gateup and output must have matching rank "
                f"({gateup.num_dims} vs {output.num_dims})"
            )
        if gateup.dim(gateup.num_dims - 1) != 2 * self.intermediate_size:
            raise ValueError(
                f"gateup trailing dim must equal 2*intermediate_size="
                f"{2*self.intermediate_size}; got {gateup.dim(gateup.num_dims - 1)}"
            )
        if output.dim(output.num_dims - 1) != self.intermediate_size:
            raise ValueError(
                f"output trailing dim must equal intermediate_size="
                f"{self.intermediate_size}; got {output.dim(output.num_dims - 1)}"
            )

        grid_dim = grid_dim or self.auto_grid_dim(gateup)
        block_dim = block_dim or self.default_block_dim()
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        if gateup.num_dims == 3:
            tb_graph.new_input(gateup, (0, 1, -1), -1, True)
            tb_graph.new_input(output, (0, 1, -1), -1, True)
        else:
            tb_graph.new_input(gateup, (0, -1, -1), -1, True)
            tb_graph.new_input(output, (0, -1, -1), -1, True)
        pk.kn_graph.customized([gateup, output], tb_graph)
        pk.kn_graph.register_task(tb_graph, "moe_silu_mul")
        return output
