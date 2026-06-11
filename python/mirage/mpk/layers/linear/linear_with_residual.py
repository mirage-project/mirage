"""Fused ``F.linear(x, W) + residual`` MPK module.

Per-arch task kernel:
* SM80-89 Ampere : ``tasks/ampere/linear.cuh``               (``linear_with_residual``)
* SM90   Hopper  : ``tasks/hopper/linear_swapAB_hopper.cuh`` (``linear_swapAB_with_residual_hopper``)
* SM100  Blackwell: ``tasks/blackwell/linear_sm100_mpk.cuh`` (``linear_with_residual_sm100``)
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import MPKModule


__all__ = ["LinearWithResidual"]


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class LinearWithResidual(MPKModule):
    """Fused ``(x @ weight.T) + residual``; no bias.

    Args:
        in_features:  Inner (contraction) dim of the weight.
        out_features: Outer dim of the weight; width of residual and output.
            Multiple of 64 on Ampere/Hopper, 128 on Blackwell SM100.
        prefix:       state_dict / tensor-name prefix.

    On non-root TP ranks the residual add is masked off via task param[0]
    so the residual is added exactly once across the shard.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.in_features = in_features
        self.out_features = out_features
        # bf16 (out_features, in_features); kernels read this layout directly
        # on Ampere/Blackwell, transposed via swapAB on Hopper.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """``F.linear(x, W) + residual``."""
        return F.linear(x, self.weight) + residual

    def auto_grid_dim(self, x_dt=None, residual_dt=None) -> GridDim:
        """``grid.x = out_features // OUTPUT_ATOM_SIZE`` (128 on SM100, else 64).

        The kernel is one-CTA-per-output-column-slab and sweeps M internally,
        so total CTAs may sit below ``num_workers`` for small out_features.
        """
        from ... import context as _ctx
        pk = _ctx.current_pk()
        atom = 128 if pk.target_cc >= 100 else 64
        if self.out_features % atom != 0:
            raise ValueError(
                f"LinearWithResidual: out_features={self.out_features} is "
                f"not divisible by the kernel output atom size {atom} "
                f"(target_cc={pk.target_cc})."
            )
        gx = self.out_features // atom
        gx = max(1, min(gx, int(pk.num_workers)))
        return (gx, 1, 1)

    def compile(
        self,
        x,
        residual,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ):
        """Register a ``linear_with_residual`` task: ``x @ weight.T + residual``.

        Tensor contract:
          x:        (B, in_features) bf16, row-major. A operand, partition (-1,-1,-1).
          weight:   (out_features, in_features) bf16, row-major. B operand, partition (0,-1,-1) — sharded along grid.x.
          residual: (B, out_features) bf16, row-major. partition (1,-1,-1).
          output:   (B, out_features) bf16, row-major. partition (1,-1,-1). None=alloc, Tensor=host-bind, else use as-is.

        Notes: out_features mult of OUTPUT_ATOM_SIZE (128 on SM100, else 64); TMA-aligned.
        params[0]=enable_residual (0 on non-root TP ranks so residual is added once across the shard).
        """
        from ... import context as _ctx
        pk = _ctx.current_pk()

        w_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")

        out_features = self.out_features
        batch_size = residual.dim(0)
        if output is None:
            out_dt = pk.new_tensor(
                dims=(batch_size, out_features),
                dtype=residual.dtype,
                name=f"{self.prefix}linear_with_residual_out",
            )
        elif isinstance(output, torch.Tensor):
            out_dt = pk.attach_input(
                output,
                name=f"{self.prefix}linear_with_residual_out",
            )
        else:
            out_dt = output

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x, residual)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert x.num_dims == 2
        assert w_dt.num_dims == 2
        assert residual.num_dims == 2
        assert out_dt.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x, (-1, -1, -1), 1, True)
        tb_graph.new_input(w_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), -1, True)
        tb_graph.new_input(out_dt, (1, -1, -1), -1, True)
        pk.kn_graph.customized([x, w_dt, residual, out_dt], tb_graph)

        # Task param[0] = enable_residual. Disabled on non-root TP ranks.
        enable_residual = 1
        if pk.world_size > 1 and pk.mpi_rank != 0:
            enable_residual = 0
        params = [enable_residual]

        if 100 <= pk.target_cc < 120:
            pk.kn_graph.register_task(
                tb_graph, "linear_with_residual_sm100", params
            )
        elif 90 <= pk.target_cc < 100:
            pk.kn_graph.register_task(
                tb_graph, "linear_swapAB_with_residual_hopper", params
            )
        elif 80 <= pk.target_cc < 90:
            pk.kn_graph.register_task(tb_graph, "linear_with_residual")
        else:
            raise RuntimeError(
                f"LinearWithResidual.compile: unsupported compute "
                f"capability {pk.target_cc}. Supported: SM80-89 (Ampere), "
                f"SM90 (Hopper), SM100-119 (Blackwell)."
            )
        return out_dt
