"""Dense bf16 linear projection (no bias, no residual).

Per-arch task kernel:
* SM80-89 Ampere : ``tasks/ampere/linear.cuh``               (``linear``)
* SM90   Hopper  : ``tasks/hopper/linear_swapAB_hopper.cuh`` (``linear_swapAB_hopper``)
* SM100  Blackwell: ``tasks/blackwell/linear_sm100_mpk.cuh`` (``linear_sm100``)
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import MPKModule


__all__ = ["Linear"]


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _grid_x_for_out_features(out_features: int) -> int:
    """Pick tile width 96 if divisible else 64."""
    if out_features % 96 == 0:
        return out_features // 96
    elif out_features % 64 == 0:
        return out_features // 64
    raise ValueError(
        f"Linear.auto_grid_dim: out_features={out_features} is not divisible "
        "by 96 or 64. Pass grid_dim explicitly to compile()."
    )


class Linear(MPKModule):
    """Plain bf16 dense linear projection: ``out = x @ weight.T``.

    Args:
        in_features:  Reduction dim. Multiple of 128 (Ampere) / 64 (Hopper/Blackwell).
        out_features: Output feature dim. Multiple of 96 or 64 for the auto-grid.
        bias:         Must be False (no kernel path); raises NotImplementedError.
        prefix:       state_dict / tensor-name prefix.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if bias:
            raise NotImplementedError(
                "Linear(bias=True) is not supported by the underlying kernel. "
                "Fold the bias into the weight at load time, or add it as a "
                "separate layers.add() call."
            )
        self.in_features = in_features
        self.out_features = out_features
        # PyTorch nn.Linear weight layout: (out_features, in_features), bf16.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``F.linear(x, self.weight)`` = ``x @ weight.T``."""
        return F.linear(x, self.weight)

    def auto_grid_dim(self, x_dt: Any) -> GridDim:
        """One CTA per output-column tile (width 96 or 64) along grid.x.

        The kernel is one-CTA-per-output-tile and sweeps M internally, so
        ``grid.x*grid.y*grid.z`` is bounded by ``out_features // tile_width``
        and may sit below ``num_workers`` for small out_features.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        gx = _grid_x_for_out_features(self.out_features)
        gx = max(1, min(gx, int(pk.num_workers)))
        return (gx, 1, 1)

    def compile(
        self,
        x: Any,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Register a ``linear`` task computing ``x @ weight.T``.

        Tensor contract:
          x:      (B, in_features) bf16, row-major contiguous. A operand, partition (-1,-1,-1).
          weight: (out_features, in_features) bf16, row-major. B operand, partition (0,-1,-1) — sharded along grid.x.
          output: (B, out_features) bf16, row-major. partition (1,-1,-1). None=alloc, Tensor=host-bind, else use as-is.

        Notes: in_features mult of 128 (Ampere) / 64 (Hopper/Blackwell); out_features mult of 96 or 64 for auto-grid.
        TMA-aligned; one CTA per output-column tile sweeps M.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        w_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")

        batch_size = x.dim(0)
        if output is None:
            out_dt = pk.new_tensor(
                dims=(batch_size, self.out_features),
                dtype=x.dtype,
                name=f"{self.prefix}linear_out",
            )
        elif isinstance(output, torch.Tensor):
            out_dt = pk.attach_input(output, name=f"{self.prefix}linear_out")
        else:
            out_dt = output

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert x.num_dims == 2
        assert w_dt.num_dims == 2
        assert out_dt.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x, (-1, -1, -1), 1, True)
        tb_graph.new_input(w_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(out_dt, (1, -1, -1), -1, True)
        pk.kn_graph.customized([x, w_dt, out_dt], tb_graph)

        if 100 <= pk.target_cc < 120:
            pk.kn_graph.register_task(tb_graph, "linear_sm100")
        elif 90 <= pk.target_cc < 100:
            pk.kn_graph.register_task(tb_graph, "linear_swapAB_hopper")
        elif 80 <= pk.target_cc < 90:
            pk.kn_graph.register_task(tb_graph, "linear")
        else:
            raise RuntimeError(
                f"Linear.compile: unsupported compute capability "
                f"{pk.target_cc}. Supported: SM80-89 (Ampere), SM90 "
                f"(Hopper), SM100-119 (Blackwell)."
            )
        return out_dt
