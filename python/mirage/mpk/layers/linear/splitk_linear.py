"""BF16 split-K dense linear.

Per-arch task kernel:
* SM90  Hopper   : ``tasks/hopper/linear_swapAB_hopper.cuh``  (``splitk_linear_swapAB_hopper``)
* SM100 Blackwell: ``tasks/blackwell/linear_sm100_mpk.cuh``   (``splitk_linear_sm100``)

The kernel reduce-adds partial products onto ``output`` via
``tma_reduce_add_async``; we prepend a ``tensor_init`` zeroer when
``accumulate=False``.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import BlockDim, GridDim, MPKModule


__all__ = ["SplitKLinear"]


class SplitKLinear(MPKModule):
    """BF16 split-K dense linear.

    Args:
        in_features:  K (reduction) axis. Multiple of TILE_SIZE (128 SM100, 64 Hopper).
        out_features: N (output) axis. Multiple of 128 on SM100.
        accumulate:   True = matmul is added onto caller-owned ``output``;
                      False = a tensor_init zero is inserted first.
        prefix:       state_dict / tensor-name prefix.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        accumulate: bool,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.in_features = in_features
        self.out_features = out_features
        self.accumulate = accumulate
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.bfloat16)
        )

    def forward(
        self,
        x: torch.Tensor,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """``F.linear(x, weight)`` + (optional) accumulate onto ``output``."""
        result = F.linear(x, self.weight)
        if self.accumulate:
            if output is None:
                raise ValueError(
                    "SplitKLinear(accumulate=True).forward requires `output`."
                )
            result = result + output
        return result

    def auto_grid_dim(self, x: Any = None) -> GridDim:
        """``(out_features // 128, 128*128 // out_features, 1)``: N along grid.x, K-split along grid.y.

        gy is shrunk so per-task K stays a multiple of 128, then capped so
        total CTAs ``<= num_workers``.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        if self.out_features % 128 != 0:
            raise ValueError(
                f"SplitKLinear: out_features={self.out_features} must be "
                "a multiple of 128."
            )
        gx = self.out_features // 128
        gy = max(1, (128 * 128) // max(1, self.out_features))
        while gy > 1 and (self.in_features // gy) % 128 != 0:
            gy -= 1
        gx = max(1, min(gx, int(pk.num_workers)))
        gy = max(1, min(gy, max(1, int(pk.num_workers) // gx)))
        return (gx, gy, 1)

    def default_block_dim(self) -> BlockDim:
        return (256, 1, 1)

    def compile(
        self,
        x: Any,
        output: Any,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Register the ``splitk_linear_*`` task: K-split GEMM with reduce-add into ``output``.

        Tensor contract:
          x:      (B, in_features) bf16, row-major. A operand, partition (-1,1,-1) — K sharded along grid.y.
          weight: (out_features, in_features) bf16, row-major. B operand, partition (0,1,-1) — N along grid.x, K along grid.y.
          output: (B, out_features) bf16, row-major, caller-allocated. partition (1,-1,-1); kernel TMA reduce-adds partials.

        Notes: out_features mult of 128 (SM100); per-task K mult of TILE_SIZE (128 SM100 / 64 Hopper); TMA-aligned.
        accumulate=False prepends ``tensor_init_layer`` to zero ``output`` before reduce-add; True keeps caller bias.
        grid.y = K-split count.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        w_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert x.num_dims == 2
        assert w_dt.num_dims == 2
        assert output.num_dims == 2
        if not self.accumulate:
            pk.tensor_init_layer(
                target=output,
                dummy=x,
                grid_dim=grid_dim,
                block_dim=block_dim,
                dummy_input_map=(-1, 1, -1),
                target_input_map=(1, -1, -1),
            )
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x,      (-1, 1, -1), 1, True)
        tb_graph.new_input(w_dt,   (0, 1, -1),  1, True)
        tb_graph.new_input(output, (1, -1, -1), -1, True)
        pk.kn_graph.customized([x, w_dt, output], tb_graph)

        if pk.target_cc == 100:
            pk.kn_graph.register_task(tb_graph, "splitk_linear_sm100")
        elif pk.target_cc == 90:
            pk.kn_graph.register_task(tb_graph, "splitk_linear_swapAB_hopper")
        else:
            raise RuntimeError(
                f"SplitKLinear.compile: unsupported compute capability "
                f"{pk.target_cc}. Supported: SM90 (Hopper), SM100 (Blackwell)."
            )
        return output
