"""BF16 split-K dense linear — qwen3 / DeepSeek V3 ``o_proj`` fast path.

Wraps :meth:`PersistentKernel.splitk_linear_layer` — task
``splitk_linear_sm100`` on Blackwell, ``splitk_linear_swapAB_hopper``
on Hopper. The split-K variant fans the K-axis reduction out across
``grid.y`` CTAs and uses ``tma_reduce_add_async`` to accumulate
partials into ``output``.

The kernel unconditionally **adds** the partial product onto whatever
``output`` already contains. The ``accumulate`` flag matches the pk
method:

* ``accumulate=True`` — caller owns ``output`` (e.g. a residual stream).
  The matmul is added on top, no pre-zero.
* ``accumulate=False`` — module prepends a ``tensor_init`` task that
  zeroes ``output`` before the linear runs, so the final result is a
  pure ``F.linear`` sum.

Tensor contract
---------------

* ``x``      : ``(batch_size, in_features)`` bf16. ``in_features`` is
               the per-rank K shard.
* ``weight`` : ``(out_features, in_features)`` bf16. Standard
               ``nn.Linear`` layout.
* ``output`` : ``(batch_size, out_features)`` bf16. **Required as a
               caller-allocated DTensor** (the kernel reduce-adds into
               it).

Grid heuristic
--------------

The qwen3 demo splits ``out_features // 128`` along grid.x and uses
``grid.y = 128 * 128 // out_features`` (a fixed ``128*128`` total
tile-count budget). We mirror that and cap at ``num_workers``.
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
        in_features:  K (reduction) axis. Must be divisible by the
            kernel TILE_SIZE — 128 on SM100, 64 on Hopper.
        out_features: N (output) axis. Must be divisible by 128 on
            SM100.
        accumulate:   See module docstring.
        prefix:       HF state_dict / tensor-name prefix.
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
        """``F.linear(x, weight)`` plus optional accumulate onto ``output``.

        For ``accumulate=True`` the caller passes the prior ``output``
        (typically the residual stream); the result is the sum.
        """
        result = F.linear(x, self.weight)
        if self.accumulate:
            if output is None:
                raise ValueError(
                    "SplitKLinear(accumulate=True).forward requires the "
                    "prior `output` tensor (the residual stream)."
                )
            result = result + output
        return result

    def auto_grid_dim(self, x: Any = None) -> GridDim:
        """``(out_features // 128, 128 * 128 // out_features, 1)``.

        Matches the qwen3 demo's pick (``demo/qwen3/demo.py`` and
        ``models/qwen3/builder.py:385``):

            grid_dim = (hidden_size // 128, 128 * 128 // hidden_size, 1)

        Cap at ``num_workers`` so the worker pool isn't oversubscribed.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        if self.out_features % 128 != 0:
            raise ValueError(
                f"SplitKLinear: out_features={self.out_features} must be "
                "a multiple of 128."
            )
        gx = self.out_features // 128
        # Default grid.y = 128*128 // out_features; clamp to >=1 and
        # ensure it doesn't oversubscribe workers.
        gy = max(1, (128 * 128) // max(1, self.out_features))
        # Ensure in_features // gy is a multiple of 128 (per-task K).
        while gy > 1 and (self.in_features // gy) % 128 != 0:
            gy -= 1
        gx = max(1, min(gx, int(pk.num_workers)))
        gy = max(1, min(gy, max(1, int(pk.num_workers) // gx)))
        return (gx, gy, 1)

    def default_block_dim(self) -> BlockDim:
        """SM100 kernel uses 256 threads; the Hopper swapAB variant the same."""
        return (256, 1, 1)

    def compile(
        self,
        x: Any,
        output: Any,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Register the ``splitk_linear`` task.

        Args:
            x:        Input DTensor ``(B, in_features)`` bf16.
            output:   Caller-allocated DTensor ``(B, out_features)`` bf16
                — see module docstring on the accumulate contract.
            grid_dim / block_dim: Explicit overrides.

        Returns:
            ``output`` (the kernel writes into it in-place).
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        w_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")

        # Inlined task registration (was pk.splitk_linear_layer). The kernel
        # always reduce-adds onto `output`; for accumulate=False we prepend
        # a tensor_init that zeroes `output` first.
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert x.num_dims == 2     # (batch_size, hidden_size / world_size)
        assert w_dt.num_dims == 2  # (hidden_size, hidden_size / world_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
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
