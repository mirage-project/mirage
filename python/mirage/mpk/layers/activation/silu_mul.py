"""SiLU-Mul activation modules.

Backed by ``tasks/{ampere,hopper}/silu_mul{,_hopper}.cuh``
(``register_task`` always uses the single name ``"silu_mul"``; both arches
share it. No blackwell variant exists.) The kernel reads
``d_mul = d_input + OUTPUT_SIZE`` per task, so the per-task layout is
**halved** (gate slab then up slab); for ``grid.x > 1`` the whole-tensor
layout becomes per-slab interleaved (upstream linear weights must be
shuffled with ``num_groups=grid.x`` to match — see
``test_qwen3_mlp_testmode.py``). bf16-only (kernels are hard-wired).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....core import DTensor
from .._base import BlockDim, GridDim, MPKModule


__all__ = ["SiluMul", "SiluMulLinearWithResidual"]


def _split_gate_up_halved(
    gateup: torch.Tensor, intermediate_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a fused ``(B, 2*intermediate)`` tensor into ``(gate, up)`` halved."""
    if gateup.dim() != 2:
        raise ValueError(
            f"gateup must be 2-D (B, 2*intermediate_size); got shape {tuple(gateup.shape)}"
        )
    if gateup.size(1) != 2 * intermediate_size:
        raise ValueError(
            f"gateup.size(1) must equal 2*intermediate_size={2 * intermediate_size}; "
            f"got {gateup.size(1)}"
        )
    return gateup[:, :intermediate_size], gateup[:, intermediate_size:]


class SiluMul(MPKModule):
    """``SiLU(gate) * up`` for a fused gate-up tensor.

    Input ``(B, 2*intermediate_size)`` bf16 → output ``(B, intermediate_size)``
    bf16. The kernel partitions on dim 1 (the output / intermediate axis);
    ``intermediate_size`` must be divisible by ``grid.x``. SiLU computed in
    fp32 internally.
    """

    def __init__(self, intermediate_size: int, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        self.intermediate_size = intermediate_size

    def forward(self, gateup: torch.Tensor) -> torch.Tensor:
        """``F.silu(gate) * up`` on halved layout (fp32 internally)."""
        gate, up = _split_gate_up_halved(gateup, self.intermediate_size)
        return (F.silu(gate.float()) * up.float()).to(gateup.dtype)

    def auto_grid_dim(self, gateup_dt) -> GridDim:
        """Stripe over the M*N output: largest divisor of
        ``intermediate_size`` <= ``min(intermediate_size // 64, num_workers)``.
        Matches qwen3 demo's ``num_tasks_gatedup // 2``.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        cap = max(1, int(getattr(pk, "num_workers", 1)))
        target = min(max(1, self.intermediate_size // 64), cap)
        gx = 1
        for d in range(1, target + 1):
            if self.intermediate_size % d == 0:
                gx = d
        return (gx, 1, 1)

    def compile(
        self,
        gateup: DTensor,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register a ``silu_mul`` task.

        Tensor contract:
          gateup: (B, 2 * intermediate_size) bf16, layout ``[gate(N) | up(N)]`` per task slab.
          output: (B, intermediate_size)     bf16, ``SiLU(gate) * up``.

        Notes: bf16-only; ``intermediate_size`` must be divisible by ``grid.x``;
        for ``grid.x > 1`` the whole-tensor layout is per-slab interleaved
        (upstream linear weights must be shuffled with ``num_groups=grid.x``).
        SiLU computed in fp32 internally.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if gateup.num_dims != 2:
            raise ValueError(
                f"SiluMul expects a 2-D gateup DTensor; got num_dims={gateup.num_dims}"
            )
        if gateup.dim(1) != 2 * self.intermediate_size:
            raise ValueError(
                "SiluMul: gateup.dim(1) must equal 2*intermediate_size="
                f"{2 * self.intermediate_size}; got {gateup.dim(1)}"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(gateup)
        if block_dim is None:
            block_dim = self.default_block_dim()

        batch_size = gateup.dim(0)
        if output is None:
            out_dt = pk.new_tensor(
                dims=(batch_size, self.intermediate_size),
                dtype=gateup.dtype,
                name=f"{self.prefix}silu_mul_out",
            )
        elif isinstance(output, torch.Tensor):
            out_dt = pk.attach_input(output, name=f"{self.prefix}silu_mul_out")
        else:
            out_dt = output

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert gateup.num_dims == 2
        assert out_dt.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(gateup, (1, -1, -1), 1, True)
        tb_graph.new_input(out_dt, (1, -1, -1), 1, True)
        pk.kn_graph.customized([gateup, out_dt], tb_graph)
        pk.kn_graph.register_task(tb_graph, "silu_mul")
        return out_dt


class SiluMulLinearWithResidual(MPKModule):
    """Fused ``F.linear(SiLU(gate)*up, w_down) + residual`` — currently broken.

    Would be backed by ``tasks/ampere/silu_mul_linear.cuh``, but the
    underlying codegen has the same int-vs-``void *`` mismatch that
    :class:`RMSNormLinear` hits; instantiation raises. Compose
    :class:`SiluMul` + :class:`LinearWithResidual` instead.
    """

    def __init__(
        self,
        intermediate_size: int,
        hidden_size: int,
        *,
        prefix: str = "",
    ) -> None:
        raise RuntimeError(
            "layers.SiluMulLinearWithResidual (wraps "
            "pk.silu_mul_linear_with_residual_layer) is broken in Mirage: "
            "the generated kernel call has an int-vs-void* argument-type "
            "mismatch (same root cause as RMSNormLinear). Compose "
            "`SiluMul` + `LinearWithResidual` instead."
        )
        super().__init__(prefix=prefix)
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.empty(hidden_size, intermediate_size))

    def forward(
        self,
        gateup: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """``F.linear(SiLU(gate)*up, weight) + residual`` (fp32 inside)."""
        if residual.dim() != 2 or residual.size(1) != self.hidden_size:
            raise ValueError(
                f"residual must have shape (B, {self.hidden_size}); "
                f"got {tuple(residual.shape)}"
            )
        gate, up = _split_gate_up_halved(gateup, self.intermediate_size)
        silu_out = F.silu(gate.float()) * up.float()
        out = F.linear(silu_out, self.weight.float())
        return (out + residual.float()).to(gateup.dtype)

    def auto_grid_dim(self, gateup_dt, residual_dt) -> GridDim:
        """Largest divisor of ``hidden_size`` <= ``min(hidden_size//64, num_workers)``."""
        from ... import context as _ctx

        pk = _ctx.current_pk()
        cap = max(1, int(getattr(pk, "num_workers", 1)))
        target = min(max(1, self.hidden_size // 64), cap)
        gx = 1
        for d in range(1, target + 1):
            if self.hidden_size % d == 0:
                gx = d
        return (gx, 1, 1)

    def compile(
        self,
        gateup: DTensor,
        residual: DTensor,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register a ``silu_mul_linear_with_residual`` task (unreachable — broken).

        Tensor contract (documented for reference; ``__init__`` raises):
          gateup:   (B, 2 * intermediate_size) bf16, ``[gate | up]``.
          weight:   (hidden_size, intermediate_size) bf16, down-projection (auto-attached).
          residual: (B, hidden_size)           bf16, fused-added to output.
          output:   (B, hidden_size)           bf16, ``F.linear(SiLU(gate)*up, w) + residual``.

        Notes: broken in Mirage (same int-vs-``void *`` codegen mismatch as
        :class:`RMSNormLinear`). Compose :class:`SiluMul` +
        :class:`LinearWithResidual` instead.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if gateup.num_dims != 2:
            raise ValueError(
                f"SiluMulLinearWithResidual expects 2-D gateup; got num_dims={gateup.num_dims}"
            )
        if gateup.dim(1) != 2 * self.intermediate_size:
            raise ValueError(
                "SiluMulLinearWithResidual: gateup.dim(1) must equal "
                f"2*intermediate_size={2 * self.intermediate_size}; got {gateup.dim(1)}"
            )
        if residual.num_dims != 2:
            raise ValueError(f"residual must be 2-D; got num_dims={residual.num_dims}")
        if residual.dim(1) != self.hidden_size:
            raise ValueError(
                f"residual.dim(1) must equal hidden_size={self.hidden_size}; "
                f"got {residual.dim(1)}"
            )
        if residual.dim(0) != gateup.dim(0):
            raise ValueError(
                f"residual.dim(0)={residual.dim(0)} must match gateup.dim(0)={gateup.dim(0)}"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(gateup, residual)
        if block_dim is None:
            block_dim = self.default_block_dim()

        weight_dt = pk.attach_input(self.weight, name=f"{self.prefix}weight")

        batch_size = gateup.dim(0)
        if output is None:
            out_dt = pk.new_tensor(
                dims=(batch_size, self.hidden_size),
                dtype=gateup.dtype,
                name=f"{self.prefix}silu_mul_linear_res_out",
            )
        elif isinstance(output, torch.Tensor):
            out_dt = pk.attach_input(
                output, name=f"{self.prefix}silu_mul_linear_res_out"
            )
        else:
            out_dt = output

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert gateup.num_dims == 2
        assert weight_dt.num_dims == 2
        assert residual.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(gateup, (-1, -1, -1), 1, True)
        tb_graph.new_input(weight_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(residual, (1, -1, -1), 1, True)
        tb_graph.new_input(out_dt, (1, -1, -1), 1, True)
        pk.kn_graph.customized([gateup, weight_dt, residual, out_dt], tb_graph)
        pk.kn_graph.register_task(tb_graph, "silu_mul_linear_with_residual")
        return out_dt
