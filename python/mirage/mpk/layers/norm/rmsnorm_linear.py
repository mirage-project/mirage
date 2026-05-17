"""Fused RMSNorm + Linear (broken in Mirage; instantiation raises).

Would be backed by ``tasks/ampere/norm_linear.cuh`` and the
``rmsnorm_linear`` task — but the compiler generates an int-vs-void*
argument-type mismatch in ``src/kernel/task_register.cc`` that nvcc
rejects. The unit test (``test_rmsnorm_linear_testmode.py``) skips this
module; callers should compose :class:`RMSNorm` + :class:`Linear`.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import MPKModule
from ...context import current_pk
from ....core import DTensor


__all__ = ["RMSNormLinear"]


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _grid_x_for_out_features(out_features: int) -> int:
    """Tile-width selector mirroring ``grid_for_rmsnorm_linear_layer``."""
    if out_features / 96 > 400:
        if out_features % 256 != 0:
            raise ValueError(
                f"RMSNormLinear.auto_grid_dim: out_features={out_features} "
                "is in the 'too big' regime (>96*400) and is not divisible "
                "by 256. Pass grid_dim explicitly to compile()."
            )
        return out_features // 256
    if out_features % 96 == 0:
        return out_features // 96
    if out_features % 64 == 0:
        return out_features // 64
    raise ValueError(
        f"RMSNormLinear.auto_grid_dim: out_features={out_features} is not "
        "divisible by 96 or 64. Pass grid_dim explicitly to compile()."
    )


class RMSNormLinear(MPKModule):
    """Fused ``F.linear(RMSNorm(x), weight_linear)`` — currently broken.

    Instantiation raises ``RuntimeError`` because the underlying
    ``pk.rmsnorm_linear_layer`` codegen path produces an
    int-vs-``void *`` argument mismatch that nvcc rejects. Use
    :class:`RMSNorm` + :class:`Linear` composed instead.
    """

    def __init__(
        self,
        hidden_size: int,
        out_features: int,
        eps: float = 1e-6,
        *,
        prefix: str = "",
    ) -> None:
        raise RuntimeError(
            "layers.RMSNormLinear (wraps pk.rmsnorm_linear_layer) is "
            "broken in Mirage: the generated kernel call has an "
            "int-vs-void* argument-type mismatch (see "
            "src/kernel/task_register.cc — the fix belongs in the Mirage "
            "compiler). Compose `RMSNorm` + `Linear` instead."
        )
        super().__init__(prefix=prefix)
        self.hidden_size = hidden_size
        self.out_features = out_features
        self.eps = eps
        self.weight_norm = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16)
        )
        self.weight_linear = nn.Parameter(
            torch.empty(out_features, hidden_size, dtype=torch.bfloat16)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """RMSNorm(x) then F.linear (no bias)."""
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.to(torch.float32) * torch.rsqrt(variance + self.eps)
        x_normed = (x_normed.to(input_dtype) * self.weight_norm).to(input_dtype)
        return F.linear(x_normed, self.weight_linear)

    def auto_grid_dim(self, x_dt) -> GridDim:
        """Tile out_features per the qwen3 helper; capped at num_workers."""
        pk = current_pk()
        gx = max(1, min(_grid_x_for_out_features(self.out_features),
                        int(pk.num_workers)))
        return (gx, 1, 1)

    def compile(
        self,
        x: DTensor,
        *,
        output: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one fused ``rmsnorm_linear`` task (unreachable — see class doc).

        Tensor contract (documented for reference; ``__init__`` raises):
          x:             (B, hidden)         bf16, per-row activations.
          weight_norm:   (hidden,)           bf16, RMSNorm scale (auto-attached).
          weight_linear: (out_features, hidden) bf16, linear weight (auto-attached).
          output:        (B, out_features)   bf16, fused norm-linear result.

        Notes: broken in Mirage — codegen emits an int-vs-``void *`` argument
        mismatch in ``src/kernel/task_register.cc`` that nvcc rejects. Compose
        :class:`RMSNorm` + :class:`Linear` instead.
        """
        pk = current_pk()
        if x.num_dims != 2:
            raise ValueError(
                f"RMSNormLinear.compile expects a 2-D input DTensor; got num_dims={x.num_dims}"
            )

        wn_dt = pk.attach_input(self.weight_norm, name=f"{self.prefix}weight_norm")
        wl_dt = pk.attach_input(self.weight_linear, name=f"{self.prefix}weight_linear")

        batch_size = x.dim(0)
        if output is None:
            out_dt = pk.new_tensor(
                dims=(batch_size, self.out_features),
                dtype=x.dtype,
                name=f"{self.prefix}rmsnorm_linear_out",
            )
        elif isinstance(output, torch.Tensor):
            out_dt = pk.attach_input(output, name=f"{self.prefix}rmsnorm_linear_out")
        elif isinstance(output, DTensor):
            out_dt = output
        else:
            raise TypeError(
                "RMSNormLinear.compile output must be None, a torch.Tensor, "
                f"or a DTensor; got {type(output).__name__}"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert x.num_dims == 2
        assert wl_dt.num_dims == 2
        assert out_dt.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x, (-1, -1, -1), 1, True)
        tb_graph.new_input(wn_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(wl_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(out_dt, (1, -1, -1), -1, True)
        pk.kn_graph.customized([x, wn_dt, wl_dt, out_dt], tb_graph)
        pk.kn_graph.register_task(tb_graph, "rmsnorm_linear")
        return out_dt
