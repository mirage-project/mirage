"""Token-wise RMSNorm.

Backed by ``tasks/{ampere,hopper}/rmsnorm{,_hopper}.cuh``. Hopper and
Blackwell share the same task name ``"rmsnorm_hopper"`` (no Blackwell-
specific .cuh exists). Ampere uses ``"rmsnorm"`` and ``ampere/rmsnorm.cuh``.
Reduction is done in fp32, output cast back to input dtype (bf16-only).
Supports column slicing via ``process_dim``/``in_offset_elems``/``out_offset_elems``
(DeepSeek-style fused QKV-a normalization).
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .._base import MPKModule
from ...context import current_pk
from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class RMSNorm(MPKModule):
    """Token-wise RMSNorm with a learnable per-channel scale.

    Constraints (from the .cuh):
      * dtype: bf16 only (kernel hard-wires ``bfloat16``).
      * ``hidden_size`` must be a multiple of ``NUM_THREADS`` (128 on
        Ampere, 256 on Hopper/Blackwell) and large enough to satisfy
        ``hidden_size * sizeof(bf16) / NUM_THREADS >= 4``.
      * The compile path uses a hard-coded ``eps = 1e-6``; ``self.eps``
        only affects the PyTorch reference.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Faithful RMSNorm in fp32, cast back to input dtype."""
        input_dtype = x.dtype
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.to(torch.float32) * torch.rsqrt(variance + self.eps)
        return (x_normed.to(input_dtype) * self.weight).to(input_dtype)

    def auto_grid_dim(self, x_dt: DTensor) -> GridDim:
        """One CTA per token (kernel partitions on dim 0), capped at num_workers."""
        pk = current_pk()
        return (max(1, min(x_dt.dim(0), pk.num_workers)), 1, 1)

    def compile(
        self,
        x: DTensor,
        *,
        process_dim: Optional[int] = None,
        in_offset_elems: int = 0,
        out_offset_elems: int = 0,
        output: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
        name: Optional[str] = None,
    ) -> DTensor:
        """Register one ``rmsnorm`` (Ampere) or ``rmsnorm_hopper`` (SM90+) task.

        Tensor contract:
          x:      (B, hidden) bf16, per-row activations.
          weight: (hidden,)   bf16, learnable per-channel scale (auto-attached).
          output: (B, hidden) bf16, same shape as x.

        Notes: bf16-only; hidden must satisfy ``hidden % NUM_THREADS == 0`` and
        ``hidden * 2 / NUM_THREADS >= 4``. eps=1e-6 is hard-coded in codegen.
        Slice-mode kwargs (``process_dim``, ``in_offset_elems``,
        ``out_offset_elems``) enable column-slice norm into a wider fused buffer.
        """
        pk = current_pk()
        if x.num_dims != 2:
            raise ValueError(
                f"RMSNorm.compile expects a 2-D input DTensor; got num_dims={x.num_dims}"
            )

        prefix = self.prefix or "rmsnorm"
        if output is None:
            out_name = name if name is not None else f"{prefix}out"
            out_dt = pk.new_tensor(
                dims=(x.dim(0), x.dim(1)),
                dtype=x.dtype,
                name=out_name,
            )
        elif isinstance(output, torch.Tensor):
            out_name = name if name is not None else f"{prefix}out"
            out_dt = pk.attach_input(output, name=out_name)
        elif isinstance(output, DTensor):
            out_dt = output
        else:
            raise TypeError(
                "RMSNorm.compile output must be None, a torch.Tensor, "
                f"or a DTensor; got {type(output).__name__}"
            )

        w_dt = pk.attach_input(self.weight.data, name=f"{prefix}weight")

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert x.num_dims == 2
        assert out_dt.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x, (0, -1, -1), 1, True)
        tb_graph.new_input(w_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(out_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized([x, w_dt, out_dt], tb_graph)

        task_name = "rmsnorm_hopper" if pk.target_cc >= 90 else "rmsnorm"
        if (process_dim is None and in_offset_elems == 0
                and out_offset_elems == 0):
            pk.kn_graph.register_task(tb_graph, task_name)
        else:
            if process_dim is None:
                process_dim = out_dt.dim(1)
            pk.kn_graph.register_task(
                tb_graph,
                task_name,
                [process_dim, in_offset_elems, out_offset_elems],
            )
        return out_dt
