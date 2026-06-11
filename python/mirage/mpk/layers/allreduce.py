"""Multi-GPU allreduce with NVSHMEM-backed implementations.

Auto-dispatches via ``auto_select_allreduce_implementation(world_size,
mpi_rank)`` (see ``python/mirage/mpk/multigpu/``) to one of several
NVSHMEM kernels (e.g. ``nvshmem_tile_allreduce``,
``nvshmem_broadcast_then_reduce``). Each implementation registers its
own task name(s) on the active PK; the kernels live alongside the
allreduce header in ``tasks/{ampere,hopper,blackwell}/allreduce.cuh``
and the NVSHMEM utilities under ``python/mirage/mpk/multigpu``.
``gate_mode != 0`` enables the gated variant (DSv3 MoE-residual);
only ``nvshmem_tile_allreduce`` honors it.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from ._base import BlockDim, GridDim, MPKModule


__all__ = ["AllReduce"]


class AllReduce(MPKModule):
    """Per-rank tensor-parallel allreduce.

    Tensor contract:
      * ``input``    : ``(B, hidden)`` bf16 per-rank partial.
      * ``buffer``   : ``(world_size, B, hidden)`` bf16 scratch.
      * ``output``   : ``(B, hidden)`` bf16 reduced result.
      * ``residual`` : optional ``(B, hidden)`` bf16, fused-added.
    """

    def __init__(
        self,
        *,
        gate_mode: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.gate_mode = gate_mode

    def forward(
        self,
        input: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single-GPU reference: identity (plus residual). Raises for ws>1."""
        from .. import context as _ctx

        try:
            pk = _ctx.current_pk()
            ws = int(getattr(pk, "world_size", 1))
        except RuntimeError:
            ws = 1
        if ws > 1:
            raise NotImplementedError(
                "AllReduce.forward (PyTorch reference) is only defined for "
                f"world_size == 1; got world_size={ws}. The compile path "
                "(NVSHMEM teams) still works."
            )
        result = input
        if residual is not None:
            result = result + residual
        return result

    def auto_grid_dim(self, input: Any) -> GridDim:
        """``(num_workers, 1, 1)`` — implementations may re-tile internally."""
        from .. import context as _ctx

        pk = _ctx.current_pk()
        return (int(pk.num_workers), 1, 1)

    def compile(
        self,
        input: Any,
        buffer: Any,
        output: Any,
        *,
        residual: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Dispatch to the best NVSHMEM allreduce backend and register its task(s).

        Tensor contract:
          input:    (B, hidden)             bf16, per-rank partial.
          buffer:   (world_size, B, hidden) bf16, NVSHMEM scratch (symmetric).
          output:   (B, hidden)             bf16, reduced (and optionally
                    residual-added) result.
          residual: (B, hidden) bf16 optional, fused-added to output.

        Notes: ``world_size > 1`` requires NVSHMEM teams; backend selected by
        ``auto_select_allreduce_implementation``. ``gate_mode != 0`` (DSv3
        MoE-residual gating) is only honored by ``nvshmem_tile_allreduce``.
        """
        from .. import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(input)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ..multigpu import auto_select_allreduce_implementation

        assert input.num_dims == 2
        assert buffer.num_dims == 3
        assert output.num_dims == 2
        if residual is not None:
            assert residual.num_dims == 2
            assert residual.dim(0) == output.dim(0)
            assert residual.dim(1) == output.dim(1)
        best_implementation = auto_select_allreduce_implementation(
            pk.world_size, pk.mpi_rank
        )
        tensors = {
            "input": input,
            "buffer": buffer,
            "output": output,
        }
        if residual is not None:
            tensors["residual"] = residual
        params = [pk.world_size, pk.mpi_rank]
        if self.gate_mode:
            if getattr(best_implementation, "name", "") != "nvshmem_tile_allreduce":
                raise RuntimeError(
                    "Gated allreduce is currently implemented only for "
                    "nvshmem_tile_allreduce."
                )
            params.append(self.gate_mode)
        best_implementation.register_tasks(
            pk,
            tensors=tensors,
            grid_dim=grid_dim,
            block_dim=block_dim,
            params=params,
        )
        return output
