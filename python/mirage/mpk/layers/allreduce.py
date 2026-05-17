"""Multi-GPU allreduce (NVSHMEM / NVLINK SHARP fast paths).

Wraps :meth:`PersistentKernel.allreduce_layer`, which internally dispatches
to one of several NVSHMEM-backed implementations via
``auto_select_allreduce_implementation(world_size, mpi_rank)`` (see
``python/mirage/mpk/allreduce.py``). Each implementation registers
different task names (``nvshmem_tile_allreduce``,
``nvshmem_broadcast_then_reduce``, etc.) — this catalog module does not
need to know which.

Tensor contract
---------------

* ``input``    : ``(batch_size, hidden_size)`` bf16 — the per-rank
                 partial.
* ``buffer``   : ``(world_size, batch_size, hidden_size)`` bf16 —
                 scratch buffer the kernel uses for the cross-GPU
                 reduce.
* ``output``   : ``(batch_size, hidden_size)`` bf16 — the reduced result.
* ``residual`` : optional ``(batch_size, hidden_size)`` bf16 — when
                 provided, the kernel adds it into ``output`` (fused
                 epilogue).

Forward reference
-----------------

For ``world_size == 1`` (single-GPU): ``forward()`` is the identity
(plus the optional residual add). This is the only case we can
faithfully reference in pure PyTorch — for ``world_size > 1`` the
algebra is the same (sum across ranks) but each rank only sees its own
input shard, so a meaningful test requires NVSHMEM and we raise
``NotImplementedError`` instead.

Gating
------

``gate_mode != 0`` enables the gated-allreduce variant used by the
DSv3 MoE-residual fast path. Only the ``nvshmem_tile_allreduce``
backend implements it (see the pk method's runtime check); passing
``gate_mode != 0`` with a different backend will raise inside
``allreduce_layer``.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from ._base import BlockDim, GridDim, MPKModule


__all__ = ["AllReduce"]


class AllReduce(MPKModule):
    """Per-rank tensor-parallel allreduce.

    Args:
        gate_mode: Passes through to the pk method. 0 = no gate
            (standard allreduce). Non-zero values select the gated
            variant (DSv3 MoE-residual). Only honored by the
            ``nvshmem_tile_allreduce`` backend.
        prefix: Reserved. No parameters live here.
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
        """Single-GPU reference: identity (plus residual if given).

        For ``world_size > 1`` the reference would need access to the
        full set of per-rank shards — that's not feasible in a unit-
        test oracle, so we raise. The compile path still works for
        multi-GPU; only ``forward()`` is single-GPU.
        """
        from .. import context as _ctx

        # Try to detect world_size from the active PK if we're inside
        # a compile scope; otherwise fall back to single-GPU semantics
        # (pure unit-test path with no PK).
        try:
            pk = _ctx.current_pk()
            ws = int(getattr(pk, "world_size", 1))
        except RuntimeError:
            ws = 1
        if ws > 1:
            raise NotImplementedError(
                "AllReduce.forward (PyTorch reference) is only defined "
                f"for world_size == 1; got world_size={ws}. The compile "
                "path (which uses NVSHMEM teams) still works."
            )
        result = input
        if residual is not None:
            result = result + residual
        return result

    def auto_grid_dim(self, input: Any) -> GridDim:
        """Grid is selected per-implementation by
        ``auto_select_allreduce_implementation`` inside the pk method.
        We default to ``(num_workers, 1, 1)``; callers can override.
        """
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
        """Register the dispatched allreduce task(s) on the active PK.

        Args:
            input:    Per-rank partial DTensor ``(B, hidden)`` bf16.
            buffer:   Scratch DTensor ``(world_size, B, hidden)`` bf16.
            output:   Reduced result DTensor ``(B, hidden)`` bf16.
            residual: Optional DTensor ``(B, hidden)`` bf16 to fuse-add.
            grid_dim / block_dim: explicit overrides; ``None`` falls
                back to :meth:`auto_grid_dim` /
                :meth:`default_block_dim`.

        Returns:
            ``output``.
        """
        from .. import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(input)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined dispatch (was pk.allreduce_layer). Each NVSHMEM-backed
        # implementation registers its own task name(s) via
        # `register_tasks(pk, tensors=..., grid_dim=..., block_dim=...,
        # params=...)`; we just pick the implementation and forward.
        from ..multigpu import auto_select_allreduce_implementation

        assert input.num_dims == 2   # (batch_size, hidden_size)
        assert buffer.num_dims == 3  # (world_size, batch_size, hidden_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        if residual is not None:
            assert residual.num_dims == 2
            assert residual.dim(0) == output.dim(0)
            assert residual.dim(1) == output.dim(1)
        best_implementation = auto_select_allreduce_implementation(
            pk.world_size, pk.mpi_rank)
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
                    "nvshmem_tile_allreduce.")
            params.append(self.gate_mode)
        best_implementation.register_tasks(
            pk, tensors=tensors, grid_dim=grid_dim,
            block_dim=block_dim, params=params)
        return output
