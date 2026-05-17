"""Multi-GPU global argmax over per-rank partials.

Wraps :meth:`PersistentKernel.nvshmem_global_argmax_layer` — task
``nvshmem_global_argmax``. Used by tensor-parallel greedy decode: each
rank produces a local ``ArgmaxPartial`` over its vocab shard; this
op then reduces across ranks via NVSHMEM and writes the global token
id.

The pk method internally allocates NVSHMEM teams via
``allocate_nvshmem_teams(pk, grid.x * grid.y * grid.z)``; the catalog
module forwards through.

Tensor contract
---------------

* ``partial_value``  : ``(B, num_partial_tasks)`` bf16 — local max per
                       chunk (from :class:`ArgmaxPartial`).
* ``partial_index``  : ``(B, num_partial_tasks)`` int64 — chunk-local
                       argmax index.
* ``scratch_value``  : ``(world_size, B)`` bf16 — NVSHMEM scratch
                       buffer for the cross-rank reduce.
* ``scratch_index``  : ``(world_size, B)`` int64 — same.
* ``output``         : ``(B, 1)`` int64 — global token id.

The ``vocab_offset`` / ``valid_vocab_size`` / ``partial_chunk_size``
parameters define the per-rank slice of the global vocab the partials
cover.

Forward reference
-----------------

Multi-GPU only — for ``world_size == 1`` we could reduce trivially,
but the typical caller would just use :class:`ArgmaxReduce` in that
case. We raise ``NotImplementedError`` from ``forward()`` because the
multi-rank reduction depends on NVSHMEM and on per-rank shards that
aren't visible to a single-process oracle.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from .._base import BlockDim, GridDim, MPKModule


__all__ = ["NVShmemGlobalArgmax"]


class NVShmemGlobalArgmax(MPKModule):
    """Tensor-parallel global argmax via NVSHMEM cross-rank reduce.

    Args:
        vocab_offset: Per-rank starting vocab index (the global vocab
            id is ``vocab_offset + local_idx``).
        valid_vocab_size: Number of valid vocab entries this rank owns
            (the local partial tensor may be padded; this is the real
            count).
        partial_chunk_size: ``CHUNK_SIZE`` used by the upstream
            :class:`ArgmaxPartial` (so the global reduction can recover
            the local position).
        prefix: Reserved. No parameters live here.
    """

    def __init__(
        self,
        vocab_offset: int,
        valid_vocab_size: int,
        partial_chunk_size: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.vocab_offset = vocab_offset
        self.valid_vocab_size = valid_vocab_size
        self.partial_chunk_size = partial_chunk_size

    def forward(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError(
            "NVShmemGlobalArgmax.forward is not implemented: the reduction "
            "is cross-rank via NVSHMEM, which requires a real multi-GPU "
            "runtime. For single-rank unit tests, use layers.Argmax / "
            "ArgmaxPartial+ArgmaxReduce instead."
        )

    def auto_grid_dim(self, partial_value: Any) -> GridDim:
        """``(batch_size, 1, 1)`` — one CTA per row.

        ``num_partial_tasks`` (the second axis of the partial tensors)
        is internal to the reduction; the kernel iterates over it
        inside one CTA per row.
        """
        return (partial_value.dim(0), 1, 1)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

    def compile(
        self,
        partial_value: Any,
        partial_index: Any,
        scratch_value: Any,
        scratch_index: Any,
        output: Any,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Register a ``nvshmem_global_argmax`` task on the active PK.

        Args:
            partial_value, partial_index: per-rank partials from
                :class:`ArgmaxPartial`.
            scratch_value, scratch_index: NVSHMEM scratch buffers
                ``(world_size, B)``.
            output: ``(B, 1)`` int64 DTensor — global token id.
            grid_dim / block_dim: explicit overrides.

        Returns:
            ``output``.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(partial_value)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (was pk.nvshmem_global_argmax_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph
        from ...multigpu import allocate_nvshmem_teams

        assert pk.world_size > 1
        assert pk.use_nvshmem
        assert partial_value.num_dims == 2  # (batch_size, num_partial_tasks)
        assert partial_index.num_dims == 2  # (batch_size, num_partial_tasks)
        assert scratch_value.num_dims == 2  # (world_size, batch_size)
        assert scratch_index.num_dims == 2  # (world_size, batch_size)
        assert output.num_dims == 2  # (batch_size, 1)
        assert partial_value.dim(0) == partial_index.dim(0)
        assert partial_value.dim(1) == partial_index.dim(1)
        assert scratch_value.dim(0) == pk.world_size
        assert scratch_index.dim(0) == pk.world_size
        assert scratch_value.dim(1) == partial_value.dim(0)
        assert scratch_index.dim(1) == partial_value.dim(0)
        assert self.partial_chunk_size > 0
        assert 0 <= self.valid_vocab_size <= partial_value.dim(1) * self.partial_chunk_size

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(partial_value, (1, 0, -1), -1, True)
        tb_graph.new_input(partial_index, (1, 0, -1), -1, True)
        tb_graph.new_input(scratch_value, (-1, -1, -1), -1, True)
        tb_graph.new_input(scratch_index, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        pk.kn_graph.customized(
            [partial_value, partial_index, scratch_value, scratch_index, output],
            tb_graph,
        )
        pk.kn_graph.register_task(
            tb_graph,
            "nvshmem_global_argmax",
            [
                pk.world_size,
                pk.mpi_rank,
                self.vocab_offset,
                self.valid_vocab_size,
                self.partial_chunk_size,
            ],
        )
        allocate_nvshmem_teams(pk, grid_dim[0] * grid_dim[1] * grid_dim[2])
        return output
