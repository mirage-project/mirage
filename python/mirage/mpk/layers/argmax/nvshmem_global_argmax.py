"""Multi-GPU global argmax over per-rank partials (NVSHMEM cross-rank reduce).

Wraps :meth:`PersistentKernel.nvshmem_global_argmax_layer` (task
``"nvshmem_global_argmax"``). Code-gen emits
``kernel::nvshmem_global_argmax_from_partials_bf16`` from
``include/mirage/persistent_kernel/tasks/blackwell/nvshmem_argmax_sm100.cuh``
(re-included into the megakernel translation unit on all archs via
``blackwell/task_header.cuh``). Requires a real NVSHMEM-enabled
multi-GPU runtime; ``pk.world_size > 1`` and ``pk.use_nvshmem`` are
asserted at compile time.
"""
from __future__ import annotations

from typing import Any, Optional

import torch

from .._base import BlockDim, GridDim, MPKModule


__all__ = ["NVShmemGlobalArgmax"]


class NVShmemGlobalArgmax(MPKModule):
    """Tensor-parallel global argmax via NVSHMEM cross-rank reduce.

    Each rank produces per-chunk partials over its vocab shard; this op
    reduces across ranks and writes the global token id. Requires
    NVSHMEM and ``world_size > 1``; ``forward()`` is not implemented
    (no single-process oracle). ``vocab_offset``/``valid_vocab_size``/
    ``partial_chunk_size`` define the per-rank slice of the global vocab.
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
            "is cross-rank via NVSHMEM. Use layers.Argmax / ArgmaxPartial+"
            "ArgmaxReduce for single-rank unit tests."
        )

    def auto_grid_dim(self, partial_value: Any) -> GridDim:
        """``(batch_size, 1, 1)`` — one CTA per row; partial axis is internal."""
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
        """Register a ``nvshmem_global_argmax`` task; allocates NVSHMEM teams.

        Tensor contract:
          partial_value: (B, num_partial_tasks)  bf16, per-rank chunked max value.
          partial_index: (B, num_partial_tasks)  int64, per-rank chunk-local idx.
          scratch_value: (world_size, B)         bf16, NVSHMEM cross-rank scratch.
          scratch_index: (world_size, B)         int64, NVSHMEM cross-rank scratch.
          output:        (B, 1)                  int64, global vocab idx.

        Notes: requires ``pk.world_size > 1`` and ``pk.use_nvshmem`` (asserted).
        Calls ``allocate_nvshmem_teams(pk, grid_dim[0]*grid_dim[1]*grid_dim[2])``.
        Params baked into task: ``[world_size, mpi_rank, vocab_offset,
        valid_vocab_size, partial_chunk_size]``.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(partial_value)
        if block_dim is None:
            block_dim = self.default_block_dim()

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
