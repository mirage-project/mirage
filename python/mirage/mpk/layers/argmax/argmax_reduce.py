"""Split-reduce argmax — the *reduce* half.

Wraps :meth:`PersistentKernel.argmax_reduce_layer`. Code-gen emits
``argmax_reduce_kernel`` from
``include/mirage/persistent_kernel/tasks/ampere/argmax.cuh`` (Ampere/Hopper)
or ``argmax_reduce_sm100_kernel`` from
``tasks/blackwell/argmax_sm100.cuh`` (Blackwell). Consumes the two
:class:`ArgmaxPartial` outputs and emits ``(B, 1)`` int64 global token id.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

if TYPE_CHECKING:
    from ....core import DTensor


class ArgmaxReduce(MPKModule):
    """Second half of split-reduce argmax: merge per-chunk partials.

    Reconstructs the global vocab index as
    ``winning_chunk_idx * CHUNK_SIZE + partial_indices[winner]``.
    ``CHUNK_SIZE`` is read from ``pk.argmax_partial_output_size``,
    written by :class:`ArgmaxPartial.compile` — that call **must** run
    earlier in the same compile scope. Parameterless; ``prefix`` only
    names the auto-allocated output.
    """

    def __init__(self, num_partial_tasks: int, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        if num_partial_tasks <= 0:
            raise ValueError(
                f"ArgmaxReduce num_partial_tasks must be positive; "
                f"got {num_partial_tasks}"
            )
        self.num_partial_tasks = num_partial_tasks

    def forward(
        self,
        partial_values: torch.Tensor,
        partial_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Reference: pick the partial with max value per row, reconstruct global idx.

        ``CHUNK_SIZE`` is read from ``self._chunk_size`` if set by the
        caller (chained pipeline pattern); otherwise inferred as
        ``partial_indices.max() + 1`` (approximate fallback for synthetic
        partials). Returns ``(B, 1)`` int64.
        """
        if partial_values.dim() != 2:
            raise ValueError(
                f"ArgmaxReduce.forward expects 2-D partial_values; "
                f"got shape {tuple(partial_values.shape)}"
            )
        if partial_indices.dim() != 2:
            raise ValueError(
                f"ArgmaxReduce.forward expects 2-D partial_indices; "
                f"got shape {tuple(partial_indices.shape)}"
            )
        if partial_values.shape != partial_indices.shape:
            raise ValueError(
                f"ArgmaxReduce.forward partial_values shape "
                f"{tuple(partial_values.shape)} must equal "
                f"partial_indices shape {tuple(partial_indices.shape)}"
            )

        winning_chunk = torch.argmax(partial_values, dim=-1)
        batch_size = partial_values.shape[0]
        row_idx = torch.arange(
            batch_size, device=partial_indices.device, dtype=torch.int64
        )
        local_idx = partial_indices[row_idx, winning_chunk]

        chunk_size = getattr(self, "_chunk_size", None)
        if chunk_size is None:
            max_local = int(partial_indices.max().item()) if partial_indices.numel() else 0
            chunk_size = max_local + 1

        global_idx = winning_chunk.to(torch.int64) * chunk_size + local_idx
        return global_idx.unsqueeze(-1)

    def auto_grid_dim(
        self,
        partial_values_dt: "DTensor",
        partial_indices_dt: Optional["DTensor"] = None,
    ) -> GridDim:
        """Always ``(1, 1, 1)``: the kernel iterates batch internally —
        partitioning the grid would corrupt ``final_output[batch_idx]`` writes."""
        return (1, 1, 1)

    def compile(
        self,
        partial_values: "DTensor",
        partial_indices: "DTensor",
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
        name: Optional[str] = None,
    ) -> "DTensor":
        """Register one ``argmax_reduce[_sm100]`` task.

        Tensor contract:
          partial_values:  (B, num_partial_tasks)  bf16, per-chunk max value.
          partial_indices: (B, num_partial_tasks)  int64, per-chunk local idx.
          output:          (B, 1)                  int64, global vocab idx.

        Notes: grid_dim MUST be ``(1, 1, 1)`` — the kernel iterates batch
        internally and partitioning would corrupt ``final_output[batch_idx]``
        writes. Requires :meth:`ArgmaxPartial.compile` earlier in the same
        scope to populate ``pk.argmax_partial_output_size``.
        """
        from ....core import DTensor

        pk = current_pk()

        if partial_values.num_dims != 2:
            raise ValueError(
                f"ArgmaxReduce.compile expects 2-D partial_values DTensor; "
                f"got num_dims={partial_values.num_dims}"
            )
        if partial_indices.num_dims != 2:
            raise ValueError(
                f"ArgmaxReduce.compile expects 2-D partial_indices DTensor; "
                f"got num_dims={partial_indices.num_dims}"
            )
        if partial_values.dim(0) != partial_indices.dim(0):
            raise ValueError(
                f"ArgmaxReduce.compile partial_values batch_size "
                f"{partial_values.dim(0)} must equal partial_indices "
                f"batch_size {partial_indices.dim(0)}"
            )
        if partial_values.dim(1) != partial_indices.dim(1):
            raise ValueError(
                f"ArgmaxReduce.compile partial_values num_partial_tasks "
                f"{partial_values.dim(1)} must equal partial_indices "
                f"num_partial_tasks {partial_indices.dim(1)}"
            )
        if partial_values.dim(1) != self.num_partial_tasks:
            raise ValueError(
                f"ArgmaxReduce.compile expects partial inputs with last "
                f"dim = num_partial_tasks={self.num_partial_tasks}; "
                f"got {partial_values.dim(1)}"
            )

        batch_size = partial_values.dim(0)
        prefix = self.prefix or "argmax_reduce."

        if output is None:
            out_name = name if name is not None else f"{prefix}out"
            out_dt = pk.new_tensor(
                dims=(batch_size, 1),
                dtype=mi.int64,
                name=out_name,
                io_category="cuda_tensor",
            )
        elif isinstance(output, torch.Tensor):
            if output.dtype != torch.int64:
                raise ValueError(
                    "ArgmaxReduce.compile output torch.Tensor must be "
                    f"int64 (the kernel writes long long); "
                    f"got dtype={output.dtype}"
                )
            out_name = name if name is not None else f"{prefix}out"
            out_dt = pk.attach_input(output, name=out_name)
        elif isinstance(output, DTensor):
            out_dt = output
        else:
            raise TypeError(
                "ArgmaxReduce.compile output must be None, a torch.Tensor, "
                f"or a DTensor; got {type(output).__name__}"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(partial_values, partial_indices)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert partial_values.num_dims == 2
        assert partial_indices.num_dims == 2
        assert out_dt.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(partial_values, (1, 0, -1), -1, True)
        tb_graph.new_input(partial_indices, (1, 0, -1), -1, True)
        tb_graph.new_input(out_dt, (0, 1, -1), -1, True)
        pk.kn_graph.customized(
            [partial_values, partial_indices, out_dt], tb_graph
        )
        if pk.target_cc == 100:
            pk.kn_graph.register_task(
                tb_graph,
                "argmax_reduce_sm100",
                [pk.argmax_partial_output_size],
            )
        else:
            pk.kn_graph.register_task(
                tb_graph,
                "argmax_reduce",
                [pk.argmax_partial_output_size],
            )
        return out_dt
