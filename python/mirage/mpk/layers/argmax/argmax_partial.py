"""Split-reduce argmax — the *partial* half.

Wraps :meth:`PersistentKernel.argmax_partial_layer`. Code-gen emits
``argmax_partial_kernel`` from
``include/mirage/persistent_kernel/tasks/ampere/argmax.cuh`` (Ampere) or
``argmax_partial_sm100_kernel`` from ``tasks/blackwell/argmax_sm100.cuh``
(Hopper/Blackwell).  Each task owns one of ``num_partial_tasks`` equal
vocab chunks and writes ``(max_value, chunk_local_idx)`` — the chunk
offset is added later by :class:`ArgmaxReduce`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
import torch.nn as nn

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

if TYPE_CHECKING:
    from ....core import DTensor


class ArgmaxPartial(MPKModule):
    """First half of split-reduce argmax for large-vocab greedy decode.

    Splits ``(B, V)`` along V into ``num_partial_tasks`` chunks of
    ``CHUNK_SIZE = V // num_partial_tasks``; emits per-chunk
    ``(max_value bf16, chunk_local_idx int64)``.  Hard alignment
    requirement: ``vocab_size % num_partial_tasks == 0``.  As a side
    effect, ``compile()`` sets ``pk.argmax_partial_output_size`` which
    :class:`ArgmaxReduce` reads to reconstruct the global vocab index.
    """

    def __init__(
        self,
        vocab_size: int,
        num_partial_tasks: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if vocab_size <= 0:
            raise ValueError(
                f"ArgmaxPartial vocab_size must be positive; got {vocab_size}"
            )
        if num_partial_tasks <= 0:
            raise ValueError(
                f"ArgmaxPartial num_partial_tasks must be positive; "
                f"got {num_partial_tasks}"
            )
        if vocab_size % num_partial_tasks != 0:
            raise AssertionError(
                f"ArgmaxPartial requires vocab_size % num_partial_tasks == 0; "
                f"got vocab_size={vocab_size}, "
                f"num_partial_tasks={num_partial_tasks} "
                f"(vocab_size % num_partial_tasks = "
                f"{vocab_size % num_partial_tasks})"
            )
        self.vocab_size = vocab_size
        self.num_partial_tasks = num_partial_tasks
        self.chunk_size = vocab_size // num_partial_tasks

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference: per-row chunked max + chunk-local argmax, matching kernel layout."""
        if x.dim() != 2:
            raise ValueError(
                f"ArgmaxPartial.forward expects a 2-D tensor "
                f"(batch_size, vocab_size); got shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.vocab_size:
            raise ValueError(
                f"ArgmaxPartial.forward got last-dim {x.shape[-1]}, "
                f"module was sized for vocab_size={self.vocab_size}"
            )

        batch_size = x.shape[0]
        chunked = x.reshape(batch_size, self.num_partial_tasks, self.chunk_size)
        partial_values, partial_indices = chunked.max(dim=-1)
        return partial_values.to(x.dtype), partial_indices.to(torch.int64)

    def auto_grid_dim(self, x_dt: "DTensor") -> GridDim:
        """``(min(num_partial_tasks, num_workers), 1, 1)`` — saturates the pool."""
        pk = current_pk()
        return (max(1, min(self.num_partial_tasks, pk.num_workers)), 1, 1)

    def compile(
        self,
        x: "DTensor",
        *,
        partial_values: Optional[Any] = None,
        partial_indices: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
        name: Optional[str] = None,
    ) -> Tuple["DTensor", "DTensor"]:
        """Register one ``argmax_partial[_sm100]`` task; returns ``(values, indices)``.

        Tensor contract:
          x:               (B, V)                   bf16, per-row logits.
          partial_values:  (B, num_partial_tasks)   bf16, per-chunk max value.
          partial_indices: (B, num_partial_tasks)   int64, per-chunk local idx.

        Notes: requires ``V % num_partial_tasks == 0`` (CHUNK_SIZE = V/num_tasks).
        SIDE EFFECT: sets ``pk.argmax_partial_output_size = V // num_tasks``,
        which :class:`ArgmaxReduce` reads later in the same compile scope.
        """
        from ....core import DTensor

        pk = current_pk()

        if x.num_dims != 2:
            raise ValueError(
                f"ArgmaxPartial.compile expects a 2-D input DTensor "
                f"(batch_size, vocab_size); got num_dims={x.num_dims}"
            )
        if x.dim(1) != self.vocab_size:
            raise ValueError(
                f"ArgmaxPartial.compile got input vocab_size={x.dim(1)}, "
                f"module was sized for vocab_size={self.vocab_size}"
            )

        batch_size = x.dim(0)
        name_prefix = name if name is not None else (self.prefix or "argmax_partial.")

        if partial_values is None:
            values_dt = pk.new_tensor(
                dims=(batch_size, self.num_partial_tasks),
                dtype=mi.bfloat16,
                name=f"{name_prefix}partial_values",
                io_category="cuda_tensor",
            )
        elif isinstance(partial_values, torch.Tensor):
            if partial_values.dtype != torch.bfloat16:
                raise ValueError(
                    "ArgmaxPartial.compile partial_values torch.Tensor "
                    f"must be bfloat16 (the kernel writes T = bf16); "
                    f"got dtype={partial_values.dtype}"
                )
            values_dt = pk.attach_input(
                partial_values, name=f"{name_prefix}partial_values"
            )
        elif isinstance(partial_values, DTensor):
            values_dt = partial_values
        else:
            raise TypeError(
                "ArgmaxPartial.compile partial_values must be None, a "
                f"torch.Tensor, or a DTensor; got {type(partial_values).__name__}"
            )

        if partial_indices is None:
            indices_dt = pk.new_tensor(
                dims=(batch_size, self.num_partial_tasks),
                dtype=mi.int64,
                name=f"{name_prefix}partial_indices",
                io_category="cuda_tensor",
            )
        elif isinstance(partial_indices, torch.Tensor):
            if partial_indices.dtype != torch.int64:
                raise ValueError(
                    "ArgmaxPartial.compile partial_indices torch.Tensor "
                    f"must be int64 (the kernel writes long long); "
                    f"got dtype={partial_indices.dtype}"
                )
            indices_dt = pk.attach_input(
                partial_indices, name=f"{name_prefix}partial_indices"
            )
        elif isinstance(partial_indices, DTensor):
            indices_dt = partial_indices
        else:
            raise TypeError(
                "ArgmaxPartial.compile partial_indices must be None, a "
                f"torch.Tensor, or a DTensor; got {type(partial_indices).__name__}"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        if grid_dim[0] != self.num_partial_tasks:
            raise ValueError(
                f"ArgmaxPartial.compile grid_dim[0]={grid_dim[0]} must "
                f"equal num_partial_tasks={self.num_partial_tasks} "
                f"(the kernel derives CHUNK_SIZE from grid_dim[0])."
            )

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert x.num_dims == 2
        assert values_dt.num_dims == 2
        assert indices_dt.num_dims == 2
        num_tasks = grid_dim[0]
        # Side effect: ArgmaxReduce reads this to reconstruct global indices.
        pk.argmax_partial_output_size = x.dim(1) // num_tasks
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x, (1, 0, -1), -1, True)
        tb_graph.new_input(values_dt, (1, 0, -1), -1, True)
        tb_graph.new_input(indices_dt, (1, 0, -1), -1, True)
        pk.kn_graph.customized([x, values_dt, indices_dt], tb_graph)
        if pk.target_cc == 100 or pk.target_cc == 90:
            pk.kn_graph.register_task(tb_graph, "argmax_partial_sm100", [num_tasks])
        else:
            pk.kn_graph.register_task(tb_graph, "argmax_partial", [num_tasks])
        return values_dt, indices_dt
