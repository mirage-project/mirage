"""Single-shot argmax (greedy token selection) catalog module.

Wraps :meth:`PersistentKernel.argmax_layer`. **Currently broken**:
``TASK_ARGMAX`` is declared in ``runtime_header.h`` but ``graph.cc``
emits no kernel body (no ``register_argmax_task`` call), so the task is
a no-op. Use :class:`ArgmaxPartial` + :class:`ArgmaxReduce` instead
(see ``tasks/{ampere,blackwell}/argmax{,_sm100}.cuh`` for the working
split-reduce kernels).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import torch
import torch.nn as nn

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

if TYPE_CHECKING:
    from ....core import DTensor


class Argmax(MPKModule):
    """Per-row argmax along the last dim. ``(B, V)`` -> ``(B, 1)`` ``int64``.

    Wraps :meth:`PersistentKernel.argmax_layer` (single-shot variant).
    Parameterless — ``state_dict`` is empty. ``prefix`` only names the
    auto-allocated output DTensor.

    NOTE: currently a no-op due to missing kernel body in ``graph.cc:493``;
    ``__init__`` raises ``RuntimeError`` until that is fixed.
    """

    def __init__(self, *, prefix: str = "") -> None:
        raise RuntimeError(
            "layers.Argmax (single-shot, wraps pk.argmax_layer) is "
            "currently broken in Mirage: TASK_ARGMAX (=109) is declared "
            "in runtime_header.h but graph.cc:493 emits no kernel body "
            "(no register_argmax_task call), so the task is a no-op and "
            "the output buffer is never written. Use the split-reduce "
            "pair ArgmaxPartial + ArgmaxReduce instead."
        )
        super().__init__(prefix=prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reference: ``torch.argmax(x, dim=-1, keepdim=True)`` -> ``(B, 1)`` int64."""
        return torch.argmax(x, dim=-1, keepdim=True)

    def auto_grid_dim(self, x_dt: "DTensor") -> GridDim:
        """``(min(batch_size, num_workers), 1, 1)`` — one task per row, capped at pool."""
        pk = current_pk()
        return (max(1, min(x_dt.dim(0), pk.num_workers)), 1, 1)

    def compile(
        self,
        x: "DTensor",
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
        name: Optional[str] = None,
    ) -> "DTensor":
        """Register one ``argmax`` task.

        Tensor contract:
          x:      (B, V)  bf16 or fp32, per-row logits.
          output: (B, 1)  int64,        per-row argmax index.

        Notes: kernel body is a NO-OP — ``graph.cc:493`` never emits a
        ``register_argmax_task`` call; the output buffer is never written.
        ``__init__`` raises ``RuntimeError`` so this path is unreachable in
        practice. Use :class:`ArgmaxPartial` + :class:`ArgmaxReduce`.
        """
        from ....core import DTensor

        pk = current_pk()

        if x.num_dims != 2:
            raise ValueError(
                f"Argmax.compile expects a 2-D input DTensor "
                f"(batch_size, vocab_size); got num_dims={x.num_dims}"
            )

        prefix = self.prefix or "argmax"

        if output is None:
            out_name = name if name is not None else f"{prefix}out"
            out_dt = pk.new_tensor(
                dims=(x.dim(0), 1),
                dtype=mi.int64,
                name=out_name,
                io_category="cuda_tensor",
            )
        elif isinstance(output, torch.Tensor):
            if output.dtype != torch.int64:
                raise ValueError(
                    "Argmax.compile output torch.Tensor must be int64 "
                    f"(the kernel writes long long); got dtype={output.dtype}"
                )
            out_name = name if name is not None else f"{prefix}out"
            out_dt = pk.attach_input(output, name=out_name)
        elif isinstance(output, DTensor):
            out_dt = output
        else:
            raise TypeError(
                "Argmax.compile output must be None, a torch.Tensor, "
                f"or a DTensor; got {type(output).__name__}"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert x.num_dims == 2
        assert out_dt.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x, (-1, -1, -1), -1, True)
        tb_graph.new_input(out_dt, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([x, out_dt], tb_graph)
        pk.kn_graph.register_task(tb_graph, "argmax")
        return out_dt
