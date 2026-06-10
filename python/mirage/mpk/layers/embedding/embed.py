"""Token-embedding lookup.

Backed by ``tasks/{ampere,hopper}/embedding{,_hopper}.cuh`` (the task
name registered is always ``"embedding"``; the Hopper header is selected
via ``#include`` in ``hopper/task_header.cuh``). bf16 weight only.
``input_source`` toggles whether tokens come from the runtime's rolling
``runtime_config.tokens + step[0]`` (0) or from ``input_dt`` /
``task_desc->input_ptrs[0]`` (1). Single-CTA op — the .cuh parallelizes
``BATCH_SIZE * OUTPUT_DIM_SIZE`` across ``blockDim.x``.
"""

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import BlockDim, GridDim, MPKModule

if TYPE_CHECKING:
    from ....core import DTensor


class Embed(MPKModule):
    """Embedding lookup ``y[i] = weight[input[i]]``.

    Weight shape ``(num_embeddings, embedding_dim)``, bf16. The kernel
    is single-CTA; growing the grid is unproductive.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Reference: ``F.embedding(input, self.weight)``."""
        return F.embedding(input, self.weight)

    def auto_grid_dim(self, input_dt: "DTensor") -> GridDim:
        """Single CTA: ``(1, 1, 1)`` — kernel has no per-CTA tile."""
        return (1, 1, 1)

    def compile(
        self,
        input_dt: "DTensor",
        *,
        input_source: int = 0,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> "DTensor":
        """Register an ``embedding`` task.

        Tensor contract:
          input_dt: (B, num_spec_tokens) int64, token-id rows (also wires the edge
                    when ``input_source=0``; data is ignored in that case).
          weight:   (num_embeddings, embedding_dim) bf16, lookup table (auto-attached).
          output:   (B, embedding_dim)   bf16, ``weight[input_dt]``.

        Notes: single-CTA — auto_grid is ``(1, 1, 1)``. Param ``input_source``:
        ``0`` → kernel reads ``runtime_config.tokens + step[0]`` (meta dep);
        ``1`` → kernel reads ``input_dt`` (``task_desc->input_ptrs[0]``).
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        grid_dim = grid_dim if grid_dim is not None else self.auto_grid_dim(input_dt)
        block_dim = block_dim if block_dim is not None else self.default_block_dim()

        weight_dt = pk.attach_input(
            torch_tensor=self.weight, name=f"{self.prefix}weight"
        )

        if output is None:
            batch_size = input_dt.dim(0)
            out_dt = pk.new_tensor(
                dims=(batch_size, self.embedding_dim),
                dtype=weight_dt.dtype,
                name=f"{self.prefix}out",
                io_category="cuda_tensor",
            )
        elif isinstance(output, torch.Tensor):
            out_dt = pk.attach_input(
                torch_tensor=output, name=f"{self.prefix}out"
            )
        else:
            out_dt = output

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_dt, (-1, 1, -1), -1, True)
        tb_graph.new_input(weight_dt, (1, -1, -1), -1, True)
        tb_graph.new_input(out_dt, (1, 0, -1), -1, True)
        pk.kn_graph.customized([input_dt, weight_dt, out_dt], tb_graph)
        pk.kn_graph.register_task(tb_graph, "embedding", [input_source])
        return out_dt
