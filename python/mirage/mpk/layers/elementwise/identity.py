"""Identity / no-op (memory copy).

Backed by ``tasks/ampere/identity.cuh`` (``identity_task_impl``). bf16
elementwise copy from input to a freshly-allocated output. Used in
DeepSeek V3's MLA path as a "phantom bridge" to legalize the task graph
when one task would otherwise be both fork- and join-producer
(``annotated_graph.cc`` case 3 rejection). The kernel partitions the
**last** dim across ``grid.x``; ``grid.x`` must divide the inner dim.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from .._base import MPKModule


__all__ = ["Identity"]


class Identity(MPKModule):
    """Element-wise bf16 copy.

    ``forward(x)`` returns ``x.clone()`` (kernel materializes a fresh
    buffer rather than aliasing). 2-D or 3-D input; same shape and
    dtype output. Used as a graph-shape primitive, not a numeric one.
    """

    def __init__(self, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x.clone()`` (kernel writes a fresh output buffer)."""
        return x.clone()

    def auto_grid_dim(self, x) -> Tuple[int, int, int]:
        """Stripe the last (inner) dim: largest divisor of inner_dim
        that is <= num_workers.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()
        if hasattr(x, "num_dims"):
            inner = x.dim(x.num_dims - 1)
        else:
            inner = int(x.shape[-1])
        cap = max(1, int(pk.num_workers))
        gx = 1
        for d in range(1, min(inner, cap) + 1):
            if inner % d == 0:
                gx = d
        return (gx, 1, 1)

    def compile(
        self,
        x,
        *,
        dependent: Optional[Any] = None,
        output: Optional[Any] = None,
        grid_dim: Optional[Tuple[int, int, int]] = None,
        block_dim: Optional[Tuple[int, int, int]] = None,
    ):
        """Register an ``identity`` task — element-wise bf16 copy.

        Tensor contract:
          x:      (*shape)  bf16, 2-D or 3-D input.
          output: (*shape)  bf16, same shape and dtype as ``x``.

        Notes: kernel partitions the LAST dim across ``grid.x``; ``grid.x``
        must divide the inner dim. ``dependent`` kwarg is reserved (not
        wired into the task graph) — use the phantom-bridge data dep instead.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        if output is None:
            shape = tuple(x.dim(i) for i in range(x.num_dims))
            out_dt = pk.new_tensor(
                dims=shape, dtype=x.dtype, name=f"{self.prefix}identity_out"
            )
        elif isinstance(output, torch.Tensor):
            out_dt = pk.attach_input(output, name=f"{self.prefix}identity_out")
        else:
            out_dt = output

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert x.num_dims == out_dt.num_dims
        last_dim = 0
        for i in range(x.num_dims):
            assert x.dim(i) == out_dt.dim(i)
            last_dim = i
        assert last_dim == 1 or last_dim == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x, (last_dim, -1, -1), 1, True)
        tb_graph.new_input(out_dt, (last_dim, -1, -1), 1, True)
        pk.kn_graph.customized([x, out_dt], tb_graph)
        pk.kn_graph.register_task(tb_graph, "identity")
        return out_dt
