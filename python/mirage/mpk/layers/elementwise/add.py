"""Element-wise ``out = a + b``.

Free function (no weights / no state). Backed by
``tasks/blackwell/elementwise_add_sm100.cuh`` — the task name registered
is always ``"elementwise_add_sm100"`` (only SM100 has an implementation).
bf16-only. The kernel partitions on dim 0 (batch); each task copies a row
slab. Matching shapes required (slice mode is not exposed here; call
``pk.elementwise_add_layer`` directly if you need ``in_a_row_stride`` /
``in_a_col_offset``).
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from ...context import current_pk
from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _default_block_dim(target_cc: int) -> BlockDim:
    return (128, 1, 1) if target_cc < 90 else (256, 1, 1)


def _auto_grid_dim(num_rows: int, num_workers: int) -> GridDim:
    """Stripe over the batch axis: one task per row, capped at num_workers."""
    return (max(1, min(num_rows, num_workers)), 1, 1)


def add(
    a: DTensor,
    b: DTensor,
    *,
    output: Optional[Union[torch.Tensor, DTensor]] = None,
    grid_dim: Optional[GridDim] = None,
    block_dim: Optional[BlockDim] = None,
    name: Optional[str] = None,
) -> DTensor:
    """Element-wise ``a + b`` registered as one ``elementwise_add_sm100`` task.

    Tensor contract:
      a:      (B, output_size) bf16, addend.
      b:      (B, output_size) bf16, addend (same shape as ``a``).
      output: (B, output_size) bf16, ``a + b`` (auto-alloc / attach / passthrough).

    Notes: SM100-only (no Ampere/Hopper variant); bf16-only; kernel
    partitions on dim 0 (batch). For column-slice mode (``in_a_row_stride``,
    ``in_a_col_offset``), call ``pk.elementwise_add_layer`` directly.
    """
    pk = current_pk()

    if a.num_dims != 2 or b.num_dims != 2:
        raise ValueError(
            f"add() expects 2-D DTensors; got a.num_dims={a.num_dims}, "
            f"b.num_dims={b.num_dims}"
        )
    if a.dim(0) != b.dim(0) or a.dim(1) != b.dim(1):
        raise ValueError(
            f"add() requires matching shapes; got a=({a.dim(0)}, {a.dim(1)}) "
            f"b=({b.dim(0)}, {b.dim(1)}). For the column-slice variant call "
            "pk.elementwise_add_layer(...) directly."
        )

    if output is None:
        out_name = name if name is not None else f"add_out_{id(a)}_{id(b)}"
        out_dt = pk.new_tensor(
            dims=(a.dim(0), a.dim(1)),
            dtype=a.dtype,
            name=out_name,
        )
    elif isinstance(output, torch.Tensor):
        out_name = name if name is not None else f"add_out_{id(output)}"
        out_dt = pk.attach_input(output, name=out_name)
    elif isinstance(output, DTensor):
        out_dt = output
    else:
        raise TypeError(
            "add() output must be None, a torch.Tensor, or a DTensor; "
            f"got {type(output).__name__}"
        )

    if grid_dim is None:
        grid_dim = _auto_grid_dim(num_rows=a.dim(0), num_workers=pk.num_workers)
    if block_dim is None:
        block_dim = _default_block_dim(pk.target_cc)

    from ....core import CyTBGraph
    from ....kernel import TBGraph

    assert a.num_dims == 2
    assert b.num_dims == 2
    assert out_dt.num_dims == 2
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(a, (0, -1, -1), -1, True)
    tb_graph.new_input(b, (0, -1, -1), -1, True)
    tb_graph.new_input(out_dt, (0, -1, -1), -1, True)
    pk.kn_graph.customized([a, b, out_dt], tb_graph)
    pk.kn_graph.register_task(tb_graph, "elementwise_add_sm100")
    return out_dt
