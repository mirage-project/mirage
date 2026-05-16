"""Functional element-wise add: ``out = a + b``.

This is the catalog counterpart to :meth:`PersistentKernel.elementwise_add_layer`.
Unlike the other catalog entries (``Linear``, ``RMSNorm``, ...), ``add`` has no
weights and no per-instance state, so it is exported as a free function rather
than a :class:`MPKModule` subclass. The PyTorch reference is simply the ``+``
operator on the input tensors; this docstring documents the equivalence so the
model author does not need a separate reference module.

Tensor contract
---------------

Both inputs and the output are 2-D ``bfloat16`` device tensors with **matching
shape** ``(batch_size, output_size)``. The kernel reads ``input_a`` and
``input_b`` row-major; it writes the output row-major with the same
``output_stride`` as ``output_size``. The dtype is hard-coded to
``cute::bfloat16_t`` at task-register time
(``src/kernel/task_register.cc:2229``) — passing any other dtype is undefined.

The kernel also exposes an optional column-slice mode for ``input_a`` via
``in_a_row_stride`` / ``in_a_col_offset`` (see ``elementwise_add_layer``
docstring) used by DeepSeek-V3 to fold a slice read of a wider buffer into the
add. ``add()`` does not surface that feature; if you need it, call the
underlying ``pk.elementwise_add_layer`` directly.

Parallelism axis
----------------

Looking at ``include/mirage/persistent_kernel/tasks/blackwell/elementwise_add_sm100.cuh``,
each task (one thread block) handles its slice of the **batch (dim 0)** —
``tb_graph.new_input(input_a, (0, -1, -1), -1, True)`` partitions on the first
axis. With ``grid_dim=(num_tasks, 1, 1)`` each task processes
``batch_size / num_tasks`` rows of ``output_size`` elements. Existing callers
(``demo/qwen3/demo.py`` does not use this op; the deepseek_v3 builder is the
only in-tree caller) pick ``grid_dim=(max_num_batched_tokens, 1, 1)``, i.e. one
task per batch row, which is the maximum-parallelism choice as long as
``num_workers >= batch_size``.

Architecture support
--------------------

Only Blackwell (SM100) has a kernel implementation today (the task name
registered is ``"elementwise_add_sm100"`` unconditionally — see
``persistent_kernel.py:3114``). The runtime task-enum is
``TASK_ELEMENTWISE_ADD_SM100 = 281``, which sits outside the 231..256 window
that the SM100 TMA-desc dispatcher autoroutes — be aware of the
``mpk_sm100_tma_dispatch_pitfall`` memory note. The current kernel uses plain
pointer reads (no TMA), so the pitfall does not bite here, but anyone adding a
TMA variant must wire it explicitly in ``runtime.cc``.

Alignment
---------

The kernel body is a strided ``for (i = threadIdx.x; i < BATCH*OUTPUT; i +=
blockDim.x)`` loop, so there is no per-row vector-load alignment requirement
beyond what ``cute::bfloat16_t`` already imposes. The element count
``BATCH_SIZE * OUTPUT_SIZE`` may be any positive integer. The
``in_a_row_stride`` slice mode (not exposed here) does add an offset
constraint; see the kernel for details.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from ...context import current_pk

# Re-export-friendly DTensor type for annotations. Importing from mirage.core
# is fine — DTensor is the public Cython class used everywhere in the codebase.
from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _default_block_dim(target_cc: int) -> BlockDim:
    # Mirrors MPKModule.default_block_dim() and the convention in
    # tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py: 256 threads on
    # Hopper/Blackwell, 128 on Ampere.
    return (128, 1, 1) if target_cc < 90 else (256, 1, 1)


def _auto_grid_dim(num_rows: int, num_workers: int) -> GridDim:
    # Heuristic: one task per output row, capped at the worker pool. The kernel
    # partitions on dim 0 (batch), so each task naturally owns its row slice.
    # If we ever want fewer than batch_size tasks (e.g. very small workers,
    # large batch) the per-task BATCH_SIZE in the generated code is
    # output_tensor.dim(0); the grid_dim only tells the runtime how many
    # task descriptors to emit.
    #
    # Existing callers (deepseek_v3/builder.py) pick
    # (max_num_batched_tokens, 1, 1) which matches this heuristic for
    # num_rows == max_num_batched_tokens.
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
    """Element-wise ``a + b`` registered as an MPK task.

    Equivalent in PyTorch to::

        out = a + b  # both 2-D bfloat16, identical shape

    Args:
        a: 2-D bfloat16 DTensor with shape ``(batch_size, output_size)``.
        b: 2-D bfloat16 DTensor, same shape as ``a``.
        output: Optional output buffer.
            * ``None`` (default, production): allocate a fresh DTensor via
              ``pk.new_tensor`` with the same shape/dtype as ``a``.
            * ``torch.Tensor``: attach it as a graph input via
              ``pk.attach_input`` so the test driver can read back from it
              (the canonical test path — see
              ``tests/runtime_python/test_mode/test_rmsnorm_testmode.py``).
            * ``DTensor``: use directly (advanced; caller owns the registration).
        grid_dim: Optional explicit grid override. ``None`` → auto-heuristic.
        block_dim: Optional explicit block override. ``None`` → architecture
            default (128 on Ampere, 256 on Hopper/Blackwell).
        name: Optional name for the auto-allocated output buffer. Only used
            when ``output is None``. Required to be unique across the PK's
            tensor registry.

    Returns:
        The output DTensor (whichever path produced it).

    Raises:
        RuntimeError: if called outside a ``pk.compile_scope()`` block (raised
            by :func:`current_pk`).
    """
    pk = current_pk()

    # Input shape sanity. The underlying pk.elementwise_add_layer asserts
    # num_dims == 2 on each tensor; we surface a friendlier error here.
    if a.num_dims != 2 or b.num_dims != 2:
        raise ValueError(
            f"add() expects 2-D DTensors; got a.num_dims={a.num_dims}, "
            f"b.num_dims={b.num_dims}"
        )
    if a.dim(0) != b.dim(0) or a.dim(1) != b.dim(1):
        # The legacy (non-slice) path requires matching shapes. The slice mode
        # is not exposed by this function — call pk.elementwise_add_layer
        # directly if you need it.
        raise ValueError(
            f"add() requires matching shapes; got a=({a.dim(0)}, {a.dim(1)}) "
            f"b=({b.dim(0)}, {b.dim(1)}). For the column-slice variant call "
            "pk.elementwise_add_layer(...) directly."
        )

    # Resolve output DTensor.
    if output is None:
        # Production path: allocate a fresh CUDA tensor.
        out_name = name if name is not None else f"add_out_{id(a)}_{id(b)}"
        out_dt = pk.new_tensor(
            dims=(a.dim(0), a.dim(1)),
            dtype=a.dtype,
            name=out_name,
        )
    elif isinstance(output, torch.Tensor):
        # Test-mode readback path: attach the torch buffer as a graph input
        # so the host can read from it after pk() returns.
        out_name = name if name is not None else f"add_out_{id(output)}"
        out_dt = pk.attach_input(output, name=out_name)
    elif isinstance(output, DTensor):
        # Advanced path: caller already has a DTensor.
        out_dt = output
    else:
        raise TypeError(
            "add() output must be None, a torch.Tensor, or a DTensor; "
            f"got {type(output).__name__}"
        )

    # Resolve grid/block.
    if grid_dim is None:
        grid_dim = _auto_grid_dim(num_rows=a.dim(0), num_workers=pk.num_workers)
    if block_dim is None:
        block_dim = _default_block_dim(pk.target_cc)

    pk.elementwise_add_layer(
        input_a=a,
        input_b=b,
        output=out_dt,
        grid_dim=grid_dim,
        block_dim=block_dim,
    )
    return out_dt
