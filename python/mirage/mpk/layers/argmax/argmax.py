"""Single-shot argmax (greedy token selection) catalog module.

Catalog wrapper around :meth:`PersistentKernel.argmax_layer`. Given a
logits matrix ``(B, V)``, returns the per-row argmax index ``(B, 1)``
``int64``. This is the greedy-decoding path used at the end of an LLM
forward: ``token_id = argmax(logits, dim=-1)``.

Single-shot vs split-reduce
---------------------------

MPK actually exposes three argmax variants:

* :meth:`PersistentKernel.argmax_layer` (this module) — one task per
  row; the full vocab is reduced inside a single threadblock.
* :meth:`PersistentKernel.argmax_partial_layer` +
  :meth:`PersistentKernel.argmax_reduce_layer` — split the vocab into
  ``num_partial_tasks`` chunks, run a tree reduction. This is the path
  qwen3 / llama3 demos use today (see ``demo/qwen3/demo.py`` lines
  728-739), giving much better occupancy for the typical
  ``vocab_size >> num_workers`` case.
* :meth:`PersistentKernel.nvshmem_global_argmax_layer` — multi-GPU
  cross-rank reduction for TP/vocab-sharded inference.

**This module only wraps the single-shot variant.** The split-reduce
form is migrated in a follow-up PR as ``ArgmaxPartial`` + ``ArgmaxReduce``
(or a fused ``Argmax`` with a ``split=`` mode), because they need a
two-call ``compile()`` and intermediate partial-value/-index buffers
that have no place in the single-task contract here.

Tensor contract
---------------
- ``logits`` (``forward``) / ``logits_dt`` (``compile``) — 2-D
  ``bfloat16`` device tensor with shape ``(batch_size, vocab_size)``.
  The kernel reduces along the last dim.
- Output — 2-D ``int64`` (``torch.long``) device tensor with shape
  ``(batch_size, 1)``. The kernel writes one ``long long`` per row at
  ``final_output[batch_idx]``; ``pk.argmax_layer`` asserts
  ``output.num_dims == 2`` with a trailing-1 dim, matching the
  ``output_tokens`` layout used by the runtime (``demo/qwen3/demo.py``
  line 250: ``torch.full((max_num_batched_tokens, 1), 0, dtype=long)``).

The PyTorch reference ``torch.argmax(x, dim=-1)`` returns ``(B,)``; we
use ``keepdim=True`` so ``forward()`` and the compiled path return
identical shapes ``(B, 1)``.

Output dtype
------------

``int64`` is **mandatory** — the kernel reinterprets the output buffer
as ``long long *`` (see
``include/mirage/persistent_kernel/tasks/ampere/argmax.cuh`` line 79 and
``blackwell/argmax_sm100.cuh`` line 91). Passing a smaller-width int
tensor will silently scribble across rows.

Tie-breaking
------------

The reduction uses a strict ``>`` comparison
(``if (val > local_max) ...``) at both the warp and block levels, so a
tie returns the **lowest** index that achieves the maximum (the first
encounter sticks). ``torch.argmax`` has the same first-wins semantics,
so ``forward()`` and ``compile()`` agree on ties.

Vocab-size alignment
--------------------

The single-shot kernel does **not** require a particular ``vocab_size``
alignment — it loops ``for (int i = tidx; i < vocab_size; i += NUM_THREADS)``,
so any positive ``vocab_size`` works. (The split-reduce variants do
require ``vocab_size % num_partial_tasks == 0`` because each task owns
a fixed ``CHUNK_SIZE = vocab_size // num_partial_tasks``; that
constraint lives on those modules when they land.)

Parallelism
-----------

One task per row of the batch: each task reduces the full vocab inside
a single threadblock. :meth:`auto_grid_dim` therefore returns
``(batch_size, 1, 1)``, capped at ``current_pk().num_workers`` so we
never overcommit the queue. For the common
``batch_size <= num_workers`` decoding case this is one task per
active token, mirroring how the existing demos size the
``argmax_partial`` grid.

For ``vocab_size >> num_workers`` (typical LLM with vocab 128K-256K)
the split-reduce variants are dramatically faster — prefer those once
they land. This module exists so a model author can wire a simple
greedy head when the vocab is small or the call site is not
performance-critical.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import torch
import torch.nn as nn

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

if TYPE_CHECKING:
    # DTensor lives in the compiled Cython core; type-only import to
    # avoid forcing the .so to load in a pure-PyTorch context.
    from ....core import DTensor


class Argmax(MPKModule):
    """Per-row argmax along the last dim. ``(B, V)`` -> ``(B, 1)`` ``int64``.

    Wraps :meth:`PersistentKernel.argmax_layer` (single-shot variant;
    partial/reduce variants are separate modules for split-V scenarios,
    see module docstring).

    The module is parameterless — there are no learnable weights, so
    ``state_dict()`` is empty. ``prefix`` is still accepted for symmetry
    with the rest of the catalog and to name the auto-allocated output
    DTensor uniquely when the same model is reused twice in one PK.

    Args:
        prefix: vLLM/HF state_dict prefix. Combined with the trailing
            ``"out"`` key gives the output DTensor name attached to the
            MPK graph (e.g. ``prefix="lm_head."`` yields ``lm_head.out``).
            Defaults to ``""``.
    """

    def __init__(self, *, prefix: str = "") -> None:
        raise RuntimeError(
            "layers.Argmax (single-shot, wraps pk.argmax_layer) is "
            "currently broken in Mirage: TASK_ARGMAX (=109) is declared "
            "in runtime_header.h but graph.cc:493 emits no kernel body "
            "(no register_argmax_task call), so the task is a no-op and "
            "the output buffer is never written. Use the split-reduce "
            "pair ArgmaxPartial + ArgmaxReduce instead (see qwen3's "
            "lm_head wiring in demo/qwen3/demo.py:728-739)."
        )
        super().__init__(prefix=prefix)

    # ------------------------------------------------------------------
    # PyTorch reference path
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reference: ``torch.argmax(x, dim=-1, keepdim=True)``.

        Returns shape ``(B, 1)`` ``int64`` so it matches the compiled
        path bit-for-bit. ``torch.argmax`` uses strict ``>`` like the
        MPK kernel, so ties pick the lowest index in both cases.
        """
        return torch.argmax(x, dim=-1, keepdim=True)

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, x_dt: "DTensor") -> GridDim:
        """One task per row, capped at the worker pool.

        ``pk.argmax_layer`` builds the TBGraph with
        ``new_input(input, (-1, -1, -1), -1, True)`` — i.e. the input
        is replicated across tasks rather than partitioned — and the
        kernel itself iterates ``for batch_idx in [0, num_active_tokens)``
        inside one task. The natural choice is therefore
        ``(batch_size, 1, 1)`` (one task per row); we cap at
        ``current_pk().num_workers`` so we never overcommit the queue.

        For ``vocab_size >> num_workers`` the split-reduce variants are
        much faster — prefer those when migrated.
        """
        pk = current_pk()
        batch_size = x_dt.dim(0)
        return (max(1, min(batch_size, pk.num_workers)), 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path
    # ------------------------------------------------------------------
    def compile(
        self,
        x: "DTensor",
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
        name: Optional[str] = None,
    ) -> "DTensor":
        """Register one ``argmax`` task on the active PersistentKernel.

        Args:
            x: 2-D ``bfloat16`` DTensor, shape ``(batch_size, vocab_size)``.
            output: Output buffer routing (same three-way contract as
                every other catalog leaf):

                * ``None`` (default, production): allocate a fresh
                  DTensor via ``pk.new_tensor`` of shape
                  ``(batch_size, 1)`` with dtype ``mirage.int64``.
                * ``torch.Tensor``: attach via ``pk.attach_input`` so
                  the test driver can read back from it after ``pk()``
                  returns. Must be ``int64`` and shape ``(B, 1)`` —
                  the kernel writes ``long long`` and ``argmax_layer``
                  asserts ``output.num_dims == 2``.
                * ``DTensor``: use directly (composite ``compile()``
                  paths where the output buffer is pre-allocated
                  upstream).
            grid_dim: Explicit override; ``None`` -> :meth:`auto_grid_dim`.
            block_dim: Explicit override; ``None`` -> :meth:`default_block_dim`.
            name: Optional name for the auto-allocated output buffer.
                Only used when ``output is None``. Must be unique within
                the PK tensor registry.

        Returns:
            The output DTensor of shape ``(batch_size, 1)``, dtype
            ``int64``.

        Raises:
            RuntimeError: when called outside ``pk.compile_scope()``.
            ValueError: when ``x.num_dims != 2``.
            TypeError: when ``output`` is none of None / torch.Tensor /
                DTensor.
        """
        # Local import keeps the core .so off the import path for
        # pure-PyTorch users of this module.
        from ....core import DTensor

        pk = current_pk()

        if x.num_dims != 2:
            raise ValueError(
                f"Argmax.compile expects a 2-D input DTensor "
                f"(batch_size, vocab_size); got num_dims={x.num_dims}"
            )

        prefix = self.prefix or "argmax"

        # Resolve output DTensor per the three-way contract.
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

        # Resolve grid / block.
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (the body that used to live on
        # ``PersistentKernel.argmax_layer``). Each catalog module owns
        # its own task wiring so adding a new layer doesn't require
        # editing ``persistent_kernel.py``.
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert x.num_dims == 2  # (batch_size, vocab_size)
        assert out_dt.num_dims == 2  # (batch_size, 1)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x, (-1, -1, -1), -1, True)
        tb_graph.new_input(out_dt, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([x, out_dt], tb_graph)
        pk.kn_graph.register_task(tb_graph, "argmax")
        return out_dt
