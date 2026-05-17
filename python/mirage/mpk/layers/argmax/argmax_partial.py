"""Split-reduce argmax — the *partial* half.

Catalog wrapper around :meth:`PersistentKernel.argmax_partial_layer`.
Together with :class:`ArgmaxReduce` this is the two-stage greedy-decode
path qwen3 / llama3 use in production for the
``vocab_size >> num_workers`` case (qwen3 has ``V = 151936``,
``num_workers`` is typically 96-128). The single-shot
:class:`Argmax` collapses the entire vocab inside one threadblock and
becomes the bottleneck of the megakernel for large vocabularies; this
split-reduce pair fans the reduction out across CTAs and finishes with
a tiny merge.

Pipeline
--------

``ArgmaxPartial`` runs first. The vocab axis is partitioned into
``num_partial_tasks`` equal chunks of ``CHUNK_SIZE = V // num_partial_tasks``
elements. Each task owns one chunk and writes two scalars per row:

* ``partial_values[b, t]`` — the chunk's maximum logit (bf16).
* ``partial_indices[b, t]`` — the **chunk-local** argmax index (int64).
  The kernel stores the chunk-internal position (``0 <= idx < CHUNK_SIZE``);
  the chunk *offset* is added later by :class:`ArgmaxReduce`. See
  ``include/mirage/persistent_kernel/tasks/ampere/argmax.cuh`` lines
  103-106: ``output_idx[batch_idx * NUM_PARTIAL_TASKS] = local_idx;``
  where ``local_idx`` is the in-chunk position.

:class:`ArgmaxReduce` then consumes these two tensors and emits the
final ``(B, 1)`` ``int64`` token-id, doing
``winning_chunk_idx * CHUNK_SIZE + winning_relative_idx`` to reconstruct
the global index.

Forward-only reference note
---------------------------

This module's :meth:`forward` mimics the kernel layout exactly: it
returns the two partials in the same shape and meaning the compiled
kernel produces, so an upstream test can compare both outputs row-by-row
against the reference. The chunk offset is **not** added here either —
:class:`ArgmaxReduce.forward` reapplies the
``chunk_index * CHUNK_SIZE + local_index`` arithmetic and the chained
``ArgmaxPartial -> ArgmaxReduce`` matches ``torch.argmax(x, dim=-1)``
bit-exactly.

Tensor contract
---------------
- Input ``x`` — 2-D ``bfloat16`` device tensor of shape
  ``(batch_size, vocab_size)``.
- Outputs (returned as a tuple, matching the kernel's twin-output
  signature):

  * ``partial_values`` — 2-D ``bfloat16`` device tensor of shape
    ``(batch_size, num_partial_tasks)``.
  * ``partial_indices`` — 2-D ``int64`` device tensor of shape
    ``(batch_size, num_partial_tasks)``. Values are chunk-local
    positions, **not** global vocab indices.

Alignment requirement
---------------------

``vocab_size % num_partial_tasks == 0`` is a **hard requirement** —
the kernel uses a fixed ``CHUNK_SIZE = vocab_size // num_partial_tasks``
as a compile-time template parameter (see ``argmax_partial_layer`` in
``persistent_kernel.py``, which records ``argmax_partial_output_size``
for the reduce step to consume). Mismatched alignment silently drops
the tail of the vocab.

Tie-breaking
------------

The reduction uses strict ``>`` at every level (warp / block / chained
reduce), so a tie returns the **lowest** in-chunk index, and across
chunks the **lowest** chunk index wins. ``torch.argmax`` uses the same
first-wins semantics, so chained ``ArgmaxPartial -> ArgmaxReduce``
matches ``torch.argmax`` on ties.

Parallelism
-----------

One task per (vocab-chunk * row group) — natural ``grid_dim`` is
``(num_partial_tasks, 1, 1)``. ``num_partial_tasks`` is selected by the
caller; the canonical heuristic used in qwen3 production is
``num_partial_tasks = pk.num_workers`` (one task per worker, saturates
the pool). The grid is capped at ``pk.num_workers`` because tasks
beyond the pool size would just queue up; we still register them all
but emit a warning in `auto_grid_dim` only if the requested value
exceeds the pool. The dtype of ``partial_values`` is ``bfloat16``
(matches the input) and ``partial_indices`` is ``int64`` (kernel stores
``long long``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import torch
import torch.nn as nn

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

if TYPE_CHECKING:
    # DTensor lives in the compiled Cython core; type-only import so the
    # .so is not forced on pure-PyTorch users.
    from ....core import DTensor


class ArgmaxPartial(MPKModule):
    """First half of split-reduce argmax for large-vocab greedy decode.

    Splits the trailing (vocab) dim of a ``(B, V)`` bf16 tensor into
    ``num_partial_tasks`` equal chunks. Each task (one CTA) computes
    ``(max_value, argmax_local_index)`` over its slice; the two outputs
    are then consumed by :class:`ArgmaxReduce` to produce the final
    per-row token id.

    The two outputs together carry enough information for
    :class:`ArgmaxReduce` to recover the global argmax index:
    ``global_idx = winning_chunk_idx * CHUNK_SIZE + partial_indices[winner]``.

    Args:
        vocab_size: Last-dim size of the logits tensor the module is
            sized for. Must be divisible by ``num_partial_tasks`` —
            asserted at construction.
        num_partial_tasks: Number of vocab-chunks (= ``grid_dim.x``).
            Canonical choice for qwen3 / llama3 is
            ``current_pk().num_workers``. Picked at module construction
            because the kernel bakes ``CHUNK_SIZE`` and
            ``NUM_PARTIAL_TASKS`` in as template parameters.
        prefix: vLLM/HF state_dict prefix. Combined with trailing
            ``"partial_values"`` / ``"partial_indices"`` to name the
            auto-allocated DTensors uniquely (e.g. ``prefix="lm_head."``
            yields ``lm_head.partial_values``).
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
            # Hard alignment requirement — the kernel uses a fixed
            # CHUNK_SIZE template parameter and a non-divisible vocab
            # silently drops its tail (no out-of-bounds, but the values
            # past num_partial_tasks * CHUNK_SIZE are skipped).
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

    # ------------------------------------------------------------------
    # PyTorch reference path
    # ------------------------------------------------------------------
    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference: per-row chunked max + chunk-local argmax.

        Mirrors the kernel layout exactly so the test can compare both
        outputs row-for-row. Chunk-local indices (``0 <= idx < CHUNK_SIZE``)
        are stored — the chunk offset is added by :class:`ArgmaxReduce`.

        Args:
            x: ``(batch_size, vocab_size)`` bf16 logits.

        Returns:
            ``(partial_values, partial_indices)`` where
            ``partial_values`` is ``(B, num_partial_tasks)`` bf16 and
            ``partial_indices`` is ``(B, num_partial_tasks)`` int64 with
            **chunk-local** positions.
        """
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
        # Reshape (B, V) -> (B, num_partial_tasks, CHUNK_SIZE) — the
        # same logical layout the kernel iterates over.
        chunked = x.reshape(batch_size, self.num_partial_tasks, self.chunk_size)
        # torch.max returns (values, indices) along the reduced dim.
        # Indices are chunk-local positions, matching the kernel.
        partial_values, partial_indices = chunked.max(dim=-1)
        # Match kernel dtypes: bf16 values, int64 indices.
        partial_values = partial_values.to(x.dtype)
        partial_indices = partial_indices.to(torch.int64)
        return partial_values, partial_indices

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, x_dt: "DTensor") -> GridDim:
        """``grid_dim.x = num_partial_tasks`` (capped at the worker pool).

        ``pk.argmax_partial_layer`` reads ``num_tasks = grid_dim[0]`` and
        registers the task with ``CHUNK_SIZE = vocab_size // num_tasks``
        as a template parameter. The module's ``num_partial_tasks`` is
        the authoritative value, so we use it directly. We additionally
        cap at ``pk.num_workers`` because tasks beyond the pool size
        cannot run concurrently — the canonical qwen3 choice is
        ``num_partial_tasks = pk.num_workers`` precisely to saturate
        without overcommitting.
        """
        pk = current_pk()
        # Cap, but keep the user's choice if it's smaller than the pool.
        x_dim = max(1, min(self.num_partial_tasks, pk.num_workers))
        return (x_dim, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path
    # ------------------------------------------------------------------
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
        """Register one ``argmax_partial`` task on the active PK.

        Both outputs follow the same three-way routing contract as the
        rest of the catalog: ``None`` -> ``pk.new_tensor`` (production),
        ``torch.Tensor`` -> ``pk.attach_input`` (test harness reads
        back), ``DTensor`` -> use as-is (composite ``compile()`` paths
        that pre-allocate).

        Args:
            x: 2-D ``bfloat16`` DTensor, shape ``(batch_size, vocab_size)``.
            partial_values: Routing for the bf16 ``(B, num_partial_tasks)``
                output of per-chunk max values. See above.
            partial_indices: Routing for the int64 ``(B, num_partial_tasks)``
                output of chunk-local argmax indices. See above.
            grid_dim: Explicit override; ``None`` -> :meth:`auto_grid_dim`.
                Note ``grid_dim[0]`` is the authoritative
                ``num_tasks`` the kernel uses; passing a value that
                conflicts with ``self.num_partial_tasks`` will give
                you the wrong ``CHUNK_SIZE`` and silently corrupt the
                output indices.
            block_dim: Explicit override; ``None`` -> :meth:`default_block_dim`.
            name: Optional prefix for auto-allocated output buffers
                (used only when ``partial_values`` / ``partial_indices``
                are ``None``). When ``None``, ``self.prefix`` is used.

        Returns:
            ``(values_dt, indices_dt)`` — the two DTensors as registered
            with the PK, ready to be passed to ``ArgmaxReduce.compile``.

        Raises:
            RuntimeError: when called outside ``pk.compile_scope()``.
            ValueError: when ``x`` is not 2-D or ``x.dim(1)`` mismatches
                ``self.vocab_size``.
            TypeError: when an output routing argument is none of None /
                torch.Tensor / DTensor.
        """
        # Local import keeps the core .so off the import path for users
        # who only need the PyTorch ``forward`` reference.
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

        # ---- partial_values (bf16) -----------------------------------
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

        # ---- partial_indices (int64) ---------------------------------
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

        # ---- grid / block --------------------------------------------
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # ``num_tasks = grid_dim[0]`` is what the kernel uses for
        # CHUNK_SIZE. If the caller overrode grid_dim with something
        # that doesn't match num_partial_tasks, the kernel will compute
        # a different CHUNK_SIZE than the module thinks — refuse early.
        if grid_dim[0] != self.num_partial_tasks:
            raise ValueError(
                f"ArgmaxPartial.compile grid_dim[0]={grid_dim[0]} must "
                f"equal num_partial_tasks={self.num_partial_tasks} "
                f"(the kernel derives CHUNK_SIZE from grid_dim[0])."
            )

        # Inlined task registration (the body that used to live on
        # ``PersistentKernel.argmax_partial_layer``). Each catalog module
        # owns its own task wiring so adding a new layer doesn't require
        # editing ``persistent_kernel.py``.
        #
        # IMPORTANT: this writes ``pk.argmax_partial_output_size`` as a
        # side effect — :class:`ArgmaxReduce` reads that attribute from
        # the PK instance to recover the per-chunk size for its
        # reconstruction kernel. ``ArgmaxPartial.compile`` MUST run
        # before ``ArgmaxReduce.compile`` in the same compile scope.
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert x.num_dims == 2  # (batch_size, vocab_size)
        assert values_dt.num_dims == 2  # (batch_size, num_tasks)
        assert indices_dt.num_dims == 2  # (batch_size, num_tasks)
        num_tasks = grid_dim[0]
        pk.argmax_partial_output_size = x.dim(1) // num_tasks
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x, (1, 0, -1), -1, True)
        tb_graph.new_input(values_dt, (1, 0, -1), -1, True)
        tb_graph.new_input(indices_dt, (1, 0, -1), -1, True)
        pk.kn_graph.customized([x, values_dt, indices_dt], tb_graph)
        if pk.target_cc == 100 or pk.target_cc == 90:
            pk.kn_graph.register_task(
                tb_graph, "argmax_partial_sm100", [num_tasks]
            )
        else:
            pk.kn_graph.register_task(
                tb_graph, "argmax_partial", [num_tasks]
            )
        return values_dt, indices_dt
