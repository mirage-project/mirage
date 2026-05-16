"""Split-reduce argmax — the *reduce* half.

Catalog wrapper around :meth:`PersistentKernel.argmax_reduce_layer`.
Consumes the two partial outputs of :class:`ArgmaxPartial`
(``partial_values`` bf16, ``partial_indices`` int64) and produces the
final per-row token-id of shape ``(B, 1)`` int64 — the same contract
as the single-shot :class:`Argmax`.

Why a split-reduce?
-------------------

The single-shot :class:`Argmax` reduces the entire vocab inside one
threadblock. For qwen3's 151,936-token vocab that's a single
``for i = tidx; i < V; i += NUM_THREADS`` loop reaching across hundreds
of thousands of elements per row — bandwidth-bound and bottlenecked on
one CTA. :class:`ArgmaxPartial` fans the reduction out across CTAs
(one chunk per CTA), then this module merges the small
``num_partial_tasks``-element vectors per row.

Two-stage reconstruction
------------------------

:class:`ArgmaxPartial` emits chunk-local indices (kernel writes the
``local_idx`` directly into ``partial_indices``). This module recovers
the global vocab index as
``winning_chunk_idx * CHUNK_SIZE + partial_indices[winner]``. The
kernel packs ``(chunk_index, relative_index)`` into a single int64 via
``((long long)i << 32) | partial_idxs[...]`` so the warp/block reduction
preserves both halves atomically; the final unpack happens on thread 0
(see ``include/mirage/persistent_kernel/tasks/ampere/argmax.cuh``
lines 142-154).

``CHUNK_SIZE`` is **not** an :meth:`__init__` argument here — it lives
on the PK instance as ``pk.argmax_partial_output_size``, set when
:meth:`PersistentKernel.argmax_partial_layer` runs. The reduce kernel
takes it as a template parameter at code-gen time
(``self.argmax_partial_output_size`` in
``argmax_reduce_layer`` register_task call). Practical consequence:
**``ArgmaxPartial.compile`` MUST be called before ``ArgmaxReduce.compile``
in the same compile scope**, otherwise the reduce step will pick up a
stale or absent chunk size.

Tensor contract
---------------
- ``partial_values`` — 2-D ``bfloat16`` DTensor of shape
  ``(batch_size, num_partial_tasks)``.
- ``partial_indices`` — 2-D ``int64`` DTensor of shape
  ``(batch_size, num_partial_tasks)``. Values are chunk-local positions
  (``0 <= idx < CHUNK_SIZE``).
- Output — 2-D ``int64`` device tensor of shape ``(batch_size, 1)``,
  matching :class:`Argmax`'s output (same ``output_tokens`` layout the
  runtime uses).

Tie-breaking
------------

Strict ``>`` at warp / block / chained reduce level, identical to
:class:`Argmax`. With chunk-local indices and ``i`` as chunk id,
both tied chunks pack ``(i, local_idx)``; the comparison is on the
value alone, so the *first* encountered chunk wins. Combined with
``ArgmaxPartial``'s first-wins per chunk, the chained pipeline returns
the lowest global index on a tie — matching ``torch.argmax``.

Parallelism
-----------

The reduce is one task per batch row (each task reduces over
``num_partial_tasks`` partials), so the canonical
``grid_dim`` is ``(batch_size, 1, 1)`` (capped at ``pk.num_workers``).
In the qwen3 demo the special case ``argmax_reduce_grid_dim = (1, 1, 1)``
is used because ``max_num_batched_requests=1`` collapses the batch to
one row; for general greedy decode the per-row grid is the right
choice.

Output dtype is ``int64`` — mandatory because the kernel casts the
output pointer to ``long long *``.
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

    Wraps :meth:`PersistentKernel.argmax_reduce_layer`. Consumes the
    two outputs of :class:`ArgmaxPartial` and produces the final
    ``(B, 1)`` ``int64`` per-row argmax index, matching the
    :class:`Argmax` single-shot contract.

    The module is parameterless — there are no learnable weights, so
    ``state_dict()`` is empty. ``prefix`` is accepted for symmetry with
    the rest of the catalog and to give the auto-allocated output a
    unique name.

    Args:
        num_partial_tasks: Number of partials per row (last dim of the
            two input tensors). Stored for ``auto_grid_dim`` heuristics
            and shape validation; the actual ``CHUNK_SIZE`` the kernel
            uses for index reconstruction comes from the PK's
            ``argmax_partial_output_size`` field, set when
            :class:`ArgmaxPartial.compile` runs earlier in the same
            scope.
        prefix: vLLM/HF state_dict prefix. Combined with the trailing
            ``"out"`` key to name the auto-allocated output DTensor
            uniquely.
    """

    def __init__(self, num_partial_tasks: int, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        if num_partial_tasks <= 0:
            raise ValueError(
                f"ArgmaxReduce num_partial_tasks must be positive; "
                f"got {num_partial_tasks}"
            )
        self.num_partial_tasks = num_partial_tasks

    # ------------------------------------------------------------------
    # PyTorch reference path
    # ------------------------------------------------------------------
    def forward(
        self,
        partial_values: torch.Tensor,
        partial_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Reference: pick the partial with the max value per row.

        Returns ``(B, 1)`` int64 matching the :class:`Argmax` convention.
        Reconstructs the global vocab index as
        ``winning_chunk * CHUNK_SIZE + partial_indices[winner]``, where
        ``CHUNK_SIZE`` is inferred from the partial-indices' max-plus-1
        only when the caller hasn't told us otherwise.

        Specifically: in the chained ``ArgmaxPartial -> ArgmaxReduce``
        reference path the partial_indices stored are chunk-local
        positions (0..CHUNK_SIZE-1), so ``CHUNK_SIZE`` must be passed
        in by the caller of the chained pipeline. To keep
        ``ArgmaxReduce.forward`` standalone and faithful to the kernel,
        we expose only the per-partial argmax-by-value here and let the
        caller add the chunk offset — but for the typical chained use
        the partial_indices' values already span ``[0, CHUNK_SIZE)``,
        and the caller knows ``CHUNK_SIZE`` from
        :attr:`ArgmaxPartial.chunk_size`.

        Args:
            partial_values: ``(B, num_partial_tasks)`` bf16. Per-chunk
                max values.
            partial_indices: ``(B, num_partial_tasks)`` int64. Per-chunk
                chunk-local argmax positions.

        Returns:
            ``(B, 1)`` int64. Per-row global argmax index reconstructed
            via ``winning_chunk * CHUNK_SIZE + partial_indices[winner]``
            (``CHUNK_SIZE = num_partial_tasks_largest_local_index + 1``
            is not knowable from the reduce inputs alone — see note).

        Important — chunk offset:
            ``partial_indices`` carries chunk-local positions; to match
            the kernel's output (a global vocab index) we need
            ``CHUNK_SIZE``. We infer it as ``partial_indices.max() + 1``
            rounded up to a power of 2 only when the caller hasn't
            passed values that allow direct reconstruction. The
            chained test in ``tests/runtime_python/layers/test_argmax_split.py``
            wires the actual ``CHUNK_SIZE`` from
            :attr:`ArgmaxPartial.chunk_size`, which is the supported
            production pattern.
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

        # Pick the chunk index with the maximum value, per row.
        # torch.argmax uses strict > internally, matching the kernel.
        # winning_chunk: (B,) int64, in [0, num_partial_tasks).
        winning_chunk = torch.argmax(partial_values, dim=-1)

        batch_size = partial_values.shape[0]
        row_idx = torch.arange(
            batch_size, device=partial_indices.device, dtype=torch.int64
        )
        # local_idx: (B,) int64, in [0, CHUNK_SIZE).
        local_idx = partial_indices[row_idx, winning_chunk]

        # We need CHUNK_SIZE to reconstruct the global index. The reduce
        # kernel gets it from `pk.argmax_partial_output_size`, which we
        # cannot read here. To keep the reference faithful to the kernel
        # we use a saved chunk size if attached, else fall back to a
        # safe heuristic that works when partial_indices spans the full
        # CHUNK_SIZE range.
        chunk_size = getattr(self, "_chunk_size", None)
        if chunk_size is None:
            # The chained test sets ``rd._chunk_size`` after constructing
            # the modules so the reference reconstruction matches the
            # kernel bit-for-bit. As a fallback (e.g. when the reference
            # is called standalone with synthetic partials) we use
            # ``int(partial_indices.max().item()) + 1`` rounded up. This
            # is approximate but documented — see the module docstring.
            max_local = int(partial_indices.max().item()) if partial_indices.numel() else 0
            chunk_size = max_local + 1

        global_idx = winning_chunk.to(torch.int64) * chunk_size + local_idx
        # Match the kernel and Argmax single-shot: (B, 1) int64.
        return global_idx.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(
        self,
        partial_values_dt: "DTensor",
        partial_indices_dt: Optional["DTensor"] = None,
    ) -> GridDim:
        """Always ``(1, 1, 1)``. The reduce kernel iterates over both
        batch (``for batch_idx < num_active_tokens``) and partials
        (``for i < NUM_PARTIAL_TASKS``) internally; partitioning the
        graph-level grid would slice the input but the per-task kernel
        still writes ``final_output[batch_idx]`` for ALL batch indices,
        leading to out-of-bounds writes. The qwen3 demo confirms this
        choice (``argmax_reduce_grid_dim = (1, 1, 1)``).
        """
        return (1, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path
    # ------------------------------------------------------------------
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
        """Register one ``argmax_reduce`` task on the active PK.

        Args:
            partial_values: 2-D ``bfloat16`` DTensor of shape
                ``(batch_size, num_partial_tasks)``. Produced by
                :class:`ArgmaxPartial.compile`.
            partial_indices: 2-D ``int64`` DTensor of shape
                ``(batch_size, num_partial_tasks)``. Produced by
                :class:`ArgmaxPartial.compile`.
            output: Output routing (same three-way contract as other
                catalog leaves):

                * ``None``: allocate via ``pk.new_tensor`` with shape
                  ``(batch_size, 1)`` and dtype ``mirage.int64``.
                * ``torch.Tensor``: attach via ``pk.attach_input`` so
                  the test driver can read the result after ``pk()``
                  returns. Must be ``int64`` and shape ``(B, 1)``.
                * ``DTensor``: use directly.

            grid_dim: Explicit override; ``None`` -> :meth:`auto_grid_dim`.
            block_dim: Explicit override; ``None`` -> :meth:`default_block_dim`.
            name: Optional name for the auto-allocated output buffer.

        Returns:
            The output DTensor of shape ``(batch_size, 1)``, dtype
            ``int64``.

        Raises:
            RuntimeError: when called outside ``pk.compile_scope()``.
            ValueError: when shapes/dtypes mismatch the partial contract.
            TypeError: when ``output`` is none of None / torch.Tensor /
                DTensor.

        Order requirement:
            :class:`ArgmaxPartial.compile` must be called earlier in the
            same compile scope — it sets ``pk.argmax_partial_output_size``
            which the reduce kernel uses to reconstruct the global index.
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

        # ---- output --------------------------------------------------
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

        # ---- grid / block -------------------------------------------
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(partial_values, partial_indices)
        if block_dim is None:
            block_dim = self.default_block_dim()

        pk.argmax_reduce_layer(
            input=(partial_values, partial_indices),
            output=out_dt,
            grid_dim=grid_dim,
            block_dim=block_dim,
        )
        return out_dt
