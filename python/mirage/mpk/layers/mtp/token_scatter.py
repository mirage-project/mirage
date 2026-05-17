"""MTP scatter primitives — token (int64) and probability (float32).

These two leaves wrap the compile-time-indexed scatter helpers that MPK
uses to accumulate per-draft-step state into a wider buffer during the
DeepSeek V3 MTP / speculative-decode loop:

* :class:`MTPTokenScatter` -> :meth:`PersistentKernel.mtp_token_scatter_layer`
  -> task ``mtp_token_scatter``. Copies one ``int64`` token-id per
  request from ``src: (batch_size, 1)`` into the ``slot_idx``-th column
  of ``dst: (batch_size, num_slots)`` (the per-iteration draft-token
  accumulator the verifier later reads).

* :class:`MTPFloatScatter` -> :meth:`PersistentKernel.mtp_float_scatter_layer`
  -> task ``mtp_float_scatter``. Same shape contract, but ``float32`` —
  used to stash the per-step draft probability (``softmax_gather``
  output) into a wider ``[batch_size, num_slots]`` float buffer for the
  probabilistic verifier.

Both kernels read ``slot_idx`` as a compile-time constant baked into
``params``; the kernel is unrolled per slot, so the caller picks the
slot at ``compile()`` time (one task per draft step). The MPK runtime
walks ``batch_size`` and writes into the chosen column.

These layers are graph-shape primitives during MTP's draft loop, not
data-dependent reductions; the :meth:`forward` reference is a plain
column-write that mirrors the kernel byte-for-byte.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

from ....core import DTensor


__all__ = ["MTPTokenScatter", "MTPFloatScatter"]


class MTPTokenScatter(MPKModule):
    """Scatter per-request int64 tokens into one column of a wide buffer.

    Wraps :meth:`PersistentKernel.mtp_token_scatter_layer` (task
    ``mtp_token_scatter``). Used by DeepSeek V3's MTP draft loop to
    accumulate each iteration's predicted token into the
    ``[batch_size, num_slots]`` draft-token buffer that the verifier
    consumes.

    Args:
        batch_size: First dim of ``src`` / ``dst`` (number of concurrent
            requests).
        num_slots:  Width of ``dst`` (max number of draft tokens MTP
            ever emits per request, i.e. one column per draft step).
        prefix:     vLLM/HF state_dict prefix; the scatter has no
            weights, so this is only used as a uniquifier when
            debugging.
    """

    def __init__(
        self,
        batch_size: int,
        num_slots: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if batch_size <= 0:
            raise ValueError(
                f"MTPTokenScatter batch_size must be positive; got {batch_size}"
            )
        if num_slots <= 0:
            raise ValueError(
                f"MTPTokenScatter num_slots must be positive; got {num_slots}"
            )
        self.batch_size = batch_size
        self.num_slots = num_slots

    # ------------------------------------------------------------------
    # PyTorch reference path — pure column write.
    # ------------------------------------------------------------------
    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        slot_idx: int,
    ) -> torch.Tensor:
        """Reference: ``dst[:, slot_idx] = src[:, 0]``.

        Args:
            src:      ``(batch_size, 1)`` int64.
            dst:      ``(batch_size, num_slots)`` int64. Mutated
                      in-place AND returned, matching the kernel's
                      side-effect-on-``dst`` contract.
            slot_idx: Column index, in ``[0, num_slots)``.

        Returns:
            ``dst`` (after the in-place write), to allow chained use in
            functional-style reference code.
        """
        if src.dim() != 2 or src.shape[1] != 1:
            raise ValueError(
                f"MTPTokenScatter.forward expects src of shape "
                f"(batch_size, 1); got {tuple(src.shape)}"
            )
        if dst.dim() != 2:
            raise ValueError(
                f"MTPTokenScatter.forward expects 2-D dst; "
                f"got shape {tuple(dst.shape)}"
            )
        if not (0 <= slot_idx < dst.shape[1]):
            raise ValueError(
                f"MTPTokenScatter.forward slot_idx {slot_idx} out of "
                f"range [0, {dst.shape[1]})"
            )
        dst[:, slot_idx] = src[:, 0]
        return dst

    # ------------------------------------------------------------------
    # Grid heuristic.
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Default grid: ``(1, 1, 1)``.

        The kernel iterates over ``batch_size`` internally (one column
        write per request), so a single CTA suffices. Matches the
        DeepSeek V3 builder caller (always ``grid_dim=(1, 1, 1)``).
        """
        return (1, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path.
    # ------------------------------------------------------------------
    def compile(
        self,
        src: DTensor,
        dst: DTensor,
        slot_idx: int,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``mtp_token_scatter`` task on the active PK.

        Args:
            src:      ``DTensor`` of shape ``(batch_size, 1)`` int64.
            dst:      ``DTensor`` of shape ``(batch_size, num_slots)``
                      int64. Mutated by the kernel and returned (so
                      caller code can chain).
            slot_idx: Compile-time column index baked into the task
                      params; one task instance per draft step.
            grid_dim: Override; ``None`` -> :meth:`auto_grid_dim`.
            block_dim: Override; ``None`` -> :meth:`default_block_dim`.

        Returns:
            ``dst`` (the scatter is consumed for its side effect on
            ``dst``; returning it lets callers thread the dependency).
        """
        pk = current_pk()

        if not (0 <= slot_idx < self.num_slots):
            raise ValueError(
                f"MTPTokenScatter.compile slot_idx {slot_idx} out of "
                f"range [0, {self.num_slots})"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(src, dst)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (formerly pk.mtp_token_scatter_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.batch_size, self.num_slots, slot_idx]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(src, (-1, -1, -1), -1, True)
        tb_graph.new_input(dst, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([src, dst], tb_graph)
        pk.kn_graph.register_task(tb_graph, "mtp_token_scatter", params)
        return dst


class MTPFloatScatter(MPKModule):
    """Scatter per-request float32 probabilities into one column.

    Wraps :meth:`PersistentKernel.mtp_float_scatter_layer` (task
    ``mtp_float_scatter``). Same compile-time-index pattern as
    :class:`MTPTokenScatter`, but for float32 — used by the
    probabilistic MTP verifier to stash per-step draft probabilities
    into a ``[batch_size, num_slots]`` float buffer.

    Args:
        batch_size: First dim of ``src`` / ``dst``.
        num_slots:  Width of ``dst`` (max number of draft tokens).
        prefix:     vLLM/HF state_dict prefix.
    """

    def __init__(
        self,
        batch_size: int,
        num_slots: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if batch_size <= 0:
            raise ValueError(
                f"MTPFloatScatter batch_size must be positive; got {batch_size}"
            )
        if num_slots <= 0:
            raise ValueError(
                f"MTPFloatScatter num_slots must be positive; got {num_slots}"
            )
        self.batch_size = batch_size
        self.num_slots = num_slots

    # ------------------------------------------------------------------
    # PyTorch reference path — pure column write.
    # ------------------------------------------------------------------
    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        slot_idx: int,
    ) -> torch.Tensor:
        """Reference: ``dst[:, slot_idx] = src[:, 0]`` (float32)."""
        if src.dim() != 2 or src.shape[1] != 1:
            raise ValueError(
                f"MTPFloatScatter.forward expects src of shape "
                f"(batch_size, 1); got {tuple(src.shape)}"
            )
        if dst.dim() != 2:
            raise ValueError(
                f"MTPFloatScatter.forward expects 2-D dst; "
                f"got shape {tuple(dst.shape)}"
            )
        if not (0 <= slot_idx < dst.shape[1]):
            raise ValueError(
                f"MTPFloatScatter.forward slot_idx {slot_idx} out of "
                f"range [0, {dst.shape[1]})"
            )
        dst[:, slot_idx] = src[:, 0]
        return dst

    # ------------------------------------------------------------------
    # Grid heuristic.
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Default grid ``(1, 1, 1)`` — same reasoning as
        :class:`MTPTokenScatter`.
        """
        return (1, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path.
    # ------------------------------------------------------------------
    def compile(
        self,
        src: DTensor,
        dst: DTensor,
        slot_idx: int,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``mtp_float_scatter`` task on the active PK.

        Args:
            src:      ``(batch_size, 1)`` float32 DTensor — typically a
                      ``softmax_gather`` output.
            dst:      ``(batch_size, num_slots)`` float32 DTensor —
                      per-step prob accumulator.
            slot_idx: Compile-time column index.
            grid_dim: Override; ``None`` -> :meth:`auto_grid_dim`.
            block_dim: Override; ``None`` -> :meth:`default_block_dim`.

        Returns:
            ``dst`` (consumed for its side effect).
        """
        pk = current_pk()

        if not (0 <= slot_idx < self.num_slots):
            raise ValueError(
                f"MTPFloatScatter.compile slot_idx {slot_idx} out of "
                f"range [0, {self.num_slots})"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(src, dst)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (formerly pk.mtp_float_scatter_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.batch_size, self.num_slots, slot_idx]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(src, (-1, -1, -1), -1, True)
        tb_graph.new_input(dst, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([src, dst], tb_graph)
        pk.kn_graph.register_task(tb_graph, "mtp_float_scatter", params)
        return dst
