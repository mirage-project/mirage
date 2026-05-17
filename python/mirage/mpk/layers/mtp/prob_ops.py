"""Probability buffer scatter/extract — symmetric pair for prob accumulation.

Two SM100 helpers that move per-step probabilities in and out of a
``(batch, max_positions)`` rolling float32 buffer keyed by a runtime
``step``/``offset`` int32 meta-tensor:

* :class:`ProbScatter` -> :meth:`PersistentKernel.prob_scatter_layer`
  -> task ``prob_scatter_sm100``. Writes ``prob[b, 0]`` into
  ``buffer[b, step_counter[b]]``.

* :class:`ProbExtract` -> :meth:`PersistentKernel.prob_extract_layer`
  -> task ``prob_extract_sm100``. Reads
  ``buffer[b, offset[b]+1 : offset[b]+1+num_extract]`` into a
  contiguous ``(batch, num_extract)`` output.

These are the read/write halves of the probabilistic-MTP target-probability
buffer: main-model softmax-gather results are scattered each iteration via
:class:`ProbScatter`, and at verify time the target probabilities for the
next ``num_extract`` positions are extracted via :class:`ProbExtract`.

Runtime-metadata dependence
---------------------------
Both kernels read an int32 ``step_counter`` / ``offset`` tensor. In
production these are MPK runtime meta-tensors (see the DeepSeek V3
builder ``_cached_attach(step_tensor, ...)`` pattern), but they appear as
ordinary input DTensors to these layers. :meth:`forward` is provided as
a faithful PyTorch reference that takes the offset/counter as a real
torch tensor.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

from ....core import DTensor


__all__ = ["ProbScatter", "ProbExtract"]


class ProbScatter(MPKModule):
    """Scatter per-request prob into a rolling per-position buffer.

    Wraps :meth:`PersistentKernel.prob_scatter_layer` (task
    ``prob_scatter_sm100``). For each batch row ``b``::

        buffer[b, step_counter[b]] = prob[b, 0]

    Args:
        max_positions: Width of ``buffer`` (max number of positions
            tracked). Baked into the task params.
        prefix:        vLLM/HF state_dict prefix.
    """

    def __init__(
        self,
        max_positions: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if max_positions <= 0:
            raise ValueError(
                f"ProbScatter max_positions must be positive; "
                f"got {max_positions}"
            )
        self.max_positions = max_positions

    # ------------------------------------------------------------------
    # PyTorch reference.
    # ------------------------------------------------------------------
    def forward(
        self,
        prob: torch.Tensor,
        step_counter: torch.Tensor,
        buffer: torch.Tensor,
    ) -> torch.Tensor:
        """Reference: ``buffer[b, step_counter[b]] = prob[b, 0]``.

        Args:
            prob:         ``(batch, 1)`` float32.
            step_counter: ``(batch,)`` int32 (or int64) — per-request
                          write position into ``buffer``.
            buffer:       ``(batch, max_positions)`` float32. Mutated.

        Returns:
            ``buffer`` (after in-place write).
        """
        if prob.dim() != 2 or prob.shape[1] != 1:
            raise ValueError(
                f"ProbScatter.forward expects prob of shape (batch, 1); "
                f"got {tuple(prob.shape)}"
            )
        if buffer.dim() != 2:
            raise ValueError(
                f"ProbScatter.forward expects 2-D buffer; "
                f"got shape {tuple(buffer.shape)}"
            )
        batch = prob.shape[0]
        if step_counter.numel() != batch:
            raise ValueError(
                f"ProbScatter.forward step_counter must have batch={batch} "
                f"elements; got {step_counter.numel()}"
            )
        batch_idx = torch.arange(
            batch, device=buffer.device, dtype=torch.int64
        )
        pos = step_counter.flatten().to(torch.int64)
        buffer[batch_idx, pos] = prob[:, 0]
        return buffer

    # ------------------------------------------------------------------
    # Grid heuristic.
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Default grid ``(1, 1, 1)``."""
        return (1, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path.
    # ------------------------------------------------------------------
    def compile(
        self,
        prob: DTensor,
        step_counter: DTensor,
        buffer: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``prob_scatter_sm100`` task on the active PK.

        Args:
            prob:         ``(batch, 1)`` float32 DTensor.
            step_counter: ``(batch,)`` int32 DTensor — runtime step.
            buffer:       ``(batch, max_positions)`` float32 DTensor.
            grid_dim:     Override; ``None`` -> :meth:`auto_grid_dim`.
            block_dim:    Override; ``None`` -> :meth:`default_block_dim`.

        Returns:
            ``buffer`` (consumed for its side effect).
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(prob, step_counter, buffer)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (formerly pk.prob_scatter_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.max_positions]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(prob, (-1, -1, -1), -1, True)
        tb_graph.new_input(step_counter, (-1, -1, -1), -1, True)
        tb_graph.new_input(buffer, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([prob, step_counter, buffer], tb_graph)
        pk.kn_graph.register_task(tb_graph, "prob_scatter_sm100", params)
        return buffer


class ProbExtract(MPKModule):
    """Extract a contiguous slice of the rolling prob buffer.

    Wraps :meth:`PersistentKernel.prob_extract_layer` (task
    ``prob_extract_sm100``). For each batch row ``b``::

        output[b, :] = buffer[b, offset[b]+1 : offset[b]+1+num_extract]

    Args:
        max_positions: Width of ``buffer``. Baked into params.
        num_extract:   Number of positions to read into ``output``
            (typically ``num_draft_tokens``). Baked into params.
        prefix:        vLLM/HF state_dict prefix.
    """

    def __init__(
        self,
        max_positions: int,
        num_extract: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if max_positions <= 0:
            raise ValueError(
                f"ProbExtract max_positions must be positive; "
                f"got {max_positions}"
            )
        if num_extract <= 0:
            raise ValueError(
                f"ProbExtract num_extract must be positive; "
                f"got {num_extract}"
            )
        self.max_positions = max_positions
        self.num_extract = num_extract

    # ------------------------------------------------------------------
    # PyTorch reference.
    # ------------------------------------------------------------------
    def forward(
        self,
        buffer: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """Reference: ``buffer[b, offset[b]+1 : offset[b]+1+num_extract]``.

        Args:
            buffer: ``(batch, max_positions)`` float32.
            offset: ``(batch,)`` int32 — per-request start (kernel reads
                    ``offset[b]+1 .. offset[b]+num_extract``).

        Returns:
            ``(batch, num_extract)`` float32.
        """
        if buffer.dim() != 2:
            raise ValueError(
                f"ProbExtract.forward expects 2-D buffer; "
                f"got shape {tuple(buffer.shape)}"
            )
        batch = buffer.shape[0]
        if offset.numel() != batch:
            raise ValueError(
                f"ProbExtract.forward offset must have batch={batch} "
                f"elements; got {offset.numel()}"
            )
        offsets = offset.flatten().to(torch.int64)
        out = torch.empty(
            (batch, self.num_extract),
            device=buffer.device,
            dtype=buffer.dtype,
        )
        # Faithful loop — kernel does the same indexing per request.
        for b in range(batch):
            start = int(offsets[b].item()) + 1
            out[b] = buffer[b, start : start + self.num_extract]
        return out

    # ------------------------------------------------------------------
    # Grid heuristic.
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Default grid ``(1, 1, 1)``."""
        return (1, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path.
    # ------------------------------------------------------------------
    def compile(
        self,
        buffer: DTensor,
        offset: DTensor,
        output: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``prob_extract_sm100`` task on the active PK.

        Args:
            buffer: ``(batch, max_positions)`` float32 DTensor.
            offset: ``(batch,)`` int32 DTensor — runtime offset.
            output: ``(batch, num_extract)`` float32 DTensor. Written by
                    the kernel.
            grid_dim: Override; ``None`` -> :meth:`auto_grid_dim`.
            block_dim: Override; ``None`` -> :meth:`default_block_dim`.

        Returns:
            ``output`` (consumed for its side effect).
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(buffer, offset, output)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (formerly pk.prob_extract_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        params = [self.max_positions, self.num_extract]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(buffer, (-1, -1, -1), -1, True)
        tb_graph.new_input(offset, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([buffer, offset, output], tb_graph)
        pk.kn_graph.register_task(tb_graph, "prob_extract_sm100", params)
        return output
