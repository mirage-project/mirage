"""Rolling probability buffer scatter / extract (SM100).

Symmetric pair around ``prob_scatter_sm100`` / ``prob_extract_sm100``
in ``include/mirage/persistent_kernel/tasks/blackwell/prob_scatter_sm100.cuh``.
``ProbScatter`` writes ``buffer[b, step_counter[b]]``; ``ProbExtract``
reads ``buffer[b, offset[b]+1 : offset[b]+1+num_extract]`` (verify
starts at ``step+1``). Both indices are int32 runtime meta-tensors.
"""
from __future__ import annotations

from typing import Optional

import torch

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

from ....core import DTensor


__all__ = ["ProbScatter", "ProbExtract"]


class ProbScatter(MPKModule):
    """Scatter ``prob[b, 0]`` into ``buffer[b, step_counter[b]]``.

    Wraps task ``prob_scatter_sm100``. Params: ``[max_positions]``.
    ``step_counter`` is int32; out-of-range positions are dropped.
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

    def forward(
        self,
        prob: torch.Tensor,
        step_counter: torch.Tensor,
        buffer: torch.Tensor,
    ) -> torch.Tensor:
        """Reference: ``buffer[b, step_counter[b]] = prob[b, 0]`` (in-place)."""
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

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """``(1, 1, 1)`` — kernel loops batch internally (``tid==0`` writes
        all rows); single scalar write per batch element."""
        return (1, 1, 1)

    def compile(
        self,
        prob: DTensor,
        step_counter: DTensor,
        buffer: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``prob_scatter_sm100`` task.

        Tensor contract:
          prob:         (batch_size, 1) float32, dense. Per-batch prob.
          step_counter: (batch_size,) int32. Runtime position index per
                        batch row; kernel reads ``step_counter[b]``.
          buffer:       (batch_size, max_positions) float32, dense.
                        In-place output; writes a single column per row
                        (``buffer[b, step_counter[b]]``), no-op if OOB.
        Params: ``[max_positions]``; ``batch_size`` inferred.

        Notes: ``step_counter`` is the runtime meta-tensor dependency.
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(prob, step_counter, buffer)
        if block_dim is None:
            block_dim = self.default_block_dim()

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
    """Read ``buffer[b, offset[b]+1 : offset[b]+1+num_extract]``.

    Wraps task ``prob_extract_sm100``. Params:
    ``[max_positions, num_extract]``. The ``+1`` matches the verifier
    starting at ``step + 1``; out-of-range reads emit ``0.0f``.
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

    def forward(
        self,
        buffer: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        """Reference: ``buffer[b, offset[b]+1 : offset[b]+1+num_extract]``."""
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
        for b in range(batch):
            start = int(offsets[b].item()) + 1
            out[b] = buffer[b, start : start + self.num_extract]
        return out

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """``(1, 1, 1)`` — kernel parallelises ``num_extract`` along
        ``threadIdx.x`` and strides batch internally; one CTA suffices."""
        return (1, 1, 1)

    def compile(
        self,
        buffer: DTensor,
        offset: DTensor,
        output: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``prob_extract_sm100`` task.

        Tensor contract:
          buffer: (batch_size, max_positions) float32, dense. Source
                  rolling prob buffer.
          offset: (batch_size,) int32. Runtime offset per row; kernel
                  reads ``buffer[b, offset[b]+1 : offset[b]+1+K]``.
          output: (batch_size, num_extract) float32, dense. Output;
                  out-of-range slots get ``0.0f``.
        Params: ``[max_positions, num_extract]``.

        Notes: ``offset`` is the runtime meta-tensor dependency; the
        ``+1`` aligns with verify starting at ``step + 1``.
        """
        pk = current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(buffer, offset, output)
        if block_dim is None:
            block_dim = self.default_block_dim()

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
