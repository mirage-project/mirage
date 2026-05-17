"""Fused softmax + single-token gather (SM100).

Wraps task ``softmax_gather_sm100`` in
``include/mirage/persistent_kernel/tasks/blackwell/softmax_gather_sm100.cuh``.
Computes ``output_probs[b, 0] = softmax(logits[b])[token_ids[b, 0]]`` in
one fused pass, reducing in fp32 and emitting fp32. Used by MTP's
probabilistic verifier to extract per-step token probabilities without
materialising the full ``(batch, vocab)`` softmax.
"""
from __future__ import annotations

from typing import Optional

import torch

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

from ....core import DTensor


__all__ = ["SoftmaxGather"]


class SoftmaxGather(MPKModule):
    """Fused per-row softmax then gather at ``token_ids``.

    Wraps task ``softmax_gather_sm100``. Output is fp32 because the
    softmax denominator can vanish at bf16 when the chosen logit lies
    far below the max, and the probabilistic-verify ratio is also fp32.
    """

    def __init__(self, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)

    def forward(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Reference: per-row ``softmax`` (fp32) then gather at
        ``token_ids``. Inputs ``(batch, vocab)`` bf16/fp32 and
        ``(batch, 1)`` int64; output ``(batch, 1)`` float32."""
        if logits.dim() != 2:
            raise ValueError(
                f"SoftmaxGather.forward expects 2-D logits; "
                f"got shape {tuple(logits.shape)}"
            )
        if token_ids.dim() != 2 or token_ids.shape[1] != 1:
            raise ValueError(
                f"SoftmaxGather.forward expects token_ids of shape "
                f"(batch, 1); got {tuple(token_ids.shape)}"
            )
        probs = torch.softmax(logits.float(), dim=-1)
        batch_idx = torch.arange(
            probs.shape[0], device=probs.device, dtype=torch.int64
        )
        gathered = probs[batch_idx, token_ids[:, 0].to(torch.int64)]
        return gathered.unsqueeze(-1).to(torch.float32)

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """``(1, 1, 1)`` — kernel loops batch internally
        (``for batch_idx in [0, BATCH_SIZE)``); one CTA suffices."""
        return (1, 1, 1)

    def compile(
        self,
        logits: DTensor,
        token_ids: DTensor,
        output_probs: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``softmax_gather_sm100`` task.

        Tensor contract:
          logits:       (batch_size, vocab_size) bf16, dense. Hard-coded
                        ``cute::bfloat16_t`` in task_register.cc.
          token_ids:    (batch_size, num_extract) int64, dense. Kernel
                        reads ``token_ids[b, 0]`` only (single gather).
          output_probs: (batch_size, num_extract) float32, dense.
                        Output; writes ``output_probs[b, 0]`` only.
        Params: none (registrar infers from ``logits`` dims).

        Notes: fp32 output regardless of input. No runtime meta deps.
        """
        pk = current_pk()

        if logits.num_dims != 2:
            raise ValueError(
                f"SoftmaxGather.compile expects 2-D logits DTensor; "
                f"got num_dims={logits.num_dims}"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(logits, token_ids, output_probs)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(logits, (-1, -1, -1), -1, True)
        tb_graph.new_input(token_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_probs, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([logits, token_ids, output_probs], tb_graph)
        pk.kn_graph.register_task(tb_graph, "softmax_gather_sm100")
        return output_probs
