"""Fused softmax + gather (single-row probability lookup).

Catalog wrapper around :meth:`PersistentKernel.softmax_gather_layer`
(task ``softmax_gather_sm100``). Computes::

    output_probs[b, 0] = softmax(logits[b])[token_ids[b, 0]]

in one fused pass — used by DeepSeek V3's probabilistic MTP verify
path to extract the per-step probability of the chosen token without
materialising the full ``(batch, vocab)`` softmax.

Tensor contract
---------------
* ``logits``       — ``(batch, vocab_size)`` bf16.
* ``token_ids``    — ``(batch, 1)`` int64.
* ``output_probs`` — ``(batch, 1)`` float32 (single fp32 because the
  downstream :class:`ProbScatter` / verify expects fp32 probs).

Why fp32 output
---------------
The softmax denominator can vanish at bf16 precision when the chosen
logit is far below the max. The kernel does the reduction in fp32 and
emits fp32; downstream probabilistic-verify math (``P_target / P_draft``
ratio) is also fp32.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .._base import BlockDim, GridDim, MPKModule
from ...context import current_pk

from ....core import DTensor


__all__ = ["SoftmaxGather"]


class SoftmaxGather(MPKModule):
    """Fused softmax + token-id gather.

    Wraps :meth:`PersistentKernel.softmax_gather_layer`.

    Args:
        prefix: vLLM/HF state_dict prefix. No weights here; used only as
            a debug name uniquifier.
    """

    def __init__(self, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)

    # ------------------------------------------------------------------
    # PyTorch reference — straightforward.
    # ------------------------------------------------------------------
    def forward(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Reference: per-row softmax then gather at ``token_ids``.

        Args:
            logits:    ``(batch, vocab_size)`` bf16-or-fp32.
            token_ids: ``(batch, 1)`` int64.

        Returns:
            ``(batch, 1)`` float32.
        """
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
        # Reduce in fp32 to match the kernel.
        probs = torch.softmax(logits.float(), dim=-1)
        # Gather: probs[b, token_ids[b, 0]] -> (batch, 1)
        batch_idx = torch.arange(
            probs.shape[0], device=probs.device, dtype=torch.int64
        )
        gathered = probs[batch_idx, token_ids[:, 0].to(torch.int64)]
        return gathered.unsqueeze(-1).to(torch.float32)

    # ------------------------------------------------------------------
    # Grid heuristic.
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Default grid ``(1, 1, 1)``.

        The kernel iterates over ``batch`` internally; one CTA is
        sufficient. Matches the DeepSeek V3 builder caller.
        """
        return (1, 1, 1)

    # ------------------------------------------------------------------
    # MPK compile path.
    # ------------------------------------------------------------------
    def compile(
        self,
        logits: DTensor,
        token_ids: DTensor,
        output_probs: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``softmax_gather_sm100`` task on the active PK.

        Args:
            logits:        ``(batch, vocab_size)`` bf16 DTensor.
            token_ids:     ``(batch, 1)`` int64 DTensor.
            output_probs:  ``(batch, 1)`` float32 DTensor (the kernel
                           writes here).
            grid_dim:      Override; ``None`` -> :meth:`auto_grid_dim`.
            block_dim:     Override; ``None`` -> :meth:`default_block_dim`.

        Returns:
            ``output_probs`` (consumed for its side effect).
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

        # Inlined task registration (formerly pk.softmax_gather_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert logits.num_dims == 2
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(logits, (-1, -1, -1), -1, True)
        tb_graph.new_input(token_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(output_probs, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([logits, token_ids, output_probs], tb_graph)
        pk.kn_graph.register_task(tb_graph, "softmax_gather_sm100")
        return output_probs
