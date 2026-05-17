"""MoE combine: weighted sum across top-k slots + residual add.

Wraps ``moe_mul_sum_add_sm100`` (kernel:
``include/mirage/persistent_kernel/tasks/blackwell/mul_sum_add_sm100.cuh``
via the moe task variant). Computes
``output[b, h] = residual[b, h] + sum_k(input[b, k, h] * weight[b, k])``.
"""
from __future__ import annotations

from typing import Optional

import torch

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["MoeMulSumAdd"]


class MoeMulSumAdd(MPKModule):
    """Top-k combine plus residual add (final step of the per-expert MoE).

    Inputs: ``input(B, K, hidden)`` bf16, ``weight(B, K)`` fp32 top-k
    weights from routing, ``residual(B, hidden)`` bf16. Output bf16.
    """

    def __init__(self, hidden_size: int, num_experts_per_tok: int, *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        self.hidden_size = hidden_size
        self.num_experts_per_tok = num_experts_per_tok

    def forward(
        self, x: torch.Tensor, topk_weights: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        """``residual + sum_k(x[..., k, :] * topk_weights[..., k, None])`` in fp32."""
        if x.dim() != 3 or x.size(-1) != self.hidden_size:
            raise ValueError(f"x must be (B, K, {self.hidden_size}); got {tuple(x.shape)}")
        if x.size(1) != self.num_experts_per_tok:
            raise ValueError(
                f"x.size(1) must equal num_experts_per_tok="
                f"{self.num_experts_per_tok}; got {x.size(1)}"
            )
        if topk_weights.shape != (x.size(0), self.num_experts_per_tok):
            raise ValueError(
                f"topk_weights must be ({x.size(0)}, {self.num_experts_per_tok}); "
                f"got {tuple(topk_weights.shape)}"
            )
        if residual.shape != (x.size(0), self.hidden_size):
            raise ValueError(
                f"residual must be ({x.size(0)}, {self.hidden_size}); "
                f"got {tuple(residual.shape)}"
            )
        combined = (x.float() * topk_weights.float().unsqueeze(-1)).sum(dim=1)
        return (residual.float() + combined).to(x.dtype)

    def auto_grid_dim(self, x_dt: DTensor) -> GridDim:
        """``(B, hidden_size // 256, 1)``, walked down to a divisor of hidden_size."""
        preferred = max(1, self.hidden_size // 256)
        gy = preferred
        while self.hidden_size % gy != 0 and gy > 1:
            gy -= 1
        return (x_dt.dim(0), gy, 1)

    def default_block_dim(self) -> BlockDim:
        """``moe_mul_sum_add_sm100`` runs 128 threads per CTA on Blackwell."""
        return (128, 1, 1)

    def compile(
        self,
        x: DTensor,
        topk_weights: DTensor,
        residual: DTensor,
        output: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``moe_mul_sum_add_sm100`` — top-k combine + residual add.

        Tensor contract:
          x: (B, num_experts_per_tok, hidden_size) bf16 — expert outputs (W2 output).
          topk_weights: (B, num_experts_per_tok) fp32 — the routing scores (kernel reads
            ``float const*``; NOT bf16 despite name overlap with routing-layer aliases).
          residual: (B, hidden_size) bf16 — additive (e.g., shared-expert output).
          output: (B, hidden_size) bf16 = residual + sum_k(x[..,k,:] * topk_weights[..,k]).

        Notes: grid is (B, hidden_size // 256_or_divisor, 1), 128 threads/CTA; all tensors
        share leading-dim B and trailing-dim hidden_size.
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        if x.num_dims != 3:
            raise ValueError(f"x must be 3-D; got num_dims={x.num_dims}")
        if topk_weights.num_dims != 2:
            raise ValueError(
                f"topk_weights must be 2-D; got num_dims={topk_weights.num_dims}"
            )
        if residual.num_dims != 2 or output.num_dims != 2:
            raise ValueError("residual and output must both be 2-D")

        grid_dim = grid_dim or self.auto_grid_dim(x)
        block_dim = block_dim or self.default_block_dim()
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x, (0, 2, -1), -1, True)
        tb_graph.new_input(topk_weights, (0, -1, -1), -1, True)
        tb_graph.new_input(residual, (0, 1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        pk.kn_graph.customized([x, topk_weights, residual, output], tb_graph)
        pk.kn_graph.register_task(tb_graph, "moe_mul_sum_add_sm100")
        return output
