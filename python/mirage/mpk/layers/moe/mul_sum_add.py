"""MoE combine — weighted sum across top-k slots + residual add.

Wraps :meth:`PersistentKernel.moe_mul_sum_add_layer` (task
``moe_mul_sum_add_sm100``, Blackwell only). This is the final step of
the (OLD) MoE pipeline: combine the per-slot W2 outputs by the
top-k weights and add the residual::

    output[b, h] = residual[b, h]
                 + sum_k(input[b, k, h] * weight[b, k])

The NEW MoE path (PR-674 group GEMM) folds this combine into
``moe_unpermute_sm100`` instead — :class:`MoEUnpermute` — and skips
``moe_mul_sum_add`` entirely.

Tensor contract
---------------

* ``input``    : ``(batch_size, num_experts_per_tok, hidden_size)``
  bf16 — the routed W2 output.
* ``weight``   : ``(batch_size, num_experts_per_tok)`` float32 — the
  renormalized top-k weights from routing.
* ``residual`` : ``(batch_size, hidden_size)`` bf16 — the
  shared-expert + transformer residual chain (DeepSeek V3) or the
  layer-input residual (qwen3).
* ``output``   : ``(batch_size, hidden_size)`` bf16 — caller-allocated.

Parallelism
-----------

``grid_dim = (batch_size, hidden_split, 1)`` where ``hidden_split``
divides ``hidden_size`` — typically ``hidden_size // 256`` (qwen3) or
the value of ``_moe_hidden_split(hidden_size)`` in the DeepSeek V3
builder. Block dim 128 (Blackwell convention; qwen3 used 256 on the
older Ampere/Hopper path, but the sm100 kernel asserts 128).

Forward (PyTorch reference) is the direct expression above in fp32,
cast back to the input dtype.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["MoeMulSumAdd"]


class MoeMulSumAdd(MPKModule):
    """MoE top-k combine + residual add.

    Args:
        hidden_size: Output / residual trailing dim.
        num_experts_per_tok: Top-k width (input dim 1, weight dim 1).
        prefix: HF state_dict prefix (no parameters live here; the
            prefix names the output DTensor when caller passes a torch
            tensor).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts_per_tok: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.hidden_size = hidden_size
        self.num_experts_per_tok = num_experts_per_tok

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """``output = residual + sum_k(x[..., k, :] * topk_weights[..., k, None])``.

        Computed in fp32 and cast back to ``x``'s dtype.
        """
        if x.dim() != 3 or x.size(-1) != self.hidden_size:
            raise ValueError(
                f"x must have shape (B, K, {self.hidden_size}); "
                f"got {tuple(x.shape)}"
            )
        if x.size(1) != self.num_experts_per_tok:
            raise ValueError(
                f"x.size(1) must equal num_experts_per_tok="
                f"{self.num_experts_per_tok}; got {x.size(1)}"
            )
        if topk_weights.shape != (x.size(0), self.num_experts_per_tok):
            raise ValueError(
                f"topk_weights must have shape "
                f"({x.size(0)}, {self.num_experts_per_tok}); "
                f"got {tuple(topk_weights.shape)}"
            )
        if residual.shape != (x.size(0), self.hidden_size):
            raise ValueError(
                f"residual must have shape ({x.size(0)}, {self.hidden_size}); "
                f"got {tuple(residual.shape)}"
            )
        combined = (
            x.float() * topk_weights.float().unsqueeze(-1)
        ).sum(dim=1)
        return (residual.float() + combined).to(x.dtype)

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, x_dt: DTensor) -> GridDim:
        """Default: ``(batch_size, hidden_split, 1)`` with ``hidden_split = hidden_size // 256``.

        Matches both qwen3 (``hidden_size // 256``) and the DeepSeek V3
        builder's ``_moe_hidden_split`` default. Falls back to the
        largest divisor of ``hidden_size`` that's <= 16 if the preferred
        split is not a divisor.
        """
        preferred = max(1, self.hidden_size // 256)
        gy = preferred
        # Walk down to a divisor of hidden_size.
        while self.hidden_size % gy != 0 and gy > 1:
            gy -= 1
        return (x_dt.dim(0), gy, 1)

    def default_block_dim(self) -> BlockDim:
        """``moe_mul_sum_add_sm100`` runs 128 threads per CTA on Blackwell."""
        return (128, 1, 1)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
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
        """Register a ``moe_mul_sum_add_sm100`` task."""
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if x.num_dims != 3:
            raise ValueError(
                f"MoeMulSumAdd.compile: x must be 3-D; got num_dims={x.num_dims}"
            )
        if topk_weights.num_dims != 2:
            raise ValueError(
                f"MoeMulSumAdd.compile: topk_weights must be 2-D; "
                f"got num_dims={topk_weights.num_dims}"
            )
        if residual.num_dims != 2 or output.num_dims != 2:
            raise ValueError(
                "MoeMulSumAdd.compile: residual and output must both be 2-D"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (formerly pk.moe_mul_sum_add_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        assert x.num_dims == 3  # (batch_size, num_experts_per_tok, hidden_size)
        assert topk_weights.num_dims == 2  # (batch_size, num_experts_per_tok)
        assert residual.num_dims == 2  # (batch_size, hidden_size)
        assert output.num_dims == 2  # (batch_size, hidden_size)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x, (0, 2, -1), -1, True)
        tb_graph.new_input(topk_weights, (0, -1, -1), -1, True)
        tb_graph.new_input(residual, (0, 1, -1), -1, True)
        tb_graph.new_input(output, (0, 1, -1), -1, True)
        pk.kn_graph.customized([x, topk_weights, residual, output], tb_graph)
        pk.kn_graph.register_task(tb_graph, "moe_mul_sum_add_sm100")
        return output
