"""MoE top-k routing (softmax + sigmoid variants).

Two single-purpose catalog modules over the moe_topk_* kernels in
``include/mirage/persistent_kernel/tasks/blackwell``:

* :class:`MoETopkSoftmaxRouting` -> ``moe_topk_softmax_sm100``.
* :class:`MoETopkSigmoidRouting` -> ``moe_topk_sigmoid_sm100``
  (DeepSeek V3, with bias + group-limited top-k).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = [
    "MoETopkSoftmaxRouting",
    "MoETopkSigmoidRouting",
    "MoETopkRouting",  # back-compat factory
]


class _MoETopkRoutingBase(MPKModule):
    """Shared base for the two routing variants.

    Holds ``(num_experts, num_experts_per_tok, hidden_size, prefix)`` and
    the common output layout: ``moe_routing_indices`` is (E, B) int32
    *expert-major* (NOT (B, K)) — slot id is 1-indexed, 0 means "this
    token did not pick this expert"; ``moe_mask`` is (E+1,) int32 prefix
    counts that the W13/W2 kernels read to skip inactive experts.
    """

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        *,
        hidden_size: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if num_experts_per_tok > num_experts:
            raise ValueError(
                f"num_experts_per_tok ({num_experts_per_tok}) must be <= "
                f"num_experts ({num_experts})"
            )
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size

    # Both kernels have one CTA do the whole batch top-k — single-CTA grid
    # is the convention every existing caller uses.
    def auto_grid_dim(self, *_) -> GridDim:
        """Single-CTA routing — both variants run one block over the batch."""
        return (1, 1, 1)

    def default_block_dim(self) -> BlockDim:
        """Routing kernels assert 8 warps (256 threads)."""
        return (256, 1, 1)

    @staticmethod
    def _build_routing_outputs(
        topk_idx: torch.Tensor,
        num_experts: int,
        num_experts_per_tok: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Lay out (E, B) expert-major indices and (E+1,) prefix mask."""
        batch_size = topk_idx.size(0)
        device = topk_idx.device
        routing_indices = torch.zeros(
            (num_experts, batch_size), dtype=torch.int32, device=device
        )
        for slot in range(num_experts_per_tok):
            experts = topk_idx[:, slot].to(torch.long)
            routing_indices[experts, torch.arange(batch_size, device=device)] = (
                slot + 1
            )
        counts = (routing_indices != 0).sum(dim=1).to(torch.int32)
        mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)
        mask[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
        return routing_indices, mask


class MoETopkSoftmaxRouting(_MoETopkRoutingBase):
    """Softmax top-k routing (qwen3 MoE).

    Output ``moe_topk_weights`` = ``softmax(top_k(logits))`` over the
    kept top-k entries (float32). No per-expert bias; full-replica
    (TP-1).
    """

    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``softmax(top_k(logits))`` then expert-major indices + prefix mask."""
        if logits.dim() != 2 or logits.size(1) != self.num_experts:
            raise ValueError(
                f"logits must have shape (B, {self.num_experts}); "
                f"got {tuple(logits.shape)}"
            )
        topk_vals, topk_idx = torch.topk(
            logits.float(), k=self.num_experts_per_tok, dim=-1
        )
        topk_weights = F.softmax(topk_vals, dim=-1).to(torch.float32)
        routing_indices, mask = self._build_routing_outputs(
            topk_idx, self.num_experts, self.num_experts_per_tok
        )
        return topk_weights, routing_indices, mask

    def compile(
        self,
        logits: DTensor,
        moe_topk_weights: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor, DTensor]:
        """Register ``moe_topk_softmax_sm100`` (single CTA, 256 threads).

        Tensor contract:
          logits: (B, num_experts) bf16, the linear-projection output.
          moe_topk_weights: (B, num_experts_per_tok) fp32, renormalized softmax weights.
          moe_routing_indices: (num_experts, B) int32, EXPERT-MAJOR (NOT (B, K)).
            ``[e, t] = slot+1`` if token ``t`` picked expert ``e``, else 0.
          moe_mask: (num_experts + 1,) int32, prefix-count over routing_indices rows.

        Notes: softmax variant has no bias param; full-replica (TP=1, no expert split).
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        grid_dim = grid_dim or self.auto_grid_dim()
        block_dim = block_dim or self.default_block_dim()

        assert logits.num_dims == 2          # (batch_size, num_experts)
        assert moe_topk_weights.num_dims == 2  # (batch_size, num_experts_per_tok)
        assert moe_routing_indices.num_dims == 2  # (num_experts, batch_size)
        assert moe_mask.num_dims == 1            # (num_experts + 1,)
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(logits, (0, -1, -1), -1, True)
        tb_graph.new_input(moe_topk_weights, (0, -1, -1), -1, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [logits, moe_topk_weights, moe_routing_indices, moe_mask], tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "moe_topk_softmax_sm100")
        return moe_topk_weights, moe_routing_indices, moe_mask


class MoETopkSigmoidRouting(_MoETopkRoutingBase):
    """Group-limited sigmoid top-k routing (DeepSeek V3).

    Owns the per-expert ``e_score_correction_bias`` as ``self.bias``
    (fp32, loaded from ``{prefix}bias``). The bias enters expert
    *selection* but NOT the returned weights, which are the renormalized
    sigmoid scores scaled by ``routed_scaling_factor``.
    """

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        *,
        num_groups: int,
        topk_group: int,
        routed_scaling_factor: float,
        local_num_experts: Optional[int] = None,
        local_expert_start: int = 0,
        hidden_size: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            num_experts,
            num_experts_per_tok,
            hidden_size=hidden_size,
            prefix=prefix,
        )
        self.num_groups = num_groups
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.local_num_experts = (
            local_num_experts if local_num_experts is not None else num_experts
        )
        self.local_expert_start = local_expert_start
        # Per-expert score-correction bias (fp32 to match the kernel).
        self.bias = nn.Parameter(torch.zeros(num_experts, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """DeepSeek V3 group-limited routing (5-step procedure).

        Steps: sigmoid -> +bias -> per-group sum-top-2 -> keep top
        ``topk_group`` groups -> top-k by *biased* score among eligible
        experts -> renormalize the *unbiased* sigmoid scores and scale.
        The reference assumes no TP (local_num_experts == num_experts).
        """
        if logits.dim() != 2 or logits.size(1) != self.num_experts:
            raise ValueError(
                f"logits must have shape (B, {self.num_experts}); "
                f"got {tuple(logits.shape)}"
            )
        batch_size = logits.size(0)
        device = logits.device

        scores = torch.sigmoid(logits.float())
        biased = scores + self.bias.float()
        grouped = biased.view(batch_size, self.num_groups, -1)
        top2_per_group, _ = grouped.topk(2, dim=-1)
        group_score = top2_per_group.sum(dim=-1)
        kept_groups = group_score.topk(self.topk_group, dim=-1).indices
        group_mask = torch.zeros(
            (batch_size, self.num_groups), dtype=torch.bool, device=device
        )
        group_mask.scatter_(1, kept_groups, True)
        expert_mask = (
            group_mask.unsqueeze(-1).expand_as(grouped).reshape(batch_size, self.num_experts)
        )
        masked = biased.masked_fill(~expert_mask, float("-inf"))
        _, topk_idx = masked.topk(self.num_experts_per_tok, dim=-1)
        topk_scores = scores.gather(1, topk_idx)
        denom = topk_scores.sum(dim=-1, keepdim=True).clamp(min=1e-20)
        topk_weights = (
            topk_scores / denom * self.routed_scaling_factor
        ).to(torch.float32)
        routing_indices, mask = self._build_routing_outputs(
            topk_idx, self.num_experts, self.num_experts_per_tok
        )
        return topk_weights, routing_indices, mask

    def compile(
        self,
        logits: DTensor,
        moe_topk_weights: DTensor,
        moe_routing_indices: DTensor,
        moe_mask: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor, DTensor]:
        """Register ``moe_topk_sigmoid_sm100`` (DeepSeek V3; attaches bias).

        Tensor contract:
          logits: (B, total_num_experts) bf16, the linear-projection output.
          bias (self.bias, attached as ``{prefix}bias``): (total_num_experts,) fp32,
            the e_score_correction_bias (used for selection, NOT in output weights).
          moe_topk_weights: (B, num_experts_per_tok) fp32, renormalized-sigmoid * routed_scaling_factor.
          moe_routing_indices: (local_num_experts, B) int32, EXPERT-MAJOR (slot+1 or 0).
          moe_mask: (local_num_experts + 1,) int32, prefix counts.

        Notes: TP-friendly — only experts in ``[local_expert_start, +local_num_experts)``
        are written; params pack (num_groups, topk_group, scaling_bits, lo, hi).
        """
        import struct
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        grid_dim = grid_dim or self.auto_grid_dim()
        block_dim = block_dim or self.default_block_dim()

        bias_dt = pk.attach_input(self.bias, name=f"{self.prefix}bias")
        assert logits.num_dims == 2  # (batch_size, num_experts)
        total_num_experts = logits.dim(1)
        assert bias_dt.num_dims == 1 and bias_dt.dim(0) == total_num_experts
        assert moe_topk_weights.num_dims == 2     # (batch_size, num_experts_per_tok)
        assert moe_routing_indices.num_dims == 2  # (local_num_experts, batch_size)
        assert moe_mask.num_dims == 1             # (local_num_experts + 1,)
        local_num_experts = moe_routing_indices.dim(0)
        assert moe_mask.dim(0) == local_num_experts + 1
        assert 0 <= self.local_expert_start
        assert self.local_expert_start + local_num_experts <= total_num_experts

        scaling_bits = struct.unpack(
            "i", struct.pack("f", self.routed_scaling_factor)
        )[0]
        params = [
            self.num_groups,
            self.topk_group,
            scaling_bits,
            self.local_expert_start,
            self.local_expert_start + local_num_experts,
        ]
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(logits, (0, -1, -1), -1, True)
        tb_graph.new_input(bias_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_topk_weights, (0, -1, -1), -1, True)
        tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
        tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [logits, bias_dt, moe_topk_weights, moe_routing_indices, moe_mask],
            tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "moe_topk_sigmoid_sm100", params)
        return moe_topk_weights, moe_routing_indices, moe_mask


def MoETopkRouting(
    num_experts: int,
    num_experts_per_tok: int,
    *,
    variant: str = "softmax",
    num_groups: int = 1,
    topk_group: int = 1,
    routed_scaling_factor: float = 1.0,
    local_num_experts: Optional[int] = None,
    local_expert_start: int = 0,
    prefix: str = "",
):
    """Back-compat factory: dispatches to the softmax/sigmoid subclass.

    Tensor contract (delegated to the chosen subclass ``compile``):
      logits: (B, num_experts) bf16.
      (sigmoid only) bias: (num_experts,) fp32 (the e_score_correction_bias).
      out moe_topk_weights: (B, num_experts_per_tok) fp32.
      out moe_routing_indices: (num_experts, B) int32, EXPERT-MAJOR.
      out moe_mask: (num_experts + 1,) int32 prefix counts.

    Notes: ``variant='softmax'`` -> :class:`MoETopkSoftmaxRouting` (no bias);
    ``variant='sigmoid'`` -> :class:`MoETopkSigmoidRouting` (DeepSeek V3, group-limited).
    """
    if variant == "softmax":
        return MoETopkSoftmaxRouting(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            prefix=prefix,
        )
    if variant == "sigmoid":
        return MoETopkSigmoidRouting(
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            num_groups=num_groups,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            local_num_experts=local_num_experts,
            local_expert_start=local_expert_start,
            prefix=prefix,
        )
    raise ValueError(
        f"MoETopkRouting variant must be 'softmax' or 'sigmoid'; got {variant!r}"
    )
