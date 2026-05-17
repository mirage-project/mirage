"""MoE top-k routing (catalog module).

Wraps the two pk-level routing methods on
:class:`PersistentKernel`:

* :meth:`PersistentKernel.moe_topk_softmax_routing_layer` -> task
  ``moe_topk_softmax_sm100``. Used by qwen3 MoE (softmax scoring).
* :meth:`PersistentKernel.moe_topk_sigmoid_routing_layer` -> task
  ``moe_topk_sigmoid_sm100``. Used by DeepSeek V3 (sigmoid scoring with
  per-expert score-correction bias, group-limited top-k and a routed
  scaling factor).

Both variants consume the **router logits** (``input``, shape
``(batch_size, num_experts)``) and write the same three outputs:

* ``moe_topk_weights``   — ``(batch_size, num_experts_per_tok)`` float32.
  Per-token, per-slot expert weights AFTER renormalization. For the
  softmax variant this is the softmax of the top-k logits; for the
  sigmoid variant it is the (normalized) sigmoid of the top-k scores
  *multiplied by* ``routed_scaling_factor`` (DeepSeek V3 contract).
* ``moe_routing_indices`` — ``(local_num_experts, batch_size)`` int32,
  expert-major. ``moe_routing_indices[e, t]`` is the **slot index**
  (1-indexed within ``[1, num_experts_per_tok]``) of token ``t`` if
  expert ``e`` is one of the top-k for token ``t``, else 0. Local in
  the sense that with TP, each rank computes the slice over its
  ``local_num_experts`` only — ``local_expert_start`` selects which
  global-expert range this rank owns (sigmoid variant only; softmax
  is always full-replica).
* ``moe_mask``            — ``(local_num_experts + 1,)`` int32. Prefix
  count of routed tokens per local expert; the W13 / W2 kernels read
  this to know how many active tokens each expert processes.

DeepSeek V3 specifics (sigmoid variant)
---------------------------------------

DeepSeek V3 uses a *group-limited* top-k:

1. Compute per-expert sigmoid score ``s_e = sigmoid(logits_e)``.
2. Add per-expert ``e_score_correction_bias`` (loaded from
   ``model.layers.{i}.mlp.gate.e_score_correction_bias``) to produce the
   *selection score* (the bias does NOT affect the final weight).
3. Partition the experts into ``num_groups`` contiguous groups, then
   pick the ``topk_group`` groups with the largest "sum of top-2
   scores in group". Only experts inside the surviving groups are
   eligible.
4. Among the eligible experts, pick the top ``num_experts_per_tok``
   by selection score.
5. Renormalize the *selected experts' sigmoid scores* (without the
   bias) to sum to 1, then multiply by ``routed_scaling_factor``.

DeepSeek V3 ships ``num_groups=8``, ``topk_group=4``,
``num_experts_per_tok=8``, ``routed_scaling_factor=2.5``.

Parallelism
-----------

The routing kernel uses ``grid_dim=(1, 1, 1)`` with one CTA (256
threads / 8 warps) handling the whole batch. Both pk methods are
called this way in every existing caller (qwen3 MoE demo, DeepSeek V3
builder). ``auto_grid_dim`` returns ``(1, 1, 1)``; the block dim
defaults to ``(256, 1, 1)`` (the kernel asserts 8 warps).

Forward (PyTorch reference)
---------------------------

``forward()`` implements the math of either variant in plain PyTorch:

* ``variant="softmax"``: ``softmax(top_k(logits, k=num_experts_per_tok))``.
* ``variant="sigmoid"``: the 5-step DeepSeek-V3 group-limited
  procedure above.

The reference returns ``(topk_weights, routing_indices, mask)`` in the
exact layout the kernel writes (expert-major routing_indices, int32
mask). The reference assumes ``local_expert_start=0`` and
``local_num_experts==num_experts`` (no TP). With TP > 1, the kernel
slices its outputs over the local range; mirroring that in the eager
reference would require encoding the rank's expert window — out of
scope for a unit-test oracle.
"""
from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["MoETopkRouting"]


RoutingVariant = Literal["softmax", "sigmoid"]


class MoETopkRouting(MPKModule):
    """MoE top-k routing for both softmax (qwen3) and sigmoid (DeepSeek V3) variants.

    The class owns one ``variant`` and routes to the matching pk
    method. DeepSeek V3 also has an ``e_score_correction_bias``
    parameter; we expose it as an ``nn.Parameter`` named ``bias`` when
    ``variant="sigmoid"`` so the standard ``state_dict`` plumbing
    works.

    Args:
        num_experts: Total number of experts in the router logits
            (input ``dim(1)``).
        num_experts_per_tok: ``top-k`` width (also the second dim of
            ``moe_topk_weights``).
        variant: ``"softmax"`` | ``"sigmoid"``.
        num_groups: (sigmoid only) Number of expert groups. DeepSeek
            V3 uses ``8``. Ignored when ``variant="softmax"``.
        topk_group: (sigmoid only) Number of groups to keep. DeepSeek
            V3 uses ``4``. Ignored when ``variant="softmax"``.
        routed_scaling_factor: (sigmoid only) Multiplier on the
            renormalized sigmoid weight. DeepSeek V3 uses ``2.5``.
            Ignored when ``variant="softmax"``.
        local_num_experts: (sigmoid only) Size of this rank's expert
            window. Defaults to ``num_experts`` (no TP). The output
            tensors' shapes must already match this.
        local_expert_start: (sigmoid only) Starting index of this
            rank's expert window in the global expert table.
        prefix: HF state_dict prefix. ``self.bias`` (sigmoid only) is
            loaded from ``state_dict[f"{prefix}bias"]``; the standard
            DeepSeek-V3 key is
            ``model.layers.{i}.mlp.gate.e_score_correction_bias``, so
            pass ``prefix=f"model.layers.{i}.mlp.gate.e_score_correction_"``
            (note the trailing underscore) or use a custom loader.

    Tensor contract — ``compile()``:
        * ``input``                 : ``(batch_size, num_experts)``
          float32 router logits.
        * ``moe_topk_weights``      : ``(batch_size, num_experts_per_tok)``
          float32 — caller-allocated.
        * ``moe_routing_indices``   : ``(local_num_experts, batch_size)``
          int32 — caller-allocated.
        * ``moe_mask``              : ``(local_num_experts + 1,)``
          int32 — caller-allocated.

    Forward (PyTorch reference) returns the same triple. With
    ``variant="sigmoid"``, the bias parameter contributes to expert
    selection only; the returned weights are renormalized sigmoid
    scores scaled by ``routed_scaling_factor``.
    """

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        *,
        variant: RoutingVariant = "softmax",
        num_groups: int = 1,
        topk_group: int = 1,
        routed_scaling_factor: float = 1.0,
        local_num_experts: Optional[int] = None,
        local_expert_start: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if variant not in ("softmax", "sigmoid"):
            raise ValueError(
                f"MoETopkRouting variant must be 'softmax' or 'sigmoid'; "
                f"got {variant!r}"
            )
        if num_experts_per_tok > num_experts:
            raise ValueError(
                f"num_experts_per_tok ({num_experts_per_tok}) must be <= "
                f"num_experts ({num_experts})"
            )
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.variant = variant
        self.num_groups = num_groups
        self.topk_group = topk_group
        self.routed_scaling_factor = routed_scaling_factor
        self.local_num_experts = (
            local_num_experts if local_num_experts is not None else num_experts
        )
        self.local_expert_start = local_expert_start

        if variant == "sigmoid":
            # DeepSeek V3 per-expert score-correction bias. fp32 to match
            # the kernel's working precision.
            self.bias = nn.Parameter(torch.zeros(num_experts, dtype=torch.float32))
        else:
            # Softmax variant has no bias. Register a None placeholder so
            # state_dict doesn't pick up a stray attribute.
            self.register_parameter("bias", None)

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Plain-PyTorch top-k routing.

        Assumes ``local_expert_start == 0`` and ``local_num_experts ==
        num_experts`` (no TP) — the reference oracle is single-rank.
        """
        if logits.dim() != 2 or logits.size(1) != self.num_experts:
            raise ValueError(
                f"logits must have shape (B, {self.num_experts}); "
                f"got {tuple(logits.shape)}"
            )
        batch_size = logits.size(0)
        device = logits.device

        if self.variant == "softmax":
            # Top-k logits then softmax over the kept entries.
            topk_vals, topk_idx = torch.topk(
                logits.float(), k=self.num_experts_per_tok, dim=-1
            )
            topk_weights = F.softmax(topk_vals, dim=-1).to(torch.float32)
        else:
            # DeepSeek V3 group-limited routing.
            scores = torch.sigmoid(logits.float())
            biased = scores + self.bias.float()
            # Step 3: pick the top ``topk_group`` groups by
            # "sum of top-2 scores within group".
            grouped = biased.view(batch_size, self.num_groups, -1)
            top2_per_group, _ = grouped.topk(2, dim=-1)
            group_score = top2_per_group.sum(dim=-1)
            kept_groups = group_score.topk(self.topk_group, dim=-1).indices
            group_mask = torch.zeros(
                (batch_size, self.num_groups), dtype=torch.bool, device=device
            )
            group_mask.scatter_(1, kept_groups, True)
            expert_mask = (
                group_mask.unsqueeze(-1)
                .expand_as(grouped)
                .reshape(batch_size, self.num_experts)
            )
            masked = biased.masked_fill(~expert_mask, float("-inf"))
            # Step 4: top-k by selection score among eligible experts.
            _, topk_idx = masked.topk(self.num_experts_per_tok, dim=-1)
            # Step 5: renormalize raw sigmoid scores, then scale.
            topk_scores = scores.gather(1, topk_idx)
            denom = topk_scores.sum(dim=-1, keepdim=True).clamp(min=1e-20)
            topk_weights = (topk_scores / denom * self.routed_scaling_factor).to(
                torch.float32
            )

        # Build expert-major routing_indices and mask. Slot ids are
        # 1-indexed (matches kernel convention).
        routing_indices = torch.zeros(
            (self.num_experts, batch_size), dtype=torch.int32, device=device
        )
        for slot in range(self.num_experts_per_tok):
            experts = topk_idx[:, slot].to(torch.long)
            routing_indices[experts, torch.arange(batch_size, device=device)] = (
                slot + 1
            )
        # mask: prefix-count cumulative is the contract the kernel emits.
        counts = (routing_indices != 0).sum(dim=1).to(torch.int32)
        mask = torch.zeros(self.num_experts + 1, dtype=torch.int32, device=device)
        mask[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
        return topk_weights.to(torch.float32), routing_indices, mask

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_) -> GridDim:
        """Single-CTA routing — both variants run one block over the batch."""
        return (1, 1, 1)

    def default_block_dim(self) -> BlockDim:
        """Routing kernels assert 8 warps (256 threads)."""
        return (256, 1, 1)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
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
        """Register the chosen routing task on the active PersistentKernel.

        All output DTensors are caller-allocated (they're consumed by
        many downstream tasks; the catalog won't second-guess where
        they live). Returns the same triple for caller convenience.
        """
        from ... import context as _ctx

        pk = _ctx.current_pk()

        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Inlined task registration (formerly pk.moe_topk_*_routing_layer).
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        if self.variant == "softmax":
            assert logits.num_dims == 2  # (batch_size, num_experts)
            assert moe_topk_weights.num_dims == 2  # (batch_size, num_experts_per_tok)
            assert moe_routing_indices.num_dims == 2  # (num_experts, batch_size)
            assert moe_mask.num_dims == 1  # (num_experts + 1)
            tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
            tb_graph.new_input(logits, (0, -1, -1), -1, True)
            tb_graph.new_input(moe_topk_weights, (0, -1, -1), -1, True)
            tb_graph.new_input(moe_routing_indices, (-1, -1, -1), -1, True)
            tb_graph.new_input(moe_mask, (-1, -1, -1), -1, True)
            pk.kn_graph.customized(
                [logits, moe_topk_weights, moe_routing_indices, moe_mask],
                tb_graph,
            )
            pk.kn_graph.register_task(tb_graph, "moe_topk_softmax_sm100")
        else:
            # Sigmoid variant: attach the per-expert bias parameter.
            import struct

            bias_dt = pk.attach_input(
                self.bias, name=f"{self.prefix}bias"
            )
            assert logits.num_dims == 2  # (batch_size, num_experts)
            total_num_experts = logits.dim(1)
            assert bias_dt.num_dims == 1  # (num_experts,)
            assert bias_dt.dim(0) == total_num_experts
            assert moe_topk_weights.num_dims == 2  # (batch_size, num_experts_per_tok)
            assert moe_routing_indices.num_dims == 2  # (local_num_experts, batch_size)
            assert moe_mask.num_dims == 1  # (local_num_experts + 1)
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
            pk.kn_graph.register_task(
                tb_graph, "moe_topk_sigmoid_sm100", params
            )
        return moe_topk_weights, moe_routing_indices, moe_mask
