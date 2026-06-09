"""
PyTorch reference implementations for sm100_moe_sigmoid tasks.

Used by both the kernel-wrapper test (test_gate_topk_sigmoid.py) and the
PersistentKernel test_mode test (test_topk_sigmoid_testmode.py).

The authoritative reference is `moe_topk_sigmoid_routing_ref`, lifted from
`reference_sigmoid_routing` in test_gate_topk_sigmoid.py (validated against
the actual SM100 hardware kernel).
"""

import torch


# ============================================================================
# DeepSeek V3 default routing configuration
# ============================================================================
DEFAULT_NUM_EXPERTS = 256          # n_routed_experts
DEFAULT_NUM_EXPERTS_PER_TOK = 8    # num_experts_per_tok
DEFAULT_NUM_GROUPS = 8             # n_group
DEFAULT_TOPK_GROUP = 4             # topk_group
DEFAULT_ROUTED_SCALING_FACTOR = 2.5  # routed_scaling_factor


def moe_topk_sigmoid_routing_ref(
    logits_bf16,
    bias,
    batch_size,
    num_experts=DEFAULT_NUM_EXPERTS,
    num_experts_per_tok=DEFAULT_NUM_EXPERTS_PER_TOK,
    num_groups=DEFAULT_NUM_GROUPS,
    topk_group=DEFAULT_TOPK_GROUP,
    routed_scaling_factor=DEFAULT_ROUTED_SCALING_FACTOR,
):
    """
    DeepSeek V3 group-aware sigmoid routing (matches DeepseekV3TopkRouter.forward
    + get_topk_indices).

    Args:
        logits_bf16: (batch_size, num_experts) gating logits, bfloat16.
        bias: (num_experts,) per-expert bias, float32.
        batch_size: int, number of tokens.
        num_experts: total number of routed experts.
        num_experts_per_tok: top-K experts selected per token.
        num_groups: number of expert groups for group-aware routing.
        topk_group: number of groups selected per token.
        routed_scaling_factor: scalar applied to normalized topk_weights.

    Returns:
        topk_weights: (batch_size, num_experts_per_tok) float32 weights, scaled.
        routing_indices: (num_experts, batch_size) int32, 1-indexed slot per token
            (0 = expert not active for that token).
        expert_active: (num_experts,) int32 mask of active experts (0/1).
    """
    experts_per_group = num_experts // num_groups

    # Step 1: sigmoid
    scores = torch.sigmoid(logits_bf16.float())

    # Step 2: add bias for selection decisions
    biased = scores + bias.unsqueeze(0)

    # Step 3: group top-2, sum -> group scores
    biased_grouped = biased.view(batch_size, num_groups, experts_per_group)
    top2_per_group, _ = biased_grouped.topk(2, dim=-1)
    group_scores = top2_per_group.sum(dim=-1)

    # Step 4: select top-K groups
    _, top_groups = group_scores.topk(topk_group, dim=-1, sorted=False)
    group_mask = torch.zeros(batch_size, num_groups, device=logits_bf16.device)
    group_mask.scatter_(1, top_groups, 1.0)
    expert_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, -1, experts_per_group)
        .reshape(batch_size, num_experts)
    )

    # Step 5: mask non-selected groups, find top-K experts
    biased_masked = biased.clone()
    biased_masked[expert_mask == 0] = -10000.0
    _, topk_indices = biased_masked.topk(num_experts_per_tok, dim=-1)

    # Step 6: gather ORIGINAL sigmoid scores (no bias)
    topk_weights = scores.gather(1, topk_indices)

    # Step 7: normalize + scale
    topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
    topk_weights = topk_weights * routed_scaling_factor

    # Build routing indices (expert-major, 1-indexed)
    routing_indices = torch.zeros(
        (num_experts, batch_size), device=logits_bf16.device, dtype=torch.int32
    )
    expert_active = torch.zeros(
        (num_experts,), device=logits_bf16.device, dtype=torch.int32
    )
    for tok in range(batch_size):
        for k in range(num_experts_per_tok):
            eidx = topk_indices[tok, k]
            routing_indices[eidx, tok] = k + 1
            expert_active[eidx] = 1

    return topk_weights, routing_indices, expert_active
