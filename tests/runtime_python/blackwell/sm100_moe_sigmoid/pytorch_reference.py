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
    local_expert_start=0,
    num_local_experts=None,
):
    """
    DeepSeek V3 group-aware sigmoid routing (matches DeepseekV3TopkRouter.forward
    + get_topk_indices), with expert-parallel (ep) local-expert slicing.

    The selection (sigmoid, group top-2, top-K groups, top-K experts) and the
    weight normalization always run over the FULL set of `num_experts` global
    experts — this mirrors the kernel, which loads the entire logit row and
    accumulates `weight_sum` over every selected expert regardless of locality.
    Locality (`local_expert_start` / `num_local_experts`) only restricts which of
    the selected experts get a NON-zero weight written, and which appear in the
    routing-indices / active-expert outputs (1-indexed within the local range).

    For ep_size=1 (`local_expert_start=0`, `num_local_experts=num_experts`) this
    is identical to the original full-routing reference (kernel start=0,end=256).

    Args:
        logits_bf16: (batch_size, num_experts) gating logits, bfloat16.
        bias: (num_experts,) per-expert bias, float32.
        batch_size: int, number of tokens.
        num_experts: total number of routed experts (global).
        num_experts_per_tok: top-K experts selected per token.
        num_groups: number of expert groups for group-aware routing.
        topk_group: number of groups selected per token.
        routed_scaling_factor: scalar applied to normalized topk_weights.
        local_expert_start: first GLOBAL expert id owned by this ep rank.
        num_local_experts: number of experts owned by this ep rank
            (defaults to `num_experts`, i.e. ep_size=1).

    Returns:
        topk_weights: (batch_size, num_experts_per_tok) float32 weights, scaled.
            Slots whose selected expert is NOT local are 0.0 (the global sum is
            still used for normalization).
        routing_indices: (num_local_experts, batch_size) int32, 1-indexed slot
            per token within the LOCAL expert range (0 = expert not active /
            not local for that token).
        expert_active: (num_local_experts,) int32 mask of active LOCAL experts.
    """
    if num_local_experts is None:
        num_local_experts = num_experts
    local_expert_end = local_expert_start + num_local_experts

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

    # Step 5: mask non-selected groups, find top-K experts (over ALL experts)
    biased_masked = biased.clone()
    biased_masked[expert_mask == 0] = -10000.0
    _, topk_indices = biased_masked.topk(num_experts_per_tok, dim=-1)

    # Step 6: gather ORIGINAL sigmoid scores (no bias)
    topk_scores = scores.gather(1, topk_indices)

    # Step 7: normalize over the GLOBAL top-K sum + scale (kernel: weight_sum is
    # accumulated over every selected expert, not just the local ones).
    weight_sum = topk_scores.sum(dim=-1, keepdim=True) + 1e-20
    topk_weights = topk_scores / weight_sum * routed_scaling_factor

    # Locality: zero the weight for any selected expert outside [start, end).
    is_local = (topk_indices >= local_expert_start) & (topk_indices < local_expert_end)
    topk_weights = topk_weights * is_local.to(topk_weights.dtype)

    # Build LOCAL routing indices (local-expert-major, 1-indexed) + active mask.
    routing_indices = torch.zeros(
        (num_local_experts, batch_size), device=logits_bf16.device, dtype=torch.int32
    )
    expert_active = torch.zeros(
        (num_local_experts,), device=logits_bf16.device, dtype=torch.int32
    )
    for tok in range(batch_size):
        for k in range(num_experts_per_tok):
            eidx = int(topk_indices[tok, k].item())
            if local_expert_start <= eidx < local_expert_end:
                local_eidx = eidx - local_expert_start
                routing_indices[local_eidx, tok] = k + 1
                expert_active[local_eidx] = 1

    return topk_weights, routing_indices, expert_active
