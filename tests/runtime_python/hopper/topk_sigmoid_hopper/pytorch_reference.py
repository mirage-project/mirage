import torch


def sigmoid_routing_reference(logits, bias, num_groups, topk_group, topk,
                              scaling_factor):
    """DeepSeek-V3 group-aware sigmoid routing (as in HF / vLLM).

    Returns (topk_weights [B, topk] fp32, routing_indices [E, B] int32 with the
    1-based slot of each pick, expert_used [E] bool)."""
    batch, num_experts = logits.shape
    per_group = num_experts // num_groups
    scores = torch.sigmoid(logits.float())
    biased = scores + bias.float().unsqueeze(0)
    if topk_group < num_groups:
        group_scores = (biased.view(batch, num_groups, per_group)
                        .topk(2, dim=-1).values.sum(-1))
        top_groups = group_scores.topk(topk_group, dim=-1).indices
        group_mask = torch.zeros_like(group_scores).scatter_(1, top_groups, 1.0)
        expert_mask = (group_mask.unsqueeze(-1)
                       .expand(-1, -1, per_group).reshape(batch, num_experts))
        biased = biased.masked_fill(expert_mask == 0, float("-inf"))
    topk_idx = biased.topk(topk, dim=-1).indices
    weights = scores.gather(1, topk_idx)
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * scaling_factor

    routing = torch.zeros(num_experts, batch, dtype=torch.int32,
                          device=logits.device)
    slots = torch.arange(1, topk + 1, dtype=torch.int32, device=logits.device)
    for row in range(batch):
        routing[topk_idx[row], row] = slots
    used = torch.zeros(num_experts, dtype=torch.bool, device=logits.device)
    used[topk_idx.flatten()] = True
    return weights, routing, used
