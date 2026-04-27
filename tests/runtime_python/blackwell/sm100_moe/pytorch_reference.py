"""PyTorch reference implementations for sm100_moe layers.

Each function is self-contained, takes all inputs as arguments, and returns
the reference output tensor(s). Used by both kernel-wrapper tests
(test_w13_linear.py, test_w2_linear.py, test_silu_mul.py, test_weighted_sum.py)
and test_mode tests in this folder.
"""

import torch
from torch.nn import functional as F


def moe_w13_linear_ref(
    x,
    w,
    topk_expert_indices,
    num_experts,
    num_topk,
    batch_size,
    reduction_size,
    output_size,
    residual=None,
    expert_offset=0,
    expert_stride=1,
):
    """Reference for moe_w13_linear: per-token top-k expert linear projection.

    Computes torch_out[token, slot, :] = x[token] @ w[expert].T for each (token,
    slot) pair where the expert was selected. Optionally adds a residual.
    Mirrors the inline reference in test_w13_linear.py (lines 34-53).
    """
    expert_mask = torch.nn.functional.one_hot(
        topk_expert_indices, num_classes=num_experts
    ).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    torch_out = torch.zeros(
        (batch_size, num_topk, output_size), device=x.device, dtype=torch.bfloat16
    )
    for i, expert_idx in enumerate(expert_hit):
        if (i + expert_offset) % expert_stride != 0:
            continue
        expert_w = w[expert_idx].squeeze(0)
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        current_state = x[None, top_x].reshape(-1, reduction_size)
        current_hidden_states = torch.matmul(current_state, expert_w.T)
        if residual is not None:
            expert_residual = residual[expert_idx].squeeze(0)
            current_residual = expert_residual[None, top_x].reshape(-1, output_size)
            current_hidden_states += current_residual
        torch_out[top_x, idx] = current_hidden_states
    return torch_out, expert_hit


def moe_w2_linear_ref(
    x,
    w,
    topk_expert_indices,
    num_experts,
    num_topk,
    batch_size,
    reduction_size,
    output_size,
    residual=None,
    expert_offset=0,
    expert_stride=1,
):
    """Reference for moe_w2_linear: per-token top-k expert down-projection.

    Like w13 but input is (batch, num_topk, reduction_size) so each (token,
    slot) pair indexes its own intermediate. Mirrors the inline reference in
    test_w2_linear.py (lines 34-53).
    """
    expert_mask = torch.nn.functional.one_hot(
        topk_expert_indices, num_classes=num_experts
    ).permute(2, 1, 0)
    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
    torch_out = torch.zeros(
        (batch_size, num_topk, output_size), device=x.device, dtype=torch.bfloat16
    )
    for i, expert_idx in enumerate(expert_hit):
        if (i + expert_offset) % expert_stride != 0:
            continue
        expert_w = w[expert_idx].squeeze(0)
        idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))
        current_state = x[None, top_x, idx].reshape(-1, reduction_size)
        current_hidden_states = torch.matmul(current_state, expert_w.T)
        if residual is not None:
            expert_residual = residual[expert_idx].squeeze(0)
            current_residual = expert_residual[None, top_x].reshape(-1, output_size)
            current_hidden_states += current_residual
        torch_out[top_x, idx] = current_hidden_states
    return torch_out, expert_hit


def moe_silu_mul_ref(input, output_size):
    """Reference for moe_silu_mul: SiLU(input[..., :I]) * input[..., I:].

    Computed in float then cast to bfloat16. Mirrors the inline reference in
    test_silu_mul.py (lines 28-30).
    """
    w1_output = F.silu(input[:, :, :output_size].to(torch.float))
    torch_output = w1_output * input[:, :, output_size:].to(torch.float)
    return torch_output.to(torch.bfloat16)


def moe_mul_sum_add_ref(x, topk_weights, residual):
    """Reference for moe_mul_sum_add: weighted sum across top-k experts + residual.

    out = sum_k (x[:, k, :] * topk_weights[:, k:k+1]) + residual.
    Mirrors the inline reference in test_weighted_sum.py (lines 31-33).
    """
    torch_out = x.to(torch.float) * topk_weights.unsqueeze(-1)
    torch_out = torch_out.sum(dim=1).to(torch.bfloat16)
    torch_out += residual
    return torch_out
