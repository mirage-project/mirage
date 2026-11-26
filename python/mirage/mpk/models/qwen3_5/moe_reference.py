"""Independent PyTorch oracle for the Qwen3.6 BF16 MoE block.

This follows the checkpoint's gate/up ordering and HF's default grouped_mm
reduction, including BF16 routing weights and the separate shared gate.
"""

import torch
import torch.nn.functional as F


def sum_expert_slots(expert_output, routing_weights, routing_indices, shared):
    """CPU/GPU diagnostic for the grouped_mm top-k-slot reduction boundary."""
    batch, topk, width = expert_output.shape
    if (routing_weights.shape != (batch, topk) or shared.shape != (batch, width)
            or routing_indices.shape[1] != batch):
        raise ValueError("Inconsistent routed expert result geometry")
    routing_cpu = routing_indices.detach().cpu()
    for row in range(batch):
        slots = [int(slot) - 1 for slot in routing_cpu[:, row] if slot > 0]
        if sorted(slots) != list(range(topk)):
            raise ValueError("Every row must route each top-k slot exactly once")
    weighted = expert_output * routing_weights.to(expert_output.dtype).unsqueeze(-1)
    return weighted.sum(dim=1) + shared


@torch.inference_mode()
def moe_forward(hidden, weights, *, top_k):
    if hidden.ndim != 2 or hidden.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 hidden states [tokens, hidden]")
    gate = weights["gate.weight"]
    w13 = weights["experts.gate_up_proj"]
    w2 = weights["experts.down_proj"]
    experts, width = gate.shape
    if (width != hidden.shape[1] or w13.shape[0] != experts or
            w13.shape[1] % 2 or w13.shape[2] != width or
            w2.shape != (experts, width, w13.shape[1] // 2) or
            not 1 <= top_k <= experts):
        raise ValueError("Inconsistent MoE expert geometry")
    logits = F.linear(hidden, gate)
    probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    values, indices = torch.topk(probs, top_k, dim=-1)
    values = (values / values.sum(dim=-1, keepdim=True)).to(hidden.dtype)
    outputs = torch.zeros((hidden.shape[0], top_k, hidden.shape[1]),
                          dtype=hidden.dtype, device=hidden.device)
    for expert in range(experts):
        token, slot = (indices == expert).nonzero(as_tuple=True)
        if not token.numel():
            continue
        gate_up = F.linear(hidden[token], w13[expert])
        gate_part, up_part = gate_up.chunk(2, dim=-1)
        down = F.linear(F.silu(gate_part) * up_part, w2[expert])
        outputs[token, slot] = down
    shared_gate_up = F.linear(hidden, weights["shared_expert.gate_proj.weight"])
    shared_up = F.linear(hidden, weights["shared_expert.up_proj.weight"])
    shared = F.linear(F.silu(shared_gate_up) * shared_up,
                      weights["shared_expert.down_proj.weight"])
    shared_scale = torch.sigmoid(F.linear(hidden, weights["shared_expert_gate.weight"]))
    routed = (outputs * values.unsqueeze(-1)).sum(dim=1)
    return routed + shared_scale * shared, indices, values
