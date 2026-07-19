"""PyTorch reference for fused softmax + gather."""

import torch
import torch.nn.functional as F


def softmax_gather_ref(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Fused softmax + gather: output[b] = softmax(logits[b])[token_ids[b]].

    Args:
        logits: [B, V] floating tensor (e.g., bfloat16).
        token_ids: [B, 1] integer tensor.

    Returns:
        [B, 1] float32 tensor of gathered probabilities.
    """
    probs = F.softmax(logits.to(torch.float32), dim=-1)
    return probs.gather(dim=-1, index=token_ids.long())
