"""Small PyTorch references for Qwen3.8-27B numerical checks.

Inputs are one token at a time. Recurrent states use [batch, value head, 128, 128].
Functions return new states so tests can compare every intermediate step.
"""

import torch
import torch.nn.functional as F


def rms_norm(x, weight, eps=1e-6, *, unit_offset=True):
    """Decoder/QK norms use (1 + weight); the GDN gated norm does not."""
    xf = x.float()
    scale = weight.float() + (1.0 if unit_offset else 0.0)
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * scale).to(x.dtype)


def gated_rms_norm(x, weight, gate, eps=1e-6):
    xf = x.float()
    # Match HF's cast before applying the norm weight and SiLU gate.
    normalized = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype)
    return (normalized * weight * F.silu(gate.float())).to(x.dtype)


def causal_conv_step(x, history, weight):
    """Depthwise causal convolution + SiLU; history is [batch, channels, K-1]."""
    if weight.ndim != 3 or weight.shape[1] != 1:
        raise ValueError("Conv weights must have shape [channels, 1, kernel_size]")
    if x.ndim != 2 or history.shape != (*x.shape, weight.shape[-1] - 1):
        raise ValueError("Conv history shape does not match input and kernel")
    window = torch.cat((history, x.unsqueeze(-1)), dim=-1)
    result = F.conv1d(window, weight, groups=x.shape[1]).squeeze(-1)
    return F.silu(result), window[..., 1:].contiguous()


def gated_delta_step(query, key, value, decay, beta, state):
    """Delta-rule update with normalized Q/K and FP32 recurrence.

    Q/K: [batch, key_heads, key_dim], V: [batch, value_heads, value_dim].
    decay is log decay (<= 0), beta is the sigmoid gate, both [batch, value_heads].
    State: [batch, value_heads, key_dim, value_dim].
    """
    if state.dtype != torch.float32:
        raise ValueError("GDN recurrent state must be float32")
    if query.shape != key.shape or query.ndim != 3 or value.ndim != 3:
        raise ValueError("Invalid Q/K/V shapes")
    batch, key_heads, key_dim = key.shape
    if value.shape[0] != batch or value.shape[1] % key_heads:
        raise ValueError("Value heads must be divisible by key heads")
    value_heads, value_dim = value.shape[1:]
    if state.shape != (batch, value_heads, key_dim, value_dim):
        raise ValueError("Invalid recurrent state shape")
    if decay.shape != (batch, value_heads) or beta.shape != decay.shape:
        raise ValueError("Invalid gate shape")
    # Transformers 5.14.1 normalizes in the input dtype *before* widening.
    # Doing this in FP32 instead changes BF16 decode results.
    q = (query * torch.rsqrt((query * query).sum(-1, keepdim=True) + 1e-6)).float()
    k = (key * torch.rsqrt((key * key).sum(-1, keepdim=True) + 1e-6)).float()
    v = value.float()
    repeats = value_heads // key_heads
    q = q.repeat_interleave(repeats, dim=1) * (key_dim ** -0.5)
    k = k.repeat_interleave(repeats, dim=1)
    decayed = state * decay.float().exp()[..., None, None]
    prediction = (decayed * k[..., None]).sum(-2)
    correction = (v - prediction) * beta.float()[..., None]
    updated = decayed + k[..., None] * correction[..., None, :]
    output = (updated * q[..., None]).sum(-2)
    return output.to(value.dtype), updated


def delta_gates(a, b, a_log, dt_bias):
    return -a_log.float().exp() * F.softplus(a.float() + dt_bias.float()), b.sigmoid()
