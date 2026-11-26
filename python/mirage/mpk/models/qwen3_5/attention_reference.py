"""PyTorch reference for dense Qwen3.8-27B text attention.

Accepts one request and any contiguous chunk within the 1024-token adapter
limit. KV tensors use [batch, sequence, KV head, head dimension] layout;
only the valid prefix is passed in. Returned caches are new tensors, so a
failed operation never partially advances caller-owned state.
"""

import torch
import torch.nn.functional as F

from .reference import rms_norm


def text_rope(config, positions, *, dtype=torch.bfloat16):
    """Text has identical temporal/height/width positions in interleaved MRoPE."""
    if positions.ndim != 1 or positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("Text positions must be a one-dimensional integer tensor")
    if (positions < 0).any() or (positions >= config.max_position_embeddings).any():
        raise ValueError("Text position is outside the configured context")
    width = config.rotary_dim
    frequency = 1.0 / (config.rope_theta ** (
        torch.arange(0, width, 2, dtype=torch.float32, device=positions.device) / width
    ))
    angle = positions.float()[:, None] * frequency[None, :]
    angle = torch.cat((angle, angle), dim=-1).unsqueeze(0)
    return angle.cos().to(dtype), angle.sin().to(dtype)


def partial_rope(query, key, cos, sin):
    """Rotate only the configured prefix of [batch, head, sequence, dim]."""
    width = cos.shape[-1]
    if width <= 0 or width % 2 or width > min(query.shape[-1], key.shape[-1]):
        raise ValueError("Invalid partial rotary width")
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)

    def rotate(x):
        first, second = x[..., :width].chunk(2, dim=-1)
        rotated_half = torch.cat((-second, first), dim=-1)
        # Keep separate BF16 products and addition, matching HF's cast order.
        prefix = x[..., :width] * cos + rotated_half * sin
        return torch.cat((prefix, x[..., width:]), dim=-1)

    return rotate(query), rotate(key)


def full_attention_forward(hidden, weights, config, key_cache=None, value_cache=None):
    """Project already-normalized hidden states through one full-attention layer.

    weights has HF attention-local keys (q_proj.weight, q_norm.weight, etc.).
    The caller owns input RMSNorm, decoder residual/MLP and model-level position
    advancement. Passing no cache starts a fresh request at text position zero.
    Returns intermediates for future GPU comparison, including new KV prefixes.
    """
    c = config
    if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[2] != c.hidden_size:
        raise ValueError("Expected hidden states [1, sequence, hidden_size]")
    if hidden.dtype != torch.bfloat16 or hidden.shape[1] < 1:
        raise ValueError("Expected at least one BF16 token")
    if (key_cache is None) != (value_cache is None):
        raise ValueError("Key and value caches must be supplied together")
    past = 0
    if key_cache is not None:
        if (key_cache.ndim != 4 or value_cache.shape != key_cache.shape or
            key_cache.shape[0] != 1 or
            key_cache.shape[2:] != (c.num_key_value_heads, c.head_dim)):
            raise ValueError("Expected matching KV prefixes [1, past, kv_heads, head_dim]")
        if any(t.dtype != hidden.dtype or t.device != hidden.device for t in (key_cache, value_cache)):
            raise ValueError("KV dtype/device must match hidden states")
        past = key_cache.shape[1]
    length = hidden.shape[1]
    if past + length > min(1024, c.max_position_embeddings):
        raise ValueError("Sequence capacity exceeded; start a new request")

    qg = F.linear(hidden, weights["q_proj.weight"])
    # Q and gate are paired *within each head*, not two contiguous half-matrices.
    query, gate = qg.reshape(1, length, c.num_attention_heads, 2 * c.head_dim).chunk(2, -1)
    query = rms_norm(query, weights["q_norm.weight"], c.rms_norm_eps).transpose(1, 2)
    key = F.linear(hidden, weights["k_proj.weight"]).reshape(1, length, c.num_key_value_heads, c.head_dim)
    key = rms_norm(key, weights["k_norm.weight"], c.rms_norm_eps).transpose(1, 2)
    value = F.linear(hidden, weights["v_proj.weight"]).reshape(1, length, c.num_key_value_heads, c.head_dim)
    positions = torch.arange(past, past + length, device=hidden.device)
    cos, sin = text_rope(c, positions, dtype=hidden.dtype)
    query, key = partial_rope(query, key, cos, sin)
    key = key.transpose(1, 2)
    if key_cache is not None:
        key = torch.cat((key_cache, key), dim=1)
        value = torch.cat((value_cache, value), dim=1)
    repeats = c.num_attention_heads // c.num_key_value_heads
    grouped_key = key.transpose(1, 2).repeat_interleave(repeats, dim=1)
    grouped_value = value.transpose(1, 2).repeat_interleave(repeats, dim=1)

    # Eager HF rounds the QK product and scaling to BF16 before FP32 softmax.
    scores = (query @ grouped_key.transpose(-1, -2)) * (c.head_dim ** -0.5)
    causal = torch.arange(past + length, device=hidden.device)[None, :] > positions[:, None]
    mask = torch.zeros(length, past + length, dtype=hidden.dtype, device=hidden.device)
    mask.masked_fill_(causal, torch.finfo(hidden.dtype).min)
    probabilities = F.softmax(scores + mask, dim=-1, dtype=torch.float32).to(hidden.dtype)
    attended = (probabilities @ grouped_value).transpose(1, 2).contiguous().reshape(1, length, -1)
    gate = gate.reshape(1, length, -1)
    gated = attended * gate.sigmoid()
    output = F.linear(gated, weights["o_proj.weight"])
    return {"output": output, "key": key, "value": value, "query": query,
            "gate": gate, "gated": gated, "attention_weights": probabilities,
            "cos": cos, "sin": sin}
