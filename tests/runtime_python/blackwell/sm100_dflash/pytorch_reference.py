"""Canonical PyTorch references for Kimi-K2.6 DFlash draft ops.

These mirror the HF reference (`/raid/.../Kimi-K2.6-DFlash-tmp/dflash.py`) op-by-op so
both the test-mode tests and any kernel-wrapper tests align on a single source.

Config (K2.6 b8): H=7168, K=6, B=8, n_q=64, n_kv=8, d=128, I=18432, eps=1e-5, YaRN rope.
"""
import json
import os

import torch
from safetensors import safe_open

CKPT = "/raid/catalyst/models/Kimi-K2.6-DFlash-tmp"
EPS = 1e-5


# ---------------------------------------------------------------- weight loading
def load_weight(name, ckpt=CKPT, device="cuda", dtype=torch.bfloat16):
    idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    shard = idx[name]
    with safe_open(os.path.join(ckpt, shard), framework="pt") as f:
        return f.get_tensor(name).to(device=device, dtype=dtype)


# ---------------------------------------------------------------- elementwise ops
def rms_norm(x, weight, eps=EPS):
    """RMSNorm with fp32 internal accumulation (matches Qwen3RMSNorm)."""
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    out = x32 * torch.rsqrt(var + eps)
    return (out * weight.to(torch.float32)).to(x.dtype)


def linear(x, w):
    """y = x @ w.T  (w is [out, in], no bias)."""
    return torch.matmul(x, w.transpose(0, 1))


def silu_mul(gate, up):
    return torch.nn.functional.silu(gate.to(torch.float32)).to(gate.dtype) * up


# ---------------------------------------------------------------- composite layers
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def dflash_norm_rope(x, weight, cos, sin, eps=EPS):
    """Per-head RMSNorm + NeoX RoPE. x:[N,H,d] weight:[d] cos/sin:[N,d] -> [N,H,d]."""
    N, H, d = x.shape
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    nv = xf * torch.rsqrt(var + eps) * weight.to(torch.float32)   # [N,H,d]
    c = cos.to(torch.float32)[:, None, :]                          # [N,1,d]
    s = sin.to(torch.float32)[:, None, :]
    out = nv * c + rotate_half(nv) * s
    return out.to(x.dtype)


def dflash_attention_core(q, k, v, sliding_window, n_q, n_kv, d):
    """Attention CORE only (matches the dflash_attention_sm100 kernel scope).

    q: [B, n_q, d] (q_norm + RoPE already applied)
    k: [T, n_kv, d] (k_norm + RoPE already applied; ctx rows first, then block)
    v: [T, n_kv, d] (raw)
    Returns [B, n_q, d]. Non-causal; key j position = j, query i position = ctx_len+i.
    """
    B = q.shape[0]
    T = k.shape[0]
    ctx_len = T - B
    g = n_q // n_kv
    qf = q.to(torch.float32)                                  # [B,n_q,d]
    kf = k.to(torch.float32).repeat_interleave(g, dim=1)      # [T,n_q,d]
    vf = v.to(torch.float32).repeat_interleave(g, dim=1)
    scale = d ** -0.5
    # scores [n_q, B, T]
    scores = torch.einsum("bhd,thd->hbt", qf, kf) * scale
    if sliding_window and sliding_window > 0:
        qpos = torch.arange(B, device=q.device) + ctx_len
        kpos = torch.arange(T, device=q.device)
        blocked = (qpos[:, None] - kpos[None, :]).abs() >= sliding_window
        scores = scores.masked_fill(blocked[None], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("hbt,thd->bhd", attn, vf)             # [B,n_q,d]
    return out.to(q.dtype)


def dflash_attention(ctx, h, q_w, k_w, v_w, o_w, q_norm_w, k_norm_w,
                     cos, sin, sliding_window, n_q, n_kv, d, eps=EPS):
    """Non-causal block attention, one draft layer (mirrors dflash.py exactly).

    ctx: [ctx_len, H] (hidden_norm(fc(target_hidden)))  -- context tokens
    h:   [B, H]       (input_layernorm output)          -- block (query) tokens
    cos/sin: [ctx_len+B, d]  positions of context++block
    Returns attn_output [B, H] (after o_proj, before residual add).
    """
    ctx_len, H = ctx.shape
    B = h.shape[0]
    # add batch dim
    ctxb = ctx.unsqueeze(0)
    hb = h.unsqueeze(0)
    cos_b = cos.unsqueeze(0).unsqueeze(1)  # [1,1,T,d]
    sin_b = sin.unsqueeze(0).unsqueeze(1)

    q = linear(hb, q_w).view(1, B, n_q, d)
    q = rms_norm(q, q_norm_w, eps).transpose(1, 2)               # [1,n_q,B,d]
    k_ctx = linear(ctxb, k_w)
    k_noise = linear(hb, k_w)
    v_ctx = linear(ctxb, v_w)
    v_noise = linear(hb, v_w)
    k = torch.cat([k_ctx, k_noise], dim=1).view(1, ctx_len + B, n_kv, d)
    v = torch.cat([v_ctx, v_noise], dim=1).view(1, ctx_len + B, n_kv, d)
    k = rms_norm(k, k_norm_w, eps).transpose(1, 2)               # [1,n_kv,T,d]
    v = v.transpose(1, 2)

    # rope: q uses last B positions, k uses all T
    qf, kf = q.to(torch.float32), k.to(torch.float32)
    cf, sf = cos_b.to(torch.float32), sin_b.to(torch.float32)
    q = (qf * cf[..., -B:, :]) + (rotate_half(qf) * sf[..., -B:, :])
    k = (kf * cf) + (rotate_half(kf) * sf)

    # GQA expand
    g = n_q // n_kv
    kx = k.repeat_interleave(g, dim=1)                            # [1,n_q,T,d]
    vx = v.repeat_interleave(g, dim=1).to(torch.float32)
    scale = d ** -0.5
    scores = torch.matmul(q, kx.transpose(-1, -2)) * scale       # [1,n_q,B,T]
    T = ctx_len + B
    if sliding_window is not None:
        qpos = torch.arange(B, device=h.device) + ctx_len
        kpos = torch.arange(T, device=h.device)
        blocked = (qpos[:, None] - kpos[None, :]).abs() >= sliding_window
        scores = scores.masked_fill(blocked[None, None], float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, vx)                                  # [1,n_q,B,d]
    out = out.transpose(1, 2).reshape(B, n_q * d).to(h.dtype)
    return linear(out, o_w)


def fc_hidden_norm(target_hidden, fc_w, hidden_norm_w, eps=EPS):
    """ctx = hidden_norm(fc(target_hidden_concat)).

    target_hidden: [S, K*H]  fc_w: [H, K*H]  hidden_norm_w: [H]  -> [S, H]
    Returns (fc_out, ctx) so callers can check both stages.
    """
    fc_out = linear(target_hidden, fc_w)
    ctx = rms_norm(fc_out, hidden_norm_w, eps)
    return fc_out, ctx
