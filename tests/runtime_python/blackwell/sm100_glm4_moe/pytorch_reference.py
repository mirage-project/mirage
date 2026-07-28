"""PyTorch references for the GLM-4.6 MPK kernels.

Math follows transformers modeling_glm4_moe.py (zai-org/GLM-4.6):
Glm4MoeTopkRouter with n_group == 1, and Glm4MoeAttention (per-head
q/k RMSNorm eps 1e-5 before RoPE, partial rotary over the first
rotary_dim dims, causal, 1/sqrt(head_dim) scaling).
"""

import torch


def glm_moe_router_ref(
    logits, bias, topk=8, n_shared=1, routed_scaling_factor=2.5
):
    """logits [rows, stride] bf16 (first num_routed cols used),
    bias [num_routed] fp32.

    Returns (weights [rows, topk+n_shared] fp32, sel [rows, topk+n_shared]),
    with the shared experts appended as ids num_routed.. at weight 1.0.
    """
    num_routed = bias.shape[0]
    l = logits[:, :num_routed].float()
    rows = l.shape[0]
    scores = torch.sigmoid(l)
    choice = scores + bias.float()
    top_idx = choice.topk(topk, dim=-1).indices          # [rows, topk]
    w = scores.gather(1, top_idx)                        # UNBIASED scores
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)        # norm_topk_prob
    w = w * routed_scaling_factor
    shared_ids = (
        torch.arange(num_routed, num_routed + n_shared, device=l.device)
        .unsqueeze(0).expand(rows, n_shared)
    )
    sel = torch.cat([top_idx, shared_ids], dim=1)
    w = torch.cat(
        [w, torch.ones(rows, n_shared, device=l.device, dtype=w.dtype)], dim=1
    )
    return w, sel


def _rms_norm(x, weight, eps):
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(var + eps) * weight.float()


def _partial_rope(x, cos, sin, rotary_dim):
    """x [..., head_dim]; cos/sin [..., rotary_dim] broadcastable.
    Rotates x[..., :rotary_dim] NeoX-style (pairs i <-> i+rotary_dim/2),
    passes the remaining dims through."""
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    half = rotary_dim // 2
    x1, x2 = x_rot[..., :half], x_rot[..., half:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return torch.cat([x_rot * cos + rotated * sin, x_pass], dim=-1)


def glm_attention_prefill_ref(
    q, k, v, q_norm_w, k_norm_w, cos, sin, rotary_dim, eps=1e-5
):
    """Causal GQA prefill attention for one request.

    q [T, NQ, D], k/v [T, NKV, D] (post-projection, pre-norm/rope);
    q_norm_w/k_norm_w [D]; cos/sin [T, rotary_dim] for positions 0..T-1.
    Head grouping: q head h belongs to kv head h // (NQ // NKV).
    Returns out [T, NQ, D] float32.
    """
    t, nq, d = q.shape
    nkv = k.shape[1]
    group = nq // nkv

    qn = _rms_norm(q, q_norm_w, eps)                 # [T, NQ, D] fp32
    kn = _rms_norm(k, k_norm_w, eps)                 # [T, NKV, D] fp32
    # bf16 round-trip between norm and rope, matching the smem layout
    qn = qn.to(q.dtype).float()
    kn = kn.to(k.dtype).float()
    cos_e = cos.float().unsqueeze(1)                 # [T, 1, rotary]
    sin_e = sin.float().unsqueeze(1)
    qr = _partial_rope(qn, cos_e, sin_e, rotary_dim)
    kr = _partial_rope(kn, cos_e, sin_e, rotary_dim)
    vf = v.float()

    scale = d ** -0.5
    causal = torch.tril(torch.ones(t, t, dtype=torch.bool, device=q.device))
    out = torch.empty(t, nq, d, device=q.device, dtype=torch.float32)
    for h in range(nq):
        g = h // group
        s = (qr[:, h] @ kr[:, g].T) * scale          # [T, T]
        s = s.masked_fill(~causal, float("-inf"))
        out[:, h] = torch.softmax(s, dim=-1) @ vf[:, g]
    return out
