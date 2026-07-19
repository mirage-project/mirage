"""PyTorch reference implementations for the Inkling SM100 MPK kernels.

One function per in-scope layer (see tests/runtime_python/<arch>/README in the
test-mode skill): both the test-mode tests and any future kernel-wrapper tests
import from this file so they share a single canonical reference.

Semantics follow the HF reference (thinkingmachines/Inkling,
transformers modular_inkling.py) as documented in the kernel headers under
include/mirage/persistent_kernel/tasks/blackwell/inkling_*.cuh.
"""

import math

import torch
import torch.nn.functional as F


def inkling_sconv_ref(x, weight, conv_state):
    """Depthwise short convolution + residual, fp32 math.

    x:          [SEQ, H] bf16 input activations (tokens in order)
    weight:     [H, K] fp32 depthwise taps, weight[:, 0] = oldest tap
    conv_state: [K-1, H] fp32 previous K-1 token activations, row 0 = oldest

    Returns (out [SEQ, H] in x.dtype, new_state [K-1, H] fp32).
    out[t, c] = x[t, c] + sum_i weight[c, i] * win[i]  with the causal window
    win = [state..., x[<=t]] (oldest first).
    """
    xf = x.float()
    seq = xf.shape[0]
    k = weight.shape[1]
    win = torch.cat([conv_state.float(), xf], dim=0)  # [K-1+SEQ, H]
    out = torch.empty_like(xf)
    for t in range(seq):
        taps = win[t : t + k]  # [K, H], oldest first
        out[t] = xf[t] + (weight.t().float() * taps).sum(dim=0)
    return out.to(x.dtype), win[seq:].clone()


def inkling_moe_router_ref(
    logits, bias, global_scale, topk=6, n_shared=2, route_scale=8.0
):
    """Sigmoid top-k router with logsigmoid-softmax weights and folded
    shared experts.

    logits:       [rows, stride] bf16, first num_routed+n_shared columns used
    bias:         [num_routed] fp32 e_score_correction_bias
    global_scale: [1] fp32

    Returns (weights [rows, topk+n_shared] fp32,
             sel     [rows, topk+n_shared] int64 selected expert ids,
             routed top-k first in descending selection-score order, then the
             shared experts num_routed..num_routed+n_shared-1).
    """
    num_routed = bias.shape[0]
    num_total = num_routed + n_shared
    l = logits[:, :num_total].float()
    rows = l.shape[0]
    choice = torch.sigmoid(l[:, :num_routed]) + bias.float()
    top_idx = choice.topk(topk, dim=-1).indices  # [rows, topk]
    shared = (
        torch.arange(num_routed, num_total, device=l.device)
        .unsqueeze(0)
        .expand(rows, n_shared)
    )
    sel = torch.cat([top_idx, shared], dim=1)  # [rows, topk+n_shared]
    lp = F.logsigmoid(torch.gather(l, 1, sel))
    weights = torch.softmax(lp, dim=-1) * route_scale * global_scale.float()
    return weights, sel


def inkling_attention_ref(
    q,
    ctx_k,
    ctx_v,
    blk_k,
    blk_v,
    bias,
    ctx_len,
    head_dim,
    sliding_window,
    extent,
    log_scaling_alpha,
    log_scaling_n_floor,
):
    """GQA decode attention with relative-position bias for the single new
    token at position P = ctx_len.

    q:            [1, NQ*D] bf16 (per-head q_norm applied upstream)
    ctx_k, ctx_v: [MAX_CTX, NKV*D] bf16 (rows < ctx_len are valid)
    blk_k, blk_v: [1, NKV*D] bf16 (this step's key/value)
    bias:         [NQ, extent] bf16 per-(head, distance) bias table

    s_j = tau * (dot(q_h, k_j) / D + bias[h, P-j])  with bias 0 for
    P-j >= extent; sliding_window > 0 keeps only 0 <= P-j < sliding_window;
    tau = 1 + alpha * ln(max((P+1)/n_floor, 1)).
    Returns out [1, NQ*D] in q.dtype.
    """
    d_head = head_dim
    nq = q.shape[1] // d_head
    nkv = ctx_k.shape[1] // d_head
    group = nq // nkv
    p_pos = ctx_len
    keys = torch.cat([ctx_k[:p_pos], blk_k], dim=0).float()
    keys = keys.view(p_pos + 1, nkv, d_head)
    vals = torch.cat([ctx_v[:p_pos], blk_v], dim=0).float()
    vals = vals.view(p_pos + 1, nkv, d_head)
    qf = q.float().view(nq, d_head)
    tau = 1.0
    if log_scaling_alpha != 0.0:
        tau = 1.0 + log_scaling_alpha * math.log(
            max(float(p_pos + 1) / float(log_scaling_n_floor), 1.0)
        )
    dist = p_pos - torch.arange(p_pos + 1, device=q.device)  # [P+1]
    bias_f = bias.float()
    out = torch.empty(nq, d_head, device=q.device, dtype=torch.float32)
    for h in range(nq):
        g = h // group
        scores = keys[:, g] @ qf[h] / d_head  # [P+1]
        b = torch.zeros(p_pos + 1, device=q.device, dtype=torch.float32)
        in_ext = dist < extent
        b[in_ext] = bias_f[h, dist[in_ext]]
        s = tau * (scores + b)
        if sliding_window > 0:
            s = s.masked_fill(dist >= sliding_window, float("-inf"))
        probs = torch.softmax(s, dim=0)
        out[h] = probs @ vals[:, g]
    return out.view(1, nq * d_head).to(q.dtype)
