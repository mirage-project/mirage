"""PyTorch reference for the DFlash non-causal block attention kernel
(dflash_attention_sm100, TASK_DFLASH_ATTENTION_SM100).

Semantics (see include/mirage/persistent_kernel/tasks/blackwell/
dflash_attention_sm100.cuh): the B block-query tokens attend NON-causally to
[context (ctx_len) ++ block (B)] keys. Absolute positions: context key j -> j,
block key/query i -> ctx_len + i. sliding_window > 0 keeps only keys with
|q_pos - key_pos| < sliding_window. Scores are scaled by 1/sqrt(head_dim).
"""

import torch


def dflash_attention_ref(
    q, ctx_k, ctx_v, blk_k, blk_v, sliding_window, head_dim
):
    """q [B, NQ*D]; ctx_k/ctx_v [ctx_len, NKV*D]; blk_k/blk_v [B, NKV*D].
    Returns out [B, NQ*D] in q.dtype."""
    d_head = head_dim
    b = q.shape[0]
    nq = q.shape[1] // d_head
    nkv = ctx_k.shape[1] // d_head
    group = nq // nkv
    ctx_len = ctx_k.shape[0]
    t_kv = ctx_len + b

    keys = torch.cat([ctx_k, blk_k], dim=0).float().view(t_kv, nkv, d_head)
    vals = torch.cat([ctx_v, blk_v], dim=0).float().view(t_kv, nkv, d_head)
    qf = q.float().view(b, nq, d_head)

    key_pos = torch.arange(t_kv, device=q.device)
    out = torch.empty(b, nq, d_head, device=q.device, dtype=torch.float32)
    for i in range(b):
        q_pos = ctx_len + i
        for h in range(nq):
            g = h // group
            scores = keys[:, g] @ qf[i, h] / (d_head**0.5)
            if sliding_window > 0:
                mask = (q_pos - key_pos).abs() >= sliding_window
                scores = scores.masked_fill(mask, float("-inf"))
            probs = torch.softmax(scores, dim=0)
            out[i, h] = probs @ vals[:, g]
    return out.view(b, nq * d_head).to(q.dtype)
