"""
PyTorch reference for the DeepSeek-V3 TP=4 decode-attention block (the part
*between* the two per-layer AllReduces — i.e. the per-rank attention compute,
which has no NVSHMEM, so it is testable standalone on ONE GPU).

The TP=4 MLA decode kernel processes NUM_HEADS=32 (= 128 / TP4) per rank. It is
the *absorbed* decode form: Q is 576-d ``[nope_abs(512) | pe(64)]``, the KV cache
row is 576-d ``[ckv(512) | kpe(64)]``, and V = the first D_V=512 of the KV row
(the absorbed ckv). Scores use the DeepSeek YARN-adjusted scale
``ss = (1/sqrt(192)) * mscale^2``, mscale = 0.1*log(40)+1 (matches
``task_register.cc`` and ``mla_mtp_decode_tp4_sm100.cuh``).

We provide a DIRECT full-attention reference (no split-K), and compare it to the
kernel's FINAL reduced output ``attn_out`` — this is layout-agnostic, so it does
not depend on the kernel's internal v_split / num_split / head_group packing.
"""

import math
import torch

# DeepSeek-V3 MLA constants, TP=4 per-rank.
NUM_HEADS = 32            # 128 / TP4
D_K = 576                # absorbed Q/K dim: [nope_abs(512) | pe(64)]
D_V = 512                # absorbed V dim (ckv / kv_lora_rank)


def deepseek_softmax_scale() -> float:
    # Matches task_register.cc / mla kernel: q_head_dim=192 (NOT 576), YARN mscale.
    mscale = 0.1 * 1.0 * math.log(40.0) + 1.0
    return (1.0 / math.sqrt(192.0)) * mscale * mscale


def mla_decode_full_ref(q, kv, batch_size, q_len, kv_len):
    """Direct (non-split-K) MLA decode attention, NUM_HEADS=32.

    Args:
        q:  bf16 [batch_size * q_len * NUM_HEADS, D_K]
        kv: bf16 [batch_size * kv_len, D_K]
    Returns:
        out: bf16 [batch_size, q_len, NUM_HEADS, D_V]  (the true attention output)

    Causal rule mirrors the kernel: for q_len==1 (pure decode) the single query
    attends to all kv_len positions; for q_len>1, query q_idx attends to absolute
    kv index k < (kv_len - q_len + q_idx + 1).
    """
    ss = deepseek_softmax_scale()
    Q = q.reshape(batch_size, q_len, NUM_HEADS, D_K).float()
    KV = kv.reshape(batch_size, kv_len, D_K).float()
    K = KV                       # [B, kv_len, D_K]  (full 576)
    V = KV[..., :D_V]            # [B, kv_len, D_V]  (first 512)

    out = torch.zeros(
        (batch_size, q_len, NUM_HEADS, D_V),
        dtype=torch.bfloat16, device=q.device)

    abs_idx = torch.arange(kv_len, device=q.device)
    for bi in range(batch_size):
        for qi in range(q_len):
            if q_len == 1:
                limit = kv_len
            else:
                limit = kv_len - q_len + qi + 1
            mask = abs_idx < limit                              # [kv_len]
            # scores: [H, kv_len]
            scores = torch.einsum("hd,kd->hk", Q[bi, qi], K[bi]) * ss
            scores = torch.where(
                mask.unsqueeze(0), scores,
                torch.tensor(-1e30, device=q.device, dtype=scores.dtype))
            sm = torch.softmax(scores, dim=-1)                  # [H, kv_len]
            o = torch.einsum("hk,kv->hv", sm, V[bi])            # [H, D_V]
            out[bi, qi] = o.to(torch.bfloat16)
    return out
