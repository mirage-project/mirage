"""
PyTorch reference for the MLA MTP decode + reduce kernels (DeepSeek V3, B200).

Two functions mirroring the CUDA implementation in
``include/mirage/persistent_kernel/tasks/blackwell/mla_mtp_decode_sm100.cuh``:

- ``mla_mtp_decode_ref``: per-split partial attention (output_partial, output_lse).
- ``mla_mtp_reduce_ref``: LSE-weighted reduction across the split-K dimension.

Constants follow the kernel (DeepSeek V3 MLA): NUM_HEADS=128, D_K=576, D_V=512,
TILE_S=128. K = kv[..., :D_K] (full 576), V = kv[..., :D_V] (first 512).

Softmax scale: ``ss = (1/sqrt(192)) * mscale^2`` where
``mscale = 0.1 * log(40) + 1.0``  (matches task_register.cc).

Buffer layouts (per block, where ``block_linear = bi*num_head_groups*sk + gi*sk + si``):
  - output_partial[block_linear, d*128 + tid]   (bf16, [D_V, 128] per block)
  - output_lse    [block_linear, tid]           (fp32, [128] per block)
where ``tid = q_idx * hpb + h_local`` and ``hpb = NUM_HEADS / num_head_groups``.
"""

import math
import torch


# DeepSeek V3 MLA constants
NUM_HEADS = 128
D_K = 576
D_V = 512
TILE_S = 128


def _deepseek_softmax_scale() -> float:
    # Match task_register.cc: q_head_dim = 192 (NOT 576), YARN-adjusted mscale.
    mscale = 0.1 * 1.0 * math.log(40.0) + 1.0
    return (1.0 / math.sqrt(192.0)) * mscale * mscale


def mla_mtp_decode_ref(q, kv, batch_size, q_len, kv_len, num_head_groups,
                       num_splits, dtype=torch.bfloat16):
    """Reference for ``mla_mtp_decode_sm100_task_impl``.

    Args:
        q: bf16 [batch_size * q_len * NUM_HEADS, D_K]
        kv: bf16 [batch_size * kv_len, D_K]
        batch_size, q_len, kv_len: int
        num_head_groups: NUM_HEADS / hpb (heads-per-block)
        num_splits: split-K count (sk). Each split covers
            ceil(ceil(kv_len/TILE_S)/sk) tiles of TILE_S=128 KV tokens.

    Returns:
        output_partial: bf16 [batch_size*num_head_groups*num_splits, D_V*128]
        output_lse:     fp32 [batch_size*num_head_groups*num_splits, 128]

    The per-block layout encodes (q_idx, h_local) along the inner-128 dim as
    ``tid = q_idx * hpb + h_local``.
    """
    assert NUM_HEADS % num_head_groups == 0
    hpb = NUM_HEADS // num_head_groups
    ss = _deepseek_softmax_scale()

    # Reshape Q -> [B, Q, H, D_K]; KV -> [B, KL, D_K]
    Q = q.reshape(batch_size, q_len, NUM_HEADS, D_K).float()
    KV = kv.reshape(batch_size, kv_len, D_K).float()
    K = KV                       # full D_K
    V = KV[..., :D_V]            # first D_V

    kvt = (kv_len + TILE_S - 1) // TILE_S
    tps = (kvt + num_splits - 1) // num_splits

    out_part = torch.zeros(
        (batch_size * num_head_groups * num_splits, D_V * 128),
        dtype=dtype, device=q.device)
    out_lse = torch.zeros(
        (batch_size * num_head_groups * num_splits, 128),
        dtype=torch.float32, device=q.device)

    for bi in range(batch_size):
        for gi in range(num_head_groups):
            head_start = gi * hpb
            head_end = head_start + hpb
            # Per-head Q for this group: [q_len, hpb, D_K]
            q_g = Q[bi, :, head_start:head_end, :]               # [Q, hpb, D_K]
            for si in range(num_splits):
                t0 = si * tps
                t1 = min(t0 + tps, kvt)
                block_linear = bi * num_head_groups * num_splits + gi * num_splits + si
                if t0 >= t1:
                    # Inactive split: leave zeros, lse defaults to log(1e-30)+(-1e30) ≈ -1e30+huge negative.
                    # Reduce kernel weights are exp(la - lse_max); inactive contributes 0 because we
                    # write nothing here. Keep lse at 0; the reduce reference must mask too.
                    # In the kernel La is uninitialized for inactive splits, but reduce uses sk
                    # such that all referenced splits have valid La. To stay safe, fill with -inf.
                    out_lse[block_linear, :].fill_(float('-inf'))
                    continue

                kvs_lo = t0 * TILE_S
                kvs_hi = min(t1 * TILE_S, kv_len)
                # Slice KV for this split window
                K_sub = K[bi, kvs_lo:kvs_hi, :]                  # [S, D_K]
                V_sub = V[bi, kvs_lo:kvs_hi, :]                  # [S, D_V]
                S = K_sub.shape[0]

                # Per-query causal limit (matches kernel):
                #   causal_limit_q = kv_len if Q_LEN==1 else (kv_len - Q_LEN + q_idx + 1)
                # so absolute kv index k must satisfy k < causal_limit_q.
                # Convert to mask within [kvs_lo, kvs_hi).
                abs_idx = torch.arange(kvs_lo, kvs_hi, device=q.device)  # [S]
                if q_len == 1:
                    causal_limit = torch.full((q_len,), kv_len, device=q.device)
                else:
                    causal_limit = torch.tensor(
                        [kv_len - q_len + qi + 1 for qi in range(q_len)],
                        device=q.device)
                # mask[q_idx, k] = True if abs_idx[k] < causal_limit[q_idx]
                mask = abs_idx.unsqueeze(0) < causal_limit.unsqueeze(1)   # [Q, S]

                # QK^T (no scale yet) — match kernel which scales after MMA.
                # q_g: [Q, hpb, D_K], K_sub: [S, D_K] -> scores: [Q, hpb, S]
                scores = torch.einsum("qhd,sd->qhs", q_g, K_sub) * ss

                neg_inf = torch.tensor(-1e30, device=q.device, dtype=scores.dtype)
                scores = torch.where(mask.unsqueeze(1), scores, neg_inf)

                # Online softmax (numerically stable). Replicate kernel formula:
                # row_max = global max over scaled scores; row_sum = sum(exp(s - row_max));
                # out = sum(softmax * V) / row_sum;
                # lse_out = log(row_sum) + row_max  (matches kernel writing
                #   ``logf(fmaxf(row_sum, 1e-30)) + row_max``).
                row_max = scores.amax(dim=-1, keepdim=True)              # [Q, hpb, 1]
                # If a query has all -1e30 (entirely masked), softmax is undefined;
                # kernel handles this via row_sum<=0 → inv=0; we mirror by zero output
                # and lse = log(1e-30) + row_max (large negative).
                exp_scores = torch.exp(scores - row_max)
                row_sum = exp_scores.sum(dim=-1)                          # [Q, hpb]
                # partial output: weighted sum of V (sum, NOT softmax — kernel divides
                # only at epilogue using inv = 1/row_sum).
                # PV: [Q, hpb, S] x [S, D_V] -> [Q, hpb, D_V]
                pv = torch.einsum("qhs,sv->qhv", exp_scores, V_sub)
                # Normalize at epilogue.
                inv = torch.where(row_sum > 0, 1.0 / row_sum,
                                  torch.zeros_like(row_sum))
                out = pv * inv.unsqueeze(-1)                              # [Q, hpb, D_V]

                # LSE per (q, h)
                row_max_f = row_max.squeeze(-1)
                lse = torch.log(torch.clamp(row_sum, min=1e-30)) + row_max_f
                # Where row_sum == 0 (fully masked), kernel writes log(1e-30)+row_max
                # (row_max stays at -1e30). We replicate.
                fully_masked = row_sum <= 0
                if fully_masked.any():
                    lse = torch.where(fully_masked,
                                      torch.log(torch.tensor(1e-30, device=q.device))
                                      + row_max_f, lse)

                # Pack into block buffer with kernel's tid layout:
                #   tid = q_idx * hpb + h_local; out_partial[block, d*128 + tid] = out[q, h, d]
                # Build [D_V, 128] then flatten.
                blk = torch.zeros((D_V, 128), dtype=torch.float32, device=q.device)
                lse_blk = torch.full((128,), float('-inf'), dtype=torch.float32,
                                     device=q.device)
                for qi in range(q_len):
                    for hl in range(hpb):
                        tid = qi * hpb + hl
                        blk[:, tid] = out[qi, hl, :]
                        lse_blk[tid] = lse[qi, hl]
                out_part[block_linear].copy_(blk.reshape(-1).to(dtype))
                out_lse[block_linear].copy_(lse_blk)

    return out_part, out_lse


def mla_mtp_reduce_ref(output_partial, output_lse, batch_size, q_len,
                       num_head_groups, num_splits):
    """Reference for ``mla_mtp_reduce_sm100_task_impl``.

    Args:
        output_partial: bf16 [B*num_head_groups*sk, D_V*128]  (from decode)
        output_lse:     fp32 [B*num_head_groups*sk, 128]      (from decode)

    Returns:
        out: bf16 [batch_size, q_len, NUM_HEADS, D_V] — LSE-weighted sum across sk.
    """
    hpb = NUM_HEADS // num_head_groups
    sk = num_splits
    out = torch.zeros(
        (batch_size, q_len, NUM_HEADS, D_V),
        dtype=torch.bfloat16, device=output_partial.device)

    Op = output_partial.float().reshape(
        batch_size, num_head_groups, sk, D_V, 128)
    La = output_lse.reshape(batch_size, num_head_groups, sk, 128)

    for bi in range(batch_size):
        for gi in range(num_head_groups):
            for qi in range(q_len):
                for hl in range(hpb):
                    tid = qi * hpb + hl
                    h_global = gi * hpb + hl
                    # LSE merge across sk.
                    la_vec = La[bi, gi, :, tid]                 # [sk]
                    lse_max = la_vec.max()
                    sum_exp = torch.exp(la_vec - lse_max).sum()
                    inv_sum = 1.0 / sum_exp if sum_exp > 0 else torch.tensor(0.0)
                    weights = torch.exp(la_vec - lse_max) * inv_sum  # [sk]
                    # Weighted sum over splits.
                    # Op[bi,gi,:,:,tid] -> [sk, D_V]
                    pv = Op[bi, gi, :, :, tid]
                    acc = (weights.unsqueeze(-1) * pv).sum(dim=0)    # [D_V]
                    out[bi, qi, h_global, :] = acc.to(torch.bfloat16)
    return out
