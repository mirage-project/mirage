#!/usr/bin/env python3
"""PyTorch reference for MLA chunked prefill attention.

Math:
    QK = Q_nope @ K_nope^T + Q_rope @ K_rope^T  (per head)
    softmax(QK * scale, dim=kv) with causal mask
    O = softmax @ V  (per head)

Usage:
    python scripts/dpskv3_pyref_chunked_prefill.py validate \\
        --ref-dir outputs/dpskv3_ref_dump_intra_v2_<ts>
"""
import argparse
import math

import torch
import torch.nn.functional as F


def chunked_prefill_attention_ref(
    q_full: torch.Tensor,    # [q_len, H, 192]
    k_nope: torch.Tensor,    # [kv_len, H, 128]
    k_pe: torch.Tensor,      # [kv_len, 1, 64]
    v: torch.Tensor,         # [kv_len, H, 128]
    softmax_scale: float = None,
    q_start: int = 0,
):
    """Returns: [q_len, H * 128] bf16."""
    q_len, H, qkd = q_full.shape
    assert qkd == 192
    kv_len = v.shape[0]
    assert v.shape == (kv_len, H, 128)
    assert k_nope.shape == (kv_len, H, 128)
    assert k_pe.shape == (kv_len, 1, 64)

    if softmax_scale is None:
        # YaRN-aware default not applied here; use simple 1/sqrt(d).
        softmax_scale = 1.0 / math.sqrt(qkd)

    # Expand k_pe to per-head (concat with k_nope to form 192-dim K)
    k_pe_expanded = k_pe.expand(-1, H, -1)  # [kv_len, H, 64]
    k_full = torch.cat([k_nope, k_pe_expanded], dim=-1)  # [kv_len, H, 192]

    # Compute QK with float32 precision
    q_f = q_full.float()
    k_f = k_full.float()
    v_f = v.float()

    # [H, q_len, 192] @ [H, kv_len, 192]^T → [H, q_len, kv_len]
    q_t = q_f.permute(1, 0, 2)
    k_t = k_f.permute(1, 0, 2)
    v_t = v_f.permute(1, 0, 2)  # [H, kv_len, 128]

    qk = torch.einsum('hqd,hkd->hqk', q_t, k_t) * softmax_scale  # [H, q_len, kv_len]

    # Causal mask: kvp <= q_start + qi
    q_idx = torch.arange(q_len, device=qk.device) + q_start
    k_idx = torch.arange(kv_len, device=qk.device)
    mask = k_idx[None, :] <= q_idx[:, None]  # [q_len, kv_len]
    qk = qk.masked_fill(~mask[None, :, :], float('-inf'))

    attn_weights = F.softmax(qk, dim=-1)
    out = torch.einsum('hqk,hkd->hqd', attn_weights, v_t)  # [H, q_len, 128]
    out = out.permute(1, 0, 2).reshape(q_len, H * 128)  # [q_len, H*128]
    return out.to(torch.bfloat16)


def chunked_prefill_via_sdpa(
    q_full: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float = None,
):
    """Match the reference impl in modeling.py (uses SDPA + V padding)."""
    q_len, H, qkd = q_full.shape
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(qkd)

    k_pe_expanded = k_pe.expand(-1, H, -1)
    k = torch.cat([k_nope, k_pe_expanded], dim=-1)  # [kv_len, H, 192]
    v_padded = F.pad(v, (0, qkd - v.shape[-1]))  # pad to 192

    q_t = q_full.permute(1, 0, 2)
    k_t = k.permute(1, 0, 2)
    v_t = v_padded.permute(1, 0, 2)

    attn = F.scaled_dot_product_attention(
        q_t.float(), k_t.float(), v_t.float(),
        is_causal=True,
        scale=softmax_scale,
    )
    attn = attn.permute(1, 0, 2)  # [q_len, H, 192]
    attn = attn[..., :128]  # crop to V_HEAD_DIM
    return attn.reshape(q_len, H * 128).to(torch.bfloat16)


def validate(args):
    ref_dir = args.ref_dir
    q_full = torch.load(f"{ref_dir}/ref_layer0_intra_q_full_post_rope.pt",
                        map_location='cpu', weights_only=True)
    k_nope = torch.load(f"{ref_dir}/ref_layer0_intra_k_nope.pt",
                        map_location='cpu', weights_only=True)
    k_pe = torch.load(f"{ref_dir}/ref_layer0_intra_k_pe_post_rope.pt",
                      map_location='cpu', weights_only=True)
    v = torch.load(f"{ref_dir}/ref_layer0_intra_v.pt",
                   map_location='cpu', weights_only=True)
    ref_out = torch.load(f"{ref_dir}/ref_layer0_intra_attn_unabsorbed.pt",
                         map_location='cpu', weights_only=True)

    print(f"Inputs:")
    print(f"  q_full:   {tuple(q_full.shape)}")
    print(f"  k_nope:   {tuple(k_nope.shape)}")
    print(f"  k_pe:     {tuple(k_pe.shape)}")
    print(f"  v:        {tuple(v.shape)}")
    print(f"  ref_out:  {tuple(ref_out.shape)}")

    # Note: DSv3 uses YaRN-mscale^2 softmax scale (not pure 1/sqrt(d)).
    # We'll try both and see which matches.
    for label, scale in [
        ("1/sqrt(192)", 1.0 / math.sqrt(192)),
        ("YaRN mscale^2=1.27**2/sqrt(192)", (1.27 ** 2) / math.sqrt(192)),
    ]:
        print(f"\n=== Trying softmax_scale = {label} ===")
        py_out_sdpa = chunked_prefill_via_sdpa(q_full, k_nope, k_pe, v,
                                                softmax_scale=scale)
        py_out_einsum = chunked_prefill_attention_ref(q_full, k_nope, k_pe, v,
                                                       softmax_scale=scale)

        for name, out in [("SDPA", py_out_sdpa), ("einsum", py_out_einsum)]:
            cos = F.cosine_similarity(out.float(), ref_out.float(), dim=-1)
            diff = (out.float() - ref_out.float()).norm(dim=-1)
            l2_o = out.float().norm(dim=-1)
            l2_r = ref_out.float().norm(dim=-1)
            print(f"  {name}: mean_cos={cos.mean():.6f}  min_cos={cos.min():.6f}  "
                  f"max_diff={diff.max():.6f}  l2_ratio={(l2_o/l2_r.clamp(min=1e-6)).mean():.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['validate'])
    p.add_argument('--ref-dir', required=True)
    args = p.parse_args()
    if args.mode == 'validate':
        validate(args)


if __name__ == '__main__':
    main()
