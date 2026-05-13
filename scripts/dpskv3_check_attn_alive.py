#!/usr/bin/env python3
"""Quick check: is attention contribution non-zero?

For a single-layer dump dir, compares layer0_attn_out.pt rows 0..N-1
against embed.pt rows 0..N-1. If attn is dead, attn_out ≈ embed
(residual passes through). If attn is alive, attn_out differs from
embed in direction.
"""
import argparse
import os
import sys
import torch
import torch.nn.functional as F


def check(dump_dir, prompt_len):
    embed_p = os.path.join(dump_dir, "embed.pt")
    attn_p = os.path.join(dump_dir, "layer0_attn_out.pt")
    mlp_p = os.path.join(dump_dir, "layer0_dense_mlp_out.pt")
    if not os.path.exists(embed_p):
        print(f"missing embed: {embed_p}", file=sys.stderr)
        return 2
    if not os.path.exists(attn_p):
        print(f"missing attn_out: {attn_p}", file=sys.stderr)
        return 2
    embed = torch.load(embed_p, map_location="cpu", weights_only=True).float()
    attn = torch.load(attn_p, map_location="cpu", weights_only=True).float()
    n = min(embed.shape[0], attn.shape[0], prompt_len)
    embed = embed[:n]
    attn = attn[:n]
    # Diff
    diff = attn - embed
    diff_l2 = diff.norm(dim=-1)
    embed_l2 = embed.norm(dim=-1)
    attn_l2 = attn.norm(dim=-1)
    cos_attn_embed = F.cosine_similarity(attn, embed, dim=-1)
    bad = bool(torch.isnan(attn).any() or torch.isinf(attn).any())

    # Row-by-row classification
    zero_rows = int((attn_l2 < 1e-3).sum().item())
    residual_rows = int((cos_attn_embed > 0.9999).sum().item())
    attn_alive_rows = int(((cos_attn_embed < 0.999) & (attn_l2 > 1e-3)).sum().item())

    print(f"# rows={n} prompt_len={prompt_len}")
    print(f"# embed L2 mean = {embed_l2.mean():.4f}")
    print(f"# attn  L2 mean = {attn_l2.mean():.4f}")
    print(f"# diff  L2 mean = {diff_l2.mean():.4f}  (= |attn - embed|)")
    print(f"# cos(attn, embed) mean = {cos_attn_embed.mean():.4f}")
    print(f"# zero rows         (L2 < 1e-3): {zero_rows}")
    print(f"# residual_only rows (cos~1.0):  {residual_rows}")
    print(f"# attn-alive rows  (cos<0.999 + L2>0): {attn_alive_rows}")
    print(f"# NaN/Inf in attn_out: {bad}")

    if attn_alive_rows >= n // 2:
        print("VERDICT: ATTENTION IS ALIVE (>=50% rows show attn contribution)")
        return 0
    elif residual_rows >= n - 5:
        print("VERDICT: ATTENTION IS DEAD (residual_only)")
        return 1
    else:
        print("VERDICT: PARTIAL (mixed pattern)")
        return 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dump_dir")
    p.add_argument("--prompt-len", type=int, default=64)
    args = p.parse_args()
    return check(args.dump_dir, args.prompt_len)


if __name__ == "__main__":
    sys.exit(main())
