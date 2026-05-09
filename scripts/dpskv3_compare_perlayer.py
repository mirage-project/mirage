#!/usr/bin/env python3
"""Compare per-layer MPK residual dumps to reference dumps.

Usage:
    python scripts/dpskv3_compare_perlayer.py \
        --mpk outputs/dpskv3_mpk_perlayer_<ts> \
        --ref outputs/dpskv3_ref_dump_20260509_001712_FIXED \
        [--prompt-len 84] [--cmp-row 83]

Outputs a per-layer table of mean_cos / row_cmp_cos / l2_ratio /
NaN-or-inf flag, plus the FIRST layer that crosses the divergence
thresholds (cos < 0.5 or |l2_ratio| < 0.5 / > 2.0 / NaN).
"""
import argparse
import os
import sys
import torch
import torch.nn.functional as F


def compare(mpk_dir: str, ref_dir: str, prompt_len: int, cmp_row: int,
            num_layers: int = 20):
    print(f"# Per-layer cosine: MPK={mpk_dir}")
    print(f"#                   REF={ref_dir}")
    print(f"# prompt_len={prompt_len} cmp_row={cmp_row} (last position)")
    print(f"{'tag':<22} {'mean_cos':>10} {'row_cos':>10} "
          f"{'l2_ratio':>10} {'bad':>6}")
    print("-" * 64)
    rows = []
    tags = ["embed"] + [f"layer_{i:02d}_residual" for i in range(num_layers)]
    for tag in tags:
        mpk_p = os.path.join(mpk_dir, tag + ".pt")
        ref_p = os.path.join(ref_dir, tag + ".pt")
        if not os.path.exists(mpk_p) or not os.path.exists(ref_p):
            print(f"{tag:<22} (missing)")
            continue
        mpk = torch.load(mpk_p, map_location="cpu", weights_only=True).float()
        ref = torch.load(ref_p, map_location="cpu", weights_only=True).float()
        n = min(mpk.shape[0], ref.shape[0], prompt_len)
        mpk = mpk[:n]
        ref = ref[:n]
        bad = bool(torch.isnan(mpk).any() or torch.isinf(mpk).any())
        cos = F.cosine_similarity(mpk, ref, dim=-1)
        l2_m = mpk.norm(dim=-1)
        l2_r = ref.norm(dim=-1)
        ratio = (l2_m / l2_r.clamp(min=1e-6)).mean().item()
        row_cos = cos[cmp_row].item() if cmp_row < n else float("nan")
        rows.append({
            "tag": tag,
            "mean_cos": cos.mean().item(),
            "row_cos": row_cos,
            "l2_ratio": ratio,
            "bad": bad,
        })
        print(f"{tag:<22} {cos.mean().item():>10.4f} {row_cos:>10.4f} "
              f"{ratio:>10.4f} {str(bad):>6}")
    # First-bad-layer
    first_bad = None
    for r in rows:
        if r["tag"] == "embed":
            continue
        if (r["mean_cos"] < 0.5 or r["row_cos"] < 0.5
                or r["l2_ratio"] < 0.5 or r["l2_ratio"] > 2.0
                or r["bad"]):
            first_bad = r
            break
    print()
    if first_bad is None:
        print("# No layer crossed divergence thresholds "
              "(cos < 0.5 / l2 < 0.5 or > 2.0 / NaN).")
    else:
        print(f"# First diverged layer: {first_bad['tag']} "
              f"(mean_cos={first_bad['mean_cos']:.3f} "
              f"row_cos={first_bad['row_cos']:.3f} "
              f"l2={first_bad['l2_ratio']:.3f} bad={first_bad['bad']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpk", required=True, help="MPK dump dir")
    ap.add_argument("--ref", required=True, help="Reference dump dir")
    ap.add_argument("--prompt-len", type=int, default=84,
                    help="Number of valid prompt positions")
    ap.add_argument("--cmp-row", type=int, default=83,
                    help="Row index to highlight (default 83 = last position "
                         "of prompt=84)")
    ap.add_argument("--num-layers", type=int, default=20)
    args = ap.parse_args()
    compare(args.mpk, args.ref, args.prompt_len, args.cmp_row,
            args.num_layers)


if __name__ == "__main__":
    main()
