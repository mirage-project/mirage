"""Compare MPK vs reference per-layer residual dumps row-by-row.

For each layer in both dump directories, report:
- per-row L2 norm in MPK and reference
- per-row absolute diff
- NaN/inf differences
- the row at which MPK first diverges substantially from reference

The MPK dump has shape (mbt, hidden) whereas reference has (q_len, hidden);
we slice both to [:q_len] for the comparison.
"""
import argparse
import re
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mpk-dir", type=Path, required=True)
    p.add_argument("--ref-dir", type=Path, required=True)
    p.add_argument("--q-len", type=int, default=84)
    p.add_argument("--diff-threshold", type=float, default=1.0,
                   help="Per-row L2 diff above this counts as 'diverged'")
    args = p.parse_args()

    mpk_files = sorted(args.mpk_dir.glob("layer_*_residual.pt"),
                       key=lambda p: int(re.search(r"layer_(\d+)", p.name).group(1)))
    ref_files = sorted(args.ref_dir.glob("layer_*_residual.pt"),
                       key=lambda p: int(re.search(r"layer_(\d+)", p.name).group(1)))

    if not mpk_files or not ref_files:
        raise SystemExit(f"No layer dumps in {args.mpk_dir} / {args.ref_dir}")

    mpk_layers = {int(re.search(r"layer_(\d+)", f.name).group(1)): f
                  for f in mpk_files}
    ref_layers = {int(re.search(r"layer_(\d+)", f.name).group(1)): f
                  for f in ref_files}
    common = sorted(set(mpk_layers) & set(ref_layers))
    if not common:
        raise SystemExit("No common layer indices between MPK and reference")

    print(f"Comparing {len(common)} layers, q_len={args.q_len}")
    print(f"MPK dir: {args.mpk_dir}")
    print(f"REF dir: {args.ref_dir}")
    print()
    print(f"{'layer':>5} {'mpk_finite':>10} {'ref_finite':>10} "
          f"{'n_div_rows':>11} {'max_l2_diff':>14} {'mean_l2_diff':>14}")

    first_div_layer = None
    for li in common:
        mpk = torch.load(mpk_layers[li], map_location="cpu",
                         weights_only=True)[:args.q_len].float()
        ref = torch.load(ref_layers[li], map_location="cpu",
                         weights_only=True)[:args.q_len].float()
        # Ensure same shape
        n_rows = min(mpk.shape[0], ref.shape[0])
        mpk = mpk[:n_rows]
        ref = ref[:n_rows]

        mpk_l2 = mpk.norm(dim=-1)
        ref_l2 = ref.norm(dim=-1)
        mpk_finite = bool((~torch.isnan(mpk_l2) & ~torch.isinf(mpk_l2)).all())
        ref_finite = bool((~torch.isnan(ref_l2) & ~torch.isinf(ref_l2)).all())

        # NaN-tolerant diff: zero out NaN/inf positions
        mpk_clean = mpk.clone()
        mpk_clean[torch.isnan(mpk_clean) | torch.isinf(mpk_clean)] = 0.0
        ref_clean = ref.clone()
        ref_clean[torch.isnan(ref_clean) | torch.isinf(ref_clean)] = 0.0
        diff = (mpk_clean - ref_clean).norm(dim=-1)

        # Mark NaN/inf in MPK as "diverged" too
        mpk_bad = torch.isnan(mpk_l2) | torch.isinf(mpk_l2)
        diverged = (diff > args.diff_threshold) | mpk_bad
        n_div = int(diverged.sum().item())
        max_d = diff.max().item()
        mean_d = diff.mean().item()

        if n_div > 0 and first_div_layer is None:
            first_div_layer = li
        print(f"{li:5d} {str(mpk_finite):>10} {str(ref_finite):>10} "
              f"{n_div:>11d} {max_d:14.4e} {mean_d:14.4e}")

    print()
    print(f"First diverged layer (>{args.diff_threshold} L2 OR NaN): "
          f"{first_div_layer}")

    if first_div_layer is not None:
        print()
        print(f"=== Detailed row breakdown of layer {first_div_layer} ===")
        mpk = torch.load(mpk_layers[first_div_layer], map_location="cpu",
                         weights_only=True)[:args.q_len].float()
        ref = torch.load(ref_layers[first_div_layer], map_location="cpu",
                         weights_only=True)[:args.q_len].float()
        n_rows = min(mpk.shape[0], ref.shape[0])
        mpk = mpk[:n_rows]
        ref = ref[:n_rows]
        mpk_l2 = mpk.norm(dim=-1)
        ref_l2 = ref.norm(dim=-1)
        mpk_clean = mpk.clone()
        mpk_clean[torch.isnan(mpk_clean) | torch.isinf(mpk_clean)] = 0.0
        ref_clean = ref.clone()
        ref_clean[torch.isnan(ref_clean) | torch.isinf(ref_clean)] = 0.0
        diff = (mpk_clean - ref_clean).norm(dim=-1)
        print(f"{'row':>4} {'mpk_l2':>14} {'ref_l2':>14} "
              f"{'diff':>14} status")
        for r in range(n_rows):
            mpk_v = mpk_l2[r].item()
            ref_v = ref_l2[r].item()
            d = diff[r].item()
            if torch.isnan(mpk_l2[r]) or torch.isinf(mpk_l2[r]):
                tag = "MPK_BAD"
            elif d > args.diff_threshold:
                tag = "DIVERGE"
            else:
                tag = "ok"
            print(f"{r:4d} {mpk_v:14.4e} {ref_v:14.4e} {d:14.4e} {tag}")

        # If first_div_layer > 0, also show the previous layer's row diff
        if first_div_layer > 0 and (first_div_layer - 1) in common:
            prev = first_div_layer - 1
            print()
            print(f"=== Sanity: layer {prev} (previous) row diff stats ===")
            mpk_p = torch.load(mpk_layers[prev], map_location="cpu",
                               weights_only=True)[:args.q_len].float()
            ref_p = torch.load(ref_layers[prev], map_location="cpu",
                               weights_only=True)[:args.q_len].float()
            n = min(mpk_p.shape[0], ref_p.shape[0])
            d_p = (mpk_p[:n] - ref_p[:n]).norm(dim=-1)
            print(f"  max diff: {d_p.max().item():.4e}")
            print(f"  mean diff: {d_p.mean().item():.4e}")
            print(f"  row 16 diff: {d_p[16].item():.4e}")
            print(f"  row 0 diff: {d_p[0].item():.4e}")
            print(f"  row 75 diff: {d_p[75].item():.4e}")


if __name__ == "__main__":
    main()
