#!/usr/bin/env python3
"""Analyze MPK per-layer residual dumps to find the FIRST layer where the
v5 zero-mask rows develop NaN/zero L2 norms.

Usage:
    python scripts/dpskv3_dump_first_bad_layer.py --dump-dir outputs/dpskv3_dump_*

Requires the demo to have been run with --dump-hidden-dir <dir>.

Reads <dir>/layer_NN_residual.pt files and reports:
1. Per-layer L2 norm of each row in [0, q_len).
2. The first layer index where rows in the v5 zero-mask
   ([16, 20-25, 29-34, ...]) develop very small / NaN L2.
3. The cumulative magnitude trend per row across layers.
"""
import argparse
import math
import os
import re
from pathlib import Path

import torch


# v5 zero-mask for q=84 mbt=128 TP=4 layers=0-19 baseline (from
# outputs/dpskv3_ablation_findings_v5_20260508.md).
V5_ZERO_MASK = [
    16, 20, 21, 22, 23, 24, 25,
    29, 30, 31, 32, 33, 34,
    38, 39, 40, 41, 42, 43,
    47, 48, 49, 50, 51, 52, 53,
    56, 57, 58, 59, 60, 61, 62,
    65, 66, 67, 68, 69, 70, 71,
]
V5_GOOD_ROWS = [r for r in range(84) if r not in V5_ZERO_MASK]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True,
                        help="Directory containing layer_NN_residual.pt files")
    parser.add_argument("--q-len", type=int, default=84,
                        help="Active prefill row count (default 84)")
    parser.add_argument("--small-threshold", type=float, default=1e-3,
                        help="L2 norm below this counts as 'degenerate'")
    args = parser.parse_args()

    files = sorted(
        args.dump_dir.glob("layer_*_residual.pt"),
        key=lambda p: int(re.search(r"layer_(\d+)", p.name).group(1)),
    )
    if not files:
        raise SystemExit(f"No layer_*_residual.pt files in {args.dump_dir}")

    print(f"Found {len(files)} layer dumps in {args.dump_dir}")
    print(f"Analyzing first {args.q_len} rows; v5 zero-mask has "
          f"{len(V5_ZERO_MASK)} bad rows, {len(V5_GOOD_ROWS)} good rows")
    print()

    first_nan_layer = None
    first_zero_layer = None
    bad_row_first_appearance: dict[int, int] = {}

    layer_idx_pattern = re.compile(r"layer_(\d+)")
    print(f"{'layer':>5} {'all_finite':>11} {'n_nan_rows':>11} "
          f"{'mean_L2_good':>14} {'mean_L2_bad':>13} "
          f"{'min_L2_good':>13} {'min_L2_bad':>13}")
    for path in files:
        m = layer_idx_pattern.search(path.name)
        layer_i = int(m.group(1))
        t = torch.load(path, map_location="cpu", weights_only=True)
        # t shape: (mbt, hidden_size)
        active = t[:args.q_len].float()
        l2 = active.norm(dim=-1)  # (q_len,)
        nan_mask = torch.isnan(l2)
        small_mask = (~nan_mask) & (l2 < args.small_threshold)
        bad_or_nan = nan_mask | small_mask

        n_nan = int(nan_mask.sum().item())
        all_finite = bool((~nan_mask).all().item())

        good_l2 = l2[V5_GOOD_ROWS]
        bad_l2 = l2[V5_ZERO_MASK]
        # filter NaN out for the means
        gl = good_l2[~torch.isnan(good_l2)]
        bl = bad_l2[~torch.isnan(bad_l2)]
        mean_good = gl.mean().item() if gl.numel() else float("nan")
        mean_bad = bl.mean().item() if bl.numel() else float("nan")
        min_good = gl.min().item() if gl.numel() else float("nan")
        min_bad = bl.min().item() if bl.numel() else float("nan")

        print(f"{layer_i:5d} {str(all_finite):>11} {n_nan:>11d} "
              f"{mean_good:14.4g} {mean_bad:13.4g} "
              f"{min_good:13.4g} {min_bad:13.4g}")

        if not all_finite and first_nan_layer is None:
            first_nan_layer = layer_i
        if int(small_mask.sum().item()) > 0 and first_zero_layer is None:
            first_zero_layer = layer_i

        # Per-row first-appearance tracking
        for r in V5_ZERO_MASK:
            if bad_or_nan[r].item() and r not in bad_row_first_appearance:
                bad_row_first_appearance[r] = layer_i

    print()
    print(f"First layer with any NaN row: {first_nan_layer}")
    print(f"First layer with any small-L2 row (< {args.small_threshold}): "
          f"{first_zero_layer}")
    if bad_row_first_appearance:
        print()
        print("v5 zero-mask rows: layer at which each row first goes bad")
        for r in V5_ZERO_MASK:
            layer = bad_row_first_appearance.get(r, "not bad in dump")
            print(f"  row {r:3d} -> layer {layer}")
    else:
        print("None of the v5 zero-mask rows had bad L2 in any dumped layer.")


if __name__ == "__main__":
    main()
