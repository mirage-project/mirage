#!/usr/bin/env python3
"""Compare two MPK per-layer residual dumps to check bit-exact determinism.

Usage:
    python scripts/dpskv3_dump_determinism.py --dir-a outputs/dpskv3_dump_run1 \\
                                              --dir-b outputs/dpskv3_dump_run2

Reports per-layer:
- max abs delta between run-A and run-B
- whether any row's L2 differs
- whether NaN positions match

Used by todo task #36 to verify that the same MPK config produces
bit-exact same per-layer outputs across re-runs.
"""
import argparse
import re
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir-a", type=Path, required=True)
    parser.add_argument("--dir-b", type=Path, required=True)
    parser.add_argument("--q-len", type=int, default=84)
    args = parser.parse_args()

    files_a = sorted(args.dir_a.glob("layer_*_residual.pt"),
                     key=lambda p: int(re.search(r"layer_(\d+)", p.name).group(1)))
    files_b = sorted(args.dir_b.glob("layer_*_residual.pt"),
                     key=lambda p: int(re.search(r"layer_(\d+)", p.name).group(1)))

    if [p.name for p in files_a] != [p.name for p in files_b]:
        raise SystemExit("File sets differ between dirs")

    print(f"{'layer':>5} {'max_abs_diff':>14} {'mean_abs_diff':>14} "
          f"{'bit_exact':>10} {'nan_pos_match':>14}")
    all_bit_exact = True
    for fa, fb in zip(files_a, files_b):
        ta = torch.load(fa, map_location="cpu", weights_only=True)[:args.q_len]
        tb = torch.load(fb, map_location="cpu", weights_only=True)[:args.q_len]
        layer_i = int(re.search(r"layer_(\d+)", fa.name).group(1))
        # Compare bit-exact via byte equality (handles NaN bits)
        bit_exact = ta.untyped_storage().nbytes() == tb.untyped_storage().nbytes() \
                    and ta.contiguous().view(torch.uint8).equal(tb.contiguous().view(torch.uint8))
        # NaN position match (NaN-tolerant comparison)
        nan_a = torch.isnan(ta.float())
        nan_b = torch.isnan(tb.float())
        nan_match = bool(torch.equal(nan_a, nan_b))
        # Numeric diff (NaN-tolerant: zero out NaN positions)
        af = ta.float().nan_to_num(nan=0.0)
        bf = tb.float().nan_to_num(nan=0.0)
        diff = (af - bf).abs()
        max_d = diff.max().item()
        mean_d = diff.mean().item()
        if not bit_exact:
            all_bit_exact = False
        print(f"{layer_i:5d} {max_d:14.6e} {mean_d:14.6e} "
              f"{str(bit_exact):>10} {str(nan_match):>14}")

    print()
    print(f"All layers bit-exact: {all_bit_exact}")


if __name__ == "__main__":
    main()
