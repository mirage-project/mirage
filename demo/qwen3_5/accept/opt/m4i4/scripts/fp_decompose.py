#!/usr/bin/env python3
"""M4-I4 -- decompose a KV/GDN fingerprint difference into WHICH tensors moved.

Why this exists. The cold gate at the shipped policy is STABLE and its tokens are
byte-identical to `results/dumps_final` at all five batch sizes -- but its per-bs
`state_sig` equals M4-I0's pinned UNCAPPED signature only at bs1, the one batch
size where the cap provably cannot bind. So the cap leaves the tokens alone while
changing some persistent state bits, and "which bits" decides whether that is a
documented mechanism or a silent degradation.

`gate_ac3_stable.py` writes each rep's per-key fingerprints to `fp_<tag>.npz`
(`w<N>_k`, `w<N>_v`, `w<N>_conv`, `w<N>_rec` per wave boundary, plus `tok_<pid>`
per prompt). This compares two reps key by key and reports which tensor families
differ -- so a difference confined to the slot-indexed GDN pools and the paged-KV
layout can be told apart from one that reaches the attention cache contents of a
live request, and from one that reaches the tokens.

Usage:
    python3 fp_decompose.py A=/path/fp_bs2_r1.npz B=/path/fp_bs2_c1.npz
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

FAMILIES = ("k", "v", "conv", "rec", "tok")


def family(key: str) -> str:
    if key.startswith("tok_"):
        return "tok"
    return key.rsplit("_", 1)[-1]


def load(path: Path) -> dict:
    import numpy as np
    z = np.load(path)
    return {k: z[k] for k in z.files}


def sig(arr) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pair", nargs=2, metavar="LABEL=NPZ")
    a = ap.parse_args(argv)

    labels, data = [], []
    for spec in a.pair:
        lab, _, p = spec.partition("=")
        labels.append(lab)
        data.append(load(Path(p)))
    A, B = data

    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))
    shared = sorted(set(A) & set(B))
    if only_a or only_b:
        print(f"KEY SETS DIFFER: only in {labels[0]}: {only_a}")
        print(f"                 only in {labels[1]}: {only_b}")

    per_family = {f: [0, 0] for f in FAMILIES}      # [same, differ]
    diffs = []
    for k in shared:
        same = A[k].shape == B[k].shape and sig(A[k]) == sig(B[k])
        fam = family(k)
        per_family.setdefault(fam, [0, 0])[0 if same else 1] += 1
        if not same:
            diffs.append(k)

    print(f"\ncomparing {labels[0]} vs {labels[1]}: {len(shared)} shared keys")
    print(f"{'family':>8} {'same':>6} {'differ':>7}")
    for fam, (s, d) in sorted(per_family.items()):
        if s or d:
            print(f"{fam:>8} {s:6d} {d:7d}")
    print(f"\nkeys that differ ({len(diffs)}): "
          + (", ".join(diffs) if len(diffs) <= 24 else
             ", ".join(diffs[:24]) + f", ... (+{len(diffs) - 24})"))
    if not diffs:
        print("=> the two reps are bit-identical in every fingerprinted tensor.")
    elif all(family(k) != "tok" for k in diffs):
        print("=> TOKENS UNCHANGED; the difference is confined to persistent state.")
    else:
        print("=> the difference REACHES THE TOKENS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
