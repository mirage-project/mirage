#!/usr/bin/env python3
"""Pooled discriminator table for M3-I11 campaign 2.

For every census set (one arm on one GPU) it reports, per rep:
  * whether ANY wave-boundary fingerprint key deviates from that set's
    consensus  -> STATE-level divergence (the ~100%-sensitive instrument)
  * whether any token array deviates -> OUTPUT-level divergence
  * the position of the rep inside its interleaved block (the blocks alternate
    ctrl/fix, so position = 2*rep-1 for ctrl and 2*rep for fix), because
    early-in-block clustering is itself a candidate mechanism
and then the arm totals plus the pre-registered one-sample binomial test of the
FIX arm against the baseline rate.
"""
from __future__ import annotations
import collections, glob, json, os, sys
import numpy as np


def load(d):
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "fp_*.npz"))):
        tag = os.path.basename(f)[3:-4]
        z = np.load(f)
        runs[tag] = {k: z[k] for k in z.files}
    return runs


def classify(runs):
    """-> {tag: (state_div, token_div, detail)} using per-key majority consensus."""
    if not runs:
        return {}
    tags = sorted(runs)
    keys = sorted(set().union(*(set(r) for r in runs.values())))
    out = {t: [False, False, []] for t in tags}
    for k in keys:
        buckets = collections.defaultdict(list)
        for t in tags:
            v = runs[t].get(k)
            buckets[b"MISS" if v is None else v.tobytes()].append(t)
        best = max(buckets.items(), key=lambda kv: len(kv[1]))
        if len(best[1]) == len(tags):
            continue
        ref = runs[best[1][0]][k]
        for t in tags:
            if t in best[1]:
                continue
            a = runs[t][k]
            n = int((a != ref).sum()) if a.shape == ref.shape else a.size
            out[t][0] = True
            if k.startswith("tok_"):
                out[t][1] = True
            out[t][2].append(f"{k}:{n}/{a.size}")
    return out


def main():
    base = sys.argv[1]
    rate = float(sys.argv[2]) if len(sys.argv) > 2 else 0.10
    sets = sorted(d for d in glob.glob(os.path.join(base, "*"))
                  if os.path.isdir(d) and glob.glob(os.path.join(d, "fp_*.npz")))
    tot = collections.defaultdict(lambda: [0, 0, 0])  # arm -> [n, state, token]
    per_gpu = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0, 0]))
    print(f"{'set':14s} {'rep':16s} {'pos':>4s} {'state':>6s} {'token':>6s}  detail")
    for d in sets:
        name = os.path.basename(d)
        runs = load(d)
        cls = classify(runs)
        arm = "fix" if name.endswith("_fix") else "ctrl" if name.endswith("_ctrl") else name
        gpu = name.split("_")[0]
        for t in sorted(cls, key=lambda x: int(x.rsplit("_c", 1)[1])):
            r = int(t.rsplit("_c", 1)[1])
            pos = 2 * r - 1 if arm == "ctrl" else 2 * r
            s, k, det = cls[t]
            tot[arm][0] += 1
            per_gpu[gpu][arm][0] += 1
            if s:
                tot[arm][1] += 1; per_gpu[gpu][arm][1] += 1
            if k:
                tot[arm][2] += 1; per_gpu[gpu][arm][2] += 1
            if s:
                print(f"{name:14s} {t:16s} {pos:4d} {'DIFF':>6s} "
                      f"{'DIFF' if k else '-':>6s}  {' '.join(det)}")
    print("\n=== per GPU ===")
    for g in sorted(per_gpu):
        for arm in ("ctrl", "fix"):
            n, s, k = per_gpu[g][arm]
            if n:
                print(f"  {g:5s} {arm:4s}: n={n:3d} state-div={s} token-div={k}")
    print("\n=== pooled ===")
    for arm in ("ctrl", "fix"):
        n, s, k = tot[arm]
        if n:
            print(f"  {arm:4s}: n={n:3d} state-div={s} ({s/n:.1%})"
                  f" token-div={k} ({k/n:.1%})")
    nf, sf, kf = tot["fix"]
    if nf:
        print(f"\npre-registered one-sample test, baseline rate {rate}:")
        print(f"  P(0 state-div in {nf} | rate {rate}) = {(1-rate)**nf:.4f}"
              f"   [observed {sf}]")
        print(f"  P(0 token-div in {nf} | rate {rate}) = {(1-rate)**nf:.4f}"
              f"   [observed {kf}]")
    nc, sc, kc = tot["ctrl"]
    if nc and nf:
        try:
            from math import comb
            def fisher(a, b, c, d):
                n = a + b + c + d
                tot_ = 0.0
                obs = comb(a + b, a) * comb(c + d, c) / comb(n, a + c)
                for i in range(0, min(a + b, a + c) + 1):
                    j = a + c - i
                    if j < 0 or j > c + d:
                        continue
                    p = comb(a + b, i) * comb(c + d, j) / comb(n, a + c)
                    if p <= obs + 1e-12:
                        tot_ += p
                return tot_
            print(f"  Fisher exact ctrl {sc}/{nc} vs fix {sf}/{nf} (state): "
                  f"p={fisher(sc, nc-sc, sf, nf-sf):.4f}")
            print(f"  Fisher exact ctrl {kc}/{nc} vs fix {kf}/{nf} (token): "
                  f"p={fisher(kc, nc-kc, kf, nf-kf):.4f}")
        except Exception as e:
            print("  fisher failed:", e)


if __name__ == "__main__":
    main()
