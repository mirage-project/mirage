#!/usr/bin/env python3
"""M4-I0 positive-control analysis: at the campaign-2 census geometry (1024 new
tokens), how DEEP into the decode does a state divergence have to run before it
surfaces in the token ids?

This is the quantitative bridge between the 1024-token census rate and the AC-3
geometry (64 new tokens). For every rep that deviates from the per-key consensus
it reports the diverging wave keys, the first diverging KV cache index (the
bitfp keeps dims (layer, page*page_size+offset), so the index localises the
first position whose K/V bytes differ), and the first divergent TOKEN position.

usage: pc_analyze.py <fps-dir>
"""
from __future__ import annotations
import collections, glob, json, os, sys
import numpy as np


def main() -> int:
    d = sys.argv[1]
    runs = {}
    for f in sorted(glob.glob(os.path.join(d, "fp_*.npz"))):
        tag = os.path.basename(f)[3:-4]
        z = np.load(f)
        runs[tag] = {k: z[k] for k in z.files}
    if not runs:
        print(f"no fp_*.npz in {d}")
        return 1
    tags = sorted(runs)
    keys = sorted(set().union(*(set(r) for r in runs.values())))
    consensus, key_dev = {}, {}
    for k in keys:
        buckets = collections.defaultdict(list)
        for t in tags:
            v = runs[t].get(k)
            buckets[b"MISSING" if v is None else v.tobytes()].append(t)
        best = max(buckets.items(), key=lambda kv: len(kv[1]))
        consensus[k] = best[1]
        dev = [t for t in tags if t not in best[1]]
        if dev:
            key_dev[k] = dev

    per_run = collections.defaultdict(list)
    for k, dev in key_dev.items():
        for t in dev:
            per_run[t].append(k)

    md5s = {}
    for t in tags:
        mp = os.path.join(d, f"meta_{t}.json")
        if os.path.exists(mp):
            md5s[t] = json.load(open(mp)).get("dump_md5")
    md5c = collections.Counter(md5s.values())
    cons_md5 = md5c.most_common(1)[0][0] if md5c else None

    state_div = sorted(t for t in per_run
                       if any(k.startswith("w") for k in per_run[t]))
    tok_div = sorted(t for t in per_run
                     if any(k.startswith("tok_") for k in per_run[t]))
    print(f"reps={len(tags)}  keys={len(keys)}")
    print(f"STATE-divergent reps: {len(state_div)}/{len(tags)} "
          f"({len(state_div)/len(tags):.1%})  -> {' '.join(state_div) or '(none)'}")
    print(f"TOKEN-divergent reps: {len(tok_div)}/{len(tags)} "
          f"({len(tok_div)/len(tags):.1%})  -> {' '.join(tok_div) or '(none)'}")
    print(f"md5 census: {dict(md5c)}")

    first_positions = []
    for t in sorted(per_run):
        ks = sorted(per_run[t])
        waves = sorted({k.rsplit('_', 1)[0] for k in ks if k.startswith('w')})
        print(f"\nrep {t}: {len(ks)} deviating keys; waves touched {waves or '(none)'}"
              f"; md5 {'CONSENSUS' if md5s.get(t) == cons_md5 else 'MINORITY'}")
        for k in ks:
            a = runs[t][k]
            ref = runs[consensus[k][0]][k]
            if a.shape != ref.shape:
                print(f"  {k}: SHAPE {a.shape} vs {ref.shape}")
                continue
            bad = np.argwhere(a != ref)
            frac = len(bad) / max(a.size, 1)
            if k.startswith("tok_"):
                fp = int(bad[0][0])
                first_positions.append((t, k[4:], fp, len(a)))
                print(f"  {k}: {len(bad)}/{a.size} tokens differ ({frac:.1%}) "
                      f"FIRST DIVERGENT TOKEN POSITION = {fp}")
            else:
                # (layer, page*page_size+offset) for k/v; (layer, slot) for conv/rec
                lo = bad.min(axis=0).tolist()
                print(f"  {k}: {len(bad)}/{a.size} entries differ ({frac:.1%}) "
                      f"min_index={lo} shape={list(a.shape)}")

    if first_positions:
        pos = sorted(p for _, _, p, _ in first_positions)
        n_lt64 = sum(1 for p in pos if p < 64)
        print(f"\n=== first-divergent-token-position distribution "
              f"({len(pos)} diverging prompt trajectories) ===")
        print(f"  positions: {pos}")
        print(f"  min {pos[0]}  median {pos[len(pos)//2]}  max {pos[-1]}")
        print(f"  BELOW POSITION 64 (the AC-3 decode length): {n_lt64}/{len(pos)} "
              f"({n_lt64/len(pos):.1%})")
    else:
        print("\nno token divergence in this arm -> no first-position distribution")
    return 0


if __name__ == "__main__":
    sys.exit(main())
