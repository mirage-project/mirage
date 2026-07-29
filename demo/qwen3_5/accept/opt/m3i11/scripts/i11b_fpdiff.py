#!/usr/bin/env python3
"""M3-I11 campaign 2 census analyser.

Reads every fp_<tag>.npz in a directory (optionally filtered by a tag
substring), takes the per-key majority value as the consensus, and reports
which reps deviate -- per wave-boundary fingerprint key and per token array.
That is the shape the M3-I5c S6 divergence had (one whole wave's KV/GDN entries
perturbed, neighbouring waves byte-clean), so the signature is printed
explicitly, not just a pair count.

usage: i11b_fpdiff.py <dir> [tag-substring]
"""
from __future__ import annotations
import collections, glob, json, os, sys
import numpy as np


def main() -> int:
    d = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else ""
    runs: dict[str, dict[str, np.ndarray]] = {}
    for f in sorted(glob.glob(os.path.join(d, "fp_*.npz"))):
        tag = os.path.basename(f)[3:-4]
        if want and want not in tag:
            continue
        z = np.load(f)
        runs[tag] = {k: z[k] for k in z.files}
    if not runs:
        print(f"no fp_*.npz matching {want!r} in {d}")
        return 1
    tags = sorted(runs)
    keys = sorted(set().union(*(set(r) for r in runs.values())))
    print(f"runs={len(tags)}  keys={len(keys)}")
    print(f"tags: {' '.join(tags)}")

    # consensus per key = the most common byte pattern across reps
    consensus, key_deviants = {}, {}
    for k in keys:
        buckets: dict[bytes, list[str]] = collections.defaultdict(list)
        for t in tags:
            v = runs[t].get(k)
            buckets[b"MISSING" if v is None else v.tobytes()].append(t)
        best = max(buckets.items(), key=lambda kv: len(kv[1]))
        consensus[k] = best[1]
        dev = [t for t in tags if t not in best[1]]
        if dev:
            key_deviants[k] = dev

    per_run = collections.defaultdict(list)
    for k, dev in key_deviants.items():
        for t in dev:
            per_run[t].append(k)

    print(f"\nkeys with any deviation: {len(key_deviants)} / {len(keys)}")
    print(f"DIVERGING REPS: {len(per_run)} / {len(tags)}"
          f"  -> {' '.join(sorted(per_run)) if per_run else '(none)'}")

    for t in sorted(per_run):
        ks = sorted(per_run[t])
        waves = sorted({k.rsplit('_', 1)[0] for k in ks if k.startswith('w')})
        toks = [k for k in ks if k.startswith('tok_')]
        print(f"\n  rep {t}: {len(ks)} deviating keys")
        print(f"    waves touched: {waves or '(none)'}")
        for k in ks:
            a = runs[t][k]
            ref = runs[consensus[k][0]][k]
            if a.shape != ref.shape:
                print(f"    {k}: SHAPE {a.shape} vs {ref.shape}")
                continue
            bad = np.argwhere(a != ref)
            frac = len(bad) / max(a.size, 1)
            extra = ""
            if k.startswith("tok_"):
                extra = f" first_pos={int(bad[0][0])}"
            print(f"    {k}: {len(bad)}/{a.size} entries differ ({frac:.1%})"
                  f"{extra}")
        if toks:
            print(f"    token arrays diverging: {len(toks)} "
                  f"({' '.join(sorted(x[4:] for x in toks))})")

    # wave-scope check: for each diverging rep, which wave keys are CLEAN
    if per_run:
        allw = sorted({k.rsplit('_', 1)[0] for k in keys if k.startswith('w')})
        print(f"\nwave-scope table (wave keys present: {allw})")
        for t in sorted(per_run):
            touched = {k.rsplit('_', 1)[0] for k in per_run[t] if k.startswith('w')}
            print(f"  {t}: " + " ".join(
                f"{w}={'DIFF' if w in touched else 'clean'}" for w in allw))

    # md5 census from the meta files, for direct comparison with older campaigns
    md5s = {}
    for t in tags:
        mp = os.path.join(d, f"meta_{t}.json")
        if os.path.exists(mp):
            md5s[t] = json.load(open(mp)).get("dump_md5")
    if md5s:
        c = collections.Counter(md5s.values())
        print(f"\ndump md5 census: {dict(c)}")
        for t, m in sorted(md5s.items()):
            if c[m] != max(c.values()):
                print(f"  minority md5 rep: {t} {m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
