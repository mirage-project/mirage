import glob, itertools, os, re, sys
import numpy as np

dirs = sys.argv[1:]
runs = {}
for d in dirs:
    for f in sorted(glob.glob(os.path.join(d, "fp_*.npz"))):
        tag = os.path.basename(f)[3:-4]
        m = re.match(r"bs(\d+)_", tag)
        if not m:
            continue
        z = np.load(f)
        runs[(int(m.group(1)), tag)] = {k: z[k] for k in z.files}
bss = sorted({bs for bs, _ in runs})
total_pairs = total_diffs = 0
print(f"loaded {len(runs)} runs at bs {bss}")
for bs in bss:
    tags = sorted(t for b, t in runs if b == bs)
    keys = sorted(runs[(bs, tags[0])].keys())
    ntraj = sum(1 for k in keys if k.startswith("tok_"))
    nwave = sum(1 for k in keys if k.endswith("_k"))
    print(f"\nbs={bs}: {len(tags)} runs, {nwave} wave fingerprints + "
          f"{ntraj} token dumps each  ({len(tags)*ntraj} trajectories)")
    diffs = []
    for a, b in itertools.combinations(tags, 2):
        A, B = runs[(bs, a)], runs[(bs, b)]
        if set(A) != set(B):
            diffs.append(f"{a} vs {b}: KEY SET differs")
            continue
        for k in keys:
            total_pairs += 1
            if not np.array_equal(A[k], B[k]):
                total_diffs += 1
                bad = np.argwhere(A[k] != B[k])
                diffs.append(f"{a} vs {b} :: {k} first={tuple(bad[0])} n={len(bad)}")
    print(f"  pairwise element comparisons: {len(tags)*(len(tags)-1)//2 * len(keys)}")
    print(f"  DIFFERENCES: {len(diffs)}")
    for x in diffs[:10]:
        print("    ", x)
print(f"\nTOTAL: {total_pairs} fingerprint/token-array comparisons, "
      f"{total_diffs} differ")
