import glob, itertools, os, sys
import numpy as np

d = sys.argv[1]
runs = {}
for f in sorted(glob.glob(os.path.join(d, "mask_*.npz"))):
    tag = os.path.basename(f)[5:-4]
    z = np.load(f)
    runs[tag] = {k: z[k] for k in z.files}
tags = sorted(runs)
print("runs:", tags)
keys = [k for k in runs[tags[0]] if k.endswith("_moe_mask")]
print("mask samples per run:", len(keys))
NUM_EXPERTS = len(runs[tags[0]][keys[0]]) - 1
print("NUM_EXPERTS:", NUM_EXPERTS)

order_diff = set_diff = count_diff = tok_diff = 0
examples = []
for a, b in itertools.combinations(tags, 2):
    for k in keys:
        A, B = runs[a][k], runs[b][k]
        na, nb = int(A[NUM_EXPERTS]), int(B[NUM_EXPERTS])
        if na != nb:
            count_diff += 1
            if len(examples) < 6:
                examples.append(f"COUNT {a} vs {b} {k}: {na} vs {nb}")
            continue
        la, lb = A[:na], B[:nb]
        if not np.array_equal(la, lb):
            order_diff += 1
            if set(la.tolist()) != set(lb.tolist()):
                set_diff += 1
                if len(examples) < 6:
                    examples.append(
                        f"SET   {a} vs {b} {k}: "
                        f"only-a={sorted(set(la.tolist())-set(lb.tolist()))[:6]} "
                        f"only-b={sorted(set(lb.tolist())-set(la.tolist()))[:6]}")
            elif len(examples) < 6:
                examples.append(f"ORDER {a} vs {b} {k}: {la[:8]} vs {lb[:8]}")
    for k in [x for x in runs[a] if x.endswith("::tokens")]:
        if not np.array_equal(runs[a][k], runs[b][k]):
            tok_diff += 1
npairs = len(tags) * (len(tags) - 1) // 2 * len(keys)
print(f"\npairs compared: {npairs}")
print(f"  count differs : {count_diff}")
print(f"  order differs : {order_diff}  (of which SET differs: {set_diff})")
print(f"  token dumps differing: {tok_diff}")
for e in examples:
    print("   ", e)
# ascending check: post-I5c the list must be strictly ascending
for t in tags:
    bad = [k for k in keys
           if not np.all(np.diff(runs[t][k][:int(runs[t][k][NUM_EXPERTS])]) > 0)]
    print(f"  {t}: non-ascending masks {len(bad)}/{len(keys)}")
