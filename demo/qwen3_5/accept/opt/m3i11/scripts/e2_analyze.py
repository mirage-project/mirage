import glob, itertools, os, sys
import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/mpk-qwen35/m3i11/out/e2")
files = sorted(glob.glob(os.path.join(d, "fp_*.npz")))
runs = {}
for f in files:
    tag = os.path.basename(f)[3:-4]
    z = np.load(f)
    for k in z.files:
        if k.endswith("_k"):
            w = k.split("_")[0]
            runs[(tag, w)] = {kk.split("_", 1)[1]: z[kk] for kk in z.files if kk.startswith(w + "_")}
keys = sorted(runs)
print("samples:", len(keys), keys)
print("k_fp shape:", runs[keys[0]]["k"].shape)

base = keys[0]
groups = {}
for k in keys:
    sig = (runs[k]["k"].tobytes(), runs[k]["v"].tobytes(),
           runs[k]["conv"].tobytes(), runs[k]["rec"].tobytes(),
           runs[k]["tok"].tobytes())
    groups.setdefault(hash(sig), []).append(k)
print("\ndistinct full-state groups:", len(groups))
for g, mem in groups.items():
    print("  ", len(mem), mem)

ndiff = 0
for a, b in itertools.combinations(keys, 2):
    A, B = runs[a], runs[b]
    parts = []
    for name in ("k", "v"):
        dif = np.argwhere(A[name] != B[name])
        if len(dif):
            L, S = dif[0]
            parts.append(f"{name}: first (layer={L}, slot={S}) ndiff={len(dif)}")
    for name in ("conv", "rec", "tok"):
        if not np.array_equal(A[name], B[name]):
            dif = np.argwhere(A[name] != B[name])
            parts.append(f"{name}: first {tuple(dif[0])} ndiff={len(dif)}")
    if parts:
        ndiff += 1
        print(f"DIFF {a} vs {b}: " + " | ".join(parts))
print(f"\npairs differing: {ndiff} / {len(keys)*(len(keys)-1)//2}")
