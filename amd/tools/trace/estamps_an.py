# estamps_an.py STAMPS.npy [STAMPS2.npy ...]: moe_experts' phases from a normal build's per-CU stamps
# (test_layer_one_gpu.py --dump-stamps with -DSTATIC_EXPERT_STAMPS; [launches][GPUs][CUs][8] ns): per CU, us after its GPU's
# first go (stamp 0): 1 z_q in LDS, 4 first W13 unit start (wave 0), 5 W13 loop end, 6 first W2 load, 7 W2 loop end,
# 2 experts task end, 3 the CU's end; medians over the launches of the per-GPU percentiles (p10 / p50 / p90 / max)
import sys
import numpy as np
NAMES = [(1, "z_q in LDS"), (4, "first W13 start"), (5, "W13 loop end"), (6, "first W2 load (mode 2: table built)"), (7, "W2 loop end (mode 2: pairs seen)"),
         (2, "experts end"), (3, "CU end")]
for path in sys.argv[1:]:
    a = np.load(path).astype(np.float64)   # [L][G][CU][8]
    L, G = a.shape[:2]
    print(f"{path}: {L} launches, {G} GPUs")
    go = np.where(a[..., 0] > 0, a[..., 0], np.inf).min(axis=2)   # [L][G]
    names = NAMES + ([(8, "fused gate passed")] if a.shape[-1] > 8 else [])
    for k, name in names:
        v = a[..., k]
        rows = []
        for l in range(L):
            for g in range(G):
                x = v[l, g][v[l, g] > 0]
                if len(x):
                    x = (x - go[l, g]) / 1e3
                    rows.append([np.percentile(x, 10), np.median(x), np.percentile(x, 90), x.max()])
        if rows:
            m = np.median(np.array(rows), axis=0)
            print(f"  {name:22s} p10 {m[0]:6.1f}  p50 {m[1]:6.1f}  p90 {m[2]:6.1f}  max {m[3]:6.1f}")
    # per CU gaps (wave 0): W13 end -> first W2 load, W2 phase length
    gap = (a[..., 6] - a[..., 5]) / 1e3
    w2 = (a[..., 7] - a[..., 6]) / 1e3
    ok = (a[..., 5] > 0) & (a[..., 6] > 0) & (a[..., 7] > 0)
    print(f"  wave 0: W13 end -> first W2 load p50 {np.median(gap[ok]):.2f} p90 {np.percentile(gap[ok], 90):.2f} us; "
          f"W2 loop p50 {np.median(w2[ok]):.2f} p90 {np.percentile(w2[ok], 90):.2f} us")
