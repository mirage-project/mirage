# w2long.py TRACE_DIR GPU...: W2 items (kind 2) by start time bin: count, median / p90 duration, and the slot (expert) of the long ones
import numpy as np, sys, os
d = sys.argv[1]
for g in [int(x) for x in sys.argv[2:]]:
    tr = np.load(os.path.join(d, f"moe_trace_gpu{g}.npy")).astype(np.int64)
    ref = np.median(np.load(os.path.join(d, f"stamps_gpu{g}.npy")).astype(np.int64)[:, 0])
    it = []   # (start, dur, slot, wave)
    w13end = []
    for w in range(tr.shape[0]):
        for i in range(tr.shape[1]):
            a, b = int(tr[w, i, 0]), int(tr[w, i, 1])
            if not a: break
            k = (b >> 32) & 0xFF
            if k == 2: it.append(((a - ref) / 1e3, (b & 0xFFFFFFFF) / 1e3, (b >> 40) & 0xFFFFFF, w))
            if k == 1: w13end.append((a - ref) / 1e3 + (b & 0xFFFFFFFF) / 1e3)
    it = np.array(it); w13end = np.array(w13end)
    print(f"GPU {g}: W2 items {len(it)}, W13 units end p50 {np.median(w13end):.1f} p90 {np.percentile(w13end, 90):.1f} max {w13end.max():.1f}")
    for lo in range(80, 180, 10):
        m = (it[:, 0] >= lo) & (it[:, 0] < lo + 10)
        if m.sum():
            du = it[m, 1]
            print(f"  start {lo:3d}-{lo + 10:3d}: n {m.sum():5d} dur p50 {np.median(du):5.2f} p90 {np.percentile(du, 90):5.2f} max {du.max():5.2f}; W13 units still running at bin start {((w13end > lo)).sum():5d}")
