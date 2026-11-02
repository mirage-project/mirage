# dieend.py TRACE_DIR GPU: per die (sm_id % 8 of the wave's CU), the waves' last W2 item end: p50 / max, and the die's W2 items / wave-time
import numpy as np, sys, os
d, g = sys.argv[1], int(sys.argv[2])
tr = np.load(os.path.join(d, f"moe_trace_gpu{g}.npy")).astype(np.int64)
ref = np.median(np.load(os.path.join(d, f"stamps_gpu{g}.npy")).astype(np.int64)[:, 0])
last = {}; cnt = {}; busy = {}; w13e = {}
for w in range(tr.shape[0]):
    die = (w // 4) % 8
    for i in range(tr.shape[1]):
        a, b = int(tr[w, i, 0]), int(tr[w, i, 1])
        if not a: break
        k = (b >> 32) & 0xFF; e = (a - ref) / 1e3 + (b & 0xFFFFFFFF) / 1e3
        if k == 2:
            last.setdefault(die, {})[w] = max(last.get(die, {}).get(w, 0), e); cnt[die] = cnt.get(die, 0) + 1; busy[die] = busy.get(die, 0) + (b & 0xFFFFFFFF) / 1e3
        if k == 1: w13e[die] = max(w13e.get(die, 0), e)
for die in sorted(last):
    v = np.array(list(last[die].values()))
    print(f"die {die}: W13 last end {w13e.get(die, 0):6.1f} | W2 items {cnt[die]:5d} wave-us {busy[die]:7.0f} | waves' last W2 end p50 {np.median(v):6.1f} max {v.max():6.1f}")
