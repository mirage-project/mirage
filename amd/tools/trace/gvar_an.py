# gvar_an.py TRACE_DIR: per GPU, moe_experts' item trace (kind 1 = W13 item, 2 = W2 item; per wave [entries][2] = t0 ns, dur | kind<<32)
# relative to the GPU's median go (stamps col 0): items, median / p90 duration, the phases' first start / last end, busy fraction
import numpy as np, sys, os
d = sys.argv[1]
for g in range(8):
    p = os.path.join(d, f"moe_trace_gpu{g}.npy")
    if not os.path.exists(p): continue
    tr = np.load(p).astype(np.int64)
    ref = np.median(np.load(os.path.join(d, f"stamps_gpu{g}.npy")).astype(np.int64)[:, 0])
    out = []
    for kind, nm in ((1, "W13"), (2, "W2")):
        t0, du = [], []
        for w in range(tr.shape[0]):
            for i in range(tr.shape[1]):
                a, b = int(tr[w, i, 0]), int(tr[w, i, 1])
                if not a: break
                if (b >> 32) & 0xFF == kind:
                    t0.append((a - ref) / 1e3); du.append((b & 0xFFFFFFFF) / 1e3)
        t0, du = np.array(t0), np.array(du)
        if not len(t0): out.append(f"{nm} none"); continue
        out.append(f"{nm} n {len(t0):5d} dur p50 {np.median(du):5.2f} p90 {np.percentile(du, 90):5.2f} sum {du.sum():8.0f} | first {t0.min():6.1f} p50 start {np.median(t0):6.1f} last end {(t0 + du).max():6.1f}")
    print(f"GPU {g}: " + " || ".join(out))
