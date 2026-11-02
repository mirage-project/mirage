# table_an.py TRACE_DIR GPU: moe item trace kind 10 (experts task start -> CSR table built): start / end percentiles (us after the go)
import numpy as np, sys, os
d, g = sys.argv[1], int(sys.argv[2])
tr = np.load(os.path.join(d, f"moe_trace_gpu{g}.npy")).astype(np.int64)
ref = np.median(np.load(os.path.join(d, f"stamps_gpu{g}.npy")).astype(np.int64)[:, 0])
st, en = [], []
for w in range(0, tr.shape[0], 4):
    for i in range(tr.shape[1]):
        a, b = int(tr[w, i, 0]), int(tr[w, i, 1])
        if not a: break
        if (b >> 32) & 0xFF == 10: st.append((a - ref) / 1e3); en.append((a - ref + (b & 0xFFFFFFFF)) / 1e3)
st, en = np.array(st), np.array(en)
q = lambda v: " ".join(f"{np.percentile(v, p):6.1f}" for p in (10, 50, 90))
print(f"GPU {g}: CUs {len(st)}; task start p10/50/90 {q(st)}; table built p10/50/90 {q(en)}; build time p50 {np.median(en - st):.1f}")
