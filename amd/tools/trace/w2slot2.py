# w2slot2.py TRACE_DIR: per GPU, the median W2 item duration of each slot in SLOTS, and the median start of the slot's items
import numpy as np, sys, os
d = sys.argv[1]; SL = list(range(168, 178)) + list(range(203, 214))
print("slot      " + " ".join(f"{s:5d}" for s in SL))
for g in range(8):
    tr = np.load(os.path.join(d, f"moe_trace_gpu{g}.npy")).astype(np.int64)
    ref = np.median(np.load(os.path.join(d, f"stamps_gpu{g}.npy")).astype(np.int64)[:, 0])
    it = []
    for w in range(tr.shape[0]):
        for i in range(tr.shape[1]):
            a, b = int(tr[w, i, 0]), int(tr[w, i, 1])
            if not a: break
            if (b >> 32) & 0xFF == 2: it.append(((a - ref) / 1e3, (b & 0xFFFFFFFF) / 1e3, (b >> 40) & 0xFFFFFF))
    it = np.array(it)
    row = [np.median(it[it[:, 2] == s, 1]) if (it[:, 2] == s).any() else np.nan for s in SL]
    st = [np.median(it[it[:, 2] == s, 0]) if (it[:, 2] == s).any() else np.nan for s in SL]
    print(f"GPU {g} dur " + " ".join(f"{x:5.1f}" for x in row))
    print(f"GPU {g} at  " + " ".join(f"{x:5.0f}" for x in st))
