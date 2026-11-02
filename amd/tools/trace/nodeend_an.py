"""STATIC_NODE_END_STAMPS (a normal build) + --dump-stamps: <out>/stamps_timed.npy [launches][GPUs][CUs][16]; per listed
node i (slot 8 + i): its last end over the CUs of each GPU, us after that GPU's first go (slot 0), median over the launches;
printed per GPU and the max over the GPUs.
  python3 amd/tools/trace/nodeend_an.py <out>/stamps_timed.npy n0,n1,..."""
import sys, numpy as np
st = np.load(sys.argv[1]).astype(np.int64)
nodes = [int(x) for x in sys.argv[2].split(",")]
L, G, C, S = st.shape
for i, n in enumerate(nodes):
    per = np.zeros((L, G))
    for l in range(L):
        for g in range(G):
            s = st[l, g]
            go = s[:, 0][s[:, 0] > 0].min()
            e = s[:, 8 + i][s[:, 8 + i] > 0]
            per[l, g] = (e.max() - go) / 1e3 if len(e) else np.nan
    med = np.nanmedian(per, 0)
    print(f"node {n}: last end (us after the go, median of {L} launches) per GPU " + " ".join(f"{v:.1f}" for v in med) +
          f" | max over GPUs {np.nanmax(med):.1f}")
