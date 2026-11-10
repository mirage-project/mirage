# start_an.py PREFIX CONFIG T: per KDA layer and rank, the CUs' go stamps [0] relative to the first CU's: p50 / p90 / max, and the
# experts task start [1] and the router / front node ends if stamped; medians over layers and ranks
import glob, json, sys, torch, numpy as np
pre, cfgp, T = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = json.load(open(cfgp)); c = c.get("text_config", c)
mla1 = set(c["linear_attn_config"]["full_attn_layers"])
rows = []
for f in sorted(glob.glob(pre + "_r*.pt")):
    for key, st in torch.load(f).items():
        lid, t = (int(x) for x in key.split("_"))
        if t != T or (lid + 2) in mla1: continue
        s = st.numpy().astype(np.int64)
        g = s[:, 0][s[:, 0] > 0]; go = g.min()
        t1 = s[:, 1][s[:, 1] > 0]
        rows.append([np.percentile((g - go) / 1e3, 50), np.percentile((g - go) / 1e3, 90), (g.max() - go) / 1e3,
                     np.percentile((t1 - go) / 1e3, 10), np.percentile((t1 - go) / 1e3, 50), len(g)])
A = np.array(rows)
print(f"T {T}: {len(A)} kernels; CU go after the first CU's: p50 {np.median(A[:, 0]):.2f} p90 {np.median(A[:, 1]):.2f} max {np.median(A[:, 2]):.2f} us "
      f"(CUs stamped {np.median(A[:, 5]):.0f}); experts task start p10 {np.median(A[:, 3]):.1f} p50 {np.median(A[:, 4]):.1f}")
