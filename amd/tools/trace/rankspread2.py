# rankspread2.py PREFIX CONFIG T: per KDA layer, the spread over ranks (max - min, own kernel time) of each stamped node's end:
# a node after a cross-GPU exchange (sum_rmsnorm 38) shows the ranks' start skew; experts (36) shows skew + real spread
import glob, json, sys, torch, numpy as np
pre, cfgp, T = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = json.load(open(cfgp)); c = c.get("text_config", c)
mla1 = set(c["linear_attn_config"]["full_attn_layers"])
per = {}
for f in sorted(glob.glob(pre + "_r*.pt")):
    rk = int(f.rsplit("_r", 1)[1].split(".")[0])
    for key, st in torch.load(f).items():
        lid, t = (int(x) for x in key.split("_"))
        if t != T or (lid + 2) in mla1: continue
        s = st.numpy().astype(np.int64)
        if s.shape[1] < 16: continue
        go = s[:, 0][s[:, 0] > 0].min()
        per.setdefault(lid, {})[rk] = [((s[:, 8 + i][s[:, 8 + i] > 0].max() - go) / 1e3) if (s[:, 8 + i] > 0).any() else np.nan for i in range(8)]
names = ["36 experts", "37 R send", "38 sum_rmsnorm", "40 latent_up", "49 agg1", "69 proj", "71 rec", "72 o_proj"]
A = np.array([[d[r] for r in sorted(d)] for d in per.values() if len(d) == 8])   # [layers][ranks][nodes]
for i, n in enumerate(names):
    sp = A[:, :, i].max(1) - A[:, :, i].min(1)
    print(f"T {T} {n:15s}: max - min over ranks p10 / 50 / 90 {np.percentile(sp, 10):5.1f} {np.median(sp):5.1f} {np.percentile(sp, 90):5.1f}")
