# phase_an.py PREFIX CONFIG T: server stamps with STATIC_CSR_PHASE_STAMPS - per KDA layer and rank (own kernel time, us after the go):
# experts task start [1] p50 / max, W13 loop end [4] p50 / max, first W2 load [5] p50, W2 loop end [6] p50 / max, task end [2] max;
# then per layer: the slowest rank (by [2] max) vs the median rank, the difference of each phase mark
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
        go = s[:, 0][s[:, 0] > 0].min()
        def col(i, f):
            v = s[:, i][s[:, i] > 0]
            return f((v - go) / 1e3) if len(v) else np.nan
        per.setdefault(lid, {})[rk] = [col(1, np.median), col(4, np.median), col(4, np.max), col(5, np.median), col(6, np.median),
                                       col(6, np.max), col(2, np.max)]
nm = ["task start p50", "W13 end p50", "W13 end max", "W2 first p50", "W2 end p50", "W2 end max", "task end max"]
L = [np.array([d[r] for r in sorted(d)]) for d in per.values() if len(d) == 8]
A = np.array(L)   # [layers][ranks][7]
print(f"T {T}: {len(L)} layers; medians over layers and ranks: " + ", ".join(f"{n} {np.nanmedian(A[:, :, i]):.1f}" for i, n in enumerate(nm)))
slow = A[np.arange(len(L)), A[:, :, 6].argmax(1)]          # the slowest rank's row per layer
med = np.median(A, axis=1)                                  # the median over ranks
d = slow - med
print("slowest rank - median rank, median over layers: " + ", ".join(f"{n} {np.nanmedian(d[:, i]):+.1f}" for i, n in enumerate(nm)))
print("                            p90 over layers:    " + ", ".join(f"{n} {np.nanpercentile(d[:, i], 90):+.1f}" for i, n in enumerate(nm)))
