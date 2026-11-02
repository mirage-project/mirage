# srvstamp_an.py PREFIX CONFIG: server stamp dumps (sglang_k3.py _stamp_dumper) -> per kernel kind (KDA / MLA), per node: its last end
# after that GPU's kernel start (slot 0 min over CUs), median over the layers, per rank and the max over ranks; plus kernel span
import glob, json, sys, torch, numpy as np
pre, cfgp = sys.argv[1], sys.argv[2]
c = json.load(open(cfgp)); c = c.get("text_config", c)
mla1 = set(c["linear_attn_config"]["full_attn_layers"])   # 1-based
NODES = {"kda": [36, 37, 39, 48, 68, 69, 70, 71], "mla": [36, 39, 48, 76, 78, 82, 84, 85]}
if len(sys.argv) > 4: NODES["kda"] = [int(x) for x in sys.argv[4].split(",")]
if len(sys.argv) > 5: NODES["mla"] = [int(x) for x in sys.argv[5].split(",")]
res = {"kda": {}, "mla": {}}
for f in sorted(glob.glob(pre + "_r*.pt")):
    rk = int(f.rsplit("_r", 1)[1].split(".")[0])
    d = torch.load(f)
    for key, st in d.items():
        lid, t = (int(x) for x in key.split("_"))
        if t != int(sys.argv[3] if len(sys.argv) > 3 else 16):
            continue
        kind = "mla" if (lid + 2) in mla1 else "kda"
        s = st.numpy().astype(np.int64)
        if s.shape[1] < 16:
            continue
        go = s[:, 0][s[:, 0] > 0].min()
        endall = s[:, 3][s[:, 3] > 0].max() if (s[:, 3] > 0).any() else 0
        row = [((s[:, 8 + i][s[:, 8 + i] > 0].max() - go) / 1e3) if (s[:, 8 + i] > 0).any() else np.nan for i in range(8)]
        row.append((endall - go) / 1e3 if endall else np.nan)
        res[kind].setdefault(rk, []).append(row)
for kind in [k for k in ("kda", "mla") if res[k]]:
    print(f"== {kind} layers: node last end (us after the GPU's kernel start), median over layers; per rank")
    hdr = NODES[kind] + ["end"]
    per = {rk: np.nanmedian(np.array(v), 0) for rk, v in sorted(res[kind].items())}
    for i, n in enumerate(hdr):
        vals = [per[rk][i] for rk in per]
        print(f"  node {str(n):>4}: " + " ".join(f"{v:6.1f}" for v in vals) + f" | max {max(vals):6.1f}  (layers per rank {len(res[kind][0]) if 0 in res[kind] else '?'})")
