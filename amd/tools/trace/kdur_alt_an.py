# kdur_alt_an.py RUN_P0_DIR RUN_P1_DIR [...more pairs]: a paired in-server A / B (sglang_k3.py MPK_AMD_DEC_F*_ALT + MPK_ALT_PARITY): each
# forward's fused kernels (moeagg*) in order, kernel j = MoE layer j + 1 (layer 0 dense); per run and kind (KDA / MLA) the median duration
# of the ALT kernels (MoE layer id parity = the run's MPK_ALT_PARITY) and of the main ones, over all ranks. Each layer is ALT in one run
# and main in the other: ALT - main per run, and the mean over the pair (each layer's own time cancels in the mean of the two runs).
import gzip, json, sys, glob, statistics as st
def run(d, par):
    out = {"kda": {"alt": [], "main": []}, "mla": {"alt": [], "main": []}}
    for f in sorted(glob.glob(f"{d}/*TP-*.trace.json.gz")):
        ev = json.load(gzip.open(f, "rt"))["traceEvents"]
        g = {e.get("args", {}).get("correlation") for e in ev if e.get("ph") == "X" and e.get("name") == "hipGraphLaunch"}
        by = {}
        for e in ev:
            if e.get("ph") == "X" and e.get("cat") == "kernel" and e.get("args", {}).get("correlation") in g and "moeagg" in e.get("name", ""):
                by.setdefault(e["args"]["correlation"], []).append(e)
        for v in by.values():
            v = sorted(v, key=lambda e: e["ts"])
            if len(v) < 40:
                continue
            for j, e in enumerate(v):
                kind = "kda" if "kda" in e["name"] else "mla"
                lid = j + 1
                sel = (lid // 4 if kind == "mla" else lid) % 2   # (as sglang_k3.py: MLA layers alternate by lid // 4)
                out[kind]["alt" if sel == par else "main"].append(e["dur"])
    return {k: {r: (st.median(x) if x else float("nan"), len(x)) for r, x in v.items()} for k, v in out.items()}
args = sys.argv[1:]
for i in range(0, len(args), 2):
    r0, r1 = run(args[i], 0), run(args[i + 1], 1)
    for k in ("kda", "mla"):
        d0 = r0[k]["alt"][0] - r0[k]["main"][0]; d1 = r1[k]["alt"][0] - r1[k]["main"][0]
        print(f"{k}: p0 alt {r0[k]['alt'][0]:.1f} main {r0[k]['main'][0]:.1f} (alt-main {d0:+.1f}, n {r0[k]['alt'][1]}/{r0[k]['main'][1]}); "
              f"p1 alt {r1[k]['alt'][0]:.1f} main {r1[k]['main'][0]:.1f} ({d1:+.1f}) -> paired ALT - main {(d0 + d1) / 2:+.2f} us")
