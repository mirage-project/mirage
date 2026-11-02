"""Where a decode step's time goes, ours vs SGLang alone, from rank-0 torch-profiler traces of the MI355X server runs (B300's
mirage_kda scripts/b300_session/step_split.py with the ROCm kernel names): per target forward (delimited by the aggregation
kernels, SGLang's HIP `_agg_kernel` for aggregation 1 and 2) the windows, each from the end of one marker to the end of the next:
  kda    aggregation 1's end -> the end of the KDA attention part (SGLang: its aggregation 2; ours: mpk_kda_kernel)
  after  the attention part's end -> the next layer's aggregation 1 end (the MoE, the next aggregation 1)
  mla    the same for MLA layers (ours: mpk_mla_kernel)
plus the forward's span (first marker's start -> last marker's end) and the step period; medians over the forwards.
    python step_split.py TRACE_A TRACE_B CONFIG_JSON [--per-layer]     (A: ours, B: SGLang alone; either may be either)"""
import gzip
import json
import statistics as st
import sys

GAP_US = 300.0
MARK = "_agg_kernel"
OURS_ATTN = ("mpk_kda_kernel", "mpk_kdad_kernel", "mpk_mla_kernel", "mpk_kdamoe_kernel", "mpk_mlamoe_kernel")   # (fused: the part window holds the MoE too)
# attention + MoE + the next layer's aggregation 1 in one kernel (MPK_K3_KDAMOEAGG / _MLAMOEAGG): a marker (it ends with the
# next aggregation 1); the layer's attention window is empty (0) and the "after" window is the kernel
FUSED_AGG = ("mpk_kdamoeagg_kernel", "mpk_mlamoeagg_kernel")
# a MoE layer + the next layer's aggregation 1 + that layer's KDA part in one kernel (MPK_K3_MOEAGGKDA): the next layer's
# marker, its attention window empty (0); the kernel itself is in layer l - 1's "after" window
MOE_ATTN = ("mpk_moeaggkda_kernel", "mpk_moeaggkdad_kernel", "mpk_moeaggmla_kernel")


def kernels(trace):
    """the kernels in time order, and the correlation ids of the graph launches (hipGraphLaunch: one per captured forward)"""
    ev = json.load(gzip.open(trace, "rt") if trace.endswith(".gz") else open(trace))["traceEvents"]
    graphs = {e.get("args", {}).get("correlation") for e in ev if e.get("ph") == "X" and e.get("name") == "hipGraphLaunch"}
    return sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"], key=lambda e: e["ts"]), graphs


def split(trace, kda, N):
    k, graphs = kernels(trace)
    end = lambda e: e["ts"] + e["dur"]
    # the markers: SGLang's aggregation-1 kernels, and our MoE kernel where it did the next layer's aggregation 1 itself
    # (MPK_K3_MOEAGG: no _agg_kernel after it)
    ar = [i for i, e in enumerate(k) if MARK in e["name"] or e["name"].startswith(FUSED_AGG + MOE_ATTN) or
          (e["name"].startswith(("mpk_moe_kernel", "mpk_moeagg_kernel")) and i + 1 < len(k) and MARK not in k[i + 1]["name"])]
    if not ar:
        raise SystemExit(f"{trace}: no {MARK} kernels")
    fw, cur = [], [ar[0]]
    corr = lambda i: k[i].get("args", {}).get("correlation")
    if graphs and any(corr(i) in graphs for i in ar):
        # (plain decode: one graph launch per forward, the steps closer than GAP_US at small batches) the markers by the
        # graph launch their kernels came from
        by = {}
        for i in ar:
            if corr(i) in graphs:
                by.setdefault(corr(i), []).append(i)
        fw = sorted(by.values(), key=lambda f: f[0])
    else:
        for a, b in zip(ar, ar[1:]):
            if k[b]["ts"] - end(k[a]) > GAP_US:
                fw.append(cur)
                cur = []
            cur.append(b)
        fw.append(cur)
    rows, per_layer = [], []
    for f in fw:
        if any("chunk_kda" in k[i]["name"] or "chunk_fwd" in k[i]["name"] for i in range(f[0], f[-1])):
            continue   # a prefill forward
        # per layer l >= 1: its aggregation 1 marker, the end of its attention part (our kernel, else SGLang's aggregation 2)
        j, ok, agg1, attn_end = 0, True, {}, {}
        for l in range(1, N):
            if j >= len(f):
                ok = False
                break
            agg1[l] = f[j]
            nxt = f[j + 1] if j + 1 < len(f) else len(k)
            lk = next((i for i in range(f[j] + 1, nxt) if any(o in k[i]["name"] for o in OURS_ATTN)), None)
            if k[f[j]]["name"].startswith(MOE_ATTN):   # (this layer's aggregation 1 and KDA part inside the marker)
                attn_end[l] = end(k[f[j]])
                j += 1
            elif lk is None and nxt < len(k) and k[nxt]["name"].startswith(FUSED_AGG):
                attn_end[l] = k[nxt]["ts"]   # (the fused kernel: the attention part inside it)
                j += 1
            elif lk is not None:
                attn_end[l] = end(k[lk])
                j += 1
            else:
                if j + 1 >= len(f):
                    ok = False
                    break
                attn_end[l] = end(k[f[j + 1]])
                j += 2
        if not ok:
            continue
        s = {"kda": 0.0, "after_kda": 0.0, "mla": 0.0, "after_mla": 0.0}
        lay = {}
        for l in range(1, N - 1):
            key = "kda" if l in kda else "mla"
            a_ = attn_end[l] - end(k[agg1[l]])
            b_ = end(k[agg1[l + 1]]) - attn_end[l]
            s[key] += a_
            s["after_" + key] += b_
            lay[l] = (a_, b_)
        s["span"] = end(k[f[-1]]) - k[f[0]]["ts"]
        s["start"] = k[f[0]]["ts"]
        rows.append(s)
        per_layer.append(lay)
    if not rows:
        raise SystemExit(f"{trace}: no decode forward of {N} layers found ({len(fw)} marker groups)")
    per = [b["start"] - a["start"] for a, b in zip(rows, rows[1:]) if b["start"] - a["start"] < 200000]
    out = {key: st.median(r[key] for r in rows) for key in ("kda", "after_kda", "mla", "after_mla", "span")}
    out["step"] = st.median(per) if per else float("nan")
    out["n"] = len(rows)
    out["layers"] = {l: tuple(st.median(p[l][i] for p in per_layer) for i in range(2)) for l in per_layer[0]}
    return out


def main():
    cfg = json.load(open(sys.argv[3]))
    cfg = cfg.get("text_config", cfg)
    N = cfg["num_hidden_layers"]
    kda = {i for i in range(N) if (i + 1) in set((cfg.get("linear_attn_config") or {}).get("kda_layers") or [])}
    a = split(sys.argv[1], kda, N)
    b = split(sys.argv[2], kda, N)
    nk, nm = len([l for l in range(1, N - 1) if l in kda]), len([l for l in range(1, N - 1) if l not in kda])
    print(f"{'':<34}{'A':>10}{'B':>10}{'A - B':>10}   (us, medians over {a['n']} / {b['n']} forwards)")
    for key, name in (("kda", f"KDA parts ({nk} layers)"), ("after_kda", "after each KDA part (MoE ..)"),
                      ("mla", f"MLA parts ({nm} layers)"), ("after_mla", "after each MLA part (MoE ..)"),
                      ("span", "target forward span"), ("step", "step period")):
        print(f"{name:<34}{a[key]:>10.1f}{b[key]:>10.1f}{a[key] - b[key]:>+10.1f}")
    print(f"{'outside the target forward':<34}{a['step'] - a['span']:>10.1f}{b['step'] - b['span']:>10.1f}"
          f"{(a['step'] - a['span']) - (b['step'] - b['span']):>+10.1f}")
    print(f"per layer (median): KDA part A {st.median(a['layers'][l][0] for l in a['layers'] if l in kda):.1f} "
          f"B {st.median(b['layers'][l][0] for l in b['layers'] if l in kda):.1f}; after KDA A "
          f"{st.median(a['layers'][l][1] for l in a['layers'] if l in kda):.1f} B "
          f"{st.median(b['layers'][l][1] for l in b['layers'] if l in kda):.1f}; MLA part A "
          f"{st.median(a['layers'][l][0] for l in a['layers'] if l not in kda):.1f} B "
          f"{st.median(b['layers'][l][0] for l in b['layers'] if l not in kda):.1f}")
    if "--per-layer" in sys.argv:
        for l in sorted(a["layers"]):
            print(f"  layer {l:3d} {'kda' if l in kda else 'mla'}: A {a['layers'][l][0]:7.1f} {a['layers'][l][1]:7.1f} | "
                  f"B {b['layers'][l][0]:7.1f} {b['layers'][l][1]:7.1f}")


if __name__ == "__main__":
    main()
