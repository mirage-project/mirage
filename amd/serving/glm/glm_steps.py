"""Where a GLM-5.3-Flash decode step's time goes, from one rank's torch-profiler trace of an SGLang ROCm server run
(srv_run_glm.sh): the decode forward is one HIP graph launch (hipGraphLaunch); its kernels carry the launch's correlation id.
Per forward: the span (first kernel's start -> last kernel's end), the period (start -> next forward's start), the kernels'
busy time; per kernel name its time per forward (median over the forwards); the names grouped by GROUPS (regexes, first
match). Medians over the forwards of the trace.
    python glm_steps.py TRACE [--top N] [--json OUT]"""
import gzip
import json
import re
import statistics as st
import sys
from collections import defaultdict

# kernel-name groups (first match); checked against the names a trace shows (--top)
GROUPS = [
    ("allreduce", r"all_?reduce|cross_device|nccl|rccl|custom_ar|one_shot|two_shot"),
    ("moe", r"moe|fused_experts|topk|expert|ck_moe|fmoe|grouped_topk|biased_grouped"),
    ("kda", r"kda|chunk_gated_delta|fused_recurrent|delta_rule|causal_conv|short_conv|gated_delta"),
    ("dsa_mla", r"mla|sparse|indexer|dsa|tilelang|fp8_index|flash_mla|paged"),
    ("mhc", r"sinkhorn|hc_|mhc|hyper"),
    ("norm", r"rms|norm"),
    ("gemm", r"gemm|matmul|Cijk|wvSplitK|hipblaslt|mfma|_gemm|scaled_mm|ck_tile"),
    ("quant", r"quant|fp8"),
    ("elementwise", r"elementwise|vectorized|copy|fill|cat|act_and_mul|silu|add"),
]


def load(path):
    ev = json.load(gzip.open(path, "rt") if path.endswith(".gz") else open(path))["traceEvents"]
    graphs = {e.get("args", {}).get("correlation") for e in ev if e.get("ph") == "X" and e.get("name") == "hipGraphLaunch"}
    ks = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"]
    return ks, graphs


def forwards(ks, graphs):
    by = defaultdict(list)
    for e in ks:
        c = e.get("args", {}).get("correlation")
        if c in graphs:
            by[c].append(e)
    fw = [sorted(v, key=lambda e: e["ts"]) for v in by.values()]
    if not fw:
        return []
    n = max(len(f) for f in fw)
    fw = [f for f in fw if len(f) >= 0.9 * n]   # the decode forward graphs (the most kernels)
    return sorted(fw, key=lambda f: f[0]["ts"])


def group(name):
    for g, rx in GROUPS:
        if re.search(rx, name, re.I):
            return g
    return "other"


def main():
    path = sys.argv[1]
    top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 30
    ks, graphs = load(path)
    fw = forwards(ks, graphs)
    if len(fw) < 3:
        raise SystemExit(f"{path}: {len(fw)} graph forwards")
    span = [f[-1]["ts"] + f[-1]["dur"] - f[0]["ts"] for f in fw]
    busy = [sum(e["dur"] for e in f) for f in fw]
    period = [b[0]["ts"] - a[0]["ts"] for a, b in zip(fw, fw[1:])]
    print(f"{path}: {len(fw)} forwards, {len(fw[0])} kernels each; span median {st.median(span):.1f} us (min {min(span):.1f}, "
          f"max {max(span):.1f}); kernel busy {st.median(busy):.1f}; period median {st.median(period):.1f}")
    per = defaultdict(list)
    cnt = defaultdict(list)
    for f in fw:
        t, c = defaultdict(float), defaultdict(int)
        for e in f:
            t[e["name"]] += e["dur"]
            c[e["name"]] += 1
        for k in set(t) | set(per):
            per[k].append(t.get(k, 0.0))
            cnt[k].append(c.get(k, 0))
    med = {k: st.median(v) for k, v in per.items()}
    groups = defaultdict(float)
    for k, v in med.items():
        groups[group(k)] += v
    print("groups (us per forward, median of each name summed): " +
          ", ".join(f"{g} {v:.1f}" for g, v in sorted(groups.items(), key=lambda x: -x[1])))
    for k, v in sorted(med.items(), key=lambda x: -x[1])[:top]:
        print(f"  {v:9.1f} us  x{st.median(cnt[k]):5.0f}  [{group(k)}]  {k[:150]}")
    if "--json" in sys.argv:
        json.dump({"span": st.median(span), "period": st.median(period), "busy": st.median(busy), "forwards": len(fw),
                   "groups": groups, "names": med}, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)


if __name__ == "__main__":
    main()
