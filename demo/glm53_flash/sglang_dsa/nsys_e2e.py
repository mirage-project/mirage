"""nsys_e2e.py RUN_DIR [scen ...] [--dev=N] | nsys_e2e.py --one DB [--dev=N]: the end-to-end table of sgl_chain.py's nsys runs
prof_{base,all3ppf}_{scen}_p{256,4000} (sgl_spec_bench.py --profile under nsys: <name>.sqlite, <name>.log). Per run the
period of the forward's CUDA graph on one GPU: start to start of consecutive replays (a decode step, or an MTP cycle = draft +
verify: each graph of a cycle replays once per cycle), median and mean over the replays in the trace (the prefill is eager).
The forward: with --cuda-graph-trace=graph (CUPTI_ACTIVITY_KIND_GRAPH_TRACE, one record per replay: sgl_chain's runs)
the graph with the longest median replay, with =node (every kernel of a replay, the kernels share its launch's correlation
id) the one with the most kernels per replay. Node tracing perturbs the timeline (MTP bs 1, step1_5: SGLang's cycle 8087.6
vs the bench's 9765.7 us, the static layers' 7170.0 vs 5548.3 us: the per-node records cost the PDL overlap of the static
kernels), so the e2e runs trace whole graphs. us per output token per request = the period (decode) or the period / the
run's accept length (MTP: its bench line at the same prompt length). Speedup = SGLang (base) / static (all3ppf) - 1.
--one DB: every graph of one trace (replays, kernels or duration per replay, period), to check which one is the forward."""
import collections
import os
import re
import sqlite3
import statistics as st
import sys

LABEL = {"spec516_bs1": "bs1 MTP 5/1/6", "spec516_bs2": "bs2 MTP 5/1/6", "off_bs8": "bs8 解码", "off_bs16": "bs16 解码",
         "spec516_bs4": "bs4 MTP 5/1/6", "off_bs32": "bs32 解码"}
LENS = (256, 4000)


def graphs(db, dev):
    """graph id -> [(start us, size)] per replay; size: the replay's duration (us, whole-graph records) or its kernel count
    (node records: the kernels of one replay share its launch's correlation id)."""
    c = sqlite3.connect(db)
    if c.execute("select count(*) from sqlite_master where name = 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE'").fetchone()[0]:
        by = collections.defaultdict(list)
        for s, e, g in c.execute("select start, end, graphId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE where deviceId = ? order by "
                                 "start", (dev,)):
            by[g].append((s / 1e3, (e - s) / 1e3))
        if by:
            return by
    rep = collections.OrderedDict()
    for s, g, cid in c.execute("select start, coalesce(graphId, 0), correlationId from CUPTI_ACTIVITY_KIND_KERNEL where "
                               "deviceId = ? order by start", (dev,)):
        if g:
            r = rep.setdefault((g, cid), [s / 1e3, 0])
            r[1] += 1
    by = collections.defaultdict(list)
    for (g, _), (s, n) in rep.items():
        by[g].append((s, n))
    return by


def period(db, dev):
    by = graphs(db, dev)
    tg = max((g for g in by if len(by[g]) >= 5), key=lambda g: st.median(n for _, n in by[g]))
    starts = sorted(s for s, _ in by[tg])
    per = [b - a for a, b in zip(starts, starts[1:])]
    return per, tg, st.median(n for _, n in by[tg])


def accept(log, p):
    for line in open(log, errors="replace"):
        m = re.search(r"prompt (\d+) tokens \((\d+) asked\).*accept length mean ([\d.]+)", line)
        if m and int(m.group(2)) == p:
            return float(m.group(3))
    return None


def main():
    dev = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--dev=")), 0))
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--one" in sys.argv:
        by = graphs(args[0], dev)
        for g in sorted(by, key=lambda g: -len(by[g])):
            s = sorted(x for x, _ in by[g])
            per = [b - a for a, b in zip(s, s[1:])]
            print(f"graph {g}: {len(by[g])} replays, size/replay {st.median(n for _, n in by[g]):.1f}, period median "
                  f"{st.median(per) if per else float('nan'):.1f} us")
        per, tg, nk = period(args[0], dev)
        print(f"forward graph {tg} (size {nk:.1f}): period median {st.median(per):.1f} mean {st.mean(per):.1f} min "
              f"{min(per):.1f} max {max(per):.1f} us over {len(per)}")
        return
    d = args[0]
    scens = args[1:] or list(LABEL)
    res = {}
    for scen in scens:
        for p in LENS:
            for cfg in ("base", "all3ppf"):
                name = f"prof_{cfg}_{scen}_p{p}"
                db, log = os.path.join(d, f"{name}.sqlite"), os.path.join(d, f"{name}.log")
                if not os.path.exists(db):
                    continue
                try:
                    per, tg, nk = period(db, dev)
                except Exception as e:
                    print(f"{name}: {e}", file=sys.stderr)
                    continue
                acc = accept(log, p) if scen.startswith("spec") else 1.0
                med = st.median(per)
                res[(scen, p, cfg)] = (med, st.mean(per), len(per), acc, med / acc if acc else None, tg, nk)
    f1 = lambda v: "—" if v is None else f"{v:.1f}"
    print("| 场景 | 提示 | SGLang µs/token | 静态层 µs/token | 提速 | 周期中位数 SGLang / 静态层 (µs) | 周期均值 SGLang / 静态层 (µs) "
          "| 接受长度 SGLang / 静态层 | 周期数 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for scen in scens:
        for p in LENS:
            b, s = res.get((scen, p, "base")), res.get((scen, p, "all3ppf"))
            if not b and not s:
                continue
            g = lambda r, i: r[i] if r else None
            sp = f"**{(b[4] / s[4] - 1) * 100:+.1f}%**" if b and s and b[4] and s[4] else "—"
            ac = (f"{g(b, 3):.2f} / {g(s, 3):.2f}" if scen.startswith("spec") and b and s and b[3] and s[3] else "—")
            print(f"| {LABEL.get(scen, scen)} | {p} | {f1(g(b, 4))} | {f1(g(s, 4))} | {sp} | {f1(g(b, 0))} / {f1(g(s, 0))} | "
                  f"{f1(g(b, 1))} / {f1(g(s, 1))} | {ac} | {g(b, 2) or '—'} / {g(s, 2) or '—'} |")
    print()
    for (scen, p, cfg), (med, mean, n, acc, tok, tg, nk) in res.items():
        print(f"prof_{cfg}_{scen}_p{p}: graph {tg} (size {nk:.1f}: us or kernels per replay), {n} periods, median {med:.1f} mean {mean:.1f} us"
              + (f", accept {acc:.2f}" if scen.startswith("spec") and acc else ""))


if __name__ == "__main__":
    main()
