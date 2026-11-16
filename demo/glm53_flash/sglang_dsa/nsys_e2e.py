"""nsys_e2e.py RUN_DIR [RUN_DIR ...] [scen ...] [--dev=N] | nsys_e2e.py --one DB [--dev=N]: the end-to-end table of
sgl_chain.py's nsys runs prof_{base,all3ppf}_{scen}_p{256,4000} (sgl_spec_bench.py --profile under nsys: <name>.sqlite,
<name>.log), each run from the first RUN_DIR that has it (e.g. the two halves' directories of one box: ours/ and base/).
--dev: the trace's deviceId of the GPU measured; default the lowest deviceId in each trace (a run on CUDA_VISIBLE_DEVICES
4,5,6,7 need not record device 0). Per run the
period of the forward's CUDA graph on one GPU: start to start of consecutive replays (a decode step), median and mean over
the replays in the trace (the prefill is eager).
The forward: with --cuda-graph-trace=graph (CUPTI_ACTIVITY_KIND_GRAPH_TRACE, one record per replay: sgl_chain's runs)
the graph with the longest median replay, with =node (every kernel of a replay, the kernels share its launch's correlation
id) the one with the most kernels per replay. Node tracing perturbs the timeline (the per-node records cost the PDL overlap
of the static kernels), so the e2e runs trace whole graphs. us per output token per request = the period. Speedup = SGLang
(base) / static (all3ppf) - 1.
--one DB: every graph of one trace (replays, kernels or duration per replay, period), to check which one is the forward."""
import collections
import os
import sqlite3
import statistics as st
import sys

LABEL = {"off_bs1": "bs1 解码", "off_bs2": "bs2 解码", "off_bs4": "bs4 解码", "off_bs8": "bs8 解码", "off_bs16": "bs16 解码",
         "off_bs32": "bs32 解码", "off_bs64": "bs64 解码"}
LENS = (256, 4000)


def graphs(db, dev):
    """graph id -> [(start us, size)] per replay; size: the replay's duration (us, whole-graph records) or its kernel count
    (node records: the kernels of one replay share its launch's correlation id). dev None: the lowest deviceId recorded."""
    c = sqlite3.connect(db)
    if dev is None:
        devs = [r[0] for t in ("CUPTI_ACTIVITY_KIND_GRAPH_TRACE", "CUPTI_ACTIVITY_KIND_KERNEL")
                if c.execute("select count(*) from sqlite_master where name = ?", (t,)).fetchone()[0]
                for r in c.execute(f"select min(deviceId) from {t}") if r[0] is not None]
        if not devs:
            raise ValueError(f"{db}: no kernel or graph records")
        dev = min(devs)
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


def main():
    dev = next((int(a.split("=")[1]) for a in sys.argv[1:] if a.startswith("--dev=")), None)
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
    dirs = [a for a in args if os.path.isdir(a)]
    scens = [a for a in args if not os.path.isdir(a)] or list(LABEL)
    res = {}
    for scen in scens:
        for p in LENS:
            for cfg in ("base", "all3ppf"):
                name = f"prof_{cfg}_{scen}_p{p}"
                d = next((d for d in dirs if os.path.exists(os.path.join(d, f"{name}.sqlite"))), None)
                if d is None:
                    continue
                db = os.path.join(d, f"{name}.sqlite")
                try:
                    per, tg, nk = period(db, dev)
                except Exception as e:
                    print(f"{name}: {e}", file=sys.stderr)
                    continue
                res[(scen, p, cfg)] = (st.median(per), st.mean(per), len(per), tg, nk)
    f1 = lambda v: "—" if v is None else f"{v:.1f}"
    print("| 场景 | 提示 | SGLang µs/token | 静态层 µs/token | 提速 | 周期中位数 SGLang / 静态层 (µs) | 周期均值 SGLang / 静态层 (µs) "
          "| 周期数 |")
    print("|---|---|---|---|---|---|---|---|")
    for scen in scens:
        for p in LENS:
            b, s = res.get((scen, p, "base")), res.get((scen, p, "all3ppf"))
            if not b and not s:
                continue
            g = lambda r, i: r[i] if r else None
            sp = f"**{(b[0] / s[0] - 1) * 100:+.1f}%**" if b and s else "—"
            print(f"| {LABEL.get(scen, scen)} | {p} | {f1(g(b, 0))} | {f1(g(s, 0))} | {sp} | {f1(g(b, 0))} / {f1(g(s, 0))} | "
                  f"{f1(g(b, 1))} / {f1(g(s, 1))} | {g(b, 2) or '—'} / {g(s, 2) or '—'} |")
    print()
    for (scen, p, cfg), (med, mean, n, tg, nk) in res.items():
        print(f"prof_{cfg}_{scen}_p{p}: graph {tg} (size {nk:.1f}: us or kernels per replay), {n} periods, median {med:.1f} "
              f"mean {mean:.1f} us")


if __name__ == "__main__":
    main()
