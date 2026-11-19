"""pair_ab.py DIR REF [--min-replays N]: the paired comparison of local/e2e_ab.sh's runs (the user 11-18: a local in-situ check
before any whole-model run, calibrated on C16 and S2-1).

DIR holds runs NAME = <arm>_r<round>: NAME.sqlite (nsys, --cuda-graph-trace=graph) and NAME.log (dsv41_spec_bench.py with
DSV41_PRINT_ALL=1: its "profiled hashes:" line, every row's output of the traced generate). Per run, GPU 0's decode graph (the
graph with the most replays there): each replay's time on the GPU and the gap before the next replay (the sampler and the
host's work run there). Two runs pair only when their profiled hashes are equal (the same tokens, so the same experts and
positions step by step) and they traced as many replays; a pair's difference is taken replay by replay (index i of one
against index i of the other) and summarized by its median and p10 / p90.

Printed: per run its medians; per arm against REF every pair's median difference; the noise floor = the pairs of one arm with
itself (REF r1 against REF r2, ..); the verdict per arm: the median over its pairs, called a difference only when it is more
than twice the largest same-arm median (or 2 us, whichever is larger)."""
import argparse
import glob
import os
import re
import sqlite3
import statistics as st


def replays(path: str):
    """GPU 0's decode graph: (each replay's duration us, each gap to the next replay us)."""
    c = sqlite3.connect(path)
    rows = list(c.execute("select start, end, deviceId, graphId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE order by start"))
    per = {}
    for s, e, d, g in rows:
        per.setdefault((d, g), []).append((s, e))
    keys = [k for k in per if k[0] == min(k[0] for k in per)]
    k = max(keys, key=lambda k: len(per[k]))
    v = per[k]
    return [(e - s) / 1e3 for s, e in v], [(v[i + 1][0] - v[i][1]) / 1e3 for i in range(len(v) - 1)]


def hashes(log: str):
    for line in open(log, errors="replace"):
        if line.startswith("profiled hashes:"):
            return line.split(":", 1)[1].split()
    return None


def q(z, f):
    z = sorted(z)
    return z[min(len(z) - 1, int(len(z) * f))]


def pair(a, b):
    """(median, p10, p90) of a's replays minus b's, index by index; None when they cannot pair."""
    if a["hash"] is None or a["hash"] != b["hash"] or len(a["dur"]) != len(b["dur"]):
        return None
    d = [x - y for x, y in zip(a["dur"], b["dur"])]
    g = [x - y for x, y in zip(a["gap"], b["gap"])]
    return st.median(d), q(d, 0.1), q(d, 0.9), st.median(g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("ref")
    ap.add_argument("--min-replays", type=int, default=40)
    a = ap.parse_args()
    runs = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "*_r[0-9]*.sqlite"))):
        name = os.path.basename(f)[:-7]
        m = re.match(r"(.+)_r(\d+)$", name)
        log = os.path.join(a.dir, name + ".log")
        if not m or not os.path.exists(log):
            continue
        fo = os.path.join(a.dir, name + ".foreign")   # (local/e2e_ab.sh's watcher: other users' processes on our GPUs)
        if os.path.exists(fo) and os.path.getsize(fo) > 0:
            print(f"   {name}: another user's process was on our GPUs during the run: left out")
            continue
        dur, gap = replays(f)
        if len(dur) < a.min_replays:
            print(f"   {name}: only {len(dur)} replays: left out")
            continue
        runs[name] = {"arm": m.group(1), "round": int(m.group(2)), "dur": dur, "gap": gap, "hash": hashes(log)}
    if not runs:
        raise SystemExit("no runs")
    print("== runs (GPU 0's decode graph): replays, median time in the graph, median gap between replays, profiled hashes")
    for n, r in sorted(runs.items(), key=lambda kv: (kv[1]["round"], kv[1]["arm"])):
        print(f"   {n:<28} {len(r['dur']):3d}  {st.median(r['dur']):8.1f}  {st.median(r['gap']):6.1f}  "
              f"{' '.join(r['hash'] or ['(none)'])}")
    arms = sorted({r["arm"] for r in runs.values()})
    if a.ref not in arms:
        raise SystemExit(f"no runs of the reference arm {a.ref} ({arms})")
    # the noise floor: each arm's runs against each other (the same code, the same tokens)
    noise = []
    print("== the same arm, run against run (the noise floor): median of the replay-by-replay difference (p10 / p90), gap")
    for arm in arms:
        rs = sorted((n for n, r in runs.items() if r["arm"] == arm), key=lambda n: runs[n]["round"])
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                p = pair(runs[rs[j]], runs[rs[i]])
                if p is None:
                    print(f"   {rs[j]} - {rs[i]}: not paired (other tokens or replay counts)")
                    continue
                noise.append(abs(p[0]))
                print(f"   {rs[j]} - {rs[i]}: {p[0]:+7.1f} ({p[1]:+.1f} / {p[2]:+.1f}), gap {p[3]:+6.1f}")
    floor = max([2.0] + [2 * x for x in noise])
    print(f"== the floor: {floor:.1f} us (twice the largest same-arm median, at least 2)")
    print(f"== each arm against {a.ref}: every pair of runs, then the median over the pairs")
    for arm in arms:
        if arm == a.ref:
            continue
        ds, gs = [], []
        for n, r in sorted(runs.items()):
            if r["arm"] != arm:
                continue
            for n0, r0 in sorted(runs.items()):
                if r0["arm"] != a.ref:
                    continue
                p = pair(r, r0)
                if p is None:
                    print(f"   {n} - {n0}: not paired (other tokens or replay counts)")
                    continue
                ds.append(p[0])
                gs.append(p[3])
                print(f"   {n} - {n0}: {p[0]:+7.1f} ({p[1]:+.1f} / {p[2]:+.1f}), gap {p[3]:+6.1f}")
        if not ds:
            print(f"   {arm}: no pair")
            continue
        md, mg = st.median(ds), st.median(gs)
        tag = "same (within the floor)" if abs(md) <= floor else ("FASTER" if md < 0 else "SLOWER")
        tg = "same" if abs(mg) <= floor else ("shorter" if mg < 0 else "longer")
        print(f"== {arm} against {a.ref}: in the graph {md:+.1f} us a step ({tag}); between replays {mg:+.1f} us ({tg}); "
              f"{len(ds)} pairs")


if __name__ == "__main__":
    main()
