# queue_an.py DUMP.json [GPU]: per node: tasks, first start, median start, median end, last end; then for each node the
# tasks that start late (> median start + 2 us) right after another node's task on their CU (queued behind it)
import json, sys, collections
d = json.load(open(sys.argv[1])); g = int(sys.argv[2]) if len(sys.argv) > 2 else 0
c = d["cols"]; R = [dict(zip(c, r)) for r in d["rows"] if r[0] == g]
for r in R:
    for k in ("start", "end", "st0", "st1"):
        r[k] = None if r[k] is None else r[k] / 1e3
cu = collections.defaultdict(list)
for r in R: cu[r["cu"]].append(r)
for v in cu.values(): v.sort(key=lambda r: r["start"])
nodes = sorted({(r["node"], r["name"]) for r in R})
med = lambda v: sorted(v)[len(v) // 2]
print("node name n  start(first/med)  end(med/last)")
for n, nm in nodes:
    rr = [r for r in R if r["node"] == n]
    print("%3d %-22s %4d  %6.1f %6.1f   %6.1f %6.1f" % (n, nm, len(rr), min(r["start"] for r in rr), med([r["start"] for r in rr]),
                                                   med([r["end"] for r in rr]), max(r["end"] for r in rr)))
print("late starts behind another node's task (start > median start + 2 us):")
for n, nm in nodes:
    rr = [r for r in R if r["node"] == n]; ms = med([r["start"] for r in rr])
    late = []
    for r in rr:
        if r["start"] <= ms + 2: continue
        L = cu[r["cu"]]; i = L.index(r)
        if i > 0 and L[i - 1]["node"] != n and r["start"] - L[i - 1]["end"] < 1.5:
            late.append((r, L[i - 1]))
    if late:
        prev = collections.Counter(p["node"] for r, p in late)
        print("  node %d %s: %d of %d late, behind nodes %s; latest start %.1f (median %.1f), their end med %.1f" % (
            n, nm, len(late), len(rr), dict(prev), max(r["start"] for r, p in late), ms, med([r["end"] for r, p in late])))
