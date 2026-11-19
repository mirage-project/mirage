"""period.py DB [--dev=N]: the decode step from an nsys trace taken with --cuda-graph-trace=graph (one record per replay of a
CUDA graph, CUPTI_ACTIVITY_KIND_GRAPH_TRACE), as PR #819's nsys_e2e.py: the forward graph = the graph with the longest median
replay among those replayed at least 5 times; its period = start to start of consecutive replays. Also the replay's duration
(the GPU's time in the graph) and the rest of the period (the GPU is not in the forward graph: the host side, the scheduler)."""
import collections
import sqlite3
import statistics as st
import sys

dev = int(next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--dev=")), 0))
db = [a for a in sys.argv[1:] if not a.startswith("--")][0]
c = sqlite3.connect(db)
by = collections.defaultdict(list)
for s, e, g in c.execute("select start, end, graphId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE where deviceId = ? order by start", (dev,)):
    by[g].append((s / 1e3, (e - s) / 1e3))
for g in sorted(by, key=lambda g: -len(by[g]))[:6]:
    s = sorted(x for x, _ in by[g])
    per = [b - a for a, b in zip(s, s[1:])]
    print(f"   graph {g}: {len(by[g])} replays, duration median {st.median(d for _, d in by[g]):.1f} us, period median "
          f"{st.median(per) if per else float('nan'):.1f} us")
tg = max((g for g in by if len(by[g]) >= 5), key=lambda g: st.median(d for _, d in by[g]))
rep = sorted(by[tg])
per = [b[0] - a[0] for a, b in zip(rep, rep[1:])]
dur = [d for _, d in rep]
gap = [b[0] - (a[0] + a[1]) for a, b in zip(rep, rep[1:])]
q = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
print(f"forward graph {tg} on GPU {dev}: {len(per)} periods; period median {st.median(per):.1f} us (p10 {q(per, 0.1):.1f}, p90 "
      f"{q(per, 0.9):.1f}); GPU in the graph median {st.median(dur):.1f} us; between replays median {st.median(gap):.1f} us")
