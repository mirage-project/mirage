"""per CU of GPU 0 (a timing build's task dump, <out>/trace): its last task end before node P (e.g. the MoE part of a
MoE + attention kernel) and its tasks of node P (count, first start, last end).
  python3 amd/tools/trace/proj_an.py <out>/trace P"""
# per CU: its last MoE task end, its projection (node P) tasks: count, first start, last end (us after start)
import json, sys, numpy as np
d = sys.argv[1]; P = int(sys.argv[2])
t = np.load(f"{d}/task_times_gpu0.npy").astype(np.int64)
s = json.load(open(f"{d}/schedule_gpu0.json")); start = int(open(f"{d}/start_gpu0.txt").read())
rows = []
for sm, q in enumerate(s["worker_task_queues"]):
    moe_end, pj = 0, []
    for i, tid in enumerate(q):
        a, b = t[sm][i][0], t[sm][i][1]
        if not (a and b): continue
        node = s["all_tasks"][tid]["node"]
        if node < P: moe_end = max(moe_end, (b - start) / 1e3)
        elif node == P: pj.append(((a - start) / 1e3, (b - start) / 1e3))
    if pj: rows.append((moe_end, len(pj), min(x for x, _ in pj), max(y for _, y in pj)))
rows.sort(key=lambda r: -r[3])
print("latest projection ends: moe_end, n tasks, proj first start, proj last end")
for r in rows[:12]: print("  %6.1f %d %6.1f %6.1f" % r)
import collections
by = collections.defaultdict(list)
for r in rows: by[(r[1], "late" if r[0] > 70 else "early")].append(r[3])
for k, v in sorted(by.items()): print(k, len(v), "proj end median %.1f max %.1f" % (np.median(v), max(v)))
