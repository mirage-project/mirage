"""per CU of GPU 0 (a timing build's task dump, <out>/trace): its consecutive tasks of node P: durations by position (1st,
2nd, 3rd task of P on the CU) and the gap from one's end to the next one's start.
  python3 amd/tools/trace/chain_an.py <out>/trace P"""
import json, sys, numpy as np
d = sys.argv[1]; P = int(sys.argv[2])
t = np.load(f"{d}/task_times_gpu0.npy").astype(np.int64)
s = json.load(open(f"{d}/schedule_gpu0.json")); start = int(open(f"{d}/start_gpu0.txt").read())
dur = {}; gap = []; first_start = []
for sm, q in enumerate(s["worker_task_queues"]):
    prev_end = None; pos = 0
    for i, tid in enumerate(q):
        a, b = t[sm][i][0], t[sm][i][1]
        if not (a and b): continue
        if s["all_tasks"][tid]["node"] != P:
            prev_end = None
            continue
        if pos == 0: first_start.append((a - start) / 1e3)
        dur.setdefault(pos, []).append((b - a) / 1e3)
        if prev_end is not None: gap.append((a - prev_end) / 1e3)
        prev_end = b; pos += 1
for p_, v in sorted(dur.items()):
    print(f"task {p_ + 1} of node {P} on a CU: {len(v)} CUs, duration p10 {np.percentile(v, 10):.2f} p50 {np.median(v):.2f} p90 {np.percentile(v, 90):.2f} us")
if gap: print(f"gap end -> next start: p10 {np.percentile(gap, 10):.2f} p50 {np.median(gap):.2f} p90 {np.percentile(gap, 90):.2f} us ({len(gap)})")
