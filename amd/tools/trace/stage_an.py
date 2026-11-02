# per node: start, stage 0, stage 1, end percentiles (us after start) from a timing build's task_times (4 slots)
import json, sys, numpy as np
d = sys.argv[1]
_s0 = __import__("json").load(open(f"{d}/schedule_gpu0.json"))
nodes = sorted(int(k) for k in _s0["nodes"]) if sys.argv[2] == "all" else [int(x) for x in sys.argv[2].split(",")]
t = np.load(f"{d}/task_times_gpu0.npy").astype(np.int64)
s = json.load(open(f"{d}/schedule_gpu0.json")); st0 = int(open(f"{d}/start_gpu0.txt").read().split()[0])
print("shape", t.shape)
for n in nodes:
    R = []
    for sm, q in enumerate(s["worker_task_queues"]):
        for i, tid in enumerate(q):
            if s["all_tasks"][tid]["node"] != n: continue
            v = t[sm][i]
            R.append([(x - st0) / 1e3 if x else np.nan for x in v[:4]])
    R = np.array(R)
    f = lambda c: " ".join(f"{np.nanpercentile(R[:, c], p):6.1f}" for p in (10, 50, 90)) if np.isfinite(R[:, c]).any() else "   -"
    print(f"node {n}: n {len(R)}  start {f(0)} | s0 {f(2)} | s1 {f(3)} | end {f(1)}")
    if np.isfinite(R[:, 2]).any():
        print(f"   s0-start p50 {np.nanmedian(R[:,2]-R[:,0]):.1f}  s1-s0 {np.nanmedian(R[:,3]-R[:,2]):.1f}  end-s1 {np.nanmedian(R[:,1]-R[:,3]):.1f}")
