# xgpu_an.py DIR: per GPU (aligned at its median go pass, timing build stamps), per node: first start, median end, last end (us)
import numpy as np, sys, os, json
d = sys.argv[1]
sched = json.load(open(os.path.join(d, "schedule_gpu0.json")))
names = {int(k): v["name"] + "#" + k for k, v in sched["nodes"].items()}
res = {}
for g in range(8):
    p = os.path.join(d, f"stamps_gpu{g}.npy")
    if not os.path.exists(p): break
    s = np.load(p).astype(np.int64)
    ref = np.median(s[:, 5] if (len(sys.argv) > 2 and sys.argv[2] == "eres") else s[:, 0])   # the go pass: eres stamp 5
    t = np.load(os.path.join(d, f"task_times_all_gpu{g}.npy")).astype(np.int64)
    sg = json.load(open(os.path.join(d, f"schedule_gpu{g}.json"))) if os.path.exists(os.path.join(d, f"schedule_gpu{g}.json")) else sched
    per = {}
    for sm, q in enumerate(sg["worker_task_queues"]):
        for i, tid in enumerate(q):
            a, b = t[sm][i][0], t[sm][i][1]
            if a and b:
                per.setdefault(sg["all_tasks"][tid]["node"], []).append(((a - ref) / 1e3, (b - ref) / 1e3))
    res[g] = per
    ent, bar = s[:, 4], s[:, 5]
    ok = ent > 0
    print(f"GPU {g}: entry {((ent[ok].min()-ref)/1e3):6.1f}..{((np.median(ent[ok])-ref)/1e3):6.1f}, barrier med {((np.median(bar[ok])-ref)/1e3):5.1f} "
          f"max {((bar[ok].max()-ref)/1e3):5.1f}, go first {((s[ok,0].min()-ref)/1e3):5.1f} last {((s[ok,0].max()-ref)/1e3):5.1f}, end {((s[:,3].max()-ref)/1e3):5.1f}")
nodes = sorted({n for g in res for n in res[g]})
print("node: per GPU last end (us after its go)")
for n in nodes:
    print(f"  {names.get(n, n):22s}", " ".join(f"{max(e for _, e in res[g][n]):6.1f}" if n in res[g] else "   -  " for g in res))
