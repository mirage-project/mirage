# expvar.py DIR NODE: per GPU the distribution of NODE's task starts / ends (us after the GPU's median go)
import numpy as np, sys, os, json
d, node = sys.argv[1], int(sys.argv[2])
for g in range(8):
    s = np.load(os.path.join(d, f"stamps_gpu{g}.npy")).astype(np.int64); ref = np.median(s[:, 0])
    t = np.load(os.path.join(d, f"task_times_all_gpu{g}.npy")).astype(np.int64)
    sg = json.load(open(os.path.join(d, f"schedule_gpu{g}.json")))
    st, en = [], []
    for sm, q in enumerate(sg["worker_task_queues"]):
        for i, tid in enumerate(q):
            if sg["all_tasks"][tid]["node"] == node and t[sm][i][0] and t[sm][i][1]:
                st.append((t[sm][i][0] - ref) / 1e3); en.append((t[sm][i][1] - ref) / 1e3)
    st, en = np.array(st), np.array(en)
    q = lambda a: " ".join(f"{np.percentile(a, p):6.1f}" for p in (0, 10, 50, 90, 99, 100))
    print(f"GPU {g} n {len(en)} start {q(st)} | end p0 p10 p50 p90 p99 max {q(en)} | 5 latest ends {np.sort(en)[-5:].round(1)}")
