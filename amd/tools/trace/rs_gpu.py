# per GPU (timing build, all GPUs' task_times_all_gpuN): node 37 (R send) last end, node 38 (sum_rmsnorm) start / s0 / s1 / end p50 & max,
# rel. that GPU's own start; the GPU that started last sees the true hop
import json, sys, numpy as np
d = sys.argv[1]
for gpu in range(8):
    try:
        t = np.load(f"{d}/task_times_all_gpu{gpu}.npy").astype(np.int64)
    except FileNotFoundError:
        continue
    s = json.load(open(f"{d}/schedule_gpu{gpu}.json")); st0 = int(open(f"{d}/start_all_gpu{gpu}.txt").read().split()[0])
    R = {}
    for sm, q in enumerate(s["worker_task_queues"]):
        for i, tid in enumerate(q):
            a, b, c, e = t[sm][i]
            if not a: continue
            n = s["all_tasks"][tid]["node"]
            R.setdefault(n, []).append([(x - st0) / 1e3 if x else np.nan for x in (a, b, c, e)])
    f = lambda n, k: np.array(R[n])[:, k]
    e36 = np.nanmax(f(36, 1)); e37 = np.nanmax(f(37, 1)); s37 = np.nanmedian(f(37, 2))
    print(f"GPU {gpu}: experts end max {e36:6.1f} | R send blocks ready p50 {s37:6.1f} end p50 {np.nanmedian(f(37,1)):6.1f} max {e37:6.1f} | "
          f"sum_rmsnorm start p50 {np.nanmedian(f(38,0)):6.1f} landed p50 {np.nanmedian(f(38,2)):6.1f} max {np.nanmax(f(38,2)):6.1f} "
          f"partner {np.nanmedian(f(38,3)):6.1f} end max {np.nanmax(f(38,1)):6.1f} | landed - own R send max {np.nanmedian(f(38,2)) - e37:5.1f}")
