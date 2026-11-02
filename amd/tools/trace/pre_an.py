# pre_an.py DIR: per GPU, the pre-go timeline from the timing build's stamps (stamps_gpu*.npy: [sm][START(go pass), TASK0, TASK1,
# END, ENTRY, BAR]), us relative to that GPU's median go pass (the GPUs' clocks are not shared; the go is ~simultaneous)
import numpy as np, sys, os
d = sys.argv[1]
for g in range(8):
    p = os.path.join(d, f"stamps_gpu{g}.npy")
    if not os.path.exists(p): break
    s = np.load(p).astype(np.int64)
    go, ent, bar, end = s[:, 0], s[:, 4], s[:, 5], s[:, 3]
    ref = np.median(go)
    f = lambda x: (x - ref) / 1e3
    ok = ent > 0
    print(f"GPU {g}: entry first {f(ent[ok].min()):6.1f} med {f(np.median(ent[ok])):6.1f} last {f(ent[ok].max()):6.1f} (CU {int(np.argmax(np.where(ok, ent, 0)))}) | "
          f"barrier passed first {f(bar[ok].min()):6.1f} med {f(np.median(bar[ok])):6.1f} last {f(bar[ok].max()):6.1f} | "
          f"go pass first {f(go[ok].min()):5.1f} last {f(go[ok].max()):5.1f} | end last {f(end.max()):6.1f}")
