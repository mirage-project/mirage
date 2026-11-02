# w2die.py TRACE_DIR GPU: per slot, the die (sm_id % 8 of the wave's CU, wave index w = 4 sm + wave) that took most of its W2 items,
# then per item: own-die (the slot's main die) vs other-die, median duration, split by start before / after 130 us
import numpy as np, sys, os, collections
d, g = sys.argv[1], int(sys.argv[2])
tr = np.load(os.path.join(d, f"moe_trace_gpu{g}.npy")).astype(np.int64)
ref = np.median(np.load(os.path.join(d, f"stamps_gpu{g}.npy")).astype(np.int64)[:, 0])
print("trace shape", tr.shape)
it = []
for w in range(tr.shape[0]):
    for i in range(tr.shape[1]):
        a, b = int(tr[w, i, 0]), int(tr[w, i, 1])
        if not a: break
        if (b >> 32) & 0xFF == 2: it.append(((a - ref) / 1e3, (b & 0xFFFFFFFF) / 1e3, (b >> 40) & 0xFFFFFF, (w // 4) % 8))
it = np.array(it)
main = {}
for s in set(it[:, 2].astype(int).tolist()):
    main[s] = collections.Counter(it[it[:, 2] == s, 3].astype(int).tolist()).most_common(1)[0][0]
own = np.array([main[int(s)] == int(dd) for s, dd in zip(it[:, 2], it[:, 3])])
for nm, m in (("before 130", it[:, 0] < 130), ("after 130", it[:, 0] >= 130)):
    for o, om in (("own die", own), ("other die", ~own)):
        x = it[m & om, 1]
        if len(x): print(f"{nm:10s} {o:9s}: n {len(x):5d} dur p50 {np.median(x):5.2f} p90 {np.percentile(x, 90):5.2f}")
sl = sorted(main)
print("main die of slots 160..215:", [(s, main[s]) for s in sl if 160 <= s <= 215])
