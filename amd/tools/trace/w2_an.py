"""moe_experts per-unit trace of a timing build (test_layer_one_gpu.py / test_kda_moe_layer.py --profile: <out>/trace):
W13 / W2 units per wave, their durations, the gap before each, start times; W2 durations by start time.
  python3 amd/tools/trace/w2_an.py <out>/trace"""
# moe_trace: per wave [entries][2] (t0 ns, dur | kind << 32). W2 items (kind 2): duration distribution, the gap before each
# (previous item's end -> this start, same wave), items per wave; W13 (kind 1) the same
import sys, numpy as np
d = sys.argv[1]
tr = np.load(f"{d}/moe_trace_gpu0.npy").astype(np.int64)
start = int(open(f"{d}/start_gpu0.txt").read())
print("trace shape", tr.shape)
for kind, name in ((1, "W13"), (2, "W2")):
    durs, gaps, starts, per_wave = [], [], [], []
    for w in range(tr.shape[0]):
        prev_end = None; n = 0
        for i in range(tr.shape[1]):
            t0, dd = int(tr[w, i, 0]), int(tr[w, i, 1])
            if t0 == 0: continue
            k = (dd >> 32) & 0xFF; du = dd & 0xFFFFFFFF
            if k in (1, 4) and kind == 1 or k == kind == 2:
                durs.append(du / 1e3); starts.append((t0 - start) / 1e3); n += 1
                if prev_end is not None: gaps.append((t0 - prev_end) / 1e3)
                prev_end = t0 + du
        if n: per_wave.append(n)
    if not durs: continue
    q = lambda v, p: np.percentile(v, p)
    print(f"{name}: {len(durs)} items on {len(per_wave)} waves (items per wave: median {np.median(per_wave):.0f}, max {max(per_wave)}); "
          f"duration p10 {q(durs,10):.1f} p50 {q(durs,50):.1f} p90 {q(durs,90):.1f} us; gap before (same wave) p50 {q(gaps,50) if gaps else 0:.2f} "
          f"p90 {q(gaps,90) if gaps else 0:.2f} us; starts p10 {q(starts,10):.1f} p50 {q(starts,50):.1f} p90 {q(starts,90):.1f}")
    if kind == 2:
        # duration by start time bins of 4 us
        st, du = np.array(starts), np.array(durs)
        for b0 in range(30, 60, 4):
            m = (st >= b0) & (st < b0 + 4)
            if m.sum(): print(f"   W2 items started {b0}-{b0+4} us: {m.sum():4d}, duration p50 {np.median(du[m]):.1f} p90 {np.percentile(du[m],90):.1f}")
