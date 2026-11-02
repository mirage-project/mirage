"""moe_experts' item trace (timing build; test_layer_one_gpu.py / test_kda_moe_layer.py --profile dump <out>/trace): per kind
the entries' start / end / duration percentiles (us after the GPU's start barrier), per wave the W13 units and W2 items.
Kinds: 1 W13 unit, 2 W2 item (loads issued -> last store), 10 task start -> z_q in LDS (table path) or -> table (CSR path),
11 z_q -> table (table path), 13 W13 sums made -> unit end, 14 a CSR W13 unit's z_q staging (summed over its quarters).
    python moe_trace.py TRACE_DIR [GPU]"""
import collections
import os
import sys

import numpy as np

NAMES = {1: "W13 unit", 2: "W2 item", 10: "start phase", 11: "z_q -> table", 13: "W13 sums -> end", 14: "W13 z_q staging"}


def main():
    d = sys.argv[1]
    gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tr = np.load(os.path.join(d, f"moe_trace_gpu{gpu}.npy"))
    start = int(open(os.path.join(d, f"start_gpu{gpu}.txt")).read())
    ent = collections.defaultdict(list)
    per_wave = collections.defaultdict(lambda: collections.Counter())
    for w in range(tr.shape[0]):
        for i in range(tr.shape[1]):
            t0, x = int(tr[w, i, 0]), int(tr[w, i, 1])
            if not t0:
                break
            kind, dur = (x >> 32) & 0xFF, x & 0xFFFFFFFF
            if kind in (0, 9):   # (0: not an entry; 9: the clock's cycle count)
                continue
            ent[kind].append(((t0 - start) / 1e3, (t0 + dur - start) / 1e3))
            per_wave[w][kind] += 1
    pc = lambda a, q: np.percentile(a, q)
    for k in sorted(ent):
        a = np.array(ent[k])
        du = a[:, 1] - a[:, 0]
        print(f"{NAMES.get(k, k)!s:>16} ({k:2d}) n {len(a):5d}  start p10/50/90 {pc(a[:, 0], 10):6.1f} {pc(a[:, 0], 50):6.1f} "
              f"{pc(a[:, 0], 90):6.1f}  end p10/50/90/max {pc(a[:, 1], 10):6.1f} {pc(a[:, 1], 50):6.1f} {pc(a[:, 1], 90):6.1f} "
              f"{a[:, 1].max():6.1f}  dur p10/50/90 {pc(du, 10):5.2f} {pc(du, 50):5.2f} {pc(du, 90):5.2f}")
    for k in (1, 2):
        c = collections.Counter(per_wave[w][k] for w in range(tr.shape[0]))
        print(f"waves per {NAMES[k]} count: {dict(sorted(c.items()))}")


if __name__ == "__main__":
    main()
