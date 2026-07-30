#!/usr/bin/env python3
"""M4-I7: the MPK_MOE_PATH_POLICY sweep -- which fetch path actually wins in MPK.

The ferret dispatcher chose among PATH 0/1/2 on "do the work items fit one wave"
(items vs %nsmid), because there one CTA ran one work item. In MPK there is one
persistent worker per SM owning the whole smem budget, so residency is fixed and
that denominator is meaningless -- which left the choice open, so it was MEASURED
with each path pinned as a -D. Arm B only (arm A is unaffected by the pin), 3 reps,
bs1 and bs16, geometry B, all inside one GPU claim.
"""
import glob, json, os, statistics, sys

root = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/m4i7_sweep"
out = sys.argv[2] if len(sys.argv) > 2 else "."
rows = {}
for p in (0, 1, 2):
    for f in sorted(glob.glob(os.path.join(root, f"noprof_B_p{p}",
                                           "meta_bs*_rep*_B.json"))):
        m = json.load(open(f))
        rows.setdefault((p, m["batch_size"]), []).append(m["waves"][0]["wall_ms"])

L = [__doc__.strip(), "",
     f"{'path':>5} {'bs':>4} {'rep0':>9} {'rep1':>9} {'rep2':>9} {'median':>9}"]
med = {}
for (p, bs), v in sorted(rows.items(), key=lambda x: (x[0][1], x[0][0])):
    med[(p, bs)] = statistics.median(v)
    L.append(f"{p:>5} {bs:>4} " + " ".join(f"{x:9.1f}" for x in v)
             + f" {med[(p,bs)]:9.1f}")
L += ["", "== relative to PATH 1, the best arm =="]
for bs in sorted({b for _, b in rows}):
    if (1, bs) not in med:
        continue
    b1 = med[(1, bs)]
    for p in (0, 2):
        if (p, bs) in med:
            L.append(f"  bs{bs:<3} PATH {p}: {med[(p,bs)]:8.1f} ms  "
                     f"{100*(b1/med[(p,bs)]-1):+6.2f}% vs PATH 1")
L += ["",
      "SHIPPED: PATH 1 when admissible, else PATH 0. PATH 2 is built and",
      "bit-exact (Gate 1 covers it) but never selected -- its premise was halving",
      "per-CTA weight bytes to recruit a second CTA wave, and in MPK the task",
      "count is fixed by the graph and the flattened work space already saturates",
      "it, so halving the tile only doubles the per-item gathers, A re-fetches and",
      "epilogues for the same MMAs. It stays reachable through MPK_MOE_PATH_POLICY",
      "so the sweep can be repeated if the geometry changes."]
txt = "\n".join(L) + "\n"
open(os.path.join(out, "path_policy.txt"), "w").write(txt)
json.dump({f"p{p}_bs{bs}": v for (p, bs), v in rows.items()},
          open(os.path.join(out, "path_policy.json"), "w"), indent=1)
print(txt)
