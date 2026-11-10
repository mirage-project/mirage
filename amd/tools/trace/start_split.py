# start_split.py DUMP.json: per GPU, the CUs' entry (before the launch count's read), start (after it), resets stored,
# first task start, last end (us after the GPU's start barrier; medians over CUs and min / max)
import json, sys, collections
d = json.load(open(sys.argv[1])); c = d["cols"]; R = [dict(zip(c, r)) for r in d["rows"]]
q = lambda v: "%5.1f/%5.1f/%5.1f" % (min(v), sorted(v)[len(v) // 2], max(v))
for g, cus in enumerate(d["cu_stamps"]):
    ent = [s[4] / 1e3 for s in cus if s[4]]; st = [s[0] / 1e3 for s in cus if s[0]]; sto = [s[6] / 1e3 for s in cus if s[6]]
    first = collections.defaultdict(lambda: 1e18)
    for r in R:
        if r["gpu"] == g: first[r["cu"]] = min(first[r["cu"]], r["start"])
    f = [v / 1e3 for v in first.values()]
    end = max(r["end"] for r in R if r["gpu"] == g) / 1e3
    print("GPU %d entry %s | start %s | stored %s | first task %s | last end %.1f" % (g, q(ent), q(st), q(sto), q(f), end))
