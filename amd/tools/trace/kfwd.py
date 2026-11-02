# kfwd.py TRACE.json.gz: within the decode forwards (between consecutive mpk / sglang marker-free spans: all GPU kernels of the busiest
# stream), the total time per kernel name, top 25, per decode step (kernels / number of steps estimated by the count of the most frequent
# mpk kernel / layers)
import gzip, json, sys, collections
d = json.load(gzip.open(sys.argv[1])); nl = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ev = [e for e in d["traceEvents"] if e.get("ph") == "X" and e.get("cat") in ("kernel", "Kernel") and e.get("dur", 0) > 0]
by = collections.defaultdict(list)
for e in ev: by[(e.get("pid"), e.get("tid"))].append(e)
k = max(by, key=lambda x: len(by[x])); L = by[k]
tot = collections.Counter(); cnt = collections.Counter()
for e in L:
    n = e["name"][:70]; tot[n] += e["dur"]; cnt[n] += 1
allt = sum(tot.values())
mk = [n for n in tot if "mpk" in n.lower() or "megakernel" in n.lower() or "static" in n.lower()]
print(f"stream {k}: {len(L)} kernels, {allt/1e3:.1f} ms total; mpk-like names: {mk[:4]}")
for n, t in tot.most_common(25):
    print(f"  {t/1e3:9.2f} ms {cnt[n]:7d} x {t/cnt[n]:8.1f} us  {n}")
