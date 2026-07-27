import gzip, json, sys, collections, statistics
p = sys.argv[1]
op = gzip.open if p.endswith(".gz") else open
with op(p, "rt") as f:
    d = json.load(f)
GPU = {"kernel", "gpu_memcpy", "gpu_memset"}
agg = collections.defaultdict(lambda: [0, 0.0])
steps = set()
streams = collections.defaultdict(float)
for e in d.get("traceEvents", []):
    n = e.get("name", "")
    if n.startswith("ProfilerStep#"):
        steps.add(n); continue
    if e.get("cat") not in GPU: continue
    a = agg[n]; a[0] += 1; a[1] += (e.get("dur", 0) or 0)
    streams[str((e.get("args") or {}).get("stream"))] += (e.get("dur",0) or 0)
tot = sum(v[1] for v in agg.values())
print(f"ProfilerStep markers: {len(steps)}   distinct kernels: {len(agg)}   total GPU us: {tot:.0f}")
print("streams:", {k: round(v) for k, v in streams.items()})
print(f"{'count':>8}{'tot_us':>11}{'us/call':>9}  name")
for n, (c, t) in sorted(agg.items(), key=lambda x: -x[1][1])[:32]:
    print(f"{c:>8}{t:>11.1f}{t/c:>9.3f}  {n[:120]}")
