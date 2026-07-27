#!/usr/bin/env python3
"""Per-(pid,tid,stream) breakdown of a kineto trace + union-of-busy vs sum-of-durations.

Answers: is the sum of kernel durations larger than the step wall because kernels genuinely
overlap on multiple streams, or because of double counting / profiler inflation?
"""
import json
import sys
from collections import defaultdict

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
path = sys.argv[1]
with open(path) as f:
    d = json.load(f)

by_track = defaultdict(lambda: {"n": 0, "us": 0.0, "names": set(), "iv": []})
anchor = sys.argv[2] if len(sys.argv) > 2 else "nvjet_sm100_tst_192x8_64x8_2x1_v_bz_TNT"
ats = []
for e in d["traceEvents"]:
    if e.get("cat") not in GPU_CATS:
        continue
    a = e.get("args") or {}
    key = (e.get("pid"), e.get("tid"), a.get("stream"), a.get("device"))
    t = by_track[key]
    t["n"] += 1
    t["us"] += e.get("dur", 0) or 0
    if len(t["names"]) < 6:
        t["names"].add(e.get("name", "")[:60])
    t["iv"].append((e.get("ts", 0), (e.get("ts", 0) or 0) + (e.get("dur", 0) or 0)))
    if e.get("name") == anchor:
        ats.append(e.get("ts", 0))

ats.sort()
t0, t1 = ats[0], ats[-1]
nsteps = len(ats) - 1
print(f"trace={path.split('/')[-1]}  anchor fires {len(ats)}x -> {nsteps} complete steps, "
      f"window {t1 - t0:.0f} us ({(t1 - t0)/nsteps:.1f} us/step)")
print(f"{'pid':>8} {'tid':>10} {'stream':>7} {'dev':>4} {'n':>8} {'us_total':>12} {'us/step':>10}   sample names")
allint = []
for k, v in sorted(by_track.items(), key=lambda x: -x[1]["us"]):
    ivs = [(a, b) for a, b in v["iv"] if a >= t0 and a < t1]
    us_win = sum(b - a for a, b in ivs)
    allint += ivs
    print(f"{str(k[0]):>8} {str(k[1]):>10} {str(k[2]):>7} {str(k[3]):>4} {len(ivs):>8} "
          f"{us_win:>12.1f} {us_win/nsteps:>10.1f}   {sorted(v['names'])[:2]}")

# union of busy intervals across all tracks
allint.sort()
merged, cur_s, cur_e = 0.0, None, None
for a, b in allint:
    if cur_s is None:
        cur_s, cur_e = a, b
    elif a <= cur_e:
        cur_e = max(cur_e, b)
    else:
        merged += cur_e - cur_s
        cur_s, cur_e = a, b
if cur_s is not None:
    merged += cur_e - cur_s
print(f"\nSUM of durations in window: {sum(b-a for a,b in allint)/nsteps:.1f} us/step")
print(f"UNION of busy intervals    : {merged/nsteps:.1f} us/step")
print(f"Window wall                : {(t1-t0)/nsteps:.1f} us/step")
print(f"=> GPU idle within window  : {((t1-t0)-merged)/nsteps:.1f} us/step")
