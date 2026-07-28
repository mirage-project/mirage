#!/usr/bin/env python3
"""M3-I6a e2e A/B: per-bs medians over 3 reps, both geometries.

Geometry A -- the AC-3 geometry: `mpk_engine_run.py` over the 10 pinned AC-3
reference prompts at msl=132.  The wave count depends on bs (10 prompts / bs),
so the e2e number is the SUM of wall_ms over that run's waves (time to produce
all 10 prompts' 64 tokens) plus the median per-wave ms_per_decode_step.

Geometry B -- the matched 256/1024 geometry: `profile_wave.py --no-profiler`,
256-token synthetic prompt, msl=353 (96 decode steps), M3-I10 armA's basis.
Metric is the unprofiled wave wall_ms.

Both arms are measured in the SAME window, alternating reps within one GPU
claim, so drift and co-tenant noise hit both arms equally.
"""
from __future__ import annotations

import glob
import json
import os
import statistics as st
import sys

M = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/mpk-qwen35/i6a")
BSL = [1, 2, 4, 8, 16]
ARMS = [4, 2]


def load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def geomA(qp, bs, rep):
    base = f"{M}/perf/dumpsA_qp{qp}"
    for name in (f"timings_bs{bs}_rep{rep}", f"timings_bs{bs}_rep{rep}.json"):
        d = load_json(os.path.join(base, name))
        if d:
            t = d.get("timings") or d.get("waves") or []
            if not t:
                return None
            wall = sum(x["wall_ms"] for x in t)
            mps = st.median([x["ms_per_decode_step"] for x in t
                             if x.get("ms_per_decode_step")])
            return wall, mps, len(t)
    return None


def geomB(qp, bs, rep):
    for pat in (f"{M}/perf/noprofB_qp{qp}/meta_bs{bs}_rep{rep}_qp{qp}.json",
                f"{M}/perf/noprofB_qp{qp}/meta_bs{bs}_rep{rep}.json"):
        d = load_json(pat)
        if d and d.get("waves"):
            return d["waves"][0]["wall_ms"]
    return None


def med(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    return st.median(vals), (max(vals) - min(vals)), len(vals)


print("=" * 104)
print("GEOMETRY A -- AC-3 geometry (10 pinned reference prompts, msl=132), "
      "e2e = sum of wave wall_ms")
print("=" * 104)
print(f"{'bs':>3} {'waves':>6} | {'qp4 wall_ms':>12} {'spread':>7} {'n':>2} | "
      f"{'qp2 wall_ms':>12} {'spread':>7} {'n':>2} | {'qp2/qp4':>8} {'speedup':>8} | "
      f"{'qp4 ms/step':>12} {'qp2 ms/step':>12} {'ratio':>7}")
sumA = {4: 0.0, 2: 0.0}
for bs in BSL:
    got = {q: [geomA(q, bs, r) for r in (0, 1, 2)] for q in ARMS}
    nw = next((g[2] for g in got[4] if g), None)
    w = {q: med([g[0] for g in got[q] if g]) for q in ARMS}
    s = {q: med([g[1] for g in got[q] if g]) for q in ARMS}
    if w[4][0] is None or w[2][0] is None:
        print(f"{bs:3d} {'?':>6} | incomplete")
        continue
    sumA[4] += w[4][0]
    sumA[2] += w[2][0]
    ratio = w[2][0] / w[4][0]
    print(f"{bs:3d} {nw:6d} | {w[4][0]:12.1f} {w[4][1]:7.1f} {w[4][2]:2d} | "
          f"{w[2][0]:12.1f} {w[2][1]:7.1f} {w[2][2]:2d} | {ratio:8.4f} "
          f"{1/ratio:7.3f}x | {s[4][0]:12.3f} {s[2][0]:12.3f} "
          f"{(s[2][0]/s[4][0]) if s[4][0] else 0:7.4f}")
if sumA[4]:
    print(f"{'ALL':>3} {'':>6} | {sumA[4]:12.1f} {'':>7} {'':>2} | {sumA[2]:12.1f} "
          f"{'':>7} {'':>2} | {sumA[2]/sumA[4]:8.4f} {sumA[4]/sumA[2]:7.3f}x")

print()
print("=" * 104)
print("GEOMETRY B -- matched 256/1024 (synth 256-token prompt, msl=353, "
      "96 decode steps), unprofiled wave wall_ms")
print("=" * 104)
print(f"{'bs':>3} | {'qp4 wall_ms':>12} {'spread':>7} {'n':>2} | "
      f"{'qp2 wall_ms':>12} {'spread':>7} {'n':>2} | {'qp2/qp4':>8} {'speedup':>8}")
sumB = {4: 0.0, 2: 0.0}
for bs in BSL:
    w = {q: med([geomB(q, bs, r) for r in (0, 1, 2)]) for q in ARMS}
    if w[4][0] is None or w[2][0] is None:
        print(f"{bs:3d} | incomplete  qp4={w[4]}  qp2={w[2]}")
        continue
    sumB[4] += w[4][0]
    sumB[2] += w[2][0]
    ratio = w[2][0] / w[4][0]
    print(f"{bs:3d} | {w[4][0]:12.1f} {w[4][1]:7.1f} {w[4][2]:2d} | "
          f"{w[2][0]:12.1f} {w[2][1]:7.1f} {w[2][2]:2d} | {ratio:8.4f} {1/ratio:7.3f}x")
if sumB[4]:
    print(f"{'ALL':>3} | {sumB[4]:12.1f} {'':>7} {'':>2} | {sumB[2]:12.1f} {'':>7} "
          f"{'':>2} | {sumB[2]/sumB[4]:8.4f} {sumB[4]/sumB[2]:7.3f}x")

print()
print("=" * 104)
print("GEOMETRY C -- deep context (synth 256-token prompt, msl=897, 640 decode "
      "steps, ctx 257->896), unprofiled wave wall_ms")
print("  at bs1 the wave is 16 prefill iterations against 640 decode steps, so "
      "this wall ratio IS the decode ratio")
print("=" * 104)


def geomC(qp, bs, rep):
    for pat in (f"{M}/perf/noprofC_qp{qp}/meta_bs{bs}_rep{rep}_qp{qp}.json",
                f"{M}/perf/noprofC_qp{qp}/meta_bs{bs}_rep{rep}.json"):
        d = load_json(pat)
        if d and d.get("waves"):
            return d["waves"][0]["wall_ms"]
    return None


def dirty(qp, bs, rep, gpu=None):
    """Reject a rep whose pinned device was already occupied at run start."""
    for pat in (f"{M}/perf/noprofC_qp{qp}/meta_bs{bs}_rep{rep}_qp{qp}.json",):
        d = load_json(pat)
        if not d:
            continue
        vis = str(d.get("cuda_visible_devices", "")).strip()
        for row in d.get("gpu_before") or []:
            f = [x.strip() for x in row.split(",")]
            if f and f[0] == vis:
                mib = int(float(f[1].replace("MiB", "").strip()))
                return mib > 500, mib
    return False, None


print(f"{'bs':>3} | {'qp4 wall_ms':>12} {'spread':>7} {'n':>2} | "
      f"{'qp2 wall_ms':>12} {'spread':>7} {'n':>2} | {'qp2/qp4':>8} {'speedup':>8} | "
      f"{'dirty reps':>10}")
sumC = {4: 0.0, 2: 0.0}
for bs in [1, 8, 16]:
    w, nd = {}, 0
    for q in ARMS:
        vals = []
        for r in (0, 1, 2):
            bad, mib = dirty(q, bs, r)
            if bad:
                nd += 1
                print(f"      DISCARDED qp{q} bs{bs} rep{r}: device already at {mib} MiB")
                continue
            vals.append(geomC(q, bs, r))
        w[q] = med(vals)
    if w[4][0] is None or w[2][0] is None:
        print(f"{bs:3d} | incomplete")
        continue
    sumC[4] += w[4][0]
    sumC[2] += w[2][0]
    ratio = w[2][0] / w[4][0]
    print(f"{bs:3d} | {w[4][0]:12.1f} {w[4][1]:7.1f} {w[4][2]:2d} | "
          f"{w[2][0]:12.1f} {w[2][1]:7.1f} {w[2][2]:2d} | {ratio:8.4f} {1/ratio:7.3f}x | "
          f"{nd:10d}")
if sumC[4]:
    print(f"{'ALL':>3} | {sumC[4]:12.1f} {'':>7} {'':>2} | {sumC[2]:12.1f} {'':>7} {'':>2} "
          f"| {sumC[2]/sumC[4]:8.4f} {sumC[4]/sumC[2]:7.3f}x")
