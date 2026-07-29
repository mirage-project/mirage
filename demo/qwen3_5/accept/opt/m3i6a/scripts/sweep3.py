#!/usr/bin/env python3
"""M3-I6a: the three-way pass-size sweep (4 / 2 / 1) at all five batch sizes.

All three arms are measured back-to-back per (bs, rep) inside one GPU claim at
integrated HEAD, with a drain gate before every rep and a per-rep device audit,
so the arms are comparable to each other and not just each to the baseline.

Every rep's PINNED DEVICE is read from that rep's OWN record -- `meta.
cuda_visible_devices` for the profile_wave geometries, the sidecar's
`pinned_device` for the AC-3 geometry -- never from the candidate list the guard
was given, which is not evidence of what actually ran (M3-I7's lesson).  A rep
whose pinned device already held >500 MiB when it started is DISCARDED and
reported, not averaged in.
"""
from __future__ import annotations

import json
import os
import statistics as st
import sys

M = sys.argv[1] if len(sys.argv) > 1 else "/var/tmp/m3i6a_sweep"
BSL = [1, 2, 4, 8, 16]
ARMS = [4, 2, 1]
REPS = [0, 1, 2]
DIRTY_MIB = 500

discarded: list[str] = []


def jload(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def sample_A(qp, bs, rep):
    """(wall_ms summed over waves, median ms/decode-step) or None."""
    tag = f"A_qp{qp}_bs{bs}_rep{rep}"
    aud = jload(f"{M}/audit/{tag}.json")
    if aud and aud.get("mib_before") is not None and aud["mib_before"] > DIRTY_MIB:
        discarded.append(f"{tag}: device {aud['pinned_device']} at "
                         f"{aud['mib_before']} MiB at start")
        return None
    for name in (f"timings_bs{bs}_rep{rep}", f"timings_bs{bs}_rep{rep}.json"):
        d = jload(os.path.join(f"{M}/dumpsA_qp{qp}", name))
        if not d:
            continue
        t = d.get("timings") or d.get("waves") or []
        if not t:
            return None
        return (sum(x["wall_ms"] for x in t),
                st.median([x["ms_per_decode_step"] for x in t
                           if x.get("ms_per_decode_step")]))
    return None


def sample_BC(geom, qp, bs, rep):
    tag = f"{geom}_qp{qp}_bs{bs}_rep{rep}"
    d = jload(f"{M}/noprof{geom}_qp{qp}/meta_bs{bs}_rep{rep}_qp{qp}.json")
    if not d or not d.get("waves"):
        return None
    vis = str(d.get("cuda_visible_devices", "")).strip()
    for row in d.get("gpu_before") or []:
        f = [x.strip() for x in row.split(",")]
        if f and f[0] == vis:
            mib = int(float(f[1].replace("MiB", "").strip()))
            if mib > DIRTY_MIB:
                discarded.append(f"{tag}: device {vis} at {mib} MiB at start")
                return None
            break
    return d["waves"][0]["wall_ms"]


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    m = st.median(vals)
    return dict(median=m, n=len(vals), spread=max(vals) - min(vals),
                pct=100.0 * (max(vals) - min(vals)) / m if m else 0.0)


def table(title, note, getter, bss):
    print()
    print("=" * 118)
    print(title)
    if note:
        print("  " + note)
    print("=" * 118)
    print(f"{'bs':>3} | {'pass4 ms':>10} {'sp%':>5} {'n':>2} | "
          f"{'pass2 ms':>10} {'sp%':>5} {'n':>2} | {'pass1 ms':>10} {'sp%':>5} {'n':>2} | "
          f"{'2 vs 4':>8} {'1 vs 4':>8} {'1 vs 2':>8} | {'winner':>8}")
    tot = {q: 0.0 for q in ARMS}
    for bs in bss:
        a = {q: agg([getter(q, bs, r) for r in REPS]) for q in ARMS}
        if any(a[q] is None for q in ARMS):
            print(f"{bs:3d} | incomplete: " +
                  " ".join(f"pass{q}={'ok' if a[q] else 'MISSING'}" for q in ARMS))
            continue
        for q in ARMS:
            tot[q] += a[q]["median"]
        r24 = a[2]["median"] / a[4]["median"]
        r14 = a[1]["median"] / a[4]["median"]
        r12 = a[1]["median"] / a[2]["median"]
        best = min(ARMS, key=lambda q: a[q]["median"])
        # a difference smaller than the larger arm's own rep spread is not a result
        tolerance = max(a[1]["spread"], a[2]["spread"])
        tie = abs(a[1]["median"] - a[2]["median"]) <= tolerance
        win = "2~1 tie" if tie and best in (1, 2) else f"pass {best}"
        print(f"{bs:3d} | {a[4]['median']:10.1f} {a[4]['pct']:5.2f} {a[4]['n']:2d} | "
              f"{a[2]['median']:10.1f} {a[2]['pct']:5.2f} {a[2]['n']:2d} | "
              f"{a[1]['median']:10.1f} {a[1]['pct']:5.2f} {a[1]['n']:2d} | "
              f"{r24:8.4f} {r14:8.4f} {r12:8.4f} | {win:>8}")
    if tot[4]:
        print(f"{'ALL':>3} | {tot[4]:10.1f} {'':>5} {'':>2} | {tot[2]:10.1f} {'':>5} {'':>2} | "
              f"{tot[1]:10.1f} {'':>5} {'':>2} | {tot[2]/tot[4]:8.4f} "
              f"{tot[1]/tot[4]:8.4f} {tot[1]/tot[2]:8.4f} |")


table("GEOMETRY A -- AC-3 geometry (10 pinned reference prompts, msl=132); "
      "e2e = sum of wave wall_ms",
      "the correctness geometry: short prompts, KV context <= 132",
      lambda q, b, r: (sample_A(q, b, r) or (None,))[0], BSL)

print()
print("  per-wave median ms/decode-step, same runs")
print(f"  {'bs':>3} | {'pass4':>9} {'pass2':>9} {'pass1':>9} | {'2/4':>7} {'1/4':>7} {'1/2':>7}")
for bs in BSL:
    a = {q: agg([(sample_A(q, bs, r) or (None, None))[1] for r in REPS]) for q in ARMS}
    if any(a[q] is None for q in ARMS):
        print(f"  {bs:3d} | incomplete")
        continue
    print(f"  {bs:3d} | {a[4]['median']:9.3f} {a[2]['median']:9.3f} {a[1]['median']:9.3f} | "
          f"{a[2]['median']/a[4]['median']:7.4f} {a[1]['median']/a[4]['median']:7.4f} "
          f"{a[1]['median']/a[2]['median']:7.4f}")

table("GEOMETRY B -- matched 256/1024 (synth 256-token prompt, msl=353, "
      "96 decode steps); unprofiled wave wall_ms",
      "the PREFILL-heavy end: 256*bs/16 prefill iterations against 96 decode "
      "steps -- where pass=1's doubled pass count should hurt most",
      lambda q, b, r: sample_BC("B", q, b, r), BSL)

table("GEOMETRY C -- deep context (synth 256-token prompt, msl=897, 640 decode "
      "steps, ctx 257->896); unprofiled wave wall_ms",
      "the DECODE-heavy end: 16 prefill iterations per request against 640 "
      "decode steps -- where pass=1's lower per-KV-token cost should help most",
      lambda q, b, r: sample_BC("C", q, b, r), BSL)

print()
print("=" * 118)
if discarded:
    print(f"DISCARDED REPS (pinned device not free at start): {len(discarded)}")
    for x in discarded:
        print("  " + x)
else:
    print("DISCARDED REPS: 0 -- every rep started on a device holding "
          f"<= {DIRTY_MIB} MiB, verified from that rep's own record")
print("=" * 118)
