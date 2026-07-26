#!/usr/bin/env python3
"""M3-I9 -- grade the captured runs against predictions.md.

Every mode compares a measurement against a number that was written down before
the GPU was touched.  A mode with no data prints what is missing and exits
non-zero; it never quietly reports a pass.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys

from cost_model import PROF_OVERHEAD, wave_ms
from protocol_sim import ac3_slots, audit, simulate

# ---- registered predictions (predictions.md) --------------------------------
# earliest generated-token index a duplicate slot's answer can differ at, IF the
# cause is compaction.  Slot 15 is migrated mid-prefill, so it does not
# discriminate.
DIVERGENCE_BOUND = {10: 60, 11: 54, 12: 46, 13: 35, 14: 19}
SLOT_ORDER_MS = (4050.0, 4350.0)      # C7 band, profiled clock, bs16
CAP_BS16_MS_UNPROF = (2750.0, 3050.0)  # C5 band, unprofiled clock
CAP_ITERS = {1: 109, 2: 110, 4: 113, 8: 119, 16: 131}


def _timings(path):
    with open(path) as f:
        return json.load(f)


def check_divergence(path) -> int:
    """C1+C2: where the bs16 duplicate slots diverge."""
    d = _timings(path)
    checks = d.get("slot_isolation_checks") or []
    if not checks:
        print("FAIL: no slot_isolation_checks in", path)
        return 1
    print("stage 0 -- duplicate-slot first divergence vs the compaction bound")
    print(f"{'slot':>5s} {'prompt':14s} {'identical':>10s} {'first_div':>10s} "
          f"{'bound':>6s} {'verdict'}")
    bad = 0
    for c in checks:
        slot = c["slots"][1]
        fd = c.get("first_divergence")
        bound = DIVERGENCE_BOUND.get(slot)
        if bound is None:
            v = "n/a (does not discriminate)"
        elif c["identical"]:
            v, bad = "UNEXPECTED MATCH -- read with stage 1", bad
        elif fd is None or fd < bound:
            v = "REFUTES compaction"
            bad += 1
        else:
            v = "consistent with compaction"
        print(f"{slot:5d} {c['prompt_id']:14s} {str(c['identical']):>10s} "
              f"{str(fd):>10s} {str(bound):>6s} {v}")
    order = [c.get("first_divergence") for c in sorted(checks, key=lambda x: x["slots"][1])
             if c["slots"][1] in DIVERGENCE_BOUND]
    mono = all(a is not None and b is not None and a > b
               for a, b in zip(order, order[1:])) if len(order) > 1 else False
    print(f"  strictly decreasing across slots 10..14: {mono} "
          f"(predicted True; the competing 'prefill chunking' explanation has no "
          f"mechanism for an ordering)")
    print("VERDICT:", "PASS" if not bad and mono else "FAIL")
    return 0 if (not bad and mono) else 1


def check_isolation(path) -> int:
    """C2 negative control at msl=212: no straddles, so every pair must match."""
    d = _timings(path)
    checks = d.get("slot_isolation_checks") or []
    w = d["waves"][0]
    pred = audit(simulate([132] * 0 or ac3_slots(16), 16, w["max_seq_length"]), 64)
    print(f"stage 1 -- msl={w['max_seq_length']}, predicted straddling slots "
          f"{pred['straddling_requests']} (must be empty), "
          f"predicted iterations {simulate(ac3_slots(16), 16, w['max_seq_length'])['n_iterations']}")
    bad = sum(1 for c in checks if not c["identical"])
    for c in checks:
        print(f"  {c['prompt_id']:14s} slots={c['slots']} identical={c['identical']} "
              f"first_divergence={c.get('first_divergence')}")
    print("VERDICT:", "PASS -- the mismatch is compaction" if bad == 0 else
          f"FAIL -- {bad} pair(s) still disagree with zero straddles; C2 is dead, "
          "root-cause before building the runtime knob")
    return 0 if bad == 0 else 1


def check_costlaw(root, reps=3) -> int:
    """C7: --slot-order sorted-padded must land in the predicted band."""
    got = []
    for p in sorted(glob.glob(os.path.join(root, "s2_sorted_rep*", "timings_bs16.json"))):
        got.append(_timings(p)["waves"][0]["wall_ms"])
    if not got:
        print("FAIL: no stage-2 runs under", root)
        return 1
    med = statistics.median(got)
    # measured runs are UNPROFILED; the band is on the profiled clock
    med_prof = med * PROF_OVERHEAD[16]
    lo, hi = SLOT_ORDER_MS
    sim = simulate(sorted(ac3_slots(16)), 16, 132)
    print(f"stage 2 -- slot-order sorted-padded, n={len(got)} reps")
    print(f"  predicted iterations {sim['n_iterations']}, predicted "
          f"{wave_ms(16, sim):.0f} ms profiled")
    print(f"  measured median {med:.0f} ms unprofiled -> {med_prof:.0f} ms profiled")
    ok = lo <= med_prof <= hi
    print(f"  band [{lo:.0f}, {hi:.0f}] ms: {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("  -> the a + b*max_chunk + c*n_live law does not generalise off its fit set."
              "\n     Every predicted policy delta in the ranking is suspect; do NOT build"
              "\n     the runtime knob on this model until the miss is explained.")
    return 0 if ok else 1


def byte_diff(report, against, base_sha=None) -> int:
    a, b = json.load(open(report)), json.load(open(against))

    def seqs(r):
        out = {}
        for k in ("results", "cases", "per_case"):
            if isinstance(r.get(k), dict):
                for pid, v in r[k].items():
                    for f in ("token_ids", "engine_token_ids", "output_ids"):
                        if isinstance(v, dict) and f in v:
                            out[pid] = tuple(v[f])
        return out
    A, B = seqs(a), seqs(b)
    if not A or not B:
        print("FAIL: could not extract per-case token ids from both reports")
        return 1
    diff = [k for k in sorted(set(A) & set(B)) if A[k] != B[k]]
    print(f"stage 4 -- per-case byte diff vs {against}"
          + (f" (base {base_sha})" if base_sha else ""))
    print(f"  compared {len(set(A) & set(B))} cases, {len(diff)} differ: {diff}")
    print("VERDICT:", "PASS -- bit-exact" if not diff else
          "FAIL -- the cap is not bit-exact; stop, cast-position root-cause (M2-I4 rule)")
    return 0 if not diff else 1


def perf(root, reps=3) -> int:
    print("stage 5/6 -- base vs cap, median of reps, unprofiled clock")
    print(f"{'bs':>3s} {'base_ms':>9s} {'cap_ms':>9s} {'gain':>8s} {'pred_gain':>10s} "
          f"{'base_iters':>11s} {'cap_iters':>10s} {'pred_cap_iters':>15s}")
    rc = 0
    for bs in (1, 2, 4, 8, 16):
        vals = {}
        for arm in ("base", "cap"):
            g = [_timings(p)["waves"][0]["wall_ms"] for p in sorted(
                glob.glob(os.path.join(root, f"s5_{arm}_bs{bs}_rep*", f"timings_bs{bs}.json")))]
            vals[arm] = statistics.median(g) if g else None
        if not all(vals.values()):
            continue
        s0 = simulate(ac3_slots(bs), 16, 132)
        s1 = simulate(ac3_slots(bs), 16, 132, cap=max(1, 16 // bs))
        pred = wave_ms(bs, s0) / wave_ms(bs, s1) - 1
        got = vals["base"] / vals["cap"] - 1
        print(f"{bs:3d} {vals['base']:9.0f} {vals['cap']:9.0f} {got:+7.1%} {pred:+9.1%} "
              f"{s0['n_iterations']:11d} {'?':>10s} {CAP_ITERS[bs]:15d}")
        if bs == 16:
            lo, hi = CAP_BS16_MS_UNPROF
            ok = lo <= vals["cap"] <= hi
            print(f"    C5 band [{lo:.0f}, {hi:.0f}] ms unprofiled: {'PASS' if ok else 'FAIL'}")
            rc |= 0 if ok else 1
    return rc


def matched(root, reps=3, vllm=None) -> int:
    print("stage 7 -- matched 256/1024 geometry (remeasure-protocol.md)")
    for bs in (1, 2, 4, 8, 16):
        sim = simulate([256] * bs, 16, 1280)
        dec = sum(1 for it in sim["iters"]
                  if it["n_live"] == bs and it["max_chunk"] == 1 and it["n_prefill"] == 0)
        for arm in ("base", "cap"):
            g = [_timings(p)["waves"][0] for p in sorted(
                glob.glob(os.path.join(root, f"s7_{arm}_bs{bs}_rep*", f"timings_bs{bs}.json")))]
            if not g:
                continue
            med = statistics.median(x["wall_ms"] for x in g)
            print(f"  bs{bs:<3d} {arm:5s} wall={med:9.0f} ms  wave tok/s="
                  f"{bs * 1024 / (med / 1000):8.1f}  predicted decode_full iters={dec}")
            if dec == 0:
                print("        NO decode number reported for this configuration "
                      "(remeasure-protocol.md 3, rule 2)")
    if vllm:
        print(f"  vLLM baseline: {vllm} (pinned, NOT re-run)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-divergence")
    ap.add_argument("--check-isolation")
    ap.add_argument("--check-costlaw")
    ap.add_argument("--byte-diff")
    ap.add_argument("--against")
    ap.add_argument("--base-sha")
    ap.add_argument("--perf")
    ap.add_argument("--matched")
    ap.add_argument("--vllm")
    ap.add_argument("--all")
    ap.add_argument("--predictions")
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args(argv)
    rc = 0
    if a.check_divergence:
        rc |= check_divergence(a.check_divergence)
    if a.check_isolation:
        rc |= check_isolation(a.check_isolation)
    if a.check_costlaw:
        rc |= check_costlaw(a.check_costlaw, a.reps)
    if a.byte_diff:
        rc |= byte_diff(a.byte_diff, a.against, a.base_sha)
    if a.perf:
        rc |= perf(a.perf, a.reps)
    if a.matched:
        rc |= matched(a.matched, a.reps, a.vllm)
    if a.all:
        for f in sorted(glob.glob(os.path.join(a.all, "*", "timings_bs16.json"))):
            print("--", f)
        rc |= perf(a.all, a.reps)
        rc |= matched(a.all, a.reps)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
