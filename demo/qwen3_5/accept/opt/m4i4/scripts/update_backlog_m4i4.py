#!/usr/bin/env python3
"""M4-I4 -- record the landed admission-cap policy in opt/backlog.json.

Idempotent: re-running rewrites the same two entries with the same content and
asserts they exist first, so a rename upstream fails loudly instead of silently
appending a duplicate lever.

Two entries change.

1. `new_levers_for_m4` "MPK prefill throughput at the 256/1024 workload" -- its
   `first_step` ("extend the admission cap to bs4/bs8") is DISCHARGED; the landed
   numbers and the remaining step (mbt for the prefill phase) go in, and the
   AC-5 consequence is stated as the decode requirement it implies.

2. `levers` rank 8 "GDN prefill chunked-matmul (WY/UT)" -- its disposition is
   RE-DERIVED. It was rejected because its precondition (more tokens per
   iteration) went the other way; that reasoning was about mbt, and the cap has
   since landed, so the rejection is restated on the new measurement rather than
   inherited.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ACC = Path(__file__).resolve().parents[3]        # .../demo/qwen3_5/accept
BACKLOG = ACC / "opt" / "backlog.json"

PREFILL_LEVER = "MPK prefill throughput at the 256/1024 workload"
WYUT_RANK = 8

PREFILL_UPDATE = {
    "first_step_status": (
        "DISCHARGED by M4-I4. The admission cap now lands from bs2 up (not bs4 -- "
        "bs2 was measured, not argued: 1.283x prefill / +1.4% e2e with "
        "non-overlapping per-rep sets), and the policy is CODE at "
        "accept/admission_policy.py rather than prose in bench-protocol.md."),
    "landed_numbers_256_1024": {
        "prefill_speedup": {"1": 1.000, "2": 1.283, "4": 1.447, "8": 1.727,
                            "16": 2.714},
        "e2e_delta_pct": {"1": 0.0, "2": 1.4, "4": 4.0, "8": 14.1, "16": 83.4},
        "decode_delta_pct": {"1": 0.0, "2": 0.1, "4": 0.7, "8": 5.0, "16": 61.8},
        "prefill_s_landed": {"1": 0.311, "2": 0.479, "4": 0.822, "8": 1.701,
                             "16": 3.040},
        "prefill_frac_of_e2e_landed_pct": {"1": 3.0, "2": 4.5, "4": 7.4,
                                           "8": 13.5, "16": 19.7},
        "basis": ("3 reps per arm, a compiled kernel per arm (the cap is a "
                  "compile-time define), both arms interleaved inside ONE GPU "
                  "claim, drain-gated with a per-run device audit. "
                  "opt/m4i4/tables/geomM.csv"),
    },
    "mechanism_correction": (
        "The recorded mechanism -- 'uncapped admission SERIALISES prefill, 1887 "
        "iterations against 1279' -- is the bs16 term only. Below bs16 the "
        "replay's iteration counts do not predict the win at all: 1057 vs 1055 at "
        "bs2 and 1094 vs 1087 at bs4 at 256/1024, and at msl=132 the CAPPED arm "
        "needs MORE iterations at bs2 (505 vs 498) and bs4 (318 vs 308) while "
        "being faster, and exactly as many at bs8 (228/228) while being 1.166x "
        "faster. The term that pays below bs16 is GRAPH WIDTH per iteration: the "
        "cap drops the widest per-slot chunk from mbt to mbt/bs, so the same token "
        "budget activates more requests, and MPK's iteration cost is set by the "
        "widest chunk (opt/m3i9/cost_model.py). At bs16 the iteration term is "
        "1.55x of the measured 1.83x."),
    "ac5_consequence": (
        "AC-5 still FAILS at every bs (mpk e2e 2.85/2.74/2.50/2.55/2.77x vLLM "
        "against a 1.25x bound) because the e2e ratio is dominated by the decode "
        "gap. What the cap changed is how much of AC-5's slack prefill spends: at "
        "the landed prefill cost AC-5 holds at 0.85/0.88/0.92x of vLLM's decode "
        "throughput at bs1/2/4 -- so AC-4 implies AC-5 there -- but needs 1.075x "
        "at bs8 and 1.381x at bs16. Uncapped, bs4 required 0.998x (AC-4 and AC-5 "
        "were the same bar) and bs16 was UNSATISFIABLE at any decode speed, since "
        "prefill alone (8.252 s) exceeded 1.25 x vLLM's whole e2e (6.960 s)."),
    "first_step": (
        "REMAINING: mbt for the prefill phase. The cap extracted the parallelism "
        "available inside a fixed 16-token budget and at bs16 that budget is now "
        "one token per request, so there is no width left to win this way. The "
        "next prefill lever is the budget itself; M3-I5b rejected raising mbt on "
        "DECODE evidence, which does not bind the prefill phase."),
    "disposition": ("PARTIALLY DISCHARGED by M4-I4 (the admission-cap half). "
                    "OPEN -> mbt-for-prefill, still ranked 1 by e2e effect at "
                    "bs8/bs16."),
    "closed_by_partial": "M4-I4 (2026-07-29)",
}

WYUT_DISPOSITION = (
    "rejected-with-evidence, RE-DERIVED by M4-I4 (2026-07-29) on a new basis. The "
    "M3 rejection said the precondition (more tokens per iteration) went the other "
    "way; that was an mbt argument made before the cap landed and before prefill "
    "was measured at the pinned geometry, so it is re-derived rather than "
    "inherited. Three findings, in the order that decides it. (1) Prefill IS still "
    "binding: after the cap it is 13.5%/19.7% of e2e at bs8/bs16 and still costs "
    "AC-5 a 1.075x/1.381x decode requirement, so the lever is not dead on "
    "relevance. (2) But WY/UT is not the shape of the remaining cost. It is a "
    "CHUNKED-MATMUL algorithm -- it wins by doing the delta rule as block-parallel "
    "matmuls over many tokens at once -- and the per-request chunk size under the "
    "landed policy is mbt/bs = 1 token at bs16, 2 at bs8. The cap REDUCED the "
    "per-request chunk to the minimum, and it did so because narrow chunks are "
    "what makes MPK fast here (the graph-width mechanism above). A chunked-matmul "
    "algorithm at chunk size 1 is the sequential recurrence plus bookkeeping: the "
    "winning direction is the opposite one, and that is now measured rather than "
    "argued. (3) Its numerics gate does not inherit. M3-I9b proved chunk "
    "boundaries bit-transparent in the CURRENT kernels (identical logit rows, "
    "bit-identical per-layer GDN conv and fp32 recurrent state at bs1/bs4 "
    "cap-vs-base, H1-H4 refuted with source citations), but that is about "
    "re-chopping the SAME per-token sequential recurrence. WY/UT is a "
    "block-parallel delta rule with a different summation order and different "
    "intermediate precision, so transparency cannot transfer; it would need its "
    "own bit-exactness gate against the HF chunk algorithm as oracle plus its own "
    "AC-3 sweep. For scale, the cap -- a one-min() change WITH a proven "
    "transparency argument -- still needed 300 cases to certify. RE-OPEN ONLY IF "
    "mbt RISES (the budget, not its division): larger chunks are the only thing "
    "that restores the precondition, so WY/UT must be sequenced strictly after an "
    "mbt-for-prefill landing, never before, and must carry its own oracle gate. "
    "Evidence: opt/m4i4/README.md sections 2d, 3 and 5; opt/m3i9b/ (transparency "
    "of the current kernels); opt/m3i5b/ (the mbt rejection, decode-side)."
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backlog", default=str(BACKLOG))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    p = Path(a.backlog)
    d = json.loads(p.read_text())

    news = [lv for lv in d["new_levers_for_m4"] if lv["lever"] == PREFILL_LEVER]
    assert len(news) == 1, f"expected exactly one {PREFILL_LEVER!r}, got {len(news)}"
    news[0].update(PREFILL_UPDATE)

    wy = [lv for lv in d["levers"] if lv.get("rank") == WYUT_RANK]
    assert len(wy) == 1, f"expected exactly one rank-{WYUT_RANK} lever"
    assert "WY/UT" in wy[0]["lever"], wy[0]["lever"]
    wy[0]["disposition"] = WYUT_DISPOSITION
    wy[0]["disposition_kind"] = "rejected-with-evidence"
    tail = ("re-derived by M4-I4 (2026-07-29, admission cap landed at "
            "348a601a/1f6848d0)")
    prev = wy[0].get("closed_by", "")
    # idempotent: never append the same provenance twice
    wy[0]["closed_by"] = prev if tail in prev else f"{prev}; {tail}".lstrip("; ")
    wy[0]["reopen_condition"] = "mbt rises for the prefill phase"

    d["m4_admission_cap_policy"] = {
        "authority": "demo/qwen3_5/accept/admission_policy.py",
        "landed": "auto from bs2 up; bs1 uncapped (auto == mbt there, provable no-op)",
        "issue": "M4-I4",
        "ac3": ("300/300 cases byte-identical to results/dumps_final across 2 arms "
                "x 3 reps x 5 bs x 10 prompts, plus the cold fingerprint gate at "
                "the shipped policy"),
    }

    out = json.dumps(d, indent=1) + "\n"
    if a.dry_run:
        print(out[:400])
        print("...")
        return 0
    p.write_text(out)
    print(f"updated {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
