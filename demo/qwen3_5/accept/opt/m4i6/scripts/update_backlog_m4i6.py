#!/usr/bin/env python3
"""M4-I6: close the MoE-router lever in opt/backlog.json with the measured result.

The entry being closed is `new_levers_for_m4` -> "MoE router top-k/softmax (task
260)", opened by M3-I7's milestone gate with disposition "OPEN -> M4" and the
note that the stage "has never had a kernel-level look" because I5c knowingly
traded speed for correctness and "that trade has never been re-costed". This
script writes the re-costing.

It only touches that one entry plus a `m4i6` provenance block, and it refuses if
the entry is not found -- a silent no-op backlog update is worse than none.
"""
import argparse
import json
import sys

LEVER = "MoE router top-k/softmax (task 260)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backlog", required=True)
    ap.add_argument("--tables-json", required=True,
                    help="tables/m4i6_tables.json from tables_m4i6.py")
    ap.add_argument("--stage-json", default=None,
                    help="tables/stage_wallspan.json from stage_tables.py")
    ap.add_argument("--ptxas", default=None,
                    help="'A_regs,B_regs,A_spill,B_spill' summary string")
    ap.add_argument("--ac3", default=None, help="one-line AC-3 verdict summary")
    ap.add_argument("--sha", required=True, help="integration commit sha")
    ap.add_argument("--disposition", required=True)
    a = ap.parse_args()

    d = json.load(open(a.backlog))
    tab = json.load(open(a.tables_json))
    stage = json.load(open(a.stage_json)) if a.stage_json else None

    target = None
    for L in d.get("new_levers_for_m4", []):
        if L.get("lever") == LEVER:
            target = L
            break
    if target is None:
        sys.exit(f"REFUSING: lever {LEVER!r} not found in new_levers_for_m4")

    e2e = {str(r["bs"]): {
        "A_median_ms": round(r["A_median_ms"], 1),
        "B_median_ms": round(r["B_median_ms"], 1),
        "speedup": round(r["speedup"], 4),
        "pct": round(100 * (r["speedup"] - 1), 2),
        "A_reps_ms": [round(x, 1) for x in r["A_reps"]],
        "B_reps_ms": [round(x, 1) for x in r["B_reps"]],
    } for r in tab["rows"]}

    st = {}
    if stage:
        for r in stage["rows"]:
            st[str(r["bs"])] = {
                "per_task_us_A": round(r["per_task_us_A"], 3),
                "per_task_us_B": round(r["per_task_us_B"], 3),
                "per_task_speedup": round(r["per_task_speedup"], 3),
                "wallspan_us_A": round(r["span_us_A"], 1),
                "wallspan_us_B": round(r["span_us_B"], 1),
                "mean_conc_B": round(r["mean_conc_B"], 2),
                "recovered_us_own_basis": (
                    round(r["recovered_us_own_basis"], 1)
                    if "recovered_us_own_basis" in r else None),
                "recovered_frac_of_m4i5_842": (
                    round(r["recovered_frac_of_m4i5_842"], 3)
                    if "recovered_frac_of_m4i5_842" in r else None),
                "vllm_floor_ratio_B": (round(r["vllm_floor_ratio_B"], 2)
                                       if r.get("vllm_floor_ratio_B") else None),
                "m4i5_target_met": r.get("m4i5_target_met"),
            }

    target["disposition"] = a.disposition
    target["closed_by"] = "M4-I6 (2026-07-30)"
    target["closed_sha"] = a.sha
    target["result"] = {
        "kernel": ("ferret workspace5 tag v013 (f370cbb), min_ratio 1.417 vs the "
                   "FlashInfer TRT-LLM routing kernel: 141.7/150.5/145.6/170.7/"
                   "170.3 % of its throughput at N_LIVE 1/2/4/8/16, i.e. 29.4-41.4 % "
                   "less time per call"),
        "e2e_ab": e2e,
        "stage": st,
        "ptxas": a.ptxas,
        "ac3": a.ac3,
        "mechanism": (
            "I5c's compaction cost is re-costed and reclaimed: the per-thread "
            "serial tile scan (NUM_EXPERTS global loads per thread, all 256 "
            "threads) becomes a warp-0 popcount over a shared-memory active-expert "
            "bitmask written at the same guarded mark sites, and the rest of the "
            "win is latency work on the single serialized CTA -- load prologue "
            "hoisted above the init loop, init moved off warp 0, the "
            "init-visibility barrier relocated below tile 0's compute with the "
            "rank/mark writes deferred and distributed across k lanes, a per-lane "
            "bitonic sort-16 of packed order-preserving 64-bit keys replacing the "
            "per-round rescan, and padding rows skipping load and compute. "
            "Concurrency stays at ~1.0 by construction (one task per layer, "
            "grid=(1,1,1)), so unlike M4-I2's dense stage there is no width term: "
            "what the kernel saves the step collects directly."),
        "residual": (
            "The stage is not at the vLLM per-call floor M4-I5 named (3.697 us/task "
            "at bs1). Further per-call latency work faces the CTA's own fixed "
            "entry/exit + gating-load round trip, which the ferret loop's own "
            "diagnostic ladder identified as the config-independent floor; the "
            "structural alternative is splitting the row loop across task "
            "instances, which needs a cross-task merge of the active-expert "
            "compaction that does not exist as a pattern in this codebase."),
    }

    d.setdefault("m4i6", {})["router_integration"] = {
        "sha": a.sha,
        "evidence": "demo/qwen3_5/accept/opt/m4i6/",
        "ferret_tag": "v013 (f370cbb) in ferret/workspace5",
        "note": ("The loop was still running on workspace5 when the tag was taken; "
                 "the import is the TAG BLOB (git show v013:kernel.cu), never the "
                 "worktree file -- the M4-I2 lesson that a live workspace carries "
                 "unfinished probe code. The tag's frozen `golden` block was "
                 "verified byte-identical to the 413-line body it replaced before "
                 "any of it was imported."),
    }

    json.dump(d, open(a.backlog, "w"), indent=1)
    print(f"backlog updated: {LEVER!r} -> {a.disposition}")
    for bs, v in sorted(e2e.items(), key=lambda x: int(x[0])):
        print(f"  bs{bs}: {v['A_median_ms']} -> {v['B_median_ms']} ms "
              f"({v['pct']:+.2f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
