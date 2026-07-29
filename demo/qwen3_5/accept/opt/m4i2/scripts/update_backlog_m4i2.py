#!/usr/bin/env python3
"""M4-I2: move backlog rank 6 (dense fp8 blockscale, task 279) to INTEGRATED.

Appends to the existing disposition rather than replacing it -- the same
convention ranks 3 and 7 already use -- so the M3 reasoning that blocked the
lever stays readable next to the M4 outcome that closed it.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKLOG = os.path.abspath(os.path.join(HERE, "..", "..", "backlog.json"))

RANK = 6
MARK = "M4-I2 (2026-07-29) TERMINAL"

APPEND = (
    "  || M4-I2 (2026-07-29) TERMINAL: INTEGRATED. The ferret loop finished at "
    "workspace4 tag v011 (min_ratio 1.011 over 30 shape/M configs vs vLLM's "
    "cutlass_3x_gemm_fp8_blockwise; worst outproj_M2 101.1%, best gdnz_M4 "
    "128.1%) and this issue ported it into task 279. The pre-existing kernel is "
    "preserved BYTE-FOR-BYTE as linear_fp8_blockscale_task_impl_golden and the "
    "entry point is now a compile-time dispatcher; the port is bit-exact against "
    "it -- 30/30 shape/bs cases in both nvcc flag lanes AND both a torch-free "
    "12.8 build (the shipped JIT toolchain) and the 13.0 pybind harness, with "
    "ZERO differing elements even on deliberately-inexact data. AC-3 STABLE at "
    "all five bs, 15/15 cold reps accepted, 0 quarantined, byte-identical to "
    "results/dumps_final; tokens_sha256 also matched between arms in all 15 e2e "
    "(bs,rep) pairs. ptxas unchanged vs pre-M4-I2 HEAD: 238 registers, 144 B "
    "stack, 0 spill st/ld -- so the M3-I6a class of cross-stage regression does "
    "not apply. e2e (3 reps/arm/bs, interleaved in one GPU claim, HEAD control "
    "within 0.01% of the baseline arm): +10.0/+10.0/+8.5/+7.2/+8.2% at bs "
    "1/2/4/8/16. THE STRUCTURAL CHANGE was a per-task N slice FINER than the "
    "checkpoint's 128-row scale block (16/32/64 per shape); MPK splits inputs by "
    "integer division, so the weight scale is attached row-replicated to one row "
    "per task and linear_fp8_blockscale_layer now asserts that, closing a "
    "silent-wrong-scale hole for every caller. WHAT THIS CLOSES AND WHAT IT "
    "HANDS ON: stage wallspan 2813.8->1483.7us at bs1 (1.896x) and "
    "2851.1->1579.6us at bs16 (1.805x), which against M3-I7's 2.07x stage "
    "deficit puts dense fp8 at roughly PARITY with vLLM -- the gap this rank "
    "tracked is gone. The win is WIDTH, not less work: aggregate per-task work "
    "ROSE 1.24x (the slice multiplies the task count ~3.2x and each task re-pays "
    "a fixed prologue) while mean concurrency during the stage went 70.8->113.0 "
    "at bs1 and 39.0->94.7 at bs16 of 128. Residual: the stage is now only 17.1% "
    "of the step at bs1 and 14.1% at bs16, so driving task 279 to ZERO would "
    "leave 7171.8 of 8655.5us and 9658.3 of 11237.9us standing -- FURTHER KERNEL "
    "WORK ON 279 IS EXHAUSTED AS A LEVER. About half of what the stage still "
    "costs is width (660.5us = 44.5% at bs1, 837.3us = 53.0% at bs16), and the "
    "largest width deficit now visible in the same profile is "
    "TASK_MOE_W13_FP8_BLOCKSCALE_SM100 (span 2343.8us at mean concurrency 95.8) "
    "-- rank 3 / M4-I5's territory, and consistent with M4-I5's finding that the "
    "compiled graph's CRITICAL PATH, not width, is the AC-4 residual. NOT DONE "
    "HERE: prefill keeps the golden path (max_num_batched_tokens > 16 falls back "
    "to slice 128, because the fast path's per-warp B ring assumes TILE_M == 16); "
    "a prefill-capable fast path needs the ring generalised to WARPS_M > 1. "
    "Evidence opt/m4i2/."
)


def main():
    d = json.load(open(BACKLOG))
    levers = d["levers"]
    for lever in levers:
        if lever.get("rank") != RANK:
            continue
        cur = lever.get("disposition", "")
        if MARK in cur:
            print(f"rank {RANK} already carries the M4-I2 note; nothing to do")
            return 0
        if not cur.startswith("blocked-with-reason"):
            print(f"REFUSING: rank {RANK} disposition does not start "
                  f"'blocked-with-reason' -- it reads {cur[:60]!r}. The lever "
                  f"this issue closed is not where it was expected; resolve by "
                  f"hand.", file=sys.stderr)
            return 2
        lever["disposition"] = cur + APPEND
        json.dump(d, open(BACKLOG, "w"), indent=1)
        print(f"rank {RANK} updated ({len(cur)} -> {len(lever['disposition'])} chars)")
        return 0
    print(f"REFUSING: no lever with rank {RANK}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
