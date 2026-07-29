#!/usr/bin/env python3
"""M3-I7 -- put every backlog lever into a TERMINAL disposition and register the
levers M3 discovered that were never on the list.

Terminal means one of exactly three things, each with the evidence attached:
  integrated             -- landed at HEAD, with the measured effect
  rejected-with-evidence -- measured or derived NOT to be worth taking, with the
                            mechanism, so nobody re-tries it blind
  blocked-with-reason    -- genuinely worth taking, not takeable inside M3, with
                            what blocked it and what M4 needs to unblock it

Idempotent: re-running rewrites the same terminal text from the same inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CLOSE = "M3-I7 (milestone gate, 2026-07-29, integrated HEAD c80ebd68)"

TERMINAL = {
1: ("integrated",
    "M3-I2b landed the quantize row-partition (input_map was unpartitioned -> 16x "
    "redundant work): +28.3% bs1, +18-31% at every bs, tokens bit-identical. Confirmed "
    "closed at the gate: the re-derived per-stage table puts quantize at 0.95/0.88/0.96x "
    "of vLLM at bs1/8/16 -- MPK is now AHEAD of the reference kernel at bs8 and at parity "
    "elsewhere, so there is no gap left to buy. Evidence opt/m3i2b/, "
    "opt/m3i7/stage/armL_m3i10/tables/comparison_by_stage.csv."),
2: ("integrated",
    "M3-I8 v1 (gate_padding_rows, default ON since 46872ad) landed: real-hardware A/B "
    "+9.7/+15.3/+7.1/+25.1/-0.0% at bs1/2/4/8/16, AC-3 byte-identical 10/10 at all five "
    "bs. The v2a/v2b grid-widen half is REJECTED-WITH-EVIDENCE: composing it with v1 gave "
    "+24.6% at bs1 but made bs8 WORSE than v1 alone (+19.6% vs +25.1%) because splitting "
    "returns bs8 to two waves -- a bs-conditional grid for a bs1-only gain is not worth "
    "the branch. Residual at the gate: MoE w13 is still rank 1 (6.96/2.85/2.56x), but "
    "that residual is kernel-side (see the ferret target), not activation right-sizing. "
    "Evidence opt/m3i8/results/VALIDATION.md."),
3: ("blocked-with-reason",
    "Graph width is still the largest structural residual and M3 only chipped at it: I3 "
    "split GDN into separate tasks, I8 gated MoE pad rows, I6a removed the register spill "
    "that was throttling every task. What remains is a per-stage task-splitting programme "
    "-- I3 alone took a full ferret campaign plus an integration issue for ONE stage -- "
    "and it does not fit in M3's remaining budget. It is not rejected: the re-derived "
    "table shows MPK still 2.2-2.8x off vLLM overall while being AHEAD on every fused "
    "stage, which is the signature of a width problem, not a kernel-quality problem. "
    "Carried to M4 with the split-KV attention lever (M3-I6a) as the first concrete "
    "instance. Evidence opt/m3i7/stage/, opt/m3i3/, opt/m3i6a/."),
4: ("integrated",
    "The per-request admission cap landed (74df41f, policy pinned in "
    "docs/qwen35/bench-protocol.md) and the gate re-validated it at integrated HEAD with "
    "PER-ARM COMPILED KERNELS -- necessary because the knob is a compile-time define, and "
    "sharing one kernel dir between arms silently measures one binary twice. bs16: 2.10x "
    "at the AC-3 geometry and +64.6% decode / +86.0% e2e at the pinned 256/1024 geometry, "
    "both LARGER than M3-I9's original +84.2%/+14.1%. Mechanism confirmed from the "
    "adapter's own admission replay: uncapped, the budget goes to the lowest live slot "
    "first, so requests prefill almost serially (1887 wave iterations vs 1279 capped). "
    "NEW at the gate: the same mechanism pays at bs4 (+3.9% e2e) and bs8 (+14.0% e2e, "
    "+4.8% decode), AC-3 byte-identical 10/10 at both -- so the policy's bs<16 exclusion "
    "is superseded (see the bench-protocol amendment). Evidence "
    "opt/m3i7/tables/cap_policy.json."),
5: ("integrated",
    "M3-I3 ported the ferret v010 GDN recurrent kernel bit-exact (b0920b28): stage "
    "wallspan 1217->236 / 2469->782 / 4950->1556 us at bs1/8/16 = e2e +9.0/+15.2/+28.1% "
    "tok/s, unit memcmp + oracle both lanes + AC-3 10/10 at five bs. Re-derived at the "
    "gate the stage now sits at 1.35/2.70/3.14x of vLLM (was 7.44/9.12/10.67x), the "
    "largest single ratio improvement in M3. The remaining bs8/bs16 ratio is width, not "
    "kernel: per-worker task time was already 0.98x vLLM's at bs16. Evidence "
    "opt/m3i3/results/."),
6: ("blocked-with-reason",
    "Dense fp8 blockscale (task 279) is rank 2 in the re-derived table and the most "
    "batch-INDEPENDENT gap left (2.07/1.98/2.08x, ~1.8 ms/step at every bs). M3-I4's "
    "ferret loop was still climbing at min_ratio 0.680 when the box's Claude weekly quota "
    "ran out (reset 2026-07-29), and no in-MPK change landed, so there is nothing to "
    "integrate and nothing to reject. Unblocked by the quota reset; carried to M4 with "
    "the loop's current best ratio as its starting point. Evidence opt/m3i4/, "
    "the ferret episode ledger at 5cedb43c."),
7: ("integrated",
    "M3-I6a set attention max_tokens_per_pass 4->2 (a86b1eb1). The knob turned out not to "
    "be an attention question at all: pass 4 was the megakernel's ONLY register-spill "
    "source (255 regs, 576 B stack, 780/976 B spill st/ld across the whole inlined "
    "kernel), so pass 2 sped up untouched task families too. Attention wallspan "
    "-48.0/-51.4%, e2e +2.1-3.1% at the AC-3 geometry and +4.6-6.2% at deep context. "
    "Attention is now rank 5 (4.50/4.63/4.98x) rather than the 8-10x the previous table "
    "showed -- most of that apparent improvement is the corrected basis, not the change. "
    "NEXT LEVER recorded, untaken: split-KV k-way via the I3 separate-task idiom. "
    "Evidence opt/m3i6a/."),
8: ("rejected-with-evidence",
    "GDN prefill chunked-matmul (WY/UT) was conditional on the admission policy letting "
    "many tokens into one iteration, and that precondition went the OTHER way: I5b "
    "rejected raising mbt with evidence, and the cap that DID land moves tokens-per-"
    "request-per-iteration DOWN, not up. At mbt=16 a prefill iteration costs 1.16-1.30x a "
    "decode iteration, so there is no WY/UT-shaped win available -- the prefill cost that "
    "does matter is scheduling (see lever 4 and the new prefill-throughput entry), not "
    "the chunked-matmul algorithm. Re-open only if mbt rises. Evidence "
    "opt/attribution.csv prefill_step_us, opt/m3i5b/ (mbt rejection)."),
9: ("rejected-with-evidence",
    "prepare_next_batch is 4.2/6.9/11.1/20.9/35.6 us per step at bs1..16, at most 0.16% "
    "of the step. Removing all of it is inside measurement noise. Unchanged at the gate. "
    "Evidence opt/attribution.csv of_which_prepare_batch_us."),
10: ("rejected-with-evidence",
     "MoE dead-task dispatch: 5652/10240 w13 and 5661/10240 w2 tasks complete in under "
     "1 us at bs1, 5.9 ms of aggregated worker time = 46 us per worker per step = 0.3% of "
     "the step. Dispatching a no-op task is cheap; the cost was in the wrongly-live "
     "experts, which lever 2 fixed. Evidence opt/pertask_by_bs.csv."),
11: ("integrated",
     "MEASUREMENT DEBT DISCHARGED by this issue, and it was worse than recorded. The "
     "256/1024 numbers now exist and are in opt/m3i7/tables/geomM_matched_256_1024.csv. "
     "Two defects found and fixed on the way: (a) M3-I9's stage-7 'matched 256/1024' run "
     "measured the AC-3 REFERENCE prompts, not 256-token prompts -- mpk_engine_run.py's "
     "--prompts-file is only read under --verify-chat-template, so the run fell back to "
     "--reference, and its committed timings record prompt_ids ['p06-poem'] with "
     "max_decode_steps 1255 while the analysis divided by a hardcoded 1024; (b) it "
     "reported bs*1024/wave_wall, which bills the prefill to decode. The gate uses a real "
     "256-token prompt source (opt/m3i7/scripts/make_matched_reference.py, the pinned "
     "baseline sampler and seed) and a prefill-subtracted slope, vLLM's own definition."),
}

NEW_LEVERS = [
    {"lever": "MPK prefill throughput at the 256/1024 workload",
     "found_by": CLOSE,
     "evidence": "Measured directly for the first time: the prefill-only arm (same "
                 "prompts, msl=259, n=3) costs 326/624/1190/2940/3041 ms at "
                 "bs1/2/4/8/16 under the pinned policy = 3.2/5.8/10.3/20.4/20.0% of the "
                 "whole 256/1024 e2e, rising to 29.2% at bs16 uncapped. The scale that "
                 "matters is against the REFERENCE, not against ourselves: at bs8 MPK's "
                 "prefill ALONE (2.94 s) is 59% of vLLM's entire end-to-end time "
                 "(4.95 s), and at bs16 it is 55%. No M3 backlog item covers prefill: "
                 "every M3 measurement was a decode step. "
                 "opt/m3i7/tables/geomM_matched_256_1024.csv.",
     "mechanism": "mbt=16 caps the whole engine at 16 prefill tokens per iteration, so a "
                  "256-token prompt needs >=16 iterations and bs*256 tokens need "
                  "bs*16 iterations at best -- and uncapped admission makes it far worse "
                  "than that best case by serialising requests (lever 4's mechanism). "
                  "vLLM prefills the same tokens in large fused chunks.",
     "first_step": "Extend the admission cap to bs4/bs8 (already measured, +3.9%/+14.0% "
                   "e2e, AC-3 byte-identical), then attack mbt itself for the prefill "
                   "phase only -- M3-I5b rejected raising mbt on DECODE evidence, which "
                   "does not bind the prefill phase.",
     "disposition": "OPEN -> M4, ranked 1 by e2e effect"},
    {"lever": "Split-KV k-way attention (the I3 separate-task idiom applied to task 257)",
     "found_by": "M3-I6a, re-ranked by " + CLOSE,
     "evidence": "Attention is rank 5 in the re-derived table (4.50/4.63/4.98x vLLM, "
                 "501/518/628 us/step of gap) and it is the only target whose ratio GROWS "
                 "with batch size. vLLM's own reference is a split-KV kernel "
                 "(MultiCtasKv). I6a predicted 39 us vs 75 us at ctx 848, k=8.",
     "mechanism": "MPK runs 2 attention tasks per layer, so the stage occupies 2 of 148 "
                  "SMs however long it takes; splitting the KV range k ways multiplies "
                  "the available parallelism by k.",
     "first_step": "ferret task spec against the vLLM MultiCtasKv reference at ctx 848.",
     "disposition": "OPEN -> M4"},
    {"lever": "MoE routed GEMM w13/w2 kernel (tasks 241/242)",
     "found_by": "M3-I1, re-ranked by " + CLOSE,
     "evidence": "Ranks 1 and 3 in the re-derived table: w13 6.96/2.85/2.56x "
                 "(2092/1847/2039 us/step of gap), w2 4.37/2.00/2.03x. Together they are "
                 "~3.2 ms of a 9.8 ms bs1 step. The bs1 ratio is far worse than bs8/bs16, "
                 "which is the tensor-core-waste signature the ferret MoE task authoring "
                 "already identified: BATCH_SIZE=16 is compile-fixed, so NUM_M_TILES=1 "
                 "and every activated expert pays a full 16-row MMA tile (~94% waste at "
                 "bs1).",
     "mechanism": "compile-fixed M tile vs live token count; distinct from lever 2, which "
                  "fixed which experts are activated rather than what each costs.",
     "first_step": "the authored ferret MoE task (workspace3, launch line staged, "
                   "target_ratio 1.3333) once the box quota resets.",
     "disposition": "OPEN -> M4"},
    {"lever": "MoE router top-k/softmax (task 260)",
     "found_by": CLOSE,
     "evidence": "Rank 4 and the biggest RATIO among the non-trivial stages: "
                 "5.70/4.93/4.12x vLLM, 731/770/801 us/step of gap, and it got WORSE "
                 "relative to the previous table (3.36->5.70x at bs1) because the "
                 "corrected window exposes it. 842-980 us/step for routing 1-16 tokens "
                 "through a top-8-of-256 softmax is far above any plausible bound.",
     "mechanism": "not yet attributed. The stage was reshaped twice in M3 (I5b's row loop, "
                  "I5c's compaction race fix, which cost +51-61% on the task by design) "
                  "and has never had a kernel-level look.",
     "first_step": "attribute before optimising -- I5c knowingly traded speed for "
                   "correctness here and that trade has never been re-costed.",
     "disposition": "OPEN -> M4"},
    {"lever": "schedule_sim diverges from the runtime at bs16/msl=897",
     "found_by": CLOSE,
     "evidence": "The exact-replay admission model predicts 1360 iterations; the trace "
                 "has 1004. bs1 (656) and bs8 (810) match exactly. Anchor QC on that "
                 "window fails (max_frac_err 0.4437 vs a 0.02 threshold, 16 task types "
                 "mismatched), so bs16's per-stage row is flagged not-valid rather than "
                 "used. opt/m3i7/stage/qc/anchor_qc_summary.json.",
     "mechanism": "unknown. It is a measurement-infrastructure defect, not a performance "
                  "lever, but it blinds every bs16 per-stage number.",
     "first_step": "diff the replay against the runtime's own iteration labels at "
                   "bs16/msl=897; the iters.csv the parse already writes has both.",
     "disposition": "OPEN -> M4 (blocks trustworthy bs16 stage attribution)"},
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backlog", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    p = Path(a.backlog)
    d = json.loads(p.read_text())

    for lever in d["levers"]:
        r = lever["rank"]
        if r not in TERMINAL:
            raise SystemExit(f"rank {r} has no terminal disposition")
        kind, text = TERMINAL[r]
        prior = lever.get("disposition")
        if prior and not prior.startswith(("integrated", "rejected-with-evidence",
                                           "blocked-with-reason")):
            lever["disposition_before_m3i7"] = prior
        lever["disposition"] = f"{kind}: {text}"
        lever["disposition_kind"] = kind
        lever["closed_by"] = CLOSE

    d["new_levers_for_m4"] = NEW_LEVERS
    kinds = {}
    for lever in d["levers"]:
        kinds[lever["disposition_kind"]] = kinds.get(lever["disposition_kind"], 0) + 1
    d["m3_close"] = {
        "closed_by": CLOSE,
        "all_terminal": True,
        "counts": kinds,
        "assertion": ("every lever in `levers` carries one of integrated / "
                      "rejected-with-evidence / blocked-with-reason with its evidence; "
                      "asserted by opt/m3i7/scripts/close_backlog.py at generation time"),
        "n_new_levers_registered": len(NEW_LEVERS),
    }
    out = Path(a.out) if a.out else p
    out.write_text(json.dumps(d, indent=1) + "\n")
    print(f"wrote {out}")
    for lever in d["levers"]:
        print(f"  rank {lever['rank']:2d}  {lever['disposition_kind']:24s} {lever['lever'][:58]}")
    print(f"  counts: {kinds}")
    print(f"  new levers registered for M4: {len(NEW_LEVERS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
