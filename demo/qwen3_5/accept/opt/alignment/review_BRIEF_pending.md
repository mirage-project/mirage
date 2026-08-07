# REVIEW TASK: adversarially review three conclusions about the standalone-vs-MPK gap

You are the independent reviewer. Your job is to try to REFUTE the claims below, then render a
verdict per claim. The coordinator (a different model) drew these conclusions and the human
challenged one of them; do not defer to either — check the artifacts.

READ-ONLY analysis. No GPU work. Do NOT write anywhere except /var/tmp/review_concl/. Do not
touch /var/tmp/combined (a gate run is live there — reading its REPORT.md is fine) or modify
anything under /var/tmp/alignment (read freely). Deliverable: /var/tmp/review_concl/REVIEW.md.

## The claims under review (as published to the human, verbatim in substance)

- C1 "Two stacked losses: the standalone→in-MPK conversion is ≈×0.53, eaten by cross-worker
  L2/DRAM CONTENTION among 136 heterogeneous workers; then the surviving in-MPK per-task win
  converts ≈×0 into step time at bs16 via packing/straggler reshuffle."
- C2 "Compile flags are NOT the mechanism behind the e2e dilution/sign reversal — the shipped
  fast-math regime is neutral-to-better for all three optimized kernels."
- C3 "The standalone harness STRUCTURALLY cannot see the co-tenancy effect; the right fix is
  measuring inside the real megakernel (the mpk-validator WALL-SPAN checkpoint), not a simulated
  136-worker standalone test."

## The human's challenge (translated)

"Why didn't the test discover this 136-worker L2/DRAM interference? If the test were designed
reasonably — e.g. also activating 136 workers — it should be AWARE of that interference. 'It
exists at runtime but not in the test' is not a credible story."

## Questions

Q1. What does the evidence actually prove about C1's attribution? Note the alignment report's own
caveat ("the sampled whole-GPU counters do not prove contention as the only cause"). Decompose:
which part of the ×0.53 is DEMONSTRATED (and by what), which part is plausible-but-unproven, and
what alternative mechanisms fit the same numbers (e.g. different effective occupancy/issue mix
inside the megakernel, task entry/exit overhead amortization differences, clock/power state,
instruction-cache effects in a 69k-task TU). Give the strongest DEFENSIBLE restatement of C1.

Q2. FACT-CHECK the harness geometry before judging C3: read the ws3 harness
(/home/muhengl/mpk-qwen35/ferret/workspace3/kernel.cu, task.yaml, build.sh, recent tag bodies)
and state what the timed window actually launches: how many CTAs, on how many SMs, what mix.
Then answer the human directly: could a standalone test "activating 136 workers" have predicted
the measured 9.3% (vs the claimed 17.55%)? Evaluate concretely:
  (a) homogeneous self-concurrency (N copies of the same kernel filling the GPU) — what it does
      and does not reproduce; note the counter evidence (standalone W13 2.51 TB/s @ 0.71% L2 hit
      vs in-MPK 1.0-2.0 TB/s @ 2-13% — which direction does homogeneous concurrency move these?);
  (b) a calibrated co-tenant (background noise kernel matched to in-MPK observed DRAM/L2 traffic)
      running concurrently with the candidate — feasibility, calibration fragility, and whether it
      would have flagged v024;
  (c) replaying the real heterogeneous task mix standalone (a mini-megakernel) — cost and drift;
and say whether any of these belongs in the ferret loop as a CHEAP SCREENING layer between the
pure standalone score and the periodic in-MPK validator run, or whether the validator alone is the
right design. If the human's proposal is partially right, say exactly which part.

Q3. Verify the packing/straggler claim from the raw artifacts (/var/tmp/alignment/part1/{off,on}:
cp136.json, gap136*.json, width136.json, the exact-iteration files): does "W13 realized span
+5.596% while per-task t_live −7.402%, self-concurrency 104.61→91.75, OFF work-bound → ON
cp-bound" survive scrutiny? Consider profiler distortion (the profiled step is ~8.9ms vs ~5.6ms
unprofiled), event-accounting artifacts, and whether "span" is well-defined for overlapping stages.

Q4. Render, for each of C1/C2/C3: CONFIRMED / OVERSTATED / REFUTED, plus the corrected one-
sentence statement the coordinator SHOULD publish. Lead REVIEW.md with the three verdicts.

## Pointers

- /var/tmp/alignment/REPORT.md (the full alignment report; its Part 1 + memory-counter section)
- /var/tmp/alignment/part1/{off,on}/ (raw npz, cp136/gap136/width136 json, exact-iteration checks)
- /var/tmp/alignment/lane_audit/ (two-lane builds backing C2)
- /var/tmp/moe_v024/REPORT.md (the earlier counterbalanced e2e A/B: bs16 −0.874%, 0/6)
- /var/tmp/combined/REPORT.md (four-arm result; read-only)
- /home/muhengl/mpk-qwen35/ferret/workspace3/ (the MoE loop harness under review in Q2)

## ADDENDUM (added after E1, 2026-08-06 ~10:50) — review these too

E1 ran: /var/tmp/e1_residency/REPORT.md (read it in full; logs + aggregate.json + patch beside it).
Key facts it established:
- The -19.88%/-13.55% "standalone claim" numbers were NOT produced by the ws3 harness: they come
  from the integration instrument bitexact_v024.cu at grid=128 with 205,824B dynamic smem —
  i.e. ALREADY one CTA per SM. The coordinator's E1 brief (and his earlier narrative) mis-attributed
  their provenance. The ws3 harness never timed its golden at all.
- The ws3 harness at its stock oversubscribed grids (3-4 CTAs/SM) reads -58%/-60% cand-vs-golden;
  clamped to 136 CTAs (1/SM) this collapses to -22.3% (W13) / -5.5% (W2).
- W2: the residency-matched standalone (-11.6% @ g148) lands ON the in-MPK -12.18%.
- W13: BOTH 1-CTA/SM instruments (record @128 with production path: -20.5/-20.8% fresh; E1 @136
  PATH0-pinned: -22.3%) remain far from the in-MPK t_live -7.40% — a ~13-point residual that
  residency does NOT explain.
- W2's claim anchor is soft: the record log predates the final header commit; fresh reruns give
  -18.6/-19.0%, not -13.55%.

Q5 (new, judge it hard): the OVERHEAD-DILUTION hypothesis for the W13 residual — in-MPK per-task
t_live = fixed per-task overhead (entry/args/smem setup/event) + body; if overhead is a large
share, a ~20% body win dilutes to -7.4% t_live. Arithmetic: t_live 18.764 -> 17.375 = -1.389us;
for that to be 20% of body, body ~= 6.95us and overhead ~= 11.8us (63% of t_live) — is that
plausible for this task given what you can read of the task impl + runtime (compare: attention
task entry measured ~0.5us)? An instrumented in-MPK phase decomposition (E6) is running to
measure it — review its REPORT (/var/tmp/e6_overhead/REPORT.md) if present when you run. Also
weigh the alternative residuals: heterogeneous co-tenancy during the W13 window (alignment data
shows W13 self-concurrency ~105/136, so co-tenancy is mostly SELF plus ~31 mixed workers + 48
schedulers), sustained-clock vs burst-clock, icache pressure in the 69k-task TU, and the
cross-instrument aggregation mismatch (whole-launch wall @ ~100us vs per-task t_live @ ~18.8us —
are % reductions even comparable across those two units? state the conditions under which they
are).

Q6: Given E1+E6, specify the CORRECTED harness scoring protocol for MoE-family loops (grid regime,
path selection, golden timing, what number gates a tag) such that the standalone score PREDICTS
the in-MPK per-task (or body) improvement within stated error bars. If no standalone protocol can,
say so and specify the minimal in-MPK cadence instead.

## ADDENDUM 2 (after E6) — more evidence for Q5/Q6

E6 report: /var/tmp/e6_overhead/REPORT.md (phase decomposition, in-MPK, perturbation-gated).
- Overhead-dilution: REFUTED (entry 0.16 + exit 0.03-0.06 + bracket residual ~0.85us = ~1.15us
  total = 6-9% of t_live, not 63%).
- Clocks: REFUTED (SM clock 1965 MHz median, flat, BOTH regimes).
- The real structure: W13 body delta is -16.08% at the MEDIAN but -9.01% at the MEAN; the ON
  arm loses its advantage in the task-time TAIL. Standalone whole-launch (makespan, i.e.
  tail-dominated) says -20.5% when alone. W2 shows no such split (-13.27% body mean).
- E7 (analysis-only, /var/tmp/e7_pairing/REPORT.md when done) pairs task instances across arms
  and separates work-size-dependence from co-tenancy-on-the-tail. Weigh its tables in your Q5/Q6
  answers; its per-decile and overlap tables are the primary evidence for the protocol design.

## ADDENDUM 3 (after E7) — the residual is now decomposed; review the chain end-to-end

E7 report: /var/tmp/e7_pairing/REPORT.md (verification log e7_verify.log beside it; the prior
partial run was independently re-verified byte-exact before continuation).

- Paired (task id, iteration) instances across OFF/ON reproduce E6 exactly (W13 mean -9.01% /
  median -16.08%). Per-pair: p50 -14.12%, p90 +17.7%, 11.5% of pairs REGRESS.
- The 7.07pp mean-median split decomposes exactly: 1.96pp estimator artifact (ratio-of-medians
  on W13's bimodal small/big-mode mix) + 0.12pp size weighting + 4.99pp within-size tail, of
  which 4.14pp is ONE EVENT CLASS: 8.74% of pairs where the ON body jumps a tight +7.4-9.9us
  QUANTUM above its own id-median while the identical (id, iteration) input runs normally in OFF.
- Size (test A): real but insufficient — big-mode deciles improve -12/-13% vs -15/-21% small;
  ids with OFF bodies stable to <1us across 348 iterations still regress >+10% in ~10% of
  iterations, spread over 10,232/10,240 ids.
- Co-tenancy (test B): placement deterministic (same worker 100%, wave-rank rho 0.91); ON does
  NOT shift starts into denser windows (|rho|<=0.033). Excursion instances show LOWER other-type
  density at onset/begin/normal-window (0.45 vs 0.56; 0.61 vs 0.78) BUT are 8.3x depleted in
  fully-isolated windows (1.53% vs 12.68% excursion rate). E7 states the data cannot separate
  same-window co-tenant cause from shared-position confound and names the separating captures
  (%smid stamps, serialized-stage replay, routed-token dump).
- W2 control: no aggregate split by CANCELLATION (size -3.89pp vs tail +3.84pp), with 4.67%
  ON-only excursions present.

Q7 (new): judge the excursion-event evidence. Is the tight +7.4-9.9us quantum consistent with a
discrete stall (one extra TMA/L2 round, an mbarrier wait, a scheduler-queue hiccup, an extra
K-loop trip from a data-dependent branch)? Does anything in the v024 body (vs v012) plausibly
create a bimodal wait? Rank the candidate causes and say whether E8 (the named captures) is
NECESSARY for the protocol design or merely for curiosity.

Q8 (protocol, extends Q6): given E1 (residency), E6 (overhead/clocks dead), and E7 (excursion
class), specify the corrected standalone scoring protocol AND the expected transfer model a loop
should quote (e.g. "typical-instance transfer ~0.8 of 1-CTA/SM standalone; mean transfer further
reduced by an ~8-9% excursion class worth ~4pp; makespan/stage-span additionally set by packing").
State what number the mpk-validator wall-span check must confirm for a tag to be integration-ready.
