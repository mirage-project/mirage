# Standalone-vs-MPK alignment — the decomposition ledger (2026-08-06)

The user's demand: name the mechanism with numbers before any loop relaunch. Status per mechanism:

| # | mechanism | status | evidence |
|---|---|---|---|
| 1 | register spill | **RULED OUT for the MoE gap** (0 spill both arms/lanes, production TU, verified twice). LIVE for attention on the combined tree (96B fast lane) — separate issue, ws6 regfit loop fixes it. | moe_v024 gates; alignment lane audit; combined REPORT §2 |
| 2 | wrong toolchain (unqualified nvcc → CUDA 13.2/C++17 vs prod 12.8/C++20) | **FOUND + FIXED** (64 vs 75 regs same source) | alignment REPORT part 2 |
| 3 | fast-math missing from scored lane | **FOUND + FIXED** (direction was conservative) | lane audits |
| 4 | analyzer 128-worker hardcode | **FOUND + FIXED** (realized wrappers, 136/48 QC PASS) | alignment REPORT |
| 5 | CTA/SM residency (harness 3-4/SM vs MPK 1/SM) | **MEASURED (E1)**: ws3 harness stock reads −58/−60% cand-vs-golden; at 136 CTAs 1/SM → −22.3% (W13) / −5.5% (W2). For W2 the residency-matched number LANDS ON in-MPK (−11.6 @g148 vs −12.18). | /var/tmp/e1_residency/REPORT.md |
| 6 | W13 residual (~13pt): 1-CTA/SM standalone −20..−22% vs in-MPK t_live −7.4% | **DECOMPOSED (E6+E7)**: overhead-dilution REFUTED (1.15µs = 6%); clocks REFUTED (1965 MHz flat). E7 paired 10,240 (id,iteration) instances across arms — the 7.07pp mean-median split decomposes EXACTLY: 1.96pp estimator artifact (ratio-of-medians on bimodal size mix) + 0.12pp size weighting + 4.99pp within-size tail, of which **4.14pp is ONE EVENT CLASS: 8.74% of ON-arm instances jump a tight +7.4–9.9µs QUANTUM above their own id-median while the identical input runs normally in OFF**. Size effect real but insufficient (stable-size ids still excurse ~10% of iterations, spread over 10,232/10,240 ids). Placement deterministic (same worker 100%, ON does NOT shift starts into denser windows). Excursions need co-presence (8.3× depleted in fully-isolated windows: 1.53% vs 12.68%) yet correlate with LOWER other-type density — same-window co-tenant cause vs shared-position confound NOT separable from this data; named separating captures: %smid stamps, serialized-stage replay, routed-token dump (E8, deferred pending review). W2 control: no aggregate split by CANCELLATION (size −3.89pp vs tail +3.84pp; 4.67% excursions exist there too). | /var/tmp/e6_overhead/REPORT.md; /var/tmp/e7_pairing/REPORT.md |
| 7 | layer 2 (in-MPK win → step ≈0 at bs16) | **PROVEN**: packing/straggler — W13 span +5.6% despite tasks −7.4%; self-concurrency 105→92; OFF work-bound → ON cp-bound. | alignment REPORT part 1 |

## PROVENANCE INVERSION (E1's biggest find — my attribution was wrong twice)
The −19.88%/−13.55% "standalone claim" did NOT come from the ws3 ferret harness. It came from the
integration instrument `bitexact_v024.cu --benchmark` at grid=128 + 205,824B dynamic smem = ALREADY
one CTA/SM (production path selection, split-2). The ws3 harness never timed its golden at all
(bit-gate only, grid=1). So my narrative "the loop's oversubscribed harness produced the inflated
claim" was false — the loop's harness is even MORE inflated (−58/−60%) but is not the claim source.
W2's claim anchor is soft besides: the record log predates the final header commit; fresh reruns of
the record instrument give −18.6/−19.0%, not −13.55%.

## Interpretation embargo
Per the standing rule (user 2026-08-06), the ATTRIBUTION of mechanisms 5/6 and the corrected
harness protocol await the codex refute-first review (brief /var/tmp/review_concl/BRIEF.md, 105
lines incl. E1 addendum + Q5 overhead-dilution + Q6 protocol design; fires at quota reset
Aug 7 23:37 EDT). Loops (ws5 topk-r2, ws6 regfit, ws7 linear — all authored READY) stay PARKED
until: E6 lands + decomposition written + protocol corrected + review verdict. Checklist:
scratchpad/aligned_relaunch_checklist.md §F.

## Combined-tree gate (context for what integration is worth right now)
run2 non-binding, fresh vLLM, AC-3 150/150: ratios 0.702/0.690/0.691/0.634/0.593 at bs1/2/4/8/16
(vs ee300d5e's 0.656/0.650/0.640/0.586/0.583). AC-4/AC-5 still FAIL — gap 1.4-1.7x remains.
