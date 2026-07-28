---
name: b200-gemm-optimization-ladder
description: >-
  Use when the user wants to implement from scratch, port, or systematically optimize a B200/Blackwell GEMM. Advances level by level along "correct single tile→K loop→spatial tiling→TMA→multi-stage pipeline→persistent→warp specialization→2-CTA cluster→multi-consumer", with a correctness and performance gate at every level. Not for cases that only want to call a mature BLAS and need no custom fusion/layout.
source_book: "Modern GPU Programming For MLSys (MLC Community) + NVIDIA Blackwell Tuning/Compatibility Guides"
source_chapter: "S11; S12; S13; S3"
tags: ["b200", "blackwell"]
related_skills: ["b200-kernel-roofline-triage", "b200-scope-layout-dispatch", "b200-tma-pipeline-designer", "b200-mbarrier-protocol-auditor", "b200-tcgen05-mma-contract-builder", "b200-cluster-persistent-scheduler", "b200-warp-specialized-debugger"]
version: "0.1.0"
---

<!-- Distilled from "Modern GPU Programming for MLSys" — https://mlc.ai/modern-gpu-programming-for-mlsys/ -->

# B200 GEMM Optimization Ladder

## R — Source evidence (Reading, paraphrased)

- [S11] Start from a minimal correct single tile, then add K accumulation and multi-CTA spatial tiling one step at a time, avoiding debugging all the complexity at once.
- [S12] First let TMA take over the regular tile copies, then use multi-stage SMEM and a persistent scheduler to reduce waiting.
- [S13] Warp specialization assigns load, MMA, and writeback to different roles; 2-CTA cluster and multi-consumer further remove serial bottlenecks.
- [S3] Every level should be driven by roofline and measured evidence; a more complex structure is not guaranteed to be faster.

> Source: distilled from "Modern GPU Programming for MLSys" (https://mlc.ai/modern-gpu-programming-for-mlsys/) and the NVIDIA Blackwell tuning/compatibility guides. Short paraphrases only; no long passages are reproduced.

---

## I — Methodology skeleton (Interpretation)

GEMM optimization is not generating the "final kernel" in one shot, but a regression-testable, bisectable upgrade path. Every level must simultaneously satisfy:

- correct against the reference;
- an explainable layout/synchronization contract;
- the performance change is measured;
- if it regresses, you know where the added resource or serialization point is.

The agent should keep the version at every level; jumping straight to a complex warp-specialized cluster kernel and then blindly guessing at bugs is forbidden.

---

## A1 — Applications in the source (Past Application)

### The nine-level route in the book
1. A single 128×128 output tile.
2. K-loop accumulation.
3. Multiple CTAs covering the full M/N.
4. TMA async load/store.
5. PIPE_DEPTH=2 software pipeline.
6. Persistent kernel + tile scheduler.
7. Warp specialization.
8. 2-CTA cluster.
9. Multi-consumer warp specialization.

Every level keeps the same basic data path: GMEM→SMEM→`tcgen05`→TMEM→register/SMEM→GMEM, changing only concurrency and scheduling.

---

## A2 — Trigger scenarios (Future Trigger) ★

### In what situations will the user need this skill?

1. "Start from a correct GEMM and optimize it step by step into a high-performance B200 version."
2. "My GEMM is at Step 5 now; should the next step be persistent or warp specialization?"
3. "Write correctness and performance acceptance criteria for each level."

### Language signals

- "Start from a correct GEMM and optimize it step by step into a high-performance B200 version."
- "My GEMM is at Step 5 now; should the next step be persistent or warp specialization?"
- "Write correctness and performance acceptance criteria for each level."

### Distinction from adjacent skills

Difference from `b200-kernel-roofline-triage`: this skill is the GEMM-specific implementation route; the roofline skill decides whether compute/overlap optimization should continue to be pursued. Combine with the individual specialized skills to complete the concrete stages.

---

## E — Executable steps (Execution)

Once the skill is activated, the agent must execute the following process:

1. **Establish the baseline contract**
   - Fix the math definition, layout, dtype, reference, timing framework, and representative shape set.
2. **Level 1: single-tile correct path**
   - Synchronous copy, one MMA, TMEM readback, store.
   - Acceptance: element-wise correct on small matrices, with every address space explainable.
3. **Level 2: K-loop**
   - Correctly handle the initial accumulator, per-K-tile accumulation, and the K tail.
4. **Level 3: spatial tiling**
   - grid→M/N tile mapping, boundary masks, full matrix coverage.
5. **Level 4: TMA**
   - descriptor/swizzle, load barrier, store drain; compare against the synchronous version bitwise/within tolerance.
6. **Level 5: multi-stage pipeline**
   - prologue/steady/epilogue, stage/phase ledger; measure the actual overlap.
7. **Level 6: persistent scheduler**
   - A fixed set of resident CTAs processes multiple tiles; check the tail and tile order.
8. **Level 7: warp specialization**
   - producer/MMA/writeback roles, the four classes of handoff, warpgroup-scoped sync.
9. **Level 8: 2-CTA cluster**
   - cluster tile, DSMEM/multicast, cta_group::2, remote barrier.
10. **Level 9: multi-consumer**
   - Multiple consumers partition the N/M/output regions, ensuring the same staged operand feeds more compute without write conflicts.
11. **Per-level gating**
   - Correctness: random shapes, misalignment, K=1/multi-tile, NaN/Inf, error thresholds.
   - Performance: compare against the previous level, the library baseline, and the roofline.
   - Resource: register/SMEM/TMEM, active clusters, spill.
12. **Stopping rule**
   - If you are already close to the practical roofline or added complexity no longer brings stable gains, stop upgrading and keep the simpler version.

### Required outputs

1. **Conclusion**: the current choice/diagnosis; do not use a vague "it could be any of them".
2. **Evidence or assumptions**: which items come from user data, and which are hypotheses awaiting verification.
3. **Contract/table/timeline**: the auditable intermediate artifacts corresponding to this skill.
4. **Minimal validation**: correctness tests, boundary tests, and one falsifiable experiment.
5. **Risks and fallback**: alternative paths when hardware, version, or resource requirements are not met.

---

## B — Boundaries (Boundary) ★

### Do not use when
- cuBLASLt/CUTLASS already meets the need and there is no fusion, special dtype/layout, or research purpose.
- There is no correctness reference or reliable timing framework.

### Failure modes
- Jumping multiple levels at once, making the source of an error impossible to localize.
- Testing only one neat large shape, ignoring small shapes and tail tiles.
- Treating warp specialization as a guaranteed speedup, ignoring the occupancy of the added roles and the sync.

### Limitations
- The book's route is based mainly on TIRx and specific example shapes; when porting to CUDA/CUTLASS/Triton, keep the principles rather than copying the APIs verbatim.

---

## Related skills

- **depends-on**: `b200-kernel-roofline-triage`, `b200-scope-layout-dispatch`
- **contrasts-with**: none
- **composes-with**: `b200-tma-pipeline-designer`, `b200-mbarrier-protocol-auditor`, `b200-tcgen05-mma-contract-builder`, `b200-cluster-persistent-scheduler`, `b200-warp-specialized-debugger`

---

## Audit info

- **Validation passed**: V1 ✓ / V2 ✓ / V3 ✓
- **Test definitions**: 6 (3 should_trigger / 2 should_not_trigger / 1 edge_case)
- **Hardware validation**: not performed; must be verified on a target B200
- **Distilled**: 2026-06-25
