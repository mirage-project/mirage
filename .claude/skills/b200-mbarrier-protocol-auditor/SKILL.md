---
name: b200-mbarrier-protocol-auditor
description: >-
  Use when a Blackwell/B200 asynchronous kernel deadlocks, fails intermittently, reads stale data, reuses a stage too early, or when the arrival, tx-count, phase, and wait of TMA/tcgen05/CLC need auditing. Produces a per-barrier protocol ledger and fix points. Not for cases already known to be out-of-bounds accesses, layout mismatches, or ordinary host-side synchronization problems.
source_book: "Modern GPU Programming For MLSys (MLC Community) + NVIDIA Blackwell Tuning/Compatibility Guides"
source_chapter: "S9; S6; S7; S13; S14"
tags: ["b200", "blackwell"]
related_skills: ["b200-scope-layout-dispatch", "b200-layout-contract-auditor", "b200-tma-pipeline-designer", "b200-tmem-lifecycle-planner", "b200-tcgen05-mma-contract-builder", "b200-warp-specialized-debugger"]
version: "0.1.0"
---

<!-- Distilled from "Modern GPU Programming for MLSys" — https://mlc.ai/modern-gpu-programming-for-mlsys/ -->

# B200 mbarrier Protocol Auditor

## R — Source evidence (Reading, paraphrased)

- [S9] An mbarrier keeps an arrival counter and a phase in SMEM; completion of asynchronous operations cannot be inferred from program order.
- [S9] A TMA load's `expect_tx` registers both the issuing thread's arrival and the expected bytes; the phase flips only when both the arrivals and the pending bytes are complete.
- [S9] A `tcgen05` MMA must explicitly associate a barrier arrival on its commit path; otherwise consumers may wait forever.
- [S13] Multi-role GEMMs commonly carry two barrier sets, forward ready and backward release; a wrong initial phase deadlocks or silently corrupts.

> Source: distilled from "Modern GPU Programming for MLSys" (https://mlc.ai/modern-gpu-programming-for-mlsys/) and the NVIDIA Blackwell tuning/compatibility guides. Short paraphrases only; no long passages are reproduced.

---

## I — Methodology skeleton (Interpretation)

Treat every barrier as a bidirectional contract, not "a wait somewhere":

- Who initializes it, and how many arrivals are expected?
- Who produces the data, and does the completion signal come from threads, a TMA byte count, an MMA commit, or a CLC response?
- Which consumers wait on which phase?
- On completion, does it release data-ready or buffer-free?
- When does the phase flip, and who holds the local expected phase for the next round?

The audit's goal is a complete barrier ledger; any field without a unique answer is a potential bug.

---

## A1 — Applications in the source (Past Application)

### Case 1: producer/consumer initial phases
- With an empty buffer, the producer's first round should start filling immediately; the consumer's first round should block until the data is ready.
- If the two sides share the same initial phase, both may block on the first round, or an old completion may be mistaken for a new one.

### Case 2: the four-barrier GEMM
- `tma2mma`: SMEM data ready.
- `mma2tma`: SMEM stage overwritable.
- `mma2ld`: TMEM result ready.
- `ld2mma`: TMEM region reusable.
- Forward data flow and backward resource release must close the loop in pairs.

---

## A2 — Trigger scenarios (Future Trigger) ★

### In what situations will the user need this skill?

1. "The kernel is stuck in mbarrier.wait — help me check the arrival/phase."
2. "My double buffering occasionally reads the previous round's tile."
3. "How should tcgen05 commit be wired to the barrier?"

### Language signals

- "The kernel is stuck in mbarrier.wait — help me check the arrival/phase."
- "My double buffering occasionally reads the previous round's tile."
- "How should tcgen05 commit be wired to the barrier?"

### Distinction from adjacent skills

Versus `b200-layout-contract-auditor`: this skill only proves "when it is safe", not "whether the address is correct". Versus `b200-warp-specialized-debugger`: this skill is a dedicated barrier audit; the debugger is a full symptom-driven workflow.

---

## E — Executable steps (Execution)

Once the skill is activated, the agent must execute the following procedure:

1. **Build the barrier ledger**
   - Fields: name, storage address, init scope, expected arrivals, tx bytes, producer, arrival mechanism, consumer, wait phase, released resource.
2. **Verify the initialization sites**
   - Barrier init must happen before the role branch and be observed by the correct scope.
   - Check alignment and the number of stages.
3. **Verify the producer side**
   - TMA load: is the expected byte count exact, and does the actual copy complete against the same barrier?
   - MMA/`tcgen05.cp`: is the commit bound to the correct barrier arrival?
   - Plain threads: does the number of threads actually arriving equal the init count?
4. **Verify the consumer side**
   - Do the waited-on barrier and the stage index match?
   - Does the local phase flip exactly once, and only once, after a successful consume?
5. **Verify the ready/free closed loop**
   - Every SMEM/TMEM slot has both a "when readable" and a "when overwritable/freeable" condition.
6. **Verify scope legality**
   - A CTA-wide collective must not hide inside a warpgroup-only branch.
   - Cluster signals must use the correct CTA mask/remote arrival.
7. **Check the first and last rounds**
   - Simulate the prologue's first wait, one steady-state lap, and the epilogue's final drain.
   - Hand-compute the 0/1/0/1 phase sequence in a table.
8. **Change only one handoff**
   - After changing exactly one of init count, phase, arrival, or fence, run minimal correctness first, then measure performance.

### Required outputs

1. **Conclusion**: the current choice/diagnosis, without vague "could be any of them" hedging.
2. **Evidence or assumptions**: which come from user data, and which are hypotheses awaiting verification.
3. **Contract/table/timeline**: the auditable intermediate artifacts corresponding to this skill.
4. **Minimal validation**: correctness tests, boundary tests, and one falsifiable experiment.
5. **Risks and fallback**: alternative paths when hardware, version, or resource conditions are not met.

---

## B — Boundaries (Boundary) ★

### Do not use when
- There is already conclusive evidence of an out-of-bounds access, an invalid pointer, or a wrong descriptor.
- Ordinary CPU mutex/condition_variable synchronization problems.

### Failure modes
- Treating `expect_tx` as a plain arrival and ignoring the byte budget.
- Forgetting the MMA commit arrival.
- Stage index correct but the phase belongs to the previous round.
- Building only the ready barrier and not the buffer-release barrier.

### Limitations
- Compiler lowering may change the surface code structure; ultimately verify the barrier initialization and commit/wait in the generated CUDA/PTX.

---

## Related skills

- **depends-on**: `b200-scope-layout-dispatch`
- **contrasts-with**: `b200-layout-contract-auditor`
- **composes-with**: `b200-tma-pipeline-designer`, `b200-tmem-lifecycle-planner`, `b200-tcgen05-mma-contract-builder`, `b200-warp-specialized-debugger`

---

## Audit info

- **Validation passed**: V1 ✓ / V2 ✓ / V3 ✓
- **Test definitions**: 6 (3 should_trigger / 2 should_not_trigger / 1 edge_case)
- **Hardware validation**: not performed; must be verified on a target B200
- **Distilled**: 2026-06-25
