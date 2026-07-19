---
name: b200-tma-pipeline-designer
description: >-
  Use when the user wants to convert regular GMEM↔SMEM tile copies to TMA on B200/Blackwell, design double-buffered/multi-stage pipelines, choose a swizzle, or distinguish the TMA load vs store completion protocols. Produces descriptor, stage ring, barrier, and prologue/steady-state/epilogue plans. Not for highly irregular gather/scatter or copies too small to be worth setting up a TMA descriptor.
source_book: "Modern GPU Programming For MLSys (MLC Community) + NVIDIA Blackwell Tuning/Compatibility Guides"
source_chapter: "S6; S9; S12"
tags: ["b200", "blackwell"]
related_skills: ["b200-scope-layout-dispatch", "b200-layout-contract-auditor", "b200-mbarrier-protocol-auditor", "b200-gemm-optimization-ladder", "b200-flash-attention4-planner"]
version: "0.1.0"
---

<!-- Distilled from "Modern GPU Programming for MLSys" — https://mlc.ai/modern-gpu-programming-for-mlsys/ -->

# B200 TMA Pipeline Designer

## R — Source evidence (Reading, paraphrased)

- [S6] TMA is issued by a single thread; the hardware asynchronously moves a regular rectangular tile; the descriptor describes the global shape/stride, tile coordinates, and SMEM swizzle.
- [S6] TMA loads complete through an mbarrier carrying a byte count; TMA stores drain through commit group / wait group.
- [S12] Double buffering gives the next K tile a landing slot; full load/compute overlap additionally requires role separation or an equivalent concurrency structure.

> Source: distilled from "Modern GPU Programming for MLSys" (https://mlc.ai/modern-gpu-programming-for-mlsys/) and the NVIDIA Blackwell tuning/compatibility guides. Short paraphrases only; no long passages are reproduced.

---

## I — Methodology skeleton (Interpretation)

TMA is not "the copy API under a new name"; it restructures a synchronous thread path into an asynchronous producer. The design counts as complete only when all of the following are given together:

- the descriptor and the SMEM layout;
- a unique issuer;
- the ready/free protocol for each stage;
- the distinct completion mechanisms for load vs store;
- the prologue, the steady-state loop, and the drain;
- the trade-off between stage count and SMEM/occupancy.

If you have only written `copy_async` without a lifecycle protocol, the TMA design is not finished yet.

---

## A1 — Applications in the source (Past Application)

### Case 1: GEMM operand double buffering
- While stage 0 is being consumed by the MMA, TMA writes the next K tile into stage 1.
- The consumer first waits on `tma2mma[stage]`; the producer waits on `mma2tma[stage]` before overwriting.

### Case 2: TMA store of results
- After the epilogue writes the result to `Dsmem`, a single thread issues the TMA store.
- `commit_group` must be followed by `wait_group`, confirming the store has drained before `Dsmem` may be reused or the associated lifetime may end.

---

## A2 — Trigger scenarios (Future Trigger) ★

### In what situations will the user need this skill?

1. "Convert this GMEM→SMEM load to B200 TMA double buffering."
2. "What should a TMA load and a TMA store each wait on?"
3. "Help me design the stage ring and barriers for PIPE_DEPTH=2/3."

### Language signals

- "Convert this GMEM→SMEM load to B200 TMA double buffering."
- "What should a TMA load and a TMA store each wait on?"
- "Help me design the stage ring and barriers for PIPE_DEPTH=2/3."

### Distinction from adjacent skills

Versus `b200-mbarrier-protocol-auditor`: this skill designs a TMA pipeline from scratch; the latter checks existing barriers item by item for bugs. Compose with `b200-layout-contract-auditor` to cross-check the swizzle.

---

## E — Executable steps (Execution)

Once the skill is activated, the agent must execute the following procedure:

1. **Assess TMA suitability**
   - Is the tile regular, rectangular, and describable by fixed strides?
   - Is the copy volume large enough to amortize the descriptor and synchronization overhead?
   - If not, keep the vectorized thread copy.
2. **Define the tensor-map descriptor**
   - Global shape/stride, element size, tile shape, coordinates, boundary policy, swizzle.
   - Completion criterion: the source address and the destination SMEM layout can be uniquely derived from the logical tile coordinates.
3. **Choose the issuer**
   - Designate one lane/thread to issue; do not let every thread redundantly issue the TMA.
   - Write the issue scope and the CTA consumption scope separately.
4. **Define the load protocol**
   - Initialize a barrier for each stage.
   - `arrive.expect_tx(bytes)` sets the expected bytes and completes the issuing thread's arrival.
   - Consumers read SMEM only after waiting on the correct phase.
5. **Define the store protocol**
   - After `commit_group`, use `wait_group` to guarantee outgoing stores have drained.
   - Do not mistakenly substitute the load-side byte-count protocol for store drain.
6. **Design the stage ring**
   - Give `PIPE_DEPTH`, per-stage A/B/temporary SMEM, the ready/free barriers, and the initial phase values.
   - Compute total SMEM; check whether it pushes residency to an unacceptable level.
7. **Write the three-phase timeline**
   - Prologue: fill the initial batch of stages.
   - Steady state: overlap load k+1 / compute k / store k-1.
   - Epilogue: stop issuing new loads, finish the remaining compute/store, drain the groups.
8. **Validate**
   - Correctness: asymmetric small matrices, boundary tiles, different K_TILES.
   - Protocol: each stage's ready and free each have a unique producer and consumer.
   - Performance: confirm the copy instructions genuinely overlap with Tensor Core work, rather than code that looks asynchronous but is still issued serially by the same role.

### Required outputs

1. **Conclusion**: the current choice/diagnosis, without vague "could be any of them" hedging.
2. **Evidence or assumptions**: which come from user data, and which are hypotheses awaiting verification.
3. **Contract/table/timeline**: the auditable intermediate artifacts corresponding to this skill.
4. **Minimal validation**: correctness tests, boundary tests, and one falsifiable experiment.
5. **Risks and fallback**: alternative paths when hardware, version, or resource conditions are not met.

---

## B — Boundaries (Boundary) ★

### Do not use when
- Highly sparse, irregular gather/scatter that a descriptor cannot express.
- A one-off, tiny copy where the TMA setup and synchronization cost may exceed the benefit.

### Failure modes
- TMA swizzle inconsistent with the MMA layout.
- Consuming right after the load is issued, missing the barrier wait.
- Reusing SMEM before the store has drained.
- Increasing pipeline depth without accounting for SMEM resources and occupancy.

### Limitations
- The best pipeline depth must be measured on real shapes, clock frequencies, and compiled output; it cannot be derived from a fixed rule.

---

## Related skills

- **depends-on**: `b200-scope-layout-dispatch`, `b200-layout-contract-auditor`
- **contrasts-with**: none
- **composes-with**: `b200-mbarrier-protocol-auditor`, `b200-gemm-optimization-ladder`, `b200-flash-attention4-planner`

---

## Audit info

- **Validation passed**: V1 ✓ / V2 ✓ / V3 ✓
- **Test definitions**: 6 (3 should_trigger / 2 should_not_trigger / 1 edge_case)
- **Hardware validation**: not performed; must be verified on a target B200
- **Distilled**: 2026-06-25
