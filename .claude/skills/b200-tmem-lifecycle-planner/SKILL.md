---
name: b200-tmem-lifecycle-planner
description: >-
  Use when the user is using `tcgen05` on Blackwell/B200 and needs to plan TMEM accumulators, block-scale factors, TMEM column allocation, the `tcgen05.ld/st/cp` paths, epilogue readback, and safe deallocation. Produces a TMEM region map, column budget, and lifecycle. Not for Hopper/Ampere or code that treats TMEM as ordinary shared memory.
source_book: "Modern GPU Programming For MLSys (MLC Community) + NVIDIA Blackwell Tuning/Compatibility Guides"
source_chapter: "S8; S7; S14"
tags: ["b200", "blackwell"]
related_skills: ["b200-layout-contract-auditor", "b200-tcgen05-mma-contract-builder", "b200-mbarrier-protocol-auditor", "b200-flash-attention4-planner"]
version: "0.1.0"
---

<!-- Distilled from "Modern GPU Programming for MLSys" — https://mlc.ai/modern-gpu-programming-for-mlsys/ -->

# B200 TMEM Lifecycle Planner

## R — Source evidence (Reading, paraphrased)

- [S8] TMEM is a Blackwell-only, CTA-scoped two-dimensional `Lane × Col` scratchpad, with 128 Lane rows and up to 512 32-bit Cols.
- [S8] TMEM must be explicitly allocated and freed in 32-column units; ordinary shared load/store cannot access it.
- [S8] `tcgen05.ld/st/cp` are all dedicated asynchronous paths, each with its own completion rules.
- [S7] Both the accumulator and the block-scale factors may live in TMEM, but their layouts and lifecycles differ.

> Source: distilled from "Modern GPU Programming for MLSys" (https://mlc.ai/modern-gpu-programming-for-mlsys/) and the NVIDIA Blackwell tuning/compatibility guides. Short paraphrases only; no long passages are reproduced.

---

## I — Methodology skeleton (Interpretation)

TMEM planning resembles an "explicitly managed accumulator heap", not compiler-allocated registers:

1. First compute the columns and alignment each object class needs;
2. then map the logical tiles onto `TLane/TCol`;
3. annotate who writes, who reads, and when each completes;
4. arrange region reuse and the final free;
5. ensure the epilogue's register fragments are consistent with the TMEM mapping used at write time.

A larger tile reducing register pressure does not mean the resources are free; it shifts the pressure onto the TMEM column budget and readback bandwidth.

---

## A1 — Applications in the source (Past Application)

### Case 1: dense MMA accumulator
- `tcgen05.mma` writes the fp32 accumulator to TMEM.
- After the compute phase, the full warpgroup uses `tcgen05.ld` to read its fragments back into registers for cast/epilogue/store.

### Case 2: block-scaled MMA
- A/B data live in SMEM.
- Scale factors first go to SMEM, then enter the dedicated TMEM scale layout via `tcgen05.cp`.
- The scale region and the accumulator region must not reuse the same mapping just because both are in TMEM.

---

## A2 — Trigger scenarios (Future Trigger) ★

### In what situations will the user need this skill?

1. "How many TMEM columns does this tcgen05 kernel need to allocate?"
2. "Help me plan the TMEM reuse of S/P/O in FlashAttention."
3. "How should the completion and free ordering of tcgen05.ld/st/cp be written?"

### Language signals

- "How many TMEM columns does this tcgen05 kernel need to allocate?"
- "Help me plan the TMEM reuse of S/P/O in FlashAttention."
- "How should the completion and free ordering of tcgen05.ld/st/cp be written?"

### Distinction from adjacent skills

Versus `b200-tcgen05-mma-contract-builder`: this skill focuses on TMEM resources, layout, and lifecycle; the MMA contract builder additionally decides the tile shape, cta_group, dtype, and operand placement.

---

## E — Executable steps (Execution)

Once the skill is activated, the agent must execute the following procedure:

1. **List the TMEM objects**
   - Accumulators, score/prob/output tiles, scale factors, temporary regions.
   - For each object, record its dtype, logical shape, and the number simultaneously live.
2. **Compute the column budget**
   - Map the objects onto 32-bit Cols; round up to the 32-column allocation unit.
   - Include the replication factors for pipeline stages, dual Q stages, or multiple consumers.
3. **Define the 2D layout**
   - Give explicit `TLane`, `TCol` formulas; "so many bytes off the base" alone is not enough.
   - For `cta_group::2`, write out each CTA's fragment in its own TMEM separately.
4. **Define the write paths**
   - MMA accumulator: `tcgen05.mma`.
   - Scale: `tcgen05.cp` from SMEM→TMEM.
   - register→TMEM: `tcgen05.st`.
5. **Define the readback path**
   - Choose the `tcgen05.ld` shape/repeat; write out the fragment each warp/lane receives.
   - After readback, wait on the corresponding `wait::ld` before consuming the registers.
6. **Draw the lifecycle**
   - allocate → producer writes → completion → consumers read/modify → completion → reuse/free.
   - Annotate every reuse edge with its barrier or wait.
7. **Check region reuse**
   - Overlay only when all old consumers have completed and the new layout does not conflict.
8. **Produce the resource report**
   - Total columns, peak simultaneously-live columns, alignment waste, the riskiest reuse edge, and the epilogue readback cost.

### Required outputs

1. **Conclusion**: the current choice/diagnosis, without vague "could be any of them" hedging.
2. **Evidence or assumptions**: which come from user data, and which are hypotheses awaiting verification.
3. **Contract/table/timeline**: the auditable intermediate artifacts corresponding to this skill.
4. **Minimal validation**: correctness tests, boundary tests, and one falsifiable experiment.
5. **Risks and fallback**: alternative paths when hardware, version, or resource conditions are not met.

---

## B — Boundaries (Boundary) ★

### Do not use when
- The target is not Blackwell, or the kernel does not use `tcgen05`/TMEM.
- The user only needs ordinary shared memory tiling.

### Failure modes
- Accessing TMEM with `ld.shared/st.shared`.
- Forgetting to allocate in 32-column units, or forgetting to free.
- Consuming registers before `tcgen05.ld` has completed.
- Mixing the accumulator layout with the scale-factor layout.

### Limitations
- The exact available column count, supported load shapes, and lowering depend on the toolchain version; verify against the target compiler and the generated PTX.

---

## Related skills

- **depends-on**: `b200-layout-contract-auditor`
- **contrasts-with**: none
- **composes-with**: `b200-tcgen05-mma-contract-builder`, `b200-mbarrier-protocol-auditor`, `b200-flash-attention4-planner`

---

## Audit info

- **Validation passed**: V1 ✓ / V2 ✓ / V3 ✓
- **Test definitions**: 6 (3 should_trigger / 2 should_not_trigger / 1 edge_case)
- **Hardware validation**: not performed; must be verified on a target B200
- **Distilled**: 2026-06-25
