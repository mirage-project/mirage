---
name: b200-flash-attention4-planner
description: >-
  Use when the user wants to design or extend a FlashAttention-style forward kernel on B200/Blackwell, involving the two MMAs QKᵀ and PV, online softmax, S/P/O in TMEM, warp roles, causal mask, GQA, tile scheduling, or final normalization. Outputs the algorithm state, tile graph, barrier graph, and validation plan. Not for cases that only use off-the-shelf framework operators, where the full backward is not yet defined, or for ordinary dense GEMM.
source_book: "Modern GPU Programming For MLSys (MLC Community) + NVIDIA Blackwell Tuning/Compatibility Guides"
source_chapter: "S14; S6–S9; S13"
tags: ["b200", "blackwell"]
related_skills: ["b200-tcgen05-mma-contract-builder", "b200-tmem-lifecycle-planner", "b200-mbarrier-protocol-auditor", "b200-gemm-optimization-ladder", "b200-tma-pipeline-designer", "b200-warp-specialized-debugger", "b200-kernel-roofline-triage"]
version: "0.1.0"
---

<!-- Distilled from "Modern GPU Programming for MLSys" — https://mlc.ai/modern-gpu-programming-for-mlsys/ -->

# B200 FlashAttention-4 Planner

## R — Source evidence (Reading, paraphrased)

- [S14] Attention is not the same MMA repeated; it is a score MMA and a value MMA with online softmax, masking, and rescaling in between.
- [S14] The streaming state is `row_max`, `row_sum`, and `O`; when a new maximum appears, the old denominator and O must both be rescaled to the same basis.
- [S14] S, P, and O mainly reside in TMEM; softmax/correction reads or modifies them in registers, then writes back to TMEM.
- [S14] Multiple warpgroups divide the work of driving TMA, MMA, softmax, and correction/epilogue; the barrier graph proves every tile can be safely consumed and reused.

> Source: distilled from "Modern GPU Programming for MLSys" (https://mlc.ai/modern-gpu-programming-for-mlsys/) and the NVIDIA Blackwell tuning/compatibility guides. Short paraphrases only; no long passages are reproduced.

---

## I — Methodology skeleton (Interpretation)

FA4 design should first draw the "algorithm state machine" and the "hardware tile graph" before writing code:

- the algorithm state machine guarantees the online softmax math is correct;
- the tile graph states where Q/K/V/S/P/O live in GMEM, SMEM, TMEM, and registers;
- the role graph states which warp/warpgroup issues TMA/MMA or does softmax;
- the barrier graph states the visibility and reuse along score→softmax→value MMA→epilogue.

Any implementation that omits the rescale, treats P as the final normalized matrix, or misses the O slot release may be fast but numerically wrong.

---

## A1 — Applications in the source (Past Application)

### Streaming online softmax
For each K/V block:
- Compute the score `S = QKᵀ`.
- Update `m_new`.
- Scale `row_sum` and the old `O` by the old/new max difference.
- Compute the current numerator `P` and accumulate `P V`.
- Only at the very end do `O / row_sum`.

### Role example
- One warp handles the TMA loads.
- One warp issues the score/value MMAs.
- Two warpgroups handle the softmax for the two-stage Q pipeline.
- One warpgroup does the O correction and epilogue.
- One warp handles the final TMA store.

---

## A2 — Trigger scenarios (Future Trigger) ★

### In what situations will the user need this skill?

1. "Implement a causal FlashAttention forward pass on B200."
2. "Help me draw the FA4 S/P/O TMEM regions and the barrier graph."
3. "Add GQA or LSE output to this attention kernel."

### Language signals

- "Implement a causal FlashAttention forward pass on B200."
- "Help me draw the FA4 S/P/O TMEM regions and the barrier graph."
- "Add GQA or LSE output to this attention kernel."

### Distinction from adjacent skills

Difference from `b200-gemm-optimization-ladder`: FA4 has two MMAs with softmax/rescaling in between and cannot be treated as a single GEMM loop. Combine with `b200-tmem-lifecycle-planner` for the region budget.

---

## E — Executable steps (Execution)

Once the skill is activated, the agent must execute the following process:

1. **Fix the math semantics**
   - Q/K/V shapes, layout, head_dim, causal, GQA ratio, scale, the output, and whether LSE is needed.
2. **Define the streaming state**
   - The initial values and per-block update formulas of `row_max`, `row_sum`, and `O` for each row.
   - Be explicit about using natural exp or the equivalent `exp2` scaling; the reference must match.
3. **Define the two MMAs**
   - score MMA: Q×Kᵀ→S.
   - value MMA: P×V→O.
   - Write out both tile shapes, dtypes, SMEM operands, and TMEM outputs.
4. **Plan the S/P/O TMEM**
   - After S is ready, read it into registers for mask/softmax; write P back to TMEM; accumulate O in TMEM and rescale when necessary.
   - Give the regions, strides, stages, and safe-reuse conditions.
5. **Assign warp roles**
   - TMA load, MMA issue, softmax stage 0/1, correction/epilogue, TMA store.
   - Check that the collective scopes are complete.
6. **Build the barrier graph**
   - Q/K/V ready→score/value MMA.
   - S ready→softmax.
   - P ready + O safe→value MMA.
   - final O ready→epilogue→store.
   - Draw the scalar mailbox's full/empty protocol separately.
7. **Implement mask/GQA**
   - causal tiles: handle fully-skipped blocks, fully-valid blocks, and diagonal-boundary masking separately.
   - GQA: make the Q-head-to-KV-head mapping, reuse, and scheduler coordinates explicit.
8. **Implement rescale and writeback**
   - The O correction is a full TMEM→register→TMEM tile operation; it cannot be removed.
   - After the loop ends: `O / row_sum`, cast, SMEM staging, TMA store drain.
9. **Validate**
   - Compare against a high-precision PyTorch reference.
   - Cover causal/noncausal, different sequence lengths, head_dim, GQA ratios, tail tiles, and extreme logits.
   - If extending to the training forward, verify the LSE definition and scale are consistent; backward requires a separate design.

### Required outputs

1. **Conclusion**: the current choice/diagnosis; do not use a vague "it could be any of them".
2. **Evidence or assumptions**: which items come from user data, and which are hypotheses awaiting verification.
3. **Contract/table/timeline**: the auditable intermediate artifacts corresponding to this skill.
4. **Minimal validation**: correctness tests, boundary tests, and one falsifiable experiment.
5. **Risks and fallback**: alternative paths when hardware, version, or resource requirements are not met.

---

## B — Boundaries (Boundary) ★

### Do not use when
- Only a stable implementation already provided by PyTorch/FlashAttention needs to be called.
- The user asks for the full backward, but the saved intermediates and gradient algorithm are not yet defined.

### Failure modes
- Not rescaling the old `row_sum/O`.
- Normalizing P too early, breaking the streaming accumulation.
- Reusing the S/P/O TMEM regions before their consumers finish.
- Using the same slow path for the causal mask on both full blocks and boundary blocks.
- Wrong GQA head mapping.

### Limitations
- This skill centers on the forward structure from the book; training backward, dropout, variable-length packed sequences, and distributed attention require additional design.

---

## Related skills

- **depends-on**: `b200-tcgen05-mma-contract-builder`, `b200-tmem-lifecycle-planner`, `b200-mbarrier-protocol-auditor`
- **contrasts-with**: `b200-gemm-optimization-ladder`
- **composes-with**: `b200-tma-pipeline-designer`, `b200-warp-specialized-debugger`, `b200-kernel-roofline-triage`

---

## Audit info

- **Validation passed**: V1 ✓ / V2 ✓ / V3 ✓
- **Test definitions**: 6 (3 should_trigger / 2 should_not_trigger / 1 edge_case)
- **Hardware validation**: not performed; must be verified on a target B200
- **Distilled**: 2026-06-25
