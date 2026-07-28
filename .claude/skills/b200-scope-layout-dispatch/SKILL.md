---
name: b200-scope-layout-dispatch
description: >-
  Use when the user wants to map an ML operator onto a B200/Blackwell kernel, or to review which threads should execute a given tile primitive, where the data should live, and whether to invoke thread code or TMA/tcgen05. Outputs a complete contract for scope, layout, dispatch, and handoff. Not for high-level model architecture design alone or pure hardware spec queries.
source_book: "Modern GPU Programming For MLSys (MLC Community) + NVIDIA Blackwell Tuning/Compatibility Guides"
source_chapter: "S2; S4; S6; S7; S8"
tags: ["b200", "blackwell"]
related_skills: ["b200-kernel-roofline-triage", "b200-layout-contract-auditor", "b200-tma-pipeline-designer", "b200-tcgen05-mma-contract-builder", "b200-mbarrier-protocol-auditor"]
version: "0.1.0"
---

<!-- Distilled from "Modern GPU Programming for MLSys" — https://mlc.ai/modern-gpu-programming-for-mlsys/ -->

# B200 Scope–Layout–Dispatch Designer

## R — Source evidence (Reading, paraphrased)

- [S2] Blackwell's execution hierarchy runs from thread, warp, warpgroup, CTA, and cluster up to grid; different hardware operations have different natural scopes.
- [S2] TMA is usually issued by a single thread, TMEM readback is done cooperatively by a warpgroup, and a `tcgen05` MMA is issued by one elected thread on behalf of the participating group.
- [S4] The layout determines the mapping from logical index to physical location, thread/register ownership, and bank/coalescing behavior.
- [S6–S9] Asynchronous hardware operations must have an explicit handoff; program order alone does not prove the data is ready.

> Source: distilled from "Modern GPU Programming for MLSys" (https://mlc.ai/modern-gpu-programming-for-mlsys/) and the NVIDIA Blackwell tuning/compatibility guides. Short paraphrases only; no long passages are reproduced.

---

## I — Methodology skeleton (Interpretation)

Treat kernel design as four mutually constraining tables, not one blob of mixed code:

- **Scope**: who participates in or issues this operation? A lane, a warp, a warpgroup, a CTA, or a two-CTA cluster?
- **Layout**: where does each element of the logical tile physically live? GMEM/SMEM/TMEM/register, and which lane/warp/CTA owns it?
- **Dispatch**: which hardware path performs the same logical action — plain thread instructions, TMA, `tcgen05`, shuffle, or a collective?
- **Handoff**: when does the producer prove the result is visible, what does the consumer wait on, and when is the storage allowed to be reused?

Whenever one table changes, the other three must be re-checked.

---

## A1 — Applications in the source (Past Application)

### Case 1: TMA-loading an operand tile
- Scope: one designated thread issues it; consumers within the CTA use it.
- Layout: a logical GMEM rectangle mapped to a swizzled SMEM tile.
- Dispatch: TMA descriptor + async copy.
- Handoff: a byte-count mbarrier.

### Case 2: Blackwell MMA and epilogue
- Scope: one elected thread issues it; a warpgroup/CTA or 2-CTA group participates.
- Layout: A/B in SMEM, the accumulator in TMEM, the epilogue reading back into registers.
- Dispatch: `tcgen05.mma` + `tcgen05.ld`.
- Handoff: MMA commit to a barrier; the epilogue waits for the result to complete.

---

## A2 — Trigger scenarios (Future Trigger) ★

### In what situations will the user need this skill?

1. "Map this fused op onto a B200 kernel — give me the scope/layout/dispatch first."
2. "Why does this collective deadlock when placed inside a warp branch?"
3. "Help me draw the tile lifetime across GMEM→SMEM→TMEM→register."

### Language signals

- "Map this fused op onto a B200 kernel — give me the scope/layout/dispatch first."
- "Why does this collective deadlock when placed inside a warp branch?"
- "Help me draw the tile lifetime across GMEM→SMEM→TMEM→register."

### Distinction from adjacent skills

Versus `b200-layout-contract-auditor`: this skill does the overall design; the layout auditor specifically verifies addresses, swizzle, banks, and Tensor Core contracts. Versus `b200-mbarrier-protocol-auditor`: this skill defines the handoffs; the latter verifies the protocol barrier by barrier.

---

## E — Executable steps (Execution)

Once the skill is activated, the agent must follow this procedure:

1. **Decompose the operator graph**
   - Break the logical operator into tile load, transform, MMA/reduction, epilogue, and store.
   - Done criterion: each node has exactly one primary output tile or scalar.
2. **Choose a scope for each node**
   - Record the "participation scope" and the "issue scope"; the two may differ.
   - Check that collectives execute over their full participation scope.
3. **Draw data residency and ownership**
   - For each tile write down: shape, dtype, storage space, layout, owner, live range.
   - Mark every ownership change; an ownership change usually implies a real data movement or a collective.
4. **Choose the dispatch**
   - Regular rectangular GMEM↔SMEM: evaluate TMA first.
   - Tensor Core tiles: evaluate `tcgen05` and its supported shapes/dtypes.
   - Small/irregular operations: plain threads, vectorized load/store, shuffle, or CUDA cores.
5. **Define the handoffs**
   - For each asynchronous edge write down the producer, consumer, signal, phase, fence/drain, and the point at which reuse is allowed.
6. **Produce the contract table**
   - Columns: operation / logical tile / scope / storage-layout / dispatch / readiness / release.
7. **Run a consistency audit**
   - The TMA descriptor, SMEM layout, and MMA expectations agree.
   - The TMEM mapping agrees with the `tcgen05.ld` readback.
   - No CTA-wide collective is placed inside a partial-thread branch.
8. **Give the minimal implementation order**
   - Build a synchronous correct version first, then replace pieces one by one with asynchronous/specialized paths.

### Required outputs

1. **Conclusion**: the current choice/diagnosis, never a vague "we may need to look at everything".
2. **Evidence or assumptions**: which items come from user data and which are assumptions pending verification.
3. **Contract/table/timeline**: the auditable intermediate artifacts corresponding to this skill.
4. **Minimal validation**: a correctness test, a boundary test, and one falsifiable experiment.
5. **Risks and fallback**: the alternative path when hardware, version, or resource requirements are not met.

---

## B — Boundaries (Boundary) ★

### Do not use when
- The user only needs to call cuBLAS/cuDNN and has no custom-kernel requirement.
- The algorithm itself is still undecided and even the tile-level data dependencies cannot be described.

### Failure modes
- Mistaking "who issues the instruction" for "who participates in the result".
- Drawing only logical tensor shapes and not physical ownership.
- Changing the dispatch while keeping the old synchronization protocol.

### Limitations
- The framework guarantees the design is auditable, not that the optimal tile falls out automatically; you still need to compile, run correctness tests, and measure performance.

---

## Related skills

- **depends-on**: `b200-kernel-roofline-triage`
- **contrasts-with**: none
- **composes-with**: `b200-layout-contract-auditor`, `b200-tma-pipeline-designer`, `b200-tcgen05-mma-contract-builder`, `b200-mbarrier-protocol-auditor`

---

## Audit info

- **Validation passed**: V1 ✓ / V2 ✓ / V3 ✓
- **Test definitions**: 6 (3 should_trigger / 2 should_not_trigger / 1 edge_case)
- **Hardware validation**: not performed; must be verified on a target B200
- **Distilled**: 2026-06-25
