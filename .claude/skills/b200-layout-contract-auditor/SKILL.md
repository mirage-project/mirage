---
name: b200-layout-contract-auditor
description: >-
  Use when a B200/Blackwell kernel shows wrong results, uncoalesced global memory access, SMEM bank conflicts, a TMA swizzle that mismatches the Tensor Core read, or confused TMEM/register ownership. Audits shape–stride, thread distribution, swizzle, and the hardware operand contract layer by layer. Not for pure synchronization deadlocks or compile errors unrelated to memory layout.
source_book: "Modern GPU Programming For MLSys (MLC Community) + NVIDIA Blackwell Tuning/Compatibility Guides"
source_chapter: "S4; S5; S6; S7; S8"
tags: ["b200", "blackwell"]
related_skills: ["b200-scope-layout-dispatch", "b200-mbarrier-protocol-auditor", "b200-tma-pipeline-designer", "b200-tmem-lifecycle-planner", "b200-tcgen05-mma-contract-builder", "b200-warp-specialized-debugger"]
version: "0.1.0"
---

<!-- Distilled from "Modern GPU Programming for MLSys" — https://mlc.ai/modern-gpu-programming-for-mlsys/ -->

# B200 Layout Contract Auditor

## R — Source evidence (Reading, paraphrased)

- [S4] A layout is the mapping from logical index to physical location; it directly determines coalescing, bank conflicts, and whether the hardware can read the tile at all.
- [S5] The two constraints invariant across generations are global coalescing and shared-memory bank conflicts; Tensor Cores add specific operand layout contracts on top.
- [S6] The TMA descriptor, the target SMEM swizzle, and the downstream MMA's interpretation of the layout must agree exactly.
- [S8] TMEM is a two-dimensional `TLane × TCol` address space; it must not be understood as an ordinary SMEM byte array.

> Source: distilled from "Modern GPU Programming for MLSys" (https://mlc.ai/modern-gpu-programming-for-mlsys/) and the NVIDIA Blackwell tuning/compatibility guides. Short paraphrases only; no long passages are reproduced.

---

## I — Methodology skeleton (Interpretation)

A layout audit must proceed hop by hop along the data path:

`logical tensor → GMEM access → SMEM tile/swizzle → Tensor Core operand → TMEM accumulator/scale → register fragment → output store`

Each hop answers four things: the index formula, the physical stride, the owner, and the consumer's expectation. If any one of them disagrees, it can show up as degraded performance or a silent error. In particular, distinguish a "view/stride rewrite" from an "ownership change": the latter usually requires a real data movement.

---

## A1 — Applications in the source (Past Application)

### Case 1: TMA writes a 128B swizzle, but the MMA reads a different layout
- The logical tile's values are unchanged, but the physical bank arrangement disagrees.
- The hardware does not report a "layout mismatch"; it interprets the addresses it receives and computes on the wrong elements.

### Case 2: a transpose mistakenly assumed to be free
- A view over a single linear storage can change only the strides.
- If the transpose also changes lane/register ownership or the SMEM swizzle, then a load/store/shuffle/specialized instruction must occur.

---

## A2 — Trigger scenarios (Future Trigger) ★

### In what situations will the user need this skill?

1. "This TMA + tcgen05 kernel is numerically wrong, but no address goes out of bounds."
2. "Help me find shared memory bank conflicts and swizzle problems."
3. "How does this TMEM accumulator map back to each lane's register fragment?"

### Language signals

- "This TMA + tcgen05 kernel is numerically wrong, but no address goes out of bounds."
- "Help me find shared memory bank conflicts and swizzle problems."
- "How does this TMEM accumulator map back to each lane's register fragment?"

### Distinction from adjacent skills

Versus `b200-mbarrier-protocol-auditor`: this skill checks "where the bytes are and who owns them"; the barrier auditor checks "when they may be read and when they may be reused". Wrong results often require invoking both in combination.

---

## E — Executable steps (Execution)

Once the skill is activated, the agent must follow this procedure:

1. **Freeze the logical semantics**
   - Write down each tensor/tile's logical shape, axis meanings, transpose relationships, and the expected element formula.
2. **Audit GMEM coalescing**
   - List lane→address for one warp; check contiguity, alignment, transaction count, and boundary tiles.
3. **Audit the SMEM bank mapping**
   - Compute banks for the consuming instruction's access pattern; identify same-bank different-address conflicts.
   - When choosing a swizzle, prefer the largest atom that the tile's contiguous dimension can fill; drop to a smaller atom when it cannot.
4. **Check the three-way contract**
   - The TMA tensor-map descriptor's shape/stride/tile/swizzle.
   - The SMEM layout declared by the DSL/code.
   - The MMA/specialized load's interpretation of the operand.
   - All three must agree item by item.
5. **Audit TMEM**
   - Pin down the `TLane`, `TCol` mapping and the column base.
   - Distinguish the accumulator layout from the block-scale layout; both live in TMEM but usually differ.
6. **Audit the register fragment**
   - Write down which elements each lane holds; confirm the `tcgen05.ld` shape/repeat matches the epilogue's expectation.
7. **Identify the real data-movement points**
   - Label views that change only shape/stride separately from rearrangements that change owner/swizzle.
8. **Output fix recommendations and validation vectors**
   - Use small, non-symmetric matrices (so transpose errors are not masked by symmetric data).
   - Design an observable sentinel pattern for each hop.

### Required outputs

1. **Conclusion**: the current choice/diagnosis, never a vague "we may need to look at everything".
2. **Evidence or assumptions**: which items come from user data and which are assumptions pending verification.
3. **Contract/table/timeline**: the auditable intermediate artifacts corresponding to this skill.
4. **Minimal validation**: a correctness test, a boundary test, and one falsifiable experiment.
5. **Risks and fallback**: the alternative path when hardware, version, or resource requirements are not met.

---

## B — Boundaries (Boundary) ★

### Do not use when
- A pure barrier deadlock with no sign of a data-layout issue.
- Merely a Python API rename or a link failure.

### Failure modes
- Validating with all-0, all-1, or symmetric inputs, which masks misplacement.
- Looking only at logical shapes without tracking lane/warp/CTA ownership.
- Assuming every transpose/view is zero-copy.

### Limitations
- Some layout contracts depend on the exact PTX instruction forms and the compiler lowering; they must be verified against the generated CUDA/PTX.

---

## Related skills

- **depends-on**: `b200-scope-layout-dispatch`
- **contrasts-with**: `b200-mbarrier-protocol-auditor`
- **composes-with**: `b200-tma-pipeline-designer`, `b200-tmem-lifecycle-planner`, `b200-tcgen05-mma-contract-builder`, `b200-warp-specialized-debugger`

---

## Audit info

- **Validation passed**: V1 ✓ / V2 ✓ / V3 ✓
- **Test definitions**: 6 (3 should_trigger / 2 should_not_trigger / 1 edge_case)
- **Hardware validation**: not performed; must be verified on a target B200
- **Distilled**: 2026-06-25
