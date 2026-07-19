---
name: b200-tcgen05-mma-contract-builder
description: >-
  Use when the user needs to choose the tile, dtype, `cta_group::1/2`, SMEM operand layout, or TMEM accumulator mapping for a `tcgen05` MMA on B200/Blackwell, or to implement an mxfp8/nvfp4 block-scaled GEMM. Produces an auditable MMA contract and completion protocol. Not for ordinary CUDA-core matmul or non-Blackwell targets.
source_book: "Modern GPU Programming For MLSys (MLC Community) + NVIDIA Blackwell Tuning/Compatibility Guides"
source_chapter: "S7; S8; S5; S16"
tags: ["b200", "blackwell"]
related_skills: ["b200-scope-layout-dispatch", "b200-layout-contract-auditor", "b200-tmem-lifecycle-planner", "b200-mbarrier-protocol-auditor", "b200-cluster-persistent-scheduler", "b200-gemm-optimization-ladder"]
version: "0.1.0"
---

<!-- Distilled from "Modern GPU Programming for MLSys" — https://mlc.ai/modern-gpu-programming-for-mlsys/ -->

# B200 tcgen05 MMA Contract Builder

## R — Source evidence (Reading, paraphrased)

- [S7] `tcgen05` is the Blackwell Tensor Core instruction family, issued by a single elected thread on behalf of the participating group; the operation itself is asynchronous.
- [S7] A/B usually reside in SMEM and the accumulator in TMEM; `cta_group::2` makes the two CTAs of the same cluster cooperate, each keeping its own accumulator fragment.
- [S7] The block-scaled mode adds SFA/SFB: the data stays in SMEM, while the scale factors are supplied through TMEM.
- [S16/S17] B200 is compute capability 10.0; when using architecture-conditional features, make the portability boundary explicit.

> Source: distilled from "Modern GPU Programming for MLSys" (https://mlc.ai/modern-gpu-programming-for-mlsys/) and the NVIDIA Blackwell tuning/compatibility guides. Short paraphrases only; no long passages are reproduced.

---

## I — Methodology skeleton (Interpretation)

A complete MMA contract contains at least:

- the math shape `M×N×K`, the input/accumulate dtypes, and whether it is block-scaled;
- the participating unit, `cta_group::1` or `cta_group::2`;
- which address space and layout each of A/B/scale/C lives in;
- who issues, who participates, and when it completes;
- the accumulator's mapping in TMEM and how the epilogue reads it back;
- boundary tiles, alignment, and the toolchain target.

If only a PTX mnemonic is given, without this contract, the agent should not generate code that merely "looks like it compiles".

---

## A1 — Applications in the source (Past Application)

### Case 1: `cta_group::1, M=128`
- A single CTA supplies the SMEM tiles of A/B.
- The 128 M rows map directly onto TMEM's 128 Lane rows; N maps onto Col.

### Case 2: `cta_group::2, M=256`
- Two CTAs cooperate; each CTA owns 128 M rows and keeps the corresponding accumulator in its own TMEM.
- The even CTA is responsible for issuing the operation and for pair completion.

### Case 3: block-scaled nvfp4/mxfp8
- The quantized A/B data is read from SMEM.
- SFA follows A's M partitioning; SFB, because both CTAs share B, must be visible/multicast to the pair.

---

## A2 — Trigger scenarios (Future Trigger) ★

### In what situations will the user need this skill?

1. "Choose the tcgen05 tile and cta_group for this B200 GEMM."
2. "Where should the scale factors of an nvfp4 block-scaled MMA go?"
3. "How is the accumulator split across the two CTAs' TMEM under cta_group::2?"

### Language signals

- "Choose the tcgen05 tile and cta_group for this B200 GEMM."
- "Where should the scale factors of an nvfp4 block-scaled MMA go?"
- "How is the accumulator split across the two CTAs' TMEM under cta_group::2?"

### Distinction from adjacent skills

Versus `b200-tmem-lifecycle-planner`: this skill first defines the MMA's math and hardware contract; the TMEM planner goes deeper into column budgeting and reuse. Versus `b200-cluster-persistent-scheduler`: this skill focuses on a single cooperative MMA; the latter focuses on cluster-level scheduling and tails.

---

## E — Executable steps (Execution)

Once the skill is activated, the agent must execute the following procedure:

1. **Confirm the target and toolchain**
   - Target GPU, compute capability, whether `sm_100a` architecture-conditional features are allowed.
   - dtype and numerical-error requirements.
2. **Define the math tile**
   - `M/N/K`, transposes, accumulate semantics, boundary/remainder handling.
3. **Choose the CTA group**
   - `group::1`: simpler single-CTA resources and synchronization.
   - `group::2`: a larger cooperative tile and cross-CTA sharing, but added cluster/DSMEM/remote-barrier complexity.
4. **Define operand placement**
   - A/B SMEM layout, swizzle, and the slice each CTA holds.
   - For block-scaled, list the SFA/SFB shapes, the K block size, and the SMEM→TMEM copy.
5. **Define the accumulator mapping**
   - Write out each CTA's `TLane/TCol` formulas.
   - For modes such as M=64/128/256, make the lane packing explicit; avoid assuming a contiguous mapping.
6. **Define issue and completion**
   - The elected thread issues; the commit group is bound to a completion barrier.
   - Every TMEM consumer must run only after the barrier completes.
7. **Define the epilogue**
   - `tcgen05.ld` fragment shape, register cast/fusion, store path.
8. **Run the three-contract check**
   - The SMEM operand layout, the TMEM layout, and the async completion must all match.
9. **Output the implementation skeleton and validation**
   - Start with small shapes and random asymmetric data.
   - For dense and block-scaled separately, build a high-precision reference, error thresholds, and boundary K-block tests.

### Required outputs

1. **Conclusion**: the current choice/diagnosis, without vague "could be any of them" hedging.
2. **Evidence or assumptions**: which come from user data, and which are hypotheses awaiting verification.
3. **Contract/table/timeline**: the auditable intermediate artifacts corresponding to this skill.
4. **Minimal validation**: correctness tests, boundary tests, and one falsifiable experiment.
5. **Risks and fallback**: alternative paths when hardware, version, or resource conditions are not met.

---

## B — Boundaries (Boundary) ★

### Do not use when
- The target is Ampere/Hopper, or only CUDA cores are used.
- The shape is too small and Tensor Core tile utilization extremely low; a custom MMA has not yet been shown to be worthwhile.

### Failure modes
- Assuming every thread should issue the MMA.
- Treating the TMEM accumulator as a register fragment.
- Splitting only the compute under `cta_group::2` without clarifying each CTA's operand/accumulator ownership.
- Wrong block-scale SFA/SFB layout or visibility.

### Limitations
- The exact shapes/dtypes the instruction supports and the compiler APIs evolve; consult the current toolchain reference before generating code.

---

## Related skills

- **depends-on**: `b200-scope-layout-dispatch`, `b200-layout-contract-auditor`
- **contrasts-with**: none
- **composes-with**: `b200-tmem-lifecycle-planner`, `b200-mbarrier-protocol-auditor`, `b200-cluster-persistent-scheduler`, `b200-gemm-optimization-ladder`

---

## Audit info

- **Validation passed**: V1 ✓ / V2 ✓ / V3 ✓
- **Test definitions**: 6 (3 should_trigger / 2 should_not_trigger / 1 edge_case)
- **Hardware validation**: not performed; must be verified on a target B200
- **Distilled**: 2026-06-25
