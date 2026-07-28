---
name: blackwell-build-compatibility-auditor
description: >-
  Use when the user wants to confirm whether an existing CUDA extension/binary can run on B200, configure `compute_100/sm_100` or the architecture-specific `sm_100a`, or check PTX/cubin, CUDA Toolkit versions, JIT, and fatbin. Outputs compatibility evidence, build flags, and a smoke test. Not for kernel performance tuning once the kernel already runs correctly.
source_book: "Modern GPU Programming For MLSys (MLC Community) + NVIDIA Blackwell Tuning/Compatibility Guides"
source_chapter: "S17; S16; S15"
tags: ["blackwell", "blackwell"]
related_skills: ["b200-kernel-roofline-triage", "b200-warp-specialized-debugger", "b200-tcgen05-mma-contract-builder"]
version: "0.1.0"
---

<!-- Distilled from "Modern GPU Programming for MLSys" — https://mlc.ai/modern-gpu-programming-for-mlsys/ -->

# Blackwell Build Compatibility Auditor

## R — Source evidence (Reading, paraphrased)

- [S17] A cubin only runs within its compatible compute-capability range; PTX can be JIT-compiled on higher capabilities, so binaries are advised to retain PTX.
- [S17] `CUDA_FORCE_PTX_JIT=1` can be used to verify whether the application contains usable PTX; the variable must be unset after the test.
- [S17] CUDA 12.8 can generate native cubin for Blackwell compute capability 10.0 while also retaining `compute_100` PTX.
- [S17] Architecture-conditional features using `sm_100a/compute_100a` have no general forward/backward compatibility.

> Source: distilled from "Modern GPU Programming for MLSys" (https://mlc.ai/modern-gpu-programming-for-mlsys/) and the NVIDIA Blackwell tuning/compatibility guides. Short paraphrases only; no long passages are reproduced.

---

## I — Methodology skeleton (Interpretation)

The compatibility audit has three layers:

1. whether the **device/driver/toolchain** recognizes the target;
2. whether the **binary** contains a native cubin usable on B200 or JIT-able PTX;
3. whether the **code semantics** depend on architecture-conditional features and legacy warp-synchronous assumptions.

"It compiles on some machine" is not compatibility evidence; the artifacts and the actual load path must be checked.

---

## A1 — Applications in the source (Past Application)

### Case 1: artifacts from an old CUDA build
- If the binary retains reasonably new PTX, B200 can JIT it at runtime.
- If there are only old-architecture cubins and no compatible PTX, the kernel launch will fail and a rebuild is needed.

### Case 2: a CUDA 12.8 build
- Generating both the `sm_100` native cubin and `compute_100` PTX reduces first-run JIT while preserving future compatibility.

### Case 3: `sm_100a`
- Used when depending on specific Blackwell architecture-conditional features.
- It should not be treated as a general Blackwell/PTX fallback; there must be a capability check and a plain path.

---

## A2 — Trigger scenarios (Future Trigger) ★

### In what situations will the user need this skill?

1. "Can this PyTorch CUDA extension run directly on B200?"
2. "How should nvcc's sm_100, compute_100, and sm_100a be configured?"
3. "How do I prove the wheel/fatbin contains PTX/cubin usable on Blackwell?"

### Language signals

- "Can this PyTorch CUDA extension run directly on B200?"
- "How should nvcc's sm_100, compute_100, and sm_100a be configured?"
- "How do I prove the wheel/fatbin contains PTX/cubin usable on Blackwell?"

### Distinction from adjacent skills

Difference from `b200-warp-specialized-debugger`: this skill first settles whether it can load/run correctly at all; the latter troubleshoots handoffs and performance once the kernel has already entered compilation or execution.

---

## E — Executable steps (Execution)

Once the skill is activated, the agent must execute the following process:

1. **Record the environment matrix**
   - GPU/capability, driver, CUDA runtime/toolkit, compiler, framework, and extension versions.
2. **Inventory the binary forms**
   - Check the fatbin/cubin/PTX targets; record whether `sm_100`, `compute_100`, or only old cubins are present.
3. **Run the PTX JIT verification**
   - Temporarily set `CUDA_FORCE_PTX_JIT=1` and run a minimal kernel smoke test.
   - Success: there is at least JIT-able PTX; failure: a rebuild or dependency fix is needed.
   - Explicitly unset it after the test.
4. **Generate the build flags**
   - CUDA 12.8+: at minimum consider `-gencode=arch=compute_100,code=sm_100` and `code=compute_100`.
   - Multi-architecture wheels: keep the cubin for each target and retain at least one PTX backend target.
5. **Audit the architecture-conditional paths**
   - If `sm_100a/compute_100a` or tcgen05-specific features are used, establish device-capability detection and a non-a fallback.
6. **Check warp-synchronous assumptions**
   - Do a migration audit of intra-warp communication without explicit synchronization and partial-mask collectives.
7. **Run the smoke matrix**
   - Load the extension, a minimal kernel, representative dtypes/shapes, error-message capture, and first-JIT vs cached runs.
8. **Output the conclusion grade**
   - A: native + PTX; B: runnable via PTX JIT; C: restricted architecture path only; F: no compatible code.
   - Attach the rebuild commands and risks.

### Required outputs

1. **Conclusion**: the current choice/diagnosis; do not use a vague "it could be any of them".
2. **Evidence or assumptions**: which items come from user data, and which are hypotheses awaiting verification.
3. **Contract/table/timeline**: the auditable intermediate artifacts corresponding to this skill.
4. **Minimal validation**: correctness tests, boundary tests, and one falsifiable experiment.
5. **Risks and fallback**: alternative paths when hardware, version, or resource requirements are not met.

---

## B — Boundaries (Boundary) ★

### Do not use when
- The kernel already runs stably and the user only asks why it is not fast enough.
- Non-CUDA package compatibility problems.

### Failure modes
- Looking only at the CUDA runtime version without checking the actual binary.
- Forgetting to unset after forcing PTX JIT, misjudging later performance/startup time.
- Treating `sm_100a` as ordinary PTX that can be JIT-compiled on any future GPU.
- Wheel builds that keep only the dev machine's architecture cubin.

### Limitations
- Frameworks differ in the environment-variable names for architecture lists and in packaging logic, so the corresponding official build docs must be consulted; this skill provides the audit method, not a replacement for framework docs.

---

## Related skills

- **depends-on**: none
- **contrasts-with**: `b200-kernel-roofline-triage`
- **composes-with**: `b200-warp-specialized-debugger`, `b200-tcgen05-mma-contract-builder`

---

## Audit info

- **Validation passed**: V1 ✓ / V2 ✓ / V3 ✓
- **Test definitions**: 6 (3 should_trigger / 2 should_not_trigger / 1 edge_case)
- **Hardware validation**: not performed; must be verified on a target B200
- **Distilled**: 2026-06-25
