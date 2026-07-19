---
name: b200-warp-specialized-debugger
description: >-
  Use when a B200/TIRx/CUDA warp-specialized kernel fails to compile, deadlocks, hits an illegal memory access, produces wrong results, or is "correct but slow". First verifies the environment and a minimal reproduction, then builds a roles/storage/handoff/lifetime worksheet from the generated CUDA/PTX, fixing one handoff at a time. Not for cases with no reproducible code yet or purely high-level model accuracy issues.
source_book: "Modern GPU Programming For MLSys (MLC Community) + NVIDIA Blackwell Tuning/Compatibility Guides"
source_chapter: "S15; S13; S14"
tags: ["b200", "blackwell"]
related_skills: ["b200-scope-layout-dispatch", "b200-mbarrier-protocol-auditor", "b200-layout-contract-auditor", "blackwell-build-compatibility-auditor"]
version: "0.1.0"
---

<!-- Distilled from "Modern GPU Programming for MLSys" — https://mlc.ai/modern-gpu-programming-for-mlsys/ -->

# B200 Warp-Specialized Kernel Debugger

## R — Source evidence (Reading, paraphrased)

- [S15] Before debugging, confirm the actually imported TVM/TIRx, the GPU capability, and the target; these examples target Blackwell `sm_100a`.
- [S15] Runtime problems usually reduce to a broken handoff: an uninitialized barrier, a wrong arrival count/phase, a collective hidden inside a partial-role branch, a missing visibility fence, or storage reused too early.
- [S15] Check role guards, mbarrier init, tcgen05, TMA, and CTA/warpgroup synchronization in the generated CUDA rather than rewriting the Python DSL first.

> Source: distilled from "Modern GPU Programming for MLSys" (https://mlc.ai/modern-gpu-programming-for-mlsys/) and the NVIDIA Blackwell tuning/compatibility guides. Short paraphrases only; no long passages are reproduced.

---

## I — Methodology skeleton (Interpretation)

The object of debugging in an asynchronous specialized kernel is not "lines of code" but the data handoffs across roles. The unified worksheet:

- Roles: who issues each asynchronous operation?
- Storage: where does the tile live in GMEM/SMEM/TMEM/register at each step?
- Handoff: producer, consumer, signal, count, phase, fence/drain.
- Lifetime: the earliest point it may be read, overwritten, or freed.

First classify the symptom as compile / deadlock / crash / wrong result / correct-but-slow, then follow a different evidence path for each.

---

## A1 — Applications in the source (Past Application)

### Case 1: deadlock
Common root causes: barrier init inside a role branch, a wrong arrival count, producer/consumer starting on the same initial phase, CTA-wide sync placed inside a warpgroup branch.

### Case 2: wrong result
Common root causes: mismatched TMA/MMA/TMEM layouts, waiting on the wrong stage/phase, stores not drained, O/S/P regions reused too early.

### Case 3: correct but slow
Common root causes: roles still running serially, pipeline bubbles, the producer becoming the submission bottleneck, resources leaving too few active clusters, the compiler not generating the expected specialized instructions.

---

## A2 — Trigger scenarios (Future Trigger) ★

### In what situations will the user need this skill?

1. "This warp-specialized kernel deadlocks — how do I debug it systematically?"
2. "The generated TIRx code is correct but very slow — help me look at the lowering."
3. "After an illegal access all subsequent Python runs are broken — how do I get a minimal reproduction?"

### Language signals

- "This warp-specialized kernel deadlocks — how do I debug it systematically?"
- "The generated TIRx code is correct but very slow — help me look at the lowering."
- "After an illegal access all subsequent Python runs are broken — how do I get a minimal reproduction?"

### Distinction from adjacent skills

Versus `b200-mbarrier-protocol-auditor`: the debugger first builds global evidence organized by symptom; once a barrier is implicated, invoke the dedicated auditor. Versus the build compatibility skill: the former handles kernel runtime/lowering, the latter handles binary/arch compatibility.

---

## E — Executable steps (Execution)

Once the skill is activated, the agent must follow this procedure:

1. **Verify the runtime context**
   - Print the actual TVM/TIRx/torch paths and versions, the GPU name/capability, and the compile target.
   - After an illegal access, restart the process/context before reproducing.
2. **Narrow the reproduction**
   - Smallest still-failing shape, fixed seed, unrelated fusion and concurrency turned off.
3. **Run minimal correctness first**
   - Prove the reference before any performance work; on a compile fail, do not descend into runtime synchronization guesses.
4. **Save the generated code**
   - Search for role guards, `mbarrier_init`, `tcgen05`, `cp.async.bulk.tensor`, cluster/CTA sync, and TMEM alloc/free.
5. **Fill in the four-column worksheet**
   - roles / storage / handoff / lifetime.
6. **Branch by symptom**
   - Compile: API, target, dispatch, buffer scope, unsupported shape.
   - Deadlock: init/count/phase/collective scope/commit arrival.
   - Crash: addresses, descriptors, TMEM/SMEM out-of-bounds, context poisoning.
   - Wrong: layout, stale phase, missing fence, premature reuse.
   - Slow: whether the specialized instructions were generated, role timelines, pipeline bubbles, resources/occupancy.
7. **Change only one handoff at a time**
   - Record the minimal test and the generated-code diff before and after the change.
8. **Re-validation order**
   - correctness → sanitizer/boundary → profiler → large-shape performance.
9. **Produce a reproducible report**
   - Environment, commands, shapes, locations of the generated-code snippets, expected/actual, minimal diff.

### Required outputs

1. **Conclusion**: the current choice/diagnosis, never a vague "we may need to look at everything".
2. **Evidence or assumptions**: which items come from user data and which are assumptions pending verification.
3. **Contract/table/timeline**: the auditable intermediate artifacts corresponding to this skill.
4. **Minimal validation**: a correctness test, a boundary test, and one falsifiable experiment.
5. **Risks and fallback**: the alternative path when hardware, version, or resource requirements are not met.

---

## B — Boundaries (Boundary) ★

### Do not use when
- There is no code, log, shape, or reproduction steps — just "it feels slow".
- A purely model-level training-loss anomaly that has not been shown to come from a custom kernel.

### Failure modes
- Rewriting the whole kernel as the first move.
- Not restarting the context after an illegal access and treating the subsequent phantom errors as new evidence.
- Reading only the DSL source and not the generated CUDA/PTX.
- Modifying multiple barriers/layouts/roles at the same time.

### Limitations
- Some hardware-level hangs or compiler bugs require a minimal reproduction filed upstream; this skill can produce a high-quality issue, but it cannot guarantee a local workaround for every toolchain defect.

---

## Related skills

- **depends-on**: `b200-scope-layout-dispatch`
- **contrasts-with**: none
- **composes-with**: `b200-mbarrier-protocol-auditor`, `b200-layout-contract-auditor`, `blackwell-build-compatibility-auditor`

---

## Audit info

- **Validation passed**: V1 ✓ / V2 ✓ / V3 ✓
- **Test definitions**: 6 (3 should_trigger / 2 should_not_trigger / 1 edge_case)
- **Hardware validation**: not performed; must be verified on a target B200
- **Distilled**: 2026-06-25
