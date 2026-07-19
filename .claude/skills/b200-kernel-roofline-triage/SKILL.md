---
name: b200-kernel-roofline-triage
description: >-
  Use when the user asks "why is this B200/Blackwell kernel slow, what should I optimize first, should I fuse or move to Tensor Cores/TMA". Classifies the problem via arithmetic intensity, dataflow, and resource occupancy as memory-bandwidth-, compute-throughput-, latency/concurrency-, or scheduling-bound, and gives a minimal falsifiable experiment. Not for queries that only ask about hardware specs with no kernel/operator context.
source_book: "Modern GPU Programming For MLSys (MLC Community) + NVIDIA Blackwell Tuning/Compatibility Guides"
source_chapter: "S3; S16"
tags: ["b200", "roofline", "performance", "triage"]
related_skills: ["b200-gemm-optimization-ladder", "b200-layout-contract-auditor", "b200-tma-pipeline-designer", "b200-warp-specialized-debugger"]
version: "0.1.0"
---

<!-- Distilled from "Modern GPU Programming for MLSys" — https://mlc.ai/modern-gpu-programming-for-mlsys/ -->

# B200 Kernel Roofline Triage

## R — Source evidence (Reading, paraphrased)

- [S3] Split the kernel's ceiling into a "compute roof" and a "bandwidth roof", and use arithmetic intensity to decide which side is more likely the current constraint.
- [S3] For low-arithmetic-intensity operators, prioritize reducing bytes, fusion, reuse, or a narrower dtype; for high-arithmetic-intensity GEMMs, the focus is keeping the Tensor Cores continuously busy.
- [S16] Blackwell still inherits the general CUDA best practices: parallelize, reduce Host↔Device transfers, coalesce accesses, and reduce redundant accesses and warp divergence.

> Source: distilled from "Modern GPU Programming for MLSys" (https://mlc.ai/modern-gpu-programming-for-mlsys/) and the NVIDIA Blackwell tuning/compatibility guides. Short paraphrases only; no long passages are reproduced.

---

## I — Methodology skeleton (Interpretation)

Do not start optimizing from "this trick is new"; answer three questions first:

1. **How much useful compute is done per output element?** Estimate the FLOPs.
2. **How many bytes were moved, from which level of storage, to do that compute?** Give at least the HBM accounting; add L2/SMEM accountings when necessary.
3. **Does the current implementation actually convert the theoretical roof into hardware busyness?** Even with high algorithmic arithmetic intensity, a wrong layout, serialized load/compute/store, resource pressure, or the launch shape can still leave the Tensor Cores idle.

The final diagnosis must not just say "memory-bound/compute-bound"; it must also give the evidence, the alternative explanations not yet ruled out, and the next minimal experiment.

---

## A1 — Applications in the source (Past Application)

### Case 1: a large GEMM that should be compute-bound yet measures low
- **Problem**: the matrices are large enough that by arithmetic intensity it should sit to the right of the ridge under the compute roof, yet Tensor Core utilization is low.
- **How the methodology was used**: after ruling out HBM bandwidth, check whether execution is still serialized as "load→compute→store" and whether TMA, software pipelining, or warp specialization is missing.
- **Conclusion**: the bottleneck is not "move a little less HBM traffic" but idle compute engines and insufficient pipeline overlap.

### Case 2: elementwise operators such as RMSNorm/GELU
- **Problem**: adding more math optimizations barely changes performance.
- **How the methodology was used**: their FLOPs/byte is very low, so first check coalescing, the number of reads/writes, fusion opportunities, and the dtype.
- **Conclusion**: the goal is to approach the bandwidth roof, not to chase Tensor Core peak.

---

## A2 — Trigger scenarios (Future Trigger) ★

### In what situations will the user need this skill?

1. "This kernel only hits xx TFLOPS on B200 — where do I look first?"
2. "Should this operator be fused, or switched to TMA/Tensor Cores?"
3. "Do a roofline diagnosis for me and give an ordered sequence of optimization experiments."

### Language signals

- "This kernel only hits xx TFLOPS on B200 — where do I look first?"
- "Should this operator be fused, or switched to TMA/Tensor Cores?"
- "Do a roofline diagnosis for me and give an ordered sequence of optimization experiments."

### Distinction from adjacent skills

Versus `b200-gemm-optimization-ladder`: this skill first determines the bottleneck and the optimization direction; the latter gives the step-by-step implementation route specifically for GEMM. Versus `b200-layout-contract-auditor`: this skill does global performance attribution; the latter audits addresses and hardware layout contracts in depth.

---

## E — Executable steps (Execution)

Once the skill is activated, the agent must follow this procedure:

1. **Collect the minimal fact set**
   - The operator formula, shapes, dtypes, batch, and whether inputs/outputs are reused.
   - The current implementation path (CUDA/Triton/TIRx/CUTLASS/framework op), the timing methodology, warmup, and whether communication is included.
   - The B200 model/power limit/clocks, the compile target, and a profiler summary.
   - Done criterion: you can write a one-line estimate of "useful FLOPs" and a one-line estimate of "HBM bytes".
2. **Compute at least one roofline accounting**
   - `AI_HBM = useful_FLOPs / HBM_bytes`.
   - Prefer the measured bandwidth on the user's machine and the same-dtype peak; without them, only order-of-magnitude inference is possible — label the assumptions explicitly.
   - Done criterion: an initial memory-bound / compute-bound / near-ridge call.
3. **Check whether the implementation contradicts the initial call**
   - memory-bound: check repeated reads/writes, intermediate tensors spilled to HBM, uncoalesced accesses, an overly wide dtype, and insufficient request concurrency.
   - compute-bound: check Tensor Core instructions, tile utilization, TMA/compute/store overlap, warp roles, tail tiles, and small shapes.
   - Neither fits: check launch latency, synchronization, CPU submission, communication, power/frequency, and occupancy resource pressure.
4. **Build the evidence matrix**
   - For each candidate bottleneck write "supporting evidence / counter-evidence / measurement needed".
   - Done criterion: at least 3 candidates listed, and they must not all be the same class of micro-optimization.
5. **Design the minimal falsifiable experiment**
   - Change only one factor at a time, e.g. disable fusion, switch to a contiguous layout, increase the pipeline depth, use a fixed shape, lock the clocks.
   - Stopping rule: if the experiment result contradicts the hypothesis, go back to step 3; do not keep stacking the same class of optimization.
6. **Output the optimization order**
   - P0: correctness and timing credibility; P1: the roof-determined primary bottleneck; P2: secondary scheduling/resource issues; P3: fine-tuning.

### Required outputs

1. **Conclusion**: the current choice/diagnosis, never a vague "we may need to look at everything".
2. **Evidence or assumptions**: which items come from user data and which are assumptions pending verification.
3. **Contract/table/timeline**: the auditable intermediate artifacts corresponding to this skill.
4. **Minimal validation**: a correctness test, a boundary test, and one falsifiable experiment.
5. **Risks and fallback**: the alternative path when hardware, version, or resource requirements are not met.

---

## B — Boundaries (Boundary) ★

### Do not use when
- The user only asks about B200 memory capacity, price, or rack specs, with no operator or performance question.
- With no shape, dtype, timing, or dataflow information at all, do not assert a bottleneck outright.

### Failure modes
- Treating the theoretical peak as a directly achievable promise.
- Looking only at occupancy while ignoring that explicit pipelining can already hide the latency; or conversely, looking only at pipelining while ignoring that resources prevent residency in the first place.
- Substituting a single profiler percentage for end-to-end evidence.

### Limitations
- Roofline has limited explanatory power for irregular accesses, short kernels, dependency chains, and cross-GPU communication; latency and communication models must be added when necessary.

---

## Related skills

- **depends-on**: none
- **contrasts-with**: `b200-gemm-optimization-ladder`
- **composes-with**: `b200-layout-contract-auditor`, `b200-tma-pipeline-designer`, `b200-warp-specialized-debugger`

---

## Audit info

- **Validation passed**: V1 ✓ / V2 ✓ / V3 ✓
- **Test definitions**: 6 (3 should_trigger / 2 should_not_trigger / 1 edge_case)
- **Hardware validation**: not performed; must be verified on a target B200
- **Distilled**: 2026-06-25
