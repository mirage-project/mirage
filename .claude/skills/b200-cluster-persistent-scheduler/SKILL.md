---
name: b200-cluster-persistent-scheduler
description: >-
  Use when the user wants to use Thread Block Cluster, DSMEM, 2-CTA cooperative MMA, persistent kernels, a tile scheduler, or Cluster Launch Control on B200 to handle uneven tails. Outputs the cluster tile, occupancy, sharing/multicast, and static or dynamic scheduling plan. Not for kernels where independent small CTAs already saturate the machine and the workload is uniform.
source_book: "Modern GPU Programming For MLSys (MLC Community) + NVIDIA Blackwell Tuning/Compatibility Guides"
source_chapter: "S2; S10; S13; S16"
tags: ["b200", "blackwell"]
related_skills: ["b200-scope-layout-dispatch", "b200-mbarrier-protocol-auditor", "b200-tcgen05-mma-contract-builder", "b200-gemm-optimization-ladder", "b200-kernel-roofline-triage"]
version: "0.1.0"
---

<!-- Distilled from "Modern GPU Programming for MLSys" — https://mlc.ai/modern-gpu-programming-for-mlsys/ -->

# B200 Cluster & Persistent Scheduler

## R — Source evidence (Reading, paraphrased)

- [S2/S16] CTAs within a cluster can access each other's SMEM (DSMEM); the B200 portable cluster size is 8, and an explicit opt-in reaches the non-portable 16, but this may reduce the number of active blocks.
- [S13] A 2-CTA cluster can jointly compute a larger MMA tile and share operands through cluster-scope handoff.
- [S10] A persistent kernel keeps a fixed set of CTAs/clusters resident to process multiple tiles; CLC allows "stealing" work at runtime from not-yet-launched cluster coordinates, improving the tail.

> Source: distilled from "Modern GPU Programming for MLSys" (https://mlc.ai/modern-gpu-programming-for-mlsys/) and the NVIDIA Blackwell tuning/compatibility guides. Short paraphrases only; no long passages are reproduced.

---

## I — Methodology skeleton (Interpretation)

Cluster and persistent scheduling solve two different problems:

- **cluster** enlarges the spatial extent and data-sharing scope of a single cooperative tile;
- **persistent scheduling** reduces the static one-tile-one-CTA binding, letting a limited set of work owners fetch tasks in a loop;
- **CLC** on Blackwell further turns task fetching from static grid-stride into hardware-assisted dynamic tail scheduling.

Before using them, you must prove the gain comes from higher reuse, larger MMAs, or a better tail — not from "Blackwell has this feature, so use it".

---

## A1 — Applications in the source (Past Application)

### Case 1: 2-CTA GEMM
- The two CTAs each hold part of A/B/accumulator.
- DSMEM or multicast gives the pair the operands they both need.
- The barrier's CTA mask changes from single-CTA to cluster remote notification.

### Case 2: uneven tile tail
- Static grid-stride may leave some SMs idle early during the final phase.
- CLC allows a finished resident cluster to cancel a not-yet-launched cluster and take over its coordinates to continue working.

---

## A2 — Trigger scenarios (Future Trigger) ★

### In what situations will the user need this skill?

1. "Is a 2-CTA cluster worth it for this GEMM?"
2. "Help me design a persistent tile scheduler and CLC tail stealing."
3. "How do I reason about DSMEM access, cluster size, and occupancy together?"

### Language signals

- "Is a 2-CTA cluster worth it for this GEMM?"
- "Help me design a persistent tile scheduler and CLC tail stealing."
- "How do I reason about DSMEM access, cluster size, and occupancy together?"

### Distinction from adjacent skills

Difference from `b200-tcgen05-mma-contract-builder`: this skill covers cluster-level cooperation and task scheduling; the MMA builder covers the contract of a single Tensor Core operation. Difference from `b200-gemm-optimization-ladder`: this skill also applies to non-GEMM persistent/cluster kernels.

---

## E — Executable steps (Execution)

Once the skill is activated, the agent must execute the following process:

1. **Prove the necessity of the cluster**
   - Is a single CTA limited by SMEM/TMEM/operand reuse?
   - Does a 2-CTA tile raise Tensor Core utilization or reduce HBM/L2 traffic?
2. **Define the cluster shape and tile mapping**
   - clusterDim, each CTA's logical coordinates, the M/N/K slices, owners, and shared operands.
3. **Plan DSMEM/multicast**
   - Make accesses as coalesced as possible and 32B-aligned; avoid non-unit strides.
   - Write out the producers/consumers and fences for local SMEM and remote SMEM.
4. **Compute residency**
   - Use `cudaOccupancyMaxActiveClusters` or an equivalent tool.
   - Compare portable size ≤8 with the B200 opt-in 16; if using 16, set the nonportable attribute and flag the portability.
5. **Upgrade the barrier scope**
   - Update the CTA mask, remote arrival, and pair completion; check the even/leader CTA responsibilities.
6. **Choose the scheduling mode**
   - Equal-cost, uniform work: static tile formula/grid-stride.
   - Uneven cost or a pronounced tail: persistent owner + CLC work stealing.
7. **Implement the CLC protocol**
   - async try-cancel → mbarrier completion → query predicate → read the cluster coordinate only on success.
   - No numeric sentinel; termination is decided by the predicate.
8. **Evaluate the tail and fairness**
   - Test the boundary cases where the tile count is fewer than, equal to, and slightly more than the resident clusters.
9. **Output the fallback**
   - Provide the portable single-CTA/static-scheduler path and the conditions for enabling it.

### Required outputs

1. **Conclusion**: the current choice/diagnosis; do not use a vague "it could be any of them".
2. **Evidence or assumptions**: which items come from user data, and which are hypotheses awaiting verification.
3. **Contract/table/timeline**: the auditable intermediate artifacts corresponding to this skill.
4. **Minimal validation**: correctness tests, boundary tests, and one falsifiable experiment.
5. **Risks and fallback**: alternative paths when hardware, version, or resource requirements are not met.

---

## B — Boundaries (Boundary) ★

### Do not use when
- A single CTA already saturates and tile costs are uniform; a cluster would only add synchronization and resource pressure.
- It must run fully identically on non-B200/non-Blackwell platforms, yet there is no fallback.

### Failure modes
- Assuming a larger cluster size is always faster.
- DSMEM accesses that are not coalesced or not correctly fenced.
- Still reading an invalid coordinate when CLC fails.
- A persistent pool that is too large or too small, ignoring resource residency.

### Limitations
- CLC and non-portable cluster sizes are architecture-dependent features; production code must have device-capability checks and a fallback.

---

## Related skills

- **depends-on**: `b200-scope-layout-dispatch`, `b200-mbarrier-protocol-auditor`
- **contrasts-with**: none
- **composes-with**: `b200-tcgen05-mma-contract-builder`, `b200-gemm-optimization-ladder`, `b200-kernel-roofline-triage`

---

## Audit info

- **Validation passed**: V1 ✓ / V2 ✓ / V3 ✓
- **Test definitions**: 6 (3 should_trigger / 2 should_not_trigger / 1 edge_case)
- **Hardware validation**: not performed; must be verified on a target B200
- **Distilled**: 2026-06-25
