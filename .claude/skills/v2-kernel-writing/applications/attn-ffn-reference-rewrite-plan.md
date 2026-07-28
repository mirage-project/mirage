# Staged plan: rewrite DSv3 attn + FFN toward the reference idiom

User directive: move the v2 attn/FFN kernels from "v1 fused scalar-MAC bodies hosted in the v2
shell" toward the `linear_sm100_v2` reference methodology (house-style.md), keeping deviations
only where m1-decode-evidence.md justifies them. This plan phases that work so every phase has
a correctness gate, a falsifiable predicted Δ, and a kill criterion. Run each phase through the
SKILL.md stage loop (SPEC → IMPLEMENT → WIRE → VALIDATE → PERF → REVIEW).

## Honest evidence caveats (pin these before starting)

1. The attn v2 CHAIN (7 per-op tasks: TASK_DSV3_ATTN_{P0_QKVA, QB_ROPE_KV, MLA_PARTIAL,
   MLA_FUSED, MLA_MERGE, WUV, OPROJ}_V2, runtime.cc:616-623) was measured **1.41-1.56× slower
   than the v1 fused mega AS A SCALAR CHAIN** (5 tuning rounds, walls cut 15-22%, v1 not beaten
   — commit fedcfd79). What the chain measurement did NOT test: the pipeline idiom INSIDE the
   tasks (TMA loader, tcgen05 where shaped) and real cross-task overlap — that is the delta this
   plan probes (evidence rows U1/U2/U3).
2. The grid-wide megas (TASK_DSV3_FFN_MEGA_V2=348, TASK_ATTN_BLOCK_MEGAKERNEL_V2=353) are the
   current best AND the lockstep counter-model: coarse monotonic barrier is optimal WITHIN that
   shape (W3/D4) — the rewrite's win, if any, comes from ABANDONING the shape for runtime
   overlap, not from a better barrier.
3. M=1 GEMV pieces are consumer-floor-bound (D3: 6.24µs armMAC floor; ceiling 1.18× unrealized)
   — per-task GEMV rewrites are dead; those pieces keep consumer-GEMV bodies.
4. FFN nwarps=7 tag-flag MAC-fold is a load-bearing WIN (W1, −7..−9%) whose mechanism is
   integer-trip wave-quantization — any re-tasking must re-run the
   `ceil(items/(ntasks*nwarps))` tie-test per op or it will silently regress.
5. Attn ≈ 58% of the decode step (2026-07-09 profile) — the plan's e2e ceiling is set by attn;
   FFN phases are the lower-risk rehearsal.
6. 8ms context: in-framework grind floors ~8.4ms; the exec-model restructure (Phase 4) is the
   only surviving 8ms path (`project_goal_8ms_decode`). This plan IS that restructure, done
   incrementally.

## Phase 0 — Baseline + instruments (no code)

Capture on the box, n≥3, TIER-1: per-op slowCTA table for (a) v1 megas, (b) current v2 megas,
(c) current v2 chains (attn + FFN 3-op). Wall per layer + e2e tpot. Profiler window traces via
`scripts/v2_perfetto_export.py`. These are the comparison anchors for every later phase.
Gate: mpk-correctness-gate PASS on the baseline config (routed-MoE non-null).

## Phase 1 — De-fuse to per-op v2 tasks at reference granularity

Both chains ALREADY EXIST (attn 7-op; FFN T1-T5 + folded 3-op variants). Work items:
- Make the chain the selectable default v2 path (env/flag, default-OFF) and re-verify.
- Split remaining grid-wide items into per-tile tasks where a tile axis exists (o_proj N-tiles,
  W13/W2 expert×N-tiles, lm_head already per-tile via TASK_DSV3_LMHEAD_GEMV_V2) so the runtime
  round-robin can overlap them cross-SM — per-tile granularity is what the reference exploits.
  NOTE: the W13/W2 expert×N-tiles split here and Phase-2 item 2 ("W13/W2 grouped GEMM as per-tile
  v2 pipeline tasks") are THE SAME work item — a per-tile W13/W2 split MUST carry the pipeline
  body (a per-tile split keeping the scalar GEMV body is a D6-class work-redistribution, measured
  4×-DEAD in m1-decode-evidence), so Phase-1's split and Phase-2's re-host are done together.
- Keep per-op nwarps per W1/W2 evidence (rq@4, w13@7, w2@7; attn GEMVs @4).
- **Correctness gate**: per-op v2 harness cases (cos ≥ 0.999) + attn/FFN bit-match harnesses +
  in-MPK 0-3 probe + iter≥3 + canary token-identity 0-3L.
- **Predicted Δ (falsifiable)**: chain ≈ mega ± 5% at this phase (overlap not yet exploited;
  granularity alone shouldn't cost). **Kill**: any op whose per-tile split regresses its op wall
  >10% with no overlap gain reverts to its per-op (non-tiled) form.

## Phase 2 — Pipeline idiom inside the GEMM-shaped pieces

Candidates, in dispatch order (each = one SKILL.md loop with a designer-subagent SPEC):
1. **MLA core (mla_partial/fused)** — compute-dense, warp-scales (W2: −36% @KV=544); the
   fmha QKᵀ/PV shape is MMA-able at KV ≥ ~64. SPEC must decide tcgen05 vs scalar per KV-length
   band and cite U3. Consult b200-flash-attention4-planner + b200-tcgen05-mma-contract-builder.
2. **W13/W2 grouped GEMM as per-tile v2 pipeline tasks** — tcgen05 already proven at M=1 in v1
   (W5: fp8_group_gemm swapAB); the v2 port is a re-hosting, not a new engine. Target: reference
   loader/launcher/consumer/storer roles + per-stage page release.
3. **q_b / o_proj / lm_head N-tiles** — GEMM-shaped only in the swapAB sense (M=1); tcgen05
   in-envelope is D2-DEAD, but as DE-FUSED per-tile tasks these sit in U3. Probe ONE (q_b,
   N=9216 K=1536) before committing the family.
4. **U1 probe (optional, cheap)**: TMA-loader INSIDE a consumer body (no dedicated warp) for one
   GEMV — pre-registered ceiling 1.18×; kill at <5% TIER-1 Δ.
- **Correctness gate**: per-op harness + §1.1/protocol audit (b200-mbarrier-protocol-auditor) +
  in-MPK probe + iter≥3 (new mbars/pages = new stale-arrival surface).
- **Predicted Δ**: per-op TIER-1 slowCTA vs Phase-1 body; pre-register per candidate. MLA core
  target ≥15% op-wall cut at KV≥512; W13/W2 target = parity with v1 group-GEMM body (12-57µs
  class) inside v2. **Kill**: TIER-1 Δ < 5% or any protocol hang → record in
  experiment_history + m1-decode-evidence.md and stop that candidate.

## Phase 3 — GEMV pieces stay consumer-GEMV

qkv_a, router, silu/quant, elementwise: keep 128T consumer bodies (D3/D5/D8/D9). Allowed
tuning: nwarps per the wave-quant tie-test (W1), L2 hints, cp.async ring depth ONLY if a
TIER-1 probe shows >5% (D9 says FLAT — expect no work here). This phase is a guard-rail, not
a workstream.

## Phase 4 — Runtime cross-task overlap replaces lockstep barriers (the exec-model move)

- Replace in-op monotonic grid barriers with cross-task event deps (the chain already does
  this: v1 grid barriers → v2 events, dsv3_ffn_v2.cuh:14-31 header note).
- Enable cross-task prefetch: loader of task N+1 starts weight TMAs during task N's tail
  (reference Phase-4.3/5 behavior — launcher early page release + storer per-stage release).
- Interleave attn-chain and FFN-chain tasks of ADJACENT layers where the DAG allows (the
  overlap SGLang gets from CUDA-graph boundaries; `project_sglang_trace_overlap_is_target`).
- **Correctness gate**: full mpk-correctness-gate + poison-fill on any scratch whose re-init is
  removed + iter≥3 + full-61L coherence.
- **Predicted Δ (the plan's verdict number)**: e2e tpot TP8. The chain starts 1.41-1.56× slower
  on attn; overlap + Phase-2 bodies must first CLOSE that, then beat the mega. Pre-register:
  Phase 4 SUCCEEDS iff e2e ≤ mega-baseline − 0.3ms; PARKS if within ±0.3ms; KILLS the
  chain-default if >0.3ms worse after the planned levers are exhausted.
- **Risk**: scheduler/hang class (EVENT_LAUNCH_TASKS history). Watchdog armed on every run;
  never crash-loop the box.

## Sequencing + reporting

FFN first (smaller, bit-exact refs, rehearses the loop), then attn (58% of step, the prize).
This ordering OVERRIDES Phase 2's candidate order (which lists MLA core #1): the first kernel
item is the W13/W2 pipeline tasks — its Stage-1 spec lives at
`.claude/skills/v2-kernel-writing/applications/ffn_item1_spec.md` (this folder; the ferret
dispatch brief for it is `applications/ferret_dispatch_w13w2.md`).
One phase = one or more SKILL.md loops; every phase ends with: TIER-1 table vs Phase-0 anchors,
ablation-logic-reviewer + Codex sign-off on the verdict, mpk-memory-keeper entry
(experiment_history INDEX row + memory update), commit via mpk-commit-reviewer (env-gated
default-OFF until the Phase-4 verdict).
