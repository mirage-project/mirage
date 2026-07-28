# ferret-v2 dispatch — using ferret as the Stage-2/Stage-5 engine

How this skill dispatches the `ferret-kernel-agent` (V2 MODE — see the
"# V2 MODE (Runtime-V2 kernels)" section of `.claude/agents/ferret-kernel-agent.md`)
to WRITE or OPTIMIZE a Runtime-V2 kernel pair. Companion contract files (MACHINE-LOCAL —
the `~/ferret/` install is a prerequisite of this engine, see SKILL.md §Environment
prerequisites; absent ⇒ use the engineer/kda routes): `~/ferret/tasks/TEMPLATE_v2.yaml`
(task schema) and `~/ferret/docs/v2_runtime_notes.md` (ferret-shaped protocol summary).
Worked dispatch-brief exemplar (in-repo): `../applications/ferret_dispatch_w13w2.md`.

## Engine routing (Stage 2 decision, one rule)

| Engine | Dispatch when | Deliverable |
|---|---|---|
| `v2-kernel-engineer` | protocol-heavy house-style port / FIRST bring-up; correctness+protocol-cleanliness is the goal, perf secondary; no faithful gate exists yet | correct pair, auditor-signed |
| **ferret V2 MODE** | a faithful FROZEN gate can exist (op wire-able + harness case + live baseline anchor) and the goal is a NUMERIC TIER-2 target over many autonomous iterations; over-claim tolerable (gate + TIER-1 catch it). Also the "write from spec" path: REPRODUCE = implement-to-gate over a wired stub | pair + gate evidence (`GATE_RESULT`, capped `KERNEL_RESULT`), "pending TIER-1" unless run |
| `kda-kernel-agent` | verdict-grade transfer: the number decides a campaign verdict (U-row kill/park threshold) and over-claim is costly | pair/kernel + faithful evidence ledger |

Default remains the engineer for a first v2 port; prefer ferret once the op is
wired and the question becomes "how fast can this body go"; prefer KDA when the
answer will be ACTED ON as a campaign verdict.

## The ferret-v2 flow (stage order is REARRANGED — wiring precedes the run)

```
S1 SPEC (designer)  ──►  S3-stub WIRE (orchestrator: compile-clean stub pair,
   pins ABI/regions/SEMs      registration per wiring-recipe.md, env-gated
                              default-OFF, harness case added)
        │                          │
        ▼                          ▼
   ferret dispatch:  test-writer freezes gate/ (harness-wrapping check.py,
   task.yaml from TEMPLATE_v2    bars from the spec §gates, baseline live-
                                 benched, gate.sha256 + manifest.sha256 over
                                 the in-tree harness+wiring files)
        │
        ▼
   ferret rounds (S2+S5 fused): REPRODUCE (implement spec → gate correctness
   PASS) → OPTIMIZE (beat target_ratio on TIER-2 body_span; Class A .cuh-only
   default, Class B spec.h-geometry budgeted w/ library rebuild)
        │
        ▼
   S4/S5 acceptance: orchestrator lands the pair → TIER-1 in-MPK slowCTA
   (box for TP8-geometry ops) → S6 review (ablation-logic-reviewer + Codex)
```

**Why wiring first:** the v2 gate substrate is the in-tree harness
(`tests/runtime_python/blackwell_v2/`), which only runs REGISTERED ops. A
standalone role-loop emulator was rejected as a gate (it re-derives the
ring/page/dep protocol — the gate-fidelity failure class). The stub pair is
mechanical from the Stage-1 spec's §SMEM/§SEM tables and pins the ABI ferret
must keep.

## What the dispatch prompt must contain (inputs)

1. `v2.spec_doc` — the Stage-1 spec absolute path (e.g.
   `<repo>/.claude/skills/v2-kernel-writing/applications/ffn_item1_spec.md`). It is the
   requirement of record.
2. The task.yaml (authored by the ferret dispatcher from `TEMPLATE_v2.yaml`):
   configs + HIGH-AMBITION `target_ratio` (derived from the spec's predicted-Δ
   band fast edge / the proven source-engine anchor — show the arithmetic),
   `protocol_frozen` restated per-op, `iteration_classes` budget,
   `stage_gate.strict: true`.
3. Wiring status + env gate name (`v2.wiring`) and the gate mirage tree
   (`mirage_root`, shadow overlay allowed).
4. Workspace index (explicit, per the parallel-dispatch rule) + a torch-probed
   free local GPU for the harness.
5. The gate bars verbatim from the spec's gates section (cos/rel_max/no-NaN,
   case matrix incl. multiple routings/activity counts for routed ops, watch
   metrics with park thresholds).

## What ferret returns (what Stage 3/5 consume)

- `~/ferret/workspace<N>/<op>_v2.cuh` + `<op>_v2_spec.h` — house-style pair,
  drop-in because the stub wiring pinned the ABI. Landing = copy the pair over
  the stub files in `include/mirage/persistent_kernel/tasks/blackwell_v2/`,
  `bash scripts/format.sh`, re-run the harness once in-tree (paranoia re-run —
  the gate ran on an overlay). Class-B (spec.h geometry) winners additionally
  need `pip install -e . --no-build-isolation --no-deps` (task_register.cc
  includes the spec.h — a JIT-only spec.h change skews planner-vs-kernel
  geometry; wiring-recipe §9's ".cuh/spec → JIT only" does NOT hold for
  planner-visible spec.h fields).
- Gate evidence: `GATE_RESULT` (correctness metrics per case), the capped
  `KERNEL_RESULT`/`KERNEL_RESULT_REFERENCE` pair, `gate.sha256` +
  `manifest.sha256` (proof the judge was never touched), progress.md history.
- Scope statement: **TIER-2 only** — the report says "pending TIER-1" unless
  the dispatcher ran it. Stage 5's verdict-grade number remains TIER-1 in-MPK
  slowCTA at the production grid (box for TP8-geometry ops, run by the
  orchestrator per the box-ownership rule).

## Seed notes (hard-won run-1 facts — paste into every ferret-v2 dispatch prompt)

From the dsv3_ffn_gg_v2 run (ws7, 2026-07-15). Each cost a debug round; seeding them is free.
(Notes 3's workspace/gate state is MACHINE-LOCAL `~/ferret/workspace7` state from the
original machine — on a fresh clone/other machine read it as campaign history + gate-design
guidance, not as live state. The run's deliverable pair is working-tree-only, not committed.)

1. **Verify TMA descriptor RANK empirically PER-OP before copying a reference idiom.** The
   pipe weight descriptor is rank-5 (`tma.cuh fill_tma_desc` hardcodes `tma_dim=5` /
   `tensorRank=5`); linear's 3D idiom
   (`cp.async.bulk.tensor.3d.shared::cluster...cta_group::1.L2::cache_hint`) is ILLEGAL against
   it → illegal-instruction at runtime. Match the instruction dimensionality to the descriptor
   (`.5d` + zero-padded coords for size-1 dims) and drop qualifiers the form doesn't accept;
   re-add L2 hints only with valid syntax for that form.
2. **Never `#include` both `blackwell/sm100_utils.cuh` and `blackwell_v2/sm100_utils.cuh` in
   one TU.** They define the SAME `kernel::sm100::` symbols; the megakernel test.cu already
   pulls the v1 header transitively → include the v1 header only (`#pragma once` makes it
   idempotent). The byte-identical v2 copy is a redefinition error, not a convenience.
3. **Profiled `body_span` scoring (TIER-2 gates) — RE-ELIGIBLE (2026-07-16).** The 2026-07-15
   "profiled-scoring fragility" that SUSPENDED profiled scoring is resolved: the wedges were
   never a profiler×tcgen05 codegen Heisenbug — they were the v2 runtime races, root-caused +
   fixed 2026-07-16 (`689dadc5`, `7d271a01`+`7b6ae2bb`, `025029a1`; mechanisms in
   `validation-debug.md` §5.1). Race 3 (iteration-barrier half-exit) explains the profiled
   BIAS specifically: profiled builds force iterations, making the racy loop-exit read the
   sole exit — that is why only profiled/gate-density runs wedged while unprofiled passed.
   Former wedge windows (reference mlp L=6/iters=32 profiled; L=4) PASS post-fix, so the ws7
   gate's fallback-(c) unprofiled iteration-wall scoring is no longer forced.
   **Re-verify note (one-time per gate freeze): run the reference-profiled pre-flight once at
   the gate's exact L/iters on a ≥`7b6ae2bb` tree before trusting profiled `body_span`** — a
   wedge there is now a NEW bug (triage per validation-debug.md §5), not the old infra class.
   Still-valid gate-design guidance: (a) that reference-only pre-flight remains the
   infra-vs-candidate discriminator; (b) a candidate wedge = round FAIL + protocol audit;
   (c) never lower `iters` below the profiling window or ship an unscored PASS.

## Friction escape (mandatory)

If ferret burns ≥2 rounds on the SAME protocol-invariant wedge class (harness
timeout/IMA traced to mbar/page/dep protocol, not math): STOP the ferret run;
dispatch `v2-kernel-engineer` to write/repair the protocol shell (role
skeletons, mbar re-init paths, page release, bail paths) with the
`b200-mbarrier-protocol-auditor` ledger; then re-dispatch ferret restricted to
the MATH-ONLY inner body (mainloop/epilogue) with the shell added to the frozen
surface (hash the shell regions of the file into the gate manifest, or split
shell/body into separate includes). This preserves the user directive (ferret
writes/optimizes v2 kernels) without letting protocol wedges eat the budget.

## Anti-loop / safety notes

- One harness wedge = round FAIL + protocol audit BEFORE the next candidate;
  never re-run a wedge unchanged; never crash-loop the megakernel.
- Ferret never edits: `gate/`, `tests/runtime_python/**`, the wiring files,
  anything in `protocol_frozen`. The gate manifest re-verify catches it.
- Default build stays byte-identical: the candidate path is env-gated
  default-OFF at the builder; new task types are additive.
- Per-op evidence rows (m1-decode-evidence.md) still gate the DISPATCH itself:
  do not ferret an op class measured DEAD without a new mechanism.
