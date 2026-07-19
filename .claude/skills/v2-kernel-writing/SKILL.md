---
name: v2-kernel-writing
description: Runtime-V2 kernel-writing workflow. Use when writing, porting, or rewriting ANY Runtime-V2 task kernel (tasks/blackwell_v2/*.cuh + registration) — a new op, a v1→v2 port, or a rewrite toward the reference linear_sm100_v2 warp-role pipeline idiom. Drives the staged loop SPEC→IMPLEMENT→WIRE→VALIDATE→PERF→REVIEW with per-stage subagents and the b200-* sub-skills, and enforces the M=1 anti-loop evidence + the v2 protocol invariants (§1.1 dep-prefix, stale-arrival re-init, skip_after_step0, task_offset wiring).
---

You are writing a **Runtime-V2 task kernel** for MPK (dsv3-decode-clean). The house style is
the reference `linear_sm100_v2.cuh` warp-role pipeline (loader W4 TMA → launcher W5
tcgen05/TMEM → consumers W0-3 epilogue → storer W6 page release), with consumer-only as the
sanctioned idiom for non-GEMM-shaped ops. Quality bar and every protocol invariant live in
this skill's references — the loop below tells you when to read what and what to dispatch.

For MODEL-level bring-up (whole compute graph → demo) use `v2-model-support`; this skill is
the per-KERNEL inner loop that pipeline dispatches into.

## Environment prerequisites (what must exist on the machine)

Everything the loop REQUIRES travels with the repo: this skill's `references/` +
`applications/` docs, the reference kernels (`include/mirage/persistent_kernel/tasks/blackwell_v2/`),
the harness (`tests/runtime_python/blackwell_v2/`), and the wiring surface. The rest is
machine-local and DEGRADES GRACEFULLY:

- **`b200-*` sub-skills** (the Skill-tool names in the index below) are USER-global
  (`~/.claude/skills/b200-*`), NOT in this repo — a fresh clone still has them only under
  the same user account. If missing: proceed anyway; `references/house-style.md` +
  `references/upstream-kernel-catalog.md` carry the distilled protocol/layout contracts.
- **Upstream catalog reads** (`git show mirage-project/runtime_refactor:<path>`) need the
  remote: `git remote add mirage-project https://github.com/mirage-project/mirage.git &&
  git fetch mirage-project runtime_refactor`. Optional — the catalog doc is self-contained.
- **`~/ferret/`** (TEMPLATE_v2.yaml, docs/v2_runtime_notes.md, workspace1..8) — required ONLY
  for the ferret-v2 engine; **`~/kda-workspaces/`** only for kda; **`~/kernel_tools/`**
  (ncu_profile.sh) only for the NCU bound-check. Absent ⇒ route Stage 2/5 work to
  `v2-kernel-engineer` (the default anyway) and use `b200-kernel-roofline-triage`/manual NCU.
- **Personal memory** (`~/.claude/projects/-home-muhengl-mirage/memory/`) — optional context
  only (same-user machines). `references/m1-decode-evidence.md` is the self-contained
  distillation of the evidence rows; treat the memory dir as its (optional) source citations.
- **Hardware**: Stages 0-3 need no GPU. Stage 4/5 need a B200 (sm_100a); TP8-geometry ops
  need the 8-GPU box (see `v2-model-support/references/box-orchestration.md`) — without it,
  deliver with local gates + an explicit "pending TIER-1".
- **Codex MCP** (`mcp__codex__codex`) for Stage-6 double-checks — if unconfigured, the
  `ablation-logic-reviewer` pass still runs; note the missing second engine in the verdict.

## Reference docs (this skill's folder)

| Doc | What it is | Read at |
|---|---|---|
| `references/house-style.md` | Reference methodology spec (roles, SEM tables, SMEM regions, TMA/tcgen05 patterns, quality bar) | Stage 0, and by every subagent |
| `references/upstream-kernel-catalog.md` | Per-kernel/family pattern catalog of upstream runtime_refactor@0eadb3fd (which family to copy, gotchas, sync-with-upstream list) | Stage 0 (find your op's family) + Stage 1 |
| `references/m1-decode-evidence.md` | ANTI-LOOP map: DEAD / WIN / UNTESTED at M=1 decode | Stage 0 + Stage 1 |
| `references/wiring-recipe.md` | The v2 8-file registration checklist + footguns | Stage 3 |
| `references/validation-debug.md` | Gates, hang/crash tooling, TIER measurement, profiler | Stage 4 + 5 |
| `references/ferret-v2-dispatch.md` | Stage-2/5 engine routing (engineer\|ferret-v2\|kda) + the ferret-v2 flow/contract | Stage 2 + 5 |
| `applications/attn-ffn-reference-rewrite-plan.md` | The staged attn/FFN rewrite (user directive) | when working that campaign |
| `applications/ffn_item1_spec.md` | Worked Stage-1 SPEC exemplar (W13/W2 per-tile pipeline) — the deliverable shape Stage 1 must produce | Stage 1 (as template) |
| `applications/ferret_dispatch_w13w2.md` | Worked ferret-v2 dispatch brief exemplar (targets/gate/protocol_frozen/budget) | Stage 2/5 ferret dispatches (as template) |

## Dispatch pattern (subagent nesting)

The MAIN THREAD (or one lead subagent for a multi-kernel campaign) is the orchestrator: it
runs Stages 0/3/6 itself and dispatches ONE subagent per heavy stage — a **designer**
(Stage 1), an **implementer** (Stage 2 — use the `v2-kernel-engineer` agent if defined in
`.claude/agents/`, else general-purpose with that discipline pasted in), a **validator**
(Stage 4). Subagents do not dispatch subagents. Each dispatch prompt MUST name: the stage's
reference docs (absolute paths), the sub-skills to load via the Skill tool, the op contract,
and the exact deliverable. Stages are sequential; iterate 2↔4 on failures, 5→1 on a perf
verdict that changes the design.

Hard rules for every stage: default build byte-identical (new task types additive; levers
env-gated default-OFF); no repo-wide refactors; GPU safety (test-mode first, never crash-loop
the megakernel); every non-trivial conclusion → Stage 6 review before acting on it.

## Stage 0 — LOAD (orchestrator, no code)

Read `references/house-style.md` + `references/m1-decode-evidence.md` IN FULL before any
design. Then classify the op: shape (M, N, K / attention / elementwise), dtype, per-token
work, TP/EP sharding, where it sits in the layer DAG (producer/consumer events), and which
evidence rows (D*/W*/U*) touch it. Match the op to an upstream family (A pipeline / B
consumer+regions / C consumer+monolith / D consumer+no-SMEM / E sub-op helper) in
`references/upstream-kernel-catalog.md` — the family names the file to crib from. If the op
matches a DEAD row and no new mechanism is on offer — STOP and say so; that is a successful
outcome of this stage.

## Stage 1 — SPEC (designer subagent)

**Deliverable: a spec.h-style design doc** (markdown). Location convention: campaign items
that should travel with the repo go to
`.claude/skills/v2-kernel-writing/applications/<item>_spec.md` (exemplar:
`applications/ffn_item1_spec.md`); throwaway/exploratory specs go to `scratch/`
(git-ignored, machine-local). The spec contains:
1. **Engine choice via the DECISION TREE**:
   - **GEMM-shaped & M ≥ tile (or a real K-pipeline: ≥ 4 K-iterations of streamed weight tiles;
     page count per tile is dtype-dependent — an fp8 128×128 tile is exactly 1 page and still
     qualifies)** →
     reference pipeline w/ TMA + tcgen05. Load `b200-tma-pipeline-designer` (stage ring,
     swizzle, load-vs-store completion), `b200-tcgen05-mma-contract-builder` (tile/dtype/
     cta_group, SMEM operand layout, I-desc), `b200-tmem-lifecycle-planner` (TMEM columns,
     alloc/dealloc, ld/wait). Anchor every choice to house-style §2/§5.
   - **M=1 GEMV / memory-bound streaming** → consumer-GEMV per m1-decode-evidence (D2/D3/D9);
     nwarps per the W1 wave-quant tie-test `ceil(items/(ntasks*nwarps))`.
   - **Attention-shaped** → consumer-only per upstream `attention_sm100.cuh` (house-style §0);
     for FA-style rewrites additionally load `b200-flash-attention4-planner`.
   - **Consumer-only ops also pick a SMEM family** (catalog §2-4): planner-region typed
     buffer struct (rmsnorm/argmax pattern — the DEFAULT for anything staging through SMEM);
     ONE monolithic honest region when porting a kernel with hand-rolled internal offsets
     (attention pattern — never declare regions the device won't address); NUM_REGIONS=0 spec
     for pure-GMEM elementwise (silu/embedding pattern). A tiny fused sub-op (norm/rope-style)
     may be a family-E SMEM-view helper inside a host kernel instead of a new task.
   - Unsure how to map the op at all → load `b200-scope-layout-dispatch` first.
2. **SMEM region plan + budget**: named regions, sizes, can_pack, page math (16KB pages, ≤14
   pages, total ≤ 224256 B), alignment 1024 — house-style §4 format.
3. **SEM ordinal table** (house-style §3 format: ordinal, count, producer→consumer, meaning;
   ≤31 op-private) — or the tag-flag alternative (W4) with its flag layout, if multi-role
   handshakes on the consumer-only idiom.
4. **Stage count + role responsibilities** per role, incl. who re-inits which async mbars and
   who releases which pages on which path (bounds-fail included).
5. **Task granularity**: DEFAULT per-tile (tile_idx = task_offset). Grid-wide fused is the
   EXCEPTION — requires written justification + the monotonic-barrier + skip_after_step0 +
   num_tasks==num_workers contract (house-style §6, wiring-recipe §7/§8).
6. **Evidence check**: per design choice, the D*/W*/U* row it rests on; for U* probes, the
   pre-registered predicted Δ + kill threshold.

## Stage 2 — IMPLEMENT (implementer subagent)

**Engine choice first** (routing rule + flow in `references/ferret-v2-dispatch.md`):
`v2-kernel-engineer` = protocol-heavy house-style port / first bring-up (default);
**ferret-v2** (`ferret-kernel-agent` V2 MODE) = beat-a-numeric-TIER-2-target optimization
loop once a faithful FROZEN gate exists (op wired + harness case + live anchor) — also the
"ferret writes from spec" path over a wired stub; **kda** = verdict-grade honest transfer
when the number decides a campaign verdict and over-claim is costly. Ferret-v2 rearranges
the stages (S3-stub wiring precedes the run — the gate substrate is the in-tree harness);
its friction escape falls back to engineer-shell + ferret-math-only-body.

Write `<op>_v2.cuh` + `<op>_v2_spec.h` to the spec. House-style code conventions:
- `namespace kernel { namespace <op>_v2 {`; spec.h constants + static_asserts pinning every
  mirrored constant; role-split `__device__ __noinline__` functions (one per role).
- mbarrier protocol per house-style §2/§3: start-of-task re-init of async-arrived mbars by
  their arriving role; `fence.mbarrier_init.release.cluster` after inits;
  `tcgen05.fence::after_thread_sync` at MMA↔wait boundaries. **Before finalizing, invoke
  `b200-mbarrier-protocol-auditor`** on the barrier ledger (every mbar: init count, arrivers,
  tx-bytes, waiters, phase evolution, re-init ownership).
- `extern __shared__ __align__(1024)`; SMEM only via `task_desc->smem_region_offset(REGION_*)`.
- **No `__syncthreads()` in role loops** — named barriers (`bar.sync <free-id>, 128`) or
  tag-flags only; `elect_sync()` for single-thread issue; no `blockIdx` for identity.
- Layout doubts (swizzle vs tcgen05 operand, coalescing, bank conflicts) →
  `b200-layout-contract-auditor`. Build-flag doubts (sm_100a, -rdc=true) →
  `blackwell-build-compatibility-auditor`.

## Stage 3 — WIRE (orchestrator)

Follow `references/wiring-recipe.md` top to bottom — enum, task_header include, register fn
(**§1.1 dep-prefix is the first line of the consumer body — MANDATORY**), graph.cc tuple,
runtime.cc task_type_to_name + task_offset=bid.x list, py wrapper (num_tasks==num_workers
gate if grid-wide), builder use_v2 branch, skip_after_step0 on any monotonic-barrier scratch.
Tick the §10 ship checklist explicitly.

## Stage 4 — VALIDATE (validator subagent + references/validation-debug.md)

In order, no skipping: (1) test-mode numeric vs torch in `tests/runtime_python/blackwell_v2/`
(cos ≥ 0.999, rel_max ≤ 3e-2, no NaN; v1-counterpart compare); (2) §1.1/protocol static audit;
(3) in-MPK `--layers 0-3` probe; (4) **MULTI-STEP run, iter ≥ 3** — iter-0-fine/iter-1-hang =
persistent-state re-init (skip_after_step0), NOT a missing event; (5) on any hang: watchdog
(`-DMPK_V2_BREADCRUMB` + `MPK_V2_HANG_WATCHDOG_S`); on any crash: compute-sanitizer memcheck
= ground truth (breadcrumb counts are base-rate-biased). Deadlock/wrong-result debugging →
`b200-warp-specialized-debugger` (roles/storage/handoff/lifetime worksheet, one handoff at a
time). Math-changing on TP8 → poison-fill gate, not token-identity.

## Stage 5 — PERF (orchestrator or validator)

TIER hierarchy is the law: **TIER 1 in-MPK %globaltimer slowCTA @ production grid = the only
verdict-grade number**; harness slowCTA corroborates; cudaEvent-wall / standalone-warm are
diagnostic-only. Compare against the reference/v1 body anchor from the spec. Bottleneck
classification → `b200-kernel-roofline-triage` (achievable-floor rules from
m1-decode-evidence §4 apply — same-grid xor-consumer floor, never theoretical peak). For a
pipeline kernel that is correct-but-slow, climb `b200-gemm-optimization-ladder` one rung at a
time. Profiler: buffer = 120000*128 entries; export via `scripts/v2_perfetto_export.py`.
For a sustained beat-a-numeric-target optimization loop on one kernel, dispatch **ferret-v2**
(`references/ferret-v2-dispatch.md`): it iterates the pair against the frozen harness gate
(TIER-2 body_span) autonomously; TIER-1 in-MPK slowCTA stays the final verdict here.
For a whole perf-optimization CAMPAIGN around this kernel (measure→plan→implement→re-measure
→land, agent roster + history contract) use the sibling skill `v2-perf-iteration`.

## Stage 6 — REVIEW (orchestrator)

- EVERY non-trivial conclusion (root-cause, DEAD/ALIVE verdict, perf claim, "matches
  reference") → `ablation-logic-reviewer` subagent + Codex MCP double-check (default params)
  BEFORE acting on or reporting it.
- Landing: `mpk-correctness-gate` for anything math-adjacent, then `mpk-commit-reviewer`
  before `git commit` (staged-path + byte-identity + message gates). Verdicts →
  `mpk-memory-keeper` (experiment_history INDEX + memory; update m1-decode-evidence sources).

## Sub-skill index (load via Skill tool, exact names)

| Sub-skill | Use at | For |
|---|---|---|
| `b200-scope-layout-dispatch` | S1 | op→kernel mapping: scope/layout/dispatch/handoff contract |
| `b200-tma-pipeline-designer` | S1/S2 | TMA descriptors, stage ring, swizzle, completion protocol |
| `b200-tcgen05-mma-contract-builder` | S1/S2 | MMA tile/dtype/descriptor contract |
| `b200-tmem-lifecycle-planner` | S1/S2 | TMEM columns, alloc/ld/wait/dealloc lifecycle |
| `b200-flash-attention4-planner` | S1 (attn rewrites) | QKᵀ/PV + online-softmax tile & barrier graph |
| `b200-mbarrier-protocol-auditor` | S2 gate | per-barrier ledger audit before finalizing |
| `b200-layout-contract-auditor` | S2/S4 | shape-stride/swizzle/operand-contract bugs |
| `blackwell-build-compatibility-auditor` | S2/S3 | sm_100a flags, PTX/cubin, JIT |
| `b200-warp-specialized-debugger` | S4 | deadlock / IMA / wrong-result / correct-but-slow |
| `b200-kernel-roofline-triage` | S5 | bound classification + minimal falsifying experiment |
| `b200-gemm-optimization-ladder` | S5 | staged GEMM perf climb with gates |
