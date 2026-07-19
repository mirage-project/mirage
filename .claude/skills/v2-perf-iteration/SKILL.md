---
name: v2-perf-iteration
description: Runtime-V2 performance-iteration workflow. Use when running a perf-optimization campaign or iteration on the v2 runtime (--use-v2) — measuring a baseline, ranking bottlenecks, planning levers, implementing, re-measuring, and landing/recording the verdict. Drives the loop MEASURE→ANALYZE→PLAN→NEXT-MOVE→REVIEW→IMPLEMENT→VALIDATE+RE-MEASURE→LOOP-OR-LAND→RECORD with the mpk-* subagent roster, the v2 profiler/perfetto toolchain, and the TIER-1 TP8 verdict discipline.
---

# V2 Perf Iteration — the measurement-driven optimization loop

This is the perf-optimization loop of the v1 multi-agent campaign that ran for months
(the old repo-root `WORKFLOW.md` is now a superseded stub pointing here), upgraded for
Runtime-V2's measurement reality. Siblings: **`v2-model-support`**
(bring-up; its Phase (d) is this skill), **`v2-kernel-writing`** (the per-KERNEL inner loop
this skill dispatches into when a lever is kernel-body work). The main thread (or one lead
orchestrator) runs the loop and does all edits/commits/box-ops; subagents measure-parse /
analyze / plan / review / record. **No subagent dispatches another subagent**, and **box
operations never go inside a subagent** (`v2-model-support/references/box-orchestration.md`).

**Goal + verdict metric — pin ONE per campaign, then do not drift.** The loop is
metric-agnostic; what is non-negotiable is that a single PRODUCTION verdict config is
declared up front and every lever's verdict-grade Δ is measured there. **Worked example —
the concluded 2026-06/07 DSv3 campaign's instance:** e2e decode tpot, **bs=1, TP8 EP2, MTP off**, toward
**8 ms/token** (SGLang 7.99 on the same box proves it reachable); v2 clean baseline
2026-07-07: 12.069 ms/tok vs v1 ~9.787; TP<8/local runs are triage only. **A new campaign
(e.g. Qwen3-8B single-GPU bs=1024 throughput) writes its own goal line in this exact shape**
— metric, config, target, why-reachable — and restates it in EVERY dispatch prompt.
Per-task numbers follow the TIER hierarchy
(`v2-kernel-writing/references/validation-debug.md` §8): **TIER-1 = in-MPK
%globaltimer / per-position slowest-CTA at the production grid — the ONLY verdict tier**;
faithful harness corroborates (TIER-2); cudaEvent-wall / standalone-warm are diagnostic only.

⚠️ The `.claude/agents/mpk-*` defs still carry v1-era DSv3 framing (150µs/MoE-layer, TP4,
`per_position_grid.py`, `scratch/` helper scripts that are git-ignored/machine-local).
Every dispatch prompt MUST restate the CURRENT campaign goal, the v2 toolchain commands
from the quickstart below, and the artifact paths — otherwise the agent drifts to the
stale v1 pipeline (each def now carries a "V2 / new-campaign note" saying exactly this).

## Environment prerequisites (what must exist on the machine)

In-repo (travels with every clone): the parser `python -m mirage.mpk.prof`
(`python/mirage/mpk/prof.py`, tracked), the demo `--profiling` plumbing, this skill's
`references/` + `tools/`. Machine-local / degradations:

- **`scripts/v2_perfetto_export.py` / `perfetto_analyze.py` / `perfetto_depgraph.py` are
  UNTRACKED on this branch** — a fresh clone will not have them under `scripts/`. Archived
  copies travel in `.claude/skills/v2-perf-iteration/tools/` — run them from there (or copy
  back to `scripts/`, which stays git-ignored for local files). `mirage.mpk.prof
  summary/check/pagewait` is the tracked no-dependency fallback for text-table analysis.
- **The TP8 box** — only for multi-GPU verdict configs (`v2-model-support/references/
  box-orchestration.md`; §1-2 there are site-specific). Single-GPU campaigns run the whole
  loop locally.
- **`experiment_history/`** — git-ignored ⇒ empty on a fresh clone; create INDEX.md + the
  journal at step 9 of the first iteration. The kernel-lever anti-loop that must survive
  clones is `v2-kernel-writing/references/m1-decode-evidence.md` (in-repo).
- **User-level agents** (`~/.claude/agents/`: `mpk-perf-analyzer`, `ablation-logic-reviewer`,
  `codex-task-dispatcher`) — same-account only; the rest of the roster is in-repo at
  `.claude/agents/`. **Codex MCP** — machine-configured (`.mcp.json` is git-ignored); absent
  ⇒ reviews degrade to subagent-only (state it). **Personal memory** — optional context.
- **`~/ref_vllm_sglang.md`** (analyzer/planner per-kernel reference table) — machine-local;
  absent ⇒ rank gaps against the SGLang/vLLM numbers recorded in the campaign goal line and
  in-repo docs, and say the external table was unavailable.

## The loop

```
(1) MEASURE ─▶ (2) ANALYZE ─▶ (3) PLAN ─▶ (4) NEXT-MOVE ─▶ (5) REVIEW-BEFORE-ACT
                                  ▲                                   │
                                  │ (new bottleneck / re-plan)        ▼
(9) RECORD ◀─ (8) LOOP-OR-LAND ◀─ (7) VALIDATE + RE-MEASURE ◀─ (6) IMPLEMENT
```

| # | Step | Who | In → Out |
|---|---|---|---|
| 1 | MEASURE | main thread (box) + `mpk-profiler` discipline | profiled `--use-v2` run → per-task-type consumer-body table + per-position slowCTA + tpot |
| 2 | ANALYZE | `mpk-perf-analyzer` (Opus) | trace/tables → ranked gaps vs refs, kernel-level vs system-level split |
| 3 | PLAN | `mpk-optimization-planner` (Opus) | report + history → µs-derived ranked batch plan, [MAIN\|ENGINEER\|FERRET\|CODEX] tags, 3-round Codex convergence |
| 4 | NEXT-MOVE | `mpk-iterator` | plan + report → reflection + the single next move w/ falsifiable predicted Δ |
| 5 | REVIEW | `ablation-logic-reviewer` + Codex MCP | the move/conclusion → first-principles audit (MANDATORY before acting) |
| 6 | IMPLEMENT | main thread (route by tag) | env-gated default-OFF change |
| 7 | VALIDATE | `mpk-correctness-gate`, then re-run (1) | PASS/FAIL + the TIER-1 TP8 Δ |
| 8 | LOOP-OR-LAND | main thread + `mpk-commit-reviewer` | commit (WIN) / revert+INDEX (NULL/REGRESS) / re-plan |
| 9 | RECORD | `mpk-memory-keeper` | journal + INDEX row + personal-memory lesson |

Full roster card (what each agent consumes/returns + key discipline): `references/loop-agents.md`.

**1. MEASURE.** The `mpk-profiler` pattern updated for v2: GPU-safety pre-flight → the
canonical config → profiled run → parse → cleanup+zombie-guard → report. At TP8 the box
session belongs to the MAIN THREAD (setup/poll split, retries, verify-STOPPED — follow
`v2-model-support/references/box-orchestration.md`; do NOT re-derive box mechanics here):
rsync → `--use-v2 --profiling` run (quickstart below) → retrieve the per-rank
`*_v2prof.npy` → `v2_perfetto_export.py` + `python -m mirage.mpk.prof summary/check`.
Dispatch `mpk-profiler` itself only for local-GPU triage runs or offline parsing of an
already-retrieved buffer — never for box ops. Report = tpot (n-of-N) + the per-task-type
consumer-body table (µs/instance × count × layers = ms and % of tpot) + per-position slowCTA
+ correctness precondition (coherent output, routed-MoE non-null).

**2. ANALYZE** (optional on small iterations, mandatory on a fresh baseline). Ranked TODO
split kernel-level (body ≫ SOTA ref at M=1 shape) vs system-level (dep-wait, page-wait,
role-coordination overhead, AR/skew — the v2 runtime-overhead axis that made v2 12.07 vs v1
9.79). It reads `experiment_history/INDEX.md` first.

**3. PLAN.** µs-derived ranked batch plan; every lever: target position + arithmetic +
on-critical-path reasoning + correctness risk + dispatch tag. Anti-loop is MANDATORY: check
`experiment_history/INDEX.md` AND `v2-kernel-writing/references/m1-decode-evidence.md`
(the DEAD/WIN/UNTESTED map) — a dead lever is only re-proposable by naming what's different.

**4. NEXT-MOVE.** One concrete single-iteration move with a falsifiable predicted Δ
("tpot 12.07 → ~11.5 because attn consumer body 108 → ~99µs and attn is on the CP").

**5. REVIEW-BEFORE-ACT (MANDATORY, user-locked).** Every non-trivial conclusion — root-cause,
ablation verdict, dead/alive, ceiling, perf claim — goes through `ablation-logic-reviewer`
(first-principles re-derivation) AND a Codex MCP cross-check (`mcp__codex__codex`, DEFAULT
params) BEFORE you act on it or report it settled. When stuck: detailed multi-turn Codex
discussion BEFORE escalating to the user (escalate only when both agree there's no room).

**6. IMPLEMENT** (main thread routes by tag):
- **[ENGINEER]** kernel-body / new-op / port work → the **`v2-kernel-writing`** skill
  (SPEC→IMPLEMENT→WIRE→VALIDATE→PERF→REVIEW; its Stage 2 dispatches `v2-kernel-engineer`,
  and `ferret-kernel-agent`/`kda-kernel-agent` are its beat-a-target engines).
- **[FERRET]/[KDA]** standalone beat-the-SOTA kernel rewrite → `ferret-kernel-agent`
  (frozen-gate autonomous loop) or `kda-kernel-agent` (verdict-grade honest transfer);
  routing one-liners in `references/loop-agents.md`.
- **[CODEX]** scoped investigation/experiment → `codex-task-dispatcher`.
- **[MAIN]** builder / plan / scheduling / system change → main thread edits directly.
Every lever lands env-gated default-OFF; default build byte-identical.

**7. VALIDATE + RE-MEASURE.** Math-changing → `mpk-correctness-gate` (test-mode + non-null
MoE + the TP8-nondeterminism-aware gates: deterministic canary, poison-fill,
coherence-in-envelope — see validation-debug.md §7). Math-neutral → token-identity on a
deterministic config. Then re-run step (1); the verdict is the TIER-1 TP8 number, and the
predicted Δ is confirmed or refuted — say which.

**8. LOOP-OR-LAND decision rules:**
- **WIN (predicted Δ held at TP8)** → land: `mpk-commit-reviewer` gate (staged-path,
  default-OFF byte-identity, message mechanism+Δ+sign-off) then commit. BLOCK → fix, re-gate.
- **NULL/REGRESS** → revert the lever (or leave default-OFF-dead), INDEX row WITH the why —
  recording the death is the deliverable, not a failure.
- **Bottleneck shifted / lever class exhausted / stalled** → back to (3) re-plan. No
  stall-stop: a stall means the next idea isn't found yet (planner researches refs).
- **STOP** only at goal, user halt, or hardware down.

**9. RECORD.** `mpk-memory-keeper` appends the journal entry + INDEX one-liner (esp.
NULL/REGRESS) + folds structural lessons into personal memory. This closes the anti-loop:
steps (2)-(4) read what (9) wrote.

## Invariants (every iteration — these encode the documented failures)

- **bs=1 ALWAYS; MTP off.** Batching/MTP-amortization = goal-drift, not a lever.
- **TP8 = the verdict tier.** Local/TP<8 = triage proxy only; TIER-1 in-MPK slowCTA per
  position, NEVER P50, NEVER per-kernel-type aggregate, NEVER cudaEvent-wall promotion.
- **4-role-track averaging trap:** a v2 task's body = its CONSUMER-group slice; loader/
  launcher/storer tracks are mostly waits — averaging across role tracks produces garbage.
- **Correctness-first.** No perf number on an unverified forward pass (routed-MoE non-null,
  num_active≈4; DECODE_LEAN ≠ correctness; coherent decode output for v2 e2e). A correctness
  bug is root-caused before any perf judgment (never-park-a-bug).
- **Default build byte-identical.** Every lever env-gated default-OFF; a default-flip needs
  measured justification + commit-reviewer sign-off.
- **Every non-trivial conclusion → ablation-logic-reviewer + Codex double-check** before it
  is acted on (the over-claim guard; this project's flip-flop history is why).
- **GPU-safety non-negotiable.** Never crash-loop the megakernel (D-state zombies);
  test-mode first; box verify-STOPPED; one hung run → clean up + stop, don't retry blind.
- **experiment_history closes the loop.** No experiment ends without its INDEX row;
  NULL/REGRESS rows are the most valuable.
- **Falsifiable moves.** Each move pre-states its predicted Δ and CP-membership; off-CP
  wins ≈ 0 e2e.

## v2 measurement quickstart (the exact commands)

**Profiled run** (add to the canonical demo invocation, per-rank under mpirun):

```
demo/deepseek_v3/demo.py ... --use-v2 --profiling --trace-name <tag> \
    [--profile-start-step N]     # profile steady-state, not warmup
```

- `--profiling` compiles with `-DMPK_ENABLE_PROFILING` (persistent_kernel.py:504).
- **Buffer contract:** v2 needs `V2_PROF_BUF_ENTRIES = 120000*128` (15.36M) uint64 entries
  — demo.py auto-sizes this (`V2_PROFILER_BUFFER_ENTRIES`, demo.py:30) and HARD-RAISES if
  `MPK_PROFILER_BUFFER_ENTRIES` is set smaller (demo.py:64-75): a smaller buffer = silent
  device OOB (the v2 profiler writes tail accumulators at absolute end-of-buffer indices).
  Don't override it; don't "fix" a >256-worker abort by shrinking the buffer.
- Only the LAST `V2_PROF_WINDOW_ITERS = 25` decode steps are recorded; 8 tracks/SM
  (consumer/loader/launcher/storer/controller + 3 phase tracks: dep-wait/page-wait/>2µs).
- Artifacts per rank: `<tag>_rank<r>_v2prof.npy` (raw buffer) + `<tag>_rank<r>.perfetto-trace`
  (v1 exporter output — **garbage for v2 buffers, ignore it**) + an auto text summary at
  run end (`prof.print_run_summary`).

**Parse** (usually rank0's npy, retrieved from the box):

```
python -m mirage.mpk.prof check    <npy>   # structural gate: needs "ALL CHECKS PASS",
                                           # dropped events MUST be 0 (else trace truncated)
python -m mirage.mpk.prof summary  <npy>   # per-task-type consumer table: n/SM/it, dep-wait,
                                           # suffix, body+disp, win-mean/p50 + busy ms/SM/step
python -m mirage.mpk.prof pagewait <npy>   # page-protocol serialization (dead prefetch)
python scripts/v2_perfetto_export.py <npy> <out.json> --last-steps 2 [--sm N]
                                           # Chrome-JSON for ui.perfetto.dev; NEVER --full
                                           # (full window OOMs the UI); deeper analysis:
                                           # scripts/perfetto_analyze.py / perfetto_depgraph.py
                                           # (fresh clone: these are untracked — run the
                                           #  archived copies in this skill's tools/ dir)
```

**What number to quote:** per-invocation task latency = ONE consumer slice; per-task-type
body = the consumer-group windows (`summary`'s table / the consumer track in perfetto).
Never average the role tracks. Headline = e2e tpot + the per-task-type decomposition.
**Worked example (2026-07-09 profile):** attn 108µs × 61 layers ≈ 58% of tpot; ffn
(52+14)µs × 58 MoE layers ≈ 30%; AR ≈ 6% → the attn consumer body is the dominant axis,
AR is not — that ranking IS the plan input.

**Hang/crash during a profiled run:** the historical profiled-only wedges were the v2 runtime
races, ALL FIXED 2026-07-16 (`689dadc5`/`7d271a01`/`025029a1`/`7b6ae2bb`; former wedge windows
pass post-fix — see `v2-kernel-writing/references/validation-debug.md` §5.1), so profiled
measurement is first-class again; a hang on a ≥`7b6ae2bb` tree is a NEW bug. Watchdog
`-DMPK_V2_BREADCRUMB` + `MPK_V2_HANG_WATCHDOG_S=<s>` names the hung task; crash →
compute-sanitizer memcheck is ground truth (breadcrumb in-flight counts are base-rate
artifacts). Full triage table: validation-debug.md §5. Remember: instrumentation changes
tpot (breadcrumb cost ~5.3ms on full-61L) — never quote an instrumented run as the baseline.

## References

| Doc | Content |
|---|---|
| `references/loop-agents.md` | Roster card: every loop agent + the kernel-perf engines + routing |
| `tools/` | Archived copies of the untracked v2 perfetto toolchain (`v2_perfetto_export.py`, `perfetto_analyze.py`, `perfetto_depgraph.py`) — the clone-safe way to run them |
| `../v2-kernel-writing/references/validation-debug.md` | TIER hierarchy §8, profiler contract §9, hang/crash triage §5 |
| `../v2-kernel-writing/references/m1-decode-evidence.md` | The DEAD/WIN/UNTESTED anti-loop map for kernel levers |
| `../v2-model-support/references/box-orchestration.md` | Box session playbook (TP8 runs live here) |
| `experiment_history/README.md` + `INDEX.md` | The durable log contract + the anti-loop source |
