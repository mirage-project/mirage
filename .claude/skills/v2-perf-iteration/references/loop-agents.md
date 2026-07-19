# Loop agents — the roster card

Defs live in `.claude/agents/` (repo — travels with every clone) and `~/.claude/agents/`
(USER-account-level, same-user machines only: `mpk-perf-analyzer`, `ablation-logic-reviewer`,
`codex-task-dispatcher`; missing on a fresh account ⇒ run that role with a general-purpose
subagent given the discipline in its row below). Universal rules: the main thread is the
sole orchestrator (edits, commits, box ops); **no subagent dispatches another subagent**
(nested dispatch fails silently); subagents MAY call the Codex MCP directly (a tool, not an
agent). ⚠️ Several defs still say "150µs/MoE-layer @ TP4 / per_position_grid.py" — v1-era
DSv3 framing (each def now carries a "V2 / new-campaign note" pointing here). Every dispatch
prompt restates the CURRENT campaign goal (DSv3 instance: 8ms e2e tpot @ TP8 EP2 bs=1) and
the v2 quickstart commands (SKILL.md) so the agent doesn't run the stale pipeline.

## The loop roster

**`mpk-profiler`** (sonnet) — loop step 1 (measure). One invocation = the whole measurement
chain: GPU-safety pre-flight + lease → canonical bs=1 decode run with profiling → trace →
per-position slowest-CTA table → cleanup + zombie-guard. Consumes: the config to measure.
Returns: headline latency (n-of-N) + ranked per-position table + trace path + correctness
precondition verdict. Key discipline: measures only, never edits; REFUSES perf on a
suspect-MoE (DECODE_LEAN) config; never crash-loops. **v2 usage:** dispatch for local-GPU
triage or offline parsing of a retrieved `*_v2prof.npy` (give it the prof/exporter commands);
TP8 box runs stay with the main thread per box-orchestration.md.

**`mpk-perf-analyzer`** (Opus) — step 2 (analyze), on a fresh baseline or after a landed
change. Consumes: the latest trace/tables. Returns: ranked TODO split into kernel-level gaps
(>20% slower than the vLLM/SGLang ref) vs system-level bubbles (serialization, dep/page-wait,
overlap). Key discipline: reads `experiment_history/INDEX.md` FIRST (history-aware ranking);
reports, never fixes.

**`mpk-optimization-planner`** (Opus) — step 3 (plan), at cold start or on a re-plan trigger
(bottleneck shifted / lever exhausted / stalled). Consumes: profiler report + builder/kernels
+ full history. Returns: the "True Plan" — ranked multi-lever batch, each lever with derived
µs arithmetic + on-critical-path reasoning + correctness risk + file:line sketch + dispatch
tag, converged with Codex over 3 rounds. Key discipline: anti-loop vs INDEX.md AND personal
memory AND (v2) `m1-decode-evidence.md`; no invented numbers; correctness-first premise;
plans only.

**`mpk-iterator`** (sonnet) — step 4 (next move), every iteration. Consumes: profiler report
+ planner roadmap + INDEX. Returns: the fixed reflection ("state the problem") + 1-5 ranked
single-iteration moves + THE single next move with a falsifiable predicted Δ. Key discipline:
reflect before proposing; anti-loop tag per move; on-critical-path or it's worthless;
one move at a time; proposes only (no Write/Edit).

**`ablation-logic-reviewer`** (Opus) — step 5 (review), MANDATORY for every non-trivial
conclusion/plan/ablation BEFORE acting on it. Consumes: the claim + raw mechanism + data.
Returns: first-principles audit verdict. Key discipline: re-derives independently (limiting-
case/invariant checks — the split-K "SK1 must cost ≈ baseline" pattern), does NOT just
sanity-check the framing; MUST cross-verify via Codex MCP (default params) without telling
Codex the expected verdict; a reviewer-Codex disagreement is itself a finding.

**`mpk-correctness-gate`** (sonnet) — step 7 (validate), before trusting any baseline and
before every math-changing commit. Consumes: the change + touched kernels. Returns: PASS/FAIL
+ the one-line correctness story for the commit-reviewer. Its four checks: test-mode kernel
numeric; routed-MoE NON-NULL (num_active≈4, DECODE_LEAN unset); token-identity (math-neutral)
or numeric check (math-changing); Qwen3 regression smoke. Key discipline: DECODE_LEAN ≠
correctness; num_active=0 is a STOP (never-park-a-bug); for v2 TP8 add the nondeterminism
protocol (canary/poison-fill/coherence — validation-debug.md §7).

**`mpk-commit-reviewer`** (sonnet) — step 8 (land), MANDATORY before every `git commit`.
Consumes: the staged diff + message. Returns: PASS/BLOCK + offending paths/hunks. Gates:
never-stage set (`.claude/` session state — `.claude/skills|agents` are exempt once
GITIGNORE_PATCH.txt is applied, see the agent def; `scratch/`, `scripts/`,
`experiment_history/`, `outputs/`,
backups, generated `test.cu`/`.so`); default build byte-identical (env-gated default-OFF; a
default-flip needs measured justification); allowed surface; message = mechanism + measured
Δ + sign-off; correctness story present. Key discipline: searches project memory for the
recorded discriminator matching the commit's CLAIM (e.g. "dead task" needs a box
token-identity A/B); never edits, never commits.

**`mpk-memory-keeper`** (sonnet) — step 9 (record), end of every experiment. Consumes: the
verdict + raw numbers + trace dir + commit hash. Returns: journal entry appended +
INDEX.md one-liner (WIN|NULL|REGRESS|PARKED + WHY — NULL/REGRESS rows are the anti-loop
payload) + structural lessons folded into personal memory. Key discipline: sole writer of
`experiment_history/`; append-only; corrections must name the prior claim; verifies numbers
against the report, never records a paraphrase.

**`codex-task-dispatcher`** — [CODEX]-tagged levers: a scoped, self-contained investigation
or experiment handed to `codex exec` with a prepared context brief. Use for read-heavy
audits and parallel experiments the main thread shouldn't context-switch into.

## Kernel-perf engines (step 6, kernel-body levers)

| Engine | One-line routing |
|---|---|
| `v2-kernel-engineer` (via the `v2-kernel-writing` skill) | House-style v2 protocol work: new op, v1→v2 port, warp-role pipeline rewrite — anything that must live in `tasks/blackwell_v2/` and obey the §1.1/mbarrier/skip_after_step0 invariants. Dispatch the SKILL (Stage 1 spec first), not the agent bare. |
| `ferret-kernel-agent` | Autonomous beat-the-target loop with a FROZEN hash-locked gate (independent test-writer → optimizer rounds → Codex integrity review). Use for fire-and-forget standalone-kernel exploration where breadth matters and over-claim is tolerable; gate must be COLD-L2. |
| `kda-kernel-agent` | Verdict-grade honest transfer: KDA prompt-driven workspace validated against the FAITHFUL in-MPK per-task gate (slowCTA @ production grid, `mpk-faithful-gate` skill). Use when the number will be ACTED ON and in-MPK transfer honesty beats breadth. |

Common to all three: the M=1 decode shape (never compile-M), the shared-worker megakernel
context, and a check against `m1-decode-evidence.md` BEFORE dispatch (don't send an engine
at a DEAD lever). Winning kernels re-enter the loop at step 7 — the TIER-1 in-MPK TP8
re-measure decides, not the engine's own bench.

## Dispatch-tag → executor map

| Tag | Executor | Typical lever |
|---|---|---|
| [MAIN] | main thread edit | builder/plan/scheduling/env-gating/buffer aliasing |
| [ENGINEER] | `v2-kernel-writing` skill → `v2-kernel-engineer` | v2 task kernel authoring/port/rewrite |
| [FERRET] / [KDA] | `ferret-kernel-agent` / `kda-kernel-agent` | standalone kernel-body beats-SOTA rewrite |
| [CODEX] | `codex-task-dispatcher` | scoped investigation / experiment |
