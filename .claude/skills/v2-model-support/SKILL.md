---
name: v2-model-support
description: End-to-end pipeline for adding or porting a model to MPK Runtime-V2 — from a compute-graph spec (shapes + draw.io graph + HF checkpoint + TP/EP plan) to a working multi-GPU demo. Use when bringing up a NEW model on the v2 (role-split, static-plan) runtime, when porting an existing v1 model to --use-v2, or when handed a compute-graph file and asked to make it run. Covers graph→plan, builder/demo bring-up, per-kernel authoring dispatch, the debug gate ladder, and the multi-agent/box workflow.
---

# V2 Model Support — compute graph → working Runtime-V2 demo

This is the GENERIC pipeline for putting ANY model on MPK Runtime-V2. It was distilled from
the campaign that took DeepSeek-V3 decode from a v1-only model to a full-61-layer Runtime-V2
e2e run at TP8 EP2 bs=1 (commit `e31b34dd`, opt-in `--use-v2`, default build byte-identical).
DSv3 examples below are clearly labeled worked-example material — the recipe does not depend
on them; the staged plan that drove that campaign is archived at
`references/V2_DSV3_DECODE_MASTER_PLAN.md` (M0→M5 ladder). A SECOND, smaller worked example
— **Qwen3-8B on v2, dense, single-GPU-capable** — is in §"Worked example #2" below (it is
the closest starting point for a dense-model campaign).
It is a context+phased-recipe skill: architectures vary, the PHASES and GATES do not.

> **Read `mpk-development-norms` FIRST.** This skill is the HOW (graph→plan→demo); that one is the WHERE + the PR-shape gate that decides what lands cleanly on `mpk` when the campaign is done — model code in `models/<model>/builder.py` + `demo/<model>/`, only GENERIC ops (never `<model>_*`) in shared `persistent_kernel.py`, no experiment env-vars in landed code, runtime fixes as separate PRs. During exploration keep levers env-gated default-OFF (`mpk-lever-cleanup`); before opening the PR, conform to the norms.

## Environment prerequisites (what must exist on the machine)

In-repo (travels with every clone): this skill's `references/`, the v2 runtime + kernels
(`include/mirage/persistent_kernel/`, `tasks/blackwell_v2/`), the harness
(`tests/runtime_python/blackwell_v2/`), both worked-example demos (`demo/deepseek_v3/`,
`demo/qwen3/`), the repo agents (`.claude/agents/*.md`), and the sibling skills. Machine-local
(keep working without them, as noted):

- **Remote multi-GPU box** — only for multi-rank gates + verdict configs. The box CLI
  (`~/nebius_box.sh`) and the machine inventory in `references/box-orchestration.md` §1-2
  are SITE-SPECIFIC (placeholders in-repo; actual IPs/users/keys/paths stay in
  operator-local notes — never committed); the structural rules there (§3-8: rsync/build,
  session discipline, safety, testing tiers) transfer to any box. No box ⇒ single-GPU phases still run end-to-end
  (Qwen3-class models need no box at TP1).
- **Local GPU(s)** — needed from the first harness gate onward (graph-build/test-mode gates
  are 0-GPU). Torch-probe cards before use.
- **Model checkpoints** — site-specific paths; the input contract below is what matters.
- **User-level agents** (`~/.claude/agents/`: `mpk-perf-analyzer`, `ablation-logic-reviewer`,
  `codex-task-dispatcher`) — present only on the same user account. The repo-level roster
  (`.claude/agents/mpk-*`, `v2-*`, `ferret-*`) travels with the clone. If
  `ablation-logic-reviewer` is missing, run the review discipline with a general-purpose
  subagent given its first-principles brief.
- **Codex MCP** (`mcp__codex__codex`) — the cross-check second engine; `.mcp.json` is
  machine-local (git-ignored). If unconfigured, reviews degrade to subagent-only — say so.
- **Personal memory** (`~/.claude/projects/-home-muhengl-mirage/memory/`) — optional context,
  same-account only. The load-bearing lessons are already distilled into this suite's docs.
- **`experiment_history/`** — git-ignored, so a FRESH CLONE STARTS EMPTY. That is expected:
  create `INDEX.md` + the journal on first use (contract in Phase d); the anti-loop evidence
  that must survive clones lives in `v2-kernel-writing/references/m1-decode-evidence.md`.

## Input contract (what you need before starting)

1. **Model config** — hidden size, layer count/types (dense vs MoE, attention variant),
   head geometry, vocab, dtype/quantization (FP8 block-scale? BF16?), norm eps.
2. **HF checkpoint** — safetensors; know which weights need conversion/absorption
   (e.g. MLA absorbed q_b_proj) and which need requantization.
3. **Parallelism plan** — TP degree, EP degree (routed_tp_size = world/EP), which
   linears are Column vs Row parallel, where the AllReduces land, vocab-parallel or
   replicated lm_head.
4. **The compute graph** — a draw.io file (ops as nodes labeled name+shape+dtype,
   edges = tensor deps, TP-sharding annotations). Format + parsing:
   `references/graph-to-plan.md`. If none is supplied, derive the graph from the HF
   `modeling_*.py` and WRITE the plan doc as if you had one — the plan doc is the
   contract, the drawio is just its serialization.
5. **Target machine(s)** — which box runs the multi-GPU verdict configs
   (`references/box-orchestration.md`). Single-GPU/local for micro-gates only.

## Why phased (the one-sentence history)

DSv3-on-v2 succeeded because it NEVER registered a live-path task without a complete
v2 consumer body (a bodyless consumer silently deadlocks the whole box — see the §1.1
trap below), validated each op bit-exact in isolation BEFORE it entered the graph, and
went e2e at the smallest possible slice (1 MoE layer) before scaling. Every shortcut
attempted around this ladder cost days (9 debug rounds on a missing AllReduce; an
iter-1 hang from re-zeroing a monotonic barrier). Do not reorder the phases.

## The pipeline

```
Phase 0  GRAPH→PLAN      drawio → op inventory → classify {reuse|new-kernel|fused-later}
Phase a  DEMO            builder-first: weights/SHARD_RULES/cache-key/lifetimes; chain of
                         existing v2 tasks; graph-build + test-mode gates (0-GPU first)
Phase b  KERNEL          per-op loop for missing/slow ops; M0→M5 staged bring-up
Phase c  DEBUG           the gate ladder (token-match first, TP-collective blind spot,
                         nondeterminism protocol, hang triage)
Phase d  WORKFLOW        the multi-agent perf loop + box orchestration + history contract
```

## Phase 0 — GRAPH→PLAN (read `references/graph-to-plan.md`)

1. Parse the draw.io XML → op list. Each op row: `name | op kind | input/output shapes
   AT THE CHOSEN TP/EP | dtype(s) | weight source key(s) | conversion/absorption needs
   | collective (none / AR / reduce-scatter / EP-dispatch) | grid intuition`.
2. Derive PER-RANK shapes: apply the sharding plan (Column→shard N, Row→shard K +
   AllReduce after, EP→local expert slice). The DSv3 worked table is in the reference.
3. Classify EVERY op:
   - **REUSE** — an existing v2 task covers it (rmsnorm_v2, silu_mul_v2, linear v2/v3
     with its `M<=16` contract, embedding_v2, argmax_partial/reduce_v2,
     nvshmem_tile_allreduce_v2(+residual), tensor_init_v2, mul_sum_add_v2 …).
     Check `runtime_header.h` enums 242/243, 326-355 and `tasks/blackwell_v2/`.
   - **NEW KERNEL** — no v2 variant exists → Phase-b item. Tag its port kind:
     leaf (role-split trivial) / collective (v2-safe sync rewrite — HARD) /
     megakernel-shape Form-2 (num_tasks==num_workers, in-op GMEM barrier).
   - **FUSED-BLOCK CANDIDATE** — defer; fusion comes only after chain correctness.
4. Output = the model plan doc: the op table, the v2-ABSENT set, the milestone ladder,
   the risk ranking. Location: `.claude/skills/v2-model-support/references/
   V2_<MODEL>_MASTER_PLAN.md` if it should travel with the repo, `scratch/` (git-ignored)
   for throwaway drafts. Mirror the archived DSv3 instance
   (`references/V2_DSV3_DECODE_MASTER_PLAN.md`). This doc is what the phase leads execute
   against.

## Phase (a) — DEMO: builder-first bring-up (read `references/demo-stage.md`)

Goal: `demo/<model>/demo.py` + `python/mirage/mpk/models/<model>/builder.py` that
BUILD the graph (no GPU needed yet) and pass test-mode with existing tasks.

- **CHAIN-FIRST rule**: assemble every layer from existing generic v2 tasks even if
  slow. Fused megakernels are Phase-b/perf work. A correct chain is your ground truth
  for every later diff (`FUSED_KERNEL_DEBUG_METHODOLOGY.md` step 2 depends on it).
- Weight mapping: SHARD_RULES live in TWO places — the builder AND the demo's
  conversion pass. Update both, always.
- Cache-key contract: the weight cache key hashes config, NOT conversion code. Bump
  the format-version string on ANY conversion-logic change (silent stale weights is
  the worst failure). Details + `MPK_CONVERT_SEMAPHORE` / `MPK_BUILD_CACHE_ONLY` in
  the reference.
- v2 wiring is 90% automatic: pass `use_v2_runtime=args.use_v2` into
  `PersistentKernel(...)`; `compile()` itself runs the v2 queue plan + SMEM plan +
  the §1.1 deadlock guard (persistent_kernel.py ~:5619-5640). Builder-side work is
  selecting v2 task names (most `*_layer` wrappers self-switch on
  `self.use_v2_runtime`) and the v2-only allocations (scratch sizing, scale packs).
- **Gates before ANY GPU run**: (1) graph-build succeeds for the smallest slice;
  (2) the §1.1 guard passes — `v2_unsafe_task_types` empty (every graph-used task
  type has a v2 role variant, else `compile()` raises instead of wedging the box);
  (3) test-mode (0-GPU-graph-build + single-pass CPU-launchable subset) green.

## Phase (b) — KERNEL: per-op loop on the M0→M5 ladder

For each **NEW KERNEL** op, dispatch the sibling skill **`v2-kernel-writing`**
(`.claude/skills/v2-kernel-writing/` — the per-kernel inner loop this pipeline plugs
into: SPEC→IMPLEMENT→WIRE→VALIDATE→PERF→REVIEW) with the op's spec row from Phase 0
(roles / SMEM regions / sync / correctness reference / validate step — the `§3`
template in the DSv3 master plan). For pure kernel-PERF rewrites of an op that already
passes correctness, `ferret-kernel-system`/`kda-kernel-agent` + `mpk-faithful-gate`
are the measurement-honest routes. Every new `_v2` task touches: `runtime_header.h` enum +
`task_register.cc` `register_*_v2_task` (consumer body MUST begin with
`emit_dep_wait_consumer_prefix`) + `graph.cc` dispatch + `runtime.cc`
`task_type_to_name` (+ the `task_offset = bid.x` block for fused megas) + the
`.cuh`/`_spec.h` pair in `tasks/blackwell_v2/` + the `persistent_kernel.py` wrapper's
`"..._v2" if self.use_v2_runtime` switch + the builder call site.

Stage the bring-up on the PROVEN ladder (mirror `references/V2_DSV3_DECODE_MASTER_PLAN.md`
— DSv3 worked example; a dense single-GPU model collapses M1 to "none" and M4/M5 shrink):

- **M0 — leaves + tail wiring.** Trivial role-split tasks (tensor_init-class), tail
  re-routes (lm_head/argmax path), confirm already-present v2 leaves are reachable
  from THIS model's builder. Validate in the `tests/runtime_python/blackwell_v2/`
  harness (per-op, deterministically-seeded, vs fp32 torch ref AND vs the v1 twin).
- **M1 — collectives at TP2 first.** The AllReduce-class ports are the highest-risk
  items (block-wide `__syncthreads()`/256-thread bodies vs the 128-thread consumer
  role → deadlock/half-compute if pasted). Validate on a TP2 micrograph (2 ranks,
  known vectors, bit-exact sum on both ranks), then TP8. Do this EARLY — it de-risks
  everything downstream and is the first multi-rank v2 proof.
- **M2 — fused blocks.** Megakernel-shape (Form-2) ports: check first whether an
  existing v2 mega can be REUSED via a builder re-route (DSv3's FFN was — kernel
  already proven, the work was builder-side tensor packing). Each fused mega:
  bit-match vs its v1 twin in a TP-shaped harness, THEN a small-slice in-MPK smoke
  (the `__align__(1024)` extern-smem footgun is only caught in-MPK).
- **M3 — FIRST E2E** at the smallest real slice (DSv3: `--use-v2 --layers 3-3`
  TP8 EP2 bs=1, `--disable-vocab-parallel-lm-head` to stay on present tail tasks).
  TWO hard pre-conditions: (1) reachability diff — build the sliced graph in
  test-mode and diff the task list vs a full build so no head/tail seed task is
  silently dropped; (2) §1.1 guard green. Correctness = the Phase-c protocol.
- **M4 — scale** layers up (+ restore any deferred tail variant). Expect the
  cold-convert OOM class here at full TP — `MPK_CONVERT_SEMAPHORE=K`.
- **M5 — remaining layer types** (DSv3: dense layers 0-2 via one Form-2 task) →
  full-model e2e. Deliverable: full-layer coherent decode on v2 + tpot vs v1.

Per-kernel gate: test-mode numeric PASS (cos ≥ 0.999, rel_max ≤ 3e-2, no NaN, and
bit-exact-vs-v1 for elementwise ops) BEFORE the task enters any e2e graph.

## Phase (c) — DEBUG (read `references/debug-gates.md` BEFORE debugging anything)

The distilled ladder — full checklists in the reference:
1. Full-layer TOKEN-MATCH first; NEVER judge correctness from few-layer coherence.
2. Broken → diff vs the CHAIN stage-by-stage (clean token position, FULL vectors).
3. Garbage at TP>1 + gate/TP1 fine ⇒ suspect a MISSING CROSS-RANK COLLECTIVE first —
   a single-rank gate is structurally blind to it (6 gate-fidelity classes).
4. Token-identity only on DETERMINISTIC configs. The TP8 FFN atomicAdd path is
   nondeterministic → use the 3-part gate: deterministic canary + NaN poison-fill +
   full-model coherence-in-envelope (with an OFF1-vs-OFF2 control).
5. iter-0-fine / iter-1-hang ⇒ a PERSISTENT state got re-initialized (monotonic
   barrier + `skip_after_step0` class), not a missing event.
6. Hangs: watchdog (names the hung task) > breadcrumb (crash-only; in-flight counts
   are base-rate artifacts). Illegal address: compute-sanitizer is ground truth.
7. "Dead task / safe to remove" claims: box token-identity A/B is the ONLY ground
   truth — static analysis + reviewers have been wrong.
Run the `mpk-correctness-gate` agent before trusting any baseline and before every
math-changing commit.

## Phase (d) — WORKFLOW orchestration (perf loop, after correctness)

**The full v2-updated loop is the sibling skill `v2-perf-iteration`
(`.claude/skills/v2-perf-iteration/`) — load it to run this phase; the summary below is
orientation only.**

The multi-agent loop, unchanged: profiler → (analyzer) → planner →
iterator → [ablation-logic-reviewer] → implement → correctness-gate → profiler →
commit-reviewer → commit → memory-keeper → decide. Standing disciplines:
- EVERY non-trivial conclusion through `ablation-logic-reviewer` + a Codex MCP
  cross-check before acting on it (the over-claim guard; defaults params only).
- Every lever lands env-gated default-OFF; DEFAULT BUILD BYTE-IDENTICAL (the whole
  v2 wiring itself followed this — `--use-v2` opt-in).
- Verdict metric at the PRODUCTION config (DSv3: bs=1 TP8 e2e tpot); smaller TP is
  triage only. Slowest-CTA per-position, never P50/per-kernel aggregates.
- `experiment_history/` contract: journal + INDEX row after every experiment,
  ESPECIALLY NULL/REGRESS (anti-loop) — via `mpk-memory-keeper`.
- GPU-safety: never crash-loop the megakernel (D-state zombies), memory-cap every
  launch, box sessions per `references/box-orchestration.md`.

## Worked example #2 — Qwen3-8B on v2 (in-tree + upstream; the dense single-GPU shape)

A COMPLETE second instance of this pipeline's endpoint already exists for a dense model, and
it is the natural starting point for any dense/single-GPU v2 campaign (e.g. Qwen3-8B
throughput work):

- **In-tree (this branch)**: `demo/qwen3/demo.py` has `--use-v2` (argparse ~:127;
  `use_v2_runtime=args.use_v2` into `PersistentKernel` ~:352). The graph is built INLINE in
  the demo (the `python/mirage/mpk/models/qwen3/builder.py` GraphBuilder has NO v2 branches —
  a wiring-style difference vs DSv3's builder-side gating). The v2 branches swap exactly the
  GEMM-shaped ops to the Channel-based per-tile linear family, `tiles_per_task=1`:
  qkv_proj + gate_up → `linear_layer_v3` (~:544, ~:711), o_proj + down_proj →
  `linear_with_residual_layer_v3` (~:637, ~:744), lm_head → `linear_layer_v3` (~:795)
  (`TASK_LINEAR_SM100_V3` = 244 / `_WITH_RESIDUAL_` = 245). Everything else keeps its task
  name and runs as the v2 role variant: rmsnorm (`TASK_RMS_NORM_HOPPER_V2` 326), paged
  attention (`TASK_ATTN_SM100_V2` 329, consumer-only), silu_mul, embedding, argmax
  partial/reduce. Task-plan wiring is EXPLICIT at demo level (~:850-854):
  `task_graph["v2_worker_task_queues"] = build_v2_worker_task_queues(...)` +
  `add_v2_region_smem_plan(...)` before `mpk.compile()` — the older of the two wiring
  styles (DSv3 relies on `compile()` doing both internally; see `references/demo-stage.md` §7).
- **Single-GPU capable**: yes — the demo runs at `world_size == 1` with a local
  `--model-path` or the HF default `Qwen/Qwen3-8B`; no NVSHMEM collectives exist at TP1, so
  no box is needed. The tracked calibration script
  `tests/runtime_python/blackwell_v2/e2e_qwen3_check.sh` runs v1-vs-`--use-v2`
  token+ms/tok on one local GPU (v1 reference ~4.03 ms/tok noted in its header) — NOTE it
  hardcodes the original machine's `PY=.../mirage/.venv/bin/python` and `DEMO_DIR`; adjust
  those two vars on a clone (it is repo code — do not expect it to self-locate).
- **Upstream twin**: `demo/qwen3/demo.py@mirage-project/runtime_refactor` (head `0eadb3fd`,
  2026-06-11) is the same demo where the Channel-based linear was PROMOTED to be THE v2
  (`linear_layer_v2`/`linear_with_residual_layer_v2`, ids 244/245; non-linear v2 ids parked
  at 224-229). Read it via `git show mirage-project/runtime_refactor:demo/qwen3/demo.py`
  (remote-add note in `v2-kernel-writing/references/upstream-kernel-catalog.md`, which also
  catalogs every upstream v2 kernel the qwen3 path uses).
- **What a Qwen3-class campaign reuses from this pipeline**: Phase 0 classifies nearly
  everything REUSE (all needed v2 tasks exist); Phase (a) is the demo/plan wiring above;
  Phase (b) shrinks to perf rewrites (M1 collectives = none at TP1); Phase (c)'s
  deterministic token-match applies directly (no TP8 nondeterminism protocol needed);
  Phase (d) = `v2-perf-iteration` with the verdict config restated for THAT campaign
  (e.g. single-GPU bs=1024 throughput instead of TP8 bs=1 tpot — restate it in every
  dispatch prompt; the mpk-* defs and this suite default to the DSv3 framing).

## Subagent dispatch model

This suite runs as a nested pipeline: the TOP orchestrator (main thread or the
`v2-model-support-orchestrator` agent) owns phase sequencing and ALL box operations;
it dispatches ONE lead subagent per phase, and a phase lead may dispatch its own
scoped workers (per-op kernel authors, harness writers, reviewers). Hard rules:
- **Box ops (start/stop/ssh/rsync/run) stay with the TOP orchestrator ONLY.** Phase
  leads and workers produce code + local gates; they hand "needs a box run" items up.
  (History: a box-touching subagent leaked an idle box for ~55 min; nested watchers
  park past their stop step. See `references/box-orchestration.md`.)
- Phase gates are blocking: a phase lead reports PASS/FAIL + evidence; the top
  orchestrator never starts phase N+1 on a FAIL.
- Reuse the existing roster where it fits: `mpk-correctness-gate`, `mpk-profiler`,
  `mpk-commit-reviewer`, `mpk-memory-keeper`, `ablation-logic-reviewer`,
  `ferret-kernel-agent`/`kda-kernel-agent` for kernel-perf work.

## Non-negotiables (the failure modes this suite exists to prevent)

1. Never register a live-path v2 task without a COMPLETE consumer body (§1.1: silent
   deadlock, D-state box wedge). The build-time guard must stay green.
2. Never re-zero monotonic barrier scratch after step 0 (`skip_after_step0=True` is
   correctness, not perf — iter-1 hang otherwise).
3. Never paste a 256-thread/`__syncthreads()` body into a 128-thread consumer role.
4. New fused-mega task types MUST be added to the `task_offset = bid.x` block in
   `runtime.cc` (else garbage CTA index → grid-barrier deadlock).
5. Extern-smem regions: `alignment=1024` in the `_spec.h`, and an in-MPK smoke after
   every fused-mega port (the harness cannot catch misalignment of OTHER tasks).
6. A rank must never silently skip a collective (epoch alignment = identical-graph
   determinism; a skipped AR desyncs the team counter → deadlock/stale).
7. Chain first, fuse later; bit-exact per-op before e2e; smallest slice before scale.
8. Default build byte-identical; all new paths opt-in.

## References

| Doc | Content |
|---|---|
| `references/graph-to-plan.md` | drawio convention + parsing, per-rank shape derivation, DSv3 worked op table, classification decision |
| `references/demo-stage.md` | builder anatomy, SHARD_RULES/cache-key/lifetime footguns, v2 wiring specifics |
| `references/debug-gates.md` | the phase-c ladder as checklists, 6 gate-fidelity classes, hang triage |
| `references/box-orchestration.md` | remote-box session playbook (setup/poll split, rsync, retries, safety) — §1-2 site-specific, §3-8 transfer |
| `references/V2_DSV3_DECODE_MASTER_PLAN.md` | the real M0→M5 plan this skill generalizes (archived worked example) |
| `FUSED_KERNEL_DEBUG_METHODOLOGY.md` (repo root) | the original debug order-of-operations |
| `../v2-perf-iteration/SKILL.md` + its `references/loop-agents.md` | the full multi-agent loop (repo-root `WORKFLOW.md` is a superseded stub) |
