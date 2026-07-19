# MPK agent skill suites

Claude Code skills (`.claude/skills/`) and subagent definitions (`.claude/agents/`)
developed during the DeepSeek-V3 decode optimization campaign.

## v1-era suites (apply to `mpk` directly)
- `add-mpk-task` — add a new task/operator to the megakernel (enum → task_register → graph → kernel → builder).
- `add-mpk-model` — bring a new model onto MPK (builder, TP sharding rules, demo wiring).
- `test-mode` — unit-test single layers or multi-layer pipelines through the full compile path.
- `mpk-internals` — reference for the compile → codegen → runtime pipeline.
- `dpskv3-logistic-review` — audit the DeepSeek-V3 demo/builder chain for drift vs the vLLM reference math.
- `ferret-kernel-system` — frozen-gate kernel optimization (dispatcher → independent test-writer → optimizer), so the optimizer can never grade its own homework.
- `mpk-faithful-gate` — faithful in-MPK per-task latency measurement (slowest-CTA at the production grid), the trusted per-kernel number.
- `mpk-lever-cleanup` — collapse env-gated default-OFF experiment levers into a clean single-path branch for a PR.

## Runtime-V2 suites (methodology docs — see note)
- `v2-kernel-writing` — staged workflow (SPEC → IMPLEMENT → WIRE → VALIDATE → PERF → REVIEW) for Runtime-V2 task kernels, with protocol references and worked applications.
- `v2-model-support` — end-to-end pipeline for bringing a model up on the V2 runtime (graph→plan, builder/demo, kernel dispatch, debug gates).
- `v2-perf-iteration` — V2 perf-iteration loop plus the perfetto export/analysis tools under `tools/`.

**Note:** the `v2-*` suites document the Runtime-V2 (`runtime_refactor` branch) methodology and
reference files that live on that branch (e.g. `tasks/blackwell_v2/`, `tests/runtime_python/blackwell_v2/`),
not on `mpk`. The v1-era suites above apply to `mpk` as-is.

## B200 / Blackwell kernel-development skills (distilled)
Distilled from ["Modern GPU Programming for MLSys"](https://mlc.ai/modern-gpu-programming-for-mlsys/)
(MLC Community) plus the NVIDIA Blackwell tuning/compatibility guides; each skill carries this
attribution in its SKILL.md. Model-agnostic B200 (SM100) kernel-engineering method cards; each
ships with its `test-prompts.json` trigger-eval fixtures.
- `b200-scope-layout-dispatch` — map an ML operator onto a B200 kernel: the scope, layout, dispatch, and handoff contract for every tile primitive.
- `b200-tma-pipeline-designer` — convert GMEM↔SMEM tile copies to TMA: descriptor, stage ring, barriers, prologue/steady-state/epilogue.
- `b200-tcgen05-mma-contract-builder` — choose tcgen05 MMA tile/dtype/`cta_group`, SMEM operand layout, TMEM accumulator mapping (incl. mxfp8/nvfp4 block-scaled).
- `b200-tmem-lifecycle-planner` — plan TMEM regions/column budget, `tcgen05.ld/st/cp` paths, epilogue readback, and safe release.
- `b200-mbarrier-protocol-auditor` — per-barrier protocol ledger (arrival, tx-count, phase, wait) for deadlocks / stale reads / premature stage reuse.
- `b200-layout-contract-auditor` — audit shape–stride, thread distribution, swizzle, and hardware operand contracts layer by layer.
- `b200-warp-specialized-debugger` — roles/storage/handoff/lifetime worksheet debugging for warp-specialized kernels; fix one handoff at a time.
- `b200-cluster-persistent-scheduler` — Thread Block Cluster / DSMEM / 2-CTA cooperative MMA / persistent-kernel tile scheduling incl. Cluster Launch Control.
- `b200-kernel-roofline-triage` — classify a slow kernel as bandwidth/compute/latency/scheduling-bound and pick the smallest falsifiable experiment.
- `b200-gemm-optimization-ladder` — staged GEMM bring-up from one correct tile to persistent, warp-specialized, 2-CTA-cluster kernels with per-rung gates.
- `b200-flash-attention4-planner` — FlashAttention-style forward planning: QKᵀ/PV MMAs, online softmax, S/P/O in TMEM, warp roles, tile/barrier graphs.
- `blackwell-build-compatibility-auditor` — verify a CUDA extension/binary really runs on B200: `compute_100`/`sm_100(a)`, PTX/cubin/fatbin, toolkit/JIT evidence.

## Subagents (`.claude/agents/`)
Ferret frozen-gate trio (`ferret-kernel-agent` incl. V2 MODE, `ferret-test-writer`, `ferret-optimizer`);
`kda-kernel-agent` (verdict-grade kernel work); the optimization-loop roster (`mpk-profiler`,
`mpk-correctness-gate`, `mpk-optimization-planner`, `mpk-iterator`, `mpk-commit-reviewer`,
`mpk-memory-keeper`); Runtime-V2 pair (`v2-kernel-engineer`, `v2-model-support-orchestrator`).
