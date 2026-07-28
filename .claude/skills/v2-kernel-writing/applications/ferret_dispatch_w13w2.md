# Ferret-v2 dispatch brief — W13/W2 grouped-GEMM per-tile pipeline (ffn item 1)

> ARCHIVED IN-SKILL COPY (2026-07-15; original authored at `scratch/v2_rewrite/`, which is
> git-ignored and absent from fresh clones). It doubles as the TEMPLATE for any future
> ferret-v2 dispatch brief: copy the section skeleton (§0 pre-dispatch gate → §1 invocation
> → §2 targets → §3 gate bars → §4 protocol_frozen → §5 budget → §6 workspace → §7 report).
> Requires the machine-local `~/ferret/` install (see SKILL.md §Environment prerequisites).

**Status (2026-07-15): the run happened on this machine** (ws7, dsv3_ffn_gg_v2 pair —
working-tree-only, not committed on dsv3-decode-clean; see ferret-v2-dispatch.md seed notes
for what it learned). A fresh clone treats this brief as the worked example, not live state.
Requirement of record: `.claude/skills/v2-kernel-writing/applications/ffn_item1_spec.md`
(Stage-1 spec, Codex-reviewed §10). Flow + contract:
`.claude/agents/ferret-kernel-agent.md` §"V2 MODE" +
`.claude/skills/v2-kernel-writing/references/ferret-v2-dispatch.md` +
`~/ferret/tasks/TEMPLATE_v2.yaml` (machine-local).

## 0. Pre-dispatch gate (orchestrator work that MUST precede the launch)

Ferret-v2 needs the gate substrate live. Owner = main thread / campaign orchestrator:

1. **S3-stub wiring** (wiring-recipe.md, spec §8 sketch): `TASK_DSV3_FFN_W13_PIPE_V2` +
   `TASK_DSV3_FFN_W2_PIPE_V2`, compile-clean stub pair `dsv3_ffn_gg_v2.cuh` +
   `dsv3_ffn_gg_v2_spec.h` (mechanical from spec §2 region table + §3 SEM table — the stub
   PINS the ABI ferret must keep), env gate `MPK_DSV3_V2_FFN_PIPE` default-OFF, chain path
   selectable (spec Q5). Both enums into the runtime.cc task_offset=bid.x list (deadliest
   footgun) + task_type_to_name.
2. **Resolve spec §9 blockers**: Q3 (page_finished accounting under launcher-blanket vs
   storer-per-stage — copy linear's proven combination verbatim into the stub) and Q4
   (input_tma_desc_ptrs population; 3D expert-weight desc vs v1 2D GMEM_ROW flattening) block
   the STUB; Q2 (sharpened pow2 assert) and Q6 (W2 segment order in the reference) go INTO
   the gate; Q1 (Phase-0 anchors) binds the ratios at freeze (below).
3. **Harness case extension**: pipe-op cases in `dsv3_ffn_harness.py`/`ffn_case_runner.py`
   (spec §7 item 1) — part of the gate substrate, written by the test-writer at freeze.
4. **Gate freeze** (ferret-test-writer, per V2 MODE (a)): frozen `gate/` wrapping
   `run_ffn_suite.py`-pattern subprocess cases + `gate.sha256` + `manifest.sha256` over the
   in-tree harness + wiring files. Baselines live-benched at freeze (Q1).

## 1. The Agent() invocation content (paste as the ferret-kernel-agent prompt)

```
Run a ferret V2 MODE optimization (your "# V2 MODE (Runtime-V2 kernels)" section governs).

TASK: dsv3-ffn-w13w2-pipe-v2-decode
task.yaml: author from ~/ferret/tasks/TEMPLATE_v2.yaml with the field values in
  .claude/skills/v2-kernel-writing/applications/ferret_dispatch_w13w2.md §2-§5 (this brief).
spec_doc (requirement of record):
  <repo>/.claude/skills/v2-kernel-writing/applications/ffn_item1_spec.md
  — implement EXACTLY it: idiom=role_pipeline, roles [loader,launcher,consumer,storer],
  W13 routed 64 tasks + W13 shared 4 tasks + W2 56 tasks, per-tile task_offset identity,
  STAGES=8/ACC=2/MMA_N=16/bK=128, SMEM 9 regions/10 pages/155664B (§2), 30 SEM ordinals (§3),
  role responsibilities + inactive-slot bail protocol + B_sf async-proxy-fence variant (§4).
source_engine: include/mirage/persistent_kernel/tasks/blackwell/fp8_group_gemm_sm100.cuh
  (v1 swapAB block-scaled UMMA — this is a re-host, not a new engine; W5 evidence row).
destination exemplar: tasks/blackwell_v2/linear_sm100_v2.cuh + _spec.h (house style),
  channel.cuh primitives preferred.
WORKSPACE: <N per §6 rule below — record it>.
GPU: torch-probe a free local B200 (probe before trusting nvidia-smi; commonly free 5/6/1).
GATE: already frozen at ~/ferret/workspace<N>/gate/ (do not proceed if absent — hand back).
MODE W (write-from-spec): REPRODUCE = make the wired stub real until gate correctness
  passes; then OPTIMIZE toward the per-config target_ratio. Class A (.cuh-only) default;
  Class B (spec.h geometry, e.g. STAGES 6..12) budget ≤3 candidates, library rebuild first.
Friction escape: ≥2 rounds on the same protocol wedge class → STOP, report; the caller
  falls back to v2-kernel-engineer shell + re-dispatch you on the math-only inner body.
TIER-1 is NOT yours: production geometry is TP8 EP2 (FFN pipe exists only at world_size=8)
  → report "pending TIER-1 (box, orchestrator)" with the TIER-2 gate evidence.
```

## 2. Configs + target_ratio derivation (HIGH-AMBITION, Step 0.5 — the arithmetic)

Anchors (spec §0, subject to Q1 live re-pin at gate freeze; the harness benches the
baseline in the SAME chain geometry):

| Anchor | Value | Source |
|---|---|---|
| v2 chain FFN consumer compute | 52.2 µs/layer (+14.2 dep-wait) | 2026-07-09 profile (unverified until Q1) |
| — of which chain W13+W2 GEMV bodies | ≈35–45 µs (split = Q1) | spec §1; placeholder split W13 ≈27 µs, W2 ≈13 µs |
| v1 tcgen05 group-GEMM W13 body | ≈12.19 µs class | W5 anchor (proven engine at M=1) |
| Weight-stream floors (achievable BW 2.0–3.4 TB/s) | W13 ≥ ~9–16 µs, W2 ≥ ~5–8 µs | spec §0 |
| Predicted Δ band (falsifiable, spec §1) | W13 body 12–20 µs, W2 body 6–12 µs | U3 probe pre-registration |

Targets = the predicted band's FAST edge, anchored by W5 (v1-parity is proven achievable
by this engine, so it is aggressive-but-not-fantasy; the floors say it is above the
achievable-BW floor):

| config | perf case | target_body_us | target_ratio (placeholder anchors) | recompute at freeze |
|---|---|---|---|---|
| `w13_body` | routed+shared W13 tile tasks, active=4 | **12.2** (v1 parity) | 27 / 12.2 ≈ **2.2** | anchor_w13_us / 12.2 |
| `w2_body`  | W2 tile tasks, active=4 | **6.0** (band fast edge) | 13 / 6.0 ≈ **2.2** | anchor_w2_us / 6.0 |

- `scoring: min_ratio`, `stage_gate: {ratio: 0.95, strict: true}` (REPRODUCE exit =
  correct + ~parity with the chain anchors).
- Best-effort floor (§6.5 applies): the band's SLOW edge (W13 20 µs / W2 12 µs ≈ the ≥25%
  combined cut) — a stalled-but-correct kernel at ≤ slow edge still DELIVERS; the spec §1
  KILL (TIER-1 Δ <5% vs chain anchor) is judged at TIER-1 by the orchestrator, not here.
- Watch metric (gate reports, PARK trigger, not a score): per-layer FFN wall + dep-wait —
  124 per-tile tasks replace 2 fat ops; body-win + wall-regress >5% ⇒ PARK pending Phase-4
  overlap (spec §1).

## 3. Gate reference (= spec §7, the frozen bars)

- Correctness per case: **cos ≥ 0.999, rel_max ≤ 3e-2, no NaN** vs the fp32 torch reference
  AND vs the chain GEMV output (v1-counterpart; math-changing swap — accumulation order
  differs, so counterpart compare is tolerance-based, segment order per Q6).
- Case matrix: **active ∈ {0, 1, 4, 8}**, duplicate-expert-ids-across-rank-filter case,
  n_tile edges, and the Q2 assert (all four scale packs positive/finite/exact-pow2 in the
  UE8M0-representable range — assert at load; failure = STOP, not a tolerance).
- Perf: TIER-2 `body_span` p50 at production grid (136), active=4 scored + active=8
  reported (routed-imbalance spread, per the load-balance gate rule). **Cold-L2: chain
  L ≥ 6** (one FFN layer instance streams ≈47.6 MB at active=4; 6 × 47.6 ≈ 286 MB ≥ 2× the
  126 MB L2 — L=4 is only 1.5× and warm-contaminated).
- Protocol: a harness wedge/timeout/IMA = round FAIL → mbarrier-protocol-auditor ledger
  (spec §3/§4 tables are the input; the inactive-slot no-handshake bail is a mandatory
  ledger row) before the next candidate. Never re-run a wedge unchanged.
- Trust gates: decode-integrity errors=0, 3-run body_span spread <5%, wall cross-check —
  all fail-closed to KERNEL_RESULT 0.0.

## 4. protocol_frozen (per-op restatement — goes into task.yaml verbatim)

Baseline list (v2_runtime_notes.md §2 / TEMPLATE_v2.yaml) PLUS the op-specific rows:
- SEM table of spec §3 (30 ordinals, counts, directions) incl. the B_sf publication rule:
  `fence.proxy.async.shared::cta` before the release-arrive (or the documented
  cp.async.mbarrier.arrive.noinc alternative — pick ONE, ledger it; Codex 4a).
- Stale-arrival re-init ownership table of spec §3 verbatim (loader: W_tma+mma+B_sf;
  launcher lane 0: mainloop+epilogue) + **re-init-before-bail on the inactive path**
  (Codex 4b hardening).
- Inactive-slot protocol of spec §4: all four roles detect from meta independently, NO
  cross-role wait on the bail path, every declared page still arrived exactly once.
- Loader dep-wait asymmetry: routed tasks dep-wait BEFORE reading meta (weights cannot
  prefetch ahead); shared-GU keeps the linear property (W TMAs first, inline dep before
  first B cp.async). Positions are contract, not knobs.
- W2 segment order = slots-ascending-then-shared (determinism pin, Q6); f32 register
  accum + single bf16 store, no atomics.
- B-tile padding: rows 1..15 zero-filled ONCE at task start before any B_sf release.
- taddr cached at alloc; launcher-blanket page release + `auto_consumer_finish=false`
  exactly as Q3 resolves it (copy linear verbatim).

**Free to optimize (Class A):** K-issue order, L2 hints, UTCCP splat scheme, tcgen05.ld
width (x16→x4/x1 pre-registered ≤1-2 µs), epilogue vectorization, elect/lane assignment.
**Class B (≤3, rebuild):** STAGES ∈ {6..12} (14-page headroom per spec §2; never <6 — v1
hung at 4), region resizing within budget, MMA_N=8 experiment.

## 5. Budget + output

- `budget: {max_iterations: 40, max_wall_minutes: 300}` — long budget; the bar stops it,
  not the wall (Step 0.5). Episode-bounded rounds via the in-session optimizer.
- Output: `result_keys: [w13_body, w2_body]`; deliverable = the pair
  `dsv3_ffn_gg_v2.cuh` + `dsv3_ffn_gg_v2_spec.h`; promotion.authority =
  v2_harness_body_span; final_acceptance = tier1_in_mpk_slowcta (orchestrator, box TP8 EP2,
  spec §1 kill/park thresholds bind THERE).

## 6. Workspace index choice rule

Claim a workspace with (i) no live process: `pgrep -af 'cc-run|ferret'` empty, AND
(ii) `progress.md` mtime > 7 days (its prior task is finished/extracted). Prefer the
OLDEST. Snapshot 2026-07-15: all 8 stale; **ws7** oldest (2026-06-06,
fp8-gemm-qkva-swaab-push) → **use workspace7**; re-verify both conditions at dispatch
time. One workspace per run; a parallel run takes the next-oldest index + a different GPU.

## 7. Report-back contract

Return: pair paths + `GATE_RESULT` per case + the capped `KERNEL_RESULT` /
`KERNEL_RESULT_REFERENCE` lines + gate.sha256/manifest.sha256 + Codex Integrity+Plan
verdicts per round + Class-B candidates used + explicit "pending TIER-1 (box,
orchestrator)". The orchestrator then runs spec §7 items 3-5 (in-MPK 0-3 probe, multi-step
iter ≥3, TP8 poison-fill, Qwen3 regression) and the TIER-1 verdict before any land/record.
