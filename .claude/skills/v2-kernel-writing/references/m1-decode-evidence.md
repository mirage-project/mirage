# M=1 decode evidence map — check BEFORE proposing (anti-loop)

Context: DSv3 decode, bs=1/M=1, B200, 136 workers × 1 CTA/SM × 256T megakernel envelope.
This doc is why our v2 kernels deviate from the reference pipeline idiom **where they do**.
Any new-kernel SPEC (SKILL.md Stage 1) must cite the row it is consistent with — or state
explicitly which UNTESTED cell it is probing and its falsifiable predicted Δ.
Source citations per row name personal-memory files
(`~/.claude/projects/-home-muhengl-mirage/memory/`) — OPTIONAL CONTEXT, present only on the
original user account, absent from a fresh clone/other account. **This doc is the
self-contained distillation**: the rows carry the verdict + number + mechanism; the memory
files are only the longer-form provenance when available.

## 1. Measured DEAD at M=1 (do NOT re-propose without new mechanism)

| # | Lever | Verdict + number | Mechanism (one line) | Date / source |
|---|---|---|---|---|
| D1 | Dedicated loader/producer warp inside the fused scalar-GEMV envelope (warp-spec producer) | DEAD, killed 3× (+67%, +17% regressions) | Loader steals a MAC slot; no register-budget gain without tcgen05/TMEM; sync cost > 0 | 2026-07-03 `feedback_blackwell_sota_fixed_warpspec_tcgen05` |
| D2 | tcgen05 for projection GEMVs IN the fused megakernel (qkv_a/q_b/o_proj) | DEAD; body 8.26µs vs achievable floor 7.65µs → gap 0.61µs (7%); real BN=16 attempt = 27µs (3.2× slower) | Fenced 256T/1-CTA/SM envelope binds; 3 kill-arms: 17/136 CTA underfill, alloc+mbar ~3.9µs ≫ gap, K-split = multi-rank crash hazard | 2026-06-29 aebd470e; `feedback_consumer_math_ceteris_paribus_protocol` |
| D3 | Decoupled-producer / TMA de-fuse / deeper prefetch for standalone projection GEMVs | DEAD, SETTLED; armMAC (consumer alone, input hot) = 6.24µs = 2.50 TB/s < 3.5 TB/s LIVE threshold; arm3a warp-spec 9.95µs REGRESS; arm3b STAGES=12 REGRESS; ceiling 1.18× UNREALIZED | Any load-side lever's best case = "loads free" = the consumer-MAC floor; the consumer forecloses the family. Prior 1.53× claim used a no-consumer denominator (WRONG); "STREAM 6.1 TB/s" was a 1-GiB number, 15.6MB cold ≈ 3.4 TB/s | 2026-07-03 ac3d9736; `feedback_roofline_floor_trap_achievable_not_theoretical` |
| D4 | Fine-grained per-slot counter release replacing the FFN megakernel's coarse grid barrier | REGRESS +11-17% (grows with active surface); FG0 control +53% | Per-slot cpasync_wait+fence+atomicAdd serializes W13 emission; W2 is output-stationary (last slot gates anyway); W13‖W2 HBM contention cancels streaming | 2026-07-04 Codex 019f2fab; `feedback_ffn_megakernel_fg_counter_regress` |
| D5 | 7-warp unlock on attn cp.async GEMVs (p0_qkva/qb/oproj) | REGRESS +32µs (R1, +11.83µs qkv_a even when trips drop) | In-flight-bytes / cp.async-issue-bound: more warps = more stragglers, per-warp throughput ~4/7; OPPOSITE regime to FFN | 2026-07-03 a9e1190f; `feedback_warp_scaling_pertype_not_aggregate` |
| D6 | MoE work-distribution (CTA-team partition, FIFO reorder, de-fuse shared/routed) | 4×-DEAD; box A/B NS=48/56/62/72 ALL REGRESS +0.058..+0.342ms | Flat warp-stride already interleaves shared+routed; even-striping physics at M=1; partition adds skew | 2026-06-20/27/28/29; `feedback_bw_bound_moe_work_distribution_dead` |
| D7 | AR-kernel axis (reduce body, RS+AG, RMSNorm-into-AR, megakernel fold, chunked-AR, radix/barrier tuning) | EXHAUSTED; skip-entire-reduce = −2.7µs only; folds +0.69/+0.92ms | Sync-limited not reduce-limited; NVLS already active; 4.1µs ≈ NVLink latency; monolithic producer → R_sched≈0 | 2026-06-29 final; `feedback_ar_kernel_axis_exhausted` |
| D8 | Naive occupancy de-fuse (more identical fenced CTAs/SM) | DEAD, FLAT 1→4 CTA/SM (±0.08 TB/s) | Bottleneck = per-CTA fenced cp.async look-ahead depth, not resident-CTA count | 2026-06-29; `feedback_gemv_occ_sweep_latency_mlp` |
| D9 | STAGES depth 4→12, split-N 1→4, split-K oracle 1→8 for qkv_a/q_b | ALL FLAT (≤6%) | Consumer-math pipeline bound (A−C = 14.5µs scalar MAC path), not load pipelining | 2026-06-29 v4/v6; `feedback_gemv_occ_sweep_latency_mlp` |
| D10 | W_UV fold into MLA body | DEAD by construction; ceiling 3.2µs < 2-8µs parallelism-loss cost | Per-sub-phase slowCTA maxima on DIFFERENT CTAs are NOT additive; true ceiling = later-wall − earlier-wall | 2026-06-28/29; `feedback_attn_body_wuv_additivity_confound` |
| D11 | In-megakernel role-TYPE morphing (per-phase loader↔MAC reassignment) | Ruled out by SOTA survey + D1/D5 | No SOTA design morphs role types; the only live dial is MAC-warp-COUNT per op | 2026-07-03; `project_hybrid_adaptive_warp_spec_directive` |

## 2. Surviving WINs (the deviations that are LOAD-BEARING — keep them)

| # | Win | Number | Mechanism (corrected) | Source |
|---|---|---|---|---|
| W1 | FFN nwarps=7 MAC-fold via salted tag-flags (helpers = loader/launcher/storer warps run the same MAC body) | −7..−9% vs v1 (commit f624d75a) | NOT generic "warp scaling": INTEGER-TRIP WAVE-QUANTIZATION — win iff `ceil(items/(ntasks*nwarps))` drops; tie → regress (threshold-shift proof). Run this arithmetic per op BEFORE porting 7w | 2026-07-03 acd86f8a; `feedback_warp_scaling_pertype_not_aggregate` |
| W2 | 7-warp on compute-dense bodies (MLA fmha accum) | −36% @ KV=544 | Dense MAC accumulation genuinely warp-scales; per-task-type probes mandatory | same |
| W3 | Coarse monotonic GMEM grid barrier in grid-wide fused ops | Rung-B mega TIES tuned 3-op chain; optimal vs FG (D4) | Barrier fences once; is NOT the perf cap at M=1 | 2026-07-04 |
| W4 | Salted tag-flags replacing op-private mbarriers for multi-role handshakes (dsv3_ffn_v2.cuh:110-153) | fixed a real deterministic multi-iteration wedge | 64-bit exact-match monotonic tags: no init, no parity, no controller involvement; stale bytes can never satisfy a wait | 2026-07-04 commit 62449212 |
| W5 | tcgen05 IS used at M=1 outside the fused envelope | W13/W2 group-GEMM (fp8_group_gemm_sm100.cuh:71 swapAB), dense small/medium-M | scalar-MAC is CONFINED to fused attn+router | `feedback_blackwell_sota_fixed_warpspec_tcgen05` |

## 3. GENUINELY UNTESTED (the open cells a new kernel may legitimately probe)

| # | Cell | Status | Bound / caveat |
|---|---|---|---|
| U1 | TMA-loader INSIDE the fused consumer body (TMA issue from a MAC warp, no dedicated producer warp) | Mechanism-excluded but never directly falsified | Upside bounded by the D3 consumer floor (6.24µs); predict Δ ≤ 1.18× and pre-register the kill threshold |
| U2 | Full reference-style de-fused rewrite: per-op/per-tile v2 tasks + RUNTIME-scheduled cross-task overlap replacing lockstep grid barriers | THE exec-model direction; not yet built for attn/FFN | This is the only surviving 8ms path (grind floors ~8.4ms, `project_goal_8ms_decode`); see applications/attn-ffn-reference-rewrite-plan.md. Attn chain AS SCALAR CHAIN measured 1.41-1.56× slower than v1 mega — the untested delta is pipeline-idiom-inside-tasks + overlap, not the chain shape itself. UPSTREAM DATA POINT (linear_spec.h@runtime_refactor CROSS_TASK_PAGES, qwen3/B200): per-stage cross-task page acquire/release with per-task footprint = full 14-page pool is pure added sync, **~+12% REGRESS**; it pays only with footprint ≤ half-pool AND planner double-buffered page assignments — budget those co-requisites into any cross-task-overlap design |
| U3 | Reference pipeline (TMA+tcgen05) for genuinely GEMM-shaped v2 work: M≥16 tiles (prefill), lm_head N-tiles, grouped W13/W2 as per-tile tasks | linear_v2/v3 prove the idiom works in-runtime | Evidence rows above do NOT kill this — they kill it only at M=1-GEMV-in-fused-envelope |

## 4. Measurement rules that gate ANY perf claim here

- **TIER hierarchy** (`feedback_gate_warm_isolated_vs_mpk_cold_barrier`): TIER 1 = in-MPK
  %globaltimer PPH/slowCTA at production grid=136 (sole verdict-grade). TIER 2 = faithful
  per-task harness slowCTA (corroborates). TIER 3 = cudaEvent-WALL (over-states ~8µs; once
  minted a bogus "4× gap") or standalone-warm — DIAGNOSTIC ONLY, never promote/reject on it.
- **%globaltimer, never cudaEvents**, for 5-40µs bodies (events quantize 10-40µs; a 20µs GEMV
  once read 0.76ms).
- **Floor claims need a SAME-GRID load-only/xor-consumer probe** (achievable floor), never a
  theoretical-peak roofline: 8 TB/s peak → real 2.0-2.5 TB/s at this geometry
  (`feedback_roofline_floor_trap_achievable_not_theoretical`). The xor-consumer arm (same fenced
  waits) is the clean floor; load-only overstates look-ahead.
- **slowCTA maxima from different CTAs are not additive** (D10). Recoverable = later-phase wall −
  earlier-phase wall.
- **Cold-L2 is the production regime** — warm-isolated gates over-stated an FFN improvement 2.5×.
- **A sub-agent µs with no on-disk CSV/trace backing it is (unverified)** — reject verdicts on it.
- **Same-binary probe-pair** dissolves binary-mixing confounds
  (`feedback_same_binary_probe_disambiguates_mixing`).

## 5. How to use this doc (the anti-loop contract)

1. Stage-1 SPEC must include an "Evidence check" section: for each design choice, the row id
   (D*/W*/U*) it rests on.
2. Proposing anything in §1 requires: (a) a NEW mechanism not covered by the row's kill logic,
   (b) a pre-registered falsifiable predicted Δ + kill threshold, (c) ablation-logic-reviewer
   sign-off BEFORE building.
3. Anything in §3 is fair game but carries its stated bound as the pre-registered ceiling.
4. After the experiment: mpk-memory-keeper updates this map's source files (this doc is a
   distillation — the memory files stay ground truth).

## Known gaps (from first live run, 2026-07-09)

- No per-op chain W13/W2 body baselines exist yet — the Phase-0 box anchors
  (attn-ffn-reference-rewrite-plan.md) must be captured before any kill threshold in that plan
  can bind.
- The 52.2/14.2µs FFN numbers trace to the 2026-07-09 v2prof run (`outputs/v2prof/` on the
  original machine; box `outputs/v2prof_run1` — both machine-local, not in a clone) — cite
  that provenance when using them, and re-pin on the current machine before verdict use.
