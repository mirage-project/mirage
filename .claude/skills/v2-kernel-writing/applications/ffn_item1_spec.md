# Stage-1 SPEC — FFN work item 1: W13/W2 grouped GEMM as per-tile v2 pipeline tasks

> ARCHIVED IN-SKILL COPY (2026-07-15; original authored at `scratch/v2_rewrite/`, which is
> git-ignored and absent from fresh clones). It is the campaign's Stage-1 spec of record AND
> the exemplar of what a v2-kernel-writing Stage-1 SPEC deliverable looks like (engine
> decision, SMEM region table, SEM ordinal table, role responsibilities, evidence check,
> gates, open questions). NOTE for clone readers: the `dsv3_ffn_gg_v2` pair this spec
> produced exists only as working-tree state on the original machine — it is NOT committed
> on dsv3-decode-clean; a fresh clone will not find it in tasks/blackwell_v2/.

**Campaign**: attn/FFN reference-style rewrite
(`.claude/skills/v2-kernel-writing/applications/attn-ffn-reference-rewrite-plan.md`).
**Work item**: the plan's FIRST kernel-writing item — FFN first (plan §Sequencing: "FFN first
(smaller, bit-exact refs, rehearses the loop), then attn"), Phase-2 candidate 2: "**W13/W2
grouped GEMM as per-tile v2 pipeline tasks** — tcgen05 already proven at M=1 in v1 (W5:
fp8_group_gemm swapAB); the v2 port is a re-hosting, not a new engine. Target: reference
loader/launcher/consumer/storer roles + per-stage page release." This item SUBSUMES the
Phase-1 bullet "W13/W2 expert×N-tiles": per-tile granularity for W13/W2 is only
evidence-viable WITH the pipeline body — a per-tile split of the *scalar* GEMV body is a
work-distribution partition of an evenly-striped M=1 body, which is the D6 kill class
(4×-DEAD, "partition adds skew; flat warp-stride already interleaves shared+routed"). See §5.

**Designer**: Stage-1 subagent, 2026-07-15. Stage-0 read: SKILL.md, references/house-style.md,
references/m1-decode-evidence.md, references/wiring-recipe.md (all in full).

**Status**: SPEC ONLY. No implementation, no registration, no box run.

---

## 0. Op contract (ground truth from code, TP8 EP2 decode, M=1, per rank)

Shapes from `include/mirage/persistent_kernel/tasks/blackwell_v2/dsv3_ffn_v2_spec.h:31-56`
(static_assert-pinned to `ffn_full_megakernel_sm100` in dsv3_ffn_v2.cuh:73-79):
HIDDEN=7168, W13_N=1024/expert, W2_K=512, W2_N=7168, GRP=128, KG1=56, KG2=4, MAX_ACTIVE=8,
E_LOCAL=128, SH_GU_N=512, SH_DN_K=256, META_INTS=24 (meta = active_count, magic, experts[8],
weights[8]).

Three GEMM-shaped sub-ops (all M=1 token):

| op | math | weights (fp8 e4m3, K-contig) | scales (f32, pow2) | out |
|---|---|---|---|---|
| W13 routed | y13[s,0:1024] = W13[e_s] @ a | w13 u8[128,1024,7168] | w13_scale [128,8,56] (128×128 blocks) | y13 f32[8,1024], per-slot, **no cross-expert reduce** |
| W13 shared | sg[0:512] = wgu @ a | wgu u8[512,7168] | wgu_scale [4,56] | sg f32[512] |
| W2 (+shared dn) | out[0:7168] = Σ_s ew_s·(W2[e_s] @ i_s) + wdn @ si | w2 u8[128,7168,512]; wdn u8[7168,256] | w2_scale [128,56,4]; wdn_scale [56,2] | out bf16[1,7168] |

Activations: a_fp8 u8[7168] + a_scale f32[56] (UE8M0-pow2 groups, produced by T1
router_quant); i_fp8 u8[8,512] + i_scale f32[8,4], si_fp8 u8[256] + si_scale f32[2]
(produced by T4 silu_quant). Routing: meta i32[24] (produced by T2 topk; EP-local filter,
active_count ∈ 0..8; ≈4 typical at EP2).

**DAG position** (v2 chain wiring, persistent_kernel.py:4051-4176; builder v2 currently
wires only the mega at builder.py:3174): `rmsnorm → T1 router_quant → T2 topk → [THIS: W13
tiles] → T4 silu_quant → [THIS: W2 tiles] → NVSHMEM AR(+residual)` (builder.py:3705-3715).
a_fp8/a_scale and i_fp8/… are hidden writes; ordering to the W13/W2 tasks is transitive via
the meta / y13-sg graph edges (same convention as the existing chain — keep it).

**Cost anchors**:
- Current v2 FFN consumer body 52.2µs compute + 14.2µs dep-wait per layer (2026-07-09
  profile, given by orchestrator — no on-disk CSV named; treat per m1 §4 as (unverified)
  until Phase 0 re-pins it).
- v1 tcgen05 group-GEMM W13 body ≈ 12.19µs class; whole-FFN body ~57µs; plan Phase-2 target
  band "parity with v1 group-GEMM body (12-57µs class) inside v2".
- Weight-stream volumes at active=4: W13 32 live tiles × 0.875MB + shared 3.5MB ≈ 31.5MB;
  W2 ≈ 16.1MB. At the m1-geometry achievable 2.0-3.4 TB/s (m1 §4 floor rules — achievable,
  never theoretical peak): W13 ≥ ~9-16µs, W2 ≥ ~5-8µs.

**Source engine** (the re-host source, verbatim where possible):
`tasks/blackwell/fp8_group_gemm_sm100.cuh` — swapAB block-scaled FP8 UMMA
(UMMA A ← weight[N,K], B ← activations[tokens,K], header :63-88), MMA_M=128, MMA_N=16,
bK=128, NUM_AB_STAGE=8, NUM_ACC_STAGE=2 (task_register.cc:3481-3494), UE8M0 scales via
UTCCP→TMEM (SM100_UTCCP_4x32dp128bit_1cta), SM100_MMA_MXF8F6F4_SS + make_instr_desc_block_scaled,
per-expert CTA + N-split decomposition (:110-119).
**Destination idiom**: `tasks/blackwell_v2/linear_sm100_v2.cuh` role pipeline
(loader W4 / launcher W5 / consumers W0-3 / storer W6; house-style §1-§2).

Warp-role mapping v1 → v2:

| v1 fp8_group_gemm | v2 role | note |
|---|---|---|
| warp 5 DMA (TMA W, cp.async B) | **loader W4** | + absorbs v1 warp-6 scale work (see §4 — block-pow2 scales collapse it to a splat) |
| warp 4 MMA (UTCCP + UMMA) | **launcher W5** | + tcgen05.alloc/dealloc + taddr publish (linear pattern) |
| warps 0-3 epilogue | **consumer W0-3** | tcgen05.ld + store; W2 adds per-segment ew-weighted register accumulation |
| warp 6 scale | (folded into loader) | v1 needed a warp for per-ROW f32 scale load+convert+transpose; our scales are per-128×128-block pow2 → 1-2 f32 per stage, splat, transpose = identity |
| — | **storer W6** | NEW vs v1: per-stage page release (reference Phase-5 behavior, linear:630-767) |

**Scale-format decision**: consume the SAME tensors the v2 mega/chain already bind
(`_w13_scale_pow2` etc., builder.py:3108-3128) — per-128×128-block, power-of-2 f32. Do NOT
attach the v1 group-gemm's per-row [E·N, K/128] scale tensors (those exist only for the
TP<8 v1 path). pow2 → UE8M0 conversion is exact; splat 1 value across the 128-row tile in
SMEM (the UTCCP 4×32 column-major layout of a constant is the constant — transpose skipped).
Runtime contract: ALL four scale packs are exact powers of two — harness must assert this
(open question Q2).

---

## 1. Engine choice (decision tree) + evidence rows

**Branch taken**: "GEMM-shaped … real K-pipeline" → **reference pipeline w/ TMA + tcgen05**
(SKILL.md Stage 1 tree, bullet 1). Qualification, stated honestly against the tree's
wording: a single fp8 weight K-stage tile is 128×128 B = 16KB = **1 page** (the tree says
"weight tiles ≥ 2 pages", written for bf16 32KB tiles); the intent-level test passes
decisively — W13 streams 56 K-iters × 16KB = 896KB of weights per tile-task through a
≥6-stage ring (≥6 pages of W in flight), W2 streams 18-34 K-iters. This is a real
K-pipeline in every sense that matters; flagged as doc-wording friction, not a deviation.

- **U3** (the row this item probes): "Reference pipeline (TMA+tcgen05) for genuinely
  GEMM-shaped v2 work: … grouped W13/W2 as per-tile tasks — linear_v2/v3 prove the idiom
  works in-runtime. Evidence rows above do NOT kill this — they kill it only at
  M=1-GEMV-in-fused-envelope."
- **W5** (feasibility anchor): tcgen05 IS used at M=1 for W13/W2 in v1
  (fp8_group_gemm_sm100.cuh:71 swapAB) — the engine body is measured at the 12µs class.
  This is a re-host, not a new engine.
- **U2** (direction): per-op/per-tile v2 tasks + runtime-scheduled overlap is THE exec-model
  direction; this item is its first FFN increment.
- **NOT D2/D3**: those kill tcgen05/de-fuse for projection GEMVs *inside the fused 256T
  envelope* and load-side levers for *scalar-consumer* bodies. Here the consumer is not a
  scalar MAC — the MMA does the math; D3's consumer-MAC floor argument does not transfer.
- **NOT D4**: D4 killed cross-grid per-slot GMEM counter release inside the lockstep
  megakernel. This design uses op-private SMEM mbarriers INSIDE a per-tile task (the
  reference protocol) and runtime events BETWEEN tasks — neither is the D4 mechanism.
- **D6 guard (why granularity and engine land together)**: per-tile split of the *scalar*
  W13/W2 body = CTA-team partition of an evenly-striped BW-bound body = D6's 4×-DEAD class.
  Therefore NO intermediate "per-tile scalar" milestone is built (§5).
- **W1 note**: the wave-quant `ceil(items/(ntasks·nwarps))` tie-test governs MAC-warp
  bodies; the pipeline body has no MAC warps → N/A. The chain GEMV bodies it replaces keep
  their tuned nwarps (w13@7, w2@7) as the A/B baseline; do not touch them.

**U-row probe pre-registration (from the plan, Phase 2, verbatim numbers)**:
- **Predicted Δ (falsifiable)**: per-op TIER-1 slowCTA of the pipe W13/W2 bodies lands at
  **parity with the v1 group-GEMM body class (12-57µs)**; derived band: W13(routed+shared)
  body 12-20µs, W2 body 6-12µs — i.e. ≥25% cut vs the Phase-1 chain GEMV bodies (chain
  W13+W2 ≈ 35-45µs of the 52.2µs consumer compute; exact split = Phase-0 anchor, Q1).
- **Kill threshold**: TIER-1 Δ < 5% vs the Phase-1 chain body anchor, OR any protocol hang
  that survives the validation-debug ladder → record in experiment_history +
  m1-decode-evidence and STOP the candidate (plan Phase-2 kill, verbatim).
- **Watch metric (park trigger)**: per-layer FFN *wall* (body + dep-wait). 124 per-tile
  tasks replace ~2 fat ops; if the body wins but dep-wait/page-churn regresses the FFN wall
  > 5%, PARK pending Phase-4 overlap (which is what per-tile granularity exists to feed) —
  do not silently ship a wall regression.

---

## 2. SMEM region plan + page math (house-style §4 format)

One plan serves both task types (W13 and W2 — same stage geometry). STAGES=8 (v1-proven
depth at this K-count, task_register.cc:3484-3491; 4 stages hung at TP4 K-depth 8 — do not
go below 6). PAGE=16KB, budget 14 pages / 224,256 B / ≤16 regions
(runtime_header.h:96-101, v2_smem_planner.py:6).

| region | ordinal | size | align | can_pack | page_count | contents |
|---|---|---|---|---|---|---|
| `W_0..W_7` | 0..7 | 16,384 B each | 1024 | false | 1 each | fp8 weight K-stage tile [128×128], TMA dst (128B-swizzle ⇒ 1024-aligned, house-style §5) |
| `BSF` | 8 | 24,592 B | 1024 | false | 2 | 8 per-stage slices @ +s·3072: B tile [16×128] u8 @+0 (2048B), SFA u32[128] @+2048 (512B), SFB u32[128] @+2560 (512B); + taddr scratch 16B @+24,576 |
| — total | 9 regions | **155,664 B** | | | **10 pages** | ≤ 224,256 ✓, ≤14 ✓, ≤16 regions ✓ |

- Per-stage BSF slice stride 3072 = 3·1024 → every B slice and SFA slice is 1024-aligned;
  SFB at +2560 is 128-aligned (matches v1's `alignas(128)` sfa/sfb, :205-207). B is written
  by cp.async (not TMA — see §4), so 1024-alignment is not load-bearing for B, only tidy.
- Why one BSF region instead of per-stage regions: 8 W + 8 B + 8 SF + scratch = 25 regions
  > MAX_SMEM_REGIONS_PER_TASK=16. Folding B+SF+scratch into one 2-page region gives 9
  regions / 10 pages and only costs holding 2 sub-1-page-class pages to task end (the W
  pages — 8 of 10 — keep per-stage release, which is where the cross-task prefetch value
  is). Headroom: 4 free pages → STAGES can grow to 12 (12 W pages + 2 BSF) as a Stage-5
  knob without a redesign.
- spec.h must carry: region ordinals, `make_smem_info(stages)`, capacity static_assert vs
  PLANNER_CAPACITY_BYTES=224,256 (== v2_smem_planner.CAPACITY_BYTES), and static_asserts
  pinning HIDDEN/W13_N/W2_K/W2_N/GRP/MAX_ACTIVE/SH_GU_N/SH_DN_K/E_LOCAL to
  `kernel::ffn_full_megakernel_sm100` and `kernel::dsv3_ffn_v2` constants (copy
  dsv3_ffn_v2.cuh:73-79 style).
- Device side: `extern __shared__ __align__(1024) char smem[]`; every address via
  `task_desc->smem_region_offset(REGION_*)` (extern_smem_align memory: 19 bare v2 sites
  once caused cross-task misalignment — never smaller than 1024).

TMEM (per b200-tmem-lifecycle-planner contract):
- Objects: ACC ×2 stages, f32 [128 lanes × MMA_N=16 cols] each → 32 cols; SFA 4 cols + SFB
  4 cols (UTCCP 4×32dp128bit layout, one col per UMMA_K=32 sub-tile). Raw 40 →
  **tcgen05.alloc 64 cols** (pow2 rounding, v1 :493-498). Layout: ACC0 [0..15], ACC1
  [16..31], SFA [32..35], SFB [36..39] (v1 kTmemStartColOfSFA/SFB scheme, :486-489).
- Write paths: UMMA → ACC; UTCCP (launcher-issued, elected lane) → SFA/SFB, re-written
  every K-stage, ordered against UMMA by tcgen05 program order within the launcher +
  `tcgen05.fence::after_thread_sync` after each mbar wait (v1 sequence, :990-1035).
- Read path: consumer `tcgen05.ld.sync.aligned.32x32b.x16` + `tcgen05.wait::ld` (linear
  :566-587 pattern; proven layout). Only col 0 is the real token — x1/x4 narrower reads are
  a Stage-5 micro-knob (pre-register ≤1-2µs, do not build first).
- Lifecycle: launcher all-32-lane alloc → taddr → BSF scratch; lane 0 arrives tmem_ready;
  **taddr cached in registers at alloc time** (linear :387 rule — scratch page may be
  released before dealloc); consumers read taddr once after tmem_ready; per-segment ACC
  handoff via mainloop/epilogue mbar ring ×2; all-128 arrive consumer_done → launcher
  `tcgen05.dealloc` 64 cols with the cached taddr. Alloc/dealloc same warp, sync.aligned,
  all 32 lanes (CLAUDE.md MLA invariant + linear pattern).

---

## 3. SEM ordinal table (op-private, ≤31; house-style §3 format)

Budget: MAX_DYNAMIC_SEMAPHORES=32, SEM_DEP_READY=0 runtime-reserved, op slots from
SEM_OP_BASE=1 → ≤31 (runtime_v2.cuh:366-376). STAGES=8 ⇒ **30 used**. (STAGES>8 would
overflow — raising stages requires folding W/B mbars or tag-flags; noted, not needed now.)

```
// Per-task SEM ordinals (relative to dyn_sem_base), STAGES=8:
//   [+0 ..+7 ]  W_tma_mbar   (count=1,      loader→launcher; TMA tx=16384B, async-arrived)
//   [+8 ..+15]  B_sf_mbar    (count=1,      loader→launcher; loader arrives ONCE per
//                             stage, but the arrive is only issued after (i) cpasync_wait<0>
//                             AND (ii) `fence.proxy.async.shared::cta` — the async-proxy
//                             fence is MANDATORY: cp.async-landed bytes are in the async
//                             proxy and a plain release-arrive alone does NOT publish them
//                             to the launcher's tcgen05 reads (Codex review 2026-07-15).
//                             The same release-arrive then also publishes the plain
//                             st.shared SFA/SFB splat stores. Alternative (v1's own B
//                             path): bind the cp.async group to the mbar via
//                             cp.async.mbarrier.arrive.noinc + a second plain arrive for
//                             the splats (count=2) — implementer picks ONE, auditor
//                             ledgers the choice.)
//   [+16..+23]  mma_mbar     (count=1,      launcher→loader, "stage K MMA done, refill ok";
//                             tcgen05.commit-arrived = async)
//   [+24..+25]  mainloop_mbar(count=1,      launcher→consumer, "ACC[stage] full";
//                             tcgen05.commit-arrived = async)
//   [+26..+27]  epilogue_mbar(count=4*32,   consumer→launcher, "ACC[stage] drained")
//   [+28]       tmem_ready   (count=1,      launcher→consumer, taddr published)
//   [+29]       consumer_done(count=4*32,   consumer→launcher, TMEM no longer read)
constexpr int NUM_OP_SEMS = 30;
```

- Justification for no tag-flags (W4): W4's tag-flags fixed a multi-role wedge in the
  *MAC-fold* tasks whose handshakes ride the consumer-only registration (no
  init_semaphores). This kernel is a full pipeline registration with controller
  init_semaphores — the reference mbar protocol + the §4 stale-arrival re-inits is the
  house idiom for it (linear_v2/v3 prove it in-runtime). Tag-flags remain the documented
  fallback if Stage-4 hits a controller-re-init wedge the re-inits don't cover.
- B_sf_mbar carries no expect_tx (cp.async, not TMA): loader drains its own cp.async group
  (`cpasync_wait<0>`) then does a release-arrive; the launcher's acquire-wait makes B and
  SF stores visible. v1's cpasync_barrier_arrive_noinc path is the equivalent precedent.
- init_semaphores (controller, lane 0, once per publish): mbar_init all 30 ordinals with
  the counts above + `fence.mbarrier_init.release.cluster` (task_register.cc:2035-2059
  pattern).

**Stale-arrival re-init ownership (async-arrived mbars only — the §2 house rule, single
most important protocol rule)**:

| mbar | async arriver | re-init owner at task start | placement guarantee |
|---|---|---|---|
| W_tma[0..7] | TMA byte delivery | **loader**, before its first TMA | linear :246-259 verbatim |
| mma[0..7] | launcher tcgen05.commit | **loader** (first waiter; re-init strictly before launcher's first commit — after page-wait, before first TMA) | linear :246-259 verbatim |
| mainloop[0..1] | launcher tcgen05.commit | **launcher lane 0**, before arriving tmem_ready | linear :391-410 verbatim |
| epilogue[0..1] | (thread-arrived, but re-init with mainloop for the prior-occupant 1-tile phase residue) | **launcher lane 0** | linear :399-409 verbatim |
| B_sf[0..7] | loader arrive (+ cp.async completion if the noinc variant is chosen) | **loader** — REQUIRED if the cp.async.mbarrier.arrive.noinc variant is chosen (async-arrived); with the fence.proxy.async variant it is thread-arrived only, re-init kept anyway (harmless, uniform) | with W re-inits |
| tmem_ready, consumer_done | thread-arrived within task lifetime | controller init only (linear precedent) | — |

Each re-init block ends with `fence.mbarrier_init.release.cluster`.

---

## 4. Stage count + per-role responsibilities

**STAGES = 8** (v1-proven; ≥6 floor per linear; 8 chosen because v1 hung at 4 with this
K-tile count and 8 is its production depth). ACC stages = 2. MMA_N = 16 (v1 constant; the
padded-token dimension). BLOCK_M = 128 weight rows/tile. bK = 128 (= GRP, one scale block
per K-stage — the invariant that makes block-scaled UMMA line up).

**K-iteration spaces**:
- W13 task (one (slot, n_tile) or shared (n_tile)): 56 K-stages, one segment, one ACC tile.
- W2 task (one n_tile): runtime-variable segment list = [slot 0..active_count-1] (4
  K-stages each, K=512) + [shared-down segment] (2 K-stages, K=256, weight 1.0, B = si_fp8,
  SFB = si_scale) → 4·active+2 ∈ [2..34] K-stages, active_count+1 ∈ [1..9] ACC segments
  cycling through the 2 ACC stages.

**Loader (W4)** — `elect_sync()` one thread, rest return (linear :200-202).
1. `prefetch.tensormap` the weight descriptor(s).
2. Stale-arrival re-init (table above) + fence.
3. **Dep-wait placement** — deviation from linear, forced by routing: the routed tasks'
   weight TMA coordinates need `meta` (expert id e_s), so weights CANNOT prefetch ahead of
   the dependency. Routed W13/W2 loader: `wait_task_dependency` FIRST (inline, before
   anything that reads meta), then read meta once, bounds-decide (slot ≥ active_count →
   bail path §below), then the stage loop. Shared-GU instance (ALWAYS_ACTIVE=1): keep the
   linear property — W TMAs start immediately, dep-wait inline once before the first
   B cp.async (linear :296-300). This asymmetry is a param, not a variant string hazard —
   it must appear in the emitted registration string (dedup footgun, wiring-recipe §4).
4. Per K-stage s (ring): `mbarrier_wait(mma[s], phase)` → splat SFA (w-scale[block(n0),
   k_tile] → one packed u32 replicated 128×, 64 uint4 stores) + SFB (a/i/si scale[k_tile]
   ditto) into the BSF slice → W TMA `tma_3d_load_l2`(dst=W_s, coords=(0, e·N_orig + n0,
   k), L2_EVICT_FIRST) + `mbarrier_arrive_expect_tx(W[s], 16384)` → B cp.async: 128B row 0
   from the activation vector (a_fp8+k·128 / i_fp8[slot]+k·128 / si_fp8+k·128; L2_EVICT_LAST
   class; 16B-aligned chunks from a 128B-aligned K-slice) → `cpasync_commit;
   cpasync_wait<0>` → **`fence.proxy.async.shared::cta`** (publishes the async-proxy
   cp.async bytes to the launcher's tcgen05/generic-proxy reads — a plain release-arrive
   alone does NOT cover cp.async data; Codex review 4a) → release-arrive B_sf[s].
   Rows 1..15 of every B slice are zero-filled ONCE at task start (before the stage loop,
   before any B_sf release);
   cp.async only ever rewrites row 0, so padding rows stay zero across the ring (zeros ×
   any UE8M0 scale = 0 — padded-column ACC garbage is never read; consumer uses col 0).
5. W2: the segment loop wraps the K-stage loop; segment order = slots ascending then
   shared — fixed order, deterministic.

**Launcher (W5)** — all 32 lanes enter.
1. Bounds/inactive fail → release-all-pages path (below) and return (linear :353-363).
2. `tcgen05.alloc` 64 cols → BSF scratch; cache taddr in registers (all lanes).
3. Lane 0: re-init mainloop+epilogue + fence + arrive tmem_ready (linear :390-412).
4. Elected lane, per segment: `mbarrier_wait(epilogue[acc], phase)` → per K-stage: wait
   W[s] + B_sf[s] → `tcgen05.fence::after_thread_sync` → UTCCP SFA slice → UTCCP SFB slice
   → 4× block-scaled UMMA (SM100_MMA_MXF8F6F4_SS; k_sub 0..3 advancing the smem descriptor
   lo-word by 32B/16, sfa_id/sfb_id=k_sub in the instruction descriptor; enable_d=0 on the
   segment's first sub-MMA only) → `tcgen05_commit(mma[s])` → advance ring. Segment end:
   `tcgen05_commit(mainloop[acc])`, flip acc stage. (v1 MMA-warp sequence :990-1035; UMMA
   issue is elect_one_sync-internal — never wrap it in an outer elect, v1 :1012-1015.)
5. Task end: lane 0 `mbarrier_wait(consumer_done, 0)`; all 32 `tcgen05.dealloc` (cached
   taddr). Page release at task end: **mirror linear exactly** — launcher lane-parallel
   blanket release :473-477 with `auto_consumer_finish=false` — subject to Q3 (the
   launcher-blanket + storer-per-stage double-arrival accounting must be confirmed against
   the runtime page protocol before Stage 2 writes it; copy whatever linear's proven
   combination is, verbatim).

**Consumer (W0-3)** — 128 threads; thread t owns output row n0 + t.
1. Lane 0 waits tmem_ready, `__syncwarp`, all read taddr from BSF scratch (linear :540-546).
2. Per segment: wait mainloop[acc] → `tcgen05.fence::after_thread_sync` → tcgen05.ld x16 +
   `tcgen05.wait::ld` → take col 0 → W13: `st.global.L1::no_allocate` f32 to
   y13[slot·1024 + n0 + t] (or sg[n0+t]); W2: `acc_f += ew_seg · col0` (ew from meta;
   shared segment ew=1.0) → arrive epilogue[acc].
3. W2 after last segment: `out[n0+t] = __float2bfloat16_rn(acc_f)` (chain T5's
   output-stationary determinism preserved: fixed segment order, fp32 register accum, one
   bf16 store, no atomics).
4. Every thread arrives consumer_done (count 4·32) — after its last tcgen05-related read.
5. No `__syncthreads`, no named barrier needed (mbars + `__syncwarp` only — quality bar §6;
   named-barrier ids 1/2/3/6 stay untouched).

**Storer (W6)** — passive per-stage page-release engine (linear :630-767 adapted):
elect 1; rides mma[s] parity in lockstep; `last_use[s] = (total_iters-1-s)/8 + 1` where
total_iters = 56 (W13) or 4·active+2 (W2 — computed from meta, so the storer body must be
ordered after the dep: give it the codegen dep-prefix, or an acquire on a
consumer-released flag; mirror linear's storer registration — Stage-3 verify). On a
stage's last fire release that stage's W page; BSF's 2 pages + any residue release in the
tail (linear's scratch-decrement pattern :754-766).

**Bounds-fail / inactive-slot path (every page-owning role)**: a routed W13 task whose
slot ≥ active_count (typical: 32 of 64 tasks dead at active=4) — after the dep-wait +
meta read, each role takes its bail path; every declared page must still be arrived
exactly the same number of times as the normal path (launcher :353-363 + storer :663-671
patterns; a bare `return` deadlocks the next slot occupant on page_ready — house-style §2
bounds-fail rule). W2 tasks are never inactive (shared-down segment always live, even at
active_count=0). Consumer/loader bail paths arrive their handshake mbars as needed so no
waiter wedges: on inactive, loader arrives nothing (no TMA issued), launcher skips alloc
and MMA but must NOT wait consumer_done unless consumers arrive it — define the inactive
protocol explicitly in the code: all four roles detect inactive independently from meta
(same branch), no cross-role wait is issued on the inactive path at all; pages are
released AND — hardening beyond linear's own bail path (which skips this and has not
wedged: loader :222-224 returns before the :246 re-init) — **each role still runs its §3
stale-arrival re-inits before bailing** (a few mbar_init instructions on a dead task,
upward-safe; adopted per Codex review 4b: relying on controller init_semaphores alone
leaves the documented post-init stray-arrival window open across the inactive publish).
The no-handshake bail + re-init-before-bail + the cross-publish stray scenario must be one
of the b200-mbarrier-protocol-auditor ledger rows at Stage 2.

**§1.1 dep-prefix assignment** (LETHAL invariant, wiring-recipe §4): consumer + launcher
get `emit_dep_wait_consumer_prefix`; loader does the dep inline (linear :2120-2123
precedent — here it is mandatory even for correctness because meta gates the coords);
storer per linear's registration. `auto_consumer_finish=false` iff non-consumer roles own
page release (linear :2150).

---

## 5. Task granularity — per-tile (the default; no grid-wide exception)

- **W13 routed**: task = (slot s ∈ 0..7, n_tile ∈ 0..7) → **64 tasks**; tile_idx =
  `task_desc->task_metadata.task_offset` (= bid.x — BOTH new enums must be added to the
  runtime.cc task_offset=bid.x list ~:571-635, the deadliest v2 footgun).
- **W13 shared**: second instance of the same task type (ALWAYS_ACTIVE=1, E=1, N=512) →
  **4 tasks** (output → sg).
- **W2**: task = n_tile ∈ 0..55 → **56 tasks** (slot segments internal — cross-slot
  weighted sum stays inside one task; keeps the chain's no-atomics output-stationary
  semantics AND avoids a per-slot-partials GMEM round-trip + combine task).
- Total 124 per-tile tasks/layer replacing the chain's 2 fat warp-strided GEMV ops. This
  IS the reference granularity the runtime round-robin can overlap cross-SM (plan Phase 1
  rationale) — no monotonic grid barrier, no num_tasks==num_workers assert, no
  skip_after_step0 scratch (nothing here persists across steps; y13/sg/i_fp8 are
  write-before-read per iter — same as the chain).
- **Why granularity and pipeline land as ONE item** (the Phase-1/Phase-2 merge): with the
  scalar body, per-tile ownership is a D6-class repartition (measured 4×-DEAD) and W1's
  tie-test predicts tie-or-regress on the re-tasking — the plan's own evidence map leaves
  no evidence-consistent intermediate. With the pipeline body, per-tile ownership is the
  natural tcgen05 tile decomposition (v1's own EXPERT_STRIDE × grid.y N-split, :110-119).
  Consequence for gates: this item carries BOTH the Phase-1 granularity watch-metric
  (dep-wait/wall regression → park/kill, §1) and the Phase-2 body target (v1-parity).
- Grid-wide-fused is NOT used → the §6 monotonic-barrier / skip_after_step0 /
  num_tasks==num_workers contract does not apply (stated per the Stage-1 checklist).

---

## 6. Evidence check table (per design choice)

| design choice | rests on | notes |
|---|---|---|
| Reference pipeline w/ TMA+tcgen05 for W13/W2 per-tile tasks | **U3** (probe; pre-registered Δ + kill in §1) + **W5** (v1 engine proven at M=1) + **U2** (direction) | the plan's Phase-2 candidate 2, verbatim |
| No per-tile *scalar* intermediate milestone | **D6** (partition of evenly-striped scalar body = 4×-DEAD) + **W1** (tie-test predicts tie/regress on re-tasking) | plan ambiguity resolved toward the evidence |
| Per-tile tasks, task_offset identity, cross-task events (no grid barrier) | **U2**; house-style §6 default | abandons the lockstep shape — plan caveat 2 says the win comes from abandoning it, not a better barrier |
| Op-private mbar protocol + stale-arrival re-inits (not tag-flags) | house-style §2/§3; linear_v2/v3 in-runtime proof; **W4** consulted and scoped to consumer-only-idiom handshakes | tag-flags = documented fallback |
| Coarse in-task protocol; no per-slot GMEM counters | **W3/D4** | D4's kill mechanism (cross-grid per-slot release) not reintroduced |
| MMA_N=16, STAGES=8, ACC=2, bK=128 | **W5** (v1 production constants, task_register.cc:3481-3494) | knobs (MMA_N=8, STAGES=12, narrower tcgen05.ld) deferred to Stage 5 with pre-registered ≤1-2µs each |
| Block-pow2 scales splat in loader (drop v1's scale warp + per-row scale tensors) | code ground truth (builder.py:3108-3128 `_w13_scale_pow2`; v1 UTCCP contract :74-88) | Q2 pins the pow2 assert; UE8M0 exact for pow2 |
| Consumer f32/bf16 epilogue, W2 in-register cross-segment sum | chain T5 design review (dsv3_ffn_v2.cuh:765-779) — determinism preserved | not bit-identical to the chain GEMV (MMA accumulation order differs) → math-changing gates, §7 |
| Chain small ops (rq/topk/silu) untouched, consumer-GEMV | plan Phase 3 guard-rail; **D3/D5/D8/D9** | out of this item's scope |
| Cold-L2 / TIER-1 measurement | m1 §4 rules | slowCTA @ grid=136, %globaltimer; no cudaEvent verdicts |

---

## 7. Correctness + measurement gates (Stage 4/5 contract)

Math-changing swap (accumulation order + scale application point differ from the chain
GEMV) ⇒ per the plan Phase-2 gate + decode-fold protocol:
1. Test-mode numeric in `tests/runtime_python/blackwell_v2/` (extend dsv3_ffn_harness /
   ffn_case_runner with the two pipe ops): cos ≥ 0.999, rel_max ≤ 3e-2, no NaN, vs the
   torch reference AND vs the chain GEMV output (v1-counterpart compare). Cases: active ∈
   {0, 1, 4, 8}, duplicate expert ids across ranks' filter, n_tile edges, pow2-scale assert.
2. §1.1/protocol static audit + **b200-mbarrier-protocol-auditor** ledger before Stage-2
   sign-off (every mbar: init count, arrivers, tx-bytes, waiters, phase evolution, re-init
   owner — §3/§4 tables are the input; the inactive-slot no-handshake path is a mandatory
   ledger row).
3. In-MPK `--layers 0-3` probe (env-gated path ON), then **multi-step iter ≥ 3**
   (iter-0-fine/iter-1-hang = persistent-state re-init class; 30 new mbars = the new
   stale-arrival surface). Watchdog armed (`MPK_V2_HANG_WATCHDOG_S`); never crash-loop.
4. TP8 full-61L: poison-fill gate on the replaced weight consumption (NaN-fill a w13/w2
   slice → output must poison ⇒ proves the pipe kernels actually consume; token-identity is
   NOT the gate on the nondeterministic TP8 path) + coherence-in-envelope.
5. Qwen3 regression untouched (additive task types; default build byte-identical —
   builder gate `MPK_DSV3_V2_FFN_PIPE=1` default-OFF; mega remains the v2 default).

Perf verdict (Stage 5): TIER-1 in-MPK %globaltimer slowCTA @ production grid (136), per-op
W13/W2 body vs (a) Phase-0 chain anchors, (b) the v1 12-57µs class, (c) mega FFN wall;
plus per-layer FFN wall + dep-wait watch metric. Kill/park thresholds in §1. Profiler:
buffer 120000·128, `scripts/v2_perfetto_export.py`.

---

## 8. Wiring sketch (Stage 3, orchestrator — for completeness, not this stage's work)

New: `TASK_DSV3_FFN_W13_PIPE_V2`, `TASK_DSV3_FFN_W2_PIPE_V2` (next-free enums; never
hardcode ids elsewhere) → `tasks/blackwell_v2/dsv3_ffn_gg_v2.cuh` + `dsv3_ffn_gg_v2_spec.h`
+ task_header.cuh include → task_register.cc register fns (pipeline role bodies + init_semaphores;
all behavior params in the emitted string) → graph.cc tuples (W13: 5 in [meta, a_fp8,
a_scale, w, w_scale] + 1 out; W2: 9 in + 1 out — final arity fixed at Stage 3 vs new_input
order) → runtime.cc task_offset list + task_type_to_name → persistent_kernel.py
`dsv3_ffn_w13_pipe_layer`/`dsv3_ffn_w2_pipe_layer` (num_tasks = 64/4/56) → builder use_v2
branch behind `MPK_DSV3_V2_FFN_PIPE` (default-OFF), which also requires the chain-path
selectability (plan Phase-1 bullet 1) as its host. Weight CUtensorMap attachment: reuse the
v1 fp8_group_gemm tma_2d emission incl. the `(E-1)·N_orig + N` GMEM_ROW expert-offset trick
(task_register.cc:3502-3521) — confirm the v2 input_tma_desc_ptrs population path (Q4).

## 9. Open questions that MUST be answered before Stage 2 dispatch

- **Q1 (blocking the verdict, not the build)**: Phase-0 anchors — per-op TIER-1 slowCTA of
  the CURRENT chain w13_gemv/w2_gemv (and mega wall) on the box, n≥3. The §1 kill threshold
  binds against these. Also re-pin the 52.2/14.2µs numbers with an on-disk trace.
- **Q2 (blocking correctness claims)**: assert all four scale packs are **positive,
  finite, exact pow2 within the UE8M0-representable exponent range** (not just
  mantissa==0.5 after frexp — exclude zero/inf/nan/exponent-overflow encodings; Codex
  bonus note) at harness load; if ANY fails, the UE8M0 conversion changes math beyond
  rounding → STOP and re-derive (the v1 TP<8 group-gemm path implies they hold; verify,
  don't assume).
- **Q3 (blocking Stage-2 storer/launcher code)**: read the runtime_v2.cuh page_finished
  protocol and confirm the per-page expected-arrival count under linear's
  launcher-blanket + storer-per-stage combination; copy the proven accounting verbatim.
- **Q4 (blocking Stage-3, affects Stage-2 signatures)**: the v2 mechanism that populates
  `task_desc->input_tma_desc_ptrs` for customized tasks (linear_v2's layer does it — read
  its persistent_kernel.py wrapper) and whether a 3D expert-weight desc or the v1 2D
  GMEM_ROW flattening is the path.
- **Q5 (sequencing)**: the pipe ops slot into the CHAIN graph; the chain-selectable-path
  env flag (plan Phase-1 bullet 1) must be wired in the same Stage 3. Confirm the
  orchestrator owns it there (recommended) rather than as a separate pre-loop.
- **Q6 (design freeze)**: W2 segment order = slots-ascending-then-shared is pinned for
  determinism; confirm the chain reference harness uses the same order for the
  v1-counterpart compare tolerance to be meaningful.

## 10. Codex design-review log (2026-07-15, default params, thread 019f668e)

Four design calls submitted: (1) Phase-1/Phase-2 merge argument — **SOUND** ("attribution
value exists academically, but not enough to justify building a predicted-regressing
configuration"); (2) B-padding zero-fill-once scheme — **SOUND** (conditions: zero-fill
before any B_sf release; 16B-aligned cp.async; pow2 assert excludes non-finite/zero —
all incorporated); (3) block-scale splat + skipped transpose — **SOUND** (condition: splat
the FULL 128-entry UTCCP buffer, not 16 — §4 says 128 ✓); (4) SEM/ordering — **FLAWED**
twice, both fixed in place: 4a cp.async async-proxy publication (→ `fence.proxy.async.
shared::cta` before the B_sf arrive, or the cp.async.mbarrier.arrive.noinc variant; §3/§4
updated); 4b inactive-slot bail (→ re-init-before-bail hardening; §4 updated). Bonus
flags: launcher-blanket + storer-per-stage double-arrival = Q3 (already open, confirmed
dangerous-until-resolved); storer dep/meta ordering (already in §4, Stage-3 verify);
sharper pow2 assert (→ Q2 updated).

## 10.5 APPENDIX — Stage-3 stub-wiring resolutions (2026-07-15, orchestrator)

Stage-3 stub landed (naive-scalar consumer body + protocol skeleton; ferret
replaces the bodies). Files: `tasks/blackwell_v2/dsv3_ffn_gg_v2.cuh` +
`dsv3_ffn_gg_v2_spec.h`; enums `TASK_DSV3_FFN_W13_PIPE_V2=356` /
`TASK_DSV3_FFN_W2_PIPE_V2=357`. §9 question resolutions:

- **Q2 (RESOLVED)**: the pow2/finite/UE8M0-range assert is implemented at
  BOTH consumption sites: `dsv3_ffn_harness.py::assert_ue8m0_pow2` (harness
  load, pipe cases) and the builder's `MPK_DSV3_V2_FFN_PIPE` branch
  (build time, all four packs). Check = positive + finite + `frexp` mantissa
  == 0.5 exactly + exponent k in [-127, 127]; failure raises (STOP). NOTE
  (TIER-1 prerequisite): the demo attaches the SHARED gate_up/down scale_inv
  RAW (no pow2 requant — only w13/w2 go through
  `_requantize_moe_fp8_for_pow2`); if a real checkpoint's shared scales are
  not pow2 the builder assert STOPS — requantize the shared weights for pow2
  before a TIER-1 run (do NOT tolerate).
- **Q3 (RESOLVED — linear's proven combination, adapted)**: page accounting
  is per-page parity keyed by instruction index; EVERY task must arrive
  EVERY page EXACTLY once (runtime_v2.cuh:472-491 + v2_role_codegen.cc). The
  wired combination: codegen loader page prefix (default ON — waits all 14
  pages, immediately arrives the pages the task does NOT use, i.e. 4 of 14
  here) + launcher task-end blanket over the pages the task USES
  (`task_uses_page`-gated — the gate is LOAD-BEARING: linear releases all 14
  ungated only because it uses 14/14) + `auto_consumer_finish=false` + EMPTY
  storer. The spec §4 per-stage storer release stays the documented
  ALTERNATIVE owner — TRANSFER by moving the release out of the launcher
  (.cuh-only), never both.
- **Q4 (RESOLVED — the population path, traced)**: `input_tma_desc_ptrs` is
  populated HOST-side at task-graph load: runtime.cc emits
  `create_tma_desc_by_task(task_desc)` calls into the generated test.cu
  gated by task-type ranges/lists (runtime.cc:1413-1460); that host fn
  (tma.cuh:2280) dispatches per task type →
  `create_tma_desc_for_tensor(task, tensor, param_id, tma_desc_id)`
  (tma.cuh:2255) → `fill_tma_desc_by_task` (the geometry switch) →
  cudaMalloc'd CUtensorMap stored in `tensor_desc.tma_desc_ptrs[k]`, copied
  to `task_desc.input_tma_desc_ptrs[i][k]` (runtime_header.h:465). linear_v2
  gets it implicitly via the 231..256 SM100-TMA enum range; our enums
  (356/357) are OUTSIDE it → explicit additions landed in (a) runtime.cc's
  call list, (b) tma.cuh `create_tma_desc_by_task` cases (W13: input 3; W2:
  inputs 5 and 7 — TWO descs), (c) tma.cuh `fill_tma_desc_by_task` case.
  **The path is the v1 2D GMEM_ROW flattening** (NOT a 3D expert desc):
  rows = (E-1)·orig_N + N (v2 tensors are unpartitioned ⇒ orig_N == N ⇒
  rows = E·N; expert e's tile row coordinate = e·N + n0), cols = K, u8
  128B-swizzle, smem box {128, 128} = one 16 KB W page — byte-compatible
  with the v1 fp8_group_gemm weight desc. Shared 2D weights ([N,K]) encode
  as E=1. The loader receives `CUtensorMap const *` by pointer.
- **Q5 (RESOLVED)**: the chain-selectable env flag is
  `MPK_DSV3_V2_FFN_PIPE=1`, owned by the builder's use_v2 branch
  (builder.py `_build_moe_mlp_ffn_full`): flag ON wires rmsnorm(existing) →
  router_quant → topk → W13 pipe (routed 64 + shared 4, meta fan-out) →
  silu_quant → W2 pipe (56) and returns before the mega; flag OFF (default)
  falls through to the mega — task graph byte-identical (verified by
  pre/post compile diff of the chain case).
- **Q6 (RESOLVED/FROZEN)**: W2 segment order = slots-ascending-then-shared,
  implemented in the stub consumer exactly in that order (f32 register
  accum, one bf16 store); the harness v1-counterpart compare
  (`ref_w2`) iterates slots in the same ascending order, so the
  tolerance-based compare is meaningful.
- **Q1 (OPEN — verdict anchor, not build)**: TIER-1 box anchors still
  pending (orchestrator). TIER-2 gate anchors are live-benched by
  gate/check.py at freeze in the same harness geometry.
- **Forced-routing note (gate case matrix)**: active∈{0,1,4,8} is forced via
  bias: targets [0,k) +1000, other locals [k,128) **−10** (NOT −1000 — the
  group top-2 sum would sink a lone target's group and silently yield
  active=0; found & fixed at Stage 3).

## 11. Self-audit vs house-style §7 quality bar

1. File-header design comment — specified (§0 tables + §4 roles feed it) ✓
2. Documented SEM ordinal table + constexpr ordinals — §3 ✓ (30 ≤ 31)
3. spec.h single source of truth: ordinals, make_smem_info, capacity + drift
   static_asserts — §2 ✓ (155,664 B / 10 pages / 9 regions)
4. Stale-arrival re-inits owned by arriving role — §3 table ✓ (linear-verbatim placements)
5. Bounds-fail releases every page exactly once; inactive-slot protocol defined and
   auditor-ledgered — §4 ✓
6. No __syncthreads/named-barrier in role bodies; no blockIdx (task_offset only) — §4/§5 ✓
7. Profiling hooks compile away; default build byte-identical (new additive task types;
   env-gated default-OFF path) — §7/§8 ✓
8. clang-format-15 at Stage 2 ✓ (n/a to this doc)
