# MASTER PLAN — DSv3 DECODE end-to-end on Runtime V2 (TP8 EP2 bs=1)

> ARCHIVED IN-SKILL COPY (the Phase-0 worked example this suite generalizes; original
> authored at `scratch/V2_DSV3_DECODE_MASTER_PLAN.md`, git-ignored and absent from fresh
> clones). Its plan was EXECUTED: full-61L DSv3 decode runs e2e on v2 (commit `e31b34dd`,
> opt-in `--use-v2`) — read it as "what a model plan doc looks like + which risks fired",
> not as open work. New-model plan docs mirror its structure (Phase 0 of SKILL.md).

Branch `dsv3-decode-clean` (repo-relative paths). This is the roadmap a series of
build agents executed. READ-ONLY planning + Codex cross-check done first; every
mechanism/edit-site below was verified against the actual files.

STATUS OF CROSS-CHECK: goal-shape (A/A), ordering, and the multi-rank-AR feasibility were
Codex-cross-checked (thread 019f3518) over 2 rounds, and the three load-bearing conclusions
were sent to `ablation-logic-reviewer` (verdict folded into the RISK section).

---

## 0. TL;DR (the decisions)

- **GOAL SHAPE = A/A (port BOTH fused megakernels as v2 megakernel-shape tasks).** Port
  `attn_block_megakernel_sm100` and `ffn_full_megakernel_sm100` as single v2 tasks using the
  **Form-2 "ffn_mega" pattern already proven** (136 co-resident tasks, `num_tasks==num_workers`,
  self-syncing via the kernel's existing in-op GMEM `grid_barrier`). Do NOT adopt the de-fused
  v2 chains for the first e2e: the de-fused **attn** chain is a measured **1.4–1.6× regression**
  vs the v1 fused (which beats SGLang), and the de-fused **FFN** chain's +7–9% single-GPU win is
  **not bankable** (measured with no AR, no attn co-residency, EP-unaware; commit `6c6a5825`
  measured the FFN megakernel-SHAPE as a **TIE** with the tuned chain — so fused-port loses ~0).
  Revisit de-fused-FFN as a *separate* perf lever AFTER e2e, with a real TP8 A/B.
- **MULTI-RANK AR ON V2 = feasible, MODERATE-HIGH risk (not a scheduler-surgery blocker).**
  The tile-AR body self-syncs across ranks via NVSHMEM team `pSync`/`sync_counter` INSIDE the
  body — it does **not** use the runtime's nvshmem-event edge (which v2 no-ops). v2 shares v1's
  full NVSHMEM init (symmetric heap + teams + `barrier_all`). The ONLY genuinely-new work is
  authoring a **v2-safe collective body** (the existing body uses block-wide `__syncthreads()`
  + `blockDim.x==256` loops; a v2 consumer role runs only 128 threads → naive reuse deadlocks/
  half-computes). Same hazard for `nvshmem_global_argmax`.
- **SMALLEST FIRST E2E = TP8 EP2, bs=1, `use_v2_runtime=True`, `--layers 3-3`** (the first MoE
  layer; skips dense layers 0–2 so no dense-path v2 tasks needed for bring-up). Single-GPU
  DSv3-fused-decode is INVALID as a first milestone (the fused kernels hard-assert
  `world_size==8` / `routed_tp_size==4`).
- **ORDER = leaf/tail wiring → TP2 AR micrograph → fused blocks (FFN before ATTN) → TP8 e2e.**
- **⚠ THE ONE CROSS-CUTTING TRAP (reviewer + Codex 019f3535, applies to AR + argmax + BOTH fused
  megas): "wrap the existing v1 body" is WRONG.** All four bodies run block-wide
  `__syncthreads()` + `blockDim.x==256`-strided loops that are correct only on 256 threads; a v2
  consumer role gives them **128 threads** (W0–3). None is a mechanical wrap — each needs a
  128-thread-safe sync/stride rewrite (or an explicit multi-role fan-out that also accounts for
  the controller warp). And because a bodyless/half-ported consumer task **wedges the box**
  (D-state zombies, no graceful degrade), the M0→M3 sequence must NEVER register a live-path task
  without a complete v2 consumer body — add a graph-build guard (§5.2).

---

## 1. The machinery, verified (what a v2 task author must satisfy)

### 1.1 The 5 fixed roles + the no-fallback / deadlock trap
- v2 warps are FIXED per 256-thread worker: **W0–3 = consumer (128 threads)**, W4 loader,
  W5 launcher, W6 storer, W7 controller (`runtime_v2.cuh:299-306`, dispatch `:1171-1193`).
- Role dispatch is generated per-role as `switch(task_type){ case X:...; default:break; }`
  (`src/kernel/v2_role_codegen.cc:185-207`). A task_type with **no body for a role** →
  `default:break` → that role's warps do nothing but the runtime STILL marks the instruction
  FINISHED (`runtime_v2.cuh:734-777`). v1 dispatch is also skipped for v2 task types
  (`runtime.cc:2268-2272`), so **there is NO v1 fallback** — a v1-only task in a v2 graph is a
  silent no-op.
- **THE LETHAL CASE = a missing CONSUMER body.** The cross-SM dep-wait + the per-slot
  `SEM_DEP_READY` arrival live inside `consumer_dep_prefix` (`runtime_v2.cuh:609-656`, arrive
  at `:646`). A task with no consumer body never arrives `SEM_DEP_READY` → the next task
  reusing that ring slot spins forever in its own `consumer_dep_prefix` mbar_wait (`:655`) →
  **silent deadlock**. `task_register.cc:56` documents the invariant: *"Works because EVERY v2
  task type runs this prefix."* ⇒ **every new v2 task MUST provide a consumer body that begins
  with `emit_dep_wait_consumer_prefix`** even if the real compute lives in helper roles.
- **7-MAC-warp fan-out (`multi_role`):** a task can replicate its compute body into
  loader/launcher/storer roles (`task_register.cc:8848-8859`, and the ffn_mega example
  `:9384-9391`) to run compute on 7 warps instead of 4. Param-gated `nwarps ∈ {4,7}`.

### 1.2 The static per-SM plan (round-robin, per-rank)
- `python/mirage/mpk/v2_task_schedule.py::build_v2_worker_task_queues`: walks `all_events`,
  keeps task-pushing events, round-robins each event's `[first_task_id,last_task_id)` range onto
  workers with a **continuous cursor across events**, prepends task 1 to worker 0. **C++ twin**
  `build_v2_plan` in `persistent_kernel_v2.cuh:50-150` must stay bit-identical (documented at
  both sites). Inputs: only event structure + `num_workers`. It has **no** per-task grid/role/
  smem awareness.
- Runs on a **per-RANK graph**: `generate_task_graph(num_gpus=world_size, my_gpu_id=rank)` (the
  same graph-gen v1 uses) runs BEFORE v2 rescheduling. Multi-rank is orthogonal to the v2
  scheduler — v2 just redistributes THIS rank's tasks onto its 136 SMs.

### 1.3 Per-CTA metadata is BAKED PER-INSTANCE at build time (the key reuse fact)
- Each grid instance `(bid.x,bid.y,bid.z)` becomes a SEPARATE `FullTaskDesc` in `all_tasks[]`,
  with its per-CTA metadata baked by `register_mugraph` in `runtime.cc` (keyed by `bid`).
- The v2 controller `cp.async`-copies `all_tasks[task_pos]` (metadata included) into the
  worker's ring slot (`runtime_v2.cuh` controller ~`:964-1035`). So **`task_offset` /
  `merge_task_offset` / `request_id` travel in the TaskDesc and work identically in v1 & v2.**
- `TaskMetadata` is a **union** (`runtime_header.h:385-398`): `task_offset` (offset 0, "nvshmem
  team mapping" — used by AR/argmax **and by the v2 chains/mega**) vs `merge_task_offset`
  (offset 4, split-kv/fused-mega CTA id — used by the **v1** fused megas). The v1 FFN mega uses
  `merge_task_offset=bid.x` (`runtime.cc:522-525`); the **v2** ffn_mega uses
  `task_offset=bid.x` (`runtime.cc:595-598`). **⇒ FOOTGUN: any NEW v2 fused-mega task_type MUST
  be added to the `task_offset = bid.x` block at `runtime.cc:588-609`, or it reads a garbage
  CTA index → grid_barrier deadlock.** (This is a "paths you touch together" edit that is easy
  to miss.)

### 1.4 Region-SMEM planner + the `_spec.h` contract
- 14 pages × 16 KB; usable `CAPACITY_BYTES = 225*1024 - 6*1024 = 224256`
  (`v2_smem_planner.py:4-6`). SM100a v2 builds bump `MAX_DYNAMIC_SHARED_MEMORY_SIZE` to 225 KB
  (`runtime_header.h:52-62`, `#ifdef USE_RUNTIME_V2`). Overflow caught at planner-time (Python
  raise) and/or a `_spec.h` `static_assert(<= PLANNER_CAPACITY_BYTES)`.
- A task declares its SMEM as **regions** via a host-safe `_spec.h` `make_smem_info(...)` that
  push_backs `{name,size,alignment,page_count(-1=auto),can_pack,release_step,contiguous}` in
  **ordinal order**; the device `.cuh` addresses region N via `task_desc->smem_region_offset(N)`
  — the push_back order IS the positional contract (example `rmsnorm_v2_spec.h:32-36,63-81`;
  richer `linear_sm100_v2_spec.h`, `dsv3_ffn_v2_spec.h`).

### 1.5 Intra-task cross-warp sync = the tag-flag protocol (the liveness fix)
- The op-private parity mbarriers (`SEM_OP_BASE..`) needed controller re-init and caused a
  deterministic multi-iteration wedge (commit `f624d75a`). The fix is **monotonic u64 tag flags
  in task SMEM** (`dsv3_ffn_v2.cuh:111-203`): a salted bijection of the per-SM instruction
  sequence (`(instruction_index+1)*0x9E3779B97F4A7C15`) that stale bytes can never satisfy;
  `sync_tag==0` compiles the handshake out (the 4-warp path). **⇒ new multi-warp v2 tasks
  should use tag-flags, not op-private mbarriers, for cross-warp sync.**

### 1.6 Megakernel-shape hosting is PROVEN (the goal-shape enabler)
- Commit `6c6a5825`: the v2 runtime CAN host a true megakernel-shape op — 136 tasks ==
  num_workers, one per worker, co-resident, self-syncing via **in-op GMEM atomic barriers**,
  bit-correct, no deadlock; `assert_mega_coresidency` machine-verifies each 136-task chunk maps
  to 136 distinct workers (the continuous round-robin cursor guarantees this for any contiguous
  num_workers-sized range). Working prototype = `ffn_mega_task_impl` (`dsv3_ffn_v2.cuh:1708-2110`,
  register `task_register.cc:9343-9399`). **Hard contract: `num_tasks == num_workers`**
  (host-assert `persistent_kernel.py:4214/4261`). There is NO `grid.sync()` in a v2 task — the
  whole-grid barrier is a hand-rolled GMEM count-barrier in a scratch tensor.

### 1.7 Multi-rank / NVSHMEM in v2 — the precise truth
- v2's runtime nvshmem-**event** edge is a **no-op**: `wait_task_dependency`/
  `task_dependency_ready`/`trigger_task_event` early-return on `is_nvshmem_event`
  (`runtime_v2.cuh:571/661/675`). v1 instead does a real `nvshmem_signal_wait_until`
  (`persistent_kernel.cuh:976`) — but that path is for the `TASK_NVSHMEM_ALLGATHER_STRIDED_PUT`
  producer/consumer edge, **NOT for tile-AR**.
- **tile-AR self-syncs across ranks INSIDE its body**: `mpkar_wait_until_ge(pSync+from_nbr,
  counter[0])` (`allreduce.cuh:192/248/299`), where `counter[0]` = NVSHMEM team
  `sync_counter[2*team_idx]` (`allreduce.cuh:52`) and `pSync` = NVSHMEM `psync_pool`
  (`allreduce.cuh:62`); it signals peers, waits, increments the team counter (`:177/:199`).
  Registration passes `runtime_config.nvshmem_teams` + `task_offset` (`task_register.cc:5275/
  5282`), runtime-version-agnostic. ⇒ **the nvshmem-event no-op is HARMLESS for tile-AR.**
- v2 **shares** v1's NVSHMEM init: `init_persistent_kernel` (nvshmem_malloc + barrier_all +
  allocate_nvshmem_teams, `persistent_kernel.cuh:1431/1578/1584/1832`) runs BEFORE
  `init_persistent_kernel_v2` (`persistent_kernel_v2.cuh:214`, and `persistent_kernel.py` calls
  `init_v2_func` after the shared init list). v2 launch is a **plain**
  `worker_v2_kernel<<<136>>>` (`runtime_v2.cuh:1219`) — fine, because each of the 8 ranks
  launches its own persistent kernel under mpirun and 136 blocks == SM count ⇒ co-resident.
- **THE ONE REAL BLOCKER for AR/argmax: block-wide sync mismatch.** The bodies use
  `__syncthreads()` + `blockDim.x==256` loops (`allreduce.cuh:196/1254`,
  `nvshmem_argmax_sm100.cuh:85`). A v2 consumer role runs only W0–3 (128 threads). A naive
  consumer-only port → `__syncthreads()` waits for 256 → **deadlock**, and 128-thread striding
  → half-computed reduce. **⇒ the AR v2 body must either (a) use the `multi_role` fan-out so all
  7 role warps (224 threads) run the body + replace `__syncthreads()` with a v2-safe
  named-barrier / tag-flag over the participating warps, or (b) restride to the actual active
  thread count.** This is the single hardest part of the AR port.
- **Cross-rank team-epoch alignment holds — but for a SUBTLE reason (reviewer + Codex 019f3535
  correction; do NOT justify it as "no rank runs ahead").** v2's per-iteration barrier
  (`runtime_v2.cuh:895/953/1066`) is **INTRA-rank, not cross-rank** — a fast rank CAN reach the
  next AR and block at the NVSHMEM team barrier. That's harmless for the team `sync_counter`
  scheme ONLY if (a) no rank epoch-skips and (b) no two same-team AR invocations are ever
  concurrent. What actually makes epochs align: all 8 ranks build an IDENTICAL graph (same
  builder, layer count, bs=1 deterministic control flow, unconditional AR) ⇒ identical AR-call
  counts in identical order ⇒ epochs line up. **Non-negotiables the AR port MUST enforce:** (1)
  NO rank-local early-return before the team rendezvous (a rank that skips its AR desyncs the
  epoch → deadlock/stale); (2) task deps must prevent overlapping same-team invocations across
  the two per-layer ARs / argmax / decode iters; (3) the §1.1 guard — never let a rank silently
  no-op a task. This is the same alignment v1 achieves, but the mechanism is
  identical-graph-determinism, not the v2 iter-barrier.

---

## 2. The DSv3 TP8 EP2 bs=1 DECODE task set (what must have a v2 variant)

Per-layer sequence for a **MoE layer** (layers 3–60, the steady state), in dep order:

| # | task type (v1) | builder call | v1 `.cuh` | v2 status | port kind |
|---|---|---|---|---|---|
| 1 | `tensor_init` (attn scratch) | `builder.py:1792` | `blackwell/tensor_init.cuh` | **ABSENT** (`.cuh` orphan exists, no enum/reg) | role-split (trivial) |
| 2 | `attn_block_megakernel_sm100` | `builder.py:1843` | `blackwell/attn_block_megakernel_sm100.cuh` | **ABSENT** | **megakernel-shape (Form-2)** |
| 3 | `nvshmem_tile_allreduce` (+`_with_residual`) | `builder.py:1869` | `blackwell/allreduce.cuh` | **ABSENT** | **collective (v2-safe sync rewrite)** |
| 4 | `rmsnorm_hopper` (post-attn) | `builder.py:3359` | `hopper/rmsnorm_hopper.cuh` | **PRESENT** (`TASK_RMS_NORM_HOPPER_V2=326`) | reuse |
| 5 | `tensor_init` (FFN scratch) | `builder.py:2931` | `blackwell/tensor_init.cuh` | **ABSENT** | role-split (trivial) |
| 6 | `ffn_full_megakernel_sm100` | `builder.py:2944` | `blackwell/ffn_full_megakernel_sm100.cuh` | **ABSENT** as such (de-fused chain exists) | **megakernel-shape (Form-2)** |
| 7 | `nvshmem_tile_allreduce` (+resid) | `builder.py:3386` | `blackwell/allreduce.cuh` | **ABSENT** | (same task as #3) |

Once-per-token head/tail:
| # | task type | builder call | v1 `.cuh` | v2 status | port kind |
|---|---|---|---|---|---|
| H | `embedding` | `builder.py:3454` | `ampere/embedding.cuh` | **PRESENT** (`TASK_EMBEDDING_V2=328`) | reuse |
| T1 | `rmsnorm_hopper` (final norm) | `builder.py:3488` | `hopper/rmsnorm_hopper.cuh` | **PRESENT** | reuse |
| T2 | `linear` (lm_head) | `builder.py:3551` | `ampere/linear.cuh` | **PRESENT as `linear_v2/v3`** but DSv3 calls base `linear_layer` | **re-route** DSv3 tail to v2 linear |
| T3a | `argmax_partial_sm100` | `builder.py:3585` | `blackwell/argmax_sm100.cuh` | **PRESENT** (`330`) | reuse |
| T3b | `nvshmem_global_argmax` (vocab-parallel default) | `builder.py:3603` | `blackwell/nvshmem_argmax_sm100.cuh` | **ABSENT** | **collective (v2-safe sync rewrite)** |
| T3b' | `argmax_reduce_sm100` (replicated-lm_head alt) | `builder.py:3627` | `blackwell/argmax_sm100.cuh` | **PRESENT** (`331`) | reuse (only if `--disable-vocab-parallel-lm-head`) |

Notes:
- At TP8 the residual add is FUSED into the tile-AR (`_with_residual`) — `elementwise_add` does
  NOT fire on the TP8 decode path (only TP=1 / diagnostics). So `elementwise_add_v2` is
  **NOT** on the critical path (deprioritize; author only if a TP=1 bring-up needs it).
- `fused_rmsnorm_quantize` (`dsv3_tasks.fused_rmsnorm_quantize_fp8_layer`, builder.py:736/816)
  is part of the **per-task attention chain** (the prefill/compat fallback), NOT the decode
  megakernel path — the attn megakernel folds RMSNorm+quantize internally (Phase-0). So
  `fused_rmsnorm_quantize_v2` is **NOT needed** for the megakernel-shape (Option A) decode path.
  (It WOULD be needed only if you chose Option B for attn — which we are not.)
- **Master V2-ABSENT set to author (Option A decode path):** `attn_block_megakernel_v2`,
  `ffn_full_megakernel_v2`, `nvshmem_tile_allreduce_v2` (+`_with_residual`),
  `nvshmem_global_argmax_v2`, `tensor_init_v2` (wire the orphan `.cuh`). Plus the tail re-route
  T2. That's **5 new tasks + 1 re-route** for the steady-state MoE decode path.
- Dense layers 0–2 (unfused: `quantize_fp8_f32scale_sm100` / `fp8_gemm_dense_finen_sm100` /
  `silu_mul`) are OFF the first-e2e path (we start at `--layers 3-3`). If full 61-layer coverage
  is later required either (a) author those 3 dense-path v2 tasks, or (b) set
  `MPK_DSV3_DENSE_MLP_MEGAKERNEL=1` and author `dsv3_dense_mlp_fused_v2` (one megakernel-shape
  task, same Form-2 pattern). Defer this to after e2e (see Milestone M5).

---

## 3. Per-task v2 specs (for the build agents)

For EACH task below: **roles / SMEM regions / sync / correctness reference / validate step.**
Every task's consumer body MUST start with `emit_dep_wait_consumer_prefix` (§1.1). Every new
`_v2` task needs the full "add-a-task" surface (enum in `runtime_header.h`, `register_*_v2_task`
in `task_register.cc`, dispatch in `graph.cc`, `task_type_to_name` + metadata in `runtime.cc`,
the `.cuh`/`_spec.h`, the `persistent_kernel.py` wrapper's `"..._v2" if self.use_v2_runtime`
switch, and — for fused-megas — the `task_offset=bid.x` line at `runtime.cc:588-609`).

### T-A. `tensor_init_v2` (LEAF, trivial) — do FIRST
- **Roles:** consumer-only (with dep-prefix). No loader/launcher/storer. It's a memset-to-zero
  of a scratch tensor with a dep-only dummy input.
- **SMEM:** none (0 regions) — `_default_plan` path; but still needs a `_spec.h` returning an
  empty `TaskSmemInfo` OR reuse the zero-region convention.
- **Sync:** none (independent per-element).
- **Correctness ref:** the v1 `tensor_init.cuh` (`tensor_init_zero_sm100_task_impl`) — must
  zero the identical byte range; honor `skip_after_step0` semantics if used (the decode path
  sets `skip_after_step0=True`, builder.py:1799 — but for a v2 megakernel-shape port the barrier
  self-maintains, so step-0 zero is the safe first cut; keep skip as a later lever).
- **Note:** `tensor_init_v2.cuh` EXISTS as a file but has NO enum/registration (grep empty) —
  wire it (enum + register + graph dispatch + `task_header.cuh` include + spec).
- **Validate:** blackwell_v2 correctness harness — add a `tensor_init` op branch + a
  zeros-reference; PASS = output all-zero (bit-exact).

### T-B. `nvshmem_tile_allreduce_v2` (+ `_with_residual`) (COLLECTIVE) — the hard one
- **Roles:** **multi_role fan-out** so the AR runs on 7 warps (224 threads) — OR consumer-only
  with a 128-thread restride. The body's `__syncthreads()` MUST be replaced by a v2-safe barrier
  over exactly the participating warps (a named `bar.sync N, <threads>` or a tag-flag), and the
  `blockDim.x`-strided loops re-based on the actual active thread count (`allreduce.cuh:196/1254`).
  This is the crux of the port — do NOT paste the body into a consumer role unchanged.
- **SMEM:** the AR body's SMEM (tile staging) — declare regions matching the v1 kernel's smem
  footprint; likely ≤1–2 pages (AR tile is small at hidden=7168/128 per CTA).
- **Sync:** cross-rank = the body's own `mpkar_wait_until_ge` on NVSHMEM team pSync (unchanged,
  reuse verbatim). Intra-task cross-warp = tag-flag / named barrier (the replacement for
  `__syncthreads()`).
- **Metadata:** needs `task_offset` baked (nvshmem team CTA mapping). `task_offset` is populated
  PER-GRID-INSTANCE inside `register_mugraph` (`runtime.cc:573-575`, `bid.x + bid.y*gx + ...`) at
  graph-build time — BEFORE v2 rescheduling. **CRITICAL interaction to VERIFY (not assume):** for
  this to be correct under v2, the AR must be registered as a grid of **N single-CTA task
  instances** (the AR grid is `hidden_size/128` CTAs), so each expanded `FullTaskDesc` carries a
  distinct `task_offset` and v2's round-robin distributes those N instances across N workers. If
  the AR were instead one wide-block task, `task_offset` wouldn't distinguish CTAs. The v1 AR
  already sets `task_offset` (`runtime.cc:571`); confirm the new v2 type is added to that block
  AND that the grid-expansion produces N distinct-`task_offset` instances the round-robin can
  spread (this is the one non-obvious multi-rank/v2 interaction — the TP2 micrograph must assert
  each CTA saw its own `task_offset`).
- **Buffers:** the AR in/out must be `nvshmem_tensor` io_category (already so in the builder,
  `builder.py:1453-1465`) — unchanged.
- **Correctness ref:** the v1 `allreduce.cuh` on the SAME symmetric buffers. The residual
  variant adds `self.x` exactly once post-reduce.
- **Validate:** a **TP2 micrograph** (2 ranks under mpirun): each rank writes a known vector to
  a symmetric buffer, run `nvshmem_tile_allreduce_v2`, assert the sum matches on both ranks
  (bit-exact). This is the FIRST multi-rank v2 test in the tree — it also proves iteration-
  lockstep + team-epoch alignment end-to-end. THEN re-validate at TP8.

### T-C. `nvshmem_global_argmax_v2` (COLLECTIVE) — same hazard as T-B
- **Roles / sync:** same block-wide-sync hazard (`nvshmem_argmax_sm100.cuh:85` uses
  `__syncthreads()` + `mpkar_sync_block`); apply the same v2-safe rewrite.
- **Metadata:** `task_offset` (nvshmem team). Grid = `lm_head_workers`.
- **Correctness ref:** v1 `nvshmem_global_argmax_from_partials_bf16` — cross-rank argmax over
  vocab shards. **De-risk option:** for the very first e2e, pass `--disable-vocab-parallel-lm-head`
  to route the tail through the **already-present** `argmax_reduce_sm100_v2` (`331`) + replicated
  lm_head, so T-C is NOT on the critical path for M3. Author T-C for M4 (restore vocab-parallel).
- **Validate:** TP2 micrograph (like T-B) once authored.

### T-D. `ffn_full_megakernel_v2` (MEGAKERNEL-SHAPE, Form-2) — do BEFORE attn
- **Roles:** consumer + `multi_role` (7-warp) exactly like `ffn_mega_task_impl`
  (`task_register.cc:9343-9399`) — that register fn is the **template**; the difference is the
  body calls the v1 `ffn_full_megakernel_sm100.cuh` kernel logic instead of the de-fused chain.
  Two viable routes:
  - **(D1) Port the existing v1 `ffn_full_megakernel_sm100.cuh` body** as the v2 task impl:
    replace its `ffn_full_grid_barrier(NUM_WORKERS)` calls (`ffn_full...cuh:2138/2678/3029`) with
    the **same GMEM count-barrier the ffn_mega Form-2 uses** (a scratch-tensor atomic barrier
    keyed on `num_tasks==num_workers`), and thread `task_offset` in place of `blockIdx.x`.
    **NOT a mechanical wrap (reviewer + Codex 019f3535 correction):** the v1 body runs on 256
    threads (`blockDim.x==256`) with block-wide `__syncthreads()`, but a v2 consumer role gives
    it only **128 threads** (W0–3). So the port needs the SAME 128-thread audit as the AR (§1.7):
    either restride to 128 + replace `__syncthreads()` with a v2-safe barrier, OR use the 7-warp
    `multi_role` fan-out (224 threads, still misses the controller warp — the barrier participant
    count must account for that). **Keep the COARSE whole-grid barrier** — the per-slot
    fine-grained-release variant REGRESSED +11–17% (memory
    `feedback_ffn_megakernel_fg_counter_regress`); do NOT "improve" it to per-slot during the
    port. The once-init barrier scratch must survive v2's per-iteration cadence (the mega's
    barrier self-maintains, but verify the step-0 zero + `num_tasks==136` hold every decode iter).
  - **(D2, if D1 fights the role model) Reuse `ffn_mega_task_impl` directly** — it already IS a
    full FFN megakernel-shape v2 task (rmsnorm→quant→router→2 all-to-alls→W13→silu→W2). Commit
    `6c6a5825` proved it TIES the tuned chain and is bit-exact. If D2's math == the production FFN
    (verify EP-locality: `ffn_mega` params `les/nle` = local_expert_start / num_local_experts —
    must be set to the TP8-EP2 per-rank slice), **D2 may be the fastest path to e2e** (the task
    already exists and passes). Prefer D2 if the EP-locality + shared-expert are covered; else D1.
- **SMEM:** reuse the v1 kernel's smem footprint (declare as regions ≤14 pages; the FFN mega is
  smem-heavy — verify against the 224 KB budget; the builder already sizes
  `ffn_full_megakernel_sm100::SCRATCH_BYTES`, builder.py:68).
- **Metadata:** `task_offset = bid.x` (ADD the new type to `runtime.cc:588-609`).
- **Correctness ref:** the v1 `ffn_full_megakernel_sm100` at TP8 EP2 (bit-match on identical
  inputs, the way `dsv3_ffn_harness` already does for the chain).
- **Validate:** TP8-shaped per-task harness (M=1, EP2 slice, num_active≈4) — bit-match vs v1.

### T-E. `attn_block_megakernel_v2` (MEGAKERNEL-SHAPE, Form-2) — the perf-critical port
- **Roles:** consumer + `multi_role`, same pattern as T-D. Body = the v1
  `attn_block_megakernel_sm100.cuh` logic with its `attn_grid_barrier(ATTN_NUM_WORKERS=136)`
  (`attn_block_megakernel_sm100.cuh:71/117`) replaced by the GMEM count-barrier (`num_tasks==
  num_workers==136`), and `blockIdx.x`→`task_offset`. **SAME 128-thread audit as T-D/AR
  (reviewer correction):** the v1 body runs on 256 threads (`block_dim=(256,1,1)`, builder.py:
  1862) with block-wide `__syncthreads()` + the CuTe `bar.sync 1,128` named-barrier pattern
  (the MLA `tcgen05.alloc` warp-0 sync — do NOT convert to `__syncthreads`, see CLAUDE.md
  invariant); a v2 consumer role gives 128 threads (W0–3). Audit every block-wide sync + strided
  loop for the actual active-thread count; the CuTe `bar.sync 1,128` warp-0 sync may map cleanly
  onto the 128-thread consumer group (a potential advantage over the 256-thread FFN body).
- **SMEM:** the attn mega's scratch (`ATTN_BLOCK_MEGAKERNEL_SCRATCH_BYTES`, builder.py:91) as a
  region set; verify ≤224 KB.
- **Metadata:** `task_offset = bid.x` (ADD to `runtime.cc:588-609`). The decode position is read
  from `runtime_config.step[0]` inside the kernel (builder.py:1667), unchanged.
- **Correctness ref:** v1 `attn_block_megakernel_sm100` at TP8 (bit-match on identical inputs,
  the way `dsv3_attn_harness` does for the chain — but here match the FUSED v1, not the chain).
  The TP-RowParallel o_proj: the kernel writes a residual-FREE partial and the AR (#3) sums +
  adds residual once (builder.py:1802-1826) — the v2 port MUST preserve the zero-residual-buffer
  binding so the AR does the combine (do NOT re-enable the fused-residual epilogue at TP8).
- **Validate:** TP8-shaped per-task harness — bit-match vs v1 fused megakernel.

### T-F. DSv3 tail linear re-route (RE-ROUTE, not a new task)
- The generic `linear_v2`/`linear_v3` already exist (`task_register.cc:1954/2110`,
  `runtime_header.h:164-166`) and are Qwen3-proven under `--use-v2`. DSv3 currently calls base
  `linear_layer` for lm_head (builder.py:3551). Re-route to `linear_layer_v3` under
  `use_v2_runtime` (mirror the qwen3 demo pattern demo/qwen3/demo.py:544/711). **Caveat:**
  `linear_v2/v3` assert `output.dim(0) <= 16` (BLOCK_N=16) — fine for bs=1 (M=1), but note it.

---

## 4. Integration path (builder plumb + demo flag + milestones)

### 4.1 The plumb (3 mechanical edits, mirrors qwen3)
1. **Demo flag:** add `--use-v2` argparse to `demo/deepseek_v3/demo.py` (copy qwen3
   demo.py:127) and pass `use_v2_runtime=args.use_v2` into the `mi.PersistentKernel(...)`
   constructor at **demo.py:449**. `PersistentKernel.__init__` already propagates it to
   `self.use_v2_runtime` (persistent_kernel.py:530) → sets the `USE_RUNTIME_V2` compile flag
   (persistent_kernel.py:413) → codegen emits the v2 role dispatch.
2. **Task-graph post-process:** after `generate_task_graph(num_gpus=world_size, my_gpu_id=rank)`
   in the DSv3 demo, add the SAME 3-line v2 block qwen3 uses (demo/qwen3/demo.py:845-853):
   `build_v2_worker_task_queues(task_graph, mpk.num_workers)` then
   `add_v2_region_smem_plan(...)`. This is model-agnostic.
3. **Builder task-name switching:** at each DSv3-specific builder call site, select the v2 task
   name under `use_v2_runtime` (mirror how persistent_kernel.py already does
   `"..._v2" if self.use_v2_runtime else "..."` for generic tasks). The DSv3 wrappers live in
   `python/mirage/mpk/models/deepseek_v3/tasks.py` (attn mega, ffn mega, global_argmax) — add
   the v2 branch there and/or in the `persistent_kernel.py` wrappers the builder calls.

### 4.1.1 CONCRETE plumb map (main-thread recon 2026-07-05 — executable checklist)
Exact sites, and — importantly — which plumbs AUTO-RESOLVE (no edit) because builds #1/#2
already put the `use_v2_runtime` switch in the wrapper the builder calls:
- **AUTO-SWITCH (NO edit needed):** `tensor_init` (build #2 → persistent_kernel.py:2241
  `tensor_init_layer` self-selects `"tensor_init_v2"`), AR (build #1 → multigpu.py
  `allreduce_layer` self-selects), rmsnorm/embed/argmax_partial/argmax_reduce (existing v2
  switches, reachability confirmed no-gap by build #2). ⇒ these fire under `use_v2_runtime`
  automatically once the demo sets the flag.
- **EDIT 1 — demo flag (`demo/deepseek_v3/demo.py`):** add `--use-v2` argparse (near
  `--disable-vocab-parallel-lm-head` at :129); add `use_v2_runtime=args.use_v2` kwarg to the
  `mi.PersistentKernel(` ctor at **:449**; add imports (mirror qwen3 demo.py:11-12:
  `from mirage.mpk.persistent_kernel import add_v2_region_smem_plan`,
  `from mirage.mpk.v2_task_schedule import build_v2_worker_task_queues`).
- **EDIT 2 — task-graph post-process (`demo/deepseek_v3/demo.py`):** after the DSv3
  `generate_task_graph(...)` call, add the qwen3 v2 block (qwen3 demo.py:850-853):
  `if args.use_v2: task_graph["v2_worker_task_queues"] = build_v2_worker_task_queues(task_graph,
  mpk.num_workers); add_v2_region_smem_plan(...)`. Model-agnostic.
- **EDIT 3 — tasks.py attn v2 branch (`deepseek_v3/tasks.py:475` `attn_block_megakernel_layer`,
  registers `"attn_block_megakernel_sm100"` at :523):** add a `pk.use_v2_runtime` branch that
  registers `"attn_block_megakernel_v2"` (via T-E's `dsv3_attn_mega_layer` wrapper). Mirror the
  FFN edit below.
- **EDIT 4 — tasks.py FFN v2 branch = T-D (`deepseek_v3/tasks.py:371`
  `ffn_full_megakernel_layer`, registers v1 `"ffn_full_megakernel_sm100"` at :422-424):** under
  `pk.use_v2_runtime`, pack the 4 scale tensors into MEGA_SC_ order + allocate xfer/bar/artifacts
  (crib sizes from `tests/runtime_python/blackwell_v2/dsv3_ffn_harness.py:100-290`), verify weight
  layouts match, and call `pk.dsv3_ffn_mega_layer(...)` instead of registering the v1 task.
- **EDIT 5 — tail linear gate = T-F (`builder.py:3551`):** `if self.mpk.use_v2_runtime:
  self.mpk.linear_layer_v3(input=..., weight=..., output=...)` else the existing
  `linear_layer(...)` (no grid_dim in v3; assert output.dim(0)<=16 satisfied at bs=1).
- **M3 tail route:** pass `--disable-vocab-parallel-lm-head` so the tail uses the PRESENT
  `argmax_reduce_v2` (331) — defers `nvshmem_global_argmax_v2` (T-C) to M4.
- **M3 PRE-CONDITIONS (§4.2 M3, do BEFORE the GPU run):** (1) reachability grep — `--layers 3-3`
  graph-build (test-mode, no GPU) diffed vs a full build to confirm no head/tail seed task is
  dropped; (2) graph-build guard (§5.2 #3) — assert every task_type in the v2 graph has a
  registered consumer body (fail loud at build, not deadlock at runtime).

### 4.2 Milestones (smallest-first)
- **M0 — leaf + tail wiring (no e2e; blackwell_v2 harness + TP2 micrograph):**
  author/validate `tensor_init_v2` (T-A), re-route DSv3 tail linear (T-F, validate via qwen3-
  style single-op), and confirm the already-present v2 leaves (rmsnorm/embed/argmax_partial/
  argmax_reduce) are reachable from the DSv3 builder under `use_v2_runtime`. Exit: all leaf ops
  bit-match in the harness.
- **M1 — AR at TP2 (`nvshmem_tile_allreduce_v2`, T-B):** author the v2-safe collective body;
  validate the TP2 micrograph (2 ranks, known-vector sum, bit-exact), then TP8. Exit: TP2+TP8
  AR bit-exact. (This is the highest-risk item — do it early to de-risk.)
- **M2 — fused blocks (FFN then ATTN):** T-D (`ffn_full_megakernel_v2`, prefer the D2 reuse of
  the proven `ffn_mega` if EP-locality checks out) → T-E (`attn_block_megakernel_v2`). Validate
  each in a TP8-shaped per-task harness bit-matching the v1 fused megakernel. Exit: both fused
  blocks bit-match v1 at TP8.
- **M3 — FIRST E2E (the path-proving milestone):** DSv3 decode **TP8 EP2, bs=1,
  `use_v2_runtime=True`, `--layers 3-3`, MTP off, `--disable-vocab-parallel-lm-head`** (routes
  the tail through the present `argmax_reduce_v2`, so T-C not yet needed). This exercises embed →
  attn_block_v2 → AR_v2 → rmsnorm_v2 → tensor_init_v2 → ffn_full_v2 → AR_v2 → final-norm →
  linear_v3 → argmax_partial_v2 → argmax_reduce_v2. **TWO HARD PRE-CONDITIONS (reviewer
  correction):** (1) **Reachability grep, NOT an assertion**, that starting at layer 3 doesn't
  drop something layer 0 / the dense path seeds — embedding tables, KV-cache init, RoPE cos/sin
  buffers, the first residual. The project has a documented dual-dispatch-deletion trap; run
  `demo.py --layers 3-3` graph-build (test-mode, no GPU) FIRST and diff the task list vs a full
  build to confirm no head/tail seed task is missing. (2) **EVERY task on the 1-layer path must
  have a COMPLETE v2 consumer body before M3 runs** (the §1.1 bodyless-consumer case wedges the
  box with D-state zombies, no graceful degrade). Add the §5.2 graph-build guard so this fails
  loud at build, not as a runtime hang. Correctness gate = poison-fill + coherence (the
  nondeterministic-TP8 protocol), NOT token-identity. Exit: one MoE layer decodes coherently on
  v2 at TP8.
- **M4 — restore vocab-parallel tail + scale layers:** author `nvshmem_global_argmax_v2` (T-C),
  validate TP2/TP8, drop `--disable-vocab-parallel-lm-head`; scale `--layers 3-N` up to all 58
  MoE layers. Exit: all-MoE-layer decode on v2 at TP8.
- **M5 — dense layers 0–2 (full 61-layer):** either author the 3 dense-path v2 tasks or
  `MPK_DSV3_DENSE_MLP_MEGAKERNEL=1` + `dsv3_dense_mlp_fused_v2` (one Form-2 task). Exit: full
  61-layer DSv3 decode on v2 at TP8 EP2 bs=1 — the deliverable. Measure tpot vs v1.
- **M6 (perf, separate) — de-fused-FFN A/B:** ONLY after M5, revisit adopting the de-fused FFN
  v2 chain (the +7–9% isolated win) as an e2e lever with a real TP8 A/B vs the fused-port. This
  is where the goal-shape B option gets its honest e2e verdict.

### 4.3 Correctness discipline (folded from CLAUDE.md / memory)
- The TP8 decode path is **token-level nondeterministic** (FFN-full cross-CTA FP atomicAdd) —
  token-identity A/B is INCONCLUSIVE as a safety gate. Use **poison-fill + coherence-in-envelope**
  for math-changing folds, and the **routed-MoE-non-null** gate (num_active≈4, DECODE_LEAN OFF)
  before trusting any perf number.
- Per-task ports (T-A..T-E) are validated **bit-exact vs the v1 kernel on identical input bytes**
  in the blackwell_v2 harness (deterministic single-op) — that IS a clean gate at the per-task
  level (the nondeterminism is a whole-graph atomicAdd effect, absent in the isolated op).
- Never let a rank silently no-op a task (the §1.1 trap) — add a **graph-build guard** (see §5).

---

## 5. Risks + rough sizing

### 5.1 Sizing
- **New v2 tasks for the steady-state MoE decode path: 5** — `tensor_init_v2` (trivial),
  `nvshmem_tile_allreduce_v2` (+resid) (HARD), `ffn_full_megakernel_v2` (MEDIUM, or reuse
  `ffn_mega` = LOW), `attn_block_megakernel_v2` (MEDIUM), `nvshmem_global_argmax_v2` (HARD,
  deferrable to M4) — **plus** the tail re-route (LOW) and the plumb (LOW).
- **Full 61-layer: +1–3** dense-path tasks (M5).
- Effort concentration: the two **collective** ports (AR, global_argmax) carry the v2-safe-sync
  rewrite risk; the two **megakernel-shape** ports are mostly mechanical grid_barrier→GMEM-
  count-barrier substitution IF the ffn_mega template holds (and FFN may be a near-free reuse).

### 5.2 Where it's most likely to break (ranked)
1. **AR/argmax block-wide-sync mismatch (§1.7).** Naive body reuse deadlocks on
   `__syncthreads()` (256 vs 128 threads). MUST rewrite to v2-safe sync + active-thread striding.
   This is THE hard part; TP2 micrograph (M1) surfaces it early.
2. **The `task_offset` metadata edit (§1.3).** Forgetting to add a new fused-mega v2 type to the
   `task_offset = bid.x` block (`runtime.cc:588-609`) → garbage CTA index → grid_barrier
   deadlock, silent. Also verify the `task_offset` (offset 0) vs `merge_task_offset` (offset 4)
   union aliasing — the v2 megas read `task_offset`, the v1 megas read `merge_task_offset`.
3. **The consumer-body dep-prefix invariant (§1.1).** Any new v2 task missing a consumer body →
   silent deadlock via SEM_DEP_READY. Add a **graph-build guard**: after graph-gen, assert every
   task_type present in the v2 graph has a registered consumer role variant (fail loud at build,
   not deadlock at runtime). This is a small, high-value defensive addition.
4. **SMEM budget + the `__align__(1024)` extern-smem footgun for the fused megas.** Both megas
   use a **DYNAMIC `extern __shared__ __align__(1024)` pool** (attn: `s_smem` at
   `attn_block_megakernel_sm100.cuh:1873`; FFN: `s_smem` at `ffn_full_megakernel_sm100.cuh:1890`)
   for a weight-prefetch ring + activation staging. The large scratch is in GMEM (FFN
   `SCRATCH_BYTES==106624` ≈104 KB, `:268/274`; attn `ATTN_SCRATCH_BYTES`, `:266` — both bound to
   a `new_tensor` scratch, builder.py:1772), so the *on-chip* extern-smem pool may be modest — but
   MUST be measured against the 14-page/224 KB region budget and declared as region(s) with
   **`alignment=1024`** in the `_spec.h`. A smaller align silently misaligns other tasks' TMA/AR
   in the shared test.cu → `cudaErrorMisalignedAddress`, caught only by an in-MPK run (known
   footgun, memory `feedback_extern_smem_align_megakernel_convention`). ⇒ do a `--layers 3-3`
   in-MPK smoke immediately after each fused-mega port, before trusting the harness.
5. **EP-locality of the FFN reuse (D2).** If reusing `ffn_mega_task_impl`, its `les/nle`
   (local_expert_start/num_local_experts) + shared-expert handling must exactly match the
   production TP8-EP2 per-rank slice, or routed-MoE is wrong (num_active mismatch). Verify against
   the v1 `ffn_full_megakernel` EP filter before trusting D2.
6. **Iteration-lockstep (LOW, but watch).** Same invariant as v1 (§1.7) — only breaks if a rank
   silently no-ops (covered by guard #3) or a rank-specific graph shape diverges (it doesn't at
   bs=1). The TP2 AR micrograph (M1) is the canary.

### 5.3 What is NOT a risk (settled by the investigation)
- v2 hosting a megakernel-shape op (PROVEN, `6c6a5825`).
- v2 multi-rank NVSHMEM init (SHARED with v1; heap/teams/barrier all set up).
- The plain (non-cooperative) v2 launch breaking the grid_barrier (136==SM count ⇒ co-resident).
- v2's nvshmem-event no-op breaking tile-AR (tile-AR never used that edge; self-syncs internally).

---

## 6. THE FIRST CONCRETE BUILD TASK (hand this to the first implementation agent)

**Task: author `nvshmem_tile_allreduce_v2` (+ `_with_residual`) as a v2-safe collective task and
validate it on a TP2 micrograph.** (This is M1 — done FIRST, before the fused blocks, because it
is the highest-risk item and its TP2 micrograph is the first multi-rank v2 test that de-risks the
whole effort. M0's `tensor_init_v2` + tail re-route are trivial and can be done in parallel by a
second agent, but the AR is the load-bearing de-risk.)

Concrete steps for the agent:
1. **Enum:** add `TASK_NVSHMEM_TILE_ALLREDUCE_V2` (+ `_WITH_RESIDUAL_V2`) to
   `runtime_header.h` (outside the TMA range, near the other multigpu enums 301–303).
2. **Kernel body:** add `nvshmem_tile_allreduce_v2` to a new `blackwell_v2/nvshmem_allreduce_v2.cuh`
   that reuses `allreduce.cuh`'s cross-rank `mpkar_wait_until_ge`/team-pSync logic VERBATIM but
   replaces every `__syncthreads()` with a v2-safe barrier over the participating warps and
   re-bases the `blockDim.x`-strided loops on the actual active-thread count. Use `multi_role`
   (7 warps / 224 threads) OR consumer-only (128) — pick 7-warp to match the v1 reduce width;
   sync the 7 warps via a named `bar.sync` or a tag-flag (§1.5). Add a `_spec.h`.
3. **Register:** `register_nvshmem_tile_allreduce_v2_task` in `task_register.cc` — consumer body
   MUST begin with `emit_dep_wait_consumer_prefix`; multi_role helper bodies for the 7-warp path;
   `register_variant_smem_info` for the tile smem.
4. **Dispatch + metadata:** add the name→enum→register mapping in `graph.cc`; add
   `task_type_to_name` + the `task_offset = bid.x` line (nvshmem team CTA mapping) for the new
   type in `runtime.cc` (mirror the v1 AR at `runtime.cc:571`).
5. **Wrapper:** add the `use_v2_runtime` branch to `allreduce_layer` in `persistent_kernel.py`
   (`"nvshmem_tile_allreduce_v2" if self.use_v2_runtime else "nvshmem_tile_allreduce"`).
6. **Validate — TP2 micrograph:** build a 2-rank test (mpirun -np 2) that allocates a symmetric
   `nvshmem_tensor`, each rank writes a known distinct vector, runs the v2 AR under
   `use_v2_runtime=True`, and asserts the reduced result == the analytic sum on BOTH ranks,
   bit-exact vs the v1 AR on the same inputs. Then repeat at TP8. Correctness = bit-exact (this
   op is deterministic in isolation). GPU-safety: gpu_safe lease, never crash-loop, D-state guard.

Acceptance: TP2 (and TP8) `nvshmem_tile_allreduce_v2` bit-matches v1 AR + residual variant adds
residual exactly once. This proves the v2-safe-sync rewrite AND cross-rank iteration-lockstep —
the two things the whole DSv3-on-v2 effort hinges on.

---

## 7. Open items to confirm during M0/M1 (cheap, do inline) — RESOLVED 2026-07-05 (recon agent a0578873)

- **SMEM footprint — RESOLVED, BOTH FIT the 224 KB / 14-page budget:**
  - ATTN `attn_block_megakernel_sm100`: on-chip extern smem ≈ **162 KB** (s_wbuf 131072 +
    s_act 28672 + s_score 2048 + red8) — GMEM scratch `ATTN_BLOCK_MEGAKERNEL_SCRATCH_BYTES=434864`
    is GMEM-only, off-budget. **162 KB < 224 KB ✓.**
  - FFN `ffn_mega` (the D2 task): on-chip ≈ **141 KB** at nwarps=7 (ring 114688 + RQR_NORM 14384
    + act 7392 + TK/W2act 7408); 4-warp variant 92.5 KB. GMEM scratch 106624 off-budget.
    **141 KB < 224 KB ✓.**
  - Both MUST declare extern smem regions with **`alignment=1024`** in `_spec.h` (footgun
    `feedback_extern_smem_align_megakernel_convention`). FFN's `dsv3_ffn_v2_spec.h` ALREADY does.
- **FFN D2 reuse — RESOLVED: D2 VIABLE (near-free), pending main-thread confirm of the exact
  builder→register mapping.** `ffn_mega_task_impl` (`TASK_DSV3_FFN_MEGA_V2=348`,
  `dsv3_ffn_v2.cuh:~1708-2110`, register `task_register.cc:~9343-9461`) is mathematically +
  structurally identical to v1 `ffn_full_megakernel_sm100` at TP8 EP2: EP-local filter
  (`dsv3_ffn_v2.cuh:494-499` == v1 `:2569-2571`), shared expert (`:727`, silu `:1860`), router
  (`router_partial_cpa<RKSPLIT,4>` verbatim v1), topk-8 sigmoid (`topk_compute` verbatim v1).
  les/nle plumbed via builder.py:~2960-2961 → register `:9461` `c.e("$, $,", les, nle)`.
  **⇒ T-D = builder re-route to the v2 FFN mega under `use_v2_runtime`, NOT a new kernel author.**
  ⚠ CAVEAT (over-claim guard, memory `feedback_subagent_wrong_code_path_overclaim`): the recon
  is an Explore agent; MAIN THREAD must confirm whether `ffn_full_megakernel_layer`/
  `dsv3_ffn_mega_layer` (persistent_kernel.py:~4184) actually registers 348 (v2) vs 325 (v1) at
  the DSv3 call site BEFORE treating T-D as done. The de-fused chain reachability (`ABSENT as
  such` in §2) refers to the BUILDER not selecting it, not the task being unregistered.
- **task_offset(0) vs merge_task_offset(4) union — RESOLVED:** v1 megas read `merge_task_offset`
  (offset 4; FFN `runtime.cc:524-525`, ATTN `:541-542`); v2 megas/chains read `task_offset`
  (offset 0; block at `runtime.cc:588-609`). `TASK_DSV3_FFN_MEGA_V2=348` is ALREADY in the FFN
  v2 `task_offset=bid.x` block (`:597-598`). **⇒ new `attn_block_megakernel_v2` MUST be added to
  the ATTN v2 block (`:603-609`)** or it reads a garbage CTA index → grid_barrier deadlock.

### 7.1 Revised M2 plan (post-recon)
- **T-D (FFN) = builder re-route, MEDIUM plumb (NOT one-line — main-thread-confirmed 2026-07-05).**
  D2 kernel confirmed proven+registered+metadata-wired (`dsv3_ffn_mega_v2`=348, wrapper
  `persistent_kernel.py:4192 dsv3_ffn_mega_layer`, asserts `num_tasks==num_workers`). BUT the v2
  wrapper takes a DIFFERENT tensor packing than the v1 `dsv3_tasks.ffn_full_megakernel_layer`
  (tasks.py:371, registers v1 `ffn_full_megakernel_sm100`, NO v2 branch): v2 wants packed
  `scales` (MEGA_SC_ layout w13|wgu|w2|wdn), a `xfer` f32 scratch (inter|y13|sg), a `bar` i64(2),
  and an `artifacts` u8 surface — vs v1's 14 separate tensors. **⇒ T-D work = builder-side: pack
  the 4 scale tensors into MEGA_SC_ order + allocate xfer/bar/artifacts (crib exact sizes from
  the reference `tests/runtime_python/blackwell_v2/dsv3_ffn_harness.py:100-290`), verify weight
  layouts match (w13 u8(128,1024,7168), wgu u8(512,7168), w2 u8(128,7168,512), wdn u8(7168,256)),
  and call `dsv3_ffn_mega_layer` under `use_v2_runtime`.** FOLD into the M3 plumb (kernel bit-match
  already proven by 6c6a5825 in the harness — no separate build/validate cycle needed). NOT a new
  kernel author.
- **T-E (attn) = the real new port**: author `attn_block_megakernel_v2` (new enum, next free
  after tensor_init's 352 — likely 353), 162 KB smem regions @ align=1024, grid_barrier→GMEM
  count-barrier, 128-thread audit of block-wide `__syncthreads()` (the CuTe `bar.sync 1,128`
  MLA warp-0 sync may map cleanly onto the 128-thread consumer group — do NOT convert it to
  `__syncthreads`), add to `runtime.cc:603-609` task_offset block, preserve zero-residual o_proj
  binding (AR does the combine). Validate bit-match vs v1 fused in TP8-shaped harness +
  `--layers 3-3` in-MPK smoke.
