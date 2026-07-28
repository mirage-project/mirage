# Runtime-V2 house style — the reference methodology (linear_sm100_v2)

The house-style reference is `include/mirage/persistent_kernel/tasks/blackwell_v2/linear_sm100_v2.cuh`
(+ `linear_sm100_v2_spec.h`), from upstream runtime_refactor. Every NEW v2 kernel is written
**toward** this idiom; deviations are allowed ONLY where `references/m1-decode-evidence.md`
shows the idiom measured DEAD for that op class. Verdict context: our DSv3 v2 kernels are v1
fused scalar-MAC bodies hosted in the v2 shell — NOT this methodology
(settled 2026-07-09, `project_v2_kernel_methodology_vs_reference`).

## 0. The two idioms (upstream's own split — pick one, don't blend)

| Idiom | When | Upstream example | Registration shape |
|---|---|---|---|
| **Role pipeline** (loader/launcher/consumer/storer) | GEMM-shaped op: real weight tiles, MMA-able, M≥16 tile or K-loop deep enough to pipeline | `linear_sm100_v2.cuh`, `linear_sm100_v3.cuh` | non-empty loader/launcher/consumer(/storer) bodies + init_semaphores (task_register.cc:2145-2151) |
| **Consumer-only** | Attention-shaped / elementwise / M=1 GEMV chains | upstream's OWN `attention_sm100.cuh` (`wg_id==0` gate at :133-134), `rmsnorm_v2`, `tensor_init_v2` | only `consumer` body non-empty (task_register.cc:2892, :7509, :7567) |

The pipeline idiom was never universal even upstream. Do not force a 4-role pipeline onto an
op whose evidence class is consumer-GEMV (see decision tree in SKILL.md Stage 1).

**Upstream tip update (0eadb3fd, studied 2026-07-15 — see
`references/upstream-kernel-catalog.md` for the per-kernel catalog):** upstream PROMOTED the
Channel-based v3 to be THE linear v2 and deleted the hand-written form (2b6a7461) — the Channel
idiom is no longer "preferred", it is the reference. Upstream also renamed the roles
(Consumer→Compute, Launcher→Mma, Controller→Dispatcher; `linear_device.cuh`→`linear_ptx.cuh`);
this doc keeps the old names, which match OUR tree — translate when reading upstream. Within
consumer-only, upstream splits on the SMEM axis: planner-region typed views (rmsnorm/argmax,
the default), one monolithic honest region (attention, port fallback), or NUM_REGIONS=0
(silu/embedding). No third sync idiom exists upstream.

## 1. Warp/role model (runtime_v2.cuh:287-306)

Fixed 8-warp worker: **W0-3 consumer (128T), W4 loader, W5 launcher, W6 storer, W7 controller**.
Controller (runtime-owned) fetches TaskDesc into a 3-slot ring, runs the op's
`init_semaphores` body (lane 0, once per publish, runtime_v2.cuh:1114-1123), arrives
`INSTRUCTION_ARRIVED`; each role loop waits it, runs its role body, arrives
`INSTRUCTION_FINISHED` (macro at runtime_v2.cuh:815-861). Role bodies are
`__device__ __noinline__` free functions, one per role — never one function branching on warp id.

## 2. Role responsibilities (cite = linear_sm100_v2.cuh)

**Loader (W4)** — `linear_loader_task` (:185-311)
- `elect_sync()` one thread; rest return (:200-202). `prefetch.tensormap` both descriptors (:205-206).
- **Start-of-task re-init of async-arrived mbars** (:246-259): mma/W_tma/A_tma mbars are arrived
  by hardware (TMA byte-delivery, tcgen05.commit) — a PRIOR ring-slot occupant's stray arrival can
  land AFTER the controller's init_semaphores re-init and flip the phase → deadlock. The role that
  ARRIVES a mbar owns re-initing it at task start, then `fence.mbarrier_init.release.cluster`.
  This is the single most important protocol rule in the file; the v3 `channel.cuh`
  `reinit_full()/reinit_empty()` (:108-121) is the packaged form.
- Pipeline loop: `mbarrier_wait(mma_mbar[stage], phase)` (stage free?) → W TMA
  (`tma_3d_load_l2` + `mbarrier_arrive_expect_tx(W_mbar, W_SIZE)`) → **inline cross-SM dep-wait
  once, before the FIRST A TMA** (`wait_task_dependency`, :296-300 — weights prefetch AHEAD of
  the dependency; activations cannot) → A TMA → advance stage/phase (:305-308).
- L2 hints: weights `L2_EVICT_FIRST`, activations `L2_EVICT_LAST` (:129-131, :292-302).

**Launcher (W5)** — `linear_launcher_task` (:322-494)
- All 32 lanes: `tcgen05.alloc.cta_group::1.sync.aligned` writes taddr into REGION_SCRATCH
  (:374-379). Lane 0: re-init mainloop/epilogue mbars (same stale-arrival rule, :391-410), then
  arrive `SEM_TMEM_READY` to publish taddr (:411).
- **Bounds-fail path MUST still release all declared pages** (:353-363) — a padding task that
  just `return`s deadlocks the next occupant on `page_ready`. Same rule in storer (:663-671).
- Elected lane MMA loop: wait W_tma + A_tma mbars, `tcgen05.fence::after_thread_sync`, then
  `tcgen05_mma` over MMA_K sub-tiles with SMEM-descriptor increments (:436-455),
  `tcgen05_commit(mma_mbar[stage])` per K-stage, `tcgen05_commit(mainloop_mbar[slot])` per tile.
- Task-end: `__syncwarp()` reconverge after the elect block FIRST — ITS does not rejoin lanes
  at the if-exit; runaway lanes 1..13 once released pages while elected lane 0 was still in
  the MMA loop (race 1, fixed `689dadc5`) — then lane-parallel blanket page release
  (`runtime_finish_page`, :475-477 — pairs with `auto_consumer_finish=false` in registration),
  lane-0 wait `consumer_done`, then `tcgen05.dealloc` using the taddr CACHED at :387 (scratch
  page may already be freed).

**Consumer (W0-3)** — `linear_consumer_task` (:497-628)
- Lane 0 waits `tmem_ready` (+`__syncwarp`), read taddr from scratch region (:540-546).
- Per tile: wait `mainloop_mbar` + `tcgen05.fence::after_thread_sync`,
  `tcgen05.ld.sync.aligned.32x32b.x16` → `tcgen05.wait::ld`, epilogue math, store via
  `st.global.L1::no_allocate` (:557-615). Arrive `epilogue_mbar` (frees the TMEM slot for the
  launcher's next tile), finally every thread arrives `consumer_done` (count 4*32, :625-627).

**Storer (W6)** — `linear_storer_task` (:630-767): passive per-stage page-release engine. Rides
the launcher's mma_mbar parity in lockstep; computes per-stage `last_use` fire counts; on a
stage's last fire, decrements per-page refcounts (sub-page-packed A regions share pages) and
`runtime_finish_page`s pages that hit 0 (:722-752). Frees the launcher (at the 255-reg cliff)
from release work and enables the NEXT task's loader to start during this task's tail.

**Launcher-blanket vs storer-per-stage release — ONE owner, never both.** Page accounting is
parity keyed by instruction index (runtime_v2.cuh:352-354): each physical page has a single
count-1 mbarrier (`page_finished[page][0]`, init runtime_v2.cuh:461-470), the occupant of
instruction i waits parity `i & 1` (:472-480), and every arrive flips it — so EVERY task must
arrive each page EXACTLY ONCE (runtime_v2.cuh:1109-1113; a second arrive flips parity back →
the next occupant deadlocks). The launcher's blanket release is the CURRENTLY-WIRED owner: the
registration sets `/*storer=*/""` + `auto_consumer_finish=false` (task_register.cc:2149-2150,
the flag that disables the codegen consumer-suffix arrive for the same exactly-once reason).
The Phase-5 storer is the ALTERNATIVE owner — wiring it TRANSFERS ownership (delete the
launcher's blanket release at :473-477 and its bounds-fail twin :353-363); it never runs
alongside it, so do not double-release. **Upstream verdict (0eadb3fd): the storer-per-stage
engine was DELETED with the old hand-written v2** — the promoted linear ships launcher(Mma)-
blanket as the only wired owner, plus a per-stage CROSS_TASK_PAGES variant compiled OFF
(linear_spec.h:41-51): mechanism verified, but without (1) footprint ≤ half the 14-page pool and
(2) planner double-buffered page assignments it is pure added sync, measured ~+12%. Cite that
before proposing cross-task page overlap.

## 3. SEM ordinal table — the documented contract (:46-62)

Every pipeline kernel declares a comment-block table + constexpr ordinals in the .cuh:

```
// Per-task SEM ordinals (relative to dyn_sem_base):
//   [+0..+5 ]  W_tma_mbar    (count=1,       loader→launcher, W only)
//   [+6..+11]  A_tma_mbar    (count=1,       loader→launcher, A only)
//   [+12..+17] mma_mbar      (count=1,       launcher→loader, "stage K MMA done")
//   [+18..+19] mainloop_mbar (count=1,       launcher→consumer)
//   [+20..+21] epilogue_mbar (count=4*32,    consumer→launcher)
//   [+22]      tmem_ready    (count=1,       launcher→consumer)
//   [+23]      consumer_done (count=4*32,    consumer→launcher)
constexpr int SEM_W_TMA_BASE = 0; ... constexpr int NUM_OP_SEMS = 24;
```

Each row: **ordinal range, count, producer→consumer direction, meaning**. Budget:
`MAX_DYNAMIC_SEMAPHORES=32` per ring slot, `SEM_DEP_READY=0` is runtime-reserved,
op-private slots start at `SEM_OP_BASE=1` → **≤31 op mbars** (runtime_v2.cuh:366-376).
Address = `op_sem_base_addr(runtime_smem, instruction_index) + ordinal*8` (runtime_v2.cuh:455-459);
the registration's `init_semaphores` body mbar_inits every ordinal with its count
(task_register.cc:2035-2059) and ends with `fence.mbarrier_init.release.cluster`.

**Re-init ownership is part of the table contract — mark every row with its owner**
(controller-only / loader / launcher-lane-0). "Controller `init_semaphores` only" is NOT a safe
default for every mbar: (a) async-arrived mbars (TMA byte-delivery, `tcgen05.commit`) always
need a start-of-task re-init by their ARRIVING role (§2 rule); (b) **any handshake mbar whose
arrivals land near END-OF-TASK — adjacent to the ring-slot-reuse boundary
(`INSTRUCTION_RING_SIZE=3`, i.e. right before the arriving role's `INSTRUCTION_FINISHED`) —
needs a DEFENSIVE role-level stale-arrival re-init too, unless its timing is PROVEN safe**: the
tail arrivals (127 of `consumer_done`'s 128 are lanes ≠ the FINISHED-arriving lane 0, so they
carry no release-chain ordering into the slot republish) can land AFTER the controller's
per-publish re-init on the reused slot and flip the fresh phase. Incident (dsv3_ffn_gg_v2 v008,
2026-07-15): `consumer_done` left controller-init-only → opposite-parity stuck phase on ring
reuse (launcher wait hung at phase 0 with `try_wait.parity(1)` ready); fix = add it to the
launcher's role-level re-init list, inheriting epilogue's proven timing. Mbars arrived EARLY
in-task by a role that runs after `INSTRUCTION_ARRIVED` (e.g. `tmem_ready`) have no stale
window and may stay controller-only.

**Preferred form: declare the table as DATA, not just a comment** (upstream-current norm,
`linear_spec.h` CHANNELS[]/ONESHOT_SEMS[] — already in our tree): each channel row carries
depth, producer/compute roles, arrival kinds, full/empty sem bases + counts, `shares_empty_with`,
and `reinit_full_by`/`reinit_empty_by`; one-shots carry `reinit_by`. The table is LIVE — consumed
by the structural `linear_init()` (controller), `reinit_for_role()` (roles), and `make_wa()`
(cursor wiring) — and guarded by `static_assert(declared_sem_count() == NUM_OP_SEMS)` so ordinal
drift can't be silent. Two upstream-encoded rules worth copying verbatim: `tmem_ready`
reinit_by=None ("re-initing it would race the consumer's wait" — it is waited at task start
CONCURRENT with the would-be re-initer) and `consumer_done` reinit_by=launcher ordered BEFORE it
publishes `tmem_ready` ("LOAD-BEARING") — upstream's independent encoding of exactly our
dsv3_ffn_gg_v2 v008 incident class. Two C++ gotchas when going table-driven (linear_ptx.cuh):
`reinit_for_role` MUST be `__forceinline__` (as a real call nvcc reorders the mbarrier-init
fence vs surrounding TMA/MMA issue, re-exposing the stale-arrival race), and device code cannot
runtime-iterate a host `constexpr` table (odr-use → "undefined in device code") — extract fields
only via constexpr-indexed macro expansion.

**Race-fix protocol rules (2026-07-16 — durable; enforce on every new/ported kernel):**
- **Arriver-set == waiter-set.** Every lane performing a lane-parallel mbar/page arrive must
  be reconverged with the work it publishes: after any `elect_sync()`/divergent block,
  `__syncwarp()` BEFORE the arrives — ITS does not rejoin lanes at an if-exit, and runaway
  lanes releasing early advance parity one use ahead (race 1, `689dadc5`).
- **Dense-observer-or-full-owner for mod-2 parity mbars.** A parity wait (`releases(p) ≡ s
  mod 2`) is sound ONLY if the waiting warp observes EVERY sequence of that page (wait-all
  dense observation) or one warp-group owns claim→body→release program-ordered
  (consumer-TOTAL). Sparse observation (SkipUsed across a chain reusing pages) cannot
  distinguish 0 releases from 2 and passes two occurrences early (race 2 + residual,
  `7d271a01` + `7b6ae2bb`).
- **Exit-reads snapshot BEFORE barrier arrival.** Any value deciding loop exit/termination
  must be read before arriving the iteration/end barrier; reading after races the next
  iteration's producer, and a straggler exits one iteration early (race 3, `025029a1`).
- **Protocol invariants become plan-time assertions.** If safety depends on a plan shape
  (page-window, chain interleave), assert it in `build_v2_plan` — the mixed-chain
  page-window assertion caught a real qwen3 shape at build time (`7b6ae2bb`).

## 4. SMEM: spec.h is the single source of truth

Pattern (`linear_sm100_v2_spec.h`): host-safe header, included by BOTH task_register.cc (planner)
and the .cuh (typed views). Contains:
- Mirrored constexpr shapes + a hand-noted "keep in sync" comment (:43-49); better: a
  `static_assert` pinning spec↔v1/cuh constants (dsv3_ffn_v2.cuh:73-79 does this — copy that).
- **Region ordinals** as constexprs; must match `make_smem_info()` push order AND the
  `task_desc->smem_region_offset(REGION_*)` calls (:56-60).
- `make_smem_info()` returning `TaskSmemInfo{total, /*alignment=*/1024, regions}` with named
  regions `{name, size, alignment, page_count=-1, can_pack, release_step, contiguous}` (:79-113).
  Page math: PAGE_SIZE=16KB, 14 pages/task; W 32KB=2 pages ×6 (not packable), A 4KB packable
  (4/page → 2 pages), 16B scratch packs into the A page = exactly 14. Split W/A rather than one
  36864B region: combined would round to 3 pages → 18 > 14 (:19-23).
- Capacity static_assert vs `PLANNER_CAPACITY_BYTES = 225*1024 − 6*1024 = 224256`, which must
  equal `python/mirage/mpk/v2_smem_planner.py:CAPACITY_BYTES` (:74-77). ≤16 regions/task
  (`MAX_SMEM_REGIONS_PER_TASK`, runtime_header.h:96).
- Device side addresses ONLY via `smem_region_offset` — never hand-rolled offsets.
- **Assign regions BY ORDINAL, not positional push_back** (upstream b27cd716):
  `info.regions.resize(NUM_REGIONS); info.regions[REGION_X] = {...};` — push-order-as-contract
  is a silent-corruption footgun the moment the list is reordered. And declare sizes once:
  shared constexpr `raw_*_bytes()` used by BOTH `make_smem_info()` and the device
  `SmemBuffer<>` template args (upstream 55dc49b2 — "no second copy of the formula to drift"),
  plus a `static_assert(NUM_REGIONS == N)` in the device buffer struct.
- **Porting a kernel with hand-rolled internal offsets?** Use upstream's honesty fallback
  (`paged_attention_sm100_v2_spec.h`): declare ONE monolithic region sized to the real footprint
  so the planner's page assignment stays truthful about which bytes the device touches — trade
  release_step granularity for correctness, and leave a FUTURE note to split per-buffer. Never
  declare per-buffer regions the device won't actually address.
- `extern __shared__ __align__(1024) char smem_ptr[]` — 1024, NEVER smaller: all tasks alias one
  test.cu dynamic-smem region; one bare/`align(16)` decl lowers the base for everyone →
  cudaErrorMisalignedAddress in OTHER tasks' TMA/AR (memory: extern_smem_align, recurred 2026-07-06
  at 19 bare v2 sites; memcheck 826→0 after fix).

## 5. TMA + tcgen05 patterns

- Pass `CUtensorMap const *` BY POINTER (cp.async.bulk.tensor needs .const/.param/.global, not
  stack — :175-177); `prefetch.tensormap` first; loads are
  `cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint`
  paired with `mbarrier_arrive_expect_tx(mbar, BYTES)` (:110-127).
- TMA SMEM dst must be 1024-aligned for 128B swizzle (channel.cuh:31-32; the linear_v3 W-TMA
  fault was exactly a misaligned dst — `aligned_smem_base` fix).
- tcgen05: SMEM descriptor + instruction descriptor as constexpr encodings (:152-161);
  MMA single-thread-issue from an elected lane; `tcgen05.commit` to a mbar for async completion;
  alloc/dealloc `sync.aligned` by the SAME warp (launcher), all 32 lanes; consumers read via
  `tcgen05.ld` + `tcgen05.wait::ld`; fences `tcgen05.fence::after_thread_sync` after any
  mbar-wait that orders MMA↔SMEM/TMEM access.
- Prefer the v3 `channel.cuh` primitives for NEW kernels: `Channel<DEPTH,By::Tma,By::Mma>` +
  `Producer/Consumer` cursors (single owner of stage index — the desync-bug killer),
  `SmemRing` (storage + per-stage page lifecycle), `TmemChannel` (taddr + st*cols addressing).

## 6. Task granularity + registration shape

- **Default: per-TILE tasks.** One TaskDesc = one output tile; `tile_idx` comes from
  `task_desc->task_metadata.task_offset` (= bid.x, set in runtime.cc:579-585); SPLIT_K /
  TILES_PER_TASK as template params (:170-177). This is what lets the runtime overlap tasks
  cross-SM and prefetch cross-task.
- **Grid-wide fused ops (num_tasks==num_workers) are the EXCEPTION** — allowed only with written
  justification, and then MUST carry: monotonic GMEM grid barrier
  (`need = num_tasks*(iter_num+1)`, never reset — attn_block_megakernel_v2.cuh:120-140), the
  host-side `num_tasks == num_workers` assert (persistent_kernel.py:2952-2955), and
  `skip_after_step0=True` on the barrier scratch's tensor_init (see wiring-recipe.md §8).
- Registration: build per-role CodeKeeper bodies; the plain variant string stays wired to the
  consumer body (dedup identity, task_register.cc:2108-2117); **§1.1 LETHAL INVARIANT — every
  role body that participates each step must start with `emit_dep_wait_consumer_prefix` (emits
  `consumer_dep_prefix(...)`) or the runtime must arrive SEM_DEP_READY on its behalf**; a
  prefixless consumer wedges the ring slot silently (task_register.cc:7501-7507,
  runtime_v2.cuh:609-656, BEGIN_TASK_GRAPH special case :1105-1123). Loader may skip the codegen
  prefix ONLY if it does the dep-wait inline (linear does, :2120-2123). Helper roles whose first
  action is an acquire on a consumer-released flag may skip it (transitively ordered,
  task_register.cc:9716-9727). `auto_consumer_finish=false` iff a non-consumer role owns page
  release (:2150).

## 7. Quality bar (what "done" looks like)

1. File header design-comment: what the op is, phase list, warp model, what changed vs source.
2. Documented SEM ordinal table (§3 format) + constexpr ordinals.
3. spec.h with `make_smem_info()`, region-ordinal constexprs, capacity static_assert, and
   static_asserts pinning every mirrored constant (byte-exact spec).
4. Stale-arrival re-inits for every async-arrived mbar, owned by its arriving role.
5. Bounds-fail paths release pages; every page arrived exactly once per task.
6. No `__syncthreads()` anywhere in a role body (only ≤its-own-warp `__syncwarp`, named
   `bar.sync N, 128` for consumer-only 128T bodies — ids 1/2/3/6 taken by
   linear/rmsnorm/ffn/attn), no `blockIdx` for identity (task_offset only).
7. Profiling hooks compile away: `MPK_V2_TIMED_WAIT` at long waits, role tracks free
   (runtime_v2.cuh:27-58); default build byte-identical. Upstream discipline
   (linear_sm100_v2.cuh@runtime_refactor): time waits on the FIRST ring lap only ("timing every
   K-iter measurably slowed the kernel"); ONE designated writer per stall track; the
   non-profiling expansion must be TEXTUALLY identical to baseline (sm100 is branch-sensitive
   around tcgen05 waits); a thread-0-timed wait diverges the warp under ITS — `__syncwarp()`
   reconverge (profiling builds only) before any following `tcgen05.ld`.
8. clang-format-15 clean (`bash scripts/format.sh`).
