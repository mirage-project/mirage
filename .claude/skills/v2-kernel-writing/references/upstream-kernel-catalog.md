# Upstream v2 kernel catalog — mirage-project/runtime_refactor @ 0eadb3fd (studied 2026-07-15)

Merge-base with our tree = e08de1df: everything below marked Δ is upstream-only. All paths are
`include/mirage/persistent_kernel/` unless noted; read any file via
`git show mirage-project/runtime_refactor:<path>` — on a fresh clone first
`git remote add mirage-project https://github.com/mirage-project/mirage.git &&
git fetch mirage-project runtime_refactor` (this doc is self-contained if you can't).

**Naming map (upstream renamed roles in 06ff516d — our docs/tree keep the OLD names):**
old Consumer→**Compute**, Launcher→**Mma**, Controller→**Dispatcher**; profiler tracks
`*_PHASE`→`*_STALL`; `linear_device.cuh`→`linear_ptx.cuh`; the Channel-based `linear_sm100_v3`
was PROMOTED to be THE `linear_sm100_v2` (2b6a7461) and the old hand-written v2 was deleted.

## 0. Inventory (17 files, ~3k lines — deliberately small)

| Kernel (tasks/blackwell_v2/) | Family | Task type (upstream id) | graph.cc tuple |
|---|---|---|---|
| linear_sm100_v2.cuh + linear_spec.h + linear_ptx.cuh | A pipeline | TASK_LINEAR[_WITH_RESIDUAL]_SM100_V2 (244/245) | (2,1)/(3,1) |
| rmsnorm_v2.cuh + rmsnorm_v2_spec.h | B regions | TASK_RMS_NORM_HOPPER_V2 (224 — "hopper" is historical) | (2,1) |
| argmax_sm100.cuh + argmax_v2_spec.h | B regions | ARGMAX_PARTIAL/REDUCE_SM100_V2 (228/229) | (1,2)/(2,1) |
| attention_sm100.cuh + paged_attention_sm100_v2_spec.h | C monolith | TASK_ATTN_SM100_V2 (227) | (7,1) |
| silu_mul_v2.cuh, embedding_v2.cuh (+specs, NUM_REGIONS=0) | D no-SMEM | 225 / 226 | (1,1)/(2,1) |
| norm_sm100.cuh, rotary_embedding_v2.cuh | E helpers | not tasks — SMEM-view device fns called inside attention | — |
| channel.cuh, task_header.cuh | infra | — | — |

Namespaces: linear/rmsnorm get their own (`kernel::linear_v2`, `kernel::rmsnorm_v2`); the rest
SHARE `kernel::v2` (task_header.cuh:11-14). Our stricter one-namespace-per-op rule is fine, but
expect `kernel::v2` when porting upstream files.

Upstream REMOVED since our merge (signals "dead" verdicts): old hand-written linear v2 + its
storer-per-stage release engine, `task_interface.cuh` (declarative task-spec layer),
blackwell_v2 copies of `sm100_ptx/sm100_utils`, `tensor_init_v2`, `mul_sum_add_v2`,
setmaxnreg/measure-role scaffolding (ec99d9b2), Channel `drain()` (2a5b2465).

## 1. Family A — Channel-based role-pipeline GEMM (`linear_sm100_v2.cuh`)

Roles: **loader W4** (elect_sync single thread), **mma W5** (all 32 lanes — alloc/dealloc are
sync.aligned), **compute W0-3** (128T epilogue + global stores). **Storer W6 = EMPTY** — upstream
deleted the per-stage storer release engine; page release lives on the mma warp.

- Sync/storage/TMEM are 3 composable primitives (channel.cuh): `Channel<DEPTH,By,By>` (mbars
  only), `SmemRing<DEPTH,PAGES_PER_SLOT>` (per-stage offsets + optional page lifecycle),
  `TmemChannel` (taddr + st*cols). Producer/Consumer cursors OWN the stage index — "the single
  owner of `st` is what keeps the four role functions in step". Commit variants by producer type:
  `commit_warp` (mbar.arrive), `commit_mma` (tcgen05.commit), `commit_tma` (no-op — the TMA op
  already carries the mbar).
- **W and A share one empty edge** (mma_mbar): one tcgen05.commit per K-iter frees both; only the
  W cursor waits it, A just advances alongside. Declared via `shares_empty_with` in CHANNELS[].
- **Stale-arrival protocol** (in-file, load-bearing): "Each role re-inits its async edges at the
  start of a task… Without that the kernel deadlocks once slots are recycled across tasks." And
  the tail: "The loader doesn't wait out its in-flight TMAs at task end: blocking on the final
  mma_mbar there would deadlock against the next task reusing the slot."
- **Dep-wait placement**: loader issues the W TMA (weights, dep-free prefetch) BEFORE
  `wait_task_dependency`, then the dep wait, then the A TMA. Registration gives the codegen
  dep-prefix to mma AND compute bodies; loader does it inline (task_register.cc:1961-1966).
- `TaskCtx ctx_from<>()`: derived shape computed by ONE function called identically from all
  three roles — "makes drift impossible".
- Epilogue: TMEM double-buffer (2 slots × BLOCK_N cols) so tile t+1's MMA overlaps compute of
  tile t. Residual = **precision-clamp round-trip** (bf16-quantize GEMM out, add residual in
  f32, bf16-quantize again — "do NOT collapse the round-trip — semantics change").
  Stores `st.global.L1::no_allocate`. SPLIT_K>1 stores f32 partials to a workspace instead.
- Page release: task-end **lane-parallel blanket** on the mma warp
  (`lane < MAX_SMEM_PAGES_PER_TASK && !Wr.owns(lane)`), `auto_compute_finish=false`;
  `__syncwarp()` before it (ITS: non-elected lanes must not free pages mid-MMA).
- **CROSS_TASK_PAGES = false** (linear_spec.h:41-51): per-stage acquire/release mechanism is
  complete + verified but pays off ONLY with (1) footprint ≤ half the 14-page pool and (2) the
  planner double-buffering page assignments; without both it's pure added sync, **measured ~+12%**.
  An honest capability-proven/default-OFF record — cite it before proposing cross-task page overlap.
- **tiles_per_task MUST be 1**: a partial last task (num_tiles % tpt) has different barrier
  accounting → deadlock on slot reuse (file header USAGE note; also slower). The mma bounds_fail
  early-return skips the blanket free → page-parity desync — unreachable at tpt=1, documented.

When to copy: any genuinely GEMM-shaped v2 op (M≥tile or a real K-pipeline) — our U3 cell.

## 2. Family B — consumer-only, planner-region typed SMEM (rmsnorm, argmax)

- Compute-only registration: loader/mma/storer bodies empty (warps no-op via dispatcher default
  case); `auto_loader_page_lifecycle` stays true → codegen emits a SYNTHETIC loader = just the
  page prefix. Body gates `if (threadIdx.x >= NUM_THREADS/*128*/) return;`.
- SMEM idiom: a typed buffer struct (`RmsNormBuffers`, `ArgmaxBuffers`) whose members are
  `SmemBuffer<raw_*_bytes(...), ALIGN>` views constructed from
  `task_desc->smem_region_offset(REGION_*)` — plus `static_assert(NUM_REGIONS == N)` pinning the
  device view to the spec.
- Spec rules (hardened upstream, b27cd716 + 55dc49b2): regions assigned by
  `info.regions[REGION_X] = {...}` after `resize(NUM_REGIONS)` — **never positional push_back**
  ("silent-corruption footgun if the list is ever reordered"); sizes come from shared constexpr
  `raw_*_bytes()` used by BOTH make_smem_info and the device SmemBuffer template args ("no second
  copy of the formula to drift").
- Sync: `cutlass::arch::NamedBarrier` / `bar.sync <id>, 128`. **Named-barrier ID registry**
  (in-file comments): 0 block-wide, 1 linear, 2 rmsnorm, 6 attention compute-WG, 7 rotary.
- rmsnorm body: cp.async double-buffered tile loop (`cp_async_wait<1>`), warp shfl_xor reduce →
  SMEM → final reduce, vectorized 16B SMEM→GMEM writeback.
- argmax: TWO tasks (partial + reduce) — grid decomposition by NUM_PARTIAL_TASKS; reduce packs
  `(chunk_idx<<32)|rel_idx` into one i64 so a single max-reduce carries both.

When to copy: any new elementwise/reduction v2 task that stages through SMEM — this is the
default consumer-only shape.

## 3. Family C — consumer-only, monolithic SMEM (attention)

`attention_sm100.cuh` = the Ampere-heritage multitoken paged attention hosted in v2: `wg_id==0`
does everything (the else branch is empty), cp.async + `mma_m16n16k16_bf16bf16bf32` (NOT
TMA/tcgen05), manual k/v double-buffer rotation, online softmax with per-warp m/d/o partials
merged through SMEM buffers, fused qk-norm (norm_sm100), fused rope, fused KV-cache append.
SMEM = `AttentionBuffersImpl(char*)` with hand-rolled compile-time offsets.

- **The planner-truthfulness fallback** (paged_attention_sm100_v2_spec.h): because the device
  ignores per-region offsets, the spec declares ONE monolithic `attention_smem` region so "the
  page IDs the planner assigns are exactly the ones the data lives on" — explicitly trading
  release_step granularity for correctness, with a FUTURE note to split back. Copy this pattern
  whenever porting a kernel with internal offsets; never declare regions the device won't honor.
- Registration comment says it plainly: "attention's body gates work via `if (wg_id == 0)` so
  threads 0..127 do everything; loader/mma/storer aren't used."

When to copy: fast ports of existing v1/ampere monolith kernels into v2 — correct first, region
split later.

## 4. Family D — consumer-only, no SMEM (silu_mul, embedding)

Spec exists but `NUM_REGIONS=0`, `make_smem_info()` returns `{0,1,{}}` — the spec file is still
the registration's single source (uniform shape). Gate on `CONSUMER_NUM_THREADS` (128). Embedding:
16B-vectorized row copy behind a runtime alignment guard with scalar fallback — rationale quoted:
"the compute owns only 128 threads (4 warps), half of v1's 256-thread worker, so per-thread
efficiency must carry the difference."

## 5. Family E — SMEM-view sub-op helpers (norm_sm100, rotary_embedding_v2)

Not tasks. Device functions over an `InputSmem` layout view, templated on the NamedBarrier IDs
they sync on, called INSIDE another kernel's body (attention's qk-norm/rope). Pattern for fusing
a small op into a host kernel without a new task type.

**Idiom verdict: still TWO sync idioms** (Channel pipeline vs consumer-only). The finer split is
the SMEM-addressing axis: planner-region typed view (B, the default) / monolithic honest region
(C, port fallback) / none (D). No third synchronization idiom exists upstream.

## 6. Registration + runtime contract digest (what every family relies on)

- `TaskRoleVariantCode{init_semaphores, loader, mma, compute, storer}` + flags
  `auto_loader_page_lifecycle` (default true: loader prefix waits every page, immediately frees
  pages the task doesn't use — "claim+release ASAP") and `auto_compute_finish` (default true:
  compute suffix frees used pages; set false iff a role releases in-body, e.g. linear's mma).
  Net invariant: **every physical page arrived exactly once per task**. The suffix iterates
  PHYSICAL PAGES not regions — packed sub-page regions would multi-flip parity
  (v2_role_codegen.cc kComputePageSuffix comment).
- `emit_dep_wait_compute_prefix` = first line of every compute body (+mma for linear). Mechanism:
  thread 0 spins the monotonic cross-SM event counter (`needed = num_triggers*(iter+1)`), arrives
  per-slot SEM_DEP_READY; ALL lanes wait that mbar directly — **do NOT lane-0-gate +
  `__syncwarp()`**: sm_100a compiles it to a WARPSYNC.COLLECTIVE whose wake crawls ~5µs/token
  (runtime_v2.cuh compute_dep_prefix comment). Single-spinner saves ~191 L2 atomic loads/task.
- Dispatcher (W7): copies TaskDesc via cp.async then **`fence.proxy.async.shared::cta`** before
  the ARRIVED mbar (v1 got this implicitly from __syncthreads; v2 must fence). Init-semaphores
  hook runs lane-0 once per publish. BEGIN_TASK_GRAPH gets SEM_DEP_READY + all page parities
  arrived on its behalf (bodyless task would desync parity). Event triggering is EAGER and
  OUT-OF-ORDER with per-slot dedup — "Out-of-order is REQUIRED: an earlier compute task can block
  on an event whose producer is a LATER, already finished task on the same ring".
- Structural init vs role re-init: dispatcher runs the op's `linear_init()`-style full mbar init
  per publish; roles re-init ONLY their async-arrived edges at task start per the spec's
  `reinit_full_by/reinit_empty_by/reinit_by` policy tables (CHANNELS[]/ONESHOT_SEMS[] in
  linear_spec.h — consumed by linear_init, reinit_for_role AND make_wa, guarded by
  `static_assert(declared_sem_count() == NUM_OP_SEMS)`). Two data-encoded rules to keep:
  `tmem_ready` reinit_by=None ("re-initing it would race the compute's wait");
  `compute_done` reinit_by=Mma BEFORE it publishes tmem_ready ("LOAD-BEARING") — upstream's
  independent encoding of our dsv3_ffn_gg_v2 v008 incident class.
- Two device-side C++ gotchas (linear_ptx.cuh): `reinit_for_role` **MUST be `__forceinline__`**
  (as a real call nvcc reorders the mbarrier-init fence vs surrounding TMA/MMA issue → re-exposes
  the stale-arrival race); device code **cannot runtime-iterate a host constexpr table** (odr-use
  → "undefined in device code") — access CHANNELS[] only via constexpr-index macro expansion.
- **Task-ID ranges have semantics** (cba02075): `create_tma_desc_by_task` auto-fires for the SM100
  TMA range **231-256** — TMA-consuming tasks must sit inside it (or be added explicitly);
  non-TMA tasks must stay OUT (upstream parked its non-linear v2 ids at 224-229; linear 244/245
  is deliberately inside). Ours renumbered independently (242/243 + 326+) — check this rule when
  allocating any new ID and at every upstream merge.
- Profiling idiom: `MPK_V2_PROF_SNAPSHOT()` once per body; `MPK_V2_TIMED_WAIT_IF(first-ring-lap
  only, …)` — "timing every K-iter measurably slowed the kernel"; ONE designated writer per stall
  track; profiling builds need a `__syncwarp()` reconverge after a thread-0-timed wait before
  tcgen05.ld; the non-profiling expansion must stay TEXTUALLY IDENTICAL to baseline (sm100 is
  branch-sensitive around tcgen05 waits).

## 7. Sync-with-upstream list (Δ e08de1df..0eadb3fd we have NOT pulled)

1. **Scheduler de-dup (b0753d51)** — upstream: Python `v2_task_schedule.py` is the single source;
   `build_v2_plan` just parses `v2_worker_task_queues` from task_graph.json. OUR tree still has
   the C++ round-robin twin with a "MUST stay in lockstep" comment — the exact latent corruption
   upstream killed; it BITES the moment cross-task page overlap / non-round-robin scheduling
   lands. Pull when touching the scheduler or enabling CROSS_TASK_PAGES.
2. **SMEM page geometry from C++ (0583aa81)** — TASK_SMEM_PAGE_SIZE / MAX_SMEM_PAGES_PER_TASK
   emitted into task_graph.json as `v2_smem_config`; planner reads them. Ours re-hardcodes in
   Python. (Caveat from the commit: MAX_DYNAMIC_SHARED_MEMORY_SIZE is deliberately NOT sourced —
   core.so is arch-agnostic and would export the wrong arch's value.)
3. **Task-ID renumber (cba02075)** + the TMA-range rule above — merge-conflict certainty; keep
   the profiler bucket maps keyed on enum names, never numbers.
4. Spec hardening (b27cd716, 55dc49b2) — REGION_*-indexed assignment + shared raw_*_bytes;
   adopt in any new/edited spec (upstream did rmsnorm/argmax; our newer specs should match).
5. Renames + promotion (06ff516d, 2b6a7461, 11a609f3) — cosmetic but pervasive; every future
   `git diff` vs upstream will be dominated by Launcher→Mma/Consumer→Compute and v3→v2 unless we
   adopt them at the next sync.
6. **No new protocol fixes beyond our merge-base** in the consumer_done / stale-re-init class:
   the channel reinit_full/empty, reinit-policy tables, eager OOO trigger, proxy fence and
   BEGIN_TASK_GRAPH parity are all already in our tree (verified by grep/diff 2026-07-15); the
   post-merge upstream work is single-sourcing, renames, dead-code deletion and comment accuracy.
