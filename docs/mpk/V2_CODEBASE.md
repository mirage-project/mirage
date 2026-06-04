# MPK v2 runtime — codebase walkthrough

A reading guide for the v2 (warp-specialized, scheduler-less) persistent
megakernel. Ordered by data flow: what happens at compile time, then at
runtime init, then on the GPU. Known debt lives in [V2_TODO.md](V2_TODO.md).

```
COMPILE TIME (python driver, offline)
  demo.py / PersistentKernel.compile()
    └─ kn_graph.generate_task_graph()           C++ transpiler
         ├─ task_register.cc                    per-op: role bodies + SMEM spec
         ├─ v2_role_codegen.cc                  emits the 5 role dispatchers
         └─ runtime.cc                          serializes task_graph JSON
    └─ build_v2_worker_task_queues()            python: static per-SM schedule
    └─ add_v2_region_smem_plan()                python: page planner
    └─ nvcc                                     compiles the megakernel .so

RUNTIME INIT (once, on the box)
  generated init code                           parses task_graph.json →
                                                device TaskDesc[] (incl. page plan)
  build_v2_plan()                               re-derives the per-SM schedule (debt:
                                                should read the JSON queues instead)

RUNTIME (per decode step, single kernel launch, 128 SMs x 256 threads)
  worker_v2_kernel                              8 warps/SM, role-specialized
```

---

## 1. File map

| File | Role |
|---|---|
| `include/mirage/kernel/task_register.h` | registration API + the 3 contract types (§2) |
| `src/kernel/task_register.cc` | per-op registrars: emit role bodies, declare SMEM regions |
| `src/kernel/v2_role_codegen.cc` | turns role bodies into the 5 `_execute_*_task_v2` dispatchers; owns the page-protocol prefix/suffix snippets |
| `src/kernel/runtime.cc` | task-graph serialization (JSON out) + generated init code (JSON back in) |
| `src/kernel/graph.cc` | `register_task(name, params)` — op-name → registrar dispatch |
| `python/mirage/mpk/v2_task_schedule.py` | static per-SM task queues (round-robin over events) |
| `python/mirage/mpk/v2_smem_planner.py` | packs each task's regions onto 14 physical SMEM pages |
| `python/mirage/mpk/page_plan_viz.py` | `--profiling` figure: per-SM page usage across tasks |
| `python/mirage/mpk/persistent_kernel.py` | compile() driver: wires the above, runs nvcc |
| `include/mirage/persistent_kernel/runtime_v2.cuh` | the device runtime: warp loops, instruction ring, page semaphores (§5) |
| `include/mirage/persistent_kernel/dispatch_v2.cuh` | includes the v2 task headers so generated calls resolve |
| `include/mirage/persistent_kernel/persistent_kernel_v2.cuh` | host launch path + `build_v2_plan` |
| `include/mirage/persistent_kernel/tasks/blackwell_v2/` | the task kernels (§6) |
| `include/mirage/persistent_kernel/runtime_header.h` | `TaskDesc`, `RuntimeConfig`, enums — shared host/device |

## 2. The three contract types (task_register.h)

**`TaskSmemRegion` / `TaskSmemInfo`** — a task's *declared SMEM shape*; the
C++ → Python planner contract. Each v2 task's spec header (e.g.
`linear_spec.h::make_smem_info()`) declares its regions:

- `size`, `alignment` — bytes; alignment 1024 for TMA 128B-swizzle targets
- `page_count` — full 4KB pages needed (or -1 for sub-page)
- `can_pack` — sub-page regions may share a physical page (linear's six 4KB
  A-stages pack onto two pages)
- `release_step` — when within the task the region's data dies; drives the
  planner's cross-task page-reuse order
- `contiguous` — must occupy adjacent physical pages (W spans 2 pages so one
  TMA covers it)

Flow: `register_variant_smem_info()` stores per (TaskType, variant_id) →
serialization (`runtime.cc:1241`) writes `smem_regions` into each task's JSON →
the planner consumes exactly those fields → emits `planned_smem_page_regions` →
generated init code (`runtime.cc:645`) parses the plan into the device
`TaskDesc.smem_regions`. The device runtime then addresses SMEM through
`task_desc->smem_region_offset(REGION_*)`.

**`TaskRoleVariantCode`** — a task's *generated device code*; the C++ →
codegen contract. Five role-body strings (`init_semaphores`, `loader`,
`launcher`, `consumer`, `storer` — empty string = role idles for this task)
plus two page-protocol flags:

- `auto_loader_page_lifecycle` (default on): codegen prepends every loader
  body with the lane-parallel page prefix — wait each page's release by the
  prior task; immediately finish pages this task does NOT use.
- `auto_consumer_finish` (default on): codegen appends the consumer suffix —
  finish the pages this task DOES use. Linear opts out and releases pages
  from its launcher instead.

Consumed by `generate_v2_role_dispatch_code` (v2_role_codegen.cc), which emits
five `switch(task_type)` dispatchers compiled into the megakernel.

Dead: `register_variant_smem_size` / `get_variant_smem_size` (scalar-size
variants; everything uses the full info) — TODO #11.

## 3. Compile-time pipeline (python)

After `generate_task_graph()` returns the JSON, two pure JSON→JSON passes run
(persistent_kernel.py ~1775, inside `compile()`):

1. **`build_v2_worker_task_queues(graph, num_workers)`** — the static
   schedule. Walks `all_events` in order; for each task-pushing event type
   (`EVENT_LAUNCH_TASKS`/`_MASSIVE`/`_DEPENDENT`, 901–903) deals its
   `[first_task_id, last_task_id)` range round-robin across SMs with a
   cursor that is continuous across events; prepends task 1
   (`BEGIN_TASK_GRAPH`) to worker 0. Qwen3-8B: 14,875 tasks/iter ≈ 116/SM.
   The order does NOT encode dependencies — those stay runtime (event
   counters, §5). NOTE: `build_v2_plan` (C++) re-derives this at init; the
   two must stay in lockstep (TODO #1: make C++ read the JSON queues).

2. **`add_v2_region_smem_plan(json)`** — the page planner. Per task: pack
   declared regions onto 14 logical pages (full-page reservations first-fit;
   `can_pack` regions share pages via offset checks), then map logical →
   physical. Single-page regions follow `preferred_physical_order` = the
   *previous* task-on-this-SM's release order (earliest-released first), so
   when cross-task overlap turns on, a task's earliest-needed data lands in
   the pages freed soonest. Multi-page regions currently take the lowest
   free contiguous run (TODO #4). Output per task:
   `planned_smem_page_regions: [{name, physical_pages, byte_offset,
   release_step}, ...]`.

`--profiling --use-v2` renders this plan: `page_plan_sm0_rank{rank}.png`
(rows = SM 0's tasks in order, columns = physical pages 0–13).

## 4. Runtime init

- Generated `construct_task_graph` (emitted by runtime.cc) parses
  `task_graph.json`: builds the device `all_tasks[]` (incl.
  `num_smem_regions`/`smem_regions` from the plan), `all_events[]`,
  event counters.
- `init_persistent_kernel_v2()` → `build_v2_plan()`
  (persistent_kernel_v2.cuh): re-runs the §3.1 round-robin on `all_events`,
  uploads `v2_per_sm_task_offsets[129]` / `v2_per_sm_task_positions[]`,
  allocates the cross-SM iteration barrier counters.
- `launch_persistent_kernel_v2()` per generation: v1's `prepare_kernel`
  (zero event counters) → reset iter counters → ONE launch of
  `worker_v2_kernel` that loops all decode steps on-device.

## 5. The device runtime (runtime_v2.cuh)

**Warp layout** (8 warps = 256 threads/SM): W0–3 consumer (128 compute
threads), W4 loader (TMA in), W5 launcher (tcgen05 MMA + TMEM), W6 storer,
W7 controller.

**Instruction ring**: 3 SMEM slots, each a `TaskDesc` copy + 2 mbarriers.
The controller walks this SM's static task list; per task it:
1. waits the slot's previous occupant via `INSTRUCTION_FINISHED[slot]`
   (count = 7 role warps), eagerly triggering finished tasks' events while
   spinning (out-of-order triggering breaks producer-later-in-ring cycles);
2. cp.async-copies the TaskDesc from `all_tasks[]` into `task_buf[slot]`,
   then `fence.proxy.async.shared::cta` (publish ASYNC-proxy writes to the
   generic proxy — hardening, see the comment there);
3. runs the task's `init_semaphores` body (single thread) — inits the
   op's dynamic semaphores in `dynamic_semaphores[slot][*]`;
4. arrives `INSTRUCTION_ARRIVED[slot]`. All role warp loops wake on that,
   dispatch by `task_type`, and arrive `INSTRUCTION_FINISHED[slot]` when
   their role body returns. Phase parity = `(sequence / 3) & 1`.

**Page protocol**: 14 page-sized SMEM regions, each guarded by a count-1
`page_finished[page]` mbarrier with parity keyed to `instruction_index & 1`.
Invariant: every task arrives every page EXACTLY once. The loader prefix
finishes unused pages; used pages are finished by the consumer suffix
(default) or the task's own release point (linear: launcher blanket,
lane-parallel). One missed/double arrive = permanent parity desync = the
next task's loader deadlocks. This was the 1-in-12 hang: the launcher's MMA
loop runs under `if (elect_sync())`, and on Volta+ independent thread
scheduling the non-elected lanes do NOT reconverge after the if-block — they
freed pages while the MMA still read them. Fix: `__syncwarp()` before the
blanket free (verified 40/40; see comment at linear_sm100_v3.cuh launcher).

**Dependencies & events**: unchanged from v1 semantics. Each task may have a
`dependent_event`; consumer warp 0 thread 0 spins on the global event counter
(`consumer_dep_prefix`), then releases the other role warps via the slot's
`SEM_DEP_READY` semaphore. Finished tasks' `trigger_event` counters are
bumped by the controller. Iteration boundary: all SMs sync on a global
counter; SM 0's controller runs `prepare_next_batch` between steps and
broadcasts early-exit on EOS via `g_v2_gen_done`.

## 6. Tasks (blackwell_v2/) and the Channel abstraction

Non-linear tasks (rmsnorm, rotary, attention, argmax, silu_mul, embedding)
are v1 kernel bodies re-registered under `TASK_*_V2` enums so the whole
graph dispatches through v2; they run in the consumer warps and rely on the
default page prefix/suffix. Attention still declares one monolithic region
(TODO #3).

Linear v3 (`linear_sm100_v3.cuh`) is the fully warp-specialized op:
loader TMAs W/A stages, launcher drives tcgen05 MMAs + owns TMEM,
consumers read TMEM and store (optional residual add). It is built on the
Channel abstraction (§6.1).

### 6.1 The Channel abstraction (channel.cuh)

**The problem it solves.** A warp-specialized op is N independent warp
functions coordinating through raw mbarriers. Before Channel, the v2 linear
hand-tracked, in EACH role function, three pieces of state that must agree
across all of them:

1. the **stage index** — which of the 6 ring slots we're on;
2. the **phase bit** — mbarrier parity (see below), with a different initial
   value per side (`mma_phase=1`, `tma_phase=0`, `epilogue_phase=1`,
   `mainloop_phase=0`) and a flip on every ring wrap;
3. the **address arithmetic** — `dyn_sem_base + SEM_MMA_BASE*8 + stage*8`.

Four hand-rolled copies of the same bookkeeping = four chances to drift by
one. That drift IS the deadlock class behind the 2026-05 v3 bring-up hang:
one role waits a barrier the other side already passed (or never armed), the
task never finishes, the whole GPU starves. Channel exists to make that
drift structurally impossible: the **cursor owns the stage+phase**, roles
never compute either.

**The design rule: sync ≠ storage.** Three primitives that compose instead
of one "pipeline" object:

- `Channel<DEPTH, PROD, CONS>` — synchronization ONLY: `full[DEPTH]` /
  `empty[DEPTH]` mbarrier addresses (stride 8 in the slot's
  dynamic-semaphore block). No data pointers. The `By::Warp/Tma/Mma` tags
  document *who arrives* each edge — `Warp` = a sync thread arrival,
  `Tma`/`Mma` = the hardware engine arrives asynchronously (these are the
  edges that need stale-arrival defense, below).
- `SmemRing<DEPTH, PAGES_PER_SLOT>` — storage ONLY: per-stage SMEM byte
  offsets (planner-fed via `smem_region_offset`, no contiguity assumed),
  plus, when `PAGES_PER_SLOT > 0`, the physical page IDs each stage owns and
  the cross-task page lifecycle (`acquire`/`release`/`owns`) for Phase-E
  overlap. With `PAGES_PER_SLOT == 0` all page methods compile to nothing.
  Unlike the constexpr CHANNELS table, the ring's CONTENTS are per-task
  runtime data — the device-side materialization of the planner's page
  assignment, rebuilt by `make_wa()` from the TaskDesc each task. It has two
  faces: intra-task it is the channel's address book (`slot_addr(cursor.st)`);
  cross-task its page methods are thin adapters onto the runtime's
  `page_finished` parity protocol (`acquire` = wait the PRIOR task's free of
  this stage's pages; `release` = publish them to the NEXT task), passed as
  functors so channel.cuh never includes the runtime. W rings get
  `PAGES_PER_SLOT=2` (each stage owns 2 dedicated pages — safe to release
  per-stage); A rings are permanently 0 because A regions PACK (A_0 and A_1
  share a physical page — a per-stage release would free a page another
  stage still owns).
- `TmemChannel<SLOTS, ...>` — same sync, but the data lives in TMEM:
  the cursor returns a column address (`taddr + st * cols_per_slot`)
  instead of an SMEM offset. (Storage is trivially linear in stage, so no
  separate ring needed.)

Why split? Because the real pipeline doesn't pair 1 sync : 1 storage.
Linear's W and A streams have **separate storage** (different rings,
different pages) but a **shared release edge**: the launcher issues ONE
`tcgen05.commit(mma_mbar[s])` per K-iteration that frees the W *and* A
stage together. With sync split from storage that's trivial — both
channels point at the same `empty` address; W's cursor does the one
`wait_free()` per iter, A's cursor just advances in lockstep. A fused
abstraction would have needed a special case.

**Cursors.** `Producer<Ch>` / `Consumer<Ch>` (and the Tmem variants) hold
`{st, ph, n_commits}` per side:

```cpp
// loader (producer side of W):
pW.wait_free();                          // wait empty[st] at phase ph
int W_smem = Wr.slot_addr(pW.st);        // storage looked up BY cursor stage
tma_3d_load_l2(W_smem, ..., pW.full_mbar(), ...);  // TMA will arrive full[st]
pW.commit_tma();                         // advance st, flip ph on wrap

// launcher (consumer side of W):
cW.wait_full();                          // wait full[st] — TMA landed
mma_k_block(tmem, Wr.slot_addr(cW.st), ...);
cW.release_mma();                        // tcgen05.commit arrives empty[st]
```

**The phase-parity trick** (why no re-init per stage): a count-N mbarrier
flips an internal phase bit every time N arrivals land; waiting means "spin
until the barrier's phase differs from MY expected bit". So the same
barrier is reused forever — each side just tracks which phase it expects
and flips its expectation on ring wrap (`advance()` does this). Initial
values encode the empty pipeline: the producer starts `ph=1` — a fresh
barrier sits at phase 0, so the first DEPTH `wait_free()`s pass immediately
("pre-empty": all slots start free, nobody had to arrive them) — while the
consumer starts `ph=0`, so its first `wait_full()` genuinely blocks until
the first commit. Those two init values replace v2's four loose
`*_phase` ints.

**Stale async arrivals — the ring-slot reuse hazard.** TMA byte delivery
and `tcgen05.commit` arrive barriers *asynchronously*: they can land after
the task body already returned and the instruction ring slot was recycled
to the NEXT op. A stray arrival on a recycled slot's barrier corrupts its
count/phase for the new occupant. Two defenses exist in the code base:

- **Start-of-task re-init** (the one in use): the role that touches an edge
  FIRST re-inits it (`reinit_for_role`, table-driven from
  `linear_spec.h CHANNELS[].reinit_*_by`). Safe because the matching waiter
  can't be waiting yet — the producer's first arrive is what unblocks the
  consumer's first wait. This placement is LOAD-BEARING and verified.
- `Producer::drain()` (available, currently unused by linear): wait
  `min(n_commits, depth)` outstanding empty arrivals before returning, so
  nothing is in flight at slot recycle. The end-of-loader drain was tried
  and REMOVED (2026-05-30) — it tangled with cross-op slot reuse and caused
  its own intermittent hang; the re-inits are the proven defense.

**What it costs: nothing.** Everything is `__forceinline__` + constexpr
ring depths; the cursors are 3 ints in registers; the emitted SASS was
validated byte-equivalent to the hand-rolled v2 op sequence during
bring-up. What stays OUTSIDE the abstraction (deliberately): the two
one-shot handshakes (`tmem_ready`, `consumer_done`) — single arrive→wait
pairs, a ring would be ceremony — and the controller-level instruction
ring / page protocol, which belong to the runtime, not the op.

**Where to read:** `channel.cuh` (primitives, ~200 lines),
`linear_spec.h` (the CHANNELS table = the op's dataflow graph as data, with
static_asserts tying it to the SEM_* ordinals), `linear_sm100_v3.cuh`
`make_wa()` (wiring), the three role functions (usage).

### 6.2 Linear v3 specifics

W and A share one empty edge (one tcgen05.commit frees both per K-iter).
Stale-arrival defense on ring-slot reuse = start-of-task re-init
(`reinit_for_role`), not an end-of-task drain (see §6.1). The launcher's
MMA loop runs under `if (elect_sync())` and MUST `__syncwarp()` before the
lane-parallel blanket page-free (the verified 1-in-12 hang fix).
`tiles_per_task` must be 1 (TODO #2).

## 7. Where to look when something hangs

1. Attach `cuda-gdb -p <pid>` to the ALREADY-hung process (launching under
   gdb perturbs the race away). Histogram all warps' PCs to find the one
   block stuck inside a task body; everyone else starves downstream in
   `consumer_dep_prefix`.
2. Read the stuck block's mbarrier words (`@shared` casts): count-1 page
   mbarriers off-parity ⇒ page-protocol violation (find the task that
   skipped/doubled an arrive). Pristine `W_tma0` + consumer at `mainloop0`
   ⇒ the loader never produced (check its early-return paths).
3. Do NOT add instrumentation to the page hot path — even a plain SMEM
   write suppresses the race. Verify fixes with 40-run stress loops; a
   single hang means the fix is wrong ("hang is hang").
