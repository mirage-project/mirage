# M4-I8 — where the step-minus-critical-path gap actually goes

**Answer: the gap is real and mostly irreducible, and closing all of it still does not reach
AC-4.** The step decomposes exactly. The two largest terms are the dependency chain's own task
time and the time the chain spends waiting behind unrelated work on its statically assigned
worker. Neither is recoverable by a scheduling policy: a perfect 128-server list schedule over
the same tasks buys 5.4–6.0%, and every reorder window from 2 to 32 buys 0.3–2.9%. What would
buy 18–28% is removing the per-dependency-level dispatch latency, and the one-line change that
should have removed part of it — the only `ld.acquire.sys` in the runtime, on the kernel's
hottest spin — measured as an exact null at every batch size.

The load-bearing number is not the gap at all. The floors computed from the same per-task
durations are

| bs | measured step | cp (exact) | work bound | binding floor | vLLM step | floor / vLLM |
|---|---|---|---|---|---|---|
| 1 | 5781.4 | **4130.7** | 2039.7 | 4130.7 | 3503.0 | **1.179x** |
| 8 | 8272.9 | **5275.7** | 4280.2 | 5275.7 | 4727.0 | **1.116x** |
| 16 | 10257.4 | 5638.7 | **5976.0** | 5976.0 | 5301.0 | **1.127x** |

so a step with **perfect packing and zero dispatch overhead** is still 12–18% slower than vLLM's
whole step. AC-4 does not need a better scheduler. It needs a shorter dependency chain (bs1, bs8)
and less total task work (bs16).

This corrects M4-I8's own framing, which read `step/cp -> 1.0` off M4-status's cp as
`1.17x / 1.02x / 1.03x` of vLLM. Two independent errors pushed that low: at bs16 the work bound
(5976.0) *exceeds* cp, so `step/cp = 1.0` is arithmetically unreachable there; and
`cp_decompose.py`'s levelmax weighting charges each level its type's **mean** live duration, which
understates a level whose binding producer is slower than its type's mean — at bs8 the exact path
is 5275.7 against levelmax's 4803.0, 9.8% higher.

---

## 1. Why the decomposition is exact rather than modelled

`sched_gap.py` reconstructs the realized schedule instead of modelling it. That is possible
because of a structural fact in `src/kernel/runtime.cc:972-993`:

```c
  // Prelaunch all tasks at the begining of an iteration
  all_events[1].first_task_id = 2;
  all_events[1].last_task_id  = all_tasks.size();
  for (e = 2; e < all_events.size(); e++)
    if (LAUNCH_TASKS || LAUNCH_MASSIVE_TASKS) {
      all_events[e].event_type = EVENT_EMPTY;
      for (t in [first,last)) all_tasks[t].dependent_event = e;
    }
```

**MPK does not schedule dynamically.** Event 1 (`EVENT_LAUNCH_DEPENDENT_TASKS`, triggered by
`TASK_BEGIN_TASK_GRAPH`) pushes *every* task of the iteration into a worker queue at iteration
start; every other event is rewritten to `EVENT_EMPTY` and becomes a pure counter. Each worker
then drains its queue **strictly in order**, blocking on each task's `dependent_event` before it
may start (`persistent_kernel.cuh:981-1009`). The schedule is a static round-robin of the graph's
task order over 128 in-order blocking queues, fixed at compile time. Confirmed in the compiled
graph: 2275 of 2278 events are type 900 (`EVENT_EMPTY`), event 1 is type 903 with range
`[2, n_tasks)`.

The assignment follows from the scheduler's loop (`persistent_kernel.cuh:1328-1376`): the
scheduler owning workers `[f,l)` walks `position = first + i*num_workers + j` for `j` in `[f,l)`,
pushing each to `next_worker` round-robin inside its own range. With `num_workers == 128` that is
exactly `worker(p) = (p-2) mod 128`, order ascending in `p`. `get_first_last_ids(128, 80, s)`
gives schedulers 0–47 a *pair* of workers and 48–79 a single one, and a pair's persistent
`next_worker` can be left one step out of phase, which swaps that pair's two sequences — so the
reconstruction is fitted per pair over two candidates and then **verified against the trace**: the
profiler records a task type per event, so the predicted per-worker type sequence must equal the
observed one element for element on all 128 workers.

It does. `assign_qc: PASS` on every decomposed iteration (126 direct + 2 phase-swapped at bs1 and
bs8, 124 + 4 on one bs16 iteration), and the resulting partition of the measured step closes to
**0 ns** (`identity_error_ns`).

Walking backwards from the last record to finish, each record's start is explained by exactly one
binding predecessor — a **data** edge (its `dependent_event` had not fired; predecessor is the
last producer to finish) or a **resource** edge (the event had fired but the worker was still
busy; predecessor is the previous record in that worker's queue). Because a resource edge's
predecessor *is* the previous record, the sum telescopes and the identity is exact:

```
step = head + SUM(record durations) + SUM(data gaps) + SUM(resource gaps) + tail
```

### Anchor QC

Every cell passes the integer per-step task-count check (`width.py`'s `anchor_qc`, the check M4-I5
built after root-causing M3-I7's bs16 failure as profiler **slot exhaustion**): `verdict PASS`,
`worst_rel_err 0.0`, all types exact in every windowed iteration. `--slots 200000000` at 640
decode steps gives fill 37.9 / 48.7 / 61.5% and `exact_prefix_iterations` 655 / 767 / 814 — the
bs16 window `[720,733)` sits inside its exact prefix, and `dropped_begin` is 0 / 0 / 95 with
`dropped_end` 0 everywhere. A profiled iteration count is a floor, so this is the value that makes
it a measurement.

Same-basis control against M4-status (`opt/m4status/cp/late/`), which measured the same windows on
the same code at a different clone: step 5821.3 / 8347.7 / 10305.8 here against 5791.8 / 8350.6 /
10308.7 there, cp 4138.3 / 4803.0 / 5484.2 against 4112.0 / 4802.3 / 5482.1 — every cell within
0.6%.

---

## 2. The decomposition

Mean of the two decomposed iterations per cell; window and geometry are M4-I5's
(msl=897, 640 decode steps, mbt=16, page 256, 256-token synthetic prompts; windows
bs1 `[288,384)`, bs8 `[365,461)`, bs16 `[720,733)`).

| term | bs1 | % | bs8 | % | bs16 | % | what bounds it |
|---|---|---|---|---|---|---|---|
| **step (measured)** | 5781.4 | 100 | 8272.9 | 100 | 10257.4 | 100 | window / `BEGIN_TASK_GRAPH` deltas |
| PATH work — dependency-chain tasks | 3043.6 | 52.6 | 3589.4 | 43.4 | 3483.9 | 34.0 | profiled task durations on data edges |
| **QUEUE work — waited behind** | 1556.4 | 26.9 | 3424.7 | 41.4 | **5365.3** | **52.3** | profiled durations on resource edges |
| TRIGGER work — `TASK_SCHD_EVENTS` | 23.2 | 0.4 | 21.3 | 0.3 | 27.4 | 0.3 | 2277 records/step, 0.40–0.44 us each |
| data gap — event visibility | 643.4 | 11.1 | 636.9 | 7.7 | 576.7 | 5.6 | 483–514 data edges x ~1.15 us median |
| resource gap — queue pop | 506.3 | 8.8 | 575.5 | 7.0 | 760.9 | 7.4 | 301–471 resource edges x ~1.58 us median |
| tail — last record to boundary | 8.4 | 0.1 | 25.0 | 0.3 | 43.2 | 0.4 | direct |

Chain shape, which is what the gap is really about:

| bs | chain records | data edges | resource edges | dispatch latency | share of step |
|---|---|---|---|---|---|
| 1 | 757 task + 52 sev | 508 x 1152 ns | 301 x 1592 ns | 1149.7 us | 19.9% |
| 8 | 798 task + 47 sev | 514 x 1168 ns | 330 x 1584 ns | 1212.4 us | 14.7% |
| 16 | 884 task + 70 sev | 483 x 1136 ns | 471 x 1568 ns | 1337.6 us | 13.0% |

Two terms answer the acceptance criterion's four questions and two of them are refuted as levers:

* **per-task dispatch / event cost** = the two gap rows, 1149.7 / 1212.4 / 1337.6 us. It is
  `n_edges x per-edge latency`, i.e. it scales with the **depth of the chain**, not with the task
  count. 508 data edges over 40 layers is ~12.7 serial dependency levels per layer at bs1, each
  costing ~1.15 us before any work happens.
* **intra-level arrival spread** does not appear as a separate term, because with full fan-in it
  *is* the data gap plus the binding producer's own delay, which the backward walk attributes
  recursively.
* **worker idle on fan-in** is 473070 / 501030 / 525988 us of worker-time (128 workers x step), of
  which 437124 / 458307 / 470157 us is genuine starvation — the worker had nothing ready. Only
  30878 / 37599 / 49723 us is poll latency.
* **the tail** is 8.4 / 25.0 / 43.2 us, 0.1–0.4%. Refuted as a lever.
* **`TASK_SCHD_EVENTS`** is 0.3–0.4%. Refuted as a lever. Note it does *not* delay the consumer:
  the counter increment happens *before* the profiled `SCHD_EVENTS` pair, so its cost lands on the
  producing worker's occupancy, not on the dependency path.

---

## 3. Ranking the mechanisms, and the two that were refuted by measuring

`sched_gap.py` also drives a discrete-event simulator over the reconstructed assignment and the
same per-task durations. Its `slide_1` policy *is* MPK's policy, so reproducing the measured step
is the validation; everything else is a prediction. Deltas are against `slide_1`.

| policy | bs1 | Δ | bs8 | Δ | bs16 | Δ |
|---|---|---|---|---|---|---|
| `slide_1` — model of HEAD | 6517.2 | — | 8937.1 | — | 10764.5 | — |
| *measured step* | *5781.4* | *−11.3%* | *8272.9* | *−7.4%* | *10257.4* | *−4.7%* |
| `slide_2` | 6520.3 | +0.0% | 8937.1 | 0.0% | 10764.5 | 0.0% |
| `batch_8` — what a minimal kernel change gives | 6494.8 | −0.3% | 8861.7 | −0.8% | 10663.1 | −0.9% |
| `slide_4` … `slide_32` | 6362.6 | −2.4% | 8693.4 | −2.7% | 10453.8 | −2.9% |
| `list_schedule` — no worker affinity at all | 6124.0 | **−6.0%** | 8456.9 | **−5.4%** | 10149.0 | **−5.7%** |
| `slide_1_nolat` — zero dispatch latency | 4717.4 | **−27.6%** | 7081.8 | **−20.8%** | 8817.1 | **−18.1%** |
| `slide_8_nolat` | 4549.3 | −30.2% | 6830.3 | −23.6% | 8502.1 | −21.0% |
| `list_schedule_nolat` | 4372.9 | −32.9% | 6612.6 | −26.0% | 8239.0 | −23.5% |

The simulator over-predicts the step by 12.7 / 8.0 / 4.9%, because it charges the median per-edge
latency to every pop. Only the relative deltas are used.

**Ranked:**

1. **Per-dependency-level dispatch latency — 18.1 to 27.6%.** The only large lever.
2. **Static task-to-worker binding — 5.4 to 6.0%.** Even a perfect affinity-free list schedule at
   the current latency buys this much and no more. Latency and policy are roughly additive
   (at bs1: 1799.8 + 393.2 = 2193 against 2144.3 for both).
3. **Queue order — 0.3 to 2.9%,** saturating at window 4. Effectively refuted.
4. **Tail, `SCHD_EVENTS` — under 0.4% each.** Refuted.

### Two candidate directions the data refutes outright

**Relaxing full fan-in.** This was the direction with the most apparent leverage, and the data
kills it. Every event is a full fan-in barrier (`num_triggers == n_producers`, 2277/2277 at all
three batch sizes, re-verified here). But the first-order stall test says **no stall is caused by
it**: at every stall in the realized trace, of the next 64 tasks in that worker's queue,
`hol_full = 0` us — *not one* was already ready when the worker went free. That is not an accident.
Because tasks are prelaunched in graph position order and that order is topological, each worker's
queue is itself topologically ordered, so its head is always its most-ready task. In-order
draining is already optimal *given this assignment*. Only the partially-recoverable measure is
non-zero — 1833 / 2352 / 2493 us of the ~500000 us of idle, 0.4–0.5% — and it saturates at window
4, which is exactly what the simulator then confirms at −2.4 to −2.9%. Nothing here would justify
changing scheduler semantics, so the soundness question never arises.

**Generalising M4-I7's work-item flattening.** At current HEAD the wide stages are already packed:
`live/lvl` at bs1 is 96.2 (dense fp8), 64.0 (MoE w13), 126.7 (MoE w2), 128.0 (GDN recurrent). The
stages still far below 128 are quantize (0.3), silu-mul (0.0), rms-norm (0.0), router (1.0) and
attention (1.0) — and those are not worst-case-grid stages with a serial inner dimension, they are
either genuinely tiny or genuinely one-per-layer. The flattening pattern has no remaining large
target at bs1.

---

## 4. What was prototyped

Both arms are default-off compile-time knobs wired through
`python/mirage/mpk/persistent_kernel.py`, so both live in one tree and one generated TU. The
extension `core.so` md5 is unchanged by either (`9d3675267bc211ca7b39b0599421879a` before and
after rebuild): both are JIT-header-only.

### Arm S — `MPK_EVENT_WAIT_GPU_SCOPE=1`

`persistent_kernel.cuh:1002` is the only `ld.acquire.sys.u64` in the runtime, and it is the
kernel's hottest spin: every task executes its body at least once before it may start. The
counter it polls is written by `atom_add_release_gpu_u64` — **device** scope — and the branch it
sits in is the `!is_nvshmem_event` branch, so the counter is a local event's and is only ever
written by this device. `ld.acquire.sys` therefore asks for system-scope coherence that the
release it pairs with never provides. The change is `ld_acquire_sys_u64 -> ld_acquire_gpu_u64` in
that branch.

*Soundness:* airtight — `release.gpu` / `acquire.gpu` is the matching pair for a location written
only by device-scope atomics on this device, and nvshmem events take the other branch.
*Bit-exactness:* by construction. No arithmetic, no dtype, no task-to-data mapping, no ordering
change; the same monotone counter is read and the same predicate evaluated.

### Arm O — `MPK_WORKER_OOO_POP=1`

Run the first **ready** task in the already-loaded task-desc buffer instead of blocking on the
head; fall back to the head if none is ready. `TASK_DESCS_BUFFER_LENGTH` is 8 on sm100
(`WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE` 3 KiB, `sizeof(TaskDesc)` 352 B with
`MPK_ENABLE_TMA`), so the window is 8 and the readiness test is done by threads 0–7 in parallel
with one `atomicMin` into shared memory — 8 B of extra static smem, no extra global traffic on the
common path.

*Soundness argument, under the persistent work-queue model:* (1) a task is legal to run the
instant its `dependent_event` reaches `num_triggers * iteration_num`, which is the runtime's *only*
precondition — that is literally the predicate of the wait loop it replaces — and event counters
are monotone, so a promoted task has satisfied it and a deferred task cannot become unready;
(2) no deadlock, because `all_tasks` is emitted in topological order (M4-I5 verified 0 topological
violations on the compiled graph) and a worker's queue is a subsequence of it, so no task in the
buffer can be a producer for an *earlier* task in the same buffer — deferring a blocked head to
run a ready successor cannot withhold anything the head waits for; (3) the scan never looks past a
not-yet-run `TASK_TERMINATE`, so a worker cannot exit while tasks its peers depend on sit unrun in
its buffer. This is the same shape as M3-I3's rule: legal precisely because no task waits on a
peer. Nothing here needs a cross-task barrier, so nothing inadmissible is required.

Arm O was pre-registered as a **predicted null** (`batch_8`: −0.3% at bs1) and run as the
simulator's falsifier: a large measured effect would mean the decomposition's ranking is wrong.

---

## 5. Gate results

*(filled in by the gate runs; see `gates/` and `tables/`)*

---

## 6. Terminal disposition

*(see the summary at the top; the backlog entry is in `tables/`)*

---

## 7. Reproducing

```bash
# on the coordinator box
bash opt/m4i8/scripts/setup_m4i8.sh                     # own clone at HEAD + fresh extension
bash opt/m4i8/scripts/gpu_guard_m4i8.sh 1,0,3,6 -- \
     bash opt/m4i8/scripts/prof_m4i8.sh                  # profiled capture, 3 cells, ~6 min
bash opt/m4i8/scripts/derive_m4i8.sh                     # decomposition + verified sims, no GPU
bash opt/m4i8/scripts/gpu_guard_m4i8.sh 1,0,3,6 -- \
     bash opt/m4i8/scripts/drive_m4i8_ab.sh              # e2e A/B, arms interleaved
bash opt/m4i8/scripts/drive_m4i8_gates.sh                # ptxas/SASS, profiler control, floors
```

Raw profiler buffers are 0.9–1.5 GB per cell and are regenerable; they are not committed. The
derived per-iteration JSONs are in `raw/gap/`.
