# AllReduce rewrite — design alternatives for user review

Started 2026-05-12 by Claude. AR is the single biggest closed-pr perf target after
the FP8_DENSE_NUM_WORKERS win (+6.3% via env override).

## What we know (data, not theory)

DSv3 prefill-128 TP=4 EP=2 layers 0-19, phase-isolated with `MPK_AR_SKIP_BARRIER` /
`MPK_AR_SKIP_REDUCE` compile gates (committed 2026-05-12 as
`d6d1730a allreduce: opt-in debug gates`):

| Variant       | per-AR-task | e2e        | delta vs baseline |
|---------------|-------------|------------|-------------------|
| baseline      | 370.4 μs    | 182.72 ms  | —                 |
| skip_barrier  | 279.0 μs    | 164.25 ms  | -18.5 ms (-10.1%) |
| skip_reduce   | 149.5 μs    | 168.97 ms  | -13.75 ms (-7.5%) |

Per-task decomposition (additive: 91 + 221 + 58 ≈ 370):
- **Barrier 91 μs (24%)** — `mpkar_sync_block` dissemination via team psync_pool
- **NVLS reduce 221 μs (60%)** — `multimem.ld_reduce` for one 128-col tile (~32KB)
- **Other 58 μs (16%)** — dep_event spin-wait + writeback fence

End-to-end "wall time saved per AR phase":
- Drop barrier: 18.5ms / 40 phases = **460 μs/phase saved** (vs per-task 91μs → ~5× amplification)
- Drop reduce: 13.75ms / 40 phases = **343 μs/phase saved** (vs per-task 221μs → ~1.5× amplification)

The 5× barrier amplification implies severe **contention** when 56 concurrent
AR-tile tasks all hit the same team psync_pool slot. This is the optimization
target; the reduce path is already running at roughly its hardware-bound time.

Reference target: vLLM custom AR is **6-8 μs/call** for a similar-size reduce.
We're at 370 μs/AR-phase (decode 14.7 μs/call). Closing the gap fully would save
~14ms (8%) of e2e on prefill-128.

## Architectural starting points

1. The AR layer is registered via `multigpu.py::AllReduceStrategy_NvshmemTile`
   (one `TASK_NVSHMEM_TILE_ALLREDUCE` per (per-PE) 128-wide column tile).
2. Each AR task: `mpkar_sync_block(team)` → `multimem.ld_reduce` from MC ptr →
   write to local HBM → `__threadfence` → `__syncthreads`.
3. The dissemination barrier uses a 2-buffered psync slot pair indexed by
   `sync_counter[0] % 2`. All 56 concurrent callers on each PE read/increment
   the SAME `sync_counter[0]` (non-atomically), and all signal/wait on slot
   `pSync[mype]` of the same buffer. This is the contention source.
4. `multimem.ld_reduce` is a hardware-multiplexed read of all peers' data with
   sum, executed entirely on NVSwitch SHARP. Each task reduces ~32KB
   (`128 cols × 128 rows × 2B bf16`).

## Design alternatives (ranked from least to most invasive)

### Option A — Per-task barrier slots (in-place fix, single-file)

**Modify** `allreduce.cuh::mpkar_sync_block`-and-friends to take `task_offset` and
index into private psync slots per task. Use atomic-increment for the per-task
phase counter so 56 callers don't contend.

Layout (using slots `2*MPKAR_NVSHMEMI_SYNC_SIZE` and beyond, currently
reserved-but-unused by the SYNC op):
- `task_counters[task_offset]` — atomic 64-bit phase counter, one per task
- `task_pSync[task_offset * npes * 2 + (phase % 2) * npes + mype]` — per-task signal slot

**Pros**:
- Single file, no new task type, no builder change.
- Easy to gate behind `MPK_AR_PER_TASK_BARRIER=1` env var for A/B testing.
- Easy revert.

**Cons / risks**:
- Cross-PE sync correctness must be verified: each PE's task counters advance
  independently via atomicAdd; under task-graph stationarity (each task fires
  exactly once per AR phase on each PE) they stay in lockstep, but ANY
  per-PE asymmetry (e.g., MTP draft-step skipping, gate-mode early-return)
  could leave PE counters diverged → deadlock.
- Pre-init: each new (team, task_offset) pair starts at counter=0; first call's
  signal value also 0, which the spin `*pSync >= 0` trivially satisfies (no
  actual cross-PE sync on first call). Must use `atomicAdd(...)+1` so first
  call signals value 1.

**Expected speedup**: best case recovers most of the 18.5ms barrier saving
(after the per-task latency for the simpler signal/wait round-trip), perhaps
14-17 ms. Worst case (if contention isn't the dominant cost): noise.

**Estimated implementation**: 1-2 hours coding, 30 min testing (Qwen3
correctness + DSv3 layers 0-3 cosine + perfetto re-measure).

### Option B — Single barrier task per AR phase (structural)

**Add** a new `TASK_NVSHMEM_ALLREDUCE_BARRIER` task type. One grid=(1,1,1)
task per AR phase that does the cross-PE barrier. The 56 tile-AR tasks then
skip their internal barrier (compile-time via `MPK_AR_SKIP_INTERNAL_BARRIER`)
and depend on the barrier task's output via the existing event-chain.

To preserve the dependency from upstream → AR-tile while letting the barrier
sit between them, the barrier task takes the upstream tensor as input and
emits a 1-element dummy output that AR-tile tasks consume as a synchronization
input. (Alternative: barrier memcpy's the full input → output and AR-tile
reads barrier's output; adds 1 HBM-pass per phase ≈ 1-2 μs, fine.)

Files to touch (~10):
- `include/mirage/persistent_kernel/runtime_header.h` — add enum
- `include/mirage/persistent_kernel/tasks/blackwell/allreduce.cuh` — barrier kernel
- `include/mirage/kernel/task_register.h` — decl
- `src/kernel/task_register.cc` — registration
- `src/kernel/graph.cc` — name dispatch
- `src/kernel/runtime.cc` — task_offset metadata + name table
- `python/mirage/mpk/multigpu.py` — new `AllReduceStrategy_NvshmemTileSingleBarrier`
- `python/mirage/mpk/persistent_kernel.py` — propagate compile macro
- `python/mirage/mpk/models/deepseek_v3/builder.py` — insert barrier in `_allreduce_residual`

**Pros**:
- Structurally clean: one place expresses "barrier across team", another expresses
  "reduce a tile". Maps to vLLM/SGLang AR design.
- Per-task contention disappears (1 caller per phase per PE).
- Foundation for further AR optimizations (fine-grained MLA↔AR event chains).

**Cons / risks**:
- Larger surface: 10 files, scheduler interaction with the new task type,
  output-tensor aliasing semantics for the dummy ready-flag.
- The barrier task adds an extra event-chain hop: upstream → barrier → AR-tile
  → consumer. Adds ~6-12 μs/phase of barrier task latency × 40 phases ≈
  0.5ms. Net savings still ~17 ms.

**Estimated implementation**: 3-5 hours coding (most spent on tensor aliasing
through annotated_graph.cc + correctness debug), 1+ hour testing.

### Option C — Eliminate the cross-PE barrier entirely (most ambitious)

Use the FACT that `multimem.ld_reduce` reads from all peers' HBM. If we ensure
peer's writes are HBM-coherent before the load, we don't need an explicit
barrier — the producer-side write fence on the upstream task (e.g., MLA reduce)
plus a `__threadfence_system()` should suffice.

But: the MPK event mechanism only syncs WITHIN a PE. Cross-PE coherency for
peer's data requires either:
(i) An NVSHMEM-aware signaling channel (= some form of barrier in disguise);
(ii) Or: structure the task graph so upstream writes are always synchronously
visible across PEs by the time AR fires.

(ii) means the upstream task's writes need to fence-system. If we add
`__threadfence_system()` to the trigger-event-firing path of every task that
produces AR input, the AR side can skip its barrier.

**Pros**: removes the barrier entirely; AR per-phase wall time → close to
just the reduce (~340 μs from skip_barrier data ÷ 40 = 8.5 μs/phase, vs vLLM
6-8 μs — competitive!).

**Cons / risks**:
- Adds `__threadfence_system` overhead to EVERY task that produces AR input
  (every MLA reduce, every MoE moe_mul_sum_add). System fences are slow
  (~hundreds of cycles each); could cancel out the AR savings.
- Requires identifying ALL upstream producers — risk of missing one and
  producing wrong AR output.
- Doesn't generalize to non-NVLS AR paths (e.g., AllgatherReduce fallback).

**Estimated implementation**: not estimable without an empirical sub-experiment
(measure `__threadfence_system` cost on a producer kernel).

## Recommendation

**Start with Option A** (per-task barrier slots). Single file, easy to revert,
covers the dominant cost, and ALSO informs Option B (if Option A doesn't help
enough, contention isn't the main cost and we know to design Option B around
something else).

If Option A delivers most of the projected savings (≥ 12 ms e2e), declare
victory and skip Option B unless the user wants further headroom.

If Option A is noise, then contention isn't the dominant cost and Option C
(no-barrier) is the right ambition target; B doesn't help because the single
barrier task itself would still cost ~90 μs.

## Open design questions (for user)

1. Is Option A's stationarity assumption (per-task counters stay in lockstep
   across PEs) safe across MTP draft loops and `runtime_m_mode=1` gated tasks?
   Need to audit `_allreduce_residual` call sites to confirm no PE skips a call
   asymmetrically.
2. For Option C, do we have a reasonable workload to micro-benchmark
   `__threadfence_system` cost on B200 SM100? The MLA reduce task is the
   primary upstream producer; adding a fence there is the obvious test.
3. Should the AR rewrite default to "on" once correct, or stay env-gated like
   `MPK_RDC_FALSE`? (Implies: revert if a regression bisects to AR.)

## Trace artifacts (for reproducibility)

- `outputs/ar_bench_112933/{baseline,skip_barrier,skip_reduce}/run.log` — full
  demo logs with per-token latency + AR cluster stats
- `outputs/ar_bench_112933/summary.txt` — one-line per variant
