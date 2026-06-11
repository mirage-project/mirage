/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/profiler.h"
#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/persistent_kernel/tasks/common/copy_sm80.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

// ── Optional per-task profiling (MPK_ENABLE_PROFILING builds only) ──────────
// Reuses v1's FlashInfer-style profiler format so profiler_persistent.py
// renders v2 traces unchanged. EIGHT tracks per SM (num_groups = 8): five role
// tracks (0-4 below) plus three stall tracks (5-7). The SM is a pipeline in v2,
// so each warp role gets its own Perfetto row and the task-level overlap
// between roles is directly visible:
//   group 0 compute  (warp 0 lane 0): BEGIN/END per task body — includes
//           the dep-wait spin; gaps = waiting on instruction publish
//   group 1 loader    (lane 0): page-prefix + loader body per task
//   group 2 mma  (lane 0): mma body (linear: TMEM/MMA driving)
//   group 3 storer    (lane 0): storer body (idle for the current tasks)
//   group 4 dispatcher(lane 0): V2_PROF_PREPARE_BATCH (worker 0),
//           V2_PROF_ITER_SYNC (end-of-iter barrier), V2_PROF_GO_WAIT
// Only the LAST V2_PROF_WINDOW_ITERS decode steps are recorded: that's the
// interesting regime (full KV length), and it bounds the event count so the
// fixed-size profiler buffer can't overflow on long runs.
// Non-profiling builds: all macros expand to nothing — zero impact.
static constexpr int V2_PROF_NUM_GROUPS = 8;
static constexpr int V2_PROF_GROUP_COMPUTE = 0;
static constexpr int V2_PROF_GROUP_LOADER = 1;
static constexpr int V2_PROF_GROUP_MMA = 2;
static constexpr int V2_PROF_GROUP_STORER = 3;
static constexpr int V2_PROF_GROUP_DISPATCHER = 4;
// Stall tracks: sub-slices WITHIN a role's task window, so a bar
// self-explains (wait vs work). Written by the closure-free emitter
// (v2_prof_emit*) because their call sites can't see the role-loop closure.
static constexpr int V2_PROF_GROUP_COMPUTE_STALL = 5;
static constexpr int V2_PROF_GROUP_LOADER_STALL = 6;
static constexpr int V2_PROF_GROUP_MMA_STALL = 7;
static constexpr int V2_PROF_PREPARE_BATCH = 204;
static constexpr int V2_PROF_ITER_SYNC = 205;
static constexpr int V2_PROF_GO_WAIT = 206;
static constexpr int V2_PROF_DEP_WAIT = 207;   // compute-stall: dep spin
static constexpr int V2_PROF_PAGE_WAIT = 208;  // loader-stall: page prefix
// Timed-wait ids (emitted by MPK_V2_TIMED_WAIT at synchronization points;
// only waits > V2_PROF_WAIT_THRESHOLD_NS produce slices):
static constexpr int V2_PROF_W_TMA_WAIT = 209;        // mma: TMA landed?
static constexpr int V2_PROF_MMA_EMPTY_WAIT = 210;    // loader: stage free?
static constexpr int V2_PROF_TMEM_READY_WAIT = 211;   // compute: tmem addr
static constexpr int V2_PROF_MAINLOOP_WAIT = 212;     // compute: MMA result
static constexpr int V2_PROF_EPILOGUE_WAIT = 213;     // mma: tmem slot
static constexpr int V2_PROF_COMPUTE_DONE_WAIT = 214;// mma tail
static constexpr unsigned long long V2_PROF_WAIT_THRESHOLD_NS = 2000;
// Total profiler-buffer entries — MUST match demo.py's profiler_tensor size.
// Sized for 8 tracks x 128 SMs x 25 windowed iters; the busiest track
// (compute-stall: dep + tmem + mainloop slices) can write ~12-17k entries,
// so per-track capacity = (ENTRIES - tail) / (128*8) ≈ 15k. The emitter
// counts (never silently drops) overflow in the MISC region.
static constexpr size_t V2_PROF_BUF_ENTRIES = 120000ull * 128;
// Tail of the profiler buffer reserved for accumulators (all in NANOSECONDS
// via %globaltimer — same timebase as the trace events, no clock-rate
// conversion). CONVENTION with demo.py's profiler_tensor; debug-only.
// Layout, growing back from the end:
//   [SPIN_BASE + bucket*256 + sm]        dep-wait ns,   per task-type bucket
//   [SPIN_BASE + bucket*256 + 128 + sm]  dep-wait count
//   [SUFFIX_BASE + sm]                   page-suffix ns, per SM (aggregate)
//   [SUFFIX_BASE + 128 + sm]             page-suffix count
// Type buckets: 0=linear(244/245) 1=attn(284) 2=rmsnorm(281) 3=silu(282)
//               4=argmax(285/286) 5=embed(283) 6=other
static constexpr int V2_PROF_NUM_BUCKETS = 7;
static constexpr size_t V2_PROF_SPIN_BASE =
    V2_PROF_BUF_ENTRIES - 256ull * V2_PROF_NUM_BUCKETS - 256;
static constexpr size_t V2_PROF_SUFFIX_BASE = V2_PROF_BUF_ENTRIES - 256;
// Per-track write cursors for the closure-free emitter (1024 slots covers
// 128 SMs x 8 groups), then a misc region: [MISC_BASE + sm] counts events
// DROPPED by the capacity guard (must be 0 in a healthy run — the checker
// reports it). Everything from MISC_BASE on is reserved tail — the exporter
// skips it (V2_PROF_TAIL_ENTRIES in profiler_persistent.py must match).
static constexpr size_t V2_PROF_CURSOR_BASE =
    V2_PROF_SPIN_BASE - 1024;
static constexpr size_t V2_PROF_MISC_BASE = V2_PROF_CURSOR_BASE - 256;
// Event-trigger log: a single global ring recording every
// trigger_task_event() fired inside the profiling window, packed as
// [63:32]=globaltimer_lo  [31:8]=event_index  [7:0]=sm. One atomic cursor.
// Diagnoses trigger latency (when did event E actually fire, and from
// which SM's dispatcher).
static constexpr size_t V2_PROF_TRIG_RING_LEN = 1048576;
static constexpr size_t V2_PROF_TRIG_BASE =
    V2_PROF_MISC_BASE - V2_PROF_TRIG_RING_LEN;
static constexpr size_t V2_PROF_TRIG_CURSOR = V2_PROF_TRIG_BASE - 1;
static constexpr size_t V2_PROF_TAIL_ENTRIES =
    V2_PROF_BUF_ENTRIES - V2_PROF_TRIG_CURSOR;
static constexpr int V2_PROF_WINDOW_ITERS = 25;

__device__ __forceinline__ unsigned long long v2_prof_now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ int v2_prof_bucket(int task_type) {
  switch (task_type) {
    case 244: case 245: return 0;   // linear v2 (+residual)
    case 284: return 1;             // attention
    case 281: return 2;             // rmsnorm
    case 282: return 3;             // silu_mul
    case 285: case 286: return 4;   // argmax
    case 283: return 5;             // embedding
    default:  return 6;
  }
}

#ifdef MPK_ENABLE_PROFILING
// Closure-free trace-event emitter for sites that cannot see the role-loop
// profiler closure (runtime helpers like compute_dep_prefix, codegen-emitted
// prefixes). Maintains per-track cursors in the reserved tail; the entry
// layout matches PROFILER_INIT's interleaving exactly, so the exporter needs
// no changes. Single writer per stall track (a designated lane), so the
// cursor bump needs no atomics. NEVER write a role group (0-4) through this
// — those tracks are owned by the role-loop closures.
__device__ __forceinline__ void v2_prof_emit(void *prof_buf,
                                             int group,
                                             uint32_t event_idx,
                                             uint32_t event_type) {
  uint64_t *buf = static_cast<uint64_t *>(prof_buf);
  unsigned int const track = blockIdx.x * V2_PROF_NUM_GROUPS + group;
  unsigned long long const k = buf[V2_PROF_CURSOR_BASE + track]++;
  unsigned int const stride = gridDim.x * V2_PROF_NUM_GROUPS;
  size_t const slot = 1 + (size_t)k * stride + track;
  if (slot >= V2_PROF_MISC_BASE) {
    // capacity guard: never bleed into the reserved tail — but COUNT the
    // drop so the checker can flag a truncated trace.
    atomicAdd(reinterpret_cast<unsigned long long *>(
                  &buf[V2_PROF_MISC_BASE + blockIdx.x]), 1ULL);
    return;
  }
  uint32_t const event_no = (uint32_t)(k >> 1) & 0x3FF;  // pair counter
  tb::ProfilerEntry e;
  e.tag = tb::encode_tag(track, 0, 0) | (event_idx << tb::EVENT_IDX_SHIFT) |
          (event_no << tb::EVENT_NO_SHIFT) | event_type;
  e.delta_time = tb::get_timestamp();
  buf[slot] = e.raw;
}

// Retro-emit a BEGIN/END pair with explicit timestamps — used by
// MPK_V2_TIMED_WAIT after a wait completes, so a slice is written only when
// the wait was long enough to matter and pairs can never dangle.
__device__ __forceinline__ void v2_prof_emit_pair(void *prof_buf,
                                                  int group,
                                                  uint32_t event_idx,
                                                  unsigned long long t0_ns,
                                                  unsigned long long t1_ns) {
  uint64_t *buf = static_cast<uint64_t *>(prof_buf);
  unsigned int const track = blockIdx.x * V2_PROF_NUM_GROUPS + group;
  unsigned long long const k = buf[V2_PROF_CURSOR_BASE + track];
  buf[V2_PROF_CURSOR_BASE + track] = k + 2;
  unsigned int const stride = gridDim.x * V2_PROF_NUM_GROUPS;
  size_t const slot0 = 1 + (size_t)k * stride + track;
  size_t const slot1 = slot0 + stride;
  if (slot1 >= V2_PROF_MISC_BASE) {
    atomicAdd(reinterpret_cast<unsigned long long *>(
                  &buf[V2_PROF_MISC_BASE + blockIdx.x]), 2ULL);
    return;
  }
  uint32_t const event_no = (uint32_t)(k >> 1) & 0x3FF;
  uint32_t const base = tb::encode_tag(track, 0, 0) |
                        (event_idx << tb::EVENT_IDX_SHIFT) |
                        (event_no << tb::EVENT_NO_SHIFT);
  tb::ProfilerEntry e;
  e.tag = base | tb::EVENT_BEGIN;
  e.delta_time = (uint32_t)t0_ns;
  buf[slot0] = e.raw;
  e.tag = base | tb::EVENT_END;
  e.delta_time = (uint32_t)t1_ns;
  buf[slot1] = e.raw;
}

// Ambient profiling context, so synchronization sites inside task bodies can
// emit without threading (config, iter_num) through every signature:
//   g_v2_prof_buf    — the profiler buffer (same for all SMs); set once in
//                      the kernel prologue.
//   g_v2_prof_window — 1 while the current decode step is inside the traced
//                      window; flipped by worker 0's dispatcher at the
//                      iteration boundary (all SMs are barrier-synced per
//                      iter, so at most one task of fuzz at the edges).
__device__ void *g_v2_prof_buf = nullptr;
__device__ volatile int g_v2_prof_window = 0;

// Snapshot the ambient window flag ONCE per task body (a volatile global
// read per wait would put L2 traffic in hot loops for the whole run).
// Required in scope before any MPK_V2_TIMED_WAIT.
#define MPK_V2_PROF_SNAPSHOT()                                                \
  bool const _mpk_prof_on =                                                   \
      (g_v2_prof_window != 0) && (g_v2_prof_buf != nullptr);

// Time `expr` (a wait) and emit a stall slice if it exceeded the threshold.
// Call from a SINGLE thread per (SM, group) — the designated writer of that
// stall track.
#define MPK_V2_TIMED_WAIT(group, ev, expr)                                    \
  do {                                                                        \
    if (_mpk_prof_on) {                                                       \
      unsigned long long const _w0 = v2_prof_now_ns();                        \
      expr;                                                                   \
      unsigned long long const _w1 = v2_prof_now_ns();                        \
      if (_w1 - _w0 > V2_PROF_WAIT_THRESHOLD_NS) {                            \
        v2_prof_emit_pair(g_v2_prof_buf, (group), (ev), _w0, _w1);            \
      }                                                                       \
    } else {                                                                  \
      expr;                                                                   \
    }                                                                         \
  } while (0)
#endif
#ifdef MPK_ENABLE_PROFILING
#define MPK_V2_PROF_DECL(grp, pred)                                           \
  PROFILER_CLOSURE_PARAMS_DECL;                                               \
  uint32_t _prof_ctr = 0;                                                     \
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer), (grp),       \
                V2_PROF_NUM_GROUPS, (pred));
#define MPK_V2_PROF_IN_WINDOW(it)                                             \
  ((it) + V2_PROF_WINDOW_ITERS >= config.v2_max_iters)
#define MPK_V2_PROF_START(ev)                                                 \
  if (MPK_V2_PROF_IN_WINDOW(iter_num)) {                                      \
    PROFILER_EVENT_START((ev), _prof_ctr);                                    \
  }
#define MPK_V2_PROF_END(ev)                                                   \
  if (MPK_V2_PROF_IN_WINDOW(iter_num)) {                                      \
    PROFILER_EVENT_END((ev), _prof_ctr);                                      \
    _prof_ctr++;                                                              \
  }
// Conditionally-timed wait: time `expr` only when `cond` (e.g. cold-start lap),
// else run it plain. The cond/branch exists ONLY in profiling builds; the
// non-profiling form (below) is a bare `expr`, textually identical to baseline
// (sm100 codegen is sensitive to if/else around tcgen05 waits — see
// sm100_branch_ima). Keeps the production hot loop free of #ifdef blocks.
#define MPK_V2_TIMED_WAIT_IF(cond, group, ev, expr)                           \
  do {                                                                        \
    if (cond) {                                                               \
      MPK_V2_TIMED_WAIT(group, ev, expr);                                     \
    } else {                                                                  \
      expr;                                                                   \
    }                                                                         \
  } while (0)
#else
#define MPK_V2_PROF_DECL(grp, pred)
#define MPK_V2_PROF_START(ev)
#define MPK_V2_PROF_END(ev)
#define MPK_V2_PROF_IN_WINDOW(it) (false)
#define MPK_V2_PROF_SNAPSHOT()
#define MPK_V2_TIMED_WAIT_IF(cond, group, ev, expr) expr
#define MPK_V2_TIMED_WAIT(group, ev, expr) expr
#endif

namespace mirage {
namespace runtime_v2 {

using namespace mirage::runtime;

// Clean v2 role runtime.
//
// Warp layout:
//   W0-W3: compute
//   W4:    loader
//   W5:    mma
//   W6:    storer
//   W7:    dispatcher
//
// The runtime owns instruction-slot scheduling, graph dependency waiting,
// event triggering, and generic SMEM page semaphores. Task-specific behavior
// must live behind the generated role dispatcher.
static constexpr int NUM_COMPUTE_WARPS = 4;
static constexpr int LOADER_WARP = 4;
static constexpr int MMA_WARP = 5;
static constexpr int STORER_WARP = 6;
static constexpr int DISPATCHER_WARP = 7;
static constexpr int NUM_ROLE_WARPS = 7;
static constexpr int NUM_WARPS = 8;
static constexpr int NUM_THREADS = NUM_WARPS * 32;

static constexpr int INSTRUCTION_RING_SIZE = 3;
// Match Megakernel's page protocol: page availability is tracked by parity
// semaphores keyed by the instruction index, not by runtime page ownership.
static constexpr int PAGE_SEMAPHORE_BITS = 1;

static constexpr int MBAR_INSTRUCTION_ARRIVED = 0;
static constexpr int MBAR_INSTRUCTION_FINISHED = 1;
static constexpr int NUM_INSTRUCTION_MBARS = 2;

// Per-instruction dynamic semaphores. Each task type may declare an
// op-specific init body that is run by the dispatcher (single thread) once
// per published instruction; the body is free to mbar_init any of these
// slots and any role body can mbar_arrive/mbar_wait on them. Drained
// implicitly when the dispatcher waits on instruction_finished[slot] before
// recycling the slot.
static constexpr int MAX_DYNAMIC_SEMAPHORES = 32;

// Slot conventions for dynamic_semaphores[slot][i]:
//   SEM_DEP_READY — compute warp 0 lane 0 spins on the cross-SM event
//   counter, then arrives this semaphore. Other compute warps wait on it
//   so they enter the compute body in lockstep with the dep being cleared.
//   SEM_OP_BASE..MAX_DYNAMIC_SEMAPHORES-1 — op-private slots. Any task
//   type that needs intra-task cross-warp coordination (e.g. linear's
//   per-stage TMA→MMA→epilogue handshakes) uses these.
static constexpr int SEM_DEP_READY = 0;
static constexpr int SEM_OP_BASE   = 1;

__device__ __forceinline__ int smem_addr(void const *ptr) {
  return static_cast<int>(__cvta_generic_to_shared(ptr));
}

// Fast, near-non-suspending poll (minimal suspend-time hint). Used by the
// dispatcher's eager trigger sweep, which polls many slots per loop and must
// not suspend ~10M cycles on each not-ready slot.
__device__ __forceinline__ bool mbar_poll(int addr, int phase) {
  int ok;
  asm volatile(
      "{\n\t.reg .pred P;\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2, 1;\n\t"
      "selp.b32 %0, 1, 0, P;\n\t}"
      : "=r"(ok) : "r"(addr), "r"(phase));
  return ok != 0;
}

__device__ __forceinline__ void mbar_init(uint64_t *mbar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
               :: "r"(smem_addr(mbar)), "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint64_t *mbar) {
  // plain mbarrier.arrive defaults to .release.cta, so role warps' .acquire
  // waits already synchronize-with this arrive.
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
               :: "r"(smem_addr(mbar)) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t *mbar, int phase) {
  int addr = smem_addr(mbar);
  asm volatile(
      "{\n\t.reg .pred P;\n\t"
      "WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1, 0x989680;\n\t"
      "@P bra DONE;\n\t"
      "bra WAIT;\n\t"
      "DONE:\n\t}"
      :: "r"(addr), "r"(phase));
}

__device__ __forceinline__ size_t get_task_iteration_num(TaskId task_id) {
  return task_id >> 32;
}

__device__ __forceinline__ size_t get_event_position_index(EventId event_id) {
  return event_id & 0xffffffff;
}

__device__ __forceinline__ bool is_nvshmem_event(EventId event_id) {
  return (event_id & EVENT_NVSHMEM_TAG) != 0;
}

struct RuntimeSMEM {
  uint64_t instruction_mbarriers[NUM_INSTRUCTION_MBARS]
                                [INSTRUCTION_RING_SIZE];
  uint64_t page_finished[MAX_SMEM_PAGES_PER_TASK][PAGE_SEMAPHORE_BITS];
  uint64_t dynamic_semaphores[INSTRUCTION_RING_SIZE][MAX_DYNAMIC_SEMAPHORES];
  __align__(16) char task_buf[INSTRUCTION_RING_SIZE][sizeof(TaskDesc)];

  __device__ __forceinline__ TaskDesc *task_slot(int slot) {
    return reinterpret_cast<TaskDesc *>(task_buf[slot]);
  }
};

__device__ __forceinline__ int ring_slot(int sequence) {
  return sequence % INSTRUCTION_RING_SIZE;
}

__device__ __forceinline__ int ring_phase(int sequence) {
  return (sequence / INSTRUCTION_RING_SIZE) & 1;
}

// SMEM address of the first op-private dynamic semaphore for the slot
// owned by `instruction_index`. Tasks that need intra-task cross-warp
// mbarriers (e.g. linear's per-stage TMA→MMA→epilogue handshakes)
// receive this base from codegen and address mbars as base + i*8.
__device__ __forceinline__ int op_sem_base_addr(RuntimeSMEM *rt,
                                                int instruction_index) {
  return smem_addr(
      &rt->dynamic_semaphores[ring_slot(instruction_index)][SEM_OP_BASE]);
}

__device__ __forceinline__ void init_page_state(RuntimeSMEM *rt) {
  if (threadIdx.x == 0) {
    for (int page = 0; page < MAX_SMEM_PAGES_PER_TASK; page++) {
      for (int bit = 0; bit < PAGE_SEMAPHORE_BITS; bit++) {
        mbar_init(&rt->page_finished[page][bit], 1);
        mbar_arrive(&rt->page_finished[page][bit]);
      }
    }
  }
}

__device__ __forceinline__ void runtime_wait_page_ready(
    RuntimeSMEM *rt, int physical_page, int instruction_index) {
  #pragma unroll
  for (int bit = 0; bit < PAGE_SEMAPHORE_BITS; bit++) {
    int const phase = (instruction_index >> bit) & 1;
    mbar_wait(&rt->page_finished[physical_page][bit], phase);
  }
}

__device__ __forceinline__ void runtime_finish_page(
    RuntimeSMEM *rt, int physical_page, int arrive_count = 1) {
  #pragma unroll
  for (int bit = 0; bit < PAGE_SEMAPHORE_BITS; bit++) {
    for (int i = 0; i < arrive_count; i++) {
      mbar_arrive(&rt->page_finished[physical_page][bit]);
    }
  }
}

// Returns true if `physical_page` falls inside any of the task's
// declared SMEM regions. Used by the codegen-emitted loader prefix to decide
// which pages to "claim+release ASAP" vs which to leave for the compute/last
// user to release after they're done with the data.
__device__ __forceinline__ bool task_uses_page(TaskDesc const *task_desc,
                                               int physical_page) {
  for (int r = 0; r < task_desc->num_smem_regions; r++) {
    SmemPageRegionDesc const &region = task_desc->smem_regions[r];
    int const start = region.physical_page_start;
    int const end = start + region.page_count;
    if (physical_page >= start && physical_page < end) {
      return true;
    }
  }
  return false;
}

__device__ __forceinline__ int runtime_region_physical_page(
    TaskDesc const *task_desc, int region_idx, int page_offset) {
  SmemPageRegionDesc const &region = task_desc->smem_regions[region_idx];
  int const physical_page = region.physical_page_start + page_offset;
  if (physical_page < 0 || physical_page >= MAX_SMEM_PAGES_PER_TASK) {
    return -1;
  }
  return physical_page;
}

__device__ __forceinline__ void wait_task_dependency(
    RuntimeConfig const &config, TaskDesc const *task, int iter_num) {
  EventId dep = task->dependent_event;
  if (dep == EVENT_INVALID_ID || is_nvshmem_event(dep)) {
    return;
  }

  size_t const event_index = get_event_position_index(dep);
  EventCounter const needed =
      static_cast<EventCounter>(config.all_event_num_triggers[event_index]) *
      static_cast<EventCounter>(iter_num + 1);
  while (ld_acquire_sys_u64(&config.all_event_counters[event_index]) < needed) {
    __nanosleep(10);
  }
}

// Same dep-wait as wait_task_dependency, but __noinline__ so when called
// from a compute task body it doesn't inflate the caller's register count.
// Safe to call from any thread; each thread independently confirms the dep.
__device__ __noinline__ void wait_task_dependency_noinline(
    RuntimeConfig const &config, TaskDesc const *task, int iter_num) {
  wait_task_dependency(config, task, iter_num);
}

// Compute-side dep-wait prefix, run by every task's compute body
// (and by linear's loader/mma bodies too, since they share the body
// string). Single-thread spin + per-slot SEM_DEP_READY mbarrier sync.
//   - thread 0 globally spins on the cross-SM event counter, then arrives
//     dynamic_semaphores[slot][SEM_DEP_READY].
//   - Lane 0 of every warp running this body waits on the same semaphore.
//   - __syncwarp() reconverges each warp's 32 lanes after its lane-0 wait.
//
// Wrapped __noinline__ so the multi-line body and its locals don't inflate
// compute_warp_loop's register frame past the launch_bounds(256) ceiling
// (same constraint that forced wait_task_dependency_noinline above).
//
// Phase: ring_phase(instruction_index) — same parity scheme as
// instruction_arrived. SEM_DEP_READY is init-once at kernel start, then
// arrived exactly once per slot use (either by the compute prefix here,
// or by the dispatcher for tasks that skip the compute body — see
// BEGIN_TASK_GRAPH special case in dispatcher_warp_loop).
__device__ __noinline__ void compute_dep_prefix(
    RuntimeConfig const &config,
    TaskDesc const *task_desc,
    RuntimeSMEM *rt,
    int instruction_index,
    int iter_num) {
  int const slot = ring_slot(instruction_index);
  int const phase = ring_phase(instruction_index);
  if (threadIdx.x == 0) {
#ifdef MPK_ENABLE_PROFILING
    bool const _in_win =
        config.profiler_buffer != nullptr && MPK_V2_PROF_IN_WINDOW(iter_num);
    unsigned long long const _t0 = v2_prof_now_ns();
    if (_in_win) {
      v2_prof_emit(config.profiler_buffer, V2_PROF_GROUP_COMPUTE_STALL,
                   V2_PROF_DEP_WAIT, tb::EVENT_BEGIN);
    }
#endif
    wait_task_dependency(config, task_desc, iter_num);
#ifdef MPK_ENABLE_PROFILING
    if (_in_win) {
      v2_prof_emit(config.profiler_buffer, V2_PROF_GROUP_COMPUTE_STALL,
                   V2_PROF_DEP_WAIT, tb::EVENT_END);
      // aggregate accumulators (ns, bucketed by task type) — kept alongside
      // the trace events for table-style analysis without trace parsing.
      unsigned long long *_spin =
          static_cast<unsigned long long *>(config.profiler_buffer);
      size_t const _b =
          V2_PROF_SPIN_BASE + 256ull * v2_prof_bucket(task_desc->task_type);
      _spin[_b + blockIdx.x] += v2_prof_now_ns() - _t0;
      _spin[_b + 128 + blockIdx.x] += 1;
    }
#endif
    mbar_arrive(&rt->dynamic_semaphores[slot][SEM_DEP_READY]);
  }
  // All lanes wait the dependency semaphore directly. Do NOT gate this on
  // lane 0 with a trailing __syncwarp(): on sm_100a that construct compiles
  // to a WARPSYNC.COLLECTIVE region around the try-wait whose wake crawls
  // ~5us per templated token at the quiet iteration head, delaying the
  // FINISHED arrival of warps 1-3 and the task's event with it
  // (~300us/step at bs16 after cascade; V2_TODO.md #17 has the full
  // evidence chain).
  mbar_wait(&rt->dynamic_semaphores[slot][SEM_DEP_READY], phase);
}

__device__ __forceinline__ void trigger_task_event(
    RuntimeConfig const &config, TaskDesc const *task) {
  EventId event_id = task->trigger_event;
  if (event_id == EVENT_INVALID_ID || is_nvshmem_event(event_id)) {
    return;
  }

  size_t const event_index = get_event_position_index(event_id);
  atom_add_release_gpu_u64(&config.all_event_counters[event_index], 1);
#ifdef MPK_ENABLE_PROFILING
  if (g_v2_prof_window != 0 && g_v2_prof_buf != nullptr) {
    uint64_t *buf = static_cast<uint64_t *>(g_v2_prof_buf);
    unsigned long long const k = atomicAdd(
        reinterpret_cast<unsigned long long *>(&buf[V2_PROF_TRIG_CURSOR]),
        1ULL);
    if (k < V2_PROF_TRIG_RING_LEN) {
      buf[V2_PROF_TRIG_BASE + k] =
          (v2_prof_now_ns() << 32) |
          ((unsigned long long)(event_index & 0xFFFFFF) << 8) |
          (blockIdx.x & 0xFF);
    }
  }
#endif
}

// Implemented by the generated v2 role dispatch code after this runtime header
// is included.
__device__ __forceinline__ void _execute_init_semaphores_v2(
    TaskDesc const *task_desc,
    RuntimeConfig const &config,
    RuntimeSMEM *runtime_smem,
    int instruction_index,
    int iter_num);

__device__ __forceinline__ void _execute_loader_task_v2(
    TaskDesc const *task_desc,
    RuntimeConfig const &config,
    RuntimeSMEM *runtime_smem,
    int instruction_index,
    int iter_num);

__device__ __forceinline__ void _execute_mma_task_v2(
    TaskDesc const *task_desc,
    RuntimeConfig const &config,
    RuntimeSMEM *runtime_smem,
    int instruction_index,
    int iter_num);

__device__ __forceinline__ void _execute_compute_task_v2(
    TaskDesc const *task_desc,
    RuntimeConfig const &config,
    RuntimeSMEM *runtime_smem,
    int instruction_index,
    int iter_num);

__device__ __forceinline__ void _execute_storer_task_v2(
    TaskDesc const *task_desc,
    RuntimeConfig const &config,
    RuntimeSMEM *runtime_smem,
    int instruction_index,
    int iter_num);

#define MIRAGE_V2_DEFINE_ROLE_WARP_LOOP(loop_name, execute_task,              \
                                        prof_group, prof_pred)               \
  __device__ __noinline__ void loop_name(                                    \
      RuntimeSMEM *rt, RuntimeConfig const &config, int lane_id) {           \
    int const worker_id = blockIdx.x;                                        \
    int const my_count = static_cast<int>(                                   \
        config.v2_per_sm_task_offsets[worker_id + 1] -                       \
        config.v2_per_sm_task_offsets[worker_id]);                           \
    int sequence = 0;                                                        \
    int iter_num = 0;                                                        \
    int sequence_in_iter = 0;                                                \
    /* profiling: one track per role. The compute loop runs on 4 warps,  */ \
    /* so its predicate must select warp 0 lane 0 (threadIdx.x == 0);     */ \
    /* single-warp roles use lane_id == 0.                                */ \
    MPK_V2_PROF_DECL(prof_group, prof_pred)                                  \
    while (true) {                                                           \
      int const slot = ring_slot(sequence);                                  \
      int const phase = ring_phase(sequence);                                \
      if (lane_id == 0) {                                                    \
        mbar_wait(&rt->instruction_mbarriers[MBAR_INSTRUCTION_ARRIVED][slot],\
                  phase);                                                    \
      }                                                                      \
      __syncwarp();                                                          \
      TaskDesc *task = rt->task_slot(slot);                                  \
      if (task->task_type == TASK_TERMINATE) {                               \
        return;                                                              \
      }                                                                      \
      if (task->task_type != TASK_BEGIN_TASK_GRAPH) {                        \
        MPK_V2_PROF_START(task->task_type);                                  \
        execute_task(task, config, rt, sequence, iter_num);                  \
        MPK_V2_PROF_END(task->task_type);                                    \
      }                                                                      \
      if (lane_id == 0) {                                                    \
        mbar_arrive(                                                         \
            &rt->instruction_mbarriers[MBAR_INSTRUCTION_FINISHED][slot]);    \
      }                                                                      \
      sequence++;                                                            \
      sequence_in_iter++;                                                    \
      if (sequence_in_iter == my_count) {                                    \
        sequence_in_iter = 0;                                                \
        iter_num++;                                                          \
      }                                                                      \
    }                                                                        \
  }

MIRAGE_V2_DEFINE_ROLE_WARP_LOOP(loader_warp_loop,
                                _execute_loader_task_v2,
                                V2_PROF_GROUP_LOADER,
                                (lane_id == 0))
MIRAGE_V2_DEFINE_ROLE_WARP_LOOP(mma_warp_loop,
                                _execute_mma_task_v2,
                                V2_PROF_GROUP_MMA,
                                (lane_id == 0))
MIRAGE_V2_DEFINE_ROLE_WARP_LOOP(compute_warp_loop,
                                _execute_compute_task_v2,
                                V2_PROF_GROUP_COMPUTE,
                                (threadIdx.x == 0))
MIRAGE_V2_DEFINE_ROLE_WARP_LOOP(storer_warp_loop,
                                _execute_storer_task_v2,
                                V2_PROF_GROUP_STORER,
                                (lane_id == 0))

#undef MIRAGE_V2_DEFINE_ROLE_WARP_LOOP

} // namespace runtime_v2
} // namespace mirage

#if defined(MODE_OFFLINE)
__device__ __forceinline__ bool
prepare_next_batch(mirage::runtime::RuntimeConfig const &config);
#elif defined(MODE_ONLINE_NOTOKEN)
__device__ __forceinline__ bool
prepare_next_batch(mirage::runtime::RuntimeConfig const &config,
                   size_t iteration_num);
#endif

namespace mirage {
namespace runtime_v2 {

// Set by worker 0 each iteration to prepare_next_batch's "generation done"
// signal (EOS reached or step >= max_seq_length). v1 stops on this return;
// v2 previously ignored it and ran the full max_seq_length iterations, so its
// per-token latency scaled with max_seq_length instead of actual output length.
__device__ unsigned int g_v2_gen_done = 0;

__device__ __noinline__ void dispatcher_warp_loop(
    RuntimeSMEM *rt, RuntimeConfig const &config, int lane_id) {
  int const worker_id = blockIdx.x;
  int const num_workers = config.num_workers;
  size_t const my_offset = config.v2_per_sm_task_offsets[worker_id];
  size_t const my_end = config.v2_per_sm_task_offsets[worker_id + 1];
  size_t const my_count = my_end - my_offset;
  int sequence = 0;

  // profiling track (dispatcher group): prepare/iter-barrier timing.
  MPK_V2_PROF_DECL(V2_PROF_GROUP_DISPATCHER, (lane_id == 0))

  // Per-slot dedup: the last absolute sequence whose graph event we already
  // triggered for this ring slot. This lets the dispatcher trigger events
  // OUT OF ORDER — eagerly, as soon as a task's role warps finish — without
  // ever double-counting. Out-of-order triggering is REQUIRED to avoid a
  // deferred-trigger deadlock: an earlier compute task (next in this SM's
  // ring) can block on a graph event whose producer is a LATER, already
  // finished task on the same ring. The old in-order
  // wait_finished_and_trigger_through could never reach that later producer,
  // so its event never fired and the whole pipeline froze.
  int triggered_seq[INSTRUCTION_RING_SIZE];
  #pragma unroll
  for (int s = 0; s < INSTRUCTION_RING_SIZE; s++) {
    triggered_seq[s] = -1;
  }

  // Lane 0 only. Non-blocking: trigger every in-flight (published, not yet
  // slot-reused) task whose role warps have finished and which we have not
  // triggered yet. task_slot(slot) still holds sequence s's TaskDesc until it
  // is reused at s + INSTRUCTION_RING_SIZE (> sequence), so this is safe.
  auto eager_trigger_inflight = [&]() {
    int lo = sequence - INSTRUCTION_RING_SIZE;
    if (lo < 0) lo = 0;
    for (int s = lo; s < sequence; s++) {
      int const slot = ring_slot(s);
      if (triggered_seq[slot] == s) {
        continue;
      }
      if (mbar_poll(
              smem_addr(
                  &rt->instruction_mbarriers[MBAR_INSTRUCTION_FINISHED][slot]),
              ring_phase(s))) {
        trigger_task_event(config, rt->task_slot(slot));
        triggered_seq[slot] = s;
      }
    }
  };

  // Lane 0 spins until done_sequence's slot has finished (so its slot can be
  // reused), eagerly triggering any other finished in-flight tasks while it
  // waits — this is what breaks the cycle.
  auto wait_slot_finished_eager = [&](int done_sequence) {
    if (lane_id == 0) {
      int const done_slot = ring_slot(done_sequence);
      int const done_phase = ring_phase(done_sequence);
      while (!mbar_poll(
                 smem_addr(
                     &rt->instruction_mbarriers[MBAR_INSTRUCTION_FINISHED]
                                               [done_slot]),
                 done_phase)) {
        eager_trigger_inflight();
      }
      eager_trigger_inflight();
    }
    __syncwarp();
  };

  // The cross-SM dependency wait lives in each task's compute prefix, which
  // calls wait_task_dependency_noinline before running its body, so the
  // dispatcher stays pure fetch+publish; only
  // wait_slot_finished_eager (above) is kept, both for slot reuse
  // and for promptly publishing intra-stream producer events that the
  // compute's spin needs to terminate.
  for (int iter_num = 0; iter_num < config.v2_max_iters; iter_num++) {
    // prepare_next_batch mutates per-iteration decode state used by every
    // worker. Worker 0 does the mutation once; all other workers wait for the
    // system-scope counter before issuing this iteration's task stream.
    if (worker_id == 0) {
      if (lane_id == 0) {
#ifdef MPK_ENABLE_PROFILING
        // arm/disarm the ambient timed-wait window for ALL SMs; they read it
        // only after their go-counter acquire below, so it is coherent.
        g_v2_prof_window = MPK_V2_PROF_IN_WINDOW(iter_num) ? 1 : 0;
#endif
        bool _cont = true;
        MPK_V2_PROF_START(V2_PROF_PREPARE_BATCH);
#if defined(MODE_OFFLINE)
        _cont = ::prepare_next_batch(config);
#elif defined(MODE_ONLINE_NOTOKEN)
        _cont = ::prepare_next_batch(config, iter_num);
#endif
        MPK_V2_PROF_END(V2_PROF_PREPARE_BATCH);
#ifdef MPK_ENABLE_PROFILING
        _cont = true;  // profiling: run all iters, don't early-exit
#endif
        // Mirror v1: prepare_next_batch returns false when generation is done
        // (EOS or step >= max_seq_length). Publish it BEFORE the go-counter
        // increment so other workers see it after their acquire-load below.
        g_v2_gen_done = _cont ? 0u : 1u;
        __threadfence_system();
        atomicAdd_system(config.v2_iter_go_counter, 1ULL);
      }
    } else {
      if (lane_id == 0) {
        MPK_V2_PROF_START(V2_PROF_GO_WAIT);
        unsigned long long const needed =
            static_cast<unsigned long long>(iter_num + 1);
        while (ld_acquire_sys_u64(config.v2_iter_go_counter) < needed) {
          __nanosleep(50);
        }
        MPK_V2_PROF_END(V2_PROF_GO_WAIT);
      }
    }
    __syncwarp();

    // THE FIX: stop as soon as generation is actually finished, instead of
    // running all v2_max_iters (= max_seq_length) iterations. worker 0 set
    // g_v2_gen_done above (ordered before its go-counter increment); every
    // other worker has passed the acquire-load of that counter, so this read
    // observes it. Mirrors v1, which terminates on prepare_next_batch's return.
    {
      unsigned int _done = 0;
      if (lane_id == 0)
        _done = *reinterpret_cast<volatile unsigned int *>(&g_v2_gen_done);
      _done = __shfl_sync(0xffffffff, _done, 0);
      if (_done) break;
    }

    for (size_t i = 0; i < my_count; i++) {
      int const slot = ring_slot(sequence);
      int const phase = ring_phase(sequence);

      // The instruction ring is finite. Before writing a new TaskDesc into a
      // reused slot, wait until the role warps have finished the previous
      // sequence that occupied the same slot.
      if (sequence >= INSTRUCTION_RING_SIZE) {
        wait_slot_finished_eager(sequence - INSTRUCTION_RING_SIZE);
      }

      size_t const task_pos =
          config.v2_per_sm_task_positions[my_offset + i];
      {
        // The dispatcher warp cooperatively copies one TaskDesc from the
        // compiled global task table into the shared-memory ring slot. The
        // following __syncwarp makes lane 0 wait for every lane's copy chunks
        // before it publishes instruction_arrived to the role warps.
        char *dst = rt->task_buf[slot];
        char const *src =
            reinterpret_cast<char const *>(&config.all_tasks[task_pos]);
        constexpr int CHUNKS = (sizeof(TaskDesc) + 15) / 16;
        for (int c = lane_id; c < CHUNKS; c += 32) {
          ::kernel::load_smem(dst + c * 16, src + c * 16);
        }
        ::kernel::cp_async_fence();
        ::kernel::cp_async_wait<0>();
        // The TaskDesc was copied via cp.async; role warps read it with normal
        // loads after the INSTRUCTION_ARRIVED mbar. Publish those async writes to
        // the generic proxy before the arrive. v1 got this implicitly from the
        // __syncthreads after cp_async_wait; v2's warp-specialized handshake
        // doesn't, and the PTX memory model requires the fence here.
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      }
      __syncwarp();

      // Op-declared per-instruction semaphore initialization. The body is
      // emitted by codegen and runs single-threaded (lane 0) once per
      // published instruction, before role warps wake. Empty for ops that
      // don't declare any dynamic semaphores.
      //
      // ALSO: BEGIN_TASK_GRAPH skips the role-warp compute body entirely
      // (the role-warp-loop macro early-returns from execute_task), so its
      // slot's SEM_DEP_READY would never be arrived. To keep the
      // ring_phase parity in sync for the next task at this slot,
      // dispatcher arrives SEM_DEP_READY here on its behalf. The per-page
      // parity needs the same protection — every task in the
      // pipeline must arrive each page exactly once, otherwise consecutive
      // tasks deadlock on page_finished. Dispatcher arrives all pages on
      // BEGIN_TASK_GRAPH's behalf.
      if (lane_id == 0) {
        _execute_init_semaphores_v2(rt->task_slot(slot), config, rt,
                                    sequence, iter_num);
        if (rt->task_slot(slot)->task_type == TASK_BEGIN_TASK_GRAPH) {
          mbar_arrive(&rt->dynamic_semaphores[slot][SEM_DEP_READY]);
          for (int p = 0; p < MAX_SMEM_PAGES_PER_TASK; p++) {
            runtime_finish_page(rt, p, 1);
          }
        }
      }
      __syncwarp();

      // No more dispatcher-side dep wait — computes handle it themselves
      // via wait_task_dependency_noinline in their compute prefix. The
      // dispatcher is now pure fetch+publish.
      //
      // Intra-stream producer events (sequence S-1, S-2 in this same ring
      // stream feeding this same SM's compute) get triggered at slot reuse
      // via wait_slot_finished_eager(sequence - INSTRUCTION_RING_SIZE)
      // above. That introduces up to RING-1 instructions of latency for
      // intra-stream compute-producer chains; in practice these are rare
      // because the worker queues round-robin tasks across SMs. If this turns
      // out hot, event triggering could move into the storer warp to cut the
      // latency.
      if (lane_id == 0) {
        // Do not release role warps until the copied TaskDesc is visible in
        // shared memory. The block fence must happen before mbar_arrive
        // because role warps wake on that mbarrier and immediately read
        // rt->task_buf[slot].
        __threadfence_block();
        mbar_arrive(
            &rt->instruction_mbarriers[MBAR_INSTRUCTION_ARRIVED][slot]);
      }
      __syncwarp();
      sequence++;
    }

    // Drain all live ring slots for this worker before ending the decode
    // iteration: block until each of the last RING tasks has finished (pumping
    // eager triggers so cross-dependencies among them resolve), then a final
    // eager sweep guarantees every published task's event has fired. This
    // preserves the iteration boundary expected by prepare_next_batch and the
    // global step/token state.
    if (lane_id == 0) {
      int lo = sequence - INSTRUCTION_RING_SIZE;
      if (lo < 0) lo = 0;
      for (int s = lo; s < sequence; s++) {
        int const slot = ring_slot(s);
        int const ph = ring_phase(s);
        while (!mbar_poll(
                   smem_addr(
                       &rt->instruction_mbarriers[MBAR_INSTRUCTION_FINISHED]
                                                 [slot]),
                   ph)) {
          eager_trigger_inflight();
        }
      }
      eager_trigger_inflight();
    }
    __syncwarp();

    // All workers must finish the current iteration before any worker starts the
    // next prepare_next_batch. The system fence orders this worker's event
    // updates before it increments the cross-worker iteration counter.
    __threadfence_system();
    if (lane_id == 0) {
      MPK_V2_PROF_START(V2_PROF_ITER_SYNC);
      atomicAdd_system(config.v2_iter_sync_counter, 1ULL);
      unsigned long long const needed =
          static_cast<unsigned long long>(num_workers) *
          static_cast<unsigned long long>(iter_num + 1);
      while (ld_acquire_sys_u64(config.v2_iter_sync_counter) < needed) {
        __nanosleep(50);
      }
      MPK_V2_PROF_END(V2_PROF_ITER_SYNC);
    }
    __syncwarp();

    // Step is updated by prepare_next_batch. Broadcast lane 0's read so the
    // dispatcher warp exits the loop uniformly.
    int step0 = 0;
    if (lane_id == 0) {
      step0 = config.step[0];
    }
    step0 = __shfl_sync(0xffffffff, step0, 0);
    if (step0 >= config.max_seq_length - 1) {
      break;
    }
  }

  // Publish a terminate instruction through the same instruction_arrived path so
  // all role warps leave their role_warp_loop cleanly.
  int const term_slot = ring_slot(sequence);
  if (sequence >= INSTRUCTION_RING_SIZE) {
    wait_slot_finished_eager(sequence - INSTRUCTION_RING_SIZE);
  }
  if (lane_id == 0) {
    rt->task_slot(term_slot)->task_type = TASK_TERMINATE;
    __threadfence_block();
    mbar_arrive(
        &rt->instruction_mbarriers[MBAR_INSTRUCTION_ARRIVED][term_slot]);
  }
}

__global__ __launch_bounds__(NUM_THREADS, 1)
void worker_v2_kernel(RuntimeConfig config) {
  __shared__ __align__(16) char rt_buf[sizeof(RuntimeSMEM)];
  RuntimeSMEM *rt = reinterpret_cast<RuntimeSMEM *>(rt_buf);

  int const warp_id = threadIdx.x / 32;
  int const lane_id = threadIdx.x % 32;

#ifdef MPK_ENABLE_PROFILING
  if (threadIdx.x == 0) {
    // same value from every block — benign race, long before first use.
    g_v2_prof_buf = config.profiler_buffer;
  }
#endif

  if (threadIdx.x == 0) {
    for (int slot = 0; slot < INSTRUCTION_RING_SIZE; slot++) {
      mbar_init(&rt->instruction_mbarriers[MBAR_INSTRUCTION_ARRIVED][slot], 1);
      mbar_init(&rt->instruction_mbarriers[MBAR_INSTRUCTION_FINISHED][slot],
                NUM_ROLE_WARPS);
      // SEM_DEP_READY: per-slot semaphore signaled by compute thread 0
      // and waited on by lane 0 of every warp running the compute body.
      // Init-once + ring_phase parity (matches instruction_arrived). Each
      // slot must be arrived once per use to keep parity in sync — see the
      // BEGIN_TASK_GRAPH special case in dispatcher_warp_loop.
      mbar_init(&rt->dynamic_semaphores[slot][SEM_DEP_READY], 1);
    }
  }
  init_page_state(rt);
  if (threadIdx.x == 0) {
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

  if (warp_id < NUM_COMPUTE_WARPS) {
    // WG0 (warps 0-3): the compute warps run the task bodies.
    compute_warp_loop(rt, config, lane_id);
  } else {
    // WG1 (warps 4-7): loader/mma/storer/dispatcher.
    if (warp_id == LOADER_WARP) {
      loader_warp_loop(rt, config, lane_id);
    } else if (warp_id == MMA_WARP) {
      mma_warp_loop(rt, config, lane_id);
    } else if (warp_id == STORER_WARP) {
      storer_warp_loop(rt, config, lane_id);
    } else if (warp_id == DISPATCHER_WARP) {
      dispatcher_warp_loop(rt, config, lane_id);
    }
  }
}

inline void launch_worker_v2(RuntimeConfig const &config,
                             int num_workers,
                             cudaStream_t stream) {
  int smem = MAX_DYNAMIC_SHARED_MEMORY_SIZE;
  cudaFuncSetAttribute(worker_v2_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem);
  worker_v2_kernel<<<dim3(num_workers, 1, 1),
                     dim3(NUM_THREADS, 1, 1),
                     smem,
                     stream>>>(config);
}

} // namespace runtime_v2
} // namespace mirage
