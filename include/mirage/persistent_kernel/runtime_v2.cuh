#pragma once

#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/persistent_kernel/tasks/blackwell_v2/task_interface.cuh"
#include "mirage/persistent_kernel/tasks/common/copy_sm80.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

namespace mirage {
namespace runtime_v2 {

using namespace mirage::runtime;

// Clean v2 role runtime.
//
// Warp layout:
//   W0-W3: consumer
//   W4:    loader
//   W5:    launcher
//   W6:    storer
//   W7:    controller
//
// The runtime owns instruction-slot scheduling, graph dependency waiting,
// event triggering, and generic SMEM page semaphores. Task-specific behavior
// must live behind the generated role dispatcher.
static constexpr int NUM_CONSUMER_WARPS = 4;
static constexpr int LOADER_WARP = 4;
static constexpr int LAUNCHER_WARP = 5;
static constexpr int STORER_WARP = 6;
static constexpr int CONTROLLER_WARP = 7;
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
// op-specific init body that is run by the controller (single thread) once
// per published instruction; the body is free to mbar_init any of these
// slots and any role body can mbar_arrive/mbar_wait on them. Drained
// implicitly when the controller waits on instruction_finished[slot] before
// recycling the slot.
static constexpr int MAX_DYNAMIC_SEMAPHORES = 32;

// Slot conventions for dynamic_semaphores[slot][i]:
//   SEM_DEP_READY — consumer warp 0 lane 0 spins on the cross-SM event
//   counter, then arrives this semaphore. Other consumer warps wait on it
//   so they enter the compute body in lockstep with the dep being cleared.
//   SEM_OP_BASE..MAX_DYNAMIC_SEMAPHORES-1 — op-private slots. Any task
//   type that needs intra-task cross-warp coordination (e.g. linear's
//   per-stage TMA→MMA→epilogue handshakes after Phase 3) uses these.
static constexpr int SEM_DEP_READY = 0;
static constexpr int SEM_OP_BASE   = 1;

__device__ __forceinline__ int smem_addr(void const *ptr) {
  return static_cast<int>(__cvta_generic_to_shared(ptr));
}

// Fast, near-non-suspending poll (minimal suspend-time hint). Used by the
// controller's eager trigger sweep, which polls many slots per loop and must
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

// Phase 3.5: returns true if `physical_page` falls inside any of the task's
// declared SMEM regions. Used by the codegen-emitted loader prefix to decide
// which pages to "claim+release ASAP" vs which to leave for the consumer/last
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

__device__ __forceinline__ void runtime_wait_region_pages(
    RuntimeSMEM *rt,
    TaskDesc const *task_desc,
    int region_idx,
    int instruction_index) {
  SmemPageRegionDesc const &region = task_desc->smem_regions[region_idx];
  for (int p = 0; p < region.page_count; p++) {
    int const physical_page =
        runtime_region_physical_page(task_desc, region_idx, p);
    if (physical_page >= 0) {
      runtime_wait_page_ready(rt, physical_page, instruction_index);
    }
  }
}

__device__ __forceinline__ void runtime_wait_region_range_pages(
    RuntimeSMEM *rt,
    TaskDesc const *task_desc,
    int first_region,
    int num_regions,
    int instruction_index) {
  for (int r = first_region; r < first_region + num_regions; r++) {
    runtime_wait_region_pages(rt, task_desc, r, instruction_index);
  }
}

__device__ __forceinline__ void runtime_finish_region_pages(
    RuntimeSMEM *rt, TaskDesc const *task_desc, int region_idx) {
  SmemPageRegionDesc const &region = task_desc->smem_regions[region_idx];
  for (int p = 0; p < region.page_count; p++) {
    int const physical_page =
        runtime_region_physical_page(task_desc, region_idx, p);
    if (physical_page >= 0) {
      runtime_finish_page(rt, physical_page, 1);
    }
  }
}

__device__ __forceinline__ void runtime_finish_region_range_pages(
    RuntimeSMEM *rt,
    TaskDesc const *task_desc,
    int first_region,
    int num_regions) {
  for (int r = first_region; r < first_region + num_regions; r++) {
    runtime_finish_region_pages(rt, task_desc, r);
  }
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
// from a consumer task body it doesn't inflate the caller's register count.
// Safe to call from any thread; each thread independently confirms the dep.
__device__ __noinline__ void wait_task_dependency_noinline(
    RuntimeConfig const &config, TaskDesc const *task, int iter_num) {
  wait_task_dependency(config, task, iter_num);
}

// Phase 2 consumer-side dep-wait prefix, run by every task's consumer body
// (and by linear's loader/launcher bodies too, since they share the body
// string). Single-thread spin + per-slot SEM_DEP_READY mbarrier sync.
//   - thread 0 globally spins on the cross-SM event counter, then arrives
//     dynamic_semaphores[slot][SEM_DEP_READY].
//   - Lane 0 of every warp running this body waits on the same semaphore.
//   - __syncwarp() reconverges each warp's 32 lanes after its lane-0 wait.
//
// Wrapped __noinline__ so the multi-line body and its locals don't inflate
// consumer_warp_loop's register frame past the launch_bounds(256) ceiling
// (same constraint that forced wait_task_dependency_noinline above).
//
// Phase: ring_phase(instruction_index) — same parity scheme as
// instruction_arrived. SEM_DEP_READY is init-once at kernel start, then
// arrived exactly once per slot use (either by the consumer prefix here,
// or by the controller for tasks that skip the consumer body — see
// BEGIN_TASK_GRAPH special case in controller_warp_loop).
__device__ __noinline__ void consumer_dep_prefix(
    RuntimeConfig const &config,
    TaskDesc const *task_desc,
    RuntimeSMEM *rt,
    int instruction_index,
    int iter_num) {
  int const slot = ring_slot(instruction_index);
  int const phase = ring_phase(instruction_index);
  if (threadIdx.x == 0) {
    wait_task_dependency(config, task_desc, iter_num);
    mbar_arrive(&rt->dynamic_semaphores[slot][SEM_DEP_READY]);
  }
  if ((threadIdx.x % 32) == 0) {
    mbar_wait(&rt->dynamic_semaphores[slot][SEM_DEP_READY], phase);
  }
  __syncwarp();
}

__device__ __forceinline__ bool task_dependency_ready(
    RuntimeConfig const &config, TaskDesc const *task, int iter_num) {
  EventId dep = task->dependent_event;
  if (dep == EVENT_INVALID_ID || is_nvshmem_event(dep)) {
    return true;
  }

  size_t const event_index = get_event_position_index(dep);
  EventCounter const needed =
      static_cast<EventCounter>(config.all_event_num_triggers[event_index]) *
      static_cast<EventCounter>(iter_num + 1);
  return ld_acquire_sys_u64(&config.all_event_counters[event_index]) >= needed;
}

__device__ __forceinline__ void trigger_task_event(
    RuntimeConfig const &config, TaskDesc const *task) {
  EventId event_id = task->trigger_event;
  if (event_id == EVENT_INVALID_ID || is_nvshmem_event(event_id)) {
    return;
  }

  size_t const event_index = get_event_position_index(event_id);
  atom_add_release_gpu_u64(&config.all_event_counters[event_index], 1);
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

__device__ __forceinline__ void _execute_launcher_task_v2(
    TaskDesc const *task_desc,
    RuntimeConfig const &config,
    RuntimeSMEM *runtime_smem,
    int instruction_index,
    int iter_num);

__device__ __forceinline__ void _execute_consumer_task_v2(
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

#define MIRAGE_V2_DEFINE_ROLE_WARP_LOOP(loop_name, execute_task)              \
  __device__ __noinline__ void loop_name(                                    \
      RuntimeSMEM *rt, RuntimeConfig const &config, int lane_id) {           \
    int const worker_id = blockIdx.x;                                        \
    int const my_count = static_cast<int>(                                   \
        config.v2_per_sm_task_offsets[worker_id + 1] -                       \
        config.v2_per_sm_task_offsets[worker_id]);                           \
    int sequence = 0;                                                        \
    int iter_num = 0;                                                        \
    int sequence_in_iter = 0;                                                \
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
        execute_task(task, config, rt, sequence, iter_num);                  \
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

MIRAGE_V2_DEFINE_ROLE_WARP_LOOP(loader_warp_loop, _execute_loader_task_v2)
MIRAGE_V2_DEFINE_ROLE_WARP_LOOP(launcher_warp_loop, _execute_launcher_task_v2)
MIRAGE_V2_DEFINE_ROLE_WARP_LOOP(consumer_warp_loop, _execute_consumer_task_v2)
MIRAGE_V2_DEFINE_ROLE_WARP_LOOP(storer_warp_loop, _execute_storer_task_v2)

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

__device__ __noinline__ void controller_warp_loop(
    RuntimeSMEM *rt, RuntimeConfig const &config, int lane_id) {
  int const worker_id = blockIdx.x;
  int const num_workers = config.num_workers;
  size_t const my_offset = config.v2_per_sm_task_offsets[worker_id];
  size_t const my_end = config.v2_per_sm_task_offsets[worker_id + 1];
  size_t const my_count = my_end - my_offset;
  int sequence = 0;

  // Per-slot dedup: the last absolute sequence whose graph event we already
  // triggered for this ring slot. This lets the controller trigger events
  // OUT OF ORDER — eagerly, as soon as a task's role warps finish — without
  // ever double-counting. Out-of-order triggering is REQUIRED to avoid a
  // deferred-trigger deadlock: an earlier consumer task (next in this SM's
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

  // Phase 2: cross-SM dependency wait moved out of the controller. Each
  // task's consumer prefix now calls wait_task_dependency_noinline before
  // running its body. Controller becomes pure fetch+publish; only
  // wait_finished_and_trigger_through (above) is kept, both for slot reuse
  // and for promptly publishing intra-stream producer events that the
  // consumer's spin needs to terminate.
  for (int iter_num = 0; iter_num < config.v2_max_iters; iter_num++) {
    // prepare_next_batch mutates per-iteration decode state used by every
    // worker. Worker 0 does the mutation once; all other workers wait for the
    // system-scope counter before issuing this iteration's task stream.
    if (worker_id == 0) {
      if (lane_id == 0) {
        bool _cont = true;
#if defined(MODE_OFFLINE)
        _cont = ::prepare_next_batch(config);
#elif defined(MODE_ONLINE_NOTOKEN)
        _cont = ::prepare_next_batch(config, iter_num);
#endif
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
        unsigned long long const needed =
            static_cast<unsigned long long>(iter_num + 1);
        while (ld_acquire_sys_u64(config.v2_iter_go_counter) < needed) {
          __nanosleep(50);
        }
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
        // The controller warp cooperatively copies one TaskDesc from the
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
      }
      __syncwarp();

      // Op-declared per-instruction semaphore initialization. The body is
      // emitted by codegen and runs single-threaded (lane 0) once per
      // published instruction, before role warps wake. Empty for ops that
      // don't declare any dynamic semaphores; Phase 3+ will populate it.
      //
      // ALSO: BEGIN_TASK_GRAPH skips the role-warp consumer body entirely
      // (the role-warp-loop macro early-returns from execute_task), so its
      // slot's SEM_DEP_READY would never be arrived. To keep the
      // ring_phase parity in sync for the next task at this slot,
      // controller arrives SEM_DEP_READY here on its behalf. Phase 3.5:
      // the per-page parity needs the same protection — every task in the
      // pipeline must arrive each page exactly once, otherwise consecutive
      // tasks deadlock on page_finished. Controller arrives all pages on
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

      // No more controller-side dep wait — consumers handle it themselves
      // via wait_task_dependency_noinline in their consumer prefix. The
      // controller is now pure fetch+publish.
      //
      // Intra-stream producer events (sequence S-1, S-2 in this same ring
      // stream feeding this same SM's consumer) get triggered at slot reuse
      // via wait_finished_and_trigger_through(sequence - INSTRUCTION_RING_SIZE)
      // above. That introduces up to RING-1 instructions of latency for
      // intra-stream consumer-producer chains; in Qwen3 these are rare
      // because the worker queues round-robin tasks across SMs. If
      // measurement shows this is hot, Phase 4 can move event triggering
      // into the storer warp and cut this latency.
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
      atomicAdd_system(config.v2_iter_sync_counter, 1ULL);
      unsigned long long const needed =
          static_cast<unsigned long long>(num_workers) *
          static_cast<unsigned long long>(iter_num + 1);
      while (ld_acquire_sys_u64(config.v2_iter_sync_counter) < needed) {
        __nanosleep(50);
      }
    }
    __syncwarp();

    // Step is updated by prepare_next_batch. Broadcast lane 0's read so the
    // controller warp exits the loop uniformly.
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

  if (threadIdx.x == 0) {
    for (int slot = 0; slot < INSTRUCTION_RING_SIZE; slot++) {
      mbar_init(&rt->instruction_mbarriers[MBAR_INSTRUCTION_ARRIVED][slot], 1);
      mbar_init(&rt->instruction_mbarriers[MBAR_INSTRUCTION_FINISHED][slot],
                NUM_ROLE_WARPS);
      // SEM_DEP_READY: per-slot semaphore signaled by consumer thread 0
      // and waited on by lane 0 of every warp running the consumer body.
      // Init-once + ring_phase parity (matches instruction_arrived). Each
      // slot must be arrived once per use to keep parity in sync — see the
      // BEGIN_TASK_GRAPH special case in controller_warp_loop.
      mbar_init(&rt->dynamic_semaphores[slot][SEM_DEP_READY], 1);
    }
  }
  init_page_state(rt);
  if (threadIdx.x == 0) {
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

  if (warp_id < NUM_CONSUMER_WARPS) {
    consumer_warp_loop(rt, config, lane_id);
  } else if (warp_id == LOADER_WARP) {
    loader_warp_loop(rt, config, lane_id);
  } else if (warp_id == LAUNCHER_WARP) {
    launcher_warp_loop(rt, config, lane_id);
  } else if (warp_id == STORER_WARP) {
    storer_warp_loop(rt, config, lane_id);
  } else if (warp_id == CONTROLLER_WARP) {
    controller_warp_loop(rt, config, lane_id);
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
