#pragma once

#include "mirage/persistent_kernel/mpk_atoms.cuh"
#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/persistent_kernel/tasks/common/copy_sm80.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

namespace mirage {
namespace runtime_v2 {

using namespace mirage::runtime;

// ============ Warp layout ============
// 11 warps = 352 threads per SM
// W0-7:   compute (run tasks)
// W8:     controller (fetch tasks, signal compute)
// W9:     prefetch (L2 prefetch for next task's weights; gated off at ≥64 SMs)
// W10:    event (dep-wait, trigger events) — pipelines with controller
//
// (Previous layout had W11-15 as idle nanosleepers. Dropped in favour of a
//  smaller block — freed ~3,840 regs plus 160 thread slots. Idle warps were
//  not doing work, and the regs they held weren't helping compute because
//  compute never called setmaxnreg.inc. Removing them is a no-op for
//  correctness and a minor SM-resource win.)

static constexpr int NUM_WARPS = 11;
static constexpr int NUM_THREADS = NUM_WARPS * 32;
static constexpr int NUM_COMPUTE_WARPS = 8;

static constexpr int CONTROLLER_WARP = 8;
static constexpr int PREFETCH_WARP = 9;
static constexpr int EVENT_WARP = 10;

// ============ Mbarrier indices ============
static constexpr int MBAR_TASK_READY = 0;   // controller → compute+event: "task desc in SMEM"
static constexpr int MBAR_TASK_DONE = 1;    // compute → controller+event: "task finished"
static constexpr int MBAR_DEPS_CLEAR = 2;   // event → compute: "dependencies satisfied"
static constexpr int NUM_MBARRIERS = 3;

// ============ Mbarrier helpers ============

__device__ __forceinline__ int smem_addr(const void* ptr) {
    return static_cast<int>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbar_init(uint64_t* mbar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :: "r"(smem_addr(mbar)), "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint64_t* mbar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                 :: "r"(smem_addr(mbar)) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint64_t* mbar, int phase) {
    int addr = smem_addr(mbar);
    asm volatile(
        "{\n\t.reg .pred P;\n\t"
        "WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%0], %1, 0x989680;\n\t"
        "@P bra DONE;\n\t"
        "bra WAIT;\n\t"
        "DONE:\n\t}"
        :: "r"(addr), "r"(phase));
}

// ============ TaskId / EventId helpers ============
// Duplicated from persistent_kernel.cuh to avoid pulling in the entire v1 runtime.

__device__ __forceinline__ size_t get_task_iteration_num(TaskId task_id) {
    return (task_id >> 32);
}

__device__ __forceinline__ size_t get_task_position_index(TaskId task_id) {
    return (task_id & 0xffffffff);
}

__device__ __forceinline__ size_t get_event_position_index(EventId event_id) {
    return (event_id & 0xffffffff);
}

__device__ __forceinline__ bool is_nvshmem_event(EventId event_id) {
    return (event_id & EVENT_NVSHMEM_TAG) > 0;
}

__device__ __forceinline__ int get_rand_sched_id(size_t event_index,
                                                 int worker_id,
                                                 int num_workers,
                                                 int num_schedulers) {
    size_t x = worker_id;
    return x / ((num_workers + num_schedulers - 1) / num_schedulers);
}

// ============ Runtime shared memory ============
// Static SMEM: runtime state. Tasks use dynamic (extern) SMEM after this.

// Lookahead entry: points to a future task in GMEM for prefetch
struct LookaheadEntry {
    size_t task_position;     // index into config.all_tasks
    volatile int ready;       // 1 = written by controller, 0 = consumed or empty
};

static constexpr int LOOKAHEAD_SLOTS = 3;

struct RuntimeSMEM {
    uint64_t mbarriers[NUM_MBARRIERS];
    __align__(16) char task_buf[2][sizeof(TaskDesc)];
    int terminate;

    size_t next_task_pos;
    size_t last_task_pos;
    TaskId current_task_id;

    volatile int pf_weight_idx;
    volatile int pf_gen;

    // Phase E: look-ahead prefetch buffer
    LookaheadEntry lookahead[LOOKAHEAD_SLOTS];
    volatile int lookahead_gen;  // incremented when new lookahead entries are written

    __device__ __forceinline__ TaskDesc* task_slot(int phase) {
        return reinterpret_cast<TaskDesc*>(task_buf[phase]);
    }
};

// Forward decl: v2 dispatch lives in dispatch_v2.cuh (same namespace).
// persistent_kernel_v2.cuh includes runtime_v2.cuh then dispatch_v2.cuh, so
// the body is visible at link time.
__device__ __forceinline__ void
_execute_task_v2(TaskDesc const *task_desc,
                 RuntimeConfig const &config);

// ============ Prefetch helpers ============

__device__ __forceinline__ void prefetch_buffer_l2(const char* ptr, size_t bytes) {
    int lane = threadIdx.x % 32;
    for (size_t off = lane * 128; off < bytes; off += 32 * 128) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(ptr + off));
    }
}

__device__ __forceinline__ bool is_weight_task(TaskType t) {
    // SM100 linear/MoE tasks that load weights via TMA
    if (t >= TASK_SM100_TMA_START_TASK && t <= TASK_SM100_TMA_END_TASK) return true;
    // Also match TASK_IDENTITY used in tests as a stand-in for linear
    if (t == TASK_IDENTITY) return true;
    return false;
}

static constexpr size_t PREFETCH_MAX_BYTES = 1024 * 1024; // 1MB cap per input

// L2 prefetch ceiling: see v2_findings_share.md §1.2. At ≥ 64 SMs the working
// set exceeds L2 (148 × 1MB > 133MB) and prefetches get evicted before use —
// wasted HBM bandwidth. Gate the prefetch warp accordingly.
static constexpr int PREFETCH_SM_CEILING = 64;

// ============ Compute warp loop (W0-7) ============

__device__ __noinline__ void compute_warp_loop(
    RuntimeSMEM* rt, RuntimeConfig const &config, int warp_id, int lane_id
) {
    int phase = 0;

    while (true) {
        // Wait for controller to set up task desc in SMEM
        if (lane_id == 0) {
            mbar_wait(&rt->mbarriers[MBAR_TASK_READY], phase);
        }
        __syncwarp();

        TaskDesc* task = rt->task_slot(phase);

        if (task->task_type == TASK_TERMINATE) return;

        // Wait for event warp to confirm dependencies are met
        if (lane_id == 0) {
            mbar_wait(&rt->mbarriers[MBAR_DEPS_CLEAR], phase);
        }
        __syncwarp();

        // Skip non-executable sentinel. Hand-written v2 dispatch in
        // dispatch_v2.cuh handles Qwen3-8B task types with the v2 warp layout.
        if (task->task_type != TASK_BEGIN_TASK_GRAPH) {
            _execute_task_v2(task, config);
        }

        if (lane_id == 0) {
            mbar_arrive(&rt->mbarriers[MBAR_TASK_DONE]);
        }
        phase ^= 1;
    }
}

// ============ Controller warp loop (W8) ============
// v2 true persistent kernel: iter loop runs entirely on device.
//   1. Walk static per-SM task list for current iter
//   2. Atomic-add iter_sync_counter at end of iter
//   3. SM 0 waits for all SMs to arrive, runs prepare_next_batch, bumps
//      iter_go_counter
//   4. All other SMs wait for iter_go_counter to advance
//   5. Check termination (step >= max_seq_length) and loop or exit
//
// Host calls launch_persistent_kernel_v2 ONCE per decode request.

} // namespace runtime_v2
} // namespace mirage

// Forward: prepare_next_batch is defined in persistent_kernel.cuh
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

__device__ __noinline__ void controller_warp_loop(
    RuntimeSMEM* rt, RuntimeConfig const &config, int lane_id
) {
    // asm volatile("setmaxnreg.dec.sync.aligned.u32 64;"); // disabled for B2
    int const worker_id = blockIdx.x;
    int const num_workers = config.num_workers;

    size_t const my_offset = config.v2_per_sm_task_offsets[worker_id];
    size_t const my_end    = config.v2_per_sm_task_offsets[worker_id + 1];
    size_t const my_count  = my_end - my_offset;

    // Init SMEM state
    if (lane_id == 0) {
        rt->pf_gen = 0;
        rt->lookahead_gen = 0;
        for (int i = 0; i < LOOKAHEAD_SLOTS; i++)
            rt->lookahead[i].ready = 0;
    }
    __syncwarp();

    int phase = 0;

    for (int iter_num = 0; iter_num < config.v2_max_iters; iter_num++) {
        // ── Pre-iter: prepare_next_batch must run BEFORE this iter's tasks
        //    (v1 does this via scheduler processing END_OF_TASK_GRAPH). It
        //    sets up step/request_ids/qo_indptr that iter 0 tasks depend on.
        //    SM 0 does the work under lane-0, everyone else waits. ──
        if (worker_id == 0) {
            if (lane_id == 0) {
#if defined(MODE_OFFLINE)
                (void)::prepare_next_batch(config);
#elif defined(MODE_ONLINE_NOTOKEN)
                (void)::prepare_next_batch(config, iter_num);
#endif
                __threadfence_system();
                atomicAdd_system(config.v2_iter_go_counter, 1ULL);
            }
        } else {
            if (lane_id == 0) {
                unsigned long long needed =
                    static_cast<unsigned long long>(iter_num + 1);
                while (::ld_acquire_sys_u64(
                           config.v2_iter_go_counter) < needed) {
                    __nanosleep(50);
                }
            }
        }
        __syncwarp();

        // ── Run one iteration's worth of tasks from the static list ──
        for (size_t i = 0; i < my_count; i++) {
            size_t task_pos = config.v2_per_sm_task_positions[my_offset + i];

            if (lane_id == 0) {
                rt->current_task_id =
                    (static_cast<TaskId>(iter_num) << 32) |
                    (task_pos & 0xffffffff);
            }
            __syncwarp();

            // Load TaskDesc (32 lanes cover 352 bytes in one round)
            {
                char* dst = rt->task_buf[phase];
                char const* src = reinterpret_cast<char const*>(
                    &config.all_tasks[task_pos]);
                constexpr int CHUNKS = (sizeof(TaskDesc) + 15) / 16;
                for (int c = lane_id; c < CHUNKS; c += 32) {
                    ::kernel::load_smem(dst + c * 16, src + c * 16);
                }
                ::kernel::cp_async_fence();
                ::kernel::cp_async_wait<0>();
            }
            __syncwarp();

            // Lookahead for loader warp
            if (lane_id == 0) {
                int written = 0;
                for (int s = 0; s < LOOKAHEAD_SLOTS; s++) {
                    size_t peek_idx = i + 1 + s;
                    if (peek_idx < my_count) {
                        rt->lookahead[s].task_position =
                            config.v2_per_sm_task_positions[my_offset + peek_idx];
                        __threadfence_block();
                        rt->lookahead[s].ready = 1;
                        written++;
                    } else {
                        rt->lookahead[s].ready = 0;
                    }
                }
                rt->pf_weight_idx = phase;
                __threadfence_block();
                rt->pf_gen++;
                if (written > 0) rt->lookahead_gen++;
            }

            if (lane_id == 0) {
                mbar_arrive(&rt->mbarriers[MBAR_TASK_READY]);
            }
            if (lane_id == 0) {
                mbar_wait(&rt->mbarriers[MBAR_TASK_DONE], phase);
            }
            __syncwarp();

            phase ^= 1;
        }

        // ── Post-iter barrier: all SMs must finish this iter before the
        //    next iter's prepare_next_batch can run. Every SM atomics and
        //    waits until all SMs have atomic'd. ──
        __threadfence_system();
        if (lane_id == 0) {
            atomicAdd_system(config.v2_iter_sync_counter, 1ULL);
            unsigned long long needed =
                static_cast<unsigned long long>(num_workers) *
                static_cast<unsigned long long>(iter_num + 1);
            while (::ld_acquire_sys_u64(
                       config.v2_iter_sync_counter) < needed) {
                __nanosleep(50);
            }
        }
        __syncwarp();

        // ── Termination check: if step hit cap, exit ──
        int step0 = 0;
        if (lane_id == 0) {
            step0 = config.step[0];
        }
        step0 = __shfl_sync(0xFFFFFFFF, step0, 0);
        if (step0 >= config.max_seq_length - 1) break;
    }

    // Signal compute + event warps to exit
    rt->terminate = 1;
    if (lane_id == 0) {
        rt->task_slot(phase)->task_type = TASK_TERMINATE;
        __threadfence_block();
        mbar_arrive(&rt->mbarriers[MBAR_TASK_READY]);
    }
}

// ============ Prefetch warp loop (W9) ============
// Phase E: look-ahead prefetch. Scans the lookahead buffer for future
// memory-bound tasks and prefetches their weight data to L2.

__device__ __noinline__ void prefetch_warp_loop(
    RuntimeSMEM* rt, RuntimeConfig const &config
) {
    // Skip prefetch entirely when past the L2 ceiling — prefetched lines would
    // be evicted before consumption, burning HBM bandwidth for no benefit.
    if (config.num_workers >= PREFETCH_SM_CEILING) {
        while (!rt->terminate) __nanosleep(1000);
        return;
    }

    int last_lookahead_gen = 0;

    while (!rt->terminate) {
        int gen = rt->lookahead_gen;
        if (gen > last_lookahead_gen) {
            // Scan lookahead entries for weight-loading tasks
            for (int s = 0; s < LOOKAHEAD_SLOTS; s++) {
                if (!rt->lookahead[s].ready) continue;

                size_t pos = rt->lookahead[s].task_position;
                TaskDesc const* future_task = &config.all_tasks[pos];

                if (is_weight_task(future_task->task_type)) {
                    // Prefetch weight inputs (typically input_ptrs[0] or [1])
                    for (int inp = 0; inp < MAX_INPUTS_PER_TASK; inp++) {
                        void* ptr = future_task->input_ptrs[inp];
                        if (ptr == nullptr) break;
                        prefetch_buffer_l2(static_cast<const char*>(ptr),
                                           PREFETCH_MAX_BYTES);
                    }
                }
            }
            last_lookahead_gen = gen;
        } else {
            __nanosleep(10);
        }
    }
}

// ============ Event warp loop (W10) ============
// Handles dep-wait (non-blocking for controller) and event triggering.
// Flow per task:
//   1. Wait TASK_READY (know which task to check deps for)
//   2. Poll dependent_event counter until satisfied → signal DEPS_CLEAR
//   3. Wait TASK_DONE (task finished executing)
//   4. atomicAdd trigger_event counter (notify downstream tasks on other SMs)

__device__ __noinline__ void event_warp_loop(
    RuntimeSMEM* rt, RuntimeConfig const &config
) {
    // asm volatile("setmaxnreg.dec.sync.aligned.u32 64;"); // disabled for B2
    int lane_id = threadIdx.x % 32;
    int phase = 0;

    while (true) {
        if (lane_id == 0) {
            mbar_wait(&rt->mbarriers[MBAR_TASK_READY], phase);
        }
        __syncwarp();

        TaskDesc* task = rt->task_slot(phase);
        if (task->task_type == TASK_TERMINATE) return;

        if (lane_id == 0) {
            EventId dep = task->dependent_event;
            if (dep != EVENT_INVALID_ID && !is_nvshmem_event(dep)) {
                size_t event_index = get_event_position_index(dep);
                // v1 formula is `needed = num_triggers * iter_num` which
                // evaluates to 0 on iter 0 — v1 survives because its scheduler
                // push-order serialized consumers behind producers. v2 has no
                // scheduler: static per-SM plan + round-robin assignment does
                // NOT enforce cross-SM order, so the consumer must actually
                // wait for its own iter's producers. Use (iter_num+1).
                EventCounter needed =
                    static_cast<EventCounter>(
                        config.all_event_num_triggers[event_index]) *
                    (get_task_iteration_num(rt->current_task_id) + 1);
                while (ld_acquire_sys_u64(
                           &config.all_event_counters[event_index]) < needed) {
                    __nanosleep(10);
                }
            }
        }
        __syncwarp();

        if (lane_id == 0) {
            mbar_arrive(&rt->mbarriers[MBAR_DEPS_CLEAR]);
        }

        if (lane_id == 0) {
            mbar_wait(&rt->mbarriers[MBAR_TASK_DONE], phase);
        }
        __syncwarp();

        if (lane_id == 0) {
            EventId event_id = task->trigger_event;
            if (event_id != EVENT_INVALID_ID && !is_nvshmem_event(event_id)) {
                size_t event_index = get_event_position_index(event_id);
                atom_add_release_gpu_u64(
                    &config.all_event_counters[event_index], 1);
            }
        }

        phase ^= 1;
    }
}

// ============ Kernel entry ============

__global__ __launch_bounds__(NUM_THREADS, 1)
void worker_v2_kernel(RuntimeConfig config) {

    __shared__ __align__(16) char rt_buf[sizeof(RuntimeSMEM)];
    RuntimeSMEM* rt = reinterpret_cast<RuntimeSMEM*>(rt_buf);

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (threadIdx.x == 0) {
        mbar_init(&rt->mbarriers[MBAR_TASK_READY], 1);        // controller arrives
        mbar_init(&rt->mbarriers[MBAR_TASK_DONE], NUM_COMPUTE_WARPS); // 8 compute warps arrive
        mbar_init(&rt->mbarriers[MBAR_DEPS_CLEAR], 1);        // event warp arrives
        asm volatile("fence.mbarrier_init.release.cluster;");
        rt->terminate = 0;
    }

    __syncthreads();

    if (warp_id < NUM_COMPUTE_WARPS) {
        compute_warp_loop(rt, config, warp_id, lane_id);
    } else if (warp_id == CONTROLLER_WARP) {
        controller_warp_loop(rt, config, lane_id);
    } else if (warp_id == PREFETCH_WARP) {
        prefetch_warp_loop(rt, config);
    } else if (warp_id == EVENT_WARP) {
        event_warp_loop(rt, config);
    }
    // No else — with NUM_WARPS=11, warp_ids 0..10 cover everything.
}

// ============ Launch helper ============
// Called from launch_persistent_kernel() in persistent_kernel.cuh

inline void launch_worker_v2(RuntimeConfig const &config,
                             int num_workers,
                             cudaStream_t stream) {
    int smem = MAX_DYNAMIC_SHARED_MEMORY_SIZE;
    cudaFuncSetAttribute(worker_v2_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    worker_v2_kernel<<<dim3(num_workers, 1, 1),
                       dim3(NUM_THREADS, 1, 1),
                       smem, stream>>>(config);
}

} // namespace runtime_v2
} // namespace mirage
