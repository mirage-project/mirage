/* Copyright 2025 CMU
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

// Warp-specialized static worker (MegaKernels pattern).
// 12 warps = 3 warpgroups:
//   WG0 (warps 0-3):  Consumer  — execute tasks
//   WG1 (warps 4-7):  Consumer  — execute tasks
//   WG2 (warps 8-11): Non-consumer
//     Warp 8-10 = Helpers (idle during task execution)
//     Warp 11   = Controller (loads tasks, resolves GMEM deps)
//
// Synchronization:
//   __syncthreads() for controller→consumer task handoff (Phase 1).
//   MPK_CONSUMER_SYNC() (named barrier, 256 threads) for consumer-only sync.
//   GMEM int32 barriers for cross-SM dependencies (release/acquire).
//
// Future: mbarrier-based pipeline (PIPELINE_STAGES=2) for overlap.

constexpr int BARRIER_SPIN_SLEEP_NS = 20;

__device__ __forceinline__ void global_barrier_arrive(int *barriers, int idx) {
  atom_add_release_gpu_s32(&barriers[idx], 1);
}

__device__ __forceinline__ void
    global_barrier_wait(int *barriers, int idx, int expected) {
  while (ld_acquire_gpu_s32(&barriers[idx]) < expected) {
    __nanosleep(BARRIER_SPIN_SLEEP_NS);
  }
}

// ── Top-level ───────────────────────────────────────────────────────────────
__device__ __forceinline__ void execute_worker_static(RuntimeConfig config) {
  int const worker_id = blockIdx.x;
  int const warp_idx = threadIdx.x / 32;
  int const lane = threadIdx.x % 32;

  // Shared instruction slot: controller loads here, consumers read after sync.
  __shared__ __align__(128) TaskDesc task_desc;
  __shared__ int trigger_event_idx;

  int const nw = config.num_workers;
  int const base = config.first_compute_task_index;
  int const total = config.num_compute_tasks;

  static_assert(sizeof(TaskDesc) % 16 == 0);
  constexpr int TASK_U128 = sizeof(TaskDesc) / 16;

#ifdef MPK_ENABLE_PROFILING
  PROFILER_CLOSURE_PARAMS_DECL;
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),
                0, 1, (threadIdx.x == 0));
  size_t task_counter = 0;
#endif

  for (size_t iter = 1;; iter++) {
    // ── Task loop ──────────────────────────────────────────────────────────
    for (int task_idx = 0; task_idx * nw + worker_id < total; task_idx++) {

      // Controller (warp 11 lane 0): load task + resolve GMEM dependency
      if (warp_idx == ROLE_CONTROLLER_WARP && lane == 0) {
        int pos = base + task_idx * nw + worker_id;
        TaskDesc const *src = &config.all_tasks[pos];
        TaskDesc *dst = &task_desc;
        for (int i = 0; i < TASK_U128; i++) {
          reinterpret_cast<uint4 *>(dst)[i] =
              reinterpret_cast<uint4 const *>(src)[i];
        }

        EventId trig = task_desc.trigger_event;
        trigger_event_idx =
            (trig != EVENT_INVALID_ID) ? (int)(trig & 0xFFFFFFFF) : -1;

        EventId dep = task_desc.dependent_event;
        if (dep != EVENT_INVALID_ID) {
          int eidx = (int)(dep & 0xFFFFFFFF);
          global_barrier_wait(
              config.barriers, eidx, config.all_event_num_triggers[eidx]);
        }
      }

      // Broadcast task to all threads
      __syncthreads();

      // Consumer warps (0-7) execute the task.
      // Non-consumer warps MUST NOT call _execute_task — tasks use
      // MPK_CONSUMER_SYNC (NamedBarrier count=256). Extra threads would
      // corrupt the barrier state (undefined behavior).
      if (warp_idx < ROLE_CONSUMER_WARPS_END) {
#ifdef MPK_ENABLE_PROFILING
        PROFILER_EVENT_START(task_desc.task_type, task_counter);
#endif

        _execute_task(&task_desc, config);
        MPK_CONSUMER_SYNC();

#ifdef MPK_ENABLE_PROFILING
        PROFILER_EVENT_END(task_desc.task_type, task_counter++);
#endif
      }

      // Re-sync all 384 threads (consumers + non-consumers) before signaling
      __syncthreads();

      // Thread 0: signal trigger event (cross-SM GMEM barrier)
      if (threadIdx.x == 0) {
        int eidx = trigger_event_idx;
        if (eidx >= 0) {
          global_barrier_arrive(config.barriers, eidx);
        }
      }
    }

    // ── End-of-iteration ───────────────────────────────────────────────────
    if (threadIdx.x == 0) {
      global_barrier_wait(
          config.barriers, config.end_barrier, config.end_barrier_count);
    }
    __syncthreads();

#if defined(MODE_OFFLINE) || defined(MODE_ONLINE) ||                           \
    defined(MODE_ONLINE_NOTOKEN)
    {
      if (worker_id == 0 && threadIdx.x == 0) {
        for (int i = 0; i < config.num_barriers; i++) {
          config.barriers[i] = 0;
        }
#ifdef MODE_ONLINE_NOTOKEN
        bool ok = prepare_next_batch(config, iter);
#else
        bool ok = prepare_next_batch(config);
#endif
        *config.continue_flag = ok;
        __threadfence();
        atomicAdd(config.prepare_done_counter, 1ULL);
      }
    }
    if (threadIdx.x == 0) {
      while (ld_acquire_sys_u64(config.prepare_done_counter) < iter) {
        __nanosleep(BARRIER_SPIN_SLEEP_NS);
      }
    }
    __syncthreads();
    if (!(*config.continue_flag)) {
      return;
    }
#else
    return;
#endif
  }
}
