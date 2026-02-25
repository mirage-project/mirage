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

// Static worker: no scheduler, no controller SM.
// Each worker computes its own round-robin task list locally.
// Cross-SM dependencies via GMEM int32 barriers (Megakernels g.Bar pattern).
// Per-task barrier info pre-computed at init — no EventId decoding on GPU.

constexpr int BARRIER_SPIN_SLEEP_NS = 20;

// Release variant: flushes L1 write-back cache to L2 before signaling.
// Required for tasks that write output via regular stores (non-TMA).
__device__ __forceinline__ void barrier_arrive(int *barriers, int idx) {
  atom_add_release_gpu_s32(&barriers[idx], 1);
}


__device__ __forceinline__ void
    barrier_wait(int *barriers, int idx, int expected) {
  while (ld_acquire_gpu_s32(&barriers[idx]) < expected) {
    __nanosleep(BARRIER_SPIN_SLEEP_NS);
  }
}

// ── Mainloop ────────────────────────────────────────────────────────────────
// Each worker executes its round-robin share of compute tasks.
// Task positions computed locally: first_compute + i * num_workers + worker_id.
// Batch-loaded: load up to BATCH_SIZE tasks at once via cp_async, then iterate
// through them with zero load overhead (same pattern as dynamic worker).
__device__ __forceinline__ void static_mainloop(RuntimeConfig const &config,
                                                int worker_id) {
  // Batch buffer: same pattern as dynamic worker's task_descs[]
  constexpr int BATCH_SIZE = std::min(
      (mirage::runtime::WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE) /
          (int)(sizeof(TaskDesc)),
      16);
  __shared__ TaskDesc task_descs[BATCH_SIZE];

#ifdef MPK_ENABLE_PROFILING
  PROFILER_CLOSURE_PARAMS_DECL;
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),
                0,
                1,
                (threadIdx.x % WORKER_NUM_THREADS == 0));
  size_t task_counter = 0;
#endif

  int const nw = config.num_workers;
  int const base = config.first_compute_task_index;
  int const total = config.num_compute_tasks;

  static_assert(sizeof(TaskDesc) % 16 == 0);
  constexpr int TASK_SIZE = sizeof(TaskDesc) / 16; // 128-bit units

  // Early exit if this worker has no tasks
  if (worker_id >= total) {
    return;
  }

  int queue_pos = 0, queue_len = 0;
  int task_idx = 0; // which round-robin slot we're on (i in the old loop)

  while (task_idx * nw + worker_id < total) {
    // Batch-load: fill task_descs[] with up to BATCH_SIZE tasks
    if (queue_pos == queue_len) {
      int remaining = 0;
      for (int k = task_idx; k * nw + worker_id < total; k++) {
        remaining++;
      }
      int num_to_load = min(remaining, BATCH_SIZE);

      // Batch cp_async: load all tasks in one shot
      for (int i = threadIdx.x; i < num_to_load * TASK_SIZE;
           i += blockDim.x) {
        int slot = i / TASK_SIZE;
        int offset = i % TASK_SIZE;
        int pos = base + (task_idx + slot) * nw + worker_id;
        load_smem(reinterpret_cast<char *>(task_descs) + i * 16,
                  reinterpret_cast<char const *>(config.all_tasks + pos) +
                      offset * 16);
      }
      kernel::cp_async_fence();
      kernel::cp_async_wait<0>();
      __syncthreads();
      queue_pos = 0;
      queue_len = num_to_load;
    }

    TaskDesc *task_desc = &task_descs[queue_pos];

    // Thread 0: decode barrier info + dependency wait
    int signal_barrier_idx = -1;
    if (threadIdx.x == 0) {
      EventId trig = task_desc->trigger_event;
      signal_barrier_idx =
          (trig != EVENT_INVALID_ID) ? (int)(trig & 0xFFFFFFFF) : -1;

      // Wait on dependency barrier
      EventId dep = task_desc->dependent_event;
      if (dep != EVENT_INVALID_ID) {
        int eidx = (int)(dep & 0xFFFFFFFF);
        barrier_wait(
            config.barriers, eidx, config.all_event_num_triggers[eidx]);
      }
    }
    __syncthreads();

#ifdef MPK_ENABLE_PROFILING
    PROFILER_EVENT_START(task_desc->task_type, task_counter);
#endif

    // Execute
    _execute_task(task_desc, config);
    __syncthreads();

#ifdef MPK_ENABLE_PROFILING
    PROFILER_EVENT_END(task_desc->task_type, task_counter++);
#endif

    // Signal completion barrier
    if (threadIdx.x == 0 && signal_barrier_idx >= 0) {
      barrier_arrive(config.barriers, signal_barrier_idx);
    }

    queue_pos++;
    task_idx++;
  }
}

// ── Top-level ───────────────────────────────────────────────────────────────
__device__ __forceinline__ void execute_worker_static(RuntimeConfig config) {
  int const worker_id = blockIdx.x;

  for (size_t iter = 1;; iter++) {
    // Execute tasks (each worker computes its own round-robin list)
    static_mainloop(config, worker_id);

    // Inter-iteration barrier: wait for all tasks to complete
    if (threadIdx.x == 0) {
      barrier_wait(
          config.barriers, config.end_barrier, config.end_barrier_count);
    }
    __syncthreads();

    // Prepare next batch (multi-iteration modes only)
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE) ||                           \
    defined(MODE_ONLINE_NOTOKEN)
    if (worker_id == 0 && threadIdx.x == 0) {
      // Reset barriers for next iteration (safe: everyone past end_barrier)
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
