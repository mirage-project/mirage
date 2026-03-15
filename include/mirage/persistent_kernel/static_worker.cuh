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

// Static worker with batched task descriptor loading.
// Tasks are loaded in small batches (like the dynamic worker path) to keep
// static SMEM usage within WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE,
// leaving the rest available as dynamic shared memory for task execution.
//
// 12 warps = 384 threads:
//   Warps 0-7  (256 threads): Consumer — execute tasks
//   Warps 8-11 (128 threads): Participate in batch load, idle during execution

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

// Same buffer length as dynamic worker path.
constexpr int TASK_DESCS_BUFFER_LENGTH_STATIC = std::min(
    (mirage::runtime::WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE - 56) /
        (int)(sizeof(TaskDesc)),
    16);

// ── Top-level ───────────────────────────────────────────────────────────────
__device__ __forceinline__ void execute_worker_static(RuntimeConfig config) {
  int const worker_id = blockIdx.x;

  __shared__ TaskDesc task_descs[TASK_DESCS_BUFFER_LENGTH_STATIC];

  int const nw = config.num_workers;
  int const base = config.first_compute_task_index;
  int const total = config.num_compute_tasks;

  // Count tasks for this worker
  int const num_tasks = (total > worker_id) ? (total - worker_id - 1) / nw + 1 : 0;

#ifdef MPK_ENABLE_PROFILING
  PROFILER_CLOSURE_PARAMS_DECL;
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),
                0, 1, (threadIdx.x == 0));
  size_t task_counter = 0;
#endif

  int const warp_idx = threadIdx.x / 32;
  // bool const is_profiler = (worker_id == 0 && threadIdx.x == 0);
  bool const is_profiler = false; // Disable profiler for static worker for now, as the timing is not very meaningful due to the batched execution and load.

  // Per-iteration cycle accumulators (worker 0, thread 0 only)
  long long cyc_load = 0, cyc_dep_wait = 0, cyc_sync1 = 0;
  long long cyc_exec = 0, cyc_sync2 = 0, cyc_trigger = 0, cyc_sync3 = 0;
  long long cyc_end_barrier = 0, cyc_barrier_reset = 0;
  long long cyc_prepare = 0, cyc_prepare_wait = 0;
  long long cyc_iter_start = 0;
  int prof_task_count = 0, prof_load_count = 0;

  for (size_t iter = 1;; iter++) {
    if (is_profiler) {
      cyc_load = cyc_dep_wait = cyc_sync1 = 0;
      cyc_exec = cyc_sync2 = cyc_trigger = cyc_sync3 = 0;
      cyc_end_barrier = cyc_barrier_reset = cyc_prepare = cyc_prepare_wait = 0;
      prof_task_count = prof_load_count = 0;
      cyc_iter_start = clock64();
    }

    if (num_tasks > 0) {
      int queue_pos = 0, queue_len = 0;
      int next_task = 0; // next task index to load

      while (next_task < num_tasks || queue_pos < queue_len) {
        // Load next batch when current batch is exhausted
        if (queue_pos == queue_len) {
          long long t0 = is_profiler ? clock64() : 0;

          int num_loaded_tasks =
              min(num_tasks - next_task, TASK_DESCS_BUFFER_LENGTH_STATIC);

          // Load task descs — same as dynamic worker
          static_assert(sizeof(TaskDesc) % 16 == 0);
          constexpr int TASK_SIZE = sizeof(TaskDesc) / 16; // 128b copy-async
          for (int i = threadIdx.x; i < num_loaded_tasks * TASK_SIZE;
               i += blockDim.x) {
            int task_idx = i / TASK_SIZE;
            int offset = i % TASK_SIZE;
            int pos = base + (next_task + task_idx) * nw + worker_id;
            load_smem(reinterpret_cast<char *>(task_descs) + i * 16,
                      reinterpret_cast<char *>(
                          config.all_tasks + pos) +
                          offset * 16);
          }
          kernel::cp_async_fence();
          kernel::cp_async_wait<0>();
          __syncthreads();
          queue_pos = 0;
          queue_len = num_loaded_tasks;
          next_task += num_loaded_tasks;

          if (is_profiler) { cyc_load += clock64() - t0; prof_load_count++; }
        }

        TaskDesc *task = task_descs + queue_pos;
        long long t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0;

        if (warp_idx < ROLE_CONSUMER_WARPS_END) {
          // Consumer thread 0: wait for cross-SM dependency
          if (is_profiler) t1 = clock64();
          if (threadIdx.x == 0) {
            EventId dep = task->dependent_event;
            if (dep != EVENT_INVALID_ID) {
              int eidx = (int)(dep & 0xFFFFFFFF);
              global_barrier_wait(config.barriers, eidx,
                                  config.all_event_num_triggers[eidx]);
            }
          }
          if (is_profiler) t2 = clock64();
          MPK_CONSUMER_SYNC();
          if (is_profiler) t3 = clock64();

          // All consumers execute
#ifdef MPK_ENABLE_PROFILING
          PROFILER_EVENT_START(task->task_type, task_counter);
#endif

          _execute_task(task, config);

          if (is_profiler) t4 = clock64();
          MPK_CONSUMER_SYNC();
          if (is_profiler) t5 = clock64();

#ifdef MPK_ENABLE_PROFILING
          PROFILER_EVENT_END(task->task_type, task_counter++);
#endif

          // Thread 0: signal cross-SM trigger event
          if (threadIdx.x == 0) {
            EventId trig = task->trigger_event;
            if (trig != EVENT_INVALID_ID) {
              int eidx = (int)(trig & 0xFFFFFFFF);
              global_barrier_arrive(config.barriers, eidx);
            }
          }
          if (is_profiler) t6 = clock64();
        }

        if (is_profiler) {
          cyc_dep_wait += t2 - t1;
          cyc_sync1 += t3 - t2;
          cyc_exec += t4 - t3;
          cyc_sync2 += t5 - t4;
          cyc_trigger += t6 - t5;
          prof_task_count++;
        }

        __syncthreads();
        queue_pos++;
      }
    }
    // ── End-of-iteration ───────────────────────────────────────────────────
    long long te0 = is_profiler ? clock64() : 0;
    if (threadIdx.x == 0) {
      global_barrier_wait(
          config.barriers, config.end_barrier, config.end_barrier_count);
    }
    long long te1 = is_profiler ? clock64() : 0;
    __syncthreads();

#if defined(MODE_OFFLINE) || defined(MODE_ONLINE) ||                           \
    defined(MODE_ONLINE_NOTOKEN)
    {
      long long te2 = 0, te3 = 0;
      if (worker_id == 0 && threadIdx.x == 0) {
        te2 = clock64();
        for (int i = 0; i < config.num_barriers; i++) {
          config.barriers[i] = 0;
        }
        te3 = clock64();
#ifdef MODE_ONLINE_NOTOKEN
        bool ok = prepare_next_batch(config, iter);
#else
        bool ok = prepare_next_batch(config);
#endif
        *config.continue_flag = ok;
        __threadfence();
        atomicAdd(config.prepare_done_counter, 1ULL);
      }
      if (is_profiler) {
        cyc_barrier_reset = te3 - te2;
        cyc_prepare = clock64() - te3;
      }
    }
    long long te4 = is_profiler ? clock64() : 0;
    if (threadIdx.x == 0) {
      while (ld_acquire_sys_u64(config.prepare_done_counter) < iter) {
        __nanosleep(BARRIER_SPIN_SLEEP_NS);
      }
    }
    __syncthreads();

    if (is_profiler) {
      cyc_end_barrier = te1 - te0;
      cyc_prepare_wait = clock64() - te4;
      long long cyc_total = clock64() - cyc_iter_start;
      printf("[STATIC iter=%llu] total=%lld  tasks=%d  loads=%d\n"
             "  load=%lld  dep_wait=%lld  sync1=%lld  exec=%lld  sync2=%lld  trigger=%lld\n"
             "  end_barrier=%lld  barrier_reset=%lld  prepare=%lld  prepare_wait=%lld\n",
             (unsigned long long)iter, cyc_total, prof_task_count, prof_load_count,
             cyc_load, cyc_dep_wait, cyc_sync1, cyc_exec, cyc_sync2, cyc_trigger,
             cyc_end_barrier, cyc_barrier_reset, cyc_prepare, cyc_prepare_wait);
    }

    if (!(*config.continue_flag)) {
      return;
    }
#else
    return;
#endif
  }
}
