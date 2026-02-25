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

#if defined(MIRAGE_GRACE_BLACKWELL)
// ── mbarrier helpers (PTX) ──────────────────────────────────────────────────
__device__ __forceinline__ void mbar_init(uint64_t *mbar, uint32_t count) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               ::"r"(smem_addr), "r"(count));
}

__device__ __forceinline__ void mbar_arrive(uint64_t *mbar) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n"
               ::"r"(smem_addr));
}

__device__ __forceinline__ void mbar_wait(uint64_t *mbar, uint32_t phase) {
  uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(mbar));
  uint32_t done;
  do {
    asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
        "  selp.u32 %0, 1, 0, p;\n"
        "}\n"
        : "=r"(done)
        : "r"(smem_addr), "r"(phase));
    if (!done) __nanosleep(BARRIER_SPIN_SLEEP_NS);
  } while (!done);
}

// ── Pipelined mainloop (Blackwell SM100) ────────────────────────────────────
// Controller warp 8 handles dependency-wait + task prefetch, pipelined with
// compute warps 0-7 executing the current task. Overlaps dep-wait for task
// N+1 with execution of task N.
//
// SMEM semaphores (mbarrier-based, 2-stage pipeline):
//   task_ready[2] — controller → compute: "dependency met, go execute"
//   task_done[2]  — compute → controller: "execution complete"
__device__ __forceinline__ void
    static_mainloop_pipelined(RuntimeConfig const &config, int worker_id) {
  __shared__ uint64_t task_ready_mbar[2];
  __shared__ uint64_t task_done_mbar[2];
  __shared__ TaskDesc task_buf[2];
  __shared__ int signal_barrier_idx;

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

  // Early exit if this worker has no tasks
  if (worker_id >= total) {
    return;
  }

  // Count how many round-robin tasks this worker owns
  int my_task_count = 0;
  for (int k = 0; k * nw + worker_id < total; k++) {
    my_task_count++;
  }

  static_assert(sizeof(TaskDesc) % 16 == 0);
  constexpr int TASK_SIZE = sizeof(TaskDesc) / 16; // 128-bit units

  // Init mbarriers (thread 0 only, then full-block sync)
  if (threadIdx.x == 0) {
    mbar_init(&task_ready_mbar[0], 1); // controller arrives once
    mbar_init(&task_ready_mbar[1], 1);
    mbar_init(&task_done_mbar[0], 1);  // one compute leader arrives once
    mbar_init(&task_done_mbar[1], 1);
  }
  __syncthreads(); // full-block sync (all 288 threads) to see init

  int const warp_id = threadIdx.x / 32;

  if (warp_id == CONTROLLER_WARP_ID) {
    // ── Controller warp (warp 8, 32 threads) ──
    int const lane = threadIdx.x % 32;

    for (int i = 0; i < my_task_count; i++) {
      int const cur = i & 1;
      int const phase = (i >> 1) & 1;
      int const pos = base + i * nw + worker_id;

      // Load current task into task_buf[cur]
      if (i == 0) {
        // First task: explicit load
        for (int j = lane; j < TASK_SIZE; j += 32) {
          load_smem(reinterpret_cast<char *>(&task_buf[cur]) + j * 16,
                    reinterpret_cast<char const *>(config.all_tasks + pos) +
                        j * 16);
        }
        kernel::cp_async_fence();
        kernel::cp_async_wait<0>();
        __syncwarp();
      } else {
        // Subsequent tasks: prefetch was issued in previous iteration
        kernel::cp_async_wait<0>();
        __syncwarp();
      }

      if (lane == 0) {
        // Wait on dependency barrier (GMEM)
        EventId dep = task_buf[cur].dependent_event;
        if (dep != EVENT_INVALID_ID) {
          int eidx = (int)(dep & 0xFFFFFFFF);
          barrier_wait(
              config.barriers, eidx, config.all_event_num_triggers[eidx]);
        }

        // Decode trigger event for later GMEM signal
        EventId trig = task_buf[cur].trigger_event;
        signal_barrier_idx =
            (trig != EVENT_INVALID_ID) ? (int)(trig & 0xFFFFFFFF) : -1;

        // Signal compute warps: task is ready
        mbar_arrive(&task_ready_mbar[cur]);
      }

      // Prefetch next task (all 32 controller threads help with cp_async)
      if (i + 1 < my_task_count) {
        int const nxt = 1 - cur;
        int const next_pos = base + (i + 1) * nw + worker_id;
        for (int j = lane; j < TASK_SIZE; j += 32) {
          load_smem(reinterpret_cast<char *>(&task_buf[nxt]) + j * 16,
                    reinterpret_cast<char const *>(config.all_tasks +
                                                   next_pos) +
                        j * 16);
        }
        kernel::cp_async_fence();
        // Don't wait — overlaps with compute execution
      }

      if (lane == 0) {
        // Wait for compute warps to finish execution
        mbar_wait(&task_done_mbar[cur], phase);
      }
      __syncwarp();
    }
  } else {
    // ── Compute warps (warps 0-7, 256 threads) ──
    for (int i = 0; i < my_task_count; i++) {
      int const cur = i & 1;
      int const phase = (i >> 1) & 1;

      // Wait for controller to signal task is ready
      if (threadIdx.x == 0) {
        mbar_wait(&task_ready_mbar[cur], phase);
      }
      TASK_SYNC(); // broadcast to all 256 compute threads

#ifdef MPK_ENABLE_PROFILING
      PROFILER_EVENT_START(task_buf[cur].task_type, task_counter);
#endif

      // Execute task (256 compute threads)
      _execute_task(&task_buf[cur], config);

      TASK_SYNC();

#ifdef MPK_ENABLE_PROFILING
      PROFILER_EVENT_END(task_buf[cur].task_type, task_counter++);
#endif

      // Signal GMEM completion barrier (cross-SM dependency).
      // Done by compute thread 0 (which participated in TASK_SYNC / membar.cta)
      // so that atom.add.release.gpu properly makes all compute writes visible.
      if (threadIdx.x == 0 && signal_barrier_idx >= 0) {
        barrier_arrive(config.barriers, signal_barrier_idx);
      }

      // Signal controller: execution complete
      if (threadIdx.x == 0) {
        mbar_arrive(&task_done_mbar[cur]);
      }
    }
  }
}

#else // !MIRAGE_GRACE_BLACKWELL

// ── Non-pipelined mainloop (batch-loaded) ───────────────────────────────────
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

#endif // MIRAGE_GRACE_BLACKWELL

// ── Top-level ───────────────────────────────────────────────────────────────
__device__ __forceinline__ void execute_worker_static(RuntimeConfig config) {
  int const worker_id = blockIdx.x;

  for (size_t iter = 1;; iter++) {
    // Execute tasks (each worker computes its own round-robin list)
#if defined(MIRAGE_GRACE_BLACKWELL)
    static_mainloop_pipelined(config, worker_id);
#else
    static_mainloop(config, worker_id);
#endif

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
