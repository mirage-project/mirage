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

#ifdef MPK_ENABLE_VERBOSE
__device__ const char *task_type_name(int t) {
  switch (t) {
    case 0:   return "TERMINATE";
    case 10:  return "BEGIN_TASK_GRAPH";
    case 101: return "EMBEDDING";
    case 102: return "RMS_NORM_LINEAR";
    case 103: return "ATTENTION_1";
    case 104: return "ATTENTION_2";
    case 105: return "SILU_MUL_LINEAR_W_RES";
    case 106: return "ALLREDUCE";
    case 107: return "REDUCE";
    case 108: return "LINEAR_W_RES";
    case 109: return "ARGMAX";
    case 110: return "ARGMAX_PARTIAL";
    case 111: return "ARGMAX_REDUCE";
    case 112: return "FIND_NGRAM_PARTIAL";
    case 113: return "FIND_NGRAM_GLOBAL";
    case 114: return "TARGET_VERIFY_GREEDY";
    case 115: return "SB_EXTEND_ATTN";
    case 116: return "PAGED_ATTN_1";
    case 117: return "PAGED_ATTN_2";
    case 118: return "SILU_MUL";
    case 119: return "RMS_NORM";
    case 120: return "LINEAR";
    case 121: return "IDENTITY";
    case 151: return "LINEAR_W_RES_HOPPER";
    case 152: return "LINEAR_HOPPER";
    case 153: return "PAGED_ATTN_HOPPER";
    case 154: return "RMS_NORM_HOPPER";
    case 155: return "LINEAR_SWAPAB_HOPPER";
    case 156: return "LINEAR_SWAPAB_W_RES_HOPPER";
    case 157: return "LINEAR_CUTLASS_HOPPER";
    case 158: return "LINEAR_CUTLASS_W_RES_HOPPER";
    case 159: return "SILU_MUL_HOPPER";
    case 160: return "EMBEDDING_HOPPER";
    case 161: return "MOE_W13_LINEAR_SM90";
    case 162: return "MOE_W2_LINEAR_SM90";
    case 163: return "SPLITK_LINEAR_SWAPAB_HOPPER";
    case 164: return "PAGED_ATTN_SPLIT_KV_HOPPER";
    case 251: return "SPLITK_LINEAR_SM100";
    case 252: return "LINEAR_W_RES_SM100";
    case 253: return "LINEAR_SM100";
    case 254: return "MOE_W13_LINEAR_SM100";
    case 255: return "MOE_W2_LINEAR_SM100";
    case 257: return "ATTN_SM100";
    case 258: return "ARGMAX_REDUCE_SM100";
    case 259: return "ARGMAX_PARTIAL_SM100";
    case 260: return "MOE_TOPK_SOFTMAX_SM100";
    case 261: return "MOE_MUL_SUM_ADD_SM100";
    case 262: return "TENSOR_INIT";
    case 263: return "PAGED_ATTN_SPLIT_KV_SM100";
    case 264: return "PAGED_ATTN_SPLIT_KV_MERGE_SM100";
    case 265: return "SAMPLING_SM100";
    case 301: return "NVSHMEM_ALLGATHER_PUT";
    case 302: return "NVSHMEM_TILE_ALLREDUCE";
    default:  return "UNKNOWN";
  }
}
#endif

__device__ __forceinline__ void
barrier_arrive(int *barriers, int idx) {
  atomicAdd(&barriers[idx], 1);
}

__device__ __forceinline__ void
barrier_wait(int *barriers, int idx, int expected) {
  while (*(volatile int *)&barriers[idx] < expected)
    __nanosleep(BARRIER_SPIN_SLEEP_NS);
}

// ── Mainloop ────────────────────────────────────────────────────────────────
// Each worker executes its round-robin share of compute tasks.
// Task positions computed locally: first_compute + i * num_workers + worker_id.
// Barrier info derived from TaskDesc fields already in SMEM — no extra GPU array.
__device__ __forceinline__ void
    static_mainloop(RuntimeConfig const &config, int worker_id) {
  __shared__ TaskDesc task_smem;

#ifdef MPK_ENABLE_PROFILING
  PROFILER_CLOSURE_PARAMS_DECL;
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),
                0, 1,
                (threadIdx.x % WORKER_NUM_THREADS == 0));
  size_t task_counter = 0;
#endif

  int const nw = config.num_workers;
  int const base = config.first_compute_task_index;
  int const total = config.num_compute_tasks;

  for (int i = 0; base + i * nw + worker_id < base + total; i++) {
    int pos = base + i * nw + worker_id;

    // Load TaskDesc collectively (all threads participate)
    static_assert(sizeof(TaskDesc) % sizeof(int) == 0);
    constexpr int WORDS = sizeof(TaskDesc) / sizeof(int);
    int *dst = reinterpret_cast<int *>(&task_smem);
    int const *src = reinterpret_cast<int const *>(&config.all_tasks[pos]);
    for (int w = threadIdx.x; w < WORDS; w += blockDim.x)
      dst[w] = src[w];
    __syncthreads();

    // Derive barrier info from TaskDesc (already in SMEM).
    // Cache signal_barrier now — task_smem will be overwritten next iteration.
    __shared__ int signal_barrier_idx;
    if (threadIdx.x == 0) {
      // Cache signal barrier before execute
      EventId trig = task_smem.trigger_event;
      signal_barrier_idx = (trig != EVENT_INVALID_ID)
                               ? (int)(trig & 0xFFFFFFFF)
                               : -1;

      // Wait on dependency barrier
      EventId dep = task_smem.dependent_event;
      if (dep != EVENT_INVALID_ID) {
        int eidx = (int)(dep & 0xFFFFFFFF);
        barrier_wait(config.barriers, eidx,
                     config.all_event_num_triggers[eidx]);
      }
    }
    __syncthreads();

#ifdef MPK_ENABLE_VERBOSE
    if (threadIdx.x == 0) {
      printf("[STATIC] worker=%d pos=%d %s dep=%llx trig=%llx\n",
             worker_id, pos, task_type_name(task_smem.task_type),
             task_smem.dependent_event, task_smem.trigger_event);
    }
#endif

#ifdef MPK_ENABLE_PROFILING
    PROFILER_EVENT_START(task_smem.task_type, task_counter);
#endif

    // Execute
    _execute_task(&task_smem, config);
    __syncthreads();

#ifdef MPK_ENABLE_PROFILING
    PROFILER_EVENT_END(task_smem.task_type, task_counter++);
#endif

    // Signal completion barrier (reads cached index, no race with next load)
    if (threadIdx.x == 0 && signal_barrier_idx >= 0)
      barrier_arrive(config.barriers, signal_barrier_idx);
  }
}

// ── Top-level ───────────────────────────────────────────────────────────────
__device__ __forceinline__ void execute_worker_static(RuntimeConfig config) {
  int const worker_id = blockIdx.x;

  for (size_t iter = 1;; iter++) {
    // Execute tasks (each worker computes its own round-robin list)
    static_mainloop(config, worker_id);

    // Inter-iteration barrier: wait for all tasks to complete
    if (threadIdx.x == 0)
      barrier_wait(config.barriers, config.end_barrier, config.end_barrier_count);
    __syncthreads();

    // Prepare next batch (multi-iteration modes only)
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE) || \
    defined(MODE_ONLINE_NOTOKEN)
    if (worker_id == 0 && threadIdx.x == 0) {
      // Reset barriers for next iteration (safe: everyone past end_barrier)
      for (int i = 0; i < config.num_barriers; i++)
        config.barriers[i] = 0;

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
      while (ld_acquire_sys_u64(config.prepare_done_counter) < iter)
        __nanosleep(BARRIER_SPIN_SLEEP_NS);
    }
    __syncthreads();
    if (!(*config.continue_flag))
      return;
#else
    return;
#endif
  }
}
