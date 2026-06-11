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

// v2 persistent kernel driver.
//
// Replaces scheduler_kernel + worker_kernel with a scheduler-less path:
//   - Host pre-computes per-SM task list via round-robin
//   - worker_v2_kernel walks its SM's list directly (runtime_v2.cuh)
//   - Host loops iterations, calling prepare_next_batch between
//
// Leaves persistent_kernel.cuh (v1) untouched. Depends on persistent_kernel.cuh
// for init_persistent_kernel/global_runtime_config/init_kernel/prepare_kernel.

#pragma once

// NOTE: this file assumes the including translation unit has ALREADY included
// "persistent_kernel.cuh" before this file — it depends on
// global_runtime_config, prepare_kernel, and prepare_next_batch being defined
// there. We do NOT re-include persistent_kernel.cuh because it lacks include
// guards.
#include "mirage/persistent_kernel/runtime_v2.cuh"
#include "mirage/persistent_kernel/tasks/blackwell_v2/task_header.cuh"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <thread>
#include <vector>

namespace mirage {
namespace runtime_v2 {

using ::mirage::runtime::EventDesc;
using ::mirage::runtime::RuntimeConfig;
using ::mirage::runtime::TaskId;

// Per-SM task queues, in execution order, produced by the Python scheduler
// (v2_task_schedule.py) and parsed out of task_graph.json by the generated
// construct_task_graph. This is the single source of the v2 task ordering —
// the SMEM page planner consumes the same queues, so the kernel's execution
// order and the page plan can never drift. Populated once at init, before
// build_v2_plan runs.
inline std::vector<std::vector<size_t>> g_v2_worker_task_queues;

// ── Host-side: build per-SM static task plan ────────────────────────────────
// Consumes g_v2_worker_task_queues (the Python scheduler's output, already in
// execution order and including task 1 on worker 0). The SMEM page planner uses
// the same queues, so the kernel and the page plan share one ordering. Flattens
// the queues into v2_per_sm_task_positions / v2_per_sm_task_offsets on device.
inline void build_v2_plan(RuntimeConfig &config) {
  int const num_workers = config.num_workers;

  std::vector<std::vector<size_t>> const &per_sm = g_v2_worker_task_queues;
  assert((int)per_sm.size() == num_workers &&
         "g_v2_worker_task_queues must hold num_workers queues — "
         "construct_task_graph must parse v2_worker_task_queues from "
         "task_graph.json before build_v2_plan runs");

  // Flatten into offsets + positions
  std::vector<size_t> h_offsets(num_workers + 1);
  size_t total = 0;
  for (int s = 0; s < num_workers; s++) {
    h_offsets[s] = total;
    total += per_sm[s].size();
  }
  h_offsets[num_workers] = total;

  std::vector<size_t> h_positions(total);
  for (int s = 0; s < num_workers; s++) {
    std::copy(
        per_sm[s].begin(), per_sm[s].end(), h_positions.begin() + h_offsets[s]);
  }

  // Allocate on device
  size_t *d_offsets = nullptr, *d_positions = nullptr;
  cudaMalloc(&d_offsets, (num_workers + 1) * sizeof(size_t));
  cudaMalloc(&d_positions, total * sizeof(size_t));
  cudaMemcpy(d_offsets,
             h_offsets.data(),
             (num_workers + 1) * sizeof(size_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_positions,
             h_positions.data(),
             total * sizeof(size_t),
             cudaMemcpyHostToDevice);

  config.v2_per_sm_task_offsets = d_offsets;
  config.v2_per_sm_task_positions = d_positions;

  // Allocate device-side iter barrier counters (zeroed once at init)
  unsigned long long *d_sync = nullptr, *d_go = nullptr;
  cudaMalloc(&d_sync, sizeof(unsigned long long));
  cudaMalloc(&d_go, sizeof(unsigned long long));
  cudaMemset(d_sync, 0, sizeof(unsigned long long));
  cudaMemset(d_go, 0, sizeof(unsigned long long));
  config.v2_iter_sync_counter = d_sync;
  config.v2_iter_go_counter = d_go;
  config.v2_max_iters = config.max_seq_length;
  config.v2_enabled = true;

  size_t max_per_sm = 0;
  for (int s = 0; s < num_workers; s++) {
    size_t n = h_offsets[s + 1] - h_offsets[s];
    if (n > max_per_sm) {
      max_per_sm = n;
    }
  }
  printf("[v2] static plan built: %d workers, %zu total tasks/iter "
         "(avg %zu/SM, max %zu/SM)\n",
         num_workers,
         total,
         total / num_workers,
         max_per_sm);
}

} // namespace runtime_v2
} // namespace mirage

// ── C entry points (global scope so HARD_CODE can call them unqualified) ───
// Must be called after init_persistent_kernel + build_v2_plan (one-time setup).
// Each launch call runs one decode step.
// Reset iter barrier counters before each launch.
__global__ inline void
    v2_reset_counters_kernel(unsigned long long *sync_counter,
                             unsigned long long *go_counter) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *sync_counter = 0;
    *go_counter = 0;
  }
}

extern "C" inline void
    launch_persistent_kernel_v2(cudaStream_t default_stream) {
  // 1. Reset task-queue state (zeroes event counters) — v1's prepare_kernel.
  int end_of_task_graph_event_pos = global_runtime_config.num_events - 1;
  ::prepare_kernel<<<dim3(global_runtime_config.num_workers, 1, 1),
                     dim3(128, 1, 1),
                     0,
                     default_stream>>>(global_runtime_config,
                                       end_of_task_graph_event_pos);

  // 2. Reset v2 iter-barrier counters.
  v2_reset_counters_kernel<<<1, 1, 0, default_stream>>>(
      global_runtime_config.v2_iter_sync_counter,
      global_runtime_config.v2_iter_go_counter);

  // 3. Single persistent kernel launch — device loops through decode steps
  //    via the per-SM static plan, calls prepare_next_batch on SM 0 at iter
  //    boundaries, and terminates on step >= max_seq_length.
  mirage::runtime_v2::launch_worker_v2(
      global_runtime_config, global_runtime_config.num_workers, default_stream);

  cudaError_t err = cudaStreamSynchronize(default_stream);
  if (err != cudaSuccess) {
    printf("[v2] worker_v2_kernel error: %s\n", cudaGetErrorString(err));
  }
}

// Must be called once, AFTER init_persistent_kernel, BEFORE first launch.
extern "C" inline void init_persistent_kernel_v2() {
  mirage::runtime_v2::build_v2_plan(global_runtime_config);
}
