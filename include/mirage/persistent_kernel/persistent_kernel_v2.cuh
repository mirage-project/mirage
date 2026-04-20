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
// "persistent_kernel.cuh" before this file — it depends on global_runtime_config,
// prepare_kernel, and prepare_next_batch being defined there. We do NOT
// re-include persistent_kernel.cuh because it lacks include guards.
#include "mirage/persistent_kernel/runtime_v2.cuh"
#include "mirage/persistent_kernel/dispatch_v2.cuh"

#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <cstdio>

namespace mirage {
namespace runtime_v2 {

using ::mirage::runtime::RuntimeConfig;
using ::mirage::runtime::EventDesc;
using ::mirage::runtime::TaskId;

// ── Host-side: build per-SM static task plan ────────────────────────────────
// Algorithm: walk all events in order, round-robin assign each event's task
// range to workers. Matches v1 scheduler semantics closely enough to preserve
// task ordering within events.
//
// Reads all_events (device) by cudaMemcpying to host scratch.
// Allocates v2_per_sm_task_positions / v2_per_sm_task_offsets on device and
// fills config.v2_* fields.
inline void build_v2_plan(RuntimeConfig &config) {
    int const num_workers = config.num_workers;
    int const num_events = config.num_events;

    // Pull all_events to host
    std::vector<EventDesc> h_events(num_events);
    cudaMemcpy(h_events.data(), config.all_events,
               num_events * sizeof(EventDesc),
               cudaMemcpyDeviceToHost);

    // Pull first_tasks (the begin-of-graph seed tasks) to host
    // They're pushed to a specific worker when EVENT_END_OF_TASK_GRAPH fires.
    // For v2, we include task_pos=1 (begin_task_graph) in SM 0's list.

    // Per-SM task position lists (one iteration's worth)
    std::vector<std::vector<size_t>> per_sm(num_workers);

    // Round-robin assign each task-pushing event's [first, last) range.
    // EVENT_LAUNCH_TASKS / _MASSIVE_TASKS / _DEPENDENT_TASKS push tasks;
    // EVENT_END_OF_TASK_GRAPH / EVENT_TERMINATION / EVENT_EMPTY do not.
    size_t next_worker = 0;
    int pushed_events = 0;
    for (int e = 0; e < num_events; e++) {
        EventDesc const &ev = h_events[e];
        if (ev.event_type != ::mirage::runtime::EVENT_LAUNCH_TASKS &&
            ev.event_type != ::mirage::runtime::EVENT_LAUNCH_MASSIVE_TASKS &&
            ev.event_type != ::mirage::runtime::EVENT_LAUNCH_DEPENDENT_TASKS) {
            continue;
        }
        if (ev.first_task_id >= ev.last_task_id) continue;
        pushed_events++;
        for (size_t t = ev.first_task_id; t < ev.last_task_id; t++) {
            per_sm[next_worker % num_workers].push_back(t);
            next_worker++;
        }
    }
    // SM 0 also runs the "begin_task_graph" task (task_pos=1) that v1's
    // scheduler pushes at iter boundary via END_OF_TASK_GRAPH. We prepend
    // it so SM 0 runs it first each iter.
    per_sm[0].insert(per_sm[0].begin(), 1);

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
        std::copy(per_sm[s].begin(), per_sm[s].end(),
                  h_positions.begin() + h_offsets[s]);
    }

    // Allocate on device
    size_t *d_offsets = nullptr, *d_positions = nullptr;
    cudaMalloc(&d_offsets, (num_workers + 1) * sizeof(size_t));
    cudaMalloc(&d_positions, total * sizeof(size_t));
    cudaMemcpy(d_offsets, h_offsets.data(),
               (num_workers + 1) * sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, h_positions.data(),
               total * sizeof(size_t), cudaMemcpyHostToDevice);

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
        if (n > max_per_sm) max_per_sm = n;
    }
    printf("[v2] static plan built: %d workers, %zu total tasks/iter "
           "(avg %zu/SM, max %zu/SM, pushed_events=%d)\n",
           num_workers, total, total / num_workers, max_per_sm, pushed_events);
    // Debug: show SM 0's first few task positions
    printf("[v2] SM 0 first 10 tasks:");
    for (int i = 0; i < 10 && i < (int)(h_offsets[1] - h_offsets[0]); i++) {
        printf(" %zu", h_positions[i]);
    }
    printf("\n");
    // Pull all_tasks[1] to host to show what begin_task_graph's task_type is
    FullTaskDesc t0;
    cudaMemcpy(&t0, config.all_tasks + 0, sizeof(TaskDesc), cudaMemcpyDeviceToHost);
    printf("[v2] all_tasks[0].task_type=%d (expect TASK_TERMINATE=%d)\n",
           (int)t0.task_type, (int)::mirage::runtime::TASK_TERMINATE);
    FullTaskDesc t1;
    cudaMemcpy(&t1, config.all_tasks + 1, sizeof(TaskDesc), cudaMemcpyDeviceToHost);
    printf("[v2] all_tasks[1].task_type=%d\n", (int)t1.task_type);
}

// ── Device kernel: advance per-iter state ──────────────────────────────────
// Runs prepare_next_batch (paged KV bookkeeping + token append) once per iter
// between worker_v2_kernel launches. Single block, single warp is enough —
// prepare_next_batch is serial inner loops, ~tens of μs.
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE) || defined(MODE_ONLINE_NOTOKEN)
__global__ void v2_iter_advance_kernel(RuntimeConfig config,
                                       int end_of_task_graph_event_pos) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
#ifdef MODE_ONLINE_NOTOKEN
    (void)::prepare_next_batch(config, 0);
#else
    (void)::prepare_next_batch(config);
#endif
}
#endif

} // namespace runtime_v2
} // namespace mirage

// ── C entry points (global scope so HARD_CODE can call them unqualified) ───
// Must be called after init_persistent_kernel + build_v2_plan (one-time setup).
// Each launch call runs one decode step.
// Reset iter barrier counters before each launch.
__global__ inline void v2_reset_counters_kernel(
    unsigned long long *sync_counter,
    unsigned long long *go_counter) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *sync_counter = 0;
        *go_counter = 0;
    }
}

extern "C" inline void launch_persistent_kernel_v2(cudaStream_t default_stream) {
    // 1. Reset task-queue state (zeroes event counters) — v1's prepare_kernel.
    int end_of_task_graph_event_pos = global_runtime_config.num_events - 1;
    ::prepare_kernel<<<
        dim3(global_runtime_config.num_workers, 1, 1),
        dim3(128, 1, 1), 0, default_stream>>>(
        global_runtime_config, end_of_task_graph_event_pos);

    // 2. Reset v2 iter-barrier counters.
    v2_reset_counters_kernel<<<1, 1, 0, default_stream>>>(
        global_runtime_config.v2_iter_sync_counter,
        global_runtime_config.v2_iter_go_counter);

    // 3. Single persistent kernel launch — device loops through decode steps
    //    via the per-SM static plan, calls prepare_next_batch on SM 0 at iter
    //    boundaries, and terminates on step >= max_seq_length.
    mirage::runtime_v2::launch_worker_v2(global_runtime_config,
                                         global_runtime_config.num_workers,
                                         default_stream);

    cudaError_t err = cudaStreamSynchronize(default_stream);
    if (err != cudaSuccess) {
        printf("[v2] worker_v2_kernel error: %s\n", cudaGetErrorString(err));
    }
}

// Must be called once, AFTER init_persistent_kernel, BEFORE first launch.
extern "C" inline void init_persistent_kernel_v2() {
    mirage::runtime_v2::build_v2_plan(global_runtime_config);
}
