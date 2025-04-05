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

#include "mirage/runtime/runtime.h"

namespace mirage {
namespace runtime {

__global__ void init_kernel(RuntimeConfig config) {
  assert(gridDim.x == 1);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Only a single thread that initializes everything
  if (threadIdx.x == 0) {
    for (int i = 0; i < config.num_workers; i++) {
      config.worker_queue_last_task_id[i] = 0;
    }
    for (int i = 0; i < config.num_schedulers; i++) {
      config.sched_queue_last_event_id[i] = 0;
    }
  }
}

__device__ int prepare_next_batch(RuntimeConfig config) {
  int batch_size = 0;
}

__device__ void terminate_workers_and_schedulers(RuntimeConfig config) {
  // Send event 0 to all workers
  // Task ID 0 is the termination task
  for (int i = 0; i < config.num_workers; i++) {
    size_t last_task_id = atomicAdd(&config.worker_queue_last_task_id[i], 1);
    config.worker_queues[i][last_task_id % config.per_worker_queue_len] = 0;
  }
  // Event ID 0 is the termination event
  for (int i = 0; i < config.num_schedulers; i++) {
    size_t last_event_id = atomicAdd(&config.sched_queue_last_event_id[i], 1);
    config.sched_queues[i][last_event_id % config.per_sched_queue_len] = 0;
  }
}

__device__ __forceinline__ bool is_termination_event(size_t event_loc,
                                                     EventDesc e) {
  return (event_loc == 0);
}

__global__ void persistent_kernel(RuntimeConfig config) {
  __shared__ TaskId cur_task_loc;
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  assert(config.num_schedulers % 4 == 0);
  assert(gridDim.x == config.num_workers + config.num_schedulers / 4);
  if (blockIdx.x < config.num_workers) {
    int worker_id = blockIdx.x;
    size_t cur_task_id = 0;
    TaskId *task_queue = config.worker_queues[worker_id];
    while (true) {
      // fetch next task from task queue
      if (threadIdx.x == 0) {
        size_t last_task_id = cur_task_id;
        while (cur_task_id == last_task_id) {
          last_task_id = config.worker_queue_last_task_id[worker_id];
        }
        assert(cur_task_id + config.per_worker_queue_len > last_task_id);
        cur_task_loc = task_queue[cur_task_id % config.per_worker_queue_len];
      }
      __syncthreads();
      TaskDesc task_desc = config.all_tasks[cur_task_loc];
      // Successfully fetched a new task
      switch (task_desc.task_type) {
        case TASK_TERMINATE: {
          return;
          break;
        }
        case TASK_RMS_NORM_LINEAR: {
          break;
        }
        default: {
          assert(false && "Unimplemented task");
        }
      }
      __syncthreads();
      // Trigger event
      if (threadIdx.x == 0) {
        EventId event_id = task_desc.trigger_event;
        int count = atomicSub(&config.all_event_counters[event_id], 1);
        if (count == 1) {
          // The event has been triggered enough times
          // Refresh the event counter
          EventDesc event_desc = config.all_events[event_id];
          atomicAdd(&config.all_event_counters[event_id],
                    event_desc.num_triggers);
          // Add the event to the schedule_queue
          int sched_id = event_id % config.num_schedulers;
          size_t last_event_id =
              atomicAdd(&config.sched_queue_last_event_id[sched_id], 1);
          config.sched_queues[sched_id][last_event_id %
                                        config.per_sched_queue_len] = event_id;
        }
      }
      cur_task_id += 1;
    }
  } else {
    // CANNOT use syncthreads on the scheduler side
    int warp_id = threadIdx.x / 32;
    int thread_id = threadIdx.x % 32;
    // assert that we have at least four warps per thread block
    assert(blockDim.x >= 128);
    if (warp_id < 4 && thread_id == 0) {
      int sched_id = (blockIdx.x - config.num_workers) * 4 + warp_id;
      EventId *sched_queue = config.sched_queues[sched_id];
      size_t cur_event_id = 0, last_event_id = 0;
      int next_worker = sched_id * (config.num_workers / config.num_schedulers);
      while (true) {
        while (cur_event_id == last_event_id) {
          last_event_id = config.sched_queue_last_event_id[sched_id];
        }
        // Make sure the schedule queue is not overflow
        assert(cur_event_id + config.per_sched_queue_len > last_event_id);
        // Launch new tasks
        EventId event_id =
            sched_queue[cur_event_id % config.per_sched_queue_len];
        EventDesc e = config.all_events[event_id];
        if (is_termination_event(event_id, e)) {
          return;
        }
        for (TaskId i = e.first_task_id; i < e.last_task_id; i++) {
          size_t last_task_id =
              atomicAdd(&(config.worker_queue_last_task_id[next_worker]), 1);
          config.worker_queues[next_worker]
                              [last_task_id % config.per_worker_queue_len] = i;
          next_worker = (next_worker + 1) % config.num_workers;
        }
        if (e.first_task_id == e.last_task_id) {
          // Terminate all schedulers & workers
          // config.all_tasks[0] and config.all_events[0]
          // are reserved to
          terminate_workers_and_schedulers(config);
          return;
        }
        cur_event_id += 1;
      }
    }
  }
}

void Runtime::launch_persistent_kernel(int num_workers, int num_schedulers) {
  RuntimeConfig config;
  config.num_workers = num_workers;
  config.num_schedulers = num_schedulers;
  config.num_graphs = num_graphs;
  config.total_num_tasks = all_tasks.size();
  config.total_num_events = all_events.size();
  config.per_worker_queue_len = 1024;
  config.per_sched_queue_len = 1024;
  cudaMalloc(&config.worker_queue_last_task_id,
             config.num_workers * sizeof(unsigned long long int));
  cudaMalloc(&config.sched_queue_last_event_id,
             config.num_schedulers * sizeof(unsigned long long int));
  cudaMalloc(&config.all_event_counters, config.total_num_events * sizeof(int));
  // Initialize all tasks
  cudaMalloc(&config.all_tasks, config.total_num_tasks * sizeof(TaskDesc));
  cudaMemcpy(config.all_tasks,
             all_tasks.data(),
             config.total_num_tasks * sizeof(TaskDesc),
             cudaMemcpyHostToDevice);
  // Initialize all events
  cudaMalloc(&config.all_events, config.total_num_events * sizeof(EventDesc));
  cudaMemcpy(config.all_events,
             all_events.data(),
             config.total_num_events * sizeof(EventDesc),
             cudaMemcpyHostToDevice);

  // Launch init kernel
  init_kernel<<<dim3(1, 1, 1), dim3(128, 1, 1)>>>(config);

  // Launcher persistent kernel
  persistent_kernel<<<dim3(108, 1, 1), dim3(128, 1, 1)>>>(config);
}

}; // namespace runtime
}; // namespace mirage
