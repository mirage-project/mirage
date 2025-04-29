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

#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>
#include <vector>
#include <unistd.h>
#include <thread>

typedef unsigned long long int TaskId;
unsigned long long int const TASK_INVALID_ID = 0x7fffffffffffffff;
typedef unsigned long long int EventId;
// Event IDs are 64-bit values encoding both the owner of the event and its
// index EVENT: nvshmem_tag: 16, owner_node: 16, event_idx: 32
unsigned long long int const EVENT_NVSHMEM_TAG = 0x1e00000000000000;
unsigned long long int const EVENT_INVALID_ID = 0x7ffffffffffffffe;
int const MAX_TENSOR_DIMS = 4;
int const MAX_INPUTS_PER_TASK = 4;
int const MAX_OUTPUTS_PER_TASK = 4;

enum TaskType {
  TASK_TERMINATE = 0,
  TASK_END_OF_ITERATION = 1,
  // compute task starts from 100
  TASK_EMBEDDING = 101,
  TASK_RMS_NORM_LINEAR = 102,
  TASK_ATTENTION_1 = 103,
  TASK_ATTENTION_2 = 104,
  TASK_SILU_MUL_LINEAR = 105,
  TASK_ALLREDUCE = 106,
  TASK_REDUCE = 107,
  TASK_MATMUL = 108,
  TASK_NVSHMEM_COPY = 199,
};

struct TensorDesc {
  int num_dims;
  void *base_ptr;
  int data_type;
  int dim[MAX_TENSOR_DIMS];
  int stride[MAX_TENSOR_DIMS];
};

struct EventDesc {
  EventDesc(void)
      : num_triggers(0), first_task_id(TASK_INVALID_ID),
        last_task_id(TASK_INVALID_ID) {}
  EventDesc(int nt, TaskId f, TaskId l)
      : num_triggers(nt), first_task_id(f), last_task_id(l) {}
  int num_triggers;
  TaskId first_task_id, last_task_id;
};

struct TaskDesc {
  TaskDesc(TaskType t)
      : task_type(t), num_inputs(0), num_outputs(0),
        trigger_event(EVENT_INVALID_ID) {}
  TaskType task_type;
  int num_inputs, num_outputs;
  EventId trigger_event;
  TensorDesc inputs[MAX_INPUTS_PER_TASK];
  TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
};

struct RuntimeConfig {
  int num_workers, num_local_schedulers, num_remote_schedulers, num_graphs;
  int num_gpus, my_gpu_id;
  int total_num_tasks, total_num_events;
  unsigned long long int per_worker_queue_len, per_sched_queue_len;
  unsigned long long int *worker_queue_last_ready_task_id;
  unsigned long long int *worker_queue_next_free_task_id;
  unsigned long long int *sched_queue_last_ready_event_id;
  unsigned long long int *sched_queue_next_free_event_id;
  int *all_event_counters;
  TaskDesc *all_tasks;
  EventDesc *all_events;
  TaskId **worker_queues;
  EventId **sched_queues;
  TaskId *first_tasks;
  bool verbose;
};

__global__ void init_kernel(RuntimeConfig config) {
  assert(gridDim.x == 1);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Only a single thread that initializes everything
  if (threadIdx.x == 0) {
    // Send first_task[0] to worker[0]
    config.worker_queue_last_ready_task_id[0] = 1;
    config.worker_queue_next_free_task_id[0] = 1;
    config.worker_queues[0][0] = config.first_tasks[0];
  }
}

__device__ int prepare_next_batch(RuntimeConfig config) {
  return 0;
}

__device__ void terminate_workers_and_schedulers(RuntimeConfig config) {
  // Send event 0 to all workers
  // Task ID 0 is the termination task
  for (int i = 0; i < config.num_workers; i++) {
    size_t last_task_id =
        atomicAdd(&config.worker_queue_next_free_task_id[i], 1);
    config.worker_queues[i][last_task_id % config.per_worker_queue_len] = 0;
    // Add threadfence to make sure worker_queue updates are visible to worker
    // CTAs before incrementing its last_ready_task_id
    __threadfence();
    size_t old = config.worker_queue_last_ready_task_id[i];
    do {
      old = atomicCAS(&config.worker_queue_last_ready_task_id[i],
                      last_task_id,
                      last_task_id + 1);
    } while (old != last_task_id);
  }
  // Event ID 0 is the termination event
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  for (int i = 0; i < num_schedulers; i++) {
    size_t last_event_id =
        atomicAdd(&config.sched_queue_next_free_event_id[i], 1);
    config.sched_queues[i][last_event_id % config.per_sched_queue_len] = 0;
    // Add threadfence to make sure sched_queue updates are visible to scheduler
    // CTAs before incrementing its last_ready_event_id
    __threadfence();
    size_t old = config.sched_queue_last_ready_event_id[i];
    do {
      old = atomicCAS(&config.sched_queue_last_ready_event_id[i],
                      last_event_id,
                      last_event_id + 1);
    } while (old != last_event_id);
  }
}

__device__ __forceinline__ bool is_termination_event(size_t event_loc,
                                                     EventDesc e) {
  return (event_loc == 0);
}

__device__ __forceinline__ bool is_nvshmem_event(EventId event_id) {
  return (event_id & EVENT_NVSHMEM_TAG) > 0;
}

__device__ __forceinline__ size_t get_event_gpu_id(EventId event_id) {
  return ((event_id >> 32) & 0xffff);
}

__device__ __forceinline__ size_t get_event_position_index(EventId event_id) {
  return (event_id & 0xffffffff);
}

__global__ void persistent_kernel(RuntimeConfig config) {
  __shared__ TaskId cur_task_loc;
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  assert(num_schedulers % 4 == 0);
  assert(gridDim.x == config.num_workers + num_schedulers / 4);
  if (blockIdx.x < config.num_workers) {
    int worker_id = blockIdx.x;
    size_t cur_task_id = 0, last_task_id = 0;
    TaskId *task_queue = config.worker_queues[worker_id];
    while (true) {
      // fetch next task from task queue
      if (threadIdx.x == 0) {
        while (cur_task_id == last_task_id) {
          __threadfence();
          last_task_id = config.worker_queue_last_ready_task_id[worker_id];
        }
        assert(cur_task_id + config.per_worker_queue_len > last_task_id);
        cur_task_loc = task_queue[cur_task_id % config.per_worker_queue_len];
        if (config.verbose) {
          printf(
              "[%d][FTCH] worker_id(%d) cur_task_pos(%llu) last_task_pos(%llu) "
              "task_id(%llu) task_type(%d) event_id(%llx) \n",
              config.my_gpu_id,
              worker_id,
              cur_task_id,
              last_task_id,
              cur_task_loc,
              config.all_tasks[cur_task_loc].task_type,
              config.all_tasks[cur_task_loc].trigger_event);
        }
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
          if (threadIdx.x == 0) {
            // printf("[EXEC] worker_id(%d) task_type(RMS)\n", worker_id);
          }
          break;
        }
        case TASK_EMBEDDING: {
          if (threadIdx.x == 0) {
            // printf("[EXEC] worker_id(%d) task_type(EMB)\n", worker_id);
          }
          break;
        }
        case TASK_ATTENTION_1: {
          if (threadIdx.x == 0) {
            // printf("worker_id(%d) task_type(Attn1)\n", worker_id);
          }
          break;
        }
        case TASK_ATTENTION_2: {
          if (threadIdx.x == 0) {
            // printf("worker_id(%d) task_type(Attn2)\n", worker_id);
          }
          break;
        }
        case TASK_SILU_MUL_LINEAR: {
          if (threadIdx.x == 0) {
            // printf("worker_id(%d) task_type(SiluMulLinear)\n", worker_id);
          }
          break;
        }
        case TASK_NVSHMEM_COPY: {
          if (threadIdx.x == 0) {
            // printf("worker_id(%d) task_type(AllReduce)\n", worker_id);
          }
          break;
        }
        case TASK_REDUCE: {
          if (threadIdx.x == 0) {
            // printf("worker_id(%d) task_type(AllReduce)\n", worker_id);
          }
          break;
        }
        case TASK_MATMUL: {
          if (threadIdx.x == 0) {
            // printf("worker_id(%d) task_type(AllReduce)\n", worker_id);
          }
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
        size_t event_index = get_event_position_index(event_id);
        if (!is_nvshmem_event(event_id)) {
          size_t gpu_id = get_event_gpu_id(event_id);
          assert(gpu_id == config.my_gpu_id);
          // Case 1: Trigger a local non-nvshmem event
          int count = atomicSub(&config.all_event_counters[event_index], 1);
          if (config.verbose) {
            printf("[%d][DONE] worker_id(%d) task_id(%llu) event_id(%llx) "
                   "event_type(local) count(%d)\n",
                   config.my_gpu_id,
                   worker_id,
                   cur_task_loc,
                   event_id,
                   count);
          }
          if (count == 1) {
            // The event has been triggered enough times
            // Refresh the event counter
            EventDesc event_desc = config.all_events[event_index];
            atomicAdd(&config.all_event_counters[event_index],
                      event_desc.num_triggers);
            // Add the event to the schedule_queue
            int sched_id = event_index % config.num_local_schedulers;
            size_t last_event_id =
                atomicAdd(&config.sched_queue_next_free_event_id[sched_id], 1);
            config.sched_queues[sched_id]
                               [last_event_id % config.per_sched_queue_len] =
                event_index;
            // Make sure that the updated event_index is visible to the
            // scheduler CTA before updating its last_ready_event_id
            __threadfence();
            size_t old = config.sched_queue_last_ready_event_id[sched_id];
            do {
              old = atomicCAS(&config.sched_queue_last_ready_event_id[sched_id],
                              last_event_id,
                              last_event_id + 1);
            } while (old != last_event_id);
          }
        } else {
          // Case 2: trigger a nvshmem event
          size_t gpu_id = get_event_gpu_id(event_id);
          assert(gpu_id < config.num_gpus);
          assert(gpu_id != config.my_gpu_id);
          int count = nvshmem_int_atomic_fetch_add(
              &config.all_event_counters[event_index], -1, gpu_id);
          if (config.verbose) {
            printf("[%d][DONE] worker_id(%d) task_id(%llu) event_id(%llx) "
                   "event_type(remote) count(%d)\n",
                   config.my_gpu_id,
                   worker_id,
                   cur_task_loc,
                   event_id,
                   count);
          }
          if (count == 1) {
            // The event has been triggered enough times
            // Refresh the event counter
            // Note that we load a local event since all task graphs
            // are replicated across gpus and therefore they have the same
            // event metadata (i.e., config.all_events[i] should be the same
            // across GPUs)
            EventDesc event_desc = config.all_events[event_index];
            nvshmem_int_atomic_add(&config.all_event_counters[event_index],
                                   event_desc.num_triggers,
                                   gpu_id);
            // Add the event to the schedule queue
            int sched_id = config.num_local_schedulers +
                           event_index % config.num_remote_schedulers;
            size_t last_event_id = nvshmem_ulonglong_atomic_fetch_add(
                &config.sched_queue_next_free_event_id[sched_id], 1, gpu_id);
            nvshmem_ulonglong_p(
                &config.sched_queues[sched_id][last_event_id %
                                               config.per_sched_queue_len],
                event_index,
                gpu_id);
            // use nvshmem_quiet to force completion of remote transfer
            // before updating the last_ready_event_id
            nvshmem_fence();
            nvshmem_ulonglong_atomic_add(
                &config.sched_queue_last_ready_event_id[sched_id], 1, gpu_id);
          }
        }
      }
      cur_task_id += 1;
    }
  } else {
    // CANNOT use syncthreads on the scheduler side
    int warp_id = threadIdx.x / 32;
    int warp_thread_id = threadIdx.x % 32;
    // assert that we have at least four warps per thread block
    assert(blockDim.x >= 128);
    if (warp_id < 4 && warp_thread_id == 0) {
      int sched_id = (blockIdx.x - config.num_workers) * 4 + warp_id;
      EventId *sched_queue = config.sched_queues[sched_id];
      size_t cur_event_id = 0, last_event_id = 0;
      int next_worker = sched_id * (config.num_workers / num_schedulers);
      while (true) {
        while (cur_event_id == last_event_id) {
          __threadfence();
          last_event_id = config.sched_queue_last_ready_event_id[sched_id];
        }
        // Make sure the schedule queue is not overflow
        assert(cur_event_id + config.per_sched_queue_len > last_event_id);
        // Launch new tasks
        EventId event_id =
            sched_queue[cur_event_id % config.per_sched_queue_len];
        EventDesc e = config.all_events[event_id];
        if (is_termination_event(event_id, e)) {
          // return;
        }
        for (TaskId i = e.first_task_id; i < e.last_task_id; i++) {
          size_t last_task_id = atomicAdd(
              &(config.worker_queue_next_free_task_id[next_worker]), 1);
          config.worker_queues[next_worker]
                              [last_task_id % config.per_worker_queue_len] = i;
          // Make sure writes to worker_queues is visible to worker CTAs before
          // we increase its last_ready_task_id
          __threadfence();
          size_t old = config.worker_queue_last_ready_task_id[next_worker];
          do {
            __threadfence();
            old =
                atomicCAS(&config.worker_queue_last_ready_task_id[next_worker],
                          last_task_id,
                          last_task_id + 1);
          } while (old != last_task_id);
          if (config.verbose) {
            printf("[%d][SCHD] schd_id(%d) task_id(%llu) worker_id(%d) "
                   "worker_last_ready_pos(%llu)\n",
                   config.my_gpu_id,
                   sched_id,
                   i,
                   next_worker,
                   last_task_id + 1);
          }
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

template <typename DT>
DT *gpu_malloc(size_t size) {
  void *dst_ptr = nvshmem_malloc(size);
  return static_cast<DT *>(dst_ptr);
}

void gpu_free(void *ptr) {
  nvshmem_free(ptr);
}

// The following function will be generated by the transpiler
static void _init_persistent_kernel(std::vector<TaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                    std::vector<TaskId> &first_tasks,
				    std::vector<void const *> const &torch_tensors,
                                    int num_gpus,
                                    int my_gpu_id);

extern "C" void init_persistent_kernel(std::vector<void const *> torch_tensors,
				       int my_rank,
                                       int num_workers,
                                       int num_local_schedulers,
                                       int num_remote_schedulers) {
  static RuntimeConfig global_runtime_config;
  global_runtime_config.num_workers = num_workers;
  global_runtime_config.num_local_schedulers = num_local_schedulers;
  global_runtime_config.num_remote_schedulers = num_remote_schedulers;
  int num_schedulers = num_local_schedulers + num_remote_schedulers;

  // Initialize nvshmem
  cudaSetDevice(my_rank);
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  nvshmem_barrier_all();
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  printf("mype(%d) npes(%d) mype_node(%d)\n", mype, npes, mype_node);
  printf("process_id(%zu) thread_id(%zu)\n", getpid(), std::this_thread::get_id());
  global_runtime_config.per_worker_queue_len = 1024;
  global_runtime_config.per_sched_queue_len = 1024;
  global_runtime_config.num_gpus = npes;
  global_runtime_config.my_gpu_id = mype;
  global_runtime_config.num_graphs = 1;
  global_runtime_config.verbose = true;

  std::vector<TaskDesc> all_tasks;
  std::vector<EventDesc> all_events;
  std::vector<TaskId> first_tasks;
  _init_persistent_kernel(all_tasks, all_events, first_tasks, torch_tensors, npes, mype);

  // Initialize worker queue last task id
  global_runtime_config.worker_queue_last_ready_task_id =
      gpu_malloc<unsigned long long int>(
          num_workers * sizeof(unsigned long long int));
  global_runtime_config.worker_queue_next_free_task_id =
      gpu_malloc<unsigned long long int>(
          num_workers * sizeof(unsigned long long int));
  std::vector<unsigned long long int> host_worker_queue_last_task_id;
  for (int i = 0; i < num_workers; i++) {
    host_worker_queue_last_task_id.push_back(0);
  }
  cudaMemcpy(global_runtime_config.worker_queue_last_ready_task_id,
             host_worker_queue_last_task_id.data(),
             num_workers * sizeof(unsigned long long int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(global_runtime_config.worker_queue_next_free_task_id,
             host_worker_queue_last_task_id.data(),
             num_workers * sizeof(unsigned long long int),
             cudaMemcpyHostToDevice);
  // Initialize scheduler queue last event id
  global_runtime_config.sched_queue_last_ready_event_id =
      gpu_malloc<unsigned long long int>(
          num_schedulers * sizeof(unsigned long long int));
  global_runtime_config.sched_queue_next_free_event_id =
      gpu_malloc<unsigned long long int>(
          num_schedulers * sizeof(unsigned long long int));

  std::vector<unsigned long long int> host_sched_queue_last_event_id;
  for (int i = 0; i < num_schedulers; i++) {
    host_sched_queue_last_event_id.push_back(0);
  }
  cudaMemcpy(global_runtime_config.sched_queue_last_ready_event_id,
             host_sched_queue_last_event_id.data(),
             num_schedulers * sizeof(unsigned long long int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(global_runtime_config.sched_queue_next_free_event_id,
             host_sched_queue_last_event_id.data(),
             num_schedulers * sizeof(unsigned long long int),
             cudaMemcpyHostToDevice);
  // Initialize all event counters
  global_runtime_config.all_event_counters =
      gpu_malloc<int>(all_events.size() * sizeof(int));
  std::vector<int> host_all_event_counters;
  for (size_t i = 0; i < all_events.size(); i++) {
    host_all_event_counters.push_back(all_events.at(i).num_triggers);
  }
  cudaMemcpy(global_runtime_config.all_event_counters,
             host_all_event_counters.data(),
             all_events.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  // Initialize all tasks
  global_runtime_config.all_tasks =
      gpu_malloc<TaskDesc>(all_tasks.size() * sizeof(TaskDesc));
  cudaMemcpy(global_runtime_config.all_tasks,
             all_tasks.data(),
             all_tasks.size() * sizeof(TaskDesc),
             cudaMemcpyHostToDevice);
  // Initialize all events
  global_runtime_config.all_events =
      gpu_malloc<EventDesc>(all_events.size() * sizeof(EventDesc));
  cudaMemcpy(global_runtime_config.all_events,
             all_events.data(),
             all_events.size() * sizeof(EventDesc),
             cudaMemcpyHostToDevice);
  // Initialize worker queues
  {
    std::vector<TaskId *> host_worker_queues;
    for (int i = 0; i < num_workers; i++) {
      TaskId *worker_queue =
          gpu_malloc<TaskId>(global_runtime_config.per_worker_queue_len * sizeof(TaskId));
      host_worker_queues.push_back(worker_queue);
    }
    global_runtime_config.worker_queues =
        gpu_malloc<TaskId *>(num_workers * sizeof(TaskId *));
    cudaMemcpy(global_runtime_config.worker_queues,
               host_worker_queues.data(),
               num_workers * sizeof(TaskId *),
               cudaMemcpyHostToDevice);
  }
  // Initialize scheduler queues
  {
    std::vector<EventId *> host_sched_queues;
    for (int i = 0; i < num_schedulers; i++) {
      EventId *sched_queue =
          gpu_malloc<EventId>(global_runtime_config.per_sched_queue_len * sizeof(EventId));
      host_sched_queues.push_back(sched_queue);
    }
    global_runtime_config.sched_queues =
        gpu_malloc<EventId *>(num_schedulers * sizeof(EventId *));
    cudaMemcpy(global_runtime_config.sched_queues,
               host_sched_queues.data(),
               num_schedulers * sizeof(EventId *),
               cudaMemcpyHostToDevice);
  }
  // Initialize first tasks
  {
    global_runtime_config.first_tasks =
        gpu_malloc<TaskId>(first_tasks.size() * sizeof(TaskId));
    cudaMemcpy(global_runtime_config.first_tasks,
               first_tasks.data(),
               first_tasks.size() * sizeof(TaskId),
               cudaMemcpyHostToDevice);
  }

  // launch init kernel
  init_kernel<<<dim3(1, 1, 1), dim3(128, 1, 1)>>>(global_runtime_config);
  cudaDeviceSynchronize();
  // Add a global barrier for all init_kernel to complete
  nvshmem_barrier_all();
}

// Entry point for C/C++
extern "C" void launch_persistent_kernel(cudaStream_t stream) {
  static RuntimeConfig global_runtime_config;
  void *args[] = {&global_runtime_config};
  // Launcher persistent kernel
  cudaFuncSetAttribute(persistent_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 22656);
  nvshmemx_collective_launch((void const *)persistent_kernel,
                             dim3(108, 1, 1),
                             dim3(128, 1, 1),
                             args,
                             1024 /*sharedmem*/,
                             stream /*stream*/);
}

extern "C" void finalize_persistent_kernel() {
  static RuntimeConfig global_runtime_config;
  gpu_free(global_runtime_config.worker_queue_last_ready_task_id);
  gpu_free(global_runtime_config.worker_queue_next_free_task_id);
  gpu_free(global_runtime_config.sched_queue_last_ready_event_id);
  gpu_free(global_runtime_config.sched_queue_next_free_event_id);
  gpu_free(global_runtime_config.all_event_counters);
  gpu_free(global_runtime_config.all_tasks);
  gpu_free(global_runtime_config.all_events);
  int num_workers = global_runtime_config.num_workers;
  std::vector<TaskId *> host_worker_queues(num_workers);
  cudaMemcpy(host_worker_queues.data(),
             global_runtime_config.worker_queues,
             num_workers * sizeof(TaskId *),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_workers; i++) {
    gpu_free(host_worker_queues[i]);
  }
  gpu_free(global_runtime_config.worker_queues);
  int num_schedulers = global_runtime_config.num_local_schedulers
                     + global_runtime_config.num_remote_schedulers;
  std::vector<EventId *> host_sched_queues(num_schedulers);
  cudaMemcpy(host_sched_queues.data(),
             global_runtime_config.sched_queues,
             num_schedulers * sizeof(EventId *),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_schedulers; i++) {
    gpu_free(host_sched_queues[i]);
  }
  gpu_free(global_runtime_config.sched_queues);
  gpu_free(global_runtime_config.first_tasks);
  nvshmem_barrier_all();
  nvshmem_finalize();
}
