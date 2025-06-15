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

#include "../persistent_kernel/tasks/kernel.h"
#include "profiler.h"
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <thread>
#include <unistd.h>
#include <vector>

typedef unsigned long long int TaskId;
unsigned long long int const TASK_INVALID_ID = 0x7fffffffffffffff;
// Task IDs are 64-bit values encoding both the current iteration of the task
// and its index TASK: iteration id: 32, task index: 32
typedef unsigned long long int EventId;
// Event IDs are 64-bit values encoding both the owner of the event and its
// index EVENT: nvshmem_tag: 16, owner_node: 16, event_idx: 32
unsigned long long int const EVENT_NVSHMEM_TAG = 0x1e00000000000000;
unsigned long long int const EVENT_INVALID_ID = 0x7ffffffffffffffe;
typedef unsigned long long int EventCounter;
int const MAX_TENSOR_DIMS = 4;
int const MAX_INPUTS_PER_TASK = 7;
int const MAX_OUTPUTS_PER_TASK = 1;
int const MAX_NUM_WORKERS = 128;

using bfloat16 = type::bfloat16_t;

enum TaskType {
  TASK_TERMINATE = 0,
  TASK_BEGIN_TASK_GRAPH = 10,
  // compute task starts from 100
  TASK_EMBEDDING = 101,
  TASK_RMS_NORM_LINEAR = 102,
  TASK_ATTENTION_1 = 103,
  TASK_ATTENTION_2 = 104,
  TASK_SILU_MUL_LINEAR = 105,
  TASK_ALLREDUCE = 106,
  TASK_REDUCE = 107,
  TASK_MATMUL = 108,
  TASK_ARGMAX = 109,
  TASK_ARGMAX_PARTIAL = 110,
  TASK_ARGMAX_REDUCE = 111,
  TASK_NVSHMEM_COPY = 199,
  TASK_SCHD_TASKS = 200,
  TASK_SCHD_EVENTS = 201,
  TASK_GET_EVENT = 202,
  TASK_GET_NEXT_TASK = 203,
};

enum EventType {
  EVENT_EMPTY = 900,
  EVENT_LAUNCH_TASKS = 901,
  EVENT_LAUNCH_MASSIVE_TASKS = 902,
  EVENT_LAUNCH_DEPENDENT_TASKS = 903,
  EVENT_END_OF_TASK_GRAPH = 910,
  EVENT_TERMINATION = 911,
  EVENT_INVALID = 999,
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
      : event_type(EVENT_INVALID), num_triggers(0),
        first_task_id(TASK_INVALID_ID), last_task_id(TASK_INVALID_ID) {}
  EventDesc(EventType type, int nt, TaskId f, TaskId l)
      : event_type(type), num_triggers(nt), first_task_id(f), last_task_id(l) {}
  EventType event_type;
  int num_triggers;
  TaskId first_task_id, last_task_id;
};

struct TaskDesc {
  TaskDesc(TaskType t)
      : task_type(t), num_inputs(0), num_outputs(0),
        trigger_event(EVENT_INVALID_ID), dependent_event(EVENT_INVALID_ID) {}
  TaskDesc() {}
  TaskType task_type;
  int num_inputs, num_outputs;
  EventId trigger_event;
  EventId dependent_event;
  TensorDesc inputs[MAX_INPUTS_PER_TASK];
  TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
};

struct RuntimeConfig {
  int num_workers, num_local_schedulers, num_remote_schedulers, num_graphs;
  int num_gpus, my_gpu_id;
  unsigned long long int per_worker_queue_len, per_sched_queue_len;
  unsigned long long int *worker_queue_last_ready_task_id;
  unsigned long long int *sched_queue_last_ready_event_id;
  unsigned long long int *sched_queue_next_free_event_id;
  EventCounter *all_event_counters;
  int *all_event_num_triggers;
  TaskDesc *all_tasks;
  EventDesc *all_events;
  TaskId **worker_queues;
  EventId **sched_queues;
  TaskId *first_tasks;
  int *step; // Metadata for LLM serving
  void *profiler_buffer;
  bool verbose;
  bool profiling;
};

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

__device__ __forceinline__ size_t get_task_iteration_num(TaskId task_id) {
  return (task_id >> 32);
}

__device__ __forceinline__ size_t get_task_position_index(TaskId task_id) {
  return (task_id & 0xffffffff);
}

__device__ __forceinline__ TaskId compute_task_id(size_t iteration_num,
                                                  size_t position_index) {
  return ((iteration_num << 32) | position_index);
}

__global__ void init_kernel(RuntimeConfig config) {
  assert(gridDim.x == 1);
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Only a single thread that initializes everything
  if (threadIdx.x == 0) {
    // Send task 1 to worker[0]
    config.worker_queue_last_ready_task_id[0] = 1;
    config.worker_queues[0][0] =
        compute_task_id(1 /*iteration_num*/, 1 /*task_begin_task_graph*/);
  }
}

__device__ __forceinline__ bool prepare_next_batch(RuntimeConfig config) {
  int step = config.step[0];
  // printf("step = %d\n", step);
  config.step[0] = step + 1;
  // return step + 1 <= 50;
  return false;
}

__device__ __forceinline__ int get_rand_sched_id(size_t event_index,
                                                 int worker_id,
                                                 int num_workers,
                                                 int num_schedulers) {
  // const size_t seed = 0xac4c1b51;
  // size_t x = event_index * seed;
  // x ^= x >> 17;
  // x *= worker_id;
  //  x *= 0xed5ad4bb;
  // x ^= x >> 11;
  size_t x = worker_id;
  return x / ((num_workers + num_schedulers - 1) / num_schedulers);
}

__device__ __forceinline__ int custom_atomic_add_s32(int *addr, int val) {
  int old_val;
  asm volatile("atom.add.release.gpu.s32 %0,[%1],%2;"
               : "=r"(old_val)
               : "l"(addr), "r"(val)
               : "memory");
  return old_val;
}

__device__ __forceinline__ unsigned long long int
    custom_atomic_add_u64(unsigned long long int *addr,
                          unsigned long long int val) {
  unsigned long long int old_val;
  asm volatile("atom.add.release.gpu.u64 %0,[%1],%2;"
               : "=l"(old_val)
               : "l"(addr), "l"(val)
               : "memory");
  return old_val;
}

__device__ __forceinline__ unsigned long long int
    custom_atomic_cas_u64(unsigned long long int *addr,
                          unsigned long long int cmp,
                          unsigned long long int val) {
  unsigned long long int old_val;
  asm volatile("atom.cas.release.gpu.b64 %0,[%1],%2,%3;"
               : "=l"(old_val)
               : "l"(addr), "l"(cmp), "l"(val)
               : "memory");
  return old_val;
}

__device__ __forceinline__ void
    get_first_last_ids(unsigned long long int num_elements,
                       unsigned long long int num_workers,
                       unsigned long long int my_id,
                       unsigned long long int *my_first_element,
                       unsigned long long int *my_last_element) {
  unsigned long long int num_elements_per_worker = num_elements / num_workers;
  unsigned long long int reminder = num_elements % num_workers;
  if (my_id < reminder) {
    *my_first_element = (num_elements_per_worker + 1) * my_id;
    *my_last_element = *my_first_element + num_elements_per_worker + 1;
  } else {
    *my_first_element = num_elements_per_worker * my_id + reminder;
    *my_last_element = *my_first_element + num_elements_per_worker;
  }
}

__device__ void terminate_schedulers(RuntimeConfig config) {
  // Event ID 0 is the termination event
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  for (int i = 0; i < num_schedulers; i++) {
    // size_t last_event_id =
    //     atomicAdd(&config.sched_queue_next_free_event_id[i], 1);
    size_t last_event_id =
        custom_atomic_add_u64(&config.sched_queue_next_free_event_id[i], 1);
    config.sched_queues[i][last_event_id % config.per_sched_queue_len] = 0;
    // Add threadfence to make sure sched_queue updates are visible to scheduler
    // CTAs before incrementing its last_ready_event_id
    __threadfence();
    size_t old;
    do {
      // old = atomicCAS(&config.sched_queue_last_ready_event_id[i],
      //                 last_event_id,
      //                 last_event_id + 1);
      old = custom_atomic_cas_u64(&config.sched_queue_last_ready_event_id[i],
                                  last_event_id,
                                  last_event_id + 1);
    } while (old != last_event_id);
  }
}

__global__ void persistent_kernel(RuntimeConfig config) {
  __shared__ TaskId cur_task_id;
  __shared__ TaskDesc task_desc;
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  assert(num_schedulers % 4 == 0);
  assert(gridDim.x == config.num_workers + num_schedulers / 4);
  assert(config.num_workers <= MAX_NUM_WORKERS);
  // We will reinterpret TaskDesc as an array of integers to
  // collectively load it from device to shared memory
  assert(sizeof(TaskDesc) % sizeof(int) == 0);
  PROFILER_CLOSURE_PARAMS_DECL;
  if (config.profiling) {
    PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),
                  0,
                  1,
                  (threadIdx.x % 128 == 0));
  }
  if (blockIdx.x < config.num_workers) {
    int worker_id = blockIdx.x;
    TaskId *worker_queues[2];
    int worker_queue_ids[2];
    TaskId *local_worker_queue = config.worker_queues[worker_id];
    worker_queues[0] = local_worker_queue;
    worker_queue_ids[0] = worker_id;
    int num_worker_queues = 1;
    if (config.num_gpus > 1) {
      TaskId *remote_worker_queue =
          config.worker_queues[worker_id + config.num_workers];
      worker_queues[num_worker_queues] = remote_worker_queue;
      worker_queue_ids[num_worker_queues] = worker_id + config.num_workers;
      num_worker_queues++;
    }
    size_t cur_task_pos[2], last_task_pos[2];
    for (int i = 0; i < 2; i++) {
      cur_task_pos[i] = 0;
      last_task_pos[i] = 0;
    }
    int queue_idx = 0;
    size_t task_counter = 0;
    while (true) {
      // fetch next task from a task queue
      if (threadIdx.x == 0) {
        while (cur_task_pos[queue_idx] == last_task_pos[queue_idx]) {
          //__threadfence();
          // last_task_id = config.worker_queue_last_ready_task_id[worker_id];
          // last_task_id =
          //    atomicAdd(&config.worker_queue_last_ready_task_id[worker_id],
          //    0);
          asm volatile("ld.acquire.gpu.u64 %0, [%1];"
                       : "=l"(last_task_pos[queue_idx])
                       : "l"(&config.worker_queue_last_ready_task_id
                              [worker_queue_ids[queue_idx]]));
          if (cur_task_pos[queue_idx] < last_task_pos[queue_idx]) {
            break;
          } else {
            queue_idx =
                (queue_idx == num_worker_queues - 1) ? 0 : queue_idx + 1;
          }
          // nanosleep to avoid overwhelming I/O
          __nanosleep(10);
        }
        assert(cur_task_pos[queue_idx] + config.per_worker_queue_len >
               last_task_pos[queue_idx]);
        __threadfence();
        cur_task_id = worker_queues[queue_idx][cur_task_pos[queue_idx] %
                                               config.per_worker_queue_len];
        if (config.verbose) {
          printf(
              "[%d][FTCH] worker_id(%d) queue_idx(%d) cur_task_pos(%llu, "
              "%llu) last_task_pos(%llu, %llu) "
              "task_id(%llu) task_type(%d) event_id(%llx) \n",
              config.my_gpu_id,
              worker_id,
              queue_idx,
              cur_task_pos[0],
              cur_task_pos[1],
              last_task_pos[0],
              last_task_pos[1],
              get_task_position_index(cur_task_id),
              config.all_tasks[get_task_position_index(cur_task_id)].task_type,
              config.all_tasks[get_task_position_index(cur_task_id)]
                  .trigger_event);
        }
      }
      __syncthreads();
      int *smem_as_int = reinterpret_cast<int *>(&task_desc);
      int const *dmem_as_int = reinterpret_cast<int *>(
          config.all_tasks + get_task_position_index(cur_task_id));
      for (int i = threadIdx.x; i * sizeof(int) < sizeof(TaskDesc);
           i += blockDim.x) {
        smem_as_int[i] = dmem_as_int[i];
      }
      __syncthreads();
      // Make sure task is ready before start execution
      if (threadIdx.x == 0) {
        if (task_desc.dependent_event != EVENT_INVALID_ID) {
          // Wait until the event has been triggered enough times
          EventId event_id = task_desc.dependent_event;
          assert(!is_nvshmem_event(event_id));
          assert(get_event_gpu_id(event_id) == config.my_gpu_id);
          size_t event_index = get_event_position_index(event_id);
          EventCounter needed_counts =
              static_cast<EventCounter>(
                  config.all_event_num_triggers[event_index]) *
              get_task_iteration_num(cur_task_id);
          EventCounter actual_counts = 0;
          while (actual_counts < needed_counts) {
            asm volatile("ld.acquire.gpu.u64 %0, [%1];"
                         : "=l"(actual_counts)
                         : "l"(&config.all_event_counters[event_index]));
            __nanosleep(10);
          }
        }
      }
      __syncthreads();

      if (config.profiling && task_desc.task_type != TASK_TERMINATE) {
        PROFILER_EVENT_START(task_desc.task_type, task_counter);
      }
      // Successfully fetched a new task
      if (task_desc.task_type == TASK_TERMINATE) {
        // Terminate
        return;
      } else if (task_desc.task_type == TASK_BEGIN_TASK_GRAPH) {
        // Do nothing
      } else {
        switch (task_desc.task_type) {
          case TASK_RMS_NORM_LINEAR: {
            if (task_desc.outputs[0].stride[0] == 6144) {
              kernel::norm_linear_task_impl<bfloat16, 1, 64, 4096, 6144>(
                  task_desc.inputs[0].base_ptr,
                  task_desc.inputs[1].base_ptr,
                  task_desc.inputs[2].base_ptr,
                  1e-6f, // eps
                  task_desc.outputs[0].base_ptr);
            } else if (task_desc.outputs[0].stride[0] == 24576) {
              kernel::norm_linear_task_impl<bfloat16, 1, 256, 4096, 24576>(
                  task_desc.inputs[0].base_ptr,
                  task_desc.inputs[1].base_ptr,
                  task_desc.inputs[2].base_ptr,
                  1e-6f, // eps
                  task_desc.outputs[0].base_ptr);
            } else if (task_desc.outputs[0].stride[0] == 153600) {
              kernel::norm_linear_task_impl<bfloat16, 1, 1600, 4096, 153600>(
                  task_desc.inputs[0].base_ptr,
                  task_desc.inputs[1].base_ptr,
                  task_desc.inputs[2].base_ptr,
                  1e-6f, // eps
                  task_desc.outputs[0].base_ptr);
            } else {
              printf("Unsupported RMSNorm linear task stride: %d\n",
                     task_desc.outputs[0].stride[0]);
              assert(false && "Unsupported RMSNorm linear task");
            }
            break;
          }
          case TASK_EMBEDDING: {
            kernel::embedding_kernel<bfloat16>(task_desc.inputs[0].base_ptr,
                                               task_desc.inputs[1].base_ptr,
                                               task_desc.outputs[0].base_ptr);
            break;
          }
          case TASK_ATTENTION_1: {
            kernel::single_batch_decoding_kernel<bfloat16, 4, 1, 128, 1024>(
                task_desc.inputs[0].base_ptr,
                task_desc.inputs[1].base_ptr,
                task_desc.inputs[2].base_ptr,
                task_desc.outputs[0].base_ptr,
                config.step[0] /*seq_len*/,
                true,
                true,
                task_desc.inputs[3].base_ptr,
                task_desc.inputs[4].base_ptr,
                task_desc.inputs[5].base_ptr,
                task_desc.inputs[6].base_ptr,
                1e-6f /*q_eps*/,
                1e-6f /*k_eps*/);
            break;
          }
          case TASK_ATTENTION_2: {
            TB_SLEEP_US(1);
            break;
          }
          case TASK_SILU_MUL_LINEAR: {
            kernel::silu_mul_linear_task<bfloat16>(
                task_desc.outputs[0].dim[task_desc.outputs[0].num_dims - 1],
                task_desc.inputs[0].base_ptr,
                task_desc.inputs[1].base_ptr,
                task_desc.inputs[2].base_ptr,
                task_desc.outputs[0].base_ptr);
            break;
          }
          case TASK_NVSHMEM_COPY: {
            size_t size_in_bytes = 2;
            for (int i = 0; i < task_desc.inputs[0].num_dims; i++) {
              size_in_bytes *= task_desc.inputs[0].dim[i];
            }
            size_t event_index =
                get_event_position_index(task_desc.trigger_event);
            int gpu_id =
                static_cast<int>(get_event_gpu_id(task_desc.trigger_event));
            assert(gpu_id < config.num_gpus);
            assert(gpu_id != config.my_gpu_id);
            nvshmemx_putmem_signal_block(
                task_desc.outputs[0].base_ptr,
                task_desc.inputs[0].base_ptr,
                size_in_bytes,
                reinterpret_cast<uint64_t *>(
                    &config.all_event_counters[event_index]),
                1 /*signal*/,
                NVSHMEM_SIGNAL_ADD,
                gpu_id);
            // nvshmem_fence();
            break;
          }
          case TASK_REDUCE: {
            TB_SLEEP_US(10);
            break;
          }
          case TASK_MATMUL: {
            kernel::linear_kernel<bfloat16, 1, 64, 4096, 6144>(
                task_desc.inputs[0].base_ptr,
                task_desc.inputs[1].base_ptr,
                task_desc.outputs[0].base_ptr,
                true,
                task_desc.inputs[2].base_ptr);
            break;
          }
          case TASK_ARGMAX: {
            // We may still need this for small vocab size.
            kernel::argmax_kernel<bfloat16, 1, 153600>(
                task_desc.inputs[0].base_ptr, task_desc.outputs[0].base_ptr);
            break;
          }
          case TASK_ARGMAX_PARTIAL: {
            EventId trigger_event_id = task_desc.trigger_event;
            size_t event_idx = get_event_position_index(trigger_event_id);
            EventDesc event_desc = config.all_events[event_idx];
            int partial_task_idx = event_desc.last_task_id - event_desc.first_task_id;
            
            assert(task_desc.inputs[0].num_dims > 0);
            int partial_vocab_size = task_desc.inputs[0].dim[task_desc.inputs[0].num_dims - 1];
            long long index_offset = (long long)partial_task_idx * partial_vocab_size;

            kernel::argmax_partial_kernel<bfloat16>(
                task_desc.inputs[0].base_ptr,   // partial vocab tensor
                task_desc.outputs[0].base_ptr,  // partial max value
                task_desc.outputs[1].base_ptr,  // partial global index
                partial_vocab_size,
                index_offset);
            break;
          }
          case TASK_ARGMAX_REDUCE: {
            assert(task_desc.inputs[0].num_dims > 0);
            int num_partial_tasks = task_desc.inputs[0].dim[task_desc.inputs[0].num_dims - 1];
            kernel::argmax_reduce_kernel<bfloat16>(
                task_desc.inputs[0].base_ptr,   // all partial max values
                task_desc.inputs[1].base_ptr,   // all partial global indices
                task_desc.outputs[0].base_ptr,  // final global index
                num_partial_tasks);
            break;
          }
          default: {
            assert(false && "Unimplemented task");
          }
        } // case
      }   // else
      __syncthreads();
      if (config.profiling && task_desc.task_type != TASK_TERMINATE) {
        PROFILER_EVENT_END(task_desc.task_type, task_counter++);
      }
      // Trigger event
      if (threadIdx.x == 0) {
        EventId event_id = task_desc.trigger_event;
        size_t event_index = get_event_position_index(event_id);
        if (!is_nvshmem_event(event_id)) {
          size_t gpu_id = get_event_gpu_id(event_id);
          assert(gpu_id == config.my_gpu_id);
          // Case 1: Trigger a local non-nvshmem event
          // int count = atomicSub(&config.all_event_counters[event_index], 1);
          EventCounter count =
              custom_atomic_add_u64(&config.all_event_counters[event_index], 1);
          int num_triggers = config.all_event_num_triggers[event_index];
          if (config.verbose) {
            printf("[%d][DONE] worker_id(%d) iter_num(%llu) task_idx(%llu) "
                   "event_id(%llx) "
                   "event_type(local) count(%llu)\n",
                   config.my_gpu_id,
                   worker_id,
                   get_task_iteration_num(cur_task_id),
                   get_task_position_index(cur_task_id),
                   event_id,
                   count);
          }
          if ((count + 1) == static_cast<EventCounter>(num_triggers) *
                                 get_task_iteration_num(cur_task_id)) {
            if (config.profiling) {
              PROFILER_EVENT_START(TASK_SCHD_EVENTS, task_counter);
            }
            EventDesc event_desc = config.all_events[event_index];
            // The event has been triggered enough times
            // Refresh the event counter
            // custom_atomic_add_u64(&config.all_event_counters[event_index],
            //                       event_desc.num_triggers);
            // Add the event to the schedule_queue
            // Note that events launching massive tasks are scheduled
            // to the global sched_queue
            if (event_desc.event_type == EVENT_EMPTY) {
              // Do nothing for empty event
            } else {
              bool use_bcast_queue = false;
              if (event_desc.event_type == EVENT_LAUNCH_MASSIVE_TASKS ||
                  event_desc.event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
                use_bcast_queue = true;
              }
              int sched_id =
                  use_bcast_queue
                      ? config.num_local_schedulers +
                            config.num_remote_schedulers
                      : get_rand_sched_id(event_index,
                                          worker_id,
                                          config.num_workers,
                                          config.num_local_schedulers);
              size_t last_event_pos = custom_atomic_add_u64(
                  &config.sched_queue_next_free_event_id[sched_id], 1);
              config.sched_queues[sched_id]
                                 [last_event_pos % config.per_sched_queue_len] =
                  event_index;
              // Make sure that the updated event_index is visible to the
              // scheduler CTA before updating its last_ready_event_id
              __threadfence();
              size_t old;
              do {
                old = custom_atomic_cas_u64(
                    &config.sched_queue_last_ready_event_id[sched_id],
                    last_event_pos,
                    last_event_pos + 1);
              } while (old != last_event_pos);
            }
            if (config.profiling) {
              PROFILER_EVENT_END(TASK_SCHD_EVENTS, task_counter++);
            }
          }
        } else {
          // Case 2: trigger a nvshmem event
          assert(task_desc.task_type == TASK_NVSHMEM_COPY);
          // Note that nvshmem copy task signal counter during data copy
          // we don't need to do anything here is the task type is NVSHMEM_COPY
          // int gpu_id = static_cast<int>(get_event_gpu_id(event_id));
          // assert(gpu_id < config.num_gpus);
          // assert(gpu_id != config.my_gpu_id);
          // EventCounter count = nvshmem_ulonglong_atomic_fetch_add(
          //    &config.all_event_counters[event_index], 1, gpu_id);
          if (config.verbose) {
            printf("[%d][DONE] worker_id(%d) task_id(%llu) event_id(%llx) "
                   "event_type(remote)\n",
                   config.my_gpu_id,
                   worker_id,
                   get_task_position_index(cur_task_id),
                   event_id);
          }
#ifdef DEADCODE
          if (count == 1) {
            // The event has been triggered enough times
            // Refresh the event counter
            // Note that we load a local event since all task graphs
            // are replicated across gpus and therefore they have the same
            // event metadata (i.e., config.all_events[i] should be the same
            // across GPUs)
            EventDesc event_desc = config.all_events[event_index];
            nvshmem_ulonglong_atomic_add(
                &config.all_event_counters[event_index],
                event_desc.num_triggers,
                gpu_id);
            // Add the event to the schedule queue
            int sched_id = config.num_local_schedulers +
                           get_rand_sched_id(event_index,
                                             worker_id,
                                             config.num_workers,
                                             config.num_remote_schedulers);
            size_t last_event_pos = nvshmem_ulonglong_atomic_fetch_add(
                &config.sched_queue_next_free_event_id[sched_id], 1, gpu_id);
            nvshmem_ulonglong_p(
                &config.sched_queues[sched_id][last_event_pos %
                                               config.per_sched_queue_len],
                event_index,
                gpu_id);
            // use nvshmem_quiet to force completion of remote transfer
            // before updating the last_ready_event_id
            nvshmem_fence();
            size_t old;
            do {
              old = nvshmem_ulonglong_atomic_compare_swap(
                  &config.sched_queue_last_ready_event_id[sched_id],
                  last_event_pos,
                  last_event_pos + 1,
                  gpu_id);
            } while (old != last_event_pos);
          }
#endif
        }
      }
      cur_task_pos[queue_idx] += 1;
    }
  } else {
    // CANNOT use syncthreads on the scheduler side
    int warp_id = threadIdx.x / 32;
    int warp_thread_id = threadIdx.x % 32;
    // assert that we have at least four warps per thread block
    assert(blockDim.x >= 128);
    if (warp_id < 4 && warp_thread_id == 0) {
      int sched_id = (blockIdx.x - config.num_workers) * 4 + warp_id;
      // if (threadIdx.x == 0) {
      //   int sched_id = (blockIdx.x - config.num_workers);
      int num_sched_queues = 1;
      size_t iteration_num = 0;
      EventId *sched_queues[2];
      int sched_queue_ids[2];
      sched_queues[0] = config.sched_queues[sched_id];
      sched_queue_ids[0] = sched_id;
      unsigned long long int my_first_worker, my_last_worker;
      if (sched_id < config.num_local_schedulers) {
        // local schedulers also (collectively) process events from
        // the global queue
        sched_queues[num_sched_queues] = config.sched_queues[num_schedulers];
        sched_queue_ids[num_sched_queues] = num_schedulers;
        num_sched_queues++;
        get_first_last_ids(config.num_workers,
                           config.num_local_schedulers,
                           sched_id,
                           &my_first_worker,
                           &my_last_worker);
      } else {
        get_first_last_ids(config.num_workers,
                           config.num_remote_schedulers,
                           sched_id - config.num_local_schedulers,
                           &my_first_worker,
                           &my_last_worker);
        // Remote schedulers send tasks to remove worker queue
        // whose ids start from config.num_workers
        my_first_worker += config.num_workers;
        my_last_worker += config.num_workers;
      }
      if (config.profiling) {
        printf("[SCHD] sched_id(%d) first_worker(%llu) last_worker(%llu)\n",
               sched_id,
               my_first_worker,
               my_last_worker);
      }
      size_t cur_event_pos[2], last_event_pos[2];
      for (int i = 0; i < 2; i++) {
        cur_event_pos[i] = 0;
        last_event_pos[i] = 0;
      }
      size_t worker_queue_next_free_task_pos[2 * MAX_NUM_WORKERS];
      for (int i = 0; i < 2 * MAX_NUM_WORKERS; i++) {
        worker_queue_next_free_task_pos[i] = 0;
      }
      worker_queue_next_free_task_pos[0] = 1;
      int next_worker = my_first_worker;
      int queue_idx = 0;
      while (true) {
        // if (config.profiling) {
        //   PROFILER_EVENT_START(TASK_GET_EVENT, event_counter);
        // }
        while (cur_event_pos[queue_idx] == last_event_pos[queue_idx]) {
          //__threadfence();
          // last_event_id = config.sched_queue_last_ready_event_id[sched_id];
          // last_event_id =
          //    atomicAdd(&config.sched_queue_last_ready_event_id[sched_id], 0);
          asm volatile("ld.acquire.gpu.u64 %0, [%1];"
                       : "=l"(last_event_pos[queue_idx])
                       : "l"(&config.sched_queue_last_ready_event_id
                              [sched_queue_ids[queue_idx]]));
          if (cur_event_pos[queue_idx] < last_event_pos[queue_idx]) {
            break;
          } else {
            queue_idx = (queue_idx == num_sched_queues - 1) ? 0 : queue_idx + 1;
          }
          // nanosleep to avoid overwhelming I/O
          __nanosleep(10);
        }
        // Make sure the schedule queue is not overflow
        assert(cur_event_pos[queue_idx] + config.per_sched_queue_len >
               last_event_pos[queue_idx]);
        __threadfence();
        // Launch new tasks
        EventId event_id = sched_queues[queue_idx][cur_event_pos[queue_idx] %
                                                   config.per_sched_queue_len];
        EventDesc e = config.all_events[event_id];
        // if (config.profiling) {
        //   PROFILER_EVENT_END(TASK_GET_EVENT, event_counter++);
        // }
        if (is_termination_event(event_id, e)) {
          // terminate all workers
          if (sched_id < config.num_local_schedulers) {
            for (int i = my_first_worker; i < my_last_worker; i++) {
              size_t last_task_id = worker_queue_next_free_task_pos[i]++;
              config.worker_queues[i][last_task_id %
                                      config.per_worker_queue_len] = 0;
              __threadfence();
              custom_atomic_add_u64(&config.worker_queue_last_ready_task_id[i],
                                    1);
            }
          }
          return;
        }
        // This is the ending task of the current task graph
        if (e.event_type == EVENT_END_OF_TASK_GRAPH) {
          // Check if we want to continue
          if (!prepare_next_batch(config)) {
            terminate_schedulers(config);
          } else {
            // Launch task 1 (begin_task_graph) for the next iteration
            size_t last_task_id =
                worker_queue_next_free_task_pos[next_worker]++;
            config.worker_queues[next_worker]
                                [last_task_id % config.per_worker_queue_len] =
                compute_task_id(iteration_num + 1, 1 /*begin_task_graph*/);
            // Make sure writes to worker_queues is visible to worker CTAs
            // before we increase its last_ready_task_id
            __threadfence();
            custom_atomic_add_u64(
                &config.worker_queue_last_ready_task_id[next_worker], 1);
            if (config.verbose) {
              printf("[%d][SCHD] schd_id(%d) iter_num(%llu) task_idx(%llu) "
                     "worker_id(%d) "
                     "worker_last_ready_pos(%llu)\n",
                     config.my_gpu_id,
                     sched_id,
                     iteration_num + 1,
                     1,
                     next_worker,
                     last_task_id + 1);
            }
            next_worker = (next_worker == my_last_worker - 1) ? my_first_worker
                                                              : next_worker + 1;
          }
        } else if (e.event_type == EVENT_LAUNCH_DEPENDENT_TASKS) {
          iteration_num = iteration_num + 1;
          // assign event in a round-robin fashion
          // Split event across local schedulers
          assert(sched_id < config.num_local_schedulers);
          for (size_t i = 0;
               i < (e.last_task_id - e.first_task_id + config.num_workers - 1) /
                       config.num_workers;
               i++) {
            for (size_t j = my_first_worker; j < my_last_worker; j++) {
              size_t position_index =
                  e.first_task_id + i * config.num_workers + j;
              if (position_index < e.last_task_id) {
                size_t last_task_id =
                    worker_queue_next_free_task_pos[next_worker]++;
                config.worker_queues[next_worker][last_task_id %
                                                  config.per_worker_queue_len] =
                    compute_task_id(iteration_num, position_index);
                // Make sure writes to worker_queues is visible to worker CTAs
                // before we increase its last_ready_task_id
                __threadfence();
                custom_atomic_add_u64(
                    &config.worker_queue_last_ready_task_id[next_worker], 1);
                if (config.verbose) {
                  printf("[%d][SCHD] schd_id(%d) iter_num(%llu) task_idx(%llu) "
                         "worker_id(%d) "
                         "worker_last_ready_pos(%llu)\n",
                         config.my_gpu_id,
                         sched_id,
                         iteration_num,
                         position_index,
                         next_worker,
                         last_task_id + 1);
                }
                next_worker = (next_worker == my_last_worker - 1)
                                  ? my_first_worker
                                  : next_worker + 1;
              }
            }
          }
        } else {
          TaskId my_first_task = e.first_task_id, my_last_task = e.last_task_id;
          if (e.event_type == EVENT_LAUNCH_MASSIVE_TASKS) {
            // Split event across local schedulers
            assert(sched_id < config.num_local_schedulers);
            get_first_last_ids(e.last_task_id - e.first_task_id,
                               config.num_local_schedulers,
                               sched_id,
                               &my_first_task,
                               &my_last_task);
            my_first_task += e.first_task_id;
            my_last_task += e.first_task_id;
          }
          for (size_t i = my_first_task; i < my_last_task; i++) {
            // if (config.profiling) {
            //   PROFILER_EVENT_START(TASK_SCHD_TASKS, event_counter);
            // }
            //  size_t last_task_id = atomicAdd(
            //      &(config.worker_queue_next_free_task_id[next_worker]), 1);
            //  size_t last_task_id = custom_atomic_add_u64(
            //     &(config.worker_queue_next_free_task_id[next_worker]), 1);
            size_t last_task_id =
                worker_queue_next_free_task_pos[next_worker]++;
            config.worker_queues[next_worker]
                                [last_task_id % config.per_worker_queue_len] =
                compute_task_id(iteration_num, i);
            // Make sure writes to worker_queues is visible to worker CTAs
            // before we increase its last_ready_task_id
            __threadfence();
            custom_atomic_add_u64(
                &config.worker_queue_last_ready_task_id[next_worker], 1);
            if (config.verbose) {
              printf("[%d][SCHD] schd_id(%d) iter_num(%llu) task_idx(%llu) "
                     "worker_id(%d) "
                     "worker_last_ready_pos(%llu)\n",
                     config.my_gpu_id,
                     sched_id,
                     iteration_num,
                     i,
                     next_worker,
                     last_task_id + 1);
            }
            next_worker = (next_worker == my_last_worker - 1) ? my_first_worker
                                                              : next_worker + 1;
            // if (config.profiling) {
            //   PROFILER_EVENT_END(TASK_SCHD_TASKS, event_counter++);
            // }
          }
        }
        cur_event_pos[queue_idx] += 1;
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
// The following function will be generated by the transpiler
static void
    _init_persistent_kernel(std::vector<TaskDesc> &all_tasks,
                            std::vector<EventDesc> &all_events,
                            std::vector<TaskId> &first_tasks,
                            std::vector<void const *> const &torch_tensors,
                            int num_gpus,
                            int my_gpu_id);

static RuntimeConfig global_runtime_config;

extern "C" void init_persistent_kernel(std::vector<void const *> torch_tensors,
                                       std::vector<void *> meta_tensors,
                                       void *profiler_buffer,
                                       int my_rank,
                                       int num_workers,
                                       int num_local_schedulers,
                                       int num_remote_schedulers) {
  assert(meta_tensors.size() == 1);
  global_runtime_config.step = static_cast<int *>(meta_tensors[0]);
  global_runtime_config.num_workers = num_workers;
  global_runtime_config.num_local_schedulers = num_local_schedulers;
  global_runtime_config.num_remote_schedulers = num_remote_schedulers;
  global_runtime_config.profiler_buffer = profiler_buffer;
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
  printf(
      "process_id(%zu) thread_id(%zu)\n", getpid(), std::this_thread::get_id());
  printf("torch_tensors.size(%zu)\n", torch_tensors.size());
  global_runtime_config.per_worker_queue_len = 1024;
  global_runtime_config.per_sched_queue_len = 1024;
  global_runtime_config.num_gpus = npes;
  global_runtime_config.my_gpu_id = mype;
  global_runtime_config.num_graphs = 1;
  global_runtime_config.verbose = false;
  global_runtime_config.profiling = profiler_buffer != nullptr;

  std::vector<TaskDesc> all_tasks;
  std::vector<EventDesc> all_events;
  std::vector<TaskId> first_tasks;
  _init_persistent_kernel(
      all_tasks, all_events, first_tasks, torch_tensors, npes, mype);
  // for (size_t i = 0; i < all_tasks.size(); i++) {
  //   printf(
  //       "task[%zu]: task_type(%d) trigger_event(%llx)
  //       dependent_event(%llx)\n", i, all_tasks[i].task_type,
  //       all_tasks[i].trigger_event,
  //       all_tasks[i].dependent_event);
  // }

  // Initialize worker queue last task id
  // Each worker now maintains a local and a remote worker queue
  global_runtime_config.worker_queue_last_ready_task_id =
      gpu_malloc<unsigned long long int>((num_workers * 2) *
                                         sizeof(unsigned long long int));
  std::vector<unsigned long long int> host_worker_queue_last_task_id;
  for (int i = 0; i < 2 * num_workers; i++) {
    host_worker_queue_last_task_id.push_back(0);
  }
  cudaMemcpy(global_runtime_config.worker_queue_last_ready_task_id,
             host_worker_queue_last_task_id.data(),
             (num_workers * 2) * sizeof(unsigned long long int),
             cudaMemcpyHostToDevice);
  // Initialize scheduler queue last event id
  // We maintain one extra scheduler queue for the global scheduler
  global_runtime_config.sched_queue_last_ready_event_id =
      gpu_malloc<unsigned long long int>((num_schedulers + 1) *
                                         sizeof(unsigned long long int));
  global_runtime_config.sched_queue_next_free_event_id =
      gpu_malloc<unsigned long long int>((num_schedulers + 1) *
                                         sizeof(unsigned long long int));

  std::vector<unsigned long long int> host_sched_queue_last_event_id;
  for (int i = 0; i < (num_schedulers + 1); i++) {
    host_sched_queue_last_event_id.push_back(0);
  }
  cudaMemcpy(global_runtime_config.sched_queue_last_ready_event_id,
             host_sched_queue_last_event_id.data(),
             (num_schedulers + 1) * sizeof(unsigned long long int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(global_runtime_config.sched_queue_next_free_event_id,
             host_sched_queue_last_event_id.data(),
             (num_schedulers + 1) * sizeof(unsigned long long int),
             cudaMemcpyHostToDevice);
  // Initialize all event counters
  global_runtime_config.all_event_counters =
      gpu_malloc<EventCounter>(all_events.size() * sizeof(EventCounter));
  global_runtime_config.all_event_num_triggers =
      gpu_malloc<int>(all_events.size() * sizeof(int));
  std::vector<int> host_all_event_counters;
  for (size_t i = 0; i < all_events.size(); i++) {
    host_all_event_counters.push_back(all_events.at(i).num_triggers);
  }
  cudaMemcpy(global_runtime_config.all_event_num_triggers,
             host_all_event_counters.data(),
             all_events.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemset(global_runtime_config.all_event_counters,
             0,
             all_events.size() * sizeof(EventCounter));
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
    for (int i = 0; i < (num_workers * 2); i++) {
      TaskId *worker_queue = gpu_malloc<TaskId>(
          global_runtime_config.per_worker_queue_len * sizeof(TaskId));
      host_worker_queues.push_back(worker_queue);
    }
    global_runtime_config.worker_queues =
        gpu_malloc<TaskId *>((num_workers * 2) * sizeof(TaskId *));
    cudaMemcpy(global_runtime_config.worker_queues,
               host_worker_queues.data(),
               (num_workers * 2) * sizeof(TaskId *),
               cudaMemcpyHostToDevice);
  }
  // Initialize scheduler queues
  {
    std::vector<EventId *> host_sched_queues;
    for (int i = 0; i < (num_schedulers + 1); i++) {
      EventId *sched_queue = gpu_malloc<EventId>(
          global_runtime_config.per_sched_queue_len * sizeof(EventId));
      host_sched_queues.push_back(sched_queue);
    }
    global_runtime_config.sched_queues =
        gpu_malloc<EventId *>((num_schedulers + 1) * sizeof(EventId *));
    cudaMemcpy(global_runtime_config.sched_queues,
               host_sched_queues.data(),
               (num_schedulers + 1) * sizeof(EventId *),
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
extern "C" void launch_persistent_kernel() {
  void *args[] = {&global_runtime_config};
  // Launcher persistent kernel
  cudaFuncSetAttribute(
      persistent_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);
  nvshmemx_collective_launch((void const *)persistent_kernel,
                             dim3(108, 1, 1),
                             dim3(128, 1, 1),
                             args,
                             98304 /*sharedmem*/,
                             0 /*stream*/);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
  printf("Finished Launch Persistent Kernel\n");
}

extern "C" void finalize_persistent_kernel() {
  gpu_free(global_runtime_config.worker_queue_last_ready_task_id);
  gpu_free(global_runtime_config.sched_queue_last_ready_event_id);
  gpu_free(global_runtime_config.sched_queue_next_free_event_id);
  gpu_free(global_runtime_config.all_event_counters);
  gpu_free(global_runtime_config.all_event_num_triggers);
  gpu_free(global_runtime_config.all_tasks);
  gpu_free(global_runtime_config.all_events);
  int num_workers = global_runtime_config.num_workers;
  std::vector<TaskId *> host_worker_queues(num_workers * 2);
  cudaMemcpy(host_worker_queues.data(),
             global_runtime_config.worker_queues,
             (num_workers * 2) * sizeof(TaskId *),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < 2 * num_workers; i++) {
    gpu_free(host_worker_queues[i]);
  }
  gpu_free(global_runtime_config.worker_queues);
  int num_schedulers = global_runtime_config.num_local_schedulers +
                       global_runtime_config.num_remote_schedulers;
  std::vector<EventId *> host_sched_queues(num_schedulers + 1);
  cudaMemcpy(host_sched_queues.data(),
             global_runtime_config.sched_queues,
             (num_schedulers + 1) * sizeof(EventId *),
             cudaMemcpyDeviceToHost);
  for (int i = 0; i < num_schedulers + 1; i++) {
    gpu_free(host_sched_queues[i]);
  }
  gpu_free(global_runtime_config.sched_queues);
  gpu_free(global_runtime_config.first_tasks);
  nvshmem_barrier_all();
  nvshmem_finalize();
}
