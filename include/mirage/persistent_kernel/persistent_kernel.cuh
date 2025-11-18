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

#include "profiler.h"
#include "tasks/common/copy_sm80.cuh"
#ifdef MPK_ENABLE_TMA
#include "tma.cuh"
#endif
#include "mpk_atoms.cuh"
#include "runtime_header.h"
#ifdef USE_NVSHMEM
#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#endif
#include <thread>
#include <unistd.h>
#include <vector>

#if defined(MIRAGE_GRACE_HOPPER)
#include "tasks/hopper/task_header.cuh"
#elif defined(MIRAGE_GRACE_BLACKWELL)
#include "tasks/blackwell/task_header.cuh"
#else
#include "tasks/ampere/task_header.cuh"
#endif

using bfloat16 = type::bfloat16_t;
using namespace mirage::runtime;
using namespace kernel;
// Configurations for the MPK runtime
// #define MPK_MAX_NUM_BATCHED_REQUESTS 16
// #define MPK_MAX_NUM_BATCHED_TOKENS 64
// #define MPK_MAX_NUM_PAGES 1024
// #define MPK_PAGE_SIZE 64

#if defined(MIRAGE_GRACE_HOPPER)
#define WORKER_NUM_THREADS 256
#define SINGLE_KERNEL_NUM_THREADS 256
#elif defined(MIRAGE_GRACE_BLACKWELL)
#define WORKER_NUM_THREADS 256
#define SINGLE_KERNEL_NUM_THREADS 256
#else
#define WORKER_NUM_THREADS 128
#define SINGLE_KERNEL_NUM_THREADS 128
#endif
#define INIT_NUM_THREADS 128

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(1);                                                          \
        }                                                                     \
    } while (0)
#endif

__device__ __forceinline__ void
    _execute_task(TaskDesc const *task_desc,
                  RuntimeConfig const &runtime_config);

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
    // initialize metadata
    for (int i = 0; i < config.total_num_requests; i++) {
      config.step[i] = 0;
    }
    *config.next_request_id = 0;
    for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
      config.request_ids[i] = -1;
    }
    for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS + 1; i++) {
      config.qo_indptr_buffer[i] = 0;
      config.paged_kv_indptr_buffer[i] = 0;
    }
    // Page manager
    *config.page_queue_head = 0;
    *config.page_queue_tail = MPK_MAX_NUM_PAGES;
    for (int i = 0; i < MPK_MAX_NUM_PAGES; i++) {
      config.page_queue[i] = i;
    }
  }
}

__global__ void prepare_kernel(RuntimeConfig config,
                               int end_of_task_graph_event_pos) {
  // Initialize worker queue last task id
  // Each worker now maintains a local and a remote worker queue
  for (int i = blockIdx.x * blockDim.x + threadIdx.x;
       i < 2 * config.num_workers;
       i += blockDim.x * gridDim.x) {
    config.worker_queue_last_ready_task_id[i] = 0;
  }
  // Initialize scheduler queue last event id
  // We maintain one extra scheduler queue for the global scheduler
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_schedulers + 1;
       i += blockDim.x * gridDim.x) {
    config.sched_queue_last_ready_event_id[i] = 0;
    config.sched_queue_next_free_event_id[i] = 0;
  }
  // Initialize all event counters
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < config.num_events;
       i += blockDim.x * gridDim.x) {
    config.all_event_counters[i] = 0;
  }
  // Send event to scheduler[0]
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    assert(config.all_events[end_of_task_graph_event_pos].event_type ==
           EVENT_END_OF_TASK_GRAPH);
    config.sched_queue_next_free_event_id[0] = 1;
    config.sched_queues[0][0] = end_of_task_graph_event_pos;
    config.sched_queue_last_ready_event_id[0] = 1;
  }
}

#ifdef MODE_OFFLINE
// TODO: parallelize this processing
__device__ __forceinline__ bool
    prepare_next_batch(RuntimeConfig const &config) {
  __shared__ int smem_kv_indices[MPK_MAX_NUM_PAGES];
  int page_queue_head = *config.page_queue_head;
  int page_queue_tail = *config.page_queue_tail;
  // Step 1: finalize previous batch
  for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    int request_id = config.request_ids[i];
    if (request_id != -1) {
      // Step 1.1: move output_tokens to tokens
      int step = config.step[request_id];
      int qo_indptr = config.qo_indptr_buffer[i];
      int num_tokens = config.qo_indptr_buffer[i + 1] - qo_indptr;
      int prompt_len = config.prompt_length[request_id];
      for (int j = 0; j < num_tokens; j++) {
        if (step + j + 1 >= prompt_len &&
            step + j + 1 < config.max_seq_length) {
          config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step + j + 1] =
              config.output_tokens[qo_indptr + j];
        }
      }
      config.step[request_id] = step + num_tokens;
#ifdef MPK_ENABLE_PROFILING
      if (true) {
#else
      if ((step + num_tokens + 1 >= config.max_seq_length) ||
          ((config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step +
                          num_tokens] == config.eos_token_id) &&
           (step + num_tokens >= prompt_len))) {
#endif
        // Request is done
        config.request_ids[i] = -1;
        // Free pages
        int kv_indptr = config.paged_kv_indptr_buffer[i];
        int num_pages = config.paged_kv_indptr_buffer[i + 1] - kv_indptr;
        for (int j = 0; j < num_pages; j++) {
          config.page_queue[page_queue_tail % MPK_MAX_NUM_PAGES] =
              config.paged_kv_indices_buffer[kv_indptr + j];
          page_queue_tail++;
        }
      }
    }
  }

  // Step 2: copy kv_indices to shared mem
  int num_pages = config.paged_kv_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS];
  for (int i = 0; i < num_pages; i++) {
    smem_kv_indices[i] = config.paged_kv_indices_buffer[i];
  }

  // Step 3: prepare next batch
  int num_reqs = 0, num_tokens = 0;
  num_pages = 0;
  for (int i = 0; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    int request_id = config.request_ids[i];
    if (request_id != -1) {
      int kv_indptr = config.paged_kv_indptr_buffer[i];
      int num_old_pages = config.paged_kv_indptr_buffer[i + 1] - kv_indptr;
      config.request_ids[num_reqs] = request_id;
      config.qo_indptr_buffer[num_reqs] = num_tokens;
      config.paged_kv_indptr_buffer[num_reqs] = num_pages;
      int step = config.step[request_id];
      int num_new_tokens = config.prompt_length[request_id] - step;
      if (num_new_tokens > 0) {
        // Prefill requests
        num_new_tokens =
            min(num_new_tokens, MPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
      } else {
        // Decode requests
        num_new_tokens = min(1, MPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
      }
      // Move tokens to input_tokens
      for (int j = 0; j < num_new_tokens; j++) {
        config.input_tokens[num_tokens + j] =
            config.tokens[request_id * MPK_MAX_SEQ_LENGTH + step + j];
      }
      // Prepare page indptrs
      int num_new_pages =
          (step + num_new_tokens + MPK_PAGE_SIZE - 1) / MPK_PAGE_SIZE;
      config.paged_kv_last_page_len_buffer[num_reqs] =
          (step + num_new_tokens) % MPK_PAGE_SIZE;
      for (int j = 0; j < num_old_pages; j++) {
        config.paged_kv_indices_buffer[num_pages + j] =
            smem_kv_indices[kv_indptr + j];
      }
      for (int j = num_old_pages; j < num_new_pages; j++) {
        config.paged_kv_indices_buffer[num_pages + j] =
            config.page_queue[page_queue_head % MPK_MAX_NUM_PAGES];
        page_queue_head++;
      }
      num_pages += num_new_pages;
      num_tokens += num_new_tokens;
      num_reqs++;
    }
  }

  // Add new prefill requests until we reach capacity
  while (num_reqs < MPK_MAX_NUM_BATCHED_REQUESTS &&
         num_tokens < MPK_MAX_NUM_BATCHED_TOKENS) {
    int next_request_id = *config.next_request_id;
    if (next_request_id >= config.total_num_requests) {
      break;
    }
    config.request_ids[num_reqs] = next_request_id;
    config.qo_indptr_buffer[num_reqs] = num_tokens;
    config.paged_kv_indptr_buffer[num_reqs] = num_pages;
    // Prefill request
    int num_new_tokens = min(config.prompt_length[next_request_id],
                             MPK_MAX_NUM_BATCHED_TOKENS - num_tokens);
    // Move tokens to input tokens
    for (int j = 0; j < num_new_tokens; j++) {
      config.input_tokens[num_tokens + j] =
          config.tokens[next_request_id * MPK_MAX_SEQ_LENGTH + j];
    }
    int num_new_pages = (num_new_tokens + MPK_PAGE_SIZE - 1) / MPK_PAGE_SIZE;
    config.paged_kv_last_page_len_buffer[num_reqs] =
        num_new_tokens % MPK_PAGE_SIZE;
    for (int j = 0; j < num_new_pages; j++) {
      config.paged_kv_indices_buffer[num_pages + j] =
          config.page_queue[page_queue_head % MPK_MAX_NUM_PAGES];
      page_queue_head++;
    }
    num_tokens += num_new_tokens;
    num_pages += num_new_pages;
    num_reqs++;
    *config.next_request_id = next_request_id + 1;
  }

  // Step 4: Update all unused requests slots
  for (int i = num_reqs; i < MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    config.request_ids[i] = -1;
  }
  for (int i = num_reqs; i <= MPK_MAX_NUM_BATCHED_REQUESTS; i++) {
    config.qo_indptr_buffer[i] = num_tokens;
    config.paged_kv_indptr_buffer[i] = num_pages;
  }

  // Step 5: update page head tail
  *config.page_queue_head = page_queue_head;
  *config.page_queue_tail = page_queue_tail;

  // printf("Next batch: steps[%d %d %d %d] num_active_tokens(%d)\n",
  //        config.step[0],
  //        config.step[1],
  //        config.step[2],
  //        config.step[3],
  //        config.qo_indptr_buffer[MPK_MAX_NUM_BATCHED_REQUESTS]);

  if (num_tokens == 0) {
    return false;
  } else {
    return true;
  }
}
#endif

#ifdef MODE_ONLINE
__device__ __forceinline__ bool
    prepare_next_batch(RuntimeConfig const &config) {
  int step = config.step[0];
#ifdef MPK_ENABLE_VERBOSE
  printf("step: %d, new_token_num(%p): %d, new_token_ids:\n",
         step,
         config.new_token_nums,
         config.new_token_nums[0]);
  for (int i = 0; i < config.new_token_nums[0]; i++) {
    printf("%lld ", config.tokens[step + 1 + i]);
  }
  printf("\n");
#endif
  config.step[0] = step + config.new_token_nums[0];

#ifdef MPK_ENABLE_PROFILING
  return false;
#else
  if ((step + 2 >= config.max_seq_length) ||
      (config.tokens[step + 1] == config.eos_token_id)) {
    return false;
  } else {
    return true;
  }
#endif
}
#endif

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

__device__ __forceinline__ void terminate_schedulers(RuntimeConfig config) {
  // Event ID 0 is the termination event
  int num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  for (int i = 0; i < num_schedulers; i++) {
    // size_t last_event_id =
    //     atomicAdd(&config.sched_queue_next_free_event_id[i], 1);
    size_t last_event_id =
        atom_add_release_gpu_u64(&config.sched_queue_next_free_event_id[i], 1);
    st_relaxed_gpu_u64(
        &config.sched_queues[i][last_event_id % config.per_sched_queue_len], 0);
    // Use st.relaxed to make sure sched_queue updates are visible to scheduler
    // CTAs before incrementing its last_ready_event_id
    size_t old;
    do {
      // old = atomicCAS(&config.sched_queue_last_ready_event_id[i],
      //                 last_event_id,
      //                 last_event_id + 1);
      old = atom_cas_release_gpu_u64(&config.sched_queue_last_ready_event_id[i],
                                     last_event_id,
                                     last_event_id + 1);
    } while (old != last_event_id);
  }
}

__device__ __forceinline__ void worker_checker(RuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  // int num_schedulers =
  //    config.num_local_schedulers + config.num_remote_schedulers;

  assert(gridDim.x == config.num_workers);
  assert(config.num_workers <= MAX_NUM_WORKERS);
  // We will reinterpret TaskDesc as an array of integers to
  // collectively load it from device to shared memory
  static_assert(sizeof(TaskDesc) % sizeof(int) == 0);
}

__device__ __forceinline__ void scheduler_checker(RuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  // int num_schedulers =
  //    config.num_local_schedulers + config.num_remote_schedulers;

  assert(config.num_workers <= MAX_NUM_WORKERS);
}

__device__ __forceinline__ void persistent_checker(RuntimeConfig config) {
  assert(gridDim.y == 1);
  assert(gridDim.z == 1);
  // Each worker SM serves a single worker
  // Each scheduelr SM serves four schedulers
  int const num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  int const num_schedulers_per_sm = std::min((int)blockDim.x / 32, 4);
  assert(num_schedulers % num_schedulers_per_sm == 0);
  assert(gridDim.x ==
         config.num_workers + num_schedulers / num_schedulers_per_sm);
  assert(config.num_workers <= MAX_NUM_WORKERS);
  // We will reinterpret TaskDesc as an array of integers to
  // collectively load it from device to shared memory
  static_assert(sizeof(TaskDesc) % sizeof(int) == 0);
  // assert(blockDim.x >= 128);
}

__device__ __forceinline__ void execute_worker(RuntimeConfig config) {
  // Make sure overall smem usage here do not exceed 3KB
  // last_task_pos: 2 * 8 = 16 B
  // next_task_pos: 2 * 8 = 16 B
  // worker_queue_ids: 2 * 4 = 8 B
  // worker_queues: 2 * 8 = 16 B
  // remaining: 3016 B

  constexpr int TASK_DESCS_BUFFER_LENGTH = std::min(
      (mirage::runtime::WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE - 56) /
          (int)(sizeof(TaskDesc) + sizeof(TaskId)),
      16);
  __shared__ TaskDesc task_descs[TASK_DESCS_BUFFER_LENGTH];
  __shared__ TaskId task_ids[TASK_DESCS_BUFFER_LENGTH];
  __shared__ TaskId *worker_queues[2];
  __shared__ int worker_queue_ids[2];
  __shared__ size_t next_task_pos[2];
  __shared__ size_t last_task_pos[2];

#ifdef MPK_ENABLE_PROFILING
  PROFILER_CLOSURE_PARAMS_DECL;
  PROFILER_INIT(static_cast<uint64_t *>(config.profiler_buffer),
                0,
                1,
                (threadIdx.x % WORKER_NUM_THREADS == 0));

#endif
  int const worker_id = blockIdx.x;
  worker_queues[0] = config.worker_queues[worker_id];
  worker_queue_ids[0] = worker_id;
  int num_worker_queues = 1;
  if (config.num_gpus > 1) {
    worker_queues[num_worker_queues] =
        config.worker_queues[worker_id + config.num_workers];
    worker_queue_ids[num_worker_queues] = worker_id + config.num_workers;
    num_worker_queues++;
  }

  if (threadIdx.x == 0) {
    for (int i = 0; i < 2; i++) {
      next_task_pos[i] = 0;
    }
    for (int i = 0; i < 2; i++) {
      last_task_pos[i] = 0;
    }
    // num_loaded_tasks = 0;
  }

  int queue_pos = 0, queue_len = 0;
#ifdef MPK_ENABLE_PROFILING
  size_t task_counter = 0;
#endif
  while (true) {
    // fetch next task from a task queue if task_descs is empty
    if (queue_pos == queue_len) {
      int queue_idx = 0;
      if (threadIdx.x == 0) {
        while (next_task_pos[queue_idx] == last_task_pos[queue_idx]) {
          last_task_pos[queue_idx] =
              ld_acquire_gpu_u64(&config.worker_queue_last_ready_task_id
                                      [worker_queue_ids[queue_idx]]);
          if (next_task_pos[queue_idx] < last_task_pos[queue_idx]) {
            break;
          } else {
            queue_idx =
                (queue_idx == num_worker_queues - 1) ? 0 : queue_idx + 1;
          }
          // nanosleep to avoid overwhelming I/O
          __nanosleep(10);
        }
        assert(next_task_pos[queue_idx] + config.per_worker_queue_len >
               last_task_pos[queue_idx]);
      }
      __syncthreads();
      int num_loaded_tasks =
          min((int)(last_task_pos[queue_idx] - next_task_pos[queue_idx]),
              TASK_DESCS_BUFFER_LENGTH);
      // Load task ids
      if (threadIdx.x < num_loaded_tasks) {
        task_ids[threadIdx.x] = ld_relaxed_gpu_u64(
            &worker_queues[queue_idx][(next_task_pos[queue_idx] + threadIdx.x) %
                                      config.per_worker_queue_len]);
      }
      __syncthreads();
      if (threadIdx.x == 0) {
#ifdef MPK_ENABLE_VERBOSE
        for (int i = 0; i < num_loaded_tasks; i++) {
          printf(
              "[%d][FTCH] worker_id(%d) queue_idx(%d) next_task_pos(%llu, "
              "%llu) last_task_pos(%llu, %llu) "
              "task_id(%llu) task_type(%d) event_id(%llx) \n",
              config.my_gpu_id,
              worker_id,
              queue_idx,
              next_task_pos[0],
              next_task_pos[1],
              last_task_pos[0],
              last_task_pos[1],
              get_task_position_index(task_ids[i]),
              config.all_tasks[get_task_position_index(task_ids[i])].task_type,
              config.all_tasks[get_task_position_index(task_ids[i])]
                  .trigger_event);
        }
#endif
        next_task_pos[queue_idx] += num_loaded_tasks;
      }
      // Load task descs
      static_assert(sizeof(TaskDesc) % 16 == 0);
      constexpr int TASK_SIZE = sizeof(TaskDesc) / 16; // 128b copy-async
      for (int i = threadIdx.x; i < num_loaded_tasks * TASK_SIZE;
           i += blockDim.x) {
        int task_idx = i / TASK_SIZE;
        int offset = i % TASK_SIZE;
        load_smem(reinterpret_cast<char *>(task_descs) + i * 16,
                  reinterpret_cast<char *>(
                      config.all_tasks +
                      get_task_position_index(task_ids[task_idx])) +
                      offset * 16);
      }
      kernel::cp_async_fence();
      kernel::cp_async_wait<0>();
      __syncthreads();
      queue_pos = 0;
      queue_len = num_loaded_tasks;
    }
    TaskDesc *task_desc = task_descs + queue_pos;
    // Make sure task is ready before start execution
    if (threadIdx.x == 0) {
      if (task_desc->dependent_event != EVENT_INVALID_ID) {
        // Wait until the event has been triggered enough times
        EventId event_id = task_desc->dependent_event;
        assert(get_event_gpu_id(event_id) == config.my_gpu_id);
        size_t event_index = get_event_position_index(event_id);
        EventCounter needed_counts =
            static_cast<EventCounter>(
                config.all_event_num_triggers[event_index]) *
            get_task_iteration_num(task_ids[queue_pos]);
        EventCounter actual_counts = 0;
        if (is_nvshmem_event(event_id)) {
          nvshmem_signal_wait_until(
            reinterpret_cast<uint64_t*>(&config.all_event_counters[event_index]), 
            NVSHMEM_CMP_EQ, 
            needed_counts);
        } else {
          while (actual_counts < needed_counts) {
            actual_counts =
                ld_acquire_sys_u64(&config.all_event_counters[event_index]);
            __nanosleep(10);
          }
        }
      }
    }
    __syncthreads();

#ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_START(task_desc->task_type, task_counter);
    }
#endif

    // Successfully fetched a new task
    if (task_desc->task_type == TASK_TERMINATE) {
      // Terminate
      return;
    } else if (task_desc->task_type == TASK_BEGIN_TASK_GRAPH) {
      // Do nothing
    } else {
#ifdef MPK_ENABLE_VERBOSE
      if (threadIdx.x == 0) {
        printf("[worker] _execute_task EXECUTE_TASK %d\n",
               task_desc->task_type);
      }
#endif
      _execute_task(task_desc, config);
    }
    __syncthreads();

#ifdef MPK_ENABLE_PROFILING
    if (task_desc->task_type != TASK_TERMINATE) {
      PROFILER_EVENT_END(task_desc->task_type, task_counter++);
    }
#endif

    // Trigger event
    if (threadIdx.x == 0) {
      EventId event_id = task_desc->trigger_event;
      size_t event_index = get_event_position_index(event_id);
      if (!is_nvshmem_event(event_id)) {
        size_t gpu_id = get_event_gpu_id(event_id);
        assert(gpu_id == config.my_gpu_id);
        // Case 1: Trigger a local non-nvshmem event
        // int count = atomicSub(&config.all_event_counters[event_index], 1);
        EventCounter count = atom_add_release_gpu_u64(
            &config.all_event_counters[event_index], 1);
        int num_triggers = config.all_event_num_triggers[event_index];
#ifdef MPK_ENABLE_VERBOSE
        printf("[%d][DONE] worker_id(%d) iter_num(%llu) task_idx(%llu) "
               "event_id(%llu) "
               "event_type(local) count(%llu)\n",
               config.my_gpu_id,
               worker_id,
               get_task_iteration_num(task_ids[queue_pos]),
               get_task_position_index(task_ids[queue_pos]),
               event_id,
               count);
#endif

        if ((count + 1) == static_cast<EventCounter>(num_triggers) *
                               get_task_iteration_num(task_ids[queue_pos])) {
#ifdef MPK_ENABLE_PROFILING
          PROFILER_EVENT_START(TASK_SCHD_EVENTS, task_counter);
#endif
          EventDesc event_desc = config.all_events[event_index];
          // The event has been triggered enough times
          // Refresh the event counter
          // atom_add_release_gpu_u64(&config.all_event_counters[event_index],
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
                    ? config.num_local_schedulers + config.num_remote_schedulers
                    : get_rand_sched_id(event_index,
                                        worker_id,
                                        config.num_workers,
                                        config.num_local_schedulers);
            size_t last_event_pos = atom_add_release_gpu_u64(
                &config.sched_queue_next_free_event_id[sched_id], 1);
            st_relaxed_gpu_u64(
                &config.sched_queues[sched_id][last_event_pos %
                                               config.per_sched_queue_len],
                event_index);
            // Use st.relaxed to make sure that the updated event_index is
            // visible to the scheduler CTA before updating its
            // last_ready_event_id
            size_t old;
            do {
              old = atom_cas_release_gpu_u64(
                  &config.sched_queue_last_ready_event_id[sched_id],
                  last_event_pos,
                  last_event_pos + 1);
            } while (old != last_event_pos);
          }
#ifdef MPK_ENABLE_PROFILING
          PROFILER_EVENT_END(TASK_SCHD_EVENTS, task_counter++);
#endif
        }
      } else {
        // Case 2: trigger a nvshmem event
        assert(task_desc->task_type == TASK_NVSHMEM_COPY);
        // Note that nvshmem copy task signal counter during data copy
        // we don't need to do anything here is the task type is NVSHMEM_COPY
#ifdef MPK_ENABLE_VERBOSE
        printf("[%d][DONE] worker_id(%d) task_id(%llu) event_id(%llx) "
               "event_type(remote)\n",
               config.my_gpu_id,
               worker_id,
               get_task_position_index(task_ids[queue_pos]),
               event_id);
#endif
      }
    }
    queue_pos += 1;
  }
}

// need to alter as there is only one warp per block
__device__ __forceinline__ void execute_scheduler(RuntimeConfig config,
                                                  int offset) {
  int const num_schedulers =
      config.num_local_schedulers + config.num_remote_schedulers;
  // if we have more than 4 warps per thread block
  // only the first 4 warps will run schedulers
  int const num_schedulers_per_sm = std::min((int)blockDim.x / 32, 4);
  int const warp_id = threadIdx.x / 32;
  // CANNOT use syncthreads below
  if (threadIdx.x % 32 == 0 && warp_id < num_schedulers_per_sm) {
    int const sched_id = blockIdx.x * num_schedulers_per_sm + warp_id + offset;
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

    // ONLY can run when comment this chunk
#ifdef MPK_ENABLE_VERBOSE
    printf("[SCHD] sched_id(%d) first_worker(%llu) last_worker(%llu)\n",
           sched_id,
           my_first_worker,
           my_last_worker);
#endif
    size_t cur_event_pos[2], last_event_pos[2];
    for (int i = 0; i < 2; i++) {
      cur_event_pos[i] = 0;
      last_event_pos[i] = 0;
    }

    size_t worker_queue_next_free_task_pos[MAX_WORKER_PER_SCHEDULER];
    for (int i = 0; i < MAX_WORKER_PER_SCHEDULER; i++) {
      worker_queue_next_free_task_pos[i] = 0;
    }

    // if (sched_id == 0) {
    //   worker_queue_next_free_task_pos[0] = 1;
    // }
    int next_worker = my_first_worker;
    int queue_idx = 0;
    while (true) {
      while (cur_event_pos[queue_idx] == last_event_pos[queue_idx]) {
        //__threadfence();
        // last_event_id = config.sched_queue_last_ready_event_id[sched_id];
        // last_event_id =
        //    atomicAdd(&config.sched_queue_last_ready_event_id[sched_id], 0);
        last_event_pos[queue_idx] = ld_acquire_gpu_u64(
            &config
                 .sched_queue_last_ready_event_id[sched_queue_ids[queue_idx]]);

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
      // Launch new tasks
      // Use ld.acquire to read latest events
      EventId event_id = ld_relaxed_gpu_u64(
          &sched_queues[queue_idx]
                       [cur_event_pos[queue_idx] % config.per_sched_queue_len]);
      EventDesc e = config.all_events[event_id];
      if (is_termination_event(event_id, e)) {
        // terminate all workers
        if (sched_id < config.num_local_schedulers) {
          for (int i = my_first_worker; i < my_last_worker; i++) {
            size_t last_task_id =
                worker_queue_next_free_task_pos[i - my_first_worker]++;
            st_relaxed_gpu_u64(
                &config.worker_queues[i][last_task_id %
                                         config.per_worker_queue_len],
                0);
            atom_add_release_gpu_u64(&config.worker_queue_last_ready_task_id[i],
                                     1);
          }
        }
        return;
      }
      // This is the ending task of the current task graph
      if (e.event_type == EVENT_END_OF_TASK_GRAPH) {
#ifdef MPK_ENABLE_VERBOSE
        printf("[SCHD] END_OF_TASK_GRAPH\n");
#endif
        // Check if we want to continue
        if (!prepare_next_batch(config)) {
          terminate_schedulers(config);
        } else {
          // Launch task 1 (begin_task_graph) for the next iteration
          size_t last_task_id =
              worker_queue_next_free_task_pos[next_worker - my_first_worker]++;
          st_relaxed_gpu_u64(
              &config.worker_queues[next_worker]
                                   [last_task_id % config.per_worker_queue_len],
              compute_task_id(iteration_num + 1, 1 /*begin_task_graph*/));
          // Use st.relaxed to make sure writes to worker_queues is visible to
          // worker CTAs before we increase its last_ready_task_id
          atom_add_release_gpu_u64(
              &config.worker_queue_last_ready_task_id[next_worker], 1);
#ifdef MPK_ENABLE_VERBOSE
          printf("[%d][SCHD]EVENT_END_OF_TASK_GRAPH schd_id(%d) "
                 "iter_num(%llu) task_idx(1) "
                 "worker_id(%d) "
                 "worker_last_ready_pos(%llu)\n",
                 config.my_gpu_id,
                 sched_id,
                 iteration_num + 1,
                 next_worker,
                 last_task_id + 1);
#endif
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
                  worker_queue_next_free_task_pos[next_worker -
                                                  my_first_worker]++;
              st_relaxed_gpu_u64(
                  &config
                       .worker_queues[next_worker][last_task_id %
                                                   config.per_worker_queue_len],
                  compute_task_id(iteration_num, position_index));
              // Use st.relaxed to make sure writes to worker_queues is visible
              // to worker CTAs before we increase its last_ready_task_id
              atom_add_release_gpu_u64(
                  &config.worker_queue_last_ready_task_id[next_worker], 1);

#ifdef MPK_ENABLE_VERBOSE
              if (sched_id == 0) {
                printf("[%d][SCHD] EVENT_LAUNCH_DEPENDENT_TASKS schd_id(%d) "
                       "iter_num(%llu) task_idx(%llu) "
                       "worker_id(%d) "
                       "worker_last_ready_pos(%llu)"
                       "event_id(%llu)"
                       "event_range(%llu-%llu)\n",
                       config.my_gpu_id,
                       sched_id,
                       iteration_num,
                       position_index,
                       next_worker,
                       last_task_id + 1,
                       event_id,
                       e.first_task_id,
                       e.last_task_id);
              }
#endif
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
          //  size_t last_task_id = atomicAdd(
          //      &(config.worker_queue_next_free_task_id[next_worker]), 1);
          //  size_t last_task_id = atom_add_release_gpu_u64(
          //     &(config.worker_queue_next_free_task_id[next_worker]), 1);
          size_t last_task_id =
              worker_queue_next_free_task_pos[next_worker - my_first_worker]++;
          st_relaxed_gpu_u64(
              &config.worker_queues[next_worker]
                                   [last_task_id % config.per_worker_queue_len],
              compute_task_id(iteration_num, i));
          // Use st.relaxed to make sure writes to worker_queues is visible to
          // worker CTAs before we increase its last_ready_task_id
          atom_add_release_gpu_u64(
              &config.worker_queue_last_ready_task_id[next_worker], 1);

#ifdef MPK_ENABLE_VERBOSE
          printf("[%d][SCHD] EXECUTE_TASK schd_id(%d) iter_num(%llu) "
                 "task_idx(%llu) "
                 "worker_id(%d) "
                 "worker_last_ready_pos(%llu)\n",
                 config.my_gpu_id,
                 sched_id,
                 iteration_num,
                 i,
                 next_worker,
                 last_task_id + 1);
#endif

          next_worker = (next_worker == my_last_worker - 1) ? my_first_worker
                                                            : next_worker + 1;
        }
      }
      cur_event_pos[queue_idx] += 1;
    }
  }
}

__global__ __launch_bounds__(WORKER_NUM_THREADS,
                             1) void persistent_kernel(RuntimeConfig config) {
  persistent_checker(config);
  if (blockIdx.x < config.num_workers) {
    execute_worker(config);
  } else {
    execute_scheduler(config, -(4 * config.num_workers));
  }
}

__global__ __launch_bounds__(WORKER_NUM_THREADS,
                             1) void worker_kernel(RuntimeConfig config) {
  worker_checker(config);
  execute_worker(config);
}

__global__ void scheduler_kernel(RuntimeConfig config) {
  scheduler_checker(config);
  execute_scheduler(config, 0);
}

template <typename DT>
DT *gpu_malloc(size_t size) {
  void *dst_ptr;
#ifdef USE_NVSHMEM
  dst_ptr = nvshmem_malloc(size);
#else
  cudaMalloc(&dst_ptr, size);
#endif
  return static_cast<DT *>(dst_ptr);
}

void gpu_free(void *ptr) {
#ifdef USE_NVSHMEM
  nvshmem_free(ptr);
#else
  cudaFree(ptr);
#endif
}

// The following function will be generated by the transpiler
static void _init_persistent_kernel(std::vector<FullTaskDesc> &all_tasks,
                                    std::vector<EventDesc> &all_events,
                                    std::vector<TaskId> &first_tasks,
                                    int num_gpus,
                                    int my_gpu_id);

static RuntimeConfig global_runtime_config;

// meta_tensors[0]: seq_length
// meta_tensors[1]: tokens
// meta_tensors[2]: input_tokens
// meta_tensors[3]: output_tokens
// meta_tensors[4]: new_tokens_nums
// meta_tensors[5]: prompt_length
// meta_tensors[6]: qo_indptr_buffer
// meta_tensors[7]: paged_kv_indptr_buffer
// meta_tensors[8]: paged_kv_indices_buffer
// meta_tensors[9]: paged_kv_last_page_len_buffer

extern "C" void init_persistent_kernel(std::vector<void *> meta_tensors,
                                       void *profiler_buffer,
                                       int my_rank,
                                       int num_workers,
                                       int num_local_schedulers,
                                       int num_remote_schedulers,
                                       int max_seq_length,
                                       int total_num_requests,
                                       long long eos_token_id) {
  assert(meta_tensors.size() == 10);
  global_runtime_config.step = static_cast<int *>(meta_tensors[0]);
  global_runtime_config.tokens = static_cast<long long *>(meta_tensors[1]);
  global_runtime_config.input_tokens =
      static_cast<long long *>(meta_tensors[2]);
  global_runtime_config.output_tokens =
      static_cast<long long *>(meta_tensors[3]);
  global_runtime_config.new_token_nums = static_cast<int *>(meta_tensors[4]);
  global_runtime_config.prompt_length = static_cast<int *>(meta_tensors[5]);
  global_runtime_config.qo_indptr_buffer = static_cast<int *>(meta_tensors[6]);
  global_runtime_config.paged_kv_indptr_buffer =
      static_cast<int *>(meta_tensors[7]);
  global_runtime_config.paged_kv_indices_buffer =
      static_cast<int *>(meta_tensors[8]);
  global_runtime_config.paged_kv_last_page_len_buffer =
      static_cast<int *>(meta_tensors[9]);
  global_runtime_config.num_workers = num_workers;
  global_runtime_config.num_local_schedulers = num_local_schedulers;
  global_runtime_config.num_remote_schedulers = num_remote_schedulers;
  global_runtime_config.max_seq_length = max_seq_length;
  global_runtime_config.eos_token_id = eos_token_id;
  global_runtime_config.profiler_buffer = profiler_buffer;
  int num_schedulers = num_local_schedulers + num_remote_schedulers;

  // Initialize nvshmem
  cudaSetDevice(my_rank);

#ifdef USE_NVSHMEM
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  attr.mpi_comm = &mpi_comm;
  nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
  nvshmem_barrier_all();
  int mype = nvshmem_my_pe();
  int npes = nvshmem_n_pes();
  int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
  printf("mype(%d) npes(%d) mype_node(%d)\n", mype, npes, mype_node);
#else
  int mype = 0;
  int npes = 1;
#endif

#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
  global_runtime_config.request_ids =
      gpu_malloc<int>(sizeof(int) * (MPK_MAX_NUM_BATCHED_REQUESTS + 1));
  global_runtime_config.next_request_id = gpu_malloc<int>(sizeof(int));
  global_runtime_config.page_queue =
      gpu_malloc<int>(MPK_MAX_NUM_PAGES * sizeof(int));
  global_runtime_config.page_queue_head = gpu_malloc<int>(sizeof(int));
  global_runtime_config.page_queue_tail = gpu_malloc<int>(sizeof(int));
  global_runtime_config.total_num_requests = total_num_requests;
#endif
  global_runtime_config.per_worker_queue_len = 1024;
  global_runtime_config.per_sched_queue_len = 1024;
  global_runtime_config.num_gpus = npes;
  global_runtime_config.my_gpu_id = mype;
  global_runtime_config.num_graphs = 1;
  global_runtime_config.split_worker_scheduler = true;

  std::vector<FullTaskDesc> all_fulltasks;
  std::vector<EventDesc> all_events;
  std::vector<TaskId> first_tasks;
  _init_persistent_kernel(all_fulltasks, all_events, first_tasks, npes, mype);
  std::vector<TaskDesc> all_tasks;
  for (auto const &ft : all_fulltasks) {
    TaskDesc task_desc(ft);
    // Reinterpret part of TaskDesc to save xfer_size information
    if (ft.task_type == TASK_NVSHMEM_COPY) {
      int size_in_bytes = 2;
      for (int i = 0; i < ft.inputs[0].num_dims; i++) {
        size_in_bytes *= ft.inputs[0].dim[i];
      }
      task_desc.xfer_size_in_bytes = size_in_bytes;
    }
    all_tasks.push_back(task_desc);
  }

  // Initialize worker queue last task id
  // Each worker now maintains a local and a remote worker queue
  global_runtime_config.worker_queue_last_ready_task_id =
      gpu_malloc<unsigned long long int>((num_workers * 2) *
                                         sizeof(unsigned long long int));
  // std::vector<unsigned long long int> host_worker_queue_last_task_id;
  // for (int i = 0; i < 2 * num_workers; i++) {
  //   host_worker_queue_last_task_id.push_back(0);
  // }
  // cudaMemcpy(global_runtime_config.worker_queue_last_ready_task_id,
  //            host_worker_queue_last_task_id.data(),
  //            (num_workers * 2) * sizeof(unsigned long long int),
  //            cudaMemcpyHostToDevice);
  //  Initialize scheduler queue last event id
  //  We maintain one extra scheduler queue for the global scheduler
  global_runtime_config.sched_queue_last_ready_event_id =
      gpu_malloc<unsigned long long int>((num_schedulers + 1) *
                                         sizeof(unsigned long long int));
  global_runtime_config.sched_queue_next_free_event_id =
      gpu_malloc<unsigned long long int>((num_schedulers + 1) *
                                         sizeof(unsigned long long int));

  // std::vector<unsigned long long int> host_sched_queue_last_event_id;
  // for (int i = 0; i < (num_schedulers + 1); i++) {
  //   host_sched_queue_last_event_id.push_back(0);
  // }
  // cudaMemcpy(global_runtime_config.sched_queue_last_ready_event_id,
  //            host_sched_queue_last_event_id.data(),
  //            (num_schedulers + 1) * sizeof(unsigned long long int),
  //            cudaMemcpyHostToDevice);
  // cudaMemcpy(global_runtime_config.sched_queue_next_free_event_id,
  //            host_sched_queue_last_event_id.data(),
  //            (num_schedulers + 1) * sizeof(unsigned long long int),
  //            cudaMemcpyHostToDevice);
  //  Initialize all event counters
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
  // cudaMemset(global_runtime_config.all_event_counters,
  //            0,
  //            all_events.size() * sizeof(EventCounter));
  //  Initialize all tasks
  global_runtime_config.all_tasks =
      gpu_malloc<TaskDesc>(all_tasks.size() * sizeof(TaskDesc));
  cudaMemcpy(global_runtime_config.all_tasks,
             all_tasks.data(),
             all_tasks.size() * sizeof(TaskDesc),
             cudaMemcpyHostToDevice);
  // Initialize all events
  global_runtime_config.num_events = (int)all_events.size();
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

  // Set configuration for kernels
  cudaFuncSetAttribute(worker_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  cudaFuncSetAttribute(scheduler_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  cudaFuncSetAttribute(persistent_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  // Create worker and scheduler streams
  cudaStreamCreate(&global_runtime_config.worker_stream);
  cudaStreamCreate(&global_runtime_config.scheduler_stream);

  // launch init kernel
  init_kernel<<<dim3(1, 1, 1), dim3(INIT_NUM_THREADS, 1, 1)>>>(
      global_runtime_config);
  cudaDeviceSynchronize();
#ifdef USE_NVSHMEM
  // Add a global barrier for all init_kernel to complete
  nvshmem_barrier_all();
#endif
}

// Entry point for C/C++
// TODO: change launch config
extern "C" void launch_persistent_kernel() {
  // int device;
  // cudaGetDevice(&device);
  // int sm_count;
  // cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  //  Prepare next persistent kernel by resetting queue pointers
  {
    int end_of_task_graph_event_pos = global_runtime_config.num_events - 1;
    prepare_kernel<<<dim3(global_runtime_config.num_workers, 1, 1),
                     dim3(128, 1, 1)>>>(global_runtime_config,
                                        end_of_task_graph_event_pos);
    cudaDeviceSynchronize();
#ifdef USE_NVSHMEM
    nvshmem_barrier_all();
#endif
  }
  int num_schedulers = global_runtime_config.num_local_schedulers +
                       global_runtime_config.num_remote_schedulers;
  if (global_runtime_config.split_worker_scheduler) {
    printf("worker kernel & scheduler kernel\n");
    printf("smem size: %d\n", MAX_DYNAMIC_SHARED_MEMORY_SIZE);

    // The split kernel does not support NVSHMEM because
    // nvshmemx_collective_launch launches kernels sequentially, which blocks
    // the interaction between the worker kernel and the scheduler kernel
    worker_kernel<<<dim3(global_runtime_config.num_workers, 1, 1),
                    dim3(WORKER_NUM_THREADS, 1, 1),
                    MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/,
                    global_runtime_config.worker_stream>>>(
        global_runtime_config);

    scheduler_kernel<<<dim3(global_runtime_config.num_local_schedulers, 1, 1),
                       dim3(32, 1, 1),
                       0 /*smem*/,
                       global_runtime_config.scheduler_stream>>>(
        global_runtime_config);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    printf("Finished Launch Persistent Kernel\n");
  } else {
    printf("a single persistent kernel\n");
    int num_sms_to_use = global_runtime_config.num_workers + num_schedulers / 4;
#ifdef USE_NVSHMEM
    void *args[] = {&global_runtime_config};
    nvshmemx_collective_launch((void const *)persistent_kernel,
                               dim3(num_sms_to_use, 1, 1),
                               dim3(SINGLE_KERNEL_NUM_THREADS, 1, 1),
                               args,
                               MAX_DYNAMIC_SHARED_MEMORY_SIZE /*sharedmem*/,
                               0 /*stream*/);
#else
    persistent_kernel<<<dim3(num_sms_to_use, 1, 1),
                        dim3(SINGLE_KERNEL_NUM_THREADS, 1, 1),
                        MAX_DYNAMIC_SHARED_MEMORY_SIZE /*smem*/>>>(
        global_runtime_config);
#endif
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    printf("Finished Launch Persistent Kernel\n");
  }
}

extern "C" void finalize_persistent_kernel() {
  gpu_free(global_runtime_config.worker_queue_last_ready_task_id);
  gpu_free(global_runtime_config.sched_queue_last_ready_event_id);
  gpu_free(global_runtime_config.sched_queue_next_free_event_id);
  gpu_free(global_runtime_config.all_event_counters);
  gpu_free(global_runtime_config.all_event_num_triggers);
  gpu_free(global_runtime_config.all_tasks);
  gpu_free(global_runtime_config.all_events);
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
  gpu_free(global_runtime_config.next_request_id);
  gpu_free(global_runtime_config.page_queue);
  gpu_free(global_runtime_config.page_queue_head);
  gpu_free(global_runtime_config.page_queue_tail);
#endif
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
#ifdef USE_NVSHMEM
  nvshmem_barrier_all();
  nvshmem_finalize();
#endif
  // Free worker and scheduler streams
  cudaStreamDestroy(global_runtime_config.worker_stream);
  cudaStreamDestroy(global_runtime_config.scheduler_stream);
}
