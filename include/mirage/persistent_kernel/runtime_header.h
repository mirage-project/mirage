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

#include "mirage/config.h"
#include <cuda_runtime.h>

namespace mirage {
namespace runtime {

#if defined(MIRAGE_GRACE_HOPPER) || defined(MIRAGE_GRACE_BLACKWELL)
constexpr int WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE = 6 * 1024;
#else
constexpr int WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE = 3 * 1024;
#endif

#if MPK_TARGET_CC >= 90
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    227 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#elif MPK_TARGET_CC >= 86
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    99 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#elif MPK_TARGET_CC >= 80
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    163 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#else
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    163 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#endif

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

int const MAX_INPUTS_PER_TASK = 7;
int const MAX_OUTPUTS_PER_TASK = 3;
int const MAX_NUM_WORKERS = 128;

enum TaskType {
  TASK_TERMINATE = 0,
  TASK_BEGIN_TASK_GRAPH = 10,
  // compute task starts from 100
  TASK_EMBEDDING = 101,
  TASK_RMS_NORM_LINEAR = 102,
  TASK_ATTENTION_1 = 103,
  TASK_ATTENTION_2 = 104,
  TASK_SILU_MUL_LINEAR_WITH_RESIDUAL = 105,
  TASK_ALLREDUCE = 106,
  TASK_REDUCE = 107,
  TASK_LINEAR_WITH_RESIDUAL = 108,
  TASK_ARGMAX = 109,
  TASK_ARGMAX_PARTIAL = 110,
  TASK_ARGMAX_REDUCE = 111,
  TASK_FIND_NGRAM_PARTIAL = 112,
  TASK_FIND_NGRAM_GLOBAL = 113,
  TASK_TARGET_VERIFY_GREEDY = 114,
  TASK_SINGLE_BATCH_EXTEND_ATTENTION = 115,
  TASK_PAGED_ATTENTION_1 = 116,
  TASK_PAGED_ATTENTION_2 = 117,
  TASK_SILU_MUL = 118,
  TASK_RMS_NORM = 119,
  TASK_LINEAR = 120,
  TASK_IDENTITY = 121,
  // Hopper Tasks
  TASK_HOPPER_TASK_BEGIN = 150, // Hopper start placeholder, not a real task
  TASK_LINEAR_WITH_RESIDUAL_HOPPER = 151,
  TASK_LINEAR_HOPPER = 152,
  TASK_PAGED_ATTENTION_HOPPER = 153,
  TASK_RMS_NORM_HOPPER = 154,
  TASK_LINEAR_SWAPAB_HOPPER = 155,
  TASK_LINEAR_SWAPAB_WITH_RESIDUAL_HOPPER = 156,
  TASK_LINEAR_CUTLASS_HOPPER = 157,
  TASK_LINEAR_CUTLASS_WITH_RESIDUAL_HOPPER = 158,
  TASK_SILU_MUL_HOPPER = 159,
  TASK_EMBEDDING_HOPPER = 160,
  TASK_MOE_W13_LINEAR_SM90 = 161,
  TASK_MOE_W2_LINEAR_SM90 = 162,
  TASK_SPLITK_LINEAR_SWAPAB_HOPPER = 163,
  TASK_HOPPER_TASK_END = 198, // Hopper end placeholder, not a real task
  // SM100 Tasks
  TASK_SM100_TASK_BEGIN = 230, // SM100 start placeholder, not a real task
  TASK_SM100_TMA_START_TASK = 231,
  TASK_SPLITK_LINEAR_SM100 = 251,
  TASK_LINEAR_WITH_RESIDUAL_SM100 = 252,
  TASK_LINEAR_SM100 = 253,
  TASK_MOE_W13_LINEAR_SM100 = 254,
  TASK_MOE_W2_LINEAR_SM100 = 255,
  TASK_SM100_TMA_END_TASK = 256,
  TASK_ATTN_SM100 = 257,
  TASK_ARGMAX_REDUCE_SM100 = 258,
  TASK_ARGMAX_PARTIAL_SM100 = 259,
  TASK_MOE_TOPK_SOFTMAX_SM100 = 260,
  TASK_MOE_MUL_SUM_ADD_SM100 = 261,
  TASK_TENSOR_INIT = 262,
  TASK_SM100_TASK_END = 298, // SM100 end placeholder, not a real task
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
#ifdef MPK_ENABLE_TMA
  void *tma_desc_ptrs[mirage::config::MAX_TMA_DESC_PER_TENSOR];
#endif
  int data_type;
  int dim[mirage::config::MAX_TENSOR_DIMS];
  int stride[mirage::config::MAX_TENSOR_DIMS];
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

struct FullTaskDesc {
  FullTaskDesc(TaskType t, int _variant_id)
      : task_type(t), variant_id(_variant_id), num_inputs(0), num_outputs(0),
        trigger_event(EVENT_INVALID_ID), dependent_event(EVENT_INVALID_ID),
        request_id(-1), head_group(-1) {}
  FullTaskDesc() {}
  TaskType task_type;
  unsigned variant_id;
  int num_inputs, num_outputs;
  EventId trigger_event;
  EventId dependent_event;
  TensorDesc inputs[MAX_INPUTS_PER_TASK];
  TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
  union {
    struct {
      int request_id; // Used for paged attention
      int head_group; // Used for paged attention hopper
    };
    int expert_offset; // Used for MoE
  };
};

struct alignas(16) TaskDesc {
  TaskDesc(FullTaskDesc t)
      : task_type(t.task_type), variant_id(t.variant_id),
        trigger_event(t.trigger_event), dependent_event(t.dependent_event),
        request_id(t.request_id), head_group(t.head_group) {
    for (int i = 0; i < t.num_inputs; i++) {
      input_ptrs[i] = t.inputs[i].base_ptr;
    }
    for (int i = 0; i < t.num_outputs; i++) {
      output_ptrs[i] = t.outputs[i].base_ptr;
    }
#ifdef MPK_ENABLE_TMA
    for (int i = 0; i < t.num_inputs; i++) {
      for (int k = 0; k < mirage::config::MAX_TMA_DESC_PER_TENSOR; k++) {
        input_tma_desc_ptrs[i][k] = t.inputs[i].tma_desc_ptrs[k];
      }
    }
    for (int i = 0; i < t.num_outputs; i++) {
      for (int k = 0; k < mirage::config::MAX_TMA_DESC_PER_TENSOR; k++) {
        output_tma_desc_ptrs[i][k] = t.outputs[i].tma_desc_ptrs[k];
      }
    }
#endif
  }
  TaskDesc() {}
  TaskType task_type;
  unsigned variant_id;
  EventId trigger_event;
  EventId dependent_event;
  void *input_ptrs[MAX_INPUTS_PER_TASK];
  void *output_ptrs[MAX_OUTPUTS_PER_TASK];
#ifdef MPK_ENABLE_TMA
  void *input_tma_desc_ptrs[MAX_INPUTS_PER_TASK]
                           [mirage::config::MAX_TMA_DESC_PER_TENSOR];
  void *output_tma_desc_ptrs[MAX_OUTPUTS_PER_TASK]
                            [mirage::config::MAX_TMA_DESC_PER_TENSOR];
#endif
  union {
    struct {
      int request_id; // Used for paged attention
      int head_group; // Used for paged attention hopper
    };
    int expert_offset;         // Used for MoE
    size_t xfer_size_in_bytes; // Used for nvshmem
  };
};

struct RuntimeConfig {
  int num_workers, num_local_schedulers, num_remote_schedulers, num_graphs;
  int num_gpus, my_gpu_id;
  int num_events;
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
  int *step;                    // Metadata for LLM serving
  long long *tokens;            // Metadata for LLM serving
  long long *input_tokens;      // Metadata for LLM serving
  long long *output_tokens;     // Metadata for LLM serving
  long long eos_token_id;       // Metadata for LLM serving
  int max_seq_length;           // Metadata for LLM serving
  int *new_token_nums;          // Metadata for LLM serving
  int *qo_indptr_buffer;        // Metadata for LLM serving (paged attention)
  int *paged_kv_indptr_buffer;  // Metadata for LLM serving (paged attention)
  int *paged_kv_indices_buffer; // Metadata for LLM serving (paged attention)
  int *paged_kv_last_page_len_buffer; // Metadata for LLM serving
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE)
  int *prompt_length;     // Metadata for online/offline serving
  int *request_ids;       // Metadata for online/offline serving
  int *page_queue;        // Metadata for online/offline serving
  int *page_queue_head;   // Metadata for online/offline serving
  int *page_queue_tail;   // Metadata for oneline/offline serving
  int *next_request_id;   // Metadata for LLM serving
  int total_num_requests; // Metadata for LLM serving
#endif
  void *profiler_buffer;
  bool split_worker_scheduler;
  cudaStream_t worker_stream, scheduler_stream;
};

} // namespace runtime
} // namespace mirage
