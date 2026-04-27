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

#ifdef USE_NVSHMEM
#if defined(MIRAGE_GRACE_BLACKWELL)
// Blackwell (SM100a): include only host API + types.
// Device-side allreduce is self-contained in tasks/blackwell/allreduce.cuh
// to avoid rdc=true register inflation (166 vs 255 regs).
//
// Define nvshmemi_device_state_d BEFORE any NVSHMEM headers so that any
// transitively-included device code (proxy_device.cuh etc.) can resolve it.
// In standard NVSHMEM this comes from libnvshmem_device.a, but we skip that
// library to avoid rdc=true.
#include "device_host/nvshmem_types.h"
#ifdef NVSHMEM_NO_DEVICE_LIB
__managed__ nvshmemi_device_host_state_t nvshmemi_device_state_d;
#endif
#include <nvshmem_host.h>
#else
// Hopper/Ampere: use standard NVSHMEM includes (rdc=true is fine on SM90).
#include <nvshmem.h>
#include <nvshmemx.h>
#endif
#endif

namespace mirage {
namespace runtime {

#if defined(MIRAGE_GRACE_HOPPER) || defined(MIRAGE_GRACE_BLACKWELL)
constexpr int WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE = 6 * 1024;
#else
constexpr int WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE = 3 * 1024;
#endif

#if defined(MODE_ONLINE_NOTOKEN) || defined(MODE_MULTI_TURN)
// Have to be smaller for vllm compatibility, or program will stuck
#if MPK_TARGET_CC >= 90
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    220 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#elif MPK_TARGET_CC >= 86
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    99 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#elif MPK_TARGET_CC >= 80
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    160 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#else
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    163 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
#endif
#else
#if MPK_TARGET_CC >= 90
// B200: 228KB total smem. PR 651 MLA reduce adds ~16KB static smem
// (la_smem[MAX_SK*128]). Reduce dynamic budget to stay under total limit.
constexpr int MAX_DYNAMIC_SHARED_MEMORY_SIZE =
    207 * 1024 - WORKER_RESERVED_STATIC_SHARED_MEMORY_SIZE;
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
// B200 has 148 SMs — need more workers than the default 128
int const MAX_NUM_WORKERS = 160;

enum TaskType {
  TASK_TERMINATE = 0,
  TASK_BEGIN_TASK_GRAPH = 10,
  // compute task starts from 100
  TASK_EMBEDDING = 101,
  TASK_RMS_NORM_LINEAR = 102,
  TASK_ATTENTION_1 = 103,
  TASK_ATTENTION_2 = 104,
  TASK_SILU_MUL_LINEAR_WITH_RESIDUAL = 105,
  TASK_ALLREDUCE = 106, // This legacy allreduce task will be removed soon
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
  TASK_PAGED_ATTENTION_SPLIT_KV_HOPPER = 164,
  TASK_HOPPER_TASK_END = 198, // Hopper end placeholder, not a real task
  // SM100 Tasks
  TASK_SM100_TASK_BEGIN = 230, // SM100 start placeholder, not a real task
  TASK_SM100_TMA_START_TASK = 231,
  TASK_MOE_W13_FP8_SM100 = 248,
  TASK_MOE_W2_FP8_SM100 = 249,
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
  TASK_PAGED_ATTENTION_SPLIT_KV_SM100 = 263,
  TASK_PAGED_ATTENTION_SPLIT_KV_MERGE_SM100 = 264,
  TASK_SAMPLING_SM100 = 265,
  TASK_MLA_DECODE_SM100 = 266,
  TASK_MLA_REDUCE_SM100 = 267,
  TASK_MLA_PREFILL_SM100 = 268,
  TASK_MLA_MTP_DECODE_SM100 = 269,
  TASK_MLA_MTP_REDUCE_SM100 = 270,
  TASK_MTP_VERIFY_STRICT = 271,
  TASK_MTP_ACCEPT_COMMIT = 272,
  TASK_MTP_TOKEN_SCATTER = 273,
  TASK_MTP_PREPARE_VERIFY = 274,
  TASK_QUANTIZE_FP8_SM100 = 275,
  TASK_LINEAR_FP8_SM100 = 276,
  TASK_LINEAR_FP8_WITH_RESIDUAL_SM100 = 277,
  TASK_MLA_KV_GATHER_SM100 = 278,
  TASK_MOE_TOPK_SIGMOID_SM100 = 280,
  TASK_ELEMENTWISE_ADD_SM100 = 281,
  TASK_SOFTMAX_GATHER_SM100 = 282,
  TASK_MTP_VERIFY_PROBABILISTIC = 283,
  TASK_PROB_SCATTER_SM100 = 284,
  TASK_MTP_FLOAT_SCATTER = 285,
  TASK_PROB_EXTRACT_SM100 = 286,
  // MLA-MTP TP variants (q1..q4, kv4096; ferret-derived, no-PDL):
  TASK_MLA_MTP_DECODE_TP2_SM100 = 287,
  TASK_MLA_MTP_DECODE_TP2_REDUCE_SM100 = 288,
  TASK_MLA_MTP_DECODE_TP4_SM100 = 289,
  TASK_MLA_MTP_DECODE_TP4_REDUCE_SM100 = 290,
  TASK_MLA_MTP_DECODE_TP8_SM100 = 291,
  TASK_MLA_MTP_DECODE_TP8_REDUCE_SM100 = 292,
  // KV gather variant that writes split CKV/KPE output (for chunked prefill):
  TASK_MLA_KV_GATHER_SPLIT_SM100 = 293,
  // MTP embedding-input builder (vLLM-aligned): produces per-iteration MTP
  // input tokens = shifted ground-truth prompt + current iter's argmax tail.
  TASK_MTP_BUILD_EMBED_INPUT = 294,
  // MLA prefill TP=8: unabsorbed, TMA K/V, seq_len<=4096.
  TASK_MLA_PREFILL_TP8_SM100 = 295,
  // Unified DeepSeek MLA dispatcher: prefill or MTP decode by runtime Q_LEN.
  TASK_MLA_UNIFIED_SM100 = 296,
  // Unified KV gather: appends once, then writes the prefill or decode layout
  // selected by runtime Q_LEN.
  TASK_MLA_KV_GATHER_UNIFIED_SM100 = 297,
  TASK_SM100_TASK_END = 298, // SM100 end placeholder, not a real task
  TASK_SCHD_TASKS = 200,
  TASK_SCHD_EVENTS = 201,
  TASK_GET_EVENT = 202,
  TASK_GET_NEXT_TASK = 203,
  // Multi-GPU tasks
  TASK_MULTIGPU_TASK_BEGIN = 300, // begin placeholder, not a real task
  TASK_NVSHMEM_ALLGATHER_STRIDED_PUT = 301,
  TASK_NVSHMEM_TILE_ALLREDUCE = 302,
  TASK_MULTIGPU_TASK_END = 349, // end placeholder, not a real task
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
        trigger_event(EVENT_INVALID_ID), dependent_event(EVENT_INVALID_ID) {
    task_metadata.raw_payload = ~0ull;
  }
  FullTaskDesc() {
    task_metadata.raw_payload = ~0ull;
  }
  TaskType task_type;
  unsigned variant_id;
  int num_inputs, num_outputs;
  EventId trigger_event;
  EventId dependent_event;
  TensorDesc inputs[MAX_INPUTS_PER_TASK];
  TensorDesc outputs[MAX_OUTPUTS_PER_TASK];
  union TaskMetadata {
    struct {
      int expert_offset; // Used for MoE
    };
    struct {
      int16_t request_id;    // Used for paged attention
      uint16_t kv_idx;       // Used for paged attention split kv
      int merge_task_offset; // Used for paged attention split kv merge
    };
    struct {
      int task_offset; // Used for nvshmem team mapping
    };
    unsigned long long raw_payload;
  } task_metadata;
};

static_assert(
    sizeof(FullTaskDesc::TaskMetadata) == sizeof(unsigned long long),
    "FullTaskDesc::TaskMetadata layout changed; update raw_payload type.");

struct alignas(16) TaskDesc {
  TaskDesc(FullTaskDesc t)
      : task_type(t.task_type), variant_id(t.variant_id),
        trigger_event(t.trigger_event), dependent_event(t.dependent_event),
        task_metadata(t.task_metadata) {
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
  TaskDesc() {
    task_metadata.raw_payload = ~0ull;
  }
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
  FullTaskDesc::TaskMetadata task_metadata;
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
  int *step;                      // Metadata for LLM serving
  long long *tokens;              // Metadata for LLM serving
  long long *input_tokens;        // Metadata for LLM serving
  long long *output_tokens;       // Metadata for LLM serving
  long long eos_token_id;         // Metadata for LLM serving
  int max_seq_length;             // Metadata for LLM serving
  int *new_token_nums;            // Metadata for LLM serving
  int *qo_indptr_buffer;          // Metadata for LLM serving (paged attention)
  int *paged_kv_indptr_buffer;    // Metadata for LLM serving (paged attention)
  int *paged_kv_indices_buffer;   // Metadata for LLM serving (paged attention)
  int *paged_kv_indices_snapshot; // Scheduler snapshot for in-place compaction
  int *paged_kv_last_page_len_buffer; // Metadata for LLM serving
#if defined(MODE_OFFLINE) || defined(MODE_ONLINE) ||                           \
    defined(MODE_ONLINE_NOTOKEN)
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
  cudaEvent_t prepare_done_event;
  cudaEvent_t worker_done_event, scheduler_done_event;
#ifdef USE_NVSHMEM
  nvshmem_team_t *nvshmem_teams;
#endif
};

} // namespace runtime
} // namespace mirage
