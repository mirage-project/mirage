/* Copyright 2023-2024 CMU
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

namespace mirage {
namespace runtime {

typedef unsigned long long int TaskId;
unsigned long long int const TASK_INVALID_ID = 0x7fffffffffffffff;
typedef unsigned long long int EventId;
// Event IDs are 64-bit values encoding both the owner of the event and its
// index EVENT: nvshmem_tag: 16, owner_node: 16, event_idx: 32
unsigned long long int const EVENT_NVSHMEM_TAG = 0x1e00000000000000;
unsigned long long int const EVENT_INVALID_ID = 0x7ffffffffffffffe;

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
  TASK_MATMUL_WITH_RESIDUAL = 108,
  TASK_ARGMAX = 109,
  TASK_ARGMAX_PARTIAL = 110,
  TASK_ARGMAX_REDUCE = 111,
  TASK_NVSHMEM_COPY = 199,
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

} // namespace runtime
} // namespace mirage
