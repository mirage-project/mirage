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

#include "mirage/kernel/graph.h"

namespace mirage {
namespace runtime {

enum TaskType {
  TASK_TERMINATE = 0,
  TASK_END_OF_ITERATION = 1,
  // compute task starts from 100
  TASK_RMS_NORM_LINEAR = 100,
};

struct TensorDesc {
  int dim[MAX_TENSOR_DIMS];
  int stride[MAX_TENSOR_DIMS];
  void* base_ptr;
  mirage::type::DataType data_type;
  EventId event_id;
};

struct EventDesc {
  int num_triggers;
  TaskId first_task_id, last_task_id;
};

struct TaskDesc {
  TaskType task_type;
  int num_inputs, num_outputs;
  TensorDesc inputs[MAX_NUM_INPUTS];
  TensorDesc outputs[MAX_NUM_OUTPUTS];
};

struct RuntimeConfig {
  int num_workers, num_schedulers, num_graphs;
  int total_num_tasks, total_num_events;
  size_t per_worker_queue_len, per_sched_queue_len;
  size_t* worker_queue_last_task_id;
  size_t* sched_queue_last_event_id;
  int* all_event_counters;
  TaskDesc* all_tasks;
  EventDesc* all_events;
  TaskId **worker_queues;
  EventId **sched_queues;
  TaskId *first_tasks;
};

class Runtime {
public:
  Runtime();
  void register_mugraph(mirage::kernel::Graph const& graph);
public:
  int total_num_tasks, total_num_events;
  int num_graphs;
};

} // namespace runtime
} // namespace mirage

