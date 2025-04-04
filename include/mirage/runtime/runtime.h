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
#include "mirage/kernel/graph.h"
#include "mirage/runtime/runtime_types.h"
#include "mirage/type.h"

namespace mirage {
namespace runtime {

struct TensorDesc {
  int num_dims;
  void *base_ptr;
  mirage::type::DataType data_type;
  int dim[mirage::config::MAX_TENSOR_DIMS];
  int stride[mirage::config::MAX_TENSOR_DIMS];
};

struct EventDesc {
  EventDesc(void) : num_triggers(0), first_task_id(0), last_task_id(0) {}
  EventDesc(int nt, TaskId f, TaskId l)
      : num_triggers(nt), first_task_id(f), last_task_id(l) {}
  int num_triggers;
  TaskId first_task_id, last_task_id;
};

struct TaskDesc {
  TaskDesc(TaskType t)
      : task_type(t), num_inputs(0), num_outputs(0), trigger_event(0) {}
  TaskType task_type;
  int num_inputs, num_outputs;
  EventId trigger_event;
  TensorDesc inputs[mirage::config::MAX_NUM_INPUTS_PER_OPERATOR];
  TensorDesc outputs[mirage::config::MAX_NUM_OUTPUTS_PER_OPERATOR];
};

struct RuntimeConfig {
  int num_workers, num_schedulers, num_graphs;
  int total_num_tasks, total_num_events;
  unsigned long long int per_worker_queue_len, per_sched_queue_len;
  unsigned long long int *worker_queue_last_task_id;
  unsigned long long int *sched_queue_last_event_id;
  int *all_event_counters;
  TaskDesc *all_tasks;
  EventDesc *all_events;
  TaskId **worker_queues;
  EventId **sched_queues;
  TaskId *first_tasks;
};

class Runtime {
public:
  Runtime();
  void register_mugraph(mirage::kernel::Graph const &graph,
                        std::vector<TaskType> const &task_types);
  void launch_persistent_kernel(int num_workers, int num_schedulers);

public:
  std::vector<TaskDesc> all_tasks;
  std::vector<EventDesc> all_events;
  std::vector<TaskId> first_tasks;
  int num_graphs;
};

} // namespace runtime
} // namespace mirage
