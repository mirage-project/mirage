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
  int innermost_dim;
  int dim[mirage::config::MAX_TENSOR_DIMS];
  int stride[mirage::config::MAX_TENSOR_DIMS];
  int dtensor_stride[mirage::config::MAX_TENSOR_DIMS];
  int dtensor_dim[mirage::config::MAX_TENSOR_DIMS];
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
  int forloop_range;
  EventId trigger_event;

  int task_id;
  int3 task_partition;
  TensorDesc inputs[mirage::config::MAX_NUM_INPUTS_PER_OPERATOR];
  TensorDesc outputs[mirage::config::MAX_NUM_OUTPUTS_PER_OPERATOR];
};

struct RuntimeConfig {
  int num_workers, num_local_schedulers, num_remote_schedulers, num_graphs;
  int num_gpus, my_gpu_id;
  int total_num_tasks, total_num_events;
  bool profiling = true;
  ;
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

  uint64_t *profiler_buffer;
  int4 *tensor_offsets;
  bool verbose;

};

class Runtime {
public:
  Runtime(int num_gpus, int my_gpu_id);
  template <typename DT>
  DT *gpu_malloc(size_t);
  void register_mugraph(
      mirage::kernel::Graph const &graph,
      std::unordered_map<mirage::kernel::KNOperator const *,
                         std::tuple<int, int, TaskType>> const &task_config);
  void add_tensor_offset(int3 const &inout_map,
                         kernel::DTensor const &dtensor,
                         std::vector<size_t> const &strides,
                         threadblock::Graph const &bgraph,
                         bool is_input);
  void launch_persistent_kernel(int num_workers,
                                int num_local_schedulers,
                                int num_remote_schedulers);
  bool sanity_check();

public:
  std::vector<TaskDesc> all_tasks;
  std::vector<size_t> task_range_begins;
  std::vector<EventDesc> all_events;
  std::vector<TaskId> first_tasks;
  std::vector<int4> tensor_offsets;
  int num_graphs;
  int num_dtensors = 0;
  int num_graphs, num_gpus, my_gpu_id;
};

} // namespace runtime
} // namespace mirage
