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
  TaskType task_type;
  int num_inputs, num_outputs;
  EventId trigger_event, dependent_event;
  TensorDesc inputs[mirage::config::MAX_NUM_INPUTS_PER_OPERATOR];
  TensorDesc outputs[mirage::config::MAX_NUM_OUTPUTS_PER_OPERATOR];
};

struct IODesc {
  enum IOType {
    TorchTensor,
    FusedTorchTensor,
    CUDAMallocTensor,
    NVSHMEMMallocTensor
  };
  IODesc(IOType _type,
         std::string _name,
         mirage::kernel::DTensor const &_tensor)
      : type(_type), name(_name) {
    tensor.num_dims = _tensor.num_dims;
    tensor.data_type = _tensor.data_type;
    assert(_tensor.owner_op->op_type == mirage::type::KN_INPUT_OP);
    mirage::kernel::KNInputOp const *op =
        static_cast<mirage::kernel::KNInputOp const *>(_tensor.owner_op);
    for (int i = 0; i < tensor.num_dims; i++) {
      tensor.dim[i] = _tensor.dim[i];
      tensor.stride[i] = op->input_strides[i];
    }
  }
  IOType type;
  std::string name;
  TensorDesc tensor;
  // Only used for fused tensors
  int num_groups;
  std::vector<IODesc> sub_descs;
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

struct Dim3Comparator {
  bool operator()(dim3 const &a, dim3 const &b) const {
    if (a.x != b.x) {
      return a.x < b.x;
    }
    if (a.y != b.y) {
      return a.y < b.y;
    }
    return a.z < b.z;
  }
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
  void launch_persistent_kernel(int num_workers,
                                int num_local_schedulers,
                                int num_remote_schedulers);
  bool sanity_check();
  std::string print_task_graph(
      mirage::kernel::Graph const &graph,
      std::unordered_map<mirage::kernel::KNOperator const *,
                         std::tuple<int, int, TaskType>> const &task_config,
      std::map<mirage::type::GuidType, IODesc> const &io_configs,
      bool use_json_format);

public:
  std::vector<TaskDesc> all_tasks;
  std::vector<EventDesc> all_events;
  std::vector<TaskId> first_tasks;
  int num_graphs, num_gpus, my_gpu_id;
  std::map<kernel::KNOperator *, std::map<dim3, TaskId, Dim3Comparator>>
      all_task_maps;
};

} // namespace runtime
} // namespace mirage
