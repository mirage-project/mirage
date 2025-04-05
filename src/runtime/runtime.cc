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

#include "mirage/runtime/runtime.h"

namespace mirage {
namespace runtime {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

Runtime::Runtime() : num_graphs(0) {
  // add the termination event to the event lists
  EventDesc e(0, 0, 0);
  all_events.push_back(e);
  TaskDesc t(TASK_TERMINATE);
  all_tasks.push_back(t);
}

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

void dfs_create_events_add_tasks(
    int depth,
    std::vector<int> const &event_dims,
    int3 const input_map,
    int3 const output_map,
    dim3 const consumer_grid_dim,
    dim3 const producer_grid_dim,
    dim3 consumer_lo_bid,
    dim3 consumer_hi_bid,
    dim3 producer_lo_bid,
    dim3 producer_hi_bid,
    std::vector<EventDesc> &all_events,
    std::vector<TaskDesc> &all_tasks,
    std::vector<TaskDesc> const &cur_op_tasks,
    std::map<dim3, TaskId, Dim3Comparator> const &pre_task_map,
    std::map<dim3, TaskId, Dim3Comparator> &cur_task_map) {
  if (depth >= mirage::config::MAX_TENSOR_DIMS) {
    EventDesc event_desc;
    event_desc.num_triggers = 0;
    event_desc.first_task_id = all_tasks.size();
    // Add consumer tasks
    dim3 bid;
    for (bid.x = consumer_lo_bid.x; bid.x < consumer_hi_bid.x; bid.x++) {
      for (bid.y = consumer_lo_bid.y; bid.y < consumer_hi_bid.y; bid.y++) {
        for (bid.z = consumer_lo_bid.z; bid.z < consumer_hi_bid.z; bid.z++) {
          cur_task_map[bid] = all_tasks.size();
          int offset = bid.x * consumer_grid_dim.y * consumer_grid_dim.z +
                       bid.y * consumer_grid_dim.z + bid.z;
          all_tasks.push_back(cur_op_tasks[offset]);
        }
      }
    }
    event_desc.last_task_id = all_tasks.size();
    // Set producer tasks
    for (bid.x = producer_lo_bid.x; bid.x < producer_hi_bid.x; bid.x++) {
      for (bid.y = producer_lo_bid.y; bid.y < producer_hi_bid.y; bid.y++) {
        for (bid.z = producer_lo_bid.z; bid.z < producer_hi_bid.z; bid.z++) {
          all_tasks[pre_task_map.find(bid)->second].trigger_event =
              all_events.size();
          event_desc.num_triggers++;
        }
      }
    }
    all_events.push_back(event_desc);
  } else {
    for (int i = 0; i < event_dims[depth]; i++) {
      dim3 new_consumer_lo_bid = consumer_lo_bid;
      dim3 new_consumer_hi_bid = consumer_hi_bid;
      dim3 new_producer_lo_bid = producer_lo_bid;
      dim3 new_producer_hi_bid = producer_hi_bid;
      if (depth == input_map.x) {
        int factor = consumer_grid_dim.x / event_dims[depth];
        new_consumer_lo_bid.x = i * factor;
        new_consumer_hi_bid.x = (i + 1) * factor;
      }
      if (depth == input_map.y) {
        int factor = consumer_grid_dim.y / event_dims[depth];
        new_consumer_lo_bid.y = i * factor;
        new_consumer_hi_bid.y = (i + 1) * factor;
      }
      if (depth == input_map.z) {
        int factor = consumer_grid_dim.z / event_dims[depth];
        new_consumer_lo_bid.z = i * factor;
        new_consumer_hi_bid.z = (i + 1) * factor;
      }
      if (depth == output_map.x) {
        int factor = producer_grid_dim.x / event_dims[depth];
        new_producer_lo_bid.x = i * factor;
        new_producer_hi_bid.x = (i + 1) * factor;
      }
      if (depth == output_map.y) {
        int factor = producer_grid_dim.y / event_dims[depth];
        new_producer_lo_bid.y = i * factor;
        new_producer_hi_bid.y = (i + 1) * factor;
      }
      if (depth == output_map.z) {
        int factor = producer_grid_dim.z / event_dims[depth];
        new_producer_lo_bid.z = i * factor;
        new_producer_hi_bid.z = (i + 1) * factor;
      }
      dfs_create_events_add_tasks(depth + 1,
                                  event_dims,
                                  input_map,
                                  output_map,
                                  consumer_grid_dim,
                                  producer_grid_dim,
                                  new_consumer_lo_bid,
                                  new_consumer_hi_bid,
                                  new_producer_lo_bid,
                                  new_producer_hi_bid,
                                  all_events,
                                  all_tasks,
                                  cur_op_tasks,
                                  pre_task_map,
                                  cur_task_map);
    }
  }
}

void Runtime::register_mugraph(mirage::kernel::Graph const &graph,
                               std::unordered_map<const kn::KNCustomizedOp*, TaskType> const &task_types) {
  std::vector<tb::TBOutputOp *> pre_output_ops;
  kn::KNCustomizedOp const *pre_op = nullptr;
  std::map<dim3, TaskId, Dim3Comparator> pre_task_map;
  for (size_t i = 0; i < graph.operators.size(); i++) {
    if (graph.operators[i]->op_type == type::KNOperatorType::KN_INPUT_OP)
      continue;
    std::map<dim3, TaskId, Dim3Comparator> cur_task_map;
    assert(graph.operators[i]->op_type ==
           type::KNOperatorType::KN_CUSTOMIZED_OP);
    // Customized op
    kn::KNCustomizedOp const *cur_op =
        dynamic_cast<kn::KNCustomizedOp const *>(graph.operators[i]);
    tb::Graph const &bgraph = cur_op->bgraph;
    dim3 bid;
    std::vector<TaskDesc> tasks;
    std::vector<tb::TBInputOp *> input_ops;
    std::vector<tb::TBOutputOp *> output_ops;
    for (auto const &op : bgraph.operators) {
      if (op->op_type == mirage::type::TB_INPUT_OP) {
        input_ops.push_back(static_cast<tb::TBInputOp *>(op));
      }
      if (op->op_type == mirage::type::TB_OUTPUT_OP) {
        output_ops.push_back(static_cast<tb::TBOutputOp *>(op));
      }
    }
    // Step 1: add all tasks based on their blockIdx
    // (bid.x, bid.y, bid.z) ordering
    for (bid.x = 0; bid.x < bgraph.grid_dim.x; bid.x++) {
      for (bid.y = 0; bid.y < bgraph.grid_dim.y; bid.y++) {
        for (bid.z = 0; bid.z < bgraph.grid_dim.z; bid.z++) {
          TaskDesc task(task_types.find(cur_op)->second);
	  // Initialize input tensors to the task
          for (auto const &input : input_ops) {
            TensorDesc desc;
            assert(input->output_tensors.size() == 1);
            tb::STensor stensor = input->output_tensors[0];
            desc.num_dims = stensor.num_dims;
            desc.data_type = stensor.data_type;
            for (int d = stensor.num_dims - 1; d >= 0; d--) {
              desc.dim[d] = stensor.dim[d];
              desc.stride[d] =
                  (d == stensor.num_dims - 1)
                      ? 1
                      : desc.stride[d + 1] * input->dtensor.dim[d + 1];
            }
            task.inputs[task.num_inputs++] = desc;
          }
	  // Initialize output tensors to the task
          for (auto const &output : output_ops) {
            TensorDesc desc;
            assert(output->input_tensors.size() == 1);
            tb::STensor stensor = output->input_tensors[0];
            desc.num_dims = stensor.num_dims;
            desc.data_type = stensor.data_type;
            for (int d = stensor.num_dims - 1; d >= 0; d--) {
              desc.dim[d] = stensor.dim[d];
              desc.stride[d] =
                  (d == stensor.num_dims - 1)
                      ? 1
                      : desc.stride[d + 1] * output->dtensor.dim[d + 1];
            }
            task.outputs[task.num_outputs++] = desc;
          }
          tasks.push_back(task);
        }
      }
    }
    // Step 2: create events between operators
    if (pre_op == nullptr) {
      // Assert that the first op launches a single task
      assert(tasks.size() == 1);
      first_tasks.push_back(all_tasks.size());
      all_tasks.push_back(tasks[0]);
    } else {
      // Step 2.1: analyze dependencies between thread blocks of the two ops
      std::vector<int> producer_partition(mirage::config::MAX_TENSOR_DIMS, 1);
      std::vector<int> consumer_partition(mirage::config::MAX_TENSOR_DIMS, 1);
      int num_shared_tensors = 0;
      int3 input_map, output_map;
      for (auto const &input : input_ops) {
        if (input->dtensor.owner_op == pre_op) {
          input_map = input->input_map;
          output_map = pre_output_ops[input->dtensor.owner_ts_idx]->output_map;
          num_shared_tensors++;
        }
      }
      // assert that their is a single tensor shared between ops
      assert(num_shared_tensors == 1);
      for (int d = 0; d < mirage::config::MAX_TENSOR_DIMS; d++) {
        if (d == input_map.x) {
          consumer_partition[d] = bgraph.grid_dim.x;
        }
        if (d == input_map.y) {
          consumer_partition[d] = bgraph.grid_dim.y;
        }
        if (d == input_map.z) {
          consumer_partition[d] = bgraph.grid_dim.z;
        }
        if (d == output_map.x) {
          producer_partition[d] = pre_op->bgraph.grid_dim.x;
        }
        if (d == output_map.y) {
          producer_partition[d] = pre_op->bgraph.grid_dim.y;
        }
        if (d == output_map.z) {
          producer_partition[d] = pre_op->bgraph.grid_dim.z;
        }
      }
      // Step 2.2: create events and add tasks
      // number of events is the product of gcd of producer/consumer
      std::vector<int> event_dims(mirage::config::MAX_TENSOR_DIMS, 1);
      for (int d = 0; d < mirage::config::MAX_TENSOR_DIMS; d++) {
        event_dims[d] = std::gcd(producer_partition[d], consumer_partition[d]);
      }
      dfs_create_events_add_tasks(0,                       /*depth*/
                                  event_dims,              /*event_dims*/
                                  input_map,               /*input_map*/
                                  output_map,              /*output_map*/
                                  bgraph.grid_dim,         /*consumer_grid_dim*/
                                  pre_op->bgraph.grid_dim, /*producer_grid_dim*/
                                  dim3(0, 0, 0),           /*consumer_lo_bid*/
                                  bgraph.grid_dim,         /*consumer_hi_bid*/
                                  dim3(0, 0, 0),           /*producer_lo_bid*/
                                  pre_op->bgraph.grid_dim, /*producer_hi_bid*/
                                  all_events,
                                  all_tasks,
                                  tasks,                   /*cur_op_tasks*/
                                  pre_task_map,            /*pre_task_map*/
                                  cur_task_map /*cur_task_map)*/);
    }
    pre_output_ops = output_ops;
    pre_op = cur_op;
    pre_task_map = cur_task_map;
  }
  num_graphs++;
}

} // namespace runtime
} // namespace mirage
