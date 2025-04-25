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
#include <queue>

namespace mirage {
namespace runtime {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

Runtime::Runtime(int _num_gpus, int _my_gpu_id)
    : num_graphs(0), num_gpus(_num_gpus), my_gpu_id(_my_gpu_id) {
  // add the termination event to the event lists
  EventDesc e(1, 0, 0);
  all_events.push_back(e);
  TaskDesc t(TASK_TERMINATE);
  all_tasks.push_back(t);
  task_range_begins.push_back(0);
}

size_t get_event_id(int my_gpu_id, size_t event_pos, bool nvshmem_event) {
  size_t event_id = ((static_cast<size_t>(my_gpu_id) << 32) | event_pos);
  if (nvshmem_event) {
    event_id = event_id | EVENT_NVSHMEM_TAG;
  }
  return event_id;
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
    int const my_gpu_id,
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
          assert(pre_task_map.find(bid) != pre_task_map.end());
          int task_id = pre_task_map.find(bid)->second;
          // encode gpu_id
          all_tasks[task_id].trigger_event = get_event_id(
              my_gpu_id, all_events.size(), false /*nvshmem_event*/);
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
                                  my_gpu_id,
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

void Runtime::add_tensor_offset(int3 const &inout_map,
                                kn::DTensor const &dtensor,
                                std::vector<size_t> const &strides,
                                tb::Graph const &bgraph,
                                bool is_input) {
  int4 offset;

  for (int dim = 0; dim < 3; ++dim) {
    int div_dim = dim == 0 ? inout_map.x : dim == 1 ? inout_map.y : inout_map.z;
    int num_tbs = dim == 0   ? bgraph.grid_dim.x
                  : dim == 1 ? bgraph.grid_dim.y
                             : bgraph.grid_dim.z;
    if (num_tbs > 1) {
      if (!is_input) {
        assert(div_dim >= 0);
      }
      if (dim == 0) {
        offset.x = dtensor.dim[div_dim] / num_tbs * strides.at(div_dim);
      } else if (dim == 1) {
        offset.y = dtensor.dim[div_dim] / num_tbs * strides.at(div_dim);
      } else {
        offset.z = dtensor.dim[div_dim] / num_tbs * strides.at(div_dim);
      }
    } else {
      if (dim == 0) {
        offset.x = 0;
      } else if (dim == 1) {
        offset.y = 0;
      } else {
        offset.z = 0;
      }
    }
  }
  tensor_offsets.push_back(offset);
}
void Runtime::register_mugraph(
    mirage::kernel::Graph const &graph,
    std::unordered_map<kn::KNOperator const *,
                       std::tuple<int, int, TaskType>> const &task_configs) {
  std::vector<tb::TBInputOp *> pre_output_ops;
  kn::KNCustomizedOp const *pre_op = nullptr;
  std::map<dim3, TaskId, Dim3Comparator> pre_task_map;
  for (auto const &op : graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      continue;
    }
    std::tuple<int, int, TaskType> task_config = task_configs.find(op)->second;
    std::map<dim3, TaskId, Dim3Comparator> cur_task_map;
    assert(op->op_type == type::KNOperatorType::KN_CUSTOMIZED_OP);
    // Customized op
    kn::KNCustomizedOp const *cur_op =
        dynamic_cast<kn::KNCustomizedOp const *>(op);
    tb::Graph const &bgraph = cur_op->bgraph;
    dim3 bid;
    std::vector<TaskDesc> tasks;
    std::vector<tb::TBInputOp *> input_ops;
    std::vector<tb::TBInputOp *> output_ops;
    int num_inputs = std::get<0>(task_config);
    int num_outputs = std::get<1>(task_config);
    TaskType task_type = std::get<2>(task_config);
    assert(bgraph.operators.size() == (size_t)num_inputs + num_outputs);
    for (auto const &op : bgraph.operators) {
      assert(op->op_type == mirage::type::TB_INPUT_OP);
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBInputOp *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBInputOp *>(op));
      }
    }

    // Specical handling for ALLREDUCE
    if (task_type == TASK_ALLREDUCE) {
      // Shouldn't have AllReduce when num_gpus == 1
      assert(num_gpus > 1);
      assert(input_ops.size() == 2);
      assert(output_ops.size() == 1);
      // To simplify the implementation, asserting that
      // produce/consumer must have the same partition
      int num_shared_tensors = 0;
      int3 input_map, output_map;
      for (auto const &input : input_ops) {
        for (auto const &output : pre_output_ops) {
          if (input->dtensor.guid == output->dtensor.guid) {
            input_map = input->input_map;
            output_map = output->input_map;
            num_shared_tensors++;
          }
        }
      }
      assert(num_shared_tensors == 1);
      assert(input_map == output_map);
      assert(bgraph.grid_dim == pre_op->bgraph.grid_dim);
      dim3 bid;
      for (bid.x = 0; bid.x < bgraph.grid_dim.x; bid.x++) {
        for (bid.y = 0; bid.y < bgraph.grid_dim.y; bid.y++) {
          for (bid.z = 0; bid.z < bgraph.grid_dim.z; bid.z++) {
            // event_desc_0 is the trigger_event of previous_task
            // event_desc_1 is the trigger_event of allgather
            EventDesc event_desc_0, event_desc_1;
            event_desc_0.num_triggers = 1;
            event_desc_0.first_task_id = all_tasks.size();
            event_desc_0.last_task_id = all_tasks.size() + num_gpus - 1;
            assert(pre_task_map.find(bid) != pre_task_map.end());
            int task_id = pre_task_map.find(bid)->second;
            all_tasks[task_id].trigger_event =
                get_event_id(my_gpu_id, all_events.size(), false);
            all_events.push_back(event_desc_0);
            // Step 1: create (num_gpus - 1) tasks for allgather
            for (int tgt_gpu_id = 0; tgt_gpu_id < num_gpus; tgt_gpu_id++) {
              if (tgt_gpu_id == my_gpu_id) {
                continue;
              }
              TaskDesc task(TASK_NVSHMEM_COPY);
              task.trigger_event = get_event_id(
                  tgt_gpu_id, all_events.size(), true /*nvshmem_event*/);
              // Initialize input/output tensors to the task
              {
                TensorDesc desc;
                assert(input_ops[0]->output_tensors.size() == 1);
                tb::STensor stensor = input_ops[0]->output_tensors[0];
                desc.num_dims = stensor.num_dims;
                desc.data_type = stensor.data_type;
                for (int d = stensor.num_dims - 1; d >= 0; d--) {
                  desc.dim[d] = stensor.dim[d];
                  desc.stride[d] = (d == stensor.num_dims - 1)
                                       ? 1
                                       : desc.stride[d + 1] *
                                             input_ops[0]->dtensor.dim[d + 1];
                }
                // Input and output have the same shape
                task.inputs[task.num_inputs++] = desc;
                task.outputs[task.num_outputs++] = desc;
              }
              all_tasks.push_back(task);
            }
            event_desc_1.first_task_id = all_tasks.size();
            event_desc_1.last_task_id = all_tasks.size() + 1;
            event_desc_1.num_triggers = num_gpus - 1;
            all_events.push_back(event_desc_1);
            // Step 2: create a task for reduce
            TaskDesc task(TASK_REDUCE);
            // Create input tensor
            {
              TensorDesc desc;
              tb::STensor stensor = input_ops[1]->output_tensors[0];
              desc.num_dims = stensor.num_dims;
              desc.data_type = stensor.data_type;
              for (int d = stensor.num_dims - 1; d >= 0; d--) {
                desc.dim[d] = stensor.dim[d];
                desc.stride[d] =
                    (d == stensor.num_dims - 1)
                        ? 1
                        : desc.stride[d + 1] * input_ops[1]->dtensor.dim[d + 1];
                task.inputs[task.num_inputs++] = desc;
              }
            }
            // Create output tensor
            {
              TensorDesc desc;
              tb::STensor stensor = output_ops[0]->output_tensors[0];
              desc.num_dims = stensor.num_dims;
              desc.data_type = stensor.data_type;
              for (int d = stensor.num_dims - 1; d >= 0; d--) {
                desc.dim[d] = stensor.dim[d];
                desc.stride[d] = (d == stensor.num_dims - 1)
                                     ? 1
                                     : desc.stride[d + 1] *
                                           output_ops[0]->dtensor.dim[d + 1];
                task.inputs[task.num_outputs++] = desc;
              }
              all_tasks.push_back(task);
              // Update current task map
              cur_task_map[bid] = all_tasks.size() - 1;
            }
          }
        }
      }
      pre_output_ops = output_ops;
      pre_op = cur_op;
      pre_task_map = cur_task_map;
      continue;
    }

    // Step 1: add all tasks based on their blockIdx
    // (bid.x, bid.y, bid.z) ordering
    for (bid.x = 0; bid.x < bgraph.grid_dim.x; bid.x++) {
      for (bid.y = 0; bid.y < bgraph.grid_dim.y; bid.y++) {
        for (bid.z = 0; bid.z < bgraph.grid_dim.z; bid.z++) {
          TaskDesc task(task_type);
          int offset = bid.x * bgraph.grid_dim.y * bgraph.grid_dim.z +
                       bid.y * bgraph.grid_dim.z + bid.z;
          task.task_id = offset;
          task.task_partition = {
              bgraph.grid_dim.x, bgraph.grid_dim.y, bgraph.grid_dim.z};
          // Initialize input tensors to the task
          for (auto const &input : input_ops) {
            TensorDesc desc;
            assert(input->output_tensors.size() == 1);
            tb::STensor stensor = input->output_tensors[0];
            std::vector<size_t> const &input_strides =
                static_cast<kn::KNInputOp *>(input->dtensor.owner_op)
                    ->input_strides;
            desc.num_dims = stensor.num_dims;
            desc.data_type = stensor.data_type;
            for (int d = stensor.num_dims - 1; d >= 0; d--) {
              desc.dim[d] = stensor.dim[d];
              desc.stride[d] =
                  (d == stensor.num_dims - 1)
                      ? 1
                      : desc.stride[d + 1] * input->dtensor.dim[d + 1];
              desc.dtensor_stride[d] = input_strides.at(d);
              desc.dtensor_dim[d] = input->dtensor.dim[d];
            }
            task.inputs[task.num_inputs++] = desc;

            // add_tensor_offset(
            //     input->input_map, input->dtensor, input_strides, bgraph, true);
            num_dtensors++;
          }
          // Initialize output tensors to the task
          for (auto const &output : output_ops) {
            TensorDesc desc;
            assert(output->output_tensors.size() == 1);
            tb::STensor stensor = output->output_tensors[0];
            // get default strides
            std::vector<size_t> output_strides = [](kn::DTensor const &A) {
              std::vector<size_t> strides(A.num_dims);
              size_t stride = 1;
              for (int i = A.num_dims - 1; i >= 0; --i) {
                strides[i] = stride;
                stride *= A.dim[i];
              }
              return strides;
            }(output->dtensor);

            // add_tensor_offset(output->input_map,
            //                   output->dtensor,
            //                   output_strides,
            //                   bgraph,
            //                   false);
            num_dtensors++;

            desc.num_dims = stensor.num_dims;
            desc.data_type = stensor.data_type;
            for (int d = stensor.num_dims - 1; d >= 0; d--) {
              desc.dim[d] = stensor.dim[d];
              desc.stride[d] =
                  (d == stensor.num_dims - 1)
                      ? 1
                      : desc.stride[d + 1] * output->dtensor.dim[d + 1];
              desc.dtensor_stride[d] = output_strides.at(d);
              desc.dtensor_dim[d] = output->dtensor.dim[d];
            }
            task.outputs[task.num_outputs++] = desc;
          }
          task.forloop_range = bgraph.forloop_range;
          tasks.push_back(task);
        }
      }
    }
    // Step 2: create events between operators
    if (pre_op == nullptr) {
      // Assert that the first op launches a single task
      assert(tasks.size() == 1);
      first_tasks.push_back(all_tasks.size());
      cur_task_map[dim3(0, 0, 0)] = all_tasks.size();
      all_tasks.push_back(tasks[0]);
      task_range_begins.push_back(1);
    } else {
      // Step 2.1: analyze dependencies between thread blocks of the two ops
      std::vector<int> producer_partition(mirage::config::MAX_TENSOR_DIMS, 1);
      std::vector<int> consumer_partition(mirage::config::MAX_TENSOR_DIMS, 1);
      int num_shared_tensors = 0;
      int3 input_map, output_map;
      for (auto const &input : input_ops) {
        for (auto const &output : pre_output_ops) {
          if (input->dtensor.guid == output->dtensor.guid) {
            input_map = input->input_map;
            output_map = output->input_map;
            num_shared_tensors++;
          }
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
                                  my_gpu_id,               /*my_gpu_id*/
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
                                  tasks,        /*cur_op_tasks*/
                                  pre_task_map, /*pre_task_map*/
                                  cur_task_map /*cur_task_map)*/);
    }
    pre_output_ops = output_ops;
    pre_op = cur_op;
    pre_task_map = cur_task_map;
    task_range_begins.push_back(all_tasks.size());
  }

  // Update the trigger event for all tasks in pre_task_map
  for (auto const &it : pre_task_map) {
    all_tasks[it.second].trigger_event =
        get_event_id(my_gpu_id, 0, false /*nvshmem_event*/);
  }
  num_graphs++;
}

bool Runtime::sanity_check() {
  std::unordered_set<EventId> triggered_events;
  std::unordered_set<TaskId> executed_tasks;
  std::vector<int> event_counts(all_events.size(), 0);
  for (size_t i = 0; i < all_events.size(); i++) {
    event_counts[i] = all_events[i].num_triggers;
  }
  std::queue<TaskId> task_queue;
  std::queue<EventId> event_queue;
  assert(first_tasks.size() == 1);
  task_queue.push(first_tasks[0]);
  while (!(task_queue.empty() && event_queue.empty())) {
    // Execute tasks
    while (!task_queue.empty()) {
      TaskId task = task_queue.front();
      task_queue.pop();
      assert(executed_tasks.count(task) == 0);
      executed_tasks.insert(task);
      TaskDesc desc = all_tasks[task];
      if (desc.trigger_event != EVENT_INVALID_ID) {
        EventId event_id = desc.trigger_event;
        size_t event_pos = event_id & 0xffffffff;
        // event_pos 0 is the end of task graph event
        if (event_pos == 0) {
          continue;
        }
        assert(event_counts[event_pos] > 0);
        event_counts[event_pos]--;
        if (event_counts[event_pos] == 0) {
          event_queue.push(event_id);
        }
      }
    }
    while (!event_queue.empty()) {
      EventId event_id = event_queue.front();
      event_queue.pop();
      assert(triggered_events.count(event_id) == 0);
      triggered_events.insert(event_id);
      size_t event_pos = event_id & 0xffffffff;
      EventDesc desc = all_events[event_pos];
      for (TaskId tid = desc.first_task_id; tid < desc.last_task_id; tid++) {
        task_queue.push(tid);
      }
    }
  }
  printf("Triggered events: %zu\n", triggered_events.size());
  printf("Executed tasks: %zu\n", executed_tasks.size());
  return true;
}

} // namespace runtime
} // namespace mirage
