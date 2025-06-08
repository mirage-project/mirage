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
#include "mirage/transpiler/utils.h"
#include "mirage/utils/json_utils.h"
#include <queue>

namespace mirage {
namespace runtime {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

Runtime::Runtime(int _num_gpus, int _my_gpu_id)
    : num_graphs(0), num_gpus(_num_gpus), my_gpu_id(_my_gpu_id) {
  // add the termination event to the event lists
  EventDesc e(EVENT_TERMINATION, 1, 0, 0);
  all_events.push_back(e);
  TaskDesc t(TASK_TERMINATE);
  all_tasks.push_back(t);
}

size_t get_event_id(int my_gpu_id, size_t event_pos, bool nvshmem_event) {
  size_t event_id = ((static_cast<size_t>(my_gpu_id) << 32) | event_pos);
  if (nvshmem_event) {
    event_id = event_id | EVENT_NVSHMEM_TAG;
  }
  return event_id;
}

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
    event_desc.event_type =
        event_desc.last_task_id >= event_desc.first_task_id + 8
            ? EVENT_LAUNCH_MASSIVE_TASKS
            : EVENT_LAUNCH_TASKS;
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

void Runtime::register_mugraph(
    mirage::kernel::Graph const &graph,
    std::unordered_map<kn::KNOperator const *,
                       std::tuple<int, int, TaskType>> const &task_configs) {
  // push a begin-graph task and a event to launch dependent asks
  {
    EventDesc e(EVENT_LAUNCH_DEPENDENT_TASKS, 1, 0, 0);
    TaskDesc t(TASK_BEGIN_TASK_GRAPH);
    t.trigger_event = get_event_id(my_gpu_id, all_events.size(), false);
    all_tasks.push_back(t);
    all_events.push_back(e);
  }
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
            event_desc_0.event_type = EVENT_LAUNCH_TASKS;
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
            event_desc_1.event_type = EVENT_LAUNCH_TASKS;
            event_desc_1.first_task_id = all_tasks.size();
            event_desc_1.last_task_id = all_tasks.size() + 1;
            event_desc_1.num_triggers = num_gpus - 1;
            all_events.push_back(event_desc_1);
            // Step 2: create a task for reduce
            TaskDesc task(TASK_REDUCE);
            for (int i = 0; i < 2; i++) {
              TensorDesc desc;
              tb::STensor stensor = input_ops[i]->output_tensors[0];
              desc.num_dims = stensor.num_dims;
              desc.data_type = stensor.data_type;
              for (int d = stensor.num_dims - 1; d >= 0; d--) {
                desc.dim[d] = stensor.dim[d];
                desc.stride[d] =
                    (d == stensor.num_dims - 1)
                        ? 1
                        : desc.stride[d + 1] * input_ops[1]->dtensor.dim[d + 1];
              }
              task.inputs[task.num_inputs++] = desc;
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
              }
              task.inputs[task.num_outputs++] = desc;
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
      all_task_maps.emplace(op, cur_task_map);
      continue;
    }
    // Step 1: add all tasks based on their blockIdx
    // (bid.x, bid.y, bid.z) ordering
    for (bid.x = 0; bid.x < bgraph.grid_dim.x; bid.x++) {
      for (bid.y = 0; bid.y < bgraph.grid_dim.y; bid.y++) {
        for (bid.z = 0; bid.z < bgraph.grid_dim.z; bid.z++) {
          TaskDesc task(task_type);
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
            assert(output->output_tensors.size() == 1);
            tb::STensor stensor = output->output_tensors[0];
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
      cur_task_map[dim3(0, 0, 0)] = all_tasks.size();
      all_tasks.push_back(tasks[0]);
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
    all_task_maps.emplace(op, cur_task_map);
  }

  // Update the trigger event for all tasks in pre_task_map
  for (auto const &it : pre_task_map) {
    all_tasks[it.second].trigger_event =
        get_event_id(my_gpu_id, all_events.size(), false /*nvshmem_event*/);
  }
  all_events.push_back(
      EventDesc(EVENT_END_OF_TASK_GRAPH, pre_task_map.size(), 0, 0));

  // Prelaunch all tasks at the begining of an iteration
  all_events[1].first_task_id = 2;
  all_events[1].last_task_id = all_tasks.size();
  for (size_t e = 2; e < all_events.size(); e++) {
    if (all_events[e].event_type == EVENT_LAUNCH_TASKS || all_events[e].event_type == EVENT_LAUNCH_MASSIVE_TASKS) {
      all_events[e].event_type = EVENT_EMPTY;
      for (size_t t = all_events[e].first_task_id; t < all_events[e].last_task_id; t++) {
        all_tasks[t].dependent_event = get_event_id(my_gpu_id, e, false/*nvshmem_event*/);
      }
    }
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

std::string Runtime::print_task_graph(
    mirage::kernel::Graph const &graph,
    std::unordered_map<kn::KNOperator const *,
                       std::tuple<int, int, TaskType>> const &task_configs,
    std::map<mirage::type::GuidType, IODesc> const &io_configs,
    bool use_json_format) {
  using mirage::runtime::IODesc;
  mirage::transpiler::CodeKeeper code;
  mirage::transpiler::CodeKeeper tgbody;
  tgbody.inc_indent();
  code.e("#include \"persistent_kernel.cuh\"");
  if (use_json_format) {
    code.e("#include <nlohmann/json.hpp>");
    code.e("#include <fstream>");
    code.e("#include <filesystem>");
    code.e("using json = nlohmann::json;");
  }
  code.e("size_t get_event_id(int my_gpu_id, size_t event_pos, bool "
         "nvshmem_event) {");
  code.e("size_t event_id = ((static_cast<size_t>(my_gpu_id) << 32) | "
         "event_pos);");
  code.e("if (nvshmem_event) {");
  code.e("event_id = event_id | EVENT_NVSHMEM_TAG;");
  code.e("}");
  code.e("return event_id;");
  code.e("}");
  code.e("");

  // function that loads json file and generates task graph
  if (use_json_format) {
    code.e("void construct_task_graph(int num_gpus,");
    code.e("                          int my_gpu_id,");
    code.e("                          std::vector<TaskDesc> &all_tasks,");
    code.e("                          std::vector<EventDesc> &all_events,");
    code.e("                          std::vector<TaskId> &first_tasks,");
    code.e("                          std::map<std::string, void*> const &all_tensors) {");
    code.e("std::filesystem::path file_path(__FILE__);");
    code.e("std::ifstream json_file(file_path.parent_path().string()+\"/task_graph.json\");");
    code.e("nlohmann::json json_task_graph;");
    code.e("json_file >> json_task_graph;");
    // load tasks
    code.e("for (json const &task : json_task_graph[\"all_tasks\"]) {");
    code.e("TaskDesc task_desc(static_cast<TaskType>(task.at(\"task_type\")));");
    code.e("if (task.at(\"trigger_event\").is_number_integer()) {");
    code.e("task_desc.trigger_event = task.at(\"trigger_event\").get<unsigned long long int>();");
    code.e("}");
    code.e("else {");
    code.e("json j = task.at(\"trigger_event\");");
    code.e("int gpu_offset = j.at(\"gpu_offset\").get<int>();");
    code.e("size_t event_pos = j.at(\"event_pos\").get<size_t>();");
    code.e("bool is_nvshmem = j.at(\"is_nvshmem\").get<bool>();");
    code.e("task_desc.trigger_event = get_event_id((my_gpu_id + gpu_offset) \% num_gpus, event_pos, is_nvshmem);");
    code.e("}");
    code.e("if (task.at(\"dependent_event\").is_number_integer()) {");
    code.e("task_desc.dependent_event = task.at(\"dependent_event\").get<unsigned long long int>();");
    code.e("}");
    code.e("else {");
    code.e("json j = task.at(\"dependent_event\");");
    code.e("int gpu_offset = j.at(\"gpu_offset\").get<int>();");
    code.e("size_t event_pos = j.at(\"event_pos\").get<size_t>();");
    code.e("bool is_nvshmem = j.at(\"is_nvshmem\").get<bool>();");
    code.e("task_desc.dependent_event = get_event_id((my_gpu_id + gpu_offset) \% num_gpus, event_pos, is_nvshmem);");
    code.e("}");

    // load inputs
    code.e("task_desc.num_inputs = 0;");
    code.e("for (json const &tensor : task[\"inputs\"]) {");
    code.e("TensorDesc input;");
    code.e("std::string name = tensor.at(\"base_ptr\").get<std::string>();");
    code.e("assert(all_tensors.find(name) != all_tensors.end());");
    code.e("off_t offset = tensor.at(\"offset\").get<off_t>();");
    code.e("input.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;");
    code.e("assert(tensor.at(\"dims\").size() == tensor.at(\"strides\").size());");
    code.e("input.num_dims = tensor.at(\"dims\").size();");
    code.e("input.data_type = tensor.at(\"data_type\").get<int>();");
    code.e("for (int i = 0; i < input.num_dims; i++) {");
    code.e("input.dim[i] = tensor[\"dims\"][i].get<int>();");
    code.e("input.stride[i] = tensor[\"strides\"][i].get<int>();");
    code.e("}");
    code.e("task_desc.inputs[task_desc.num_inputs++] = input;");
    code.e("}");
    // load outputs
    code.e("task_desc.num_outputs = 0;");
    code.e("for (json const &tensor : task[\"outputs\"]) {");
    code.e("TensorDesc output;");
    code.e("std::string name = tensor.at(\"base_ptr\").get<std::string>();");
    code.e("assert(all_tensors.find(name) != all_tensors.end());");
    code.e("off_t offset = tensor.at(\"offset\").get<off_t>();");
    code.e("output.base_ptr = static_cast<char*>(all_tensors.at(name))+offset;");
    code.e("assert(tensor.at(\"dims\").size() == tensor.at(\"strides\").size());");
    code.e("output.num_dims = tensor.at(\"dims\").size();");
    code.e("output.data_type = tensor.at(\"data_type\").get<int>();");
    code.e("for (int i = 0; i < output.num_dims; i++) {");
    code.e("output.dim[i] = tensor[\"dims\"][i];");
    code.e("output.stride[i] = tensor[\"strides\"][i];");
    code.e("}");
    code.e("task_desc.outputs[task_desc.num_outputs++] = output;");
    code.e("}");
    code.e("all_tasks.push_back(task_desc);");
    code.e("}");
    // load events
    code.e("for (json const &e : json_task_graph[\"all_events\"]) {");
    code.e("EventType event_type = static_cast<EventType>(e.at(\"event_type\").get<int>());");
    code.e("int num_triggers = e.at(\"num_triggers\").get<int>();");
    code.e("int first_task_id = e.at(\"first_task_id\").get<int>();");
    code.e("int last_task_id = e.at(\"last_task_id\").get<int>();");
    code.e("all_events.push_back(EventDesc(event_type, num_triggers, first_task_id, last_task_id));");
    code.e("}");
    // load first tasks
    code.e("for (json const &t : json_task_graph[\"first_tasks\"]) {");
    code.e("first_tasks.push_back(t.get<int>());");
    code.e("}");
    code.e("}");
    code.e("");
  }

  code.e(
      "static void _init_persistent_kernel(std::vector<TaskDesc> &all_tasks,");
  code.e(
      "                                  std::vector<EventDesc> &all_events,");
  code.e("                                  std::vector<TaskId> &first_tasks,");
  code.e("                                  std::vector<void const*> const "
         "&torch_tensors,");
  code.e("                                  int num_gpus,");
  code.e("                                  int my_gpu_id) {");
  code.e("assert(num_gpus = $);", num_gpus);

  int num_torch_tensors = 0;
  if (use_json_format) {
    code.e("std::map<std::string, void*> all_tensors;");
  }
  for (auto const &iter : io_configs) {
    IODesc desc = iter.second;
    switch (desc.type) {
      case IODesc::TorchTensor: {
        code.e("char *$ = (char*) torch_tensors[$];",
               desc.name,
               num_torch_tensors);
        if (use_json_format) {
          code.e("all_tensors[\"$\"] = $;", desc.name, desc.name);
        }
        num_torch_tensors++;
        break;
      }
      case IODesc::FusedTorchTensor: {
        for (auto const &desc : desc.sub_descs) {
          code.e("char *$ = (char*)torch_tensors[$];",
                 desc.name,
                 num_torch_tensors);
          if (use_json_format) {
            code.e("all_tensors[\"$\"] = $;", desc.name, desc.name);
          }
          num_torch_tensors++;
        }
        break;
      }
      case IODesc::CUDAMallocTensor: {
        code.e("void *$;", desc.name);
        size_t size = mirage::type::get_datatype_size(desc.tensor.data_type);
        for (int i = 0; i < desc.tensor.num_dims; i++) {
          size *= desc.tensor.dim[i];
        }
        code.e("cudaMalloc(&$, $);", desc.name, size);
        if (use_json_format) {
          code.e("all_tensors[\"$\"] = $;", desc.name, desc.name);
        }
        break;
      }
      case IODesc::NVSHMEMMallocTensor: {
        size_t size = mirage::type::get_datatype_size(desc.tensor.data_type);
        for (int i = 0; i < desc.tensor.num_dims; i++) {
          size *= desc.tensor.dim[i];
        }
        code.e("void *$ = nvshmem_malloc($);", desc.name, size);
        if (use_json_format) {
          code.e("all_tensors[\"$\"] = $;", desc.name, desc.name);
        }
        break;
      }
      default:
        assert(false);
    }
  }
  json json_task_graph = {
      {"all_tasks", {}}, {"all_events", {}}, {"first_tasks", {}}};
  // generate task[0]
  {
    tgbody.e("all_tasks.push_back(TaskDesc(TASK_TERMINATE));");
    json_task_graph["all_tasks"].push_back(json{
        {"task_type", TASK_TERMINATE},
        {"inputs", {}},
        {"outputs", {}},
        {"trigger_event", EVENT_INVALID_ID},
	{"dependent_event", EVENT_INVALID_ID}});
  }
  // generate task[1]
  {
    tgbody.e("all_tasks.push_back(TaskDesc(TASK_BEGIN_TASK_GRAPH));");
    json_task_graph["all_tasks"].push_back(json{
        {"task_type", TASK_BEGIN_TASK_GRAPH},
        {"inputs", {}},
        {"outputs", {}},
        {"trigger_event", json{{"gpu_offset", 0}, {"event_pos", 1}, {"is_nvshmem", false}}},
	{"dependent_event", EVENT_INVALID_ID}});
  }
  // generate all other tasks
  size_t task_pos = 2;
  for (auto const &op : graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      continue;
    }
    assert(op->op_type == type::KNOperatorType::KN_CUSTOMIZED_OP);
    std::tuple<int, int, TaskType> task_config = task_configs.find(op)->second;

    assert(all_task_maps.find(op) != all_task_maps.end());
    std::map<dim3, TaskId, Dim3Comparator> const &task_map = all_task_maps[op];
    // Customized op
    kn::KNCustomizedOp const *cur_op =
        dynamic_cast<kn::KNCustomizedOp const *>(op);
    tb::Graph const &bgraph = cur_op->bgraph;
    dim3 bid;
    std::vector<tb::TBInputOp *> input_ops;
    std::vector<tb::TBInputOp *> output_ops;
    int num_inputs = std::get<0>(task_config);
    // int num_outputs = std::get<1>(task_config);
    TaskType task_type = std::get<2>(task_config);
    for (auto const &op : bgraph.operators) {
      assert(op->op_type == mirage::type::TB_INPUT_OP);
      if (input_ops.size() < (size_t)num_inputs) {
        input_ops.push_back(static_cast<tb::TBInputOp *>(op));
      } else {
        output_ops.push_back(static_cast<tb::TBInputOp *>(op));
      }
    }

    for (bid.x = 0; bid.x < bgraph.grid_dim.x; bid.x++) {
      for (bid.y = 0; bid.y < bgraph.grid_dim.y; bid.y++) {
        for (bid.z = 0; bid.z < bgraph.grid_dim.z; bid.z++) {
          // To perform allreduce, we first launch (num_gpus-1) tasks for
          // allgather
          if (task_type == TASK_ALLREDUCE) {
            for (int tgt_gpu_id = 0; tgt_gpu_id < num_gpus; tgt_gpu_id++) {
              if (tgt_gpu_id == my_gpu_id) {
                continue;
              }
              TaskDesc task_desc = all_tasks[task_pos];
              assert(task_desc.task_type == TASK_NVSHMEM_COPY);
              tgbody.e("// task[$]", task_pos);
              tgbody.e("{");
              tgbody.e("TaskDesc task_desc(static_cast<TaskType>($));",
                     task_desc.task_type);
              size_t gpu_offset = ((task_desc.trigger_event >> 32) & 0xffff);
              size_t event_pos = (task_desc.trigger_event & 0xffffffff);
              bool is_nvshmem_event =
                  ((task_desc.trigger_event & EVENT_NVSHMEM_TAG) > 0);
              assert(is_nvshmem_event);
              tgbody.e("task_desc.trigger_event = get_event_id((my_gpu_id + $) "
                       "\% num_gpus, $, $);",
                       gpu_offset,
                       event_pos,
                       is_nvshmem_event);
	      assert(task_desc.dependent_event != EVENT_INVALID_ID);
              tgbody.e("task_desc.dependent_event = get_event_id(my_gpu_id, $, false);",
                       (task_desc.dependent_event & 0xffffffff));
              assert(task_desc.num_inputs == 1);
              assert(task_desc.num_outputs == 1);
              json json_task = {{"task_type", task_desc.task_type},
                                {"inputs", {}},
                                {"outputs", {}},
                                {"trigger_event", json{{"gpu_offset", gpu_offset}, {"event_pos", event_pos}, {"is_nvshmem", is_nvshmem_event}}},
                                {"dependent_event", json{{"gpu_offset", 0}, {"event_pos", (task_desc.dependent_event & 0xffffffff)}, {"is_nvshmem", false}}}};
              tgbody.e("task_desc.num_inputs = $;", task_desc.num_inputs);
              tgbody.e("task_desc.num_outputs = $;", task_desc.num_outputs);
              off_t offset = 0;
              // Add input
              int3 input_map = input_ops[0]->input_map;
              IODesc io_desc =
                  io_configs.find(input_ops[0]->dtensor.guid)->second;
              if (input_map.x >= 0) {
                size_t block_size =
                    io_desc.tensor.dim[input_map.x] / bgraph.grid_dim.x;
                offset +=
                    block_size * bid.x * io_desc.tensor.stride[input_map.x];
              }
              if (input_map.y >= 0) {
                size_t block_size =
                    io_desc.tensor.dim[input_map.y] / bgraph.grid_dim.y;
                offset +=
                    block_size * bid.y * io_desc.tensor.stride[input_map.y];
              }
              if (input_map.z >= 0) {
                size_t block_size =
                    io_desc.tensor.dim[input_map.z] / bgraph.grid_dim.z;
                offset +=
                    block_size * bid.z * io_desc.tensor.stride[input_map.z];
              }
              tgbody.e("TensorDesc input$;", 0);
              tgbody.e("input$.base_ptr = static_cast<char*>($) + $;",
                       0,
                       io_desc.name,
                       offset *
                           type::get_datatype_size(io_desc.tensor.data_type));
              tgbody.e("input$.num_dims = $;", 0, task_desc.inputs[0].num_dims);
              tgbody.e("input$.data_type = $;", 0, task_desc.inputs[0].data_type);
              json json_dims = json::array(), json_strides = json::array();
              for (int d = 0; d < task_desc.inputs[0].num_dims; d++) {
                tgbody.e("input$.dim[$] = $;", 0, d, task_desc.inputs[0].dim[d]);
                tgbody.e("input$.stride[$] = $;",
                         0,
                         d,
                         task_desc.inputs[0].stride[d]);
                json_dims.push_back(task_desc.inputs[0].dim[d]);
                json_strides.push_back(task_desc.inputs[0].stride[d]);
              }
              tgbody.e("task_desc.inputs[$] = input$;", 0, 0);
              json_task["inputs"].push_back(json{{"base_ptr", io_desc.name},
                                                 {"offset", offset * type::get_datatype_size(io_desc.tensor.data_type)},
                                                 {"data_type", task_desc.inputs[0].data_type},
                                                 {"dims", json_dims},
                                                 {"strides", json_strides}});
              // Add nvshmem_copy output
              // Note that nvshmem_copy's output is stored in input_ops[1]
              offset = tgt_gpu_id * input_ops[0]->dtensor.num_elements();
              int3 output_map = input_ops[1]->input_map;
              io_desc = io_configs.find(input_ops[1]->dtensor.guid)->second;
              if (output_map.x >= 0) {
                size_t block_size =
                    io_desc.tensor.dim[output_map.x] / bgraph.grid_dim.x;
                offset +=
                    block_size * bid.x * io_desc.tensor.stride[output_map.x];
              }
              if (output_map.y >= 0) {
                size_t block_size =
                    io_desc.tensor.dim[output_map.y] / bgraph.grid_dim.y;
                offset +=
                    block_size * bid.y * io_desc.tensor.stride[output_map.y];
              }
              if (output_map.z >= 0) {
                size_t block_size =
                    io_desc.tensor.dim[output_map.z] / bgraph.grid_dim.z;
                offset +=
                    block_size * bid.z * io_desc.tensor.stride[output_map.z];
              }
              tgbody.e("TensorDesc output$;", 0);
              tgbody.e("output$.base_ptr = static_cast<char*>($) + $;",
                       0,
                       io_desc.name,
                       offset *
                           type::get_datatype_size(io_desc.tensor.data_type));
              tgbody.e("output$.num_dims = $;", 0, task_desc.outputs[0].num_dims);
              tgbody.e(
                  "output$.data_type = $;", 0, task_desc.outputs[0].data_type);
              json_dims = json::array();
              json_strides = json::array();
              for (int d = 0; d < task_desc.outputs[0].num_dims; d++) {
                tgbody.e(
                    "output$.dim[$] = $;", 0, d, task_desc.outputs[0].dim[d]);
                tgbody.e("output$.stride[$] = $;",
                         0,
                         d,
                         task_desc.outputs[0].stride[d]);
                json_dims.push_back(task_desc.outputs[0].dim[d]);
                json_strides.push_back(task_desc.outputs[0].stride[d]);
              }
              tgbody.e("task_desc.outputs[$] = output$;", 0, 0);
              json_task["outputs"].push_back(json{{"base_ptr", io_desc.name},
                                                  {"offset", offset * type::get_datatype_size(io_desc.tensor.data_type)},
                                                  {"data_type", task_desc.outputs[0].data_type},
                                                  {"dims", json_dims},
                                                  {"strides", json_strides}});
              tgbody.e("all_tasks.push_back(task_desc);");
              json_task_graph["all_tasks"].push_back(json_task);
              tgbody.e("}");
              task_pos++;
            }
          }
          TaskId task_id = task_map.at(bid);
          TaskDesc task_desc = all_tasks[task_pos];
          assert(task_desc.task_type == task_type ||
                 task_type == TASK_ALLREDUCE);
          assert(task_pos == (task_id & 0xffffffff));
          tgbody.e("// task[$]", task_pos);
          tgbody.e("{");
          tgbody.e("TaskDesc task_desc(static_cast<TaskType>($));",
                 task_desc.task_type);
          size_t gpu_offset = ((task_desc.trigger_event >> 32) & 0xffff);
          size_t event_pos = (task_desc.trigger_event & 0xffffffff);
          bool is_nvshmem_event =
              ((task_desc.trigger_event & EVENT_NVSHMEM_TAG) > 0);
          assert(gpu_offset == 0);
          assert(!is_nvshmem_event);
          tgbody.e("task_desc.trigger_event = get_event_id(my_gpu_id, $, $);",
                 event_pos,
                 is_nvshmem_event);
          json json_task;
	  if (task_desc.dependent_event == EVENT_INVALID_ID) {
            tgbody.e("task_desc.dependent_event = EVENT_INVALID_ID;");
            json_task = {{"task_type", task_desc.task_type},
                         {"inputs", {}},
                         {"outputs", {}},
                         {"trigger_event", json{{"gpu_offset", gpu_offset}, {"event_pos", event_pos}, {"is_nvshmem", is_nvshmem_event}}},
                         {"dependent_event", EVENT_INVALID_ID}};
          } else {
            tgbody.e("task_desc.dependent_event = get_event_id(my_gpu_id, $, false);",
                   (task_desc.dependent_event & 0xffffffff));
            json_task = {{"task_type", task_desc.task_type},
                         {"inputs", {}},
                         {"outputs", {}},
                         {"trigger_event", json{{"gpu_offset", gpu_offset}, {"event_pos", event_pos}, {"is_nvshmem", is_nvshmem_event}}},
                         {"dependent_event", json{{"gpu_offset", 0}, {"event_pos", (task_desc.dependent_event & 0xffffffff)}, {"is_nvshmem", false}}}};
          }
          tgbody.e("task_desc.num_inputs = $;", task_desc.num_inputs);
          tgbody.e("task_desc.num_outputs = $;", task_desc.num_outputs);
          for (int i = 0; i < task_desc.num_inputs; i++) {
            off_t offset = 0;
            int num_dims = input_ops[i]->dtensor.num_dims;
            int3 input_map = input_ops[i]->input_map;
            IODesc io_desc =
                io_configs.find(input_ops[i]->dtensor.guid)->second;
            assert(input_ops[i]->dtensor.owner_op->op_type ==
                   type::KN_INPUT_OP);
            if (io_desc.type == IODesc::FusedTorchTensor) {
              // Currently assert that we fuse the last dim (i.e.,num_dims - 1)
              int fused_group_size = 0;
              std::vector<int> group_sizes;
              for (auto const &sub_desc : io_desc.sub_descs) {
                assert(sub_desc.tensor.num_dims == num_dims);
                assert(sub_desc.tensor.dim[num_dims - 1] % io_desc.num_groups ==
                       0);
                int my_group_size =
                    sub_desc.tensor.dim[num_dims - 1] / io_desc.num_groups;
                fused_group_size += my_group_size;
                group_sizes.push_back(my_group_size);
              }
              assert(io_desc.tensor.dim[num_dims - 1] ==
                     fused_group_size * io_desc.num_groups);
              assert(io_desc.tensor.num_dims == num_dims);
              int fused_dim_off = 0;
              if (input_map.x == num_dims - 1) {
                fused_dim_off = io_desc.tensor.dim[num_dims - 1] /
                                bgraph.grid_dim.x * bid.x;
              }
              if (input_map.y == num_dims - 1) {
                fused_dim_off = io_desc.tensor.dim[num_dims - 1] /
                                bgraph.grid_dim.y * bid.y;
              }
              if (input_map.z == num_dims - 1) {
                fused_dim_off = io_desc.tensor.dim[num_dims - 1] /
                                bgraph.grid_dim.z * bid.z;
              }
              int fused_dim_off_in_group = fused_dim_off % fused_group_size;
              size_t index = 0;
              while (index < group_sizes.size()) {
                if (fused_dim_off_in_group >= group_sizes[index]) {
                  fused_dim_off_in_group -= group_sizes[index];
                  index++;
                } else {
                  break;
                }
              }
              IODesc sub_desc = io_desc.sub_descs[index];
              int fused_dim_off_subtensor =
                  fused_dim_off / fused_group_size * group_sizes[index] +
                  fused_dim_off_in_group;
              // Assert that it is within range
              assert(fused_dim_off_subtensor <
                     sub_desc.tensor.dim[num_dims - 1]);
              if (input_map.x >= 0 && input_map.x != num_dims - 1) {
                size_t block_size =
                    sub_desc.tensor.dim[input_map.x] / bgraph.grid_dim.x;
                offset +=
                    block_size * bid.x * sub_desc.tensor.stride[input_map.x];
              } else if (input_map.x == num_dims - 1) {
                offset += fused_dim_off_subtensor *
                          sub_desc.tensor.stride[input_map.x];
              }
              if (input_map.y >= 0 && input_map.y != num_dims - 1) {
                size_t block_size =
                    sub_desc.tensor.dim[input_map.y] / bgraph.grid_dim.y;
                offset +=
                    block_size * bid.y * sub_desc.tensor.stride[input_map.y];
              } else if (input_map.y == num_dims - 1) {
                offset += fused_dim_off_subtensor *
                          sub_desc.tensor.stride[input_map.y];
              }
              if (input_map.z >= 0 && input_map.z != num_dims - 1) {
                size_t block_size =
                    sub_desc.tensor.dim[input_map.z] / bgraph.grid_dim.z;
                offset +=
                    block_size * bid.z * sub_desc.tensor.stride[input_map.z];
              } else if (input_map.z == num_dims - 1) {
                offset += fused_dim_off_subtensor *
                          sub_desc.tensor.stride[input_map.z];
              }
              tgbody.e("TensorDesc input$;", i);
              tgbody.e("input$.base_ptr = static_cast<char*>($) + $;",
                     i,
                     sub_desc.name,
                     offset *
                         type::get_datatype_size(sub_desc.tensor.data_type));
              tgbody.e("input$.num_dims = $;", i, task_desc.inputs[i].num_dims);
              tgbody.e("input$.data_type = $;", i, task_desc.inputs[i].data_type);
              json json_dims = json::array();
              json json_strides = json::array();
              for (int d = 0; d < task_desc.inputs[i].num_dims; d++) {
                tgbody.e("input$.dim[$] = $;", i, d, task_desc.inputs[i].dim[d]);
                tgbody.e("input$.stride[$] = $;",
                         i,
                         d,
                         sub_desc.tensor.stride[d]);
                json_dims.push_back(task_desc.inputs[i].dim[d]);
                json_strides.push_back(sub_desc.tensor.stride[d]);
              }
              tgbody.e("task_desc.inputs[$] = input$;", i, i);
              json_task["inputs"].push_back(
                  json{{"base_ptr", sub_desc.name},
                       {"offset", offset * type::get_datatype_size(sub_desc.tensor.data_type)},
                       {"data_type", task_desc.inputs[i].data_type},
                       {"dims", json_dims},
                       {"strides", json_strides}});
            } else {
              // Non-fused case, use io_desc
              if (input_map.x >= 0) {
                size_t block_size =
                    io_desc.tensor.dim[input_map.x] / bgraph.grid_dim.x;
                offset +=
                    block_size * bid.x * io_desc.tensor.stride[input_map.x];
              }
              if (input_map.y >= 0) {
                size_t block_size =
                    io_desc.tensor.dim[input_map.y] / bgraph.grid_dim.y;
                offset +=
                    block_size * bid.y * io_desc.tensor.stride[input_map.y];
              }
              if (input_map.z >= 0) {
                size_t block_size =
                    io_desc.tensor.dim[input_map.z] / bgraph.grid_dim.z;
                offset +=
                    block_size * bid.z * io_desc.tensor.stride[input_map.z];
              }
              tgbody.e("TensorDesc input$;", i);
              tgbody.e("input$.base_ptr = static_cast<char*>($) + $;",
                     i,
                     io_desc.name,
                     offset *
                         type::get_datatype_size(io_desc.tensor.data_type));
              tgbody.e("input$.num_dims = $;", i, task_desc.inputs[i].num_dims);
              tgbody.e("input$.data_type = $;", i, task_desc.inputs[i].data_type);
              json json_dims = json::array();
              json json_strides = json::array();
              for (int d = 0; d < task_desc.inputs[i].num_dims; d++) {
                tgbody.e("input$.dim[$] = $;", i, d, task_desc.inputs[i].dim[d]);
                tgbody.e("input$.stride[$] = $;",
                       i,
                       d,
                       task_desc.inputs[i].stride[d]);
                json_dims.push_back(task_desc.inputs[i].dim[d]);
                json_strides.push_back(task_desc.inputs[i].stride[d]);
              }
              tgbody.e("task_desc.inputs[$] = input$;", i, i);
              json_task["inputs"].push_back(
                  json{{"base_ptr", io_desc.name},
                       {"offset", offset * type::get_datatype_size(io_desc.tensor.data_type)},
                       {"data_type", task_desc.inputs[i].data_type},
                       {"dims", json_dims},
                       {"strides", json_strides}});
            }
          }
          for (int i = 0; i < task_desc.num_outputs; i++) {
            off_t offset = 0;
            int3 output_map = output_ops[i]->input_map;
            IODesc io_desc =
                io_configs.find(output_ops[i]->dtensor.guid)->second;
            assert(io_desc.type != IODesc::FusedTorchTensor);
            if (output_map.x >= 0) {
              size_t block_size =
                  io_desc.tensor.dim[output_map.x] / bgraph.grid_dim.x;
              offset +=
                  block_size * bid.x * io_desc.tensor.stride[output_map.x];
            }
            if (output_map.y >= 0) {
              size_t block_size =
                  io_desc.tensor.dim[output_map.y] / bgraph.grid_dim.y;
              offset +=
                  block_size * bid.y * io_desc.tensor.stride[output_map.y];
            }
            if (output_map.z >= 0) {
              size_t block_size =
                  io_desc.tensor.dim[output_map.z] / bgraph.grid_dim.z;
              offset +=
                  block_size * bid.z * io_desc.tensor.stride[output_map.z];
            }

            tgbody.e("TensorDesc output$;", i);
            tgbody.e("output$.base_ptr = static_cast<char*>($) + $;",
                   i,
                   io_desc.name,
                   offset * type::get_datatype_size(io_desc.tensor.data_type));
            tgbody.e("output$.num_dims = $;", i, task_desc.outputs[i].num_dims);
            tgbody.e("output$.data_type = $;", i, task_desc.outputs[i].data_type);
            json json_dims = json::array();
            json json_strides = json::array();
            for (int d = 0; d < task_desc.outputs[i].num_dims; d++) {
              tgbody.e("output$.dim[$] = $;", i, d, task_desc.outputs[i].dim[d]);
              tgbody.e("output$.stride[$] = $;",
                     i,
                     d,
                     task_desc.outputs[i].stride[d]);
              json_dims.push_back(task_desc.outputs[i].dim[d]);
              json_strides.push_back(task_desc.outputs[i].stride[d]);
            }
            tgbody.e("task_desc.outputs[$] = output$;", i, i);
              json_task["outputs"].push_back(
                  json{{"base_ptr", io_desc.name},
                       {"offset", offset * type::get_datatype_size(io_desc.tensor.data_type)},
                       {"data_type", task_desc.outputs[i].data_type},
                       {"dims", json_dims},
                       {"strides", json_strides}});
          }
          tgbody.e("all_tasks.push_back(task_desc);");
          tgbody.e("}");
          json_task_graph["all_tasks"].push_back(json_task);
          task_pos++;
        }
      }
    }
  }
  assert(task_pos == all_tasks.size());
  // Add all events
  for (auto const &event : all_events) {
    tgbody.e(
        "all_events.push_back(EventDesc(static_cast<EventType>($), $, $, $));",
        event.event_type,
        event.num_triggers,
        event.first_task_id,
        event.last_task_id);
    json_task_graph["all_events"].push_back(json{{"event_type", event.event_type}, {"num_triggers", event.num_triggers}, {"first_task_id", event.first_task_id}, {"last_task_id", event.last_task_id}});
  }
  // Add first task
  for (auto const &task : first_tasks) {
    tgbody.e("first_tasks.push_back($);", task);
    json_task_graph["first_tasks"].push_back(task);
  }
  if (use_json_format) {
    code.e("construct_task_graph(num_gpus, my_gpu_id, all_tasks, all_events, first_tasks, all_tensors);");
  } else {
    code.e(tgbody.to_string());
  }
  code.e("}");
  // Write json to output file
  std::ofstream out("task_graph.json");
  out << json_task_graph.dump(2);
  out.close();
  return code.to_string();
}

} // namespace runtime
} // namespace mirage
