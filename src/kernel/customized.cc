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

#include "mirage/kernel/customized.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/reduction.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/utils/hash_utils.h"
#include <cassert>

namespace mirage {
namespace kernel {

using mirage::threadblock::ExecutionPlan;
using mirage::threadblock::STensor;

std::vector<DTensor> Graph::customized(std::vector<DTensor> const &inputs,
                                       ExecutionPlan const &plan) {
  KNOperator *op = create_customized_op(inputs, plan);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors;
}

KNOperator *Graph::create_customized_op(std::vector<DTensor> const &inputs,
                                        ExecutionPlan const &plan) {
  KNCustomizedOp *op = new KNCustomizedOp(inputs, plan);
  return op;
}

KNOperator *Graph::create_customized_op(std::vector<DTensor> const &inputs,
                                        threadblock::Graph const &_graph) {
  size_t output_size = 0;
  for (threadblock::TBOperator *op : _graph.operators) {
    if (op->op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      output_size +=
          static_cast<threadblock::TBOutputOp *>(op)->dtensor.data_size();
    }
  }

  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  if (dmm->offset + output_size > dmm->total_size) {
    return nullptr;
  }

  KNCustomizedOp *op = new KNCustomizedOp(inputs, _graph);
  return op;
}

KNCustomizedOp::KNCustomizedOp(std::vector<DTensor> const &_inputs,
                               ExecutionPlan const &_plan)
    : KNOperator(mirage::type::KN_CUSTOMIZED_OP, _inputs), plan(_plan),
      bgraph(_plan.grid_dim, _plan.block_dim, _plan.forloop_range, _plan.reduction_dimx) {
  assert(_inputs.size() == plan.input_map.size());
  assert(plan.forloop_dim.size() == plan.input_map.size());
  assert(plan.input_smem_layouts.size() == plan.input_map.size());
  // Step 1: computing input shapes
  // Step 1: creating a stensor for each input
  for (size_t i = 0; i < input_tensors.size(); i++) {
    bgraph.new_input(input_tensors[i],
                     plan.input_map[i],
                     plan.forloop_dim[i],
                     plan.input_smem_layouts[i]);
  }

  auto const &ops = plan.ops;
  for (auto const &op : ops) {
    std::vector<STensor> my_inputs;
    for (auto const &idx : op.second) {
      // assert(bgraph.tensors.find(idx) != bgraph.tensors.end());
      // my_inputs.push_back(bgraph.tensors[idx]);
      assert((int)bgraph.operators.size() > idx.first);
      assert((int)bgraph.operators[idx.first]->output_tensors.size() >
             idx.second);
      my_inputs.push_back(
          bgraph.operators[idx.first]->output_tensors[idx.second]);
    }
    switch (op.first) {
      case mirage::type::TB_MATMUL_OP: {
        assert(my_inputs.size() == 2);
        bgraph.matmul(my_inputs[0], my_inputs[1]);
        break;
      }
      case mirage::type::TB_EXP_OP: {
        assert(my_inputs.size() == 1);
        bgraph.exp(my_inputs[0]);
        break;
      }
      case mirage::type::TB_ADD_OP: {
        assert(my_inputs.size() == 2);
        bgraph.add(my_inputs[0], my_inputs[1]);
        break;
      }
      case mirage::type::TB_DIV_OP: {
        assert(my_inputs.size() == 2);
        bgraph.div(my_inputs[0], my_inputs[1]);
        break;
      }
      case mirage::type::TB_REDUCTION_0_OP:
      case mirage::type::TB_REDUCTION_1_OP:
      case mirage::type::TB_REDUCTION_2_OP: {
        assert(my_inputs.size() == 1);
        int reduce_dim = op.first - mirage::type::TB_REDUCTION_0_OP;
        bgraph.reduction(my_inputs[0], reduce_dim);
        break;
      }
      case mirage::type::TB_REDUCTION_0_TO_DIMX_OP:
      case mirage::type::TB_REDUCTION_1_TO_DIMX_OP:
      case mirage::type::TB_REDUCTION_2_TO_DIMX_OP: {
        assert(my_inputs.size() == 1);
        int reduce_dim = op.first - mirage::type::TB_REDUCTION_0_TO_DIMX_OP;
        bgraph.reduction_to_dimx(my_inputs[0], reduce_dim);
        break;
      }
      case mirage::type::TB_CONCAT_0_OP:
      case mirage::type::TB_CONCAT_1_OP:
      case mirage::type::TB_CONCAT_2_OP: {
        assert(my_inputs.size() == 2);
        int concat_dim = op.first - mirage::type::TB_CONCAT_0_OP;
        bgraph.concat(my_inputs[0], my_inputs[1], concat_dim);
        break;
      }
      default: {
        assert(false && "Unsupported kernel operator");
      }
    }
  }

  assert(output_tensors.size() == 0);
  // Identify outputs: a tensor is an output if it is not used by
  // any other operators
  size_t num_operators = bgraph.operators.size();
  for (size_t op1_idx = 0; op1_idx < num_operators; op1_idx++) {
    mirage::threadblock::TBOperator const *op = bgraph.operators[op1_idx];
    if (op->op_type == mirage::type::TB_INPUT_OP) {
      // Skip input loader
      continue;
    }
    for (size_t i = 0; i < op->output_tensors.size(); i++) {
      bool found = false;
      for (size_t op2_idx = op1_idx + 1; op2_idx < num_operators; op2_idx++) {
        mirage::threadblock::TBOperator const *op2 = bgraph.operators[op2_idx];
        for (size_t j = 0; j < op2->input_tensors.size(); j++) {
          if (op2->input_tensors[j] == op->output_tensors[i]) {
            found = true;
          }
        }
      }
      if (!found) {
        // TODO: change output tensor_shape
        STensor stensor = op->output_tensors[i];
        DTensor dtensor = bgraph.new_output(stensor, plan.output_map);
        // printf("stensor.offset(%d)\n", stensor.smem_offset);
        dtensor.owner_op = this;
        dtensor.owner_ts_idx = static_cast<int>(output_tensors.size());
        dtensor.guid = DTensor::next_guid++;
        DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
        dmm->allocate(dtensor);
        // Update dtensor saved by the output operator
        {
          assert(bgraph.operators.back()->op_type == mirage::type::TB_OUTPUT_OP);
          mirage::threadblock::TBOutputOp *output =
              static_cast<mirage::threadblock::TBOutputOp *>(
                  bgraph.operators.back());
          output->dtensor = dtensor;
        }
        output_tensors.push_back(dtensor);
      }
    }
  }
}

KNCustomizedOp::KNCustomizedOp(std::vector<DTensor> const &_inputs,
                               mirage::threadblock::Graph const &_graph)
    : KNOperator(mirage::type::KN_CUSTOMIZED_OP, _inputs),
      bgraph(_graph.grid_dim, _graph.block_dim, _graph.forloop_range, _graph.reduction_dimx) {
  plan.grid_dim = _graph.grid_dim;
  plan.block_dim = _graph.block_dim;
  plan.forloop_range = _graph.forloop_range;
  plan.reduction_dimx = _graph.reduction_dimx;

  for (auto const &op : _graph.operators) {
    std::vector<STensor> my_inputs;
    std::vector<std::pair<int, int>> indices;
    for (size_t i = 0; i < op->input_tensors.size(); i++) {
      int op_idx = -1, ts_idx = op->input_tensors[i].owner_ts_idx;
      for (size_t l = 0; l < _graph.operators.size(); l++) {
        if (_graph.operators[l] == op->input_tensors[i].owner_op) {
          assert(op_idx == -1);
          op_idx = static_cast<int>(l);
        }
      }
      assert(op_idx != -1);
      my_inputs.push_back(bgraph.operators[op_idx]->output_tensors[ts_idx]);
      indices.push_back({op_idx, ts_idx});
    }
    if (op->op_type != mirage::type::TB_INPUT_OP &&
        op->op_type != mirage::type::TB_OUTPUT_OP) {
      plan.ops.push_back({op->op_type, indices});
    }
    switch (op->op_type) {
      case mirage::type::TB_INPUT_OP: {
        assert(my_inputs.size() == 0);
        mirage::threadblock::TBInputOp *input_op =
            static_cast<mirage::threadblock::TBInputOp *>(op);
        bgraph.new_input(input_op->dtensor,
                         input_op->input_map,
                         input_op->forloop_dim,
                         input_op->output_tensors[0].layout);
        plan.input_map.push_back(input_op->input_map);
        plan.forloop_dim.push_back(input_op->forloop_dim);
        plan.input_smem_layouts.push_back(input_op->output_tensors[0].layout);
        break;
      }
      case mirage::type::TB_OUTPUT_OP: {
        assert(my_inputs.size() == 1);
        mirage::threadblock::TBOutputOp *output_op =
            static_cast<mirage::threadblock::TBOutputOp *>(op);
        DTensor dtensor =
            bgraph.new_output(my_inputs[0], output_op->output_map);
        dtensor.owner_op = this;
        dtensor.owner_ts_idx = static_cast<int>(output_tensors.size());
        dtensor.guid = DTensor::next_guid++;
        DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
        dmm->allocate(dtensor);
        // Update dtensor saved by the output operator
        {
          assert(bgraph.operators.back()->op_type == mirage::type::TB_OUTPUT_OP);
          mirage::threadblock::TBOutputOp *output =
              static_cast<mirage::threadblock::TBOutputOp *>(
                  bgraph.operators.back());
          output->dtensor = dtensor;
        }
        output_tensors.push_back(dtensor);
        plan.output_map = output_op->output_map;
        break;
      }
      case mirage::type::TB_MATMUL_OP: {
        assert(my_inputs.size() == 2);
        bgraph.matmul(my_inputs[0], my_inputs[1]);
        break;
      }
      case mirage::type::TB_EXP_OP: {
        assert(my_inputs.size() == 1);
        bgraph.exp(my_inputs[0]);
        break;
      }
      case mirage::type::TB_ADD_OP: {
        assert(my_inputs.size() == 2);
        bgraph.add(my_inputs[0], my_inputs[1]);
        break;
      }
      case mirage::type::TB_DIV_OP: {
        assert(my_inputs.size() == 2);
        bgraph.div(my_inputs[0], my_inputs[1]);
        break;
      }
      case mirage::type::TB_REDUCTION_0_OP:
      case mirage::type::TB_REDUCTION_1_OP:
      case mirage::type::TB_REDUCTION_2_OP: {
        assert(my_inputs.size() == 1);
        int reduce_dim = op->op_type - mirage::type::TB_REDUCTION_0_OP;
        bgraph.reduction(my_inputs[0], reduce_dim);
        break;
      }
      case mirage::type::TB_REDUCTION_0_TO_DIMX_OP:
      case mirage::type::TB_REDUCTION_1_TO_DIMX_OP:
      case mirage::type::TB_REDUCTION_2_TO_DIMX_OP: {
        assert(my_inputs.size() == 1);
        int reduce_dim = op->op_type - mirage::type::TB_REDUCTION_0_TO_DIMX_OP;
        bgraph.reduction_to_dimx(my_inputs[0], reduce_dim);
        break;
      }
      case mirage::type::TB_CONCAT_0_OP:
      case mirage::type::TB_CONCAT_1_OP:
      case mirage::type::TB_CONCAT_2_OP: {
        assert(my_inputs.size() == 2);
        int concat_dim = op->op_type - mirage::type::TB_CONCAT_FIRST_OP_ID;
        bgraph.concat(my_inputs[0], my_inputs[1], concat_dim);
        break;
      }
      default: {
        assert(false && "Unsupported kernel operator");
      }
    }
  }
}

KNCustomizedOp::~KNCustomizedOp() {
  while (!bgraph.operators.empty()) {
    delete bgraph.operators.back();
    bgraph.operators.pop_back();
  }
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    dmm->free(output_tensors[i]);
  }
}

KNCustomizedOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"bgraph", bgraph},
              {"plan", plan}};
}

} // namespace kernel
} // namespace mirage
