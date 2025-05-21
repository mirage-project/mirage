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
#include "mirage/threadblock/element_unary.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/reduction.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/utils/hash_utils.h"
#include <cassert>

namespace mirage {
namespace kernel {

using mirage::threadblock::STensor;

std::vector<DTensor> Graph::customized(std::vector<DTensor> const &inputs,
                                       threadblock::Graph const &bgraph) {
  KNOperator *op = create_customized_op(inputs, bgraph);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors;
}

int Graph::customized(std::vector<DTensor const *> _inputs,
                      DTensor **outputs,
                      mirage::threadblock::Graph const *bgraph) {
  std::vector<DTensor> inputs;
  for (auto const &t : _inputs) {
    inputs.push_back(*t);
  }
  KNOperator *op = create_customized_op(inputs, *bgraph);
  assert(op != nullptr);
  operators.push_back(op);
  for (size_t i = 0; i < op->output_tensors.size(); i++) {
    outputs[i] = &op->output_tensors[i];
  }
  return op->output_tensors.size();
}

KNOperator *Graph::create_customized_op(std::vector<DTensor> const &inputs,
                                        threadblock::Graph const &_graph) {
  // Assert that _graph's dtensor inputs align with inputs
  {
    int num_inputs = 0;
    for (auto const &op : _graph.operators) {
      if (op->op_type == mirage::type::TB_INPUT_OP) {
        mirage::threadblock::TBInputOp const *input_op =
            static_cast<mirage::threadblock::TBInputOp const *>(op);
        assert(inputs[num_inputs] == input_op->dtensor);
        num_inputs++;
      }
    }
    assert(num_inputs == (int)inputs.size());
  }
  // Calculate fingerprint sizes
  size_t output_data_size = 0, output_fp_size = 0;
  for (threadblock::TBOperator *op : _graph.operators) {
    if (op->op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      output_data_size +=
          static_cast<threadblock::TBOutputOp *>(op)->dtensor.data_size();
      output_fp_size += static_cast<threadblock::TBOutputOp *>(op)
                            ->dtensor.fingerprint_size();
    }
  }

  if (!can_allocate(output_data_size, output_fp_size)) {
    return nullptr;
  }

  KNCustomizedOp *op = new KNCustomizedOp(this, inputs, _graph);
  return op;
}

KNCustomizedOp::KNCustomizedOp(mirage::kernel::Graph *_kgraph,
                               std::vector<DTensor> const &_inputs,
                               mirage::threadblock::Graph const &_graph)
    : KNOperator(_kgraph, mirage::type::KN_CUSTOMIZED_OP, _inputs),
      bgraph(_graph.grid_dim,
             _graph.block_dim,
             _graph.forloop_range,
             _graph.reduction_dimx) {
  // plan.grid_dim = _graph.grid_dim;
  // plan.block_dim = _graph.block_dim;
  // plan.forloop_range = _graph.forloop_range;
  // plan.reduction_dimx = _graph.reduction_dimx;
  size_t input_idx = 0;
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
      // plan.ops.push_back({op->op_type, indices});
    }
    switch (op->op_type) {
      case mirage::type::TB_INPUT_OP: {
        assert(my_inputs.size() == 0);
        mirage::threadblock::TBInputOp *input_op =
            static_cast<mirage::threadblock::TBInputOp *>(op);
        DTensor const &dtensor = _inputs[input_idx++];
        bgraph.new_input(dtensor,
                         input_op->input_map,
                         input_op->forloop_dim,
                         input_op->output_tensors[0].layout);
        // plan.input_map.push_back(input_op->input_map);
        // plan.input_forloop_dim.push_back(input_op->forloop_dim);
        // plan.input_smem_layouts.push_back(input_op->output_tensors[0].layout);
        break;
      }
      case mirage::type::TB_OUTPUT_OP: {
        assert(my_inputs.size() == 1);
        mirage::threadblock::TBOutputOp *output_op =
            static_cast<mirage::threadblock::TBOutputOp *>(op);
        DTensor dtensor = bgraph.mark_output(my_inputs[0],
                                             output_op->output_map,
                                             output_op->forloop_dim,
                                             output_op->epilogue);
        dtensor.owner_op = this;
        dtensor.owner_ts_idx = static_cast<int>(output_tensors.size());
        dtensor.guid = DTensor::next_guid++;
        // DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
        // dmm->allocate(dtensor);
        kgraph->allocate(dtensor);
        // Update dtensor saved by the output operator
        {
          assert(bgraph.operators.back()->op_type ==
                 mirage::type::TB_OUTPUT_OP);
          mirage::threadblock::TBOutputOp *output =
              static_cast<mirage::threadblock::TBOutputOp *>(
                  bgraph.operators.back());
          output->dtensor = dtensor;
        }
        output_tensors.push_back(dtensor);
        // plan.output_map = output_op->output_map;
        break;
      }
      case mirage::type::TB_MATMUL_OP: {
        assert(my_inputs.size() == 2);
        bgraph.matmul(my_inputs[0], my_inputs[1]);
        break;
      }
      case mirage::type::TB_EXP_OP:
      case mirage::type::TB_SQUARE_OP:
      case mirage::type::TB_SQRT_OP:
      case mirage::type::TB_SILU_OP:
      case mirage::type::TB_GELU_OP:
      case mirage::type::TB_RELU_OP:
      case mirage::type::TB_CLAMP_OP:
      case mirage::type::TB_MUL_SCALAR_OP: {
        assert(my_inputs.size() == 1);
        mirage::threadblock::TBElementUnaryOp const *cur_op =
            dynamic_cast<mirage::threadblock::TBElementUnaryOp const *>(op);
        bgraph.elementunary(my_inputs[0], cur_op->op_type, cur_op->scalar);
        break;
      }
      case mirage::type::TB_ADD_OP:
      case mirage::type::TB_MUL_OP:
      case mirage::type::TB_DIV_OP:
      case mirage::type::TB_SUB_OP:
      case mirage::type::TB_POW_OP: {
        assert(my_inputs.size() == 2);
        bgraph.elementbinary(my_inputs[0], my_inputs[1], op->op_type);
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
      case mirage::type::TB_REDUCTION_0_MAX_OP:
      case mirage::type::TB_REDUCTION_1_MAX_OP:
      case mirage::type::TB_REDUCTION_2_MAX_OP: {
        assert(my_inputs.size() == 1);
        int reduce_dim = op->op_type - mirage::type::TB_REDUCTION_0_MAX_OP;
        bgraph.reduction_max(my_inputs[0], reduce_dim);
        break;
      }
      case mirage::type::TB_RMS_NORM_OP: {
        assert(my_inputs.size() == 1);
        bgraph.rms_norm(my_inputs[0]);
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
      case mirage::type::TB_FORLOOP_ACCUM_NO_RED_OP:
      case mirage::type::TB_FORLOOP_ACCUM_RED_LD_SUM_OP:
      case mirage::type::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP:
      case mirage::type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP:
      case mirage::type::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP: {
        assert(my_inputs.size() == 1);
        bgraph.forloop_accum(my_inputs[0], op->op_type);
        break;
      }
      case mirage::type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP:
      case mirage::type::TB_FORLOOP_ACCUM_RED_LD_SUM_RESCALE_OP: {
        assert(my_inputs.size() == 2);
        bgraph.forloop_accum_rescale(my_inputs[0], my_inputs[1], op->op_type);
        break;
      }
      case mirage::type::TB_FORLOOP_ACCUM_MAX_OP: {
        assert(my_inputs.size() == 1);
        bgraph.forloop_accum_max(my_inputs[0]);
        break;
      }
      default: {
        assert(false && "Unsupported threadblock operator");
      }
    }
  }
}

void KNCustomizedOp::get_bgraph(mirage::threadblock::Graph **bgraph_) {
  *bgraph_ = &(this->bgraph);
}

KNCustomizedOp::~KNCustomizedOp() {
  // while (!bgraph.operators.empty()) {
  //   delete bgraph.operators.back();
  //   bgraph.operators.pop_back();
  // }
  // DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    kgraph->free(output_tensors[i]);
  }
}

KNCustomizedOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"bgraph", bgraph}};
}

size_t KNCustomizedOp::get_owner_independent_hash() const {
  assert(false && "To be implemented");
}

} // namespace kernel
} // namespace mirage
