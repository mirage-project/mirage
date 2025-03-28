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

#include "mirage/transpiler/transpile.h"
#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"
#include "mirage/transpiler/transpiler.h"
#include <cassert>

namespace mirage {
namespace transpiler {

template <typename DT>
DT get_tensor_in_new_graph(std::unordered_map<size_t, DT> mapping,
                           DT const &tensor_in_old_graph) {
  assert(mapping.find(tensor_in_old_graph.guid) != mapping.end());
  return mapping[tensor_in_old_graph.guid];
}

Transpiler::Transpiler(kernel::Graph const *_graph,
                       TranspilerConfig const &_config,
                       vector<vector<size_t>> const &_input_strides)
    : config(_config), input_strides(_input_strides) {
  // Currently we only support GPUs with compute capability >= 8.0 (A100+)
  // TODO(intlsy): Support older GPUs
  if (config.target_cc < GPU_CC::A100) {
    throw std::runtime_error("Unsupported target compute capability");
  }

  // using mirage::type namespace to simplify code
  using namespace mirage::type;
  // We need to construct a new kernel graph by decomposing forloop accumulators
  // into the non-reduction accumulator type to enable transpiler optimizations
  g = std::make_shared<kernel::Graph>();
  std::unordered_map<size_t, kernel::DTensor> dtensor_mapping;

  int input_dtensor_idx = 0;
  for (auto const &op : _graph->operators) {
    // Preparing dtensors in the new graph
    std::vector<kernel::DTensor> dtensor_inputs;
    for (auto const &t : op->input_tensors) {
      dtensor_inputs.push_back(get_tensor_in_new_graph(dtensor_mapping, t));
    }
    switch (op->op_type) {
      case KN_INPUT_OP: {
        // Assert that an input op has exactly one output dtensor
        kernel::KNInputOp *input_op = static_cast<kernel::KNInputOp *>(op);
        assert(op->output_tensors.size() == 1);
        kernel::DTensor const &dtensor = op->output_tensors[0];
        std::vector<int> dims;
        for (int i = 0; i < dtensor.num_dims; i++) {
          dims.push_back(dtensor.dim[i]);
        }
        // Assert that the input_strides of given tensors match the input_stride
        // defined in mugraph
        assert(input_dtensor_idx < (int)input_strides.size());
        assert(input_op->input_strides == input_strides[input_dtensor_idx++]);
        kernel::DTensor dt = g->new_input(
            dims, input_op->input_strides, dtensor.data_type, dtensor.layout);
        dtensor_mapping[op->output_tensors[0].guid] = dt;
        break;
      }
      case KN_OUTPUT_OP: {
        // Each KNOutputOp takes one input and has no output
        assert(dtensor_inputs.size() == 1);
        kernel::KNOutputOp *output_op = static_cast<kernel::KNOutputOp *>(op);
        g->mark_output(dtensor_inputs[0], output_op->output_strides);
        if (!output_op->output_strides.empty()) {
          assert(output_op->output_strides.size() ==
                 dtensor_inputs[0].num_dims);
          output_strides.push_back(output_op->output_strides);
        }
        this->mugraph_output_tensors.insert(this->mugraph_output_tensors.end(),
                                            dtensor_inputs.begin(),
                                            dtensor_inputs.end());
        break;
      }
      case KN_MATMUL_OP: {
        // Assert that a matmul has two input dtensors
        assert(dtensor_inputs.size() == 2);
        assert(op->output_tensors.size() == 1);
        kernel::DTensor dt = g->matmul(dtensor_inputs[0], dtensor_inputs[1]);
        dtensor_mapping[op->output_tensors[0].guid] = dt;
        break;
      }
      case KN_EXP_OP:
      case KN_SQUARE_OP:
      case KN_SQRT_OP:
      case KN_SILU_OP:
      case KN_GELU_OP:
      case KN_RELU_OP:
      case KN_CLAMP_OP: {
        assert(dtensor_inputs.size() == 1);
        assert(op->output_tensors.size() == 1);
        kernel::DTensor dt = g->elementunary(dtensor_inputs[0], op->op_type);
        dtensor_mapping[op->output_tensors[0].guid] = dt;
        break;
      }
      case KN_ADD_OP:
      case KN_MUL_OP:
      case KN_DIV_OP: {
        assert(dtensor_inputs.size() == 2);
        assert(op->output_tensors.size() == 1);
        kernel::DTensor dt =
            g->elementbinary(dtensor_inputs[0], dtensor_inputs[1], op->op_type);
        dtensor_mapping[op->output_tensors[0].guid] = dt;
        break;
      }
      case KN_REDUCTION_0_OP:
      case KN_REDUCTION_1_OP:
      case KN_REDUCTION_2_OP: {
        assert(false && "To be implemented");
        break;
      }
      case KN_RMS_NORM_OP: {
        assert(false && "To be implemented");
        break;
      }
      case KN_CUSTOMIZED_OP: {
        // Create a new threadblock graph
        kernel::KNCustomizedOp *customized_op =
            static_cast<kernel::KNCustomizedOp *>(op);
        std::shared_ptr<threadblock::Graph> tbg =
            std::make_shared<threadblock::Graph>(
                customized_op->bgraph.grid_dim,
                customized_op->bgraph.block_dim,
                customized_op->bgraph.forloop_range,
                customized_op->bgraph.reduction_dimx);
        std::unordered_map<size_t, threadblock::STensor> stensor_mapping;
        for (auto const &bop : customized_op->bgraph.operators) {
          // Preparing dtensors in the new graph
          std::vector<threadblock::STensor> stensor_inputs;
          for (auto const &t : bop->input_tensors) {
            stensor_inputs.push_back(
                get_tensor_in_new_graph(stensor_mapping, t));
          }
          switch (bop->op_type) {
            case TB_INPUT_OP: {
              threadblock::TBInputOp *input_op =
                  static_cast<threadblock::TBInputOp *>(bop);
              assert(bop->input_tensors.size() == 0);
              threadblock::STensor st = tbg->new_input(
                  get_tensor_in_new_graph(dtensor_mapping, input_op->dtensor),
                  input_op->input_map,
                  input_op->forloop_dim,
                  input_op->output_tensors[0].layout);
              stensor_mapping[bop->output_tensors[0].guid] = st;
              break;
            }
            case TB_OUTPUT_OP: {
              threadblock::TBOutputOp *output_op =
                  static_cast<threadblock::TBOutputOp *>(bop);
              assert(stensor_inputs.size() == 1);
              tbg->mark_output(stensor_inputs[0],
                               output_op->output_map,
                               output_op->forloop_dim,
                               output_op->epilogue);
              break;
            }
            case TB_MATMUL_OP: {
              threadblock::STensor st =
                  tbg->matmul(stensor_inputs[0], stensor_inputs[1]);
              stensor_mapping[bop->output_tensors[0].guid] = st;
              stensor_metas[bop->input_tensors[0].guid].m_input = true;
              break;
            }
            case TB_EXP_OP:
            case TB_SQUARE_OP:
            case TB_SQRT_OP:
            case TB_SILU_OP:
            case TB_GELU_OP:
            case TB_RELU_OP:
            case TB_CLAMP_OP:
            case TB_MUL_SCALAR_OP: {
              assert(stensor_inputs.size() == 1);
              threadblock::STensor st =
                  tbg->elementunary(stensor_inputs[0], bop->op_type);
              assert(bop->output_tensors.size() == 1);
              stensor_mapping[bop->output_tensors[0].guid] = st;
              break;
            }
            case TB_ADD_OP:
            case TB_MUL_OP:
            case TB_DIV_OP: {
              assert(stensor_inputs.size() == 2);
              threadblock::STensor st = tbg->elementbinary(
                  stensor_inputs[0], stensor_inputs[1], bop->op_type);
              assert(bop->output_tensors.size() == 1);
              stensor_mapping[bop->output_tensors[0].guid] = st;
              break;
            }
            case TB_FORLOOP_ACCUM_NO_RED_OP: {
              assert(stensor_inputs.size() == 1);
              threadblock::STensor st = tbg->forloop_accum(
                  stensor_inputs[0], TB_FORLOOP_ACCUM_NO_RED_OP);
              assert(bop->output_tensors.size() == 1);
              stensor_mapping[bop->output_tensors[0].guid] = st;
              break;
            }
            case TB_FORLOOP_ACCUM_RED_LD_SUM_OP: {
              assert(stensor_inputs.size() == 1);
              assert(bop->output_tensors.size() == 1);
              threadblock::STensor st = tbg->forloop_accum(
                  stensor_inputs[0], TB_FORLOOP_ACCUM_NO_RED_OP);
              st = tbg->reduction(st, st.num_dims - 1);
              stensor_mapping[bop->output_tensors[0].guid] = st;
              break;
            }
            case TB_FORLOOP_ACCUM_RED_LD_MEAN_OP: {
              assert(false && "To be implemented");
              break;
            }
            case TB_FORLOOP_ACCUM_RED_LD_RMS_OP: {
              assert(stensor_inputs.size() == 1);
              assert(bop->output_tensors.size() == 1);
              threadblock::STensor st = stensor_inputs[0];
              st = tbg->square(st);
              size_t normalization_factor =
                  st.dim[st.num_dims - 1] * customized_op->bgraph.forloop_range;
              st = tbg->mul_scalar(st, (1.0f / normalization_factor));
              st = tbg->forloop_accum(st, TB_FORLOOP_ACCUM_NO_RED_OP);
              st = tbg->reduction(st, st.num_dims - 1);
              st = tbg->sqrt(st);
              stensor_mapping[bop->output_tensors[0].guid] = st;
              break;
            }
            case TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP: {
              assert(stensor_inputs.size() == 1);
              assert(bop->output_tensors.size() == 1);
              threadblock::STensor st = tbg->forloop_accum(
                  stensor_inputs[0], TB_FORLOOP_ACCUM_NO_RED_OP);
              st = tbg->reduction_to_dimx(st, st.num_dims - 1);
              stensor_mapping[bop->output_tensors[0].guid] = st;
              break;
            }
            case TB_RMS_NORM_OP: {
              assert(stensor_inputs.size() == 1);
              threadblock::STensor st = stensor_inputs[0];
              st = tbg->square(st);
              st = tbg->mul_scalar(st, (1.0f / st.dim[st.num_dims - 1]));
              st = tbg->reduction(st, st.num_dims - 1);
              st = tbg->sqrt(st);
              st = tbg->div(stensor_inputs[0], st);
              stensor_mapping[bop->output_tensors[0].guid] = st;
              break;
            }
            default: {
              assert(false && "Unsupported tb operator");
            }
          }
        }
        std::vector<kernel::DTensor> dts = g->customized(dtensor_inputs, *tbg);
        assert(dts.size() == op->output_tensors.size());
        for (size_t i = 0; i < dts.size(); i++) {
          dtensor_mapping[op->output_tensors[i].guid] = dts[i];
        }
        break;
      }
      default: {
        assert(false && "Unsupported operator");
      }
    }
  }

  // Check the following:
  // 1. there is no non-default forloop accum tb operators in g
  // 2. there is no threadblock rms_norm operators in g (should be decomposed)
  for (auto const &op : g->operators) {
    if (op->op_type == KN_CUSTOMIZED_OP) {
      kernel::KNCustomizedOp *customized_op =
          static_cast<kernel::KNCustomizedOp *>(op);
      for (auto const &bop : customized_op->bgraph.operators) {
        if (bop->op_type >= TB_FORLOOP_ACCUM_FIRST_OP &&
            bop->op_type <= TB_FORLOOP_ACCUM_LAST_OP) {
          assert(bop->op_type == TB_FORLOOP_ACCUM_NO_RED_OP);
        }
        if (bop->op_type == TB_RMS_NORM_OP) {
          assert(false);
        }
      }
    }
  }
}

// Transpile a kernel graph into CUDA code
// Return (code, global memory buffer size (in bytes))
TranspileResult
    transpile(kernel::Graph const *g,
              TranspilerConfig const &config,
              std::vector<std::vector<size_t>> const &input_strides) {
  Transpiler transpiler(g, config, input_strides);
  TranspileResult result = transpiler.generate_code();
  return result;
}

} // namespace transpiler
} // namespace mirage
