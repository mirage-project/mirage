/* Copyright 2023-2025 CMU
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

#include "mirage/transpiler/graph_normalizer.h"

#include <cassert>

#include "mirage/kernel/element_unary.h"
#include "mirage/threadblock/element_unary.h"

namespace mirage {
namespace transpiler {

namespace {

using namespace mirage::type;

template <typename TensorT>
TensorT get_tensor_in_new_graph(
    std::unordered_map<decltype(TensorT::guid), TensorT> const &mapping,
    TensorT const &tensor_in_old_graph) {
  auto it = mapping.find(tensor_in_old_graph.guid);
  assert(it != mapping.end());
  return it->second;
}

threadblock::STensor lower_tb_op(
    threadblock::Graph &tbg,
    kernel::KNCustomizedOp const *customized_op,
    threadblock::TBOperator const *bop,
    std::vector<threadblock::STensor> const &stensor_inputs) {
  switch (bop->op_type) {
    case TB_MATMUL_OP:
      return tbg.matmul(stensor_inputs[0], stensor_inputs[1]);
    case TB_EXP_OP:
    case TB_SQUARE_OP:
    case TB_SQRT_OP:
    case TB_SILU_OP:
    case TB_SIGMOID_OP:
    case TB_GELU_OP:
    case TB_RELU_OP:
    case TB_LOG_OP:
      return tbg.elementunary(stensor_inputs[0], bop->op_type);
    case TB_CLAMP_OP: {
      auto const *clamp_op =
          dynamic_cast<threadblock::TBClampUnaryOp const *>(bop);
      assert(clamp_op != nullptr);
      return tbg.clamp(stensor_inputs[0], clamp_op->min_val, clamp_op->max_val);
    }
    case TB_MUL_SCALAR_OP: {
      auto const *unary =
          dynamic_cast<threadblock::TBElementUnaryOp const *>(bop);
      assert(unary != nullptr);
      return tbg.elementunary(stensor_inputs[0], bop->op_type, unary->scalar);
    }
    case TB_ADD_OP:
    case TB_MUL_OP:
    case TB_DIV_OP:
    case TB_SUB_OP:
    case TB_POW_OP:
      return tbg.elementbinary(stensor_inputs[0], stensor_inputs[1], bop->op_type);
    case TB_REDUCTION_0_OP:
    case TB_REDUCTION_1_OP:
    case TB_REDUCTION_2_OP:
      return tbg.reduction(stensor_inputs[0], bop->op_type - TB_REDUCTION_0_OP);
    case TB_REDUCTION_0_TO_DIMX_OP:
    case TB_REDUCTION_1_TO_DIMX_OP:
    case TB_REDUCTION_2_TO_DIMX_OP:
      return tbg.reduction_to_dimx(stensor_inputs[0],
                                   bop->op_type - TB_REDUCTION_0_TO_DIMX_OP);
    case TB_FORLOOP_ACCUM_NO_RED_OP:
      return tbg.forloop_accum(stensor_inputs[0], TB_FORLOOP_ACCUM_NO_RED_OP);
    case TB_FORLOOP_ACCUM_RED_LD_SUM_OP: {
      threadblock::STensor st =
          tbg.forloop_accum(stensor_inputs[0], TB_FORLOOP_ACCUM_NO_RED_OP);
      return tbg.reduction(st, st.num_dims - 1);
    }
    case TB_FORLOOP_ACCUM_RED_LD_MEAN_OP: {
      threadblock::STensor st =
          tbg.forloop_accum(stensor_inputs[0], TB_FORLOOP_ACCUM_NO_RED_OP);
      st = tbg.reduction(st, st.num_dims - 1);
      return tbg.mul_scalar(
          st,
          1.0f / static_cast<float>(stensor_inputs[0].dim[stensor_inputs[0].num_dims -
                                                           1] *
                                    customized_op->bgraph.forloop_range));
    }
    case TB_FORLOOP_ACCUM_RED_LD_RMS_OP: {
      threadblock::STensor st = tbg.square(stensor_inputs[0]);
      size_t normalization_factor =
          stensor_inputs[0].dim[stensor_inputs[0].num_dims - 1] *
          customized_op->bgraph.forloop_range;
      st = tbg.mul_scalar(st, 1.0f / normalization_factor);
      st = tbg.forloop_accum(st, TB_FORLOOP_ACCUM_NO_RED_OP);
      st = tbg.reduction(st, st.num_dims - 1);
      return tbg.sqrt(st);
    }
    case TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP: {
      threadblock::STensor st =
          tbg.forloop_accum(stensor_inputs[0], TB_FORLOOP_ACCUM_NO_RED_OP);
      return tbg.reduction_to_dimx(st, st.num_dims - 1);
    }
    case TB_RMS_NORM_OP: {
      threadblock::STensor st = tbg.square(stensor_inputs[0]);
      st = tbg.mul_scalar(st, 1.0f / stensor_inputs[0].dim[stensor_inputs[0].num_dims -
                                                           1]);
      st = tbg.reduction(st, st.num_dims - 1);
      st = tbg.sqrt(st);
      return tbg.div(stensor_inputs[0], st);
    }
    default:
      assert(false && "Unsupported tb operator in graph normalizer");
  }
}

} // namespace

GraphNormalizationResult normalize_graph(kernel::Graph const *graph) {
  GraphNormalizationResult result;
  result.graph = std::make_shared<kernel::Graph>();

  for (auto const &op : graph->operators) {
    std::vector<kernel::DTensor> dtensor_inputs;
    for (auto const &t : op->input_tensors) {
      dtensor_inputs.push_back(get_tensor_in_new_graph(result.dtensor_mapping, t));
    }

    switch (op->op_type) {
      case KN_INPUT_OP: {
        auto const *input_op = static_cast<kernel::KNInputOp const *>(op);
        assert(op->output_tensors.size() == 1);
        kernel::DTensor const &dtensor = op->output_tensors[0];
        std::vector<int> dims;
        for (int i = 0; i < dtensor.num_dims; ++i) {
          dims.push_back(dtensor.dim[i]);
        }
        kernel::DTensor dt = result.graph->new_input(
            dims, input_op->input_strides, dtensor.data_type, dtensor.layout);
        result.dtensor_mapping[op->output_tensors[0].guid] = dt;
        break;
      }
      case KN_OUTPUT_OP: {
        auto const *output_op = static_cast<kernel::KNOutputOp const *>(op);
        assert(dtensor_inputs.size() == 1);
        result.graph->mark_output(dtensor_inputs[0], output_op->output_strides);
        result.mugraph_output_tensors.push_back(dtensor_inputs[0]);
        break;
      }
      case KN_MATMUL_OP: {
        kernel::DTensor dt = result.graph->matmul(dtensor_inputs[0], dtensor_inputs[1]);
        result.dtensor_mapping[op->output_tensors[0].guid] = dt;
        break;
      }
      case KN_EXP_OP:
      case KN_SQUARE_OP:
      case KN_SQRT_OP:
      case KN_SILU_OP:
      case KN_SIGMOID_OP:
      case KN_GELU_OP:
      case KN_RELU_OP:
      case KN_LOG_OP: {
        kernel::DTensor dt = result.graph->elementunary(dtensor_inputs[0], op->op_type);
        result.dtensor_mapping[op->output_tensors[0].guid] = dt;
        break;
      }
      case KN_CLAMP_OP: {
        auto const *clamp_op = dynamic_cast<kernel::KNClampUnaryOp const *>(op);
        assert(clamp_op != nullptr);
        kernel::DTensor dt = result.graph->clamp(
            dtensor_inputs[0], clamp_op->min_val, clamp_op->max_val);
        result.dtensor_mapping[op->output_tensors[0].guid] = dt;
        break;
      }
      case KN_ADD_OP:
      case KN_MUL_OP:
      case KN_DIV_OP:
      case KN_POW_OP: {
        kernel::DTensor dt =
            result.graph->elementbinary(dtensor_inputs[0], dtensor_inputs[1], op->op_type);
        result.dtensor_mapping[op->output_tensors[0].guid] = dt;
        break;
      }
      case KN_REDUCTION_0_OP:
      case KN_REDUCTION_1_OP:
      case KN_REDUCTION_2_OP: {
        kernel::DTensor dt = result.graph->reduction(
            dtensor_inputs[0], op->op_type - KN_REDUCTION_0_OP);
        result.dtensor_mapping[op->output_tensors[0].guid] = dt;
        break;
      }
      case KN_CUSTOMIZED_OP: {
        auto const *customized_op = static_cast<kernel::KNCustomizedOp const *>(op);
        std::shared_ptr<threadblock::Graph> tbg = std::make_shared<threadblock::Graph>(
            customized_op->bgraph.grid_dim,
            customized_op->bgraph.block_dim,
            customized_op->bgraph.forloop_range,
            customized_op->bgraph.reduction_dimx);

        for (auto const &bop : customized_op->bgraph.operators) {
          std::vector<threadblock::STensor> stensor_inputs;
          for (auto const &t : bop->input_tensors) {
            stensor_inputs.push_back(
                get_tensor_in_new_graph(result.stensor_mapping, t));
          }

          switch (bop->op_type) {
            case TB_INPUT_OP: {
              auto const *input_op =
                  static_cast<threadblock::TBInputOp const *>(bop);
              threadblock::STensor st = tbg->new_input(
                  get_tensor_in_new_graph(result.dtensor_mapping, input_op->dtensor),
                  input_op->input_map,
                  input_op->forloop_dim,
                  input_op->output_tensors[0].layout,
                  input_op->output_tensors[0].store_in_dmem);
              result.stensor_mapping[bop->output_tensors[0].guid] = st;
              break;
            }
            case TB_OUTPUT_OP: {
              auto const *output_op =
                  static_cast<threadblock::TBOutputOp const *>(bop);
              tbg->mark_output(stensor_inputs[0],
                               output_op->output_map,
                               output_op->forloop_dim,
                               output_op->epilogue);
              break;
            }
            default: {
              threadblock::STensor st = lower_tb_op(
                  *tbg, customized_op, bop, stensor_inputs);
              if (!bop->output_tensors.empty()) {
                result.stensor_mapping[bop->output_tensors[0].guid] = st;
              }
              break;
            }
          }
        }

        std::vector<kernel::DTensor> outputs =
            result.graph->customized(dtensor_inputs, *tbg);
        assert(outputs.size() == op->output_tensors.size());
        for (size_t i = 0; i < outputs.size(); ++i) {
          result.dtensor_mapping[op->output_tensors[i].guid] = outputs[i];
        }
        break;
      }
      case KN_RMS_NORM_OP:
      case KN_ALLREDUCE_OP:
      case KN_CONCAT_0_OP:
      case KN_CONCAT_1_OP:
      case KN_CONCAT_2_OP:
      case KN_SPLIT_0_OP:
      case KN_SPLIT_1_OP:
      case KN_SPLIT_2_OP:
      case KN_CHUNK_0_OP:
      case KN_CHUNK_1_OP:
      case KN_CHUNK_2_OP:
      default:
        assert(false && "Unsupported kernel operator in graph normalizer");
    }
  }

  return result;
}

} // namespace transpiler
} // namespace mirage
