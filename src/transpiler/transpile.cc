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

#include <cassert>

#include "mirage/transpiler/transpiler.h"

namespace mirage {
namespace transpiler {

DTensor get_dtensor_in_new_graph(
    std::unordered_map<size_t, DTensor> mapping,
    DTensor const & dtensor_in_old_graph) {
  assert(mapping.find(dtensor_in_old_graph.guid) != mapping.end());
  return mapping[dtensor_in_old_graph.guid];
}

Transpiler(kernel::Graph const *graph,
           TranspilerConfig const &config,
           vector<vector<size_t>> const &input_strides,
           vector<kn::DTensor const *> const &output_tensors);
    : config(config), input_strides(input_strides),
      output_tensors(output_tensors) {
  // Currently we only support GPUs with compute capability >= 8.0 (A100+)
  // TODO(intlsy): Support older GPUs
  if (config.target_cc < GPU_CC::A100) {
    throw std::runtime_error("Unsupported target compute capability");
  }
  // We need to construct a new kernel graph by decomposing forloop accumulators
  // into the non-reduction accumulator type to enable transpiler optimizations
  g = std::shared_ptr<kernel::Graph>();
  std::unordered_map<size_t, DTensor> dtensor_mapping;
  for (const auto & op : graph->operators) {
    using namespace mirage::type;
    // Preparing dtensors in the new graph
    std::vector<DTensor> inputs;
    for (const auto & t : op->input_tensors) {
      inputs.push_back(get_dtensor_in_new_graph(dtensor_mapping, t));
    }
    switch (op->op_type) {
      case KN_INPUT_OP: {
        // Assert that an input op has exactly one output dtensor
        assert(op->output_tensors.size() == 1);
        kernel::DTensor const& dtensor = op->output_tensors[0];
        std::vector<int> dims;
        for (int i = 0; i < dtensor.num_dims; i++) {
          dims.push_back(dtensor.dim[i]);
        }
        g->new_input(dims, dtensor->data_type, dtensor->layout);
        dtensor_mapping[op] = g->operators.back();
        break;
      }
      case KN_MATMUL_OP: {
        // Assert that a matmul has two input dtensors
        assert(inputs.size() == 2);
        DTensor output = g->matmul(inputs[0], inputs[1]);
        assert(op->output_tensors.size() == 1);
        dtensor_mapping[op->output_tensors[0].guid] = output;
        break;
      }
      case KN_EXP_OP:
      case KN_SQUARE_OP:
      case KN_SQRT_OP:
      case KN_SILU_OP: {
        assert(inputs.size() == 1);
        DTensor output = g->elementunary(inputs[0], op->op_type);
        assert(op->output_tensors.size() == 1);
        dtensor_mapping[op->output_tensors[0].guid] = output;
        break;
      }
      case KN_ADD_OP:
      case KN_MUL_OP:
      case KN_DIV_OP: {
        break;
      }
      case KN_REDUCTION_0_OP:
      case KN_REDUCTION_1_OP:
      case KN_REDUCTION_2_OP: {
      }
      case KN_RMS_NORM_OP: {
      }
      case KN_CUSTOMIZED_OP: {
        // Create a new threadblock graph
        kernel::KNCustomizedOp * cop = static_cast<kernel::KNCustomizedOp*>(op);
        threadblock::Graph *tbg = sdt::shared_ptr<threadblock::Graph*>(
            op->bgraph.grid_dim, op->bgraph.block_dim, op->bgraph.forloop_range,
            op->bgraph.reduction_dimx);
        std::unordered_map<size_t, STensor> smapping;
        std::vector<STensor> sinputs;
        for (const auto& bop : op->bgraph.operators) {
          switch (bop->op_type) {
            case TB_INPUT_OP:
            case TB_OUTPUT_OP: {
            }
            case TB_MATMUL_OP: {
              STensor output = tbg->matmul(sinputs[0], sinputs[1]);
              smapping[bop->output_tensors[0].guid] = output;
              break;
            }
            case TB_EXP_OP:
            case TB_SQUARE_OP:
            case TB_SQRT_OP:
            case TB_SILU_OP: {
              break;
            }
            case TB_ADD_OP:
            case TB_MUL_OP:
            case TB_DIV_OP: {
              break;
            }
            case TB_FORLOOP_ACCUM_NO_RED_OP: {
              STensor st = tbg->forloop_accum(sinputs[0], TB_FORLOOP_ACCUM_NO_RED_OP);
              smapping[bop->output_tensors[0].guid] = st;
              break;
            }
            case TB_FORLOOP_ACCUM_RED_LD_SUM_OP: {
              STensor st = tbg->forloop_accum(sinputs[0], TB_FORLOOP_ACCUM_NO_RED_OP);
              st = tbg->reduction(st, st.num_dims-1);
              smapping[bop->output_tensors[0].guid] = st;
              break;
            }
            case TB_FORLOOP_ACCUM_RED_LD_MEAN_OP: {
              assert(false && "TBI");
              break;
            }
            case TB_FORLOOP_ACCUM_RED_LD_RMS_OP: {
              STensor st = tbg->square(sinputs[0]);
              st = tbg->forloop_accum(st, TB_FORLOOP_ACCUM_NO_RED_OP);
              // FIXME: add mul_scalar
              st = tbg->sqrt(st);
              smapping[bop->output_tensors[0].guid] = st;
              break;
            }
            case TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP: {
              STensor st = tbg->forloop_accum(sinputs[0], TB_FORLOOP_ACCUM_NO_RED_OP);
              st = tbg->reduction_to_dimx(st, st.num_dims-1);
              smapping[bop->output_tensors[0].guid] = st;
              break;
            }
            default: {
              assert(false && "Unsupported tb operator");
            }
          }
        }
      }
      default: {
        assert(false && "Unsupported operator");
      }
    }
  }
}


// Transpile a kernel graph into CUDA code
// Return (code, global memory buffer size (in bytes))
TranspileResult
    transpile(kernel::Graph const *g,
              TranspilerConfig const &config,
              std::vector<std::vector<size_t>> const &input_strides,
              std::vector<kernel::DTensor const *> const &output_tensors) {
  Transpiler transpiler(g, config, input_strides, output_tensors);
  TranspileResult result = transpiler.generate_code();
  return result;
}

} // namespace transpiler
} // namespace mirage
