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
#include "mirage/triton_transpiler/transpile.h"
#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"
#include "mirage/transpiler/utils.h"

namespace mirage {
namespace triton_transpiler {

using mirage::transpiler::CodeKeeper;
using mirage::transpiler::Combine;
using mirage::transpiler::fmt;

int TritonTranspiler::kernel_idx_counter = 0;

template <typename DT>
DT get_tensor_in_new_graph(std::unordered_map<size_t, DT> mapping,
                           DT const &tensor_in_old_graph) {
  assert(mapping.find(tensor_in_old_graph.guid) != mapping.end());
  return mapping[tensor_in_old_graph.guid];
}

TritonTranspiler::TritonTranspiler(kernel::Graph const *_graph,
                                   TritonTranspilerConfig const &_config)
    : config(_config) {
  // Create a new kernel graph
  using namespace mirage::type;
  g = std::make_shared<kernel::Graph>();
  std::unordered_map<size_t, kernel::DTensor> dtensor_mapping;

  // Process operators from input graph
  for (auto const &op : _graph->operators) {
    std::vector<kernel::DTensor> dtensor_inputs;
    for (auto const &t : op->input_tensors) {
      dtensor_inputs.push_back(get_tensor_in_new_graph(dtensor_mapping, t));
    }

    switch (op->op_type) {
      case KN_INPUT_OP: {
        kernel::KNInputOp *input_op = static_cast<kernel::KNInputOp *>(op);
        assert(op->output_tensors.size() == 1);
        kernel::DTensor const &dtensor = op->output_tensors[0];
        std::vector<int> dims;
        for (int i = 0; i < dtensor.num_dims; i++) {
          dims.push_back(dtensor.dim[i]);
        }

        kernel::DTensor dt = g->new_input(
            dims, input_op->input_strides, dtensor.data_type, dtensor.layout);
        dtensor_mapping[op->output_tensors[0].guid] = dt;
        break;
      }

      case KN_OUTPUT_OP: {
        assert(dtensor_inputs.size() == 1);
        kernel::KNOutputOp *output_op = static_cast<kernel::KNOutputOp *>(op);
        g->mark_output(dtensor_inputs[0], output_op->output_strides);
        if (!output_op->output_strides.empty()) {
          assert(output_op->output_strides.size() ==
                 dtensor_inputs[0].num_dims);
        }
        this->mugraph_output_tensors.insert(this->mugraph_output_tensors.end(),
                                            dtensor_inputs.begin(),
                                            dtensor_inputs.end());
        break;
      }

      case KN_MATMUL_OP: {
        assert(dtensor_inputs.size() == 2);
        assert(op->output_tensors.size() == 1);
        kernel::DTensor dt = g->matmul(dtensor_inputs[0], dtensor_inputs[1]);
        dtensor_mapping[op->output_tensors[0].guid] = dt;
        break;
      }
      case KN_EXP_OP:
      case KN_SQUARE_OP:
      case KN_SQRT_OP:
      case KN_SILU_OP: {
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
              break;
            }
            case TB_EXP_OP:
            case TB_SQUARE_OP:
            case TB_SQRT_OP:
            case TB_SILU_OP:
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
              // st = tbg->reduction(st, st.num_dims - 1);
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

      default:
        assert(false && "Unsupported operator type");
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
TritonTranspileResult TritonTranspiler::transpile_ugraph() {
  // Generate header
  CodeKeeper header;
  header.e("import triton");
  header.e("import triton.language as tl");
  header.e("import triton.ops as ops");
  header.e("import torch");

  // Generate execution section
  CodeKeeper exec;
  exec.e("if __name__ == \"__main__\":");
  exec.inc_indent();
  exec.e("device = torch.device('cuda')");

  CodeKeeper entrance_func;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;

  using namespace mirage::type;
  // Initialize input and output tensors
  for (kn::KNOperator *const op : g->operators) {
    if (op->op_type == KN_INPUT_OP) {
      std::string shape;
      kn::DTensor dtensor = op->output_tensors.at(0);
      for (int i = 0; i < dtensor.num_dims; i++) {
        shape += fmt("$,", dtensor.dim[i]);
      }
      exec.e("$ = torch.randn(($), dtype=torch.float16).to(device=device)",
             fmt("dtensor$", dtensor.guid),
             shape);
      input_tensor_names.push_back(fmt("dtensor$", dtensor.guid));
    }
    if (op->op_type == KN_OUTPUT_OP) {
      std::string shape;
      kn::DTensor dtensor = op->input_tensors.at(0);
      for (int i = 0; i < dtensor.num_dims; i++) {
        shape += fmt("$,", dtensor.dim[i]);
      }
      exec.e("$ = torch.zeros(($), dtype=torch.float16).to(device=device)",
             fmt("dtensor$", dtensor.guid),
             shape);
      output_tensor_names.push_back(fmt("dtensor$", dtensor.guid));
    }
  }
  std::string input_tensor_str;
  std::string output_tensor_str;
  for (int i = 0; i < input_tensor_names.size(); i++) {
    input_tensor_str += input_tensor_names[i];
    if (i != input_tensor_names.size() - 1) {
      input_tensor_str += ", ";
    }
  }
  for (int i = 0; i < output_tensor_names.size(); i++) {
    output_tensor_str += output_tensor_names[i];
    if (i != output_tensor_names.size() - 1) {
      output_tensor_str += ", ";
    }
  }
  entrance_func.e(
      "def execute_mugraph($, $):", input_tensor_str, output_tensor_str);
  entrance_func.inc_indent();
  entrance_func.e("device = torch.device('cuda')");

  // Generate custom kernels (only for truly custom operations)
  CodeKeeper custom_kernels;
  for (kn::KNOperator *const op : g->operators) {
    switch (op->op_type) {
      case KN_INPUT_OP:
      case KN_OUTPUT_OP: {
        break;
      }
      case KN_CUSTOMIZED_OP: {
        kn::KNCustomizedOp const *cur_op =
            dynamic_cast<kn::KNCustomizedOp const *>(op);
        tb::Graph const &bgraph = cur_op->bgraph;

        // Prepare tensor names
        std::vector<std::string> tensor_names;
        for (kn::DTensor const &dtensor :
             Combine(cur_op->output_tensors, cur_op->input_tensors)) {
          std::string tensor_name = fmt("dtensor$", dtensor.guid);
          tensor_names.push_back(tensor_name);
        }

        // Generate kernel
        TritonCustomOPTranspileResult result = transpile_kn_custom_op(cur_op);

        // Add kernel definition and launch
        custom_kernels.e(result.code);
        std::string new_line = fmt("$[($, $, $)]($)",
                                   result.func_name,
                                   bgraph.grid_dim.x,
                                   bgraph.grid_dim.y,
                                   bgraph.grid_dim.z,
                                   tensor_names);
        exec.e(new_line);
        entrance_func.e(new_line);
        break;
      }

      case KN_MATMUL_OP: {
        kn::DTensor &input0 = op->input_tensors[0];
        kn::DTensor &input1 = op->input_tensors[1];
        kn::DTensor &output = op->output_tensors[0];
        std::string new_line = fmt("$ = ops.matmul($, $)",
                                   fmt("dtensor$", output.guid),
                                   fmt("dtensor$", input0.guid),
                                   fmt("dtensor$", input1.guid));
        exec.e(new_line);
        entrance_func.e(new_line);
        break;
      }

      case KN_ADD_OP: {
        kn::DTensor &input0 = op->input_tensors[0];
        kn::DTensor &input1 = op->input_tensors[1];
        kn::DTensor &output = op->output_tensors[0];
        std::string new_line = fmt("$ = ops.add($, $)",
                                   fmt("dtensor$", output.guid),
                                   fmt("dtensor$", input0.guid),
                                   fmt("dtensor$", input1.guid));
        exec.e(new_line);
        entrance_func.e(new_line);
        break;
      }

      case KN_MUL_OP: {
        kn::DTensor &input0 = op->input_tensors[0];
        kn::DTensor &input1 = op->input_tensors[1];
        kn::DTensor &output = op->output_tensors[0];
        std::string new_line = fmt("$ = ops.multiply($, $)",
                                   fmt("dtensor$", output.guid),
                                   fmt("dtensor$", input0.guid),
                                   fmt("dtensor$", input1.guid));
        exec.e(new_line);
        entrance_func.e(new_line);
        break;
      }

      case KN_DIV_OP: {
        kn::DTensor &input0 = op->input_tensors[0];
        kn::DTensor &input1 = op->input_tensors[1];
        kn::DTensor &output = op->output_tensors[0];
        std::string new_line = fmt("$ = ops.divide($, $)",
                                   fmt("dtensor$", output.guid),
                                   fmt("dtensor$", input0.guid),
                                   fmt("dtensor$", input1.guid));
        exec.e(new_line);
        entrance_func.e(new_line);
        break;
      }

      case KN_EXP_OP: {
        kn::DTensor &input = op->input_tensors[0];
        kn::DTensor &output = op->output_tensors[0];
        std::string new_line = fmt("$ = ops.exp($)",
                                   fmt("dtensor$", output.guid),
                                   fmt("dtensor$", input.guid));
        exec.e(new_line);
        entrance_func.e(new_line);
        break;
      }

      case KN_SQRT_OP: {
        kn::DTensor &input = op->input_tensors[0];
        kn::DTensor &output = op->output_tensors[0];
        std::string new_line = fmt("$ = ops.sqrt($)",
                                   fmt("dtensor$", output.guid),
                                   fmt("dtensor$", input.guid));
        exec.e(new_line);
        entrance_func.e(new_line);
        break;
      }

      case KN_SQUARE_OP: {
        kn::DTensor &input = op->input_tensors[0];
        kn::DTensor &output = op->output_tensors[0];
        std::string new_line = fmt("$ = $ * $",
                                   fmt("dtensor$", output.guid),
                                   fmt("dtensor$", input.guid),
                                   fmt("dtensor$", input.guid));
        exec.e(new_line);
        entrance_func.e(new_line);
        break;
      }

      case KN_REDUCTION_0_OP:
      case KN_REDUCTION_1_OP:
      case KN_REDUCTION_2_OP: {
        kn::DTensor &input = op->input_tensors[0];
        kn::DTensor &output = op->output_tensors[0];
        int dim = op->op_type - KN_REDUCTION_0_OP;
        std::string new_line = fmt("$ = ops.reduce.sum($, dim=$)",
                                   fmt("dtensor$", output.guid),
                                   fmt("dtensor$", input.guid),
                                   dim);
        exec.e(new_line);
        entrance_func.e(new_line);
        break;
      }
    }
  }

  // Combine all sections
  std::string code = fmt("$\n$\n$\n$",
                         header.to_string(),
                         custom_kernels.to_string(),
                         entrance_func.to_string(),
                         exec.to_string());
  std::vector<std::vector<int>> output_shapes;
  for (kn::DTensor const &dtensor : this->mugraph_output_tensors) {
    output_shapes.push_back(
        std::vector<int>(dtensor.dim, dtensor.dim + dtensor.num_dims));
  }
  return TritonTranspileResult{code, output_shapes};
}

TritonTranspileResult TritonTranspiler::generate_code() {
  TritonTranspileResult result = transpile_ugraph();
  return result;
}

TritonTranspileResult transpile(kernel::Graph const *g,
                                TritonTranspilerConfig const &config) {
  TritonTranspiler transpiler(g, config);
  return transpiler.generate_code();
}

} // namespace triton_transpiler
} // namespace mirage