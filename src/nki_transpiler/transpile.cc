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

#include "mirage/nki_transpiler/transpile.h"
#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"
#include "mirage/transpiler/graph_normalizer.h"
#include "mirage/transpiler/utils.h"
#include <cassert>

#include "z3++.h"

namespace mirage {
namespace nki_transpiler {

using mirage::transpiler::ceil_div;
using mirage::transpiler::CodeKeeper;
using mirage::transpiler::Combine;
using mirage::transpiler::fmt;

namespace cost {
using cost_t = int;

// The cost of allocate 1B of shared memory for stensor is 1
cost_t NKI_TB_TRANSPOSE = 1;

} // namespace cost

NKITranspiler::NKITranspiler(kernel::Graph const *_graph,
                             NKITranspilerConfig const &_config)
    : config(_config), nki_custom_kernel_idx_counter(0) {
  auto normalized = mirage::transpiler::normalize_graph(_graph);
  g = std::move(normalized.graph);
  mugraph_output_tensors = std::move(normalized.mugraph_output_tensors);

  // Check the following:
  // 1. there is no non-default forloop accum tb operators in g
  // 2. there is no threadblock rms_norm operators in g (should be decomposed)
  for (auto const &op : g->operators) {
    if (op->op_type == type::KN_CUSTOMIZED_OP) {
      kernel::KNCustomizedOp *customized_op =
          static_cast<kernel::KNCustomizedOp *>(op);
      for (auto const &bop : customized_op->bgraph.operators) {
        if (bop->op_type >= type::TB_FORLOOP_ACCUM_FIRST_OP &&
            bop->op_type <= type::TB_FORLOOP_ACCUM_LAST_OP) {
          assert(bop->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP);
        }
        if (bop->op_type == type::TB_RMS_NORM_OP) {
          assert(false);
        }
      }
    }
  }
}

std::optional<NKIErrorInfo> NKITranspiler::resolve_tensor_layout() {
  // Get a list of all STensors
  std::vector<tb::STensor> all_stensors;
  std::unordered_set<sguid_t> processed_sguids;
  for (kn::KNOperator *const op : this->g->operators) {
    if (op->op_type == type::KN_CUSTOMIZED_OP) {
      kn::KNCustomizedOp *cur_op = dynamic_cast<kn::KNCustomizedOp *>(op);
      for (tb::TBOperator *const tb_op : cur_op->bgraph.operators) {
        for (tb::STensor const &stensor :
             Combine(tb_op->input_tensors, tb_op->output_tensors)) {
          if (processed_sguids.count(stensor.guid) == 0) {
            processed_sguids.insert(stensor.guid);
            all_stensors.push_back(stensor);
          }
        }
      }
    }
  }
  // Create z3 context and optimizer
  z3::context ctx;
  z3::optimize opt(ctx);
  z3::expr_vector costs(ctx);
  // Create variables denoting whether a dimension is the partition dimension
  // sp_x_y denotes whether the y-th dimension of STensor x is the partition dim
  std::unordered_map<sguid_t, std::vector<z3::expr>> s_is_partition;
  for (tb::STensor const &stensor : all_stensors) {
    int num_dims = stensor.num_dims;
    for (int i = 0; i < num_dims; i++) {
      std::string var_name = fmt("sp_$_$", stensor.guid, i);
      s_is_partition[stensor.guid].push_back(ctx.bool_const(var_name.c_str()));
    }
  }
  // Create constraints that limit the number of partition dimension to 1
  for (tb::STensor const &stensor : all_stensors) {
    int num_dims = stensor.num_dims;
    // Every stensor can have at most 1 innermost dim
    z3::expr_vector partition_exprs(ctx);
    for (int i = 0; i < num_dims; i++) {
      partition_exprs.push_back(s_is_partition[stensor.guid][i]);
      // A partition dimension cannot be larger than 128
      // if (stensor.dim[i] > 128) {
      //   opt.add(!s_is_partition[stensor.guid][i]);
      // }
      // A partition dimension must be the last two dims
      if ((i != num_dims - 1) && (i != num_dims - 2)) {
        opt.add(!s_is_partition[stensor.guid][i]);
      }
    }
    opt.add(z3::atmost(partition_exprs, 1));
    opt.add(z3::atleast(partition_exprs, 1));
  }

  // Constraints and costs for every threadblock-level operator
  for (kn::KNOperator const *kn_op : this->g->operators) {
    if (kn_op->op_type == type::KN_CUSTOMIZED_OP) {
      kn::KNCustomizedOp const *kn_customized_op =
          static_cast<kn::KNCustomizedOp const *>(kn_op);
      tb::Graph const &tb_graph = kn_customized_op->bgraph;
      for (tb::TBOperator const *tb_op : tb_graph.operators) {
        switch (tb_op->op_type) {
          case type::TB_INPUT_OP: {
            // TB input operator
            // tb::TBInputOp const *tb_input_op =
            //    static_cast<tb::TBInputOp const *>(tb_op);
            // tb::STensor const &output = tb_op->output_tensors.at(0);
            // TODO: does the cost of loading stensor from HBM to SBUF
            // depend on the partition dimension???
            // Currently do nothing
            break;
          }
          case type::TB_OUTPUT_OP: {
            // TODO: does the cost of saving stensor from SBUF to HBM
            // depend on the partition diemnsion???
            // Currently do nothing
            break;
          }
          case type::TB_MATMUL_OP: {
            tb::STensor const &input0 = tb_op->input_tensors.at(0);
            tb::STensor const &input1 = tb_op->input_tensors.at(1);
            tb::STensor const &output = tb_op->output_tensors.at(0);
            assert(input0.num_dims == input1.num_dims &&
                   input0.num_dims == output.num_dims);
            int num_dims = input0.num_dims;
            assert(num_dims >= 2);
            // Add a transpose cost if input0's last dim is not the
            // partition dim
            costs.push_back(z3::ite(!s_is_partition[input0.guid][num_dims - 1],
                                    ctx.int_val(cost::NKI_TB_TRANSPOSE),
                                    ctx.int_val(0)));
            // Add a transpose cost if input1's last dim is not the
            // partition dim
            costs.push_back(z3::ite(!s_is_partition[input1.guid][num_dims - 2],
                                    ctx.int_val(cost::NKI_TB_TRANSPOSE),
                                    ctx.int_val(0)));
            // NKI's matmul interface:
            // (https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.isa.nc_matmul.html)
            // requires the stationary operand is at most 128x128, while the
            // moving operand is at most 128x512. As a result,
            // the transpiler has two choices when transpiling matmul:
            // Case 1: input0 is stationary and input1 is moving
            // Case 2: input1 is stationary and input0 is moving
            // We compute the number of matmuls for them and pick the more
            // efficient one
            int num_matmul_for_case1 = ceil_div(input0.dim[num_dims - 2], 128) *
                                       ceil_div(input1.dim[num_dims - 1], 512);
            int num_matmul_for_case2 = ceil_div(input1.dim[num_dims - 1], 128) *
                                       ceil_div(input0.dim[num_dims - 2], 512);
            if (num_matmul_for_case1 < num_matmul_for_case2) {
              // Enforce using case 1 by setting output.dim[num_dims-2] as
              // the partition dim for the output
              opt.add(s_is_partition[output.guid][num_dims - 2]);
            } else if (num_matmul_for_case1 > num_matmul_for_case2) {
              // Enforce using case 2 by setting output.dim[num_dims-1] as
              // the partition dim for the output
              opt.add(s_is_partition[output.guid][num_dims - 1]);
            }
            break;
          }
          case type::TB_EXP_OP:
          case type::TB_SILU_OP:
          case type::TB_SQUARE_OP:
          case type::TB_SQRT_OP:
          case type::TB_RELU_OP:
          case type::TB_CLAMP_OP:
          case type::TB_MUL_SCALAR_OP: {
            tb::STensor const &input = tb_op->input_tensors.at(0);
            tb::STensor const &output = tb_op->output_tensors.at(0);
            assert(input.num_dims == output.num_dims);
            int num_dims = input.num_dims;
            // Need a transpose is input and output pick different partition dim
            for (int i = 0; i < num_dims; i++) {
              costs.push_back(z3::ite(!s_is_partition[input.guid][i] &&
                                          s_is_partition[output.guid][i],
                                      ctx.int_val(cost::NKI_TB_TRANSPOSE),
                                      ctx.int_val(0)));
            }
            break;
          }
          case type::TB_ADD_OP:
          case type::TB_MUL_OP:
          case type::TB_DIV_OP:
          case type::TB_SUB_OP:
          case type::TB_POW_OP: {
            tb::STensor const &input0 = tb_op->input_tensors.at(0);
            tb::STensor const &input1 = tb_op->input_tensors.at(1);
            tb::STensor const &output = tb_op->output_tensors.at(0);
            assert(input0.num_dims == input1.num_dims &&
                   input0.num_dims == output.num_dims);
            int num_dims = input0.num_dims;
            for (int i = 0; i < num_dims; i++) {
              // Need a transpose if input0 and output pick different partition
              // dim
              costs.push_back(z3::ite(!s_is_partition[input0.guid][i] &&
                                          s_is_partition[output.guid][i],
                                      ctx.int_val(cost::NKI_TB_TRANSPOSE),
                                      ctx.int_val(0)));
              // Need a transpose if input1 and output pick different partition
              // dim
              costs.push_back(z3::ite(!s_is_partition[input1.guid][i] &&
                                          s_is_partition[output.guid][i],
                                      ctx.int_val(cost::NKI_TB_TRANSPOSE),
                                      ctx.int_val(0)));
            }
            break;
          }
          case type::TB_FORLOOP_ACCUM_NO_RED_OP: {
            // Do nothing
            break;
          }
          case type::TB_REDUCTION_0_OP:
          case type::TB_REDUCTION_1_OP:
          case type::TB_REDUCTION_2_OP:
          case type::TB_REDUCTION_0_TO_DIMX_OP:
          case type::TB_REDUCTION_1_TO_DIMX_OP:
          case type::TB_REDUCTION_2_TO_DIMX_OP: {
            int reduc_dim =
                tb_op->op_type >= type::TB_REDUCTION_0_TO_DIMX_OP
                    ? tb_op->op_type - type::TB_REDUCTION_0_TO_DIMX_OP
                    : tb_op->op_type - type::TB_REDUCTION_0_OP;
            tb::STensor const &input = tb_op->input_tensors.at(0);
            tb::STensor const &output = tb_op->output_tensors.at(0);
            int num_dims = input.num_dims;
            assert(input.num_dims == output.num_dims);
            assert(0 <= reduc_dim && reduc_dim < num_dims);
            // The reduction dim cannot be the partition dim for both
            // input and output
            opt.add(!s_is_partition[input.guid][reduc_dim]);
            opt.add(!s_is_partition[output.guid][reduc_dim]);
            for (int i = 0; i < num_dims; i++) {
              // Need a transpose if input and output pick different partition
              // dim
              costs.push_back(z3::ite(!s_is_partition[input.guid][i] &&
                                          s_is_partition[output.guid][i],
                                      ctx.int_val(cost::NKI_TB_TRANSPOSE),
                                      ctx.int_val(0)));
            }
            break;
          }
          default: {
            assert(fmt("Unsupported TB op: $", tb_op->op_type).c_str());
          }
        }
      }
    }
  }

  // Optimize
  if (costs.empty()) {
    costs.push_back(ctx.int_val(0));
  }
  z3::expr objective = z3::sum(costs);
  opt.minimize(objective);
  z3::check_result check_result = opt.check();

  // z3 can't provide a sat, generate the error state.
  if (check_result != z3::sat) {
    std::string msg = check_result == z3::unsat
                          ? "Z3 unsat: No valid stensor layout found."
                          : "Z3 unknown: While resolving stensor layout.";
    std::vector<std::string> error_msgs;
    error_msgs.emplace_back(std::move(msg));
    return error_msgs;
  }

  assert(check_result == z3::sat);

  // Retrieve the result
  z3::model m = opt.get_model();
  for (tb::STensor const &stensor : all_stensors) {
    int num_dims = stensor.num_dims;
    int partition_dim = -1;
    for (int i = 0; i < num_dims; i++) {
      if (m.eval(s_is_partition[stensor.guid][i]).is_true()) {
        partition_dim = i;
        break;
      }
    }
    assert(partition_dim != -1);
    this->stensor_metas[stensor.guid].partition_dim = partition_dim;
  }
  return std::nullopt;
}

NKITranspileResult NKITranspiler::transpile_ugraph() {
  // Generate header
  CodeKeeper header;
  header.e("import neuronxcc.nki as nki");
  header.e("import neuronxcc.nki.language as nl");
  header.e("import neuronxcc.nki.isa as nisa");
  CodeKeeper exec;
  exec.e("if __name__ == \"__main__\":");
  exec.inc_indent();
  exec.e("import torch");
  exec.e("from torch_xla.core import xla_model as xm");
  exec.e("device = xm.xla_device()");
  for (kn::KNOperator *const op : g->operators) {
    for (kn::DTensor const &dtensor : op->output_tensors) {
      std::string shape;
      for (int i = 0; i < dtensor.num_dims; i++) {
        shape += fmt("$,", dtensor.dim[i]);
      }
      exec.e("$ = torch.randn(($), dtype=torch.float16).to(device=device)",
             fmt("dtensor$", dtensor.guid),
             shape);
    }
  }
  // Generate helper functions
  CodeKeeper helper;
  std::vector<HelperFunction> helper_functions;
  helper_functions.push_back(tiled_transpose_function());
  helper_functions.push_back(tiled_matmul_function());
  helper_functions.push_back(tiled_matmul_accum_function());
  for (HelperFunction const &hf : helper_functions) {
    helper.e(hf.get_code());
  }
  CodeKeeper custom_kernels;
  for (kn::KNOperator *const op : g->operators) {
    switch (op->op_type) {
      case type::KNOperatorType::KN_INPUT_OP:
      case type::KNOperatorType::KN_OUTPUT_OP: {
        // input and output have been processed before
        break;
      }
      case type::KNOperatorType::KN_CUSTOMIZED_OP: {
        // Customized op
        // Customized op
        kn::KNCustomizedOp const *cur_op =
            dynamic_cast<kn::KNCustomizedOp const *>(op);
        // tb::ExecutionPlan const &plan = cur_op->plan;
        tb::Graph const &bgraph = cur_op->bgraph;
        std::vector<std::string> dtensor_names;
        for (kn::DTensor const &dtensor : cur_op->input_tensors) {
          std::string dtensor_name = fmt("dtensor$", dtensor.guid);
          dtensor_names.push_back(dtensor_name);
        }
        // Transpile
        NKICustomOPTranspileResult result = transpile_kn_custom_op(cur_op);
        // Launch kernels
        custom_kernels.e(result.code);
        exec.e("$($)", result.func_name, dtensor_names);
        break;
      }
      case type::KN_ADD_OP:
      case type::KN_MUL_OP:
      case type::KN_DIV_OP:
      case type::KN_POW_OP: {
        kn::KNElementBinaryOp const *cur_op =
            dynamic_cast<kn::KNElementBinaryOp const *>(op);
        std::vector<std::string> dtensor_names;
        for (kn::DTensor const &dtensor :
             Combine(cur_op->output_tensors, cur_op->input_tensors)) {
          std::string dtensor_name = fmt("dtensor$", dtensor.guid);
          dtensor_names.push_back(dtensor_name);
        }
        // Transpile
        auto result = transpile_kn_op(cur_op);
        if (result.has_value()) {
          custom_kernels.e(result.value().code);
          // launch a single SPMD kernel
          exec.e("$($)", result.value().func_name, dtensor_names);
        }
        break;
      }

      default: {
        // TODO: discuss with the NKI team on how to implement
        // operators at the kernel level
        assert(false && "To be implemented");
      }
    }
  }

  std::string code = fmt("$\n$\n$\n$",
                         header.to_string(),
                         helper.to_string(),
                         custom_kernels.to_string(),
                         exec.to_string());
  return NKITranspileResult{std::move(code)};
}

NKITranspileResult NKITranspiler::generate_code() {
  auto maybe_error = this->resolve_tensor_layout();
  if (maybe_error.has_value()) { // contains error
    return NKITranspileResult{"", maybe_error.value()};
  }
  return this->transpile_ugraph();
}

NKITranspileResult transpile(kernel::Graph const *g,
                             NKITranspilerConfig const &config) {
  NKITranspiler transpiler(g, config);
  NKITranspileResult result = transpiler.generate_code();
  return result;
}

} // namespace nki_transpiler
} // namespace mirage
