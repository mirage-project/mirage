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

#include "mirage/threadblock/element_unary.h"
#include "mirage/threadblock/forloop_accum.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/transpiler/utils.h"
#include "mirage/triton_transpiler/transpile.h"

inline int round_up_to_power_of_2(int n) {
  if (n <= 0) {
    return 1;
  }
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return (n + 1) >= 16 ? (n + 1) : 16;
}

inline bool is_power_of_2(int n) {
  return n > 0 && (n & (n - 1)) == 0;
}

namespace mirage {
namespace triton_transpiler {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

using mirage::transpiler::CodeKeeper;
using mirage::transpiler::Combine;
using mirage::transpiler::fmt;
using mirage::transpiler::map;
using std::string;

inline std::string get_tensor_shape(tb::STensor const &stensor) {
  std::string shape = "";
  for (int i = 0; i < stensor.num_dims; i++) {
    shape += fmt("$,", stensor.dim[i]);
  }
  return shape;
}

std::vector<int> adjust_tensor_dims(tb::STensor const &stensor) {
  std::vector<int> adjusted_dims;
  for (int i = 0; i < stensor.num_dims; i++) {
    adjusted_dims.push_back(round_up_to_power_of_2(stensor.dim[i]));
  }
  return adjusted_dims;
}

string operator_type_to_triton(type::TBOperatorType type) {
  switch (type) {
    case type::TB_EXP_OP:
      return "tl.exp";
    case type::TB_SILU_OP:
      return "tl.sigmoid";
    case type::TB_SQRT_OP:
      return "tl.sqrt";
    case type::TB_SQUARE_OP:
      return "tl.square";
    case type::TB_DIV_OP:
      return "tl.fdiv"; // TODO: AttributeError: module 'triton.language' has no
                        // attribute 'div_rn'
    case type::TB_POW_OP:
      return "tl.power";
    case type::TB_ADD_OP:
      return "+";
    case type::TB_SUB_OP:
      return "-";
    case type::TB_MUL_OP:
      return "*";
    default:
      assert(false && "Unsupported operator type in operator_type_to_triton()");
  }
}

std::string generate_mask_expr(tb::STensor const &stensor,
                               std::vector<int> const &adjusted_dims,
                               int3 const &map,
                               int forloop_dim = -1) {
  std::vector<std::string> mask_conditions;

  for (int i = 0; i < stensor.num_dims; i++) {
    // Only create mask if dimension was padded
    if (stensor.dim[i] != adjusted_dims[i]) {
      std::string base_expr;

      // Handle different dimensions
      if (i == stensor.num_dims - 1) {
        base_expr = "tl.arange(0, $)[None, :]";
      } else {
        base_expr = "tl.arange(0, $)[:, None]";
      }

      std::string condition =
          fmt("$ < $", fmt(base_expr, adjusted_dims[i]), stensor.dim[i]);

      condition =
          fmt("($) < $", fmt(base_expr, adjusted_dims[i]), stensor.dim[i]);

      mask_conditions.push_back(condition);
    }
  }

  if (mask_conditions.empty()) {
    return "";
  }

  // Combine all conditions
  std::string start = "";
  std::string end = "";
  if (mask_conditions.size() > 1) {
    start = "((";
    end = "))";
  } else {
    start = "(";
    end = ")";
  }
  std::string mask_expr = start + mask_conditions[0];
  for (size_t i = 1; i < mask_conditions.size(); i++) {
    mask_expr += ")&(" + mask_conditions[i];
  }
  mask_expr += end;

  return mask_expr;
}

// Generate expression for a dimension
std::string get_input_dim_expr(int dim_idx,
                               int3 const &imap,
                               int forloop_dim,
                               int forloop_range,
                               int block_size) {
  std::string base_expr = "tl.arange(0, $)";
  std::vector<std::string> offset_terms;

  if (imap.x == dim_idx) {
    offset_terms.push_back(fmt("tl.program_id(0) * $", block_size));
  }
  if (imap.y == dim_idx) {
    offset_terms.push_back(fmt("tl.program_id(1) * $", block_size));
  }
  if (imap.z == dim_idx) {
    offset_terms.push_back(fmt("tl.program_id(2) * $", block_size));
  }

  if (forloop_dim == dim_idx) {
    offset_terms.push_back(fmt("i * $", block_size));
  }

  std::string result = fmt(base_expr, round_up_to_power_of_2(block_size));
  if (!offset_terms.empty()) {
    std::string offset = offset_terms[0];
    for (size_t i = 1; i < offset_terms.size(); i++) {
      offset += " + " + offset_terms[i];
    }
    result = fmt("$ + $", offset, result);
  }

  return result;
}

std::string get_output_dim_expr(int dim_idx, int3 const &omap, int block_size) {
  std::string base_expr = "tl.arange(0, $)";
  std::vector<std::string> offset_terms;

  // 处理program_id映射
  if (omap.x == dim_idx) {
    offset_terms.push_back(fmt("tl.program_id(0) * $", block_size));
  }
  if (omap.y == dim_idx) {
    offset_terms.push_back(fmt("tl.program_id(1) * $", block_size));
  }
  if (omap.z == dim_idx) {
    offset_terms.push_back(fmt("tl.program_id(2) * $", block_size));
  }

  // 组合表达式
  std::string result = fmt(base_expr, round_up_to_power_of_2(block_size));
  if (!offset_terms.empty()) {
    std::string offset = offset_terms[0];
    for (size_t i = 1; i < offset_terms.size(); i++) {
      offset += " + " + offset_terms[i];
    }
    result = fmt("$ + $", offset, result);
  }

  return result;
}

TritonCustomOPTranspileResult
    TritonTranspiler::transpile_kn_custom_op(kn::KNCustomizedOp const *op) {
  tb::Graph const &g = op->bgraph;
  int cur_kernel_idx = kernel_idx_counter++;
  string func_name = fmt("custom_kernel_$", cur_kernel_idx);

  // Generate kernel function
  CodeKeeper code;
  code.e("@triton.jit");
  code.e("def $($, $):",
         func_name,
         map<kn::DTensor, string>(op->output_tensors,
                                  [](kn::DTensor const &dtensor) -> string {
                                    return fmt("dtensor$: tl.tensor",
                                               dtensor.guid);
                                  }),
         map<kn::DTensor, string>(
             op->input_tensors, [](kn::DTensor const &dtensor) -> string {
               return fmt("dtensor$: tl.tensor", dtensor.guid);
             }));
  code.inc_indent();

  // Initialize accumulation tensors
  for (tb::TBOperator *tb_op : g.operators) {
    if (tb_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
      tb::STensor const &stensor = tb_op->output_tensors.at(0);
      std::vector<int> adjusted_dims = adjust_tensor_dims(stensor);
      // STensorMeta meta = stensor_metas.at(stensor.guid);

      // Create accumulator with appropriate shape
      std::string shape = "";
      for (int i = 0; i < stensor.num_dims; i++) {
        // shape += fmt("$,", stensor.dim[i]);
        shape += fmt("$,", adjusted_dims[i]);
      }
      code.e("$ = tl.zeros(($), dtype=tl.float32)",
             fmt("stensor$", stensor.guid),
             shape);
      code.e("# Original shape: ($)", get_tensor_shape(stensor));
    }
  }
  // Generate forloop if needed
  if (g.forloop_range > 1) {
    code.e("for i in range($):", g.forloop_range);
    code.inc_indent();
  }

  // Generate code for operators before accum
  for (tb::TBOperator *tb_op : g.operators) {
    bool after_accum = false;
    for (tb::STensor const &input : tb_op->input_tensors) {
      if (input.after_accum) {
        after_accum = true;
      }
    }
    if (after_accum) {
      continue;
    }
    switch (tb_op->op_type) {
      case type::TB_INPUT_OP: {
        tb::TBInputOp const *input_op =
            static_cast<tb::TBInputOp const *>(tb_op);
        kn::DTensor const &dtensor = input_op->dtensor;
        tb::STensor const &stensor = input_op->output_tensors.at(0);
        std::vector<int> adjusted_dims = adjust_tensor_dims(stensor);

        int3 imap = input_op->input_map;
        int forloop_dim = input_op->forloop_dim;

        std::vector<int> stride(dtensor.num_dims, 1);
        for (int i = dtensor.num_dims - 2; i >= 0; i--) {
          stride[i] = stride[i + 1] * dtensor.dim[i + 1];
        }

        std::vector<std::string> dim_exprs;
        for (int i = 0; i < stensor.num_dims; i++) {
          dim_exprs.push_back(get_input_dim_expr(
              i, imap, forloop_dim, g.forloop_range, stensor.dim[i]));
        }
        std::string ptr_expr = dim_exprs[0];
        if (stensor.num_dims > 1) {
          ptr_expr = fmt("($)[:, None] * $ + ($)[None, :] * $",
                         dim_exprs[0],
                         stride[0],
                         dim_exprs[1],
                         stride[1]);
        }
        // Generate mask for valid data
        std::string mask_expr = generate_mask_expr(
            stensor, adjusted_dims, input_op->input_map, input_op->forloop_dim);

        // Generate load instruction
        code.e("$ = tl.load($ + $, mask=$)",
               fmt("stensor$", stensor.guid),
               fmt("dtensor$", dtensor.guid),
               ptr_expr,
               mask_expr.empty() ? "None" : mask_expr);
        break;
      }

      case type::TB_OUTPUT_OP: {
        assert(false && "Cannot have output op before forloop_accum");
        break;
      }

      case type::TB_MATMUL_OP: {
        tb::STensor const &input0 = tb_op->input_tensors.at(0);
        tb::STensor const &input1 = tb_op->input_tensors.at(1);
        tb::STensor const &output = tb_op->output_tensors.at(0);

        std::vector<int> adjusted_dims0 = adjust_tensor_dims(input0);
        std::vector<int> adjusted_dims1 = adjust_tensor_dims(input1);
        std::string mask0 =
            generate_mask_expr(input0, adjusted_dims0, {-1, -1, -1});
        std::string mask1 =
            generate_mask_expr(input1, adjusted_dims1, {-1, -1, -1});

        code.e("$ = tl.dot($, $)",
               fmt("stensor$", output.guid),
               fmt("stensor$.to(tl.float32)", input0.guid),
               fmt("stensor$.to(tl.float32)", input1.guid));
        break;
      }

      case type::TB_FORLOOP_ACCUM_NO_RED_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        code.e("$ += $",
               fmt("stensor$", output.guid),
               fmt("stensor$", input.guid));
        break;
      }

      case type::TB_EXP_OP:
      case type::TB_SILU_OP:
      case type::TB_SQRT_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        assert(input.num_dims == output.num_dims);

        string op_str = operator_type_to_triton(tb_op->op_type);

        code.e("$ = $($)",
               fmt("stensor$", output.guid),
               op_str,
               fmt("stensor$", input.guid));
        break;
      }
      case type::TB_SQUARE_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        code.e("$ = $ * $",
               fmt("stensor$", output.guid),
               fmt("stensor$", input.guid),
               fmt("stensor$", input.guid));
        break;
      }
      case type::TB_MUL_SCALAR_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        tb::TBElementUnaryOp *unary =
            static_cast<tb::TBElementUnaryOp *>(tb_op);
        code.e("$ = $ * $",
               fmt("stensor$", output.guid),
               fmt("stensor$", input.guid),
               unary->scalar);
        break;
      }
      case type::TB_ADD_OP:
      case type::TB_MUL_OP:
      case type::TB_SUB_OP: {
        tb::STensor const &input0 = tb_op->input_tensors.at(0);
        tb::STensor const &input1 = tb_op->input_tensors.at(1);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        assert(input0.num_dims == input1.num_dims);
        assert(input1.num_dims == output.num_dims);
        string op_str = operator_type_to_triton(tb_op->op_type);
        code.e("$ = $ $ $",
               fmt("stensor$", output.guid),
               fmt("stensor$", input0.guid),
               op_str,
               fmt("stensor$", input1.guid));
      }
      case type::TB_DIV_OP: {
        tb::STensor const &input0 = tb_op->input_tensors.at(0);
        tb::STensor const &input1 = tb_op->input_tensors.at(1);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        assert(input0.num_dims == input1.num_dims);
        assert(input1.num_dims == output.num_dims);
        code.e("$ = tl.fdiv($, $)",
               fmt("stensor$", output.guid),
               fmt("stensor$", input0.guid),
               fmt("stensor$", input1.guid));
        break;
      }
      case type::TB_POW_OP: {
        tb::STensor const &input0 = tb_op->input_tensors.at(0);
        tb::STensor const &input1 = tb_op->input_tensors.at(1);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        assert(input0.num_dims == input1.num_dims);
        assert(input1.num_dims == output.num_dims);
        code.e("$ = tl.power($, $)",
               fmt("stensor$", output.guid),
               fmt("stensor$", input0.guid),
               fmt("stensor$", input1.guid));
        break;
      }

      case type::TB_REDUCTION_0_OP:
      case type::TB_REDUCTION_1_OP:
      case type::TB_REDUCTION_2_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        std::vector<int> adjusted_dims = adjust_tensor_dims(input);
        std::string mask_expr =
            generate_mask_expr(input, adjusted_dims, {-1, -1, -1});

        int reduc_dim = tb_op->op_type - type::TB_REDUCTION_0_OP;

        if (!mask_expr.empty()) {
          code.e("$ = tl.sum($ * $, axis=$, keep_dims=True)",
                 fmt("stensor$", output.guid),
                 mask_expr,
                 fmt("stensor$", input.guid),
                 reduc_dim);
        } else {
          code.e("$ = tl.sum($, axis=$, keep_dims=True)",
                 fmt("stensor$", output.guid),
                 fmt("stensor$", input.guid),
                 reduc_dim);
        }
        break;
      }

      default:
        assert(false && "Unsupported threadblock operator");
    }
  }
  if (g.forloop_range > 1) {
    code.dec_indent();
  }

  // Perform post processing after forloop accum
  for (tb::TBOperator *tb_op : g.operators) {
    bool after_accum = false;
    for (tb::STensor const &input : tb_op->input_tensors) {
      if (input.after_accum) {
        after_accum = true;
      }
    }
    if (!after_accum) {
      continue;
    }

    switch (tb_op->op_type) {
      case type::TB_INPUT_OP: {
        assert(false && "Cannot have input op after forloop_accum");
        break;
      }

      case type::TB_OUTPUT_OP: {
        tb::TBOutputOp const *output_op =
            static_cast<tb::TBOutputOp const *>(tb_op);
        kn::DTensor const &dtensor = output_op->dtensor;
        tb::STensor const &stensor = output_op->input_tensors.at(0);
        std::vector<int> adjusted_dims = adjust_tensor_dims(stensor);

        int3 omap = output_op->output_map;

        std::vector<int> stride(dtensor.num_dims, 1);
        for (int i = dtensor.num_dims - 2; i >= 0; i--) {
          stride[i] = stride[i + 1] * dtensor.dim[i + 1];
        }

        std::vector<std::string> dim_exprs;
        for (int i = 0; i < stensor.num_dims; i++) {
          dim_exprs.push_back(get_output_dim_expr(i, omap, stensor.dim[i]));
        }

        std::string ptr_expr;

        ptr_expr = dim_exprs[0];
        if (stensor.num_dims > 1) {
          ptr_expr = fmt("($)[:, None] * $ + ($)[None, :]",
                         dim_exprs[0],
                         stride[0],
                         dim_exprs[1]);
        }
        std::string mask_expr =
            generate_mask_expr(stensor, adjusted_dims, output_op->output_map);

        if (!mask_expr.empty()) {
          code.e("tl.store($ + $, $, mask=$)",
                 fmt("dtensor$", output_op->dtensor.guid),
                 ptr_expr,
                 fmt("stensor$", stensor.guid),
                 mask_expr == "" ? "None" : mask_expr);
        } else {
          code.e("tl.store($ + $, $)",
                 fmt("dtensor$", output_op->dtensor.guid),
                 ptr_expr,
                 fmt("stensor$", stensor.guid));
        }

        break;
      }

      case type::TB_EXP_OP:
      case type::TB_SILU_OP:
      case type::TB_SQRT_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);

        string op_str = operator_type_to_triton(tb_op->op_type);
        code.e("$ = $($)",
               fmt("stensor$", output.guid),
               op_str,
               fmt("stensor$", input.guid));
        break;
      }
      case type::TB_SQUARE_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        code.e("$ = $ * $",
               fmt("stensor$", output.guid),
               fmt("stensor$", input.guid),
               fmt("stensor$", input.guid));
        break;
      }
      case type::TB_ADD_OP:
      case type::TB_MUL_OP:
      case type::TB_SUB_OP: {
        tb::STensor const &input0 = tb_op->input_tensors.at(0);
        tb::STensor const &input1 = tb_op->input_tensors.at(1);
        tb::STensor const &output = tb_op->output_tensors.at(0);

        string op_symbol = "";
        switch (tb_op->op_type) {
          case type::TB_ADD_OP:
            op_symbol = "+";
            break;
          case type::TB_MUL_OP:
            op_symbol = "*";
            break;
          case type::TB_SUB_OP:
            op_symbol = "-";
            break;
          default:
            assert(false);
        }

        code.e("$ = $ $ $",
               fmt("stensor$", output.guid),
               fmt("stensor$", input0.guid),
               op_symbol,
               fmt("stensor$", input1.guid));
        break;
      }
      case type::TB_DIV_OP: {
        tb::STensor const &input0 = tb_op->input_tensors.at(0);
        tb::STensor const &input1 = tb_op->input_tensors.at(1);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        code.e(
            "$ = tl.fdiv($, $)", // TODO: AttributeError: module
                                 // 'triton.language' has no attribute 'div_rn'
            fmt("stensor$", output.guid),
            fmt("stensor$", input0.guid),
            fmt("stensor$", input1.guid));
        break;
      }
      case type::TB_POW_OP: {
        tb::STensor const &input0 = tb_op->input_tensors.at(0);
        tb::STensor const &input1 = tb_op->input_tensors.at(1);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        assert(input0.num_dims == input1.num_dims);
        assert(input1.num_dims == output.num_dims);
        code.e("$ = tl.power($, $)",
               fmt("stensor$", output.guid),
               fmt("stensor$", input0.guid),
               fmt("stensor$", input1.guid));
        break;
      }

      case type::TB_REDUCTION_0_OP:
      case type::TB_REDUCTION_1_OP:
      case type::TB_REDUCTION_2_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        std::vector<int> adjusted_dims = adjust_tensor_dims(input);
        std::string mask_expr =
            generate_mask_expr(input, adjusted_dims, {-1, -1, -1});

        int reduc_dim = tb_op->op_type - type::TB_REDUCTION_0_OP;

        if (!mask_expr.empty()) {
          code.e("$ = tl.sum($ * $, axis=$, keep_dims=True)",
                 fmt("stensor$", output.guid),
                 mask_expr == "" ? "1.0" : mask_expr,
                 fmt("stensor$", input.guid),
                 reduc_dim);
        } else {
          code.e("$ = tl.sum($, axis=$, keep_dims=True)",
                 fmt("stensor$", output.guid),
                 fmt("stensor$", input.guid),
                 reduc_dim);
        }
        break;
      }
      case type::TB_MATMUL_OP: {
        tb::STensor const &input0 = tb_op->input_tensors.at(0);
        tb::STensor const &input1 = tb_op->input_tensors.at(1);
        tb::STensor const &output = tb_op->output_tensors.at(0);

        std::vector<int> adjusted_dims0 = adjust_tensor_dims(input0);
        std::vector<int> adjusted_dims1 = adjust_tensor_dims(input1);
        std::string mask0 =
            generate_mask_expr(input0, adjusted_dims0, {-1, -1, -1});
        std::string mask1 =
            generate_mask_expr(input1, adjusted_dims1, {-1, -1, -1});

        code.e("$ = tl.dot($, $)",
               fmt("stensor$", output.guid),
               fmt("stensor$.to(tl.float32)", input0.guid),
               fmt("stensor$.to(tl.float32)", input1.guid));
        break;
      }
      default: {
        std::cout << "Unsupported op_type: " << tb_op->op_type << std::endl;
        throw std::runtime_error(fmt("Unsupported op_type: $", tb_op->op_type));
      }
    }
  }

  return TritonCustomOPTranspileResult{func_name, code.to_string()};
}

} // namespace triton_transpiler
} // namespace mirage