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
#include "mirage/threadblock/element_unary.h"
#include "mirage/threadblock/forloop_accum.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/transpiler/utils.h"

#include <algorithm>

namespace mirage {
namespace nki_transpiler {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;

using mirage::transpiler::CodeKeeper;
using mirage::transpiler::fmt;
using mirage::transpiler::map;
using std::string;

string tb_operator_type_to_nki(type::TBOperatorType type) {
  switch (type) {
    case type::TB_EXP_OP:
      return "nl.exp";
    case type::TB_SILU_OP:
      return "nl.erf";
    case type::TB_SQUARE_OP:
      return "nl.square";
    case type::TB_SQRT_OP:
      return "nl.sqrt";
    case type::TB_MUL_SCALAR_OP:
      return "nl.multiply";
    case type::TB_ADD_OP:
      return "nl.add";
    case type::TB_MUL_OP:
      return "nl.multiply";
    case type::TB_DIV_OP:
      return "nl.divide";
    default:
      assert(false);
  }
}

NKICustomOPTranspileResult
    NKITranspiler::transpile_kn_custom_op(kn::KNCustomizedOp const *op) {
  tb::Graph const &g = op->bgraph;
  static int nki_custom_kernel_idx_counter = 0;
  int cur_custom_kernel_idx = nki_custom_kernel_idx_counter++;
  string func_name = fmt("custom_kernel_$", cur_custom_kernel_idx);

  // Generate code prologue
  CodeKeeper code;
  code.e("@nki_jit");
  code.e("def $($, $):",
         func_name,
         map<kn::DTensor, string>(op->output_tensors,
                                  [](kn::DTensor const &dtensor) -> string {
                                    return fmt("dtensor$", dtensor.guid);
                                  }),
         map<kn::DTensor, string>(op->input_tensors,
                                  [](kn::DTensor const &dtensor) -> string {
                                    return fmt("dtensor$", dtensor.guid);
                                  }));
  code.inc_indent();
  // Initialize all accum stensors
  for (tb::TBOperator *tb_op : g.operators) {
    if (tb_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
      // We determine whether we want to transpose the accum based on
      // tb_op's input. This is because, in the case tb_op's input
      // and output have different partition_dim, we defer the transpose
      // to be performed after forloop to reduce cost
      tb::STensor const &stensor = tb_op->input_tensors.at(0);
      STensorMeta meta = stensor_metas.at(stensor.guid);
      // Assert that only the last two dims can have size larger than 1
      for (int i = 0; i < stensor.num_dims - 2; i++) {
        assert(stensor.dim[i] == 1);
      }
      if (stensor.num_dims == 1) {
        // Create a 1D tile
        code.e("$ = nl.zeros(($), dtype=nl.float16, buffer=nl.sbuf)",
               fmt("stensor$", stensor.guid),
               stensor.dim[0]);
      } else {
        // Create a 2D tile
        if (meta.partition_dim == stensor.num_dims - 2) {
          code.e("$ = nl.zeros(($, $), dtype=nl.float16, buffer=nl.sbuf)",
                 fmt("stensor$", stensor.guid),
                 stensor.dim[stensor.num_dims - 2],
                 stensor.dim[stensor.num_dims - 1]);
        } else {
          assert(meta.partition_dim == stensor.num_dims - 1);
          // num_dims - 1 is the partition_dim
          code.e("$ = nl.zeros(($, $), dtype=nl.float16, buffer=nl.sbuf)",
                 fmt("stensor$", stensor.guid),
                 stensor.dim[stensor.num_dims - 1],
                 stensor.dim[stensor.num_dims - 2]);
        }
      }
    }
  }
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
        STensorMeta meta = stensor_metas.at(stensor.guid);
        // Assert that only the last two dims can have size larger than 1
        for (int i = 0; i < stensor.num_dims - 2; i++) {
          assert(stensor.dim[i] == 1);
        }
        string instruction = "nl.load";
        if (meta.partition_dim == stensor.num_dims - 1) {
          instruction = "nl.load_transpose2d";
        }
        int3 imap = input_op->input_map;
        int forloop_dim = input_op->forloop_dim;
        std::string range = "";
        for (int i = 0; i < stensor.num_dims; i++) {
          if (imap.x == i) {
            range += fmt("$*$:$*$",
                         "nl.program_id(0)",
                         stensor.dim[i],
                         "(nl.program_id(0)+1)",
                         stensor.dim[i]);
          } else if (imap.y == i) {
            range += fmt("$*$:$*$",
                         "nl.program_id(1)",
                         stensor.dim[i],
                         "(nl.program_id(1)+1)",
                         stensor.dim[i]);
          } else if (imap.z == i) {
            range += fmt("$*$:$*$",
                         "nl.program_id(2)",
                         stensor.dim[i],
                         "(nl.program_id(2)+1)",
                         stensor.dim[i]);
          } else if (forloop_dim == i) {
            range +=
                fmt("$*$:$*$", "i", stensor.dim[i], "(i+1)", stensor.dim[i]);
          } else {
            range += fmt("$:$", 0, stensor.dim[i]);
          }
          if (i < stensor.num_dims - 1) {
            range += ",";
          }
        }
        std::string range_suffix = "";
        if (stensor.num_dims == 2 || stensor.num_dims == 1) {
        } else if (stensor.num_dims == 3) {
          range_suffix = "[0]";
        } else if (stensor.num_dims == 4) {
          range_suffix = "[0,0]";
        } else {
          assert(false && "Currently unsupported dim size");
        }
        code.e("$ = $($[$])",
               fmt("stensor$", stensor.guid),
               instruction,
               fmt("dtensor$", dtensor.guid),
               range);
        if (range_suffix.length() > 0) {
          code.e("$ = $$",
                 fmt("stensor$", stensor.guid),
                 fmt("stensor$", stensor.guid),
                 range_suffix);
        }
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
        STensorMeta meta0 = stensor_metas.at(input0.guid);
        STensorMeta meta1 = stensor_metas.at(input1.guid);
        // FIXME:Currently assert no transpose
        assert(meta0.partition_dim == output.num_dims - 1);
        assert(meta1.partition_dim == output.num_dims - 2);
        STensorMeta meta2 = stensor_metas.at(output.guid);
        if (meta2.partition_dim == output.num_dims - 2) {
          // First oprand: input0
          // Second operand: input1
          code.e("$ = nisa.nc_matmul($, $)",
                 fmt("stensor$", output.guid),
                 fmt("stensor$", input0.guid),
                 fmt("stensor$", input1.guid));
        } else {
          // First oprand: input1
          // Second operand: input0
          assert(meta2.partition_dim == output.num_dims - 1);
          code.e("$ = nisa.nc_matmul($, $)",
                 fmt("stensor$", output.guid),
                 fmt("stensor$", input1.guid),
                 fmt("stensor$", input0.guid));
        }
        break;
      }
      case type::TB_FORLOOP_ACCUM_NO_RED_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        STensorMeta meta0 = stensor_metas.at(input.guid);
        STensorMeta meta1 = stensor_metas.at(output.guid);
        assert(input.num_dims == output.num_dims);
        if (meta0.partition_dim != meta1.partition_dim) {
          // Need to perform a transpose before accum
          // we defer the transpose to be after forloop
          // to reduce transpose costs
          // Do nothing here
        }
        code.e("$ += $",
               fmt("stensor$", output.guid),
               fmt("stensor$", input.guid));
        break;
      }
      case type::TB_EXP_OP:
      case type::TB_SILU_OP:
      case type::TB_SQUARE_OP:
      case type::TB_SQRT_OP:
      case type::TB_MUL_SCALAR_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        STensorMeta meta0 = stensor_metas.at(input.guid);
        STensorMeta meta1 = stensor_metas.at(output.guid);
        assert(input.num_dims == output.num_dims);
        string optional_second_operand = "";
        if (tb_op->op_type == type::TB_MUL_SCALAR_OP) {
          tb::TBElementUnaryOp *unary =
              static_cast<tb::TBElementUnaryOp *>(tb_op);
          optional_second_operand = fmt(", $", unary->scalar);
        }
        if (meta0.partition_dim != meta1.partition_dim) {
          // Need a transpose before elementwise
          code.e("$ = $(nl.transpose($)$)",
                 fmt("stensor$", output.guid),
                 tb_operator_type_to_nki(tb_op->op_type),
                 fmt("stensor$", input.guid),
                 optional_second_operand);
        } else {
          code.e("$ = $($$)",
                 fmt("stensor$", output.guid),
                 tb_operator_type_to_nki(tb_op->op_type),
                 fmt("stensor$", input.guid),
                 optional_second_operand);
        }
        break;
      }
      case type::TB_ADD_OP:
      case type::TB_MUL_OP:
      case type::TB_DIV_OP: {
        tb::STensor const &input0 = tb_op->input_tensors.at(0);
        tb::STensor const &input1 = tb_op->input_tensors.at(1);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        STensorMeta meta0 = stensor_metas.at(input0.guid);
        STensorMeta meta1 = stensor_metas.at(input1.guid);
        STensorMeta meta2 = stensor_metas.at(output.guid);
        assert(input0.num_dims == input1.num_dims);
        assert(input1.num_dims == output.num_dims);
        bool transpose0 = false, transpose1 = false;
        if (meta0.partition_dim != meta2.partition_dim) {
          transpose0 = true;
        }
        if (meta1.partition_dim != meta2.partition_dim) {
          transpose1 = true;
        }
        code.e("$ = $($, $)",
               fmt("stensor$", output.guid),
               tb_operator_type_to_nki(tb_op->op_type),
               transpose0 ? fmt("nl.transpose(stensor$)", input0.guid)
                          : fmt("stensor$", input0.guid),
               transpose1 ? fmt("nl.transpose(stensor$)", input1.guid)
                          : fmt("stensor$", input1.guid));
        break;
      }
      case type::TB_REDUCTION_0_OP:
      case type::TB_REDUCTION_1_OP:
      case type::TB_REDUCTION_2_OP: {
        // May need to recompute dim size since we may omit leading dims
        int reduc_dim = tb_op->op_type >= type::TB_REDUCTION_0_TO_DIMX_OP
                            ? tb_op->op_type - type::TB_REDUCTION_0_TO_DIMX_OP
                            : tb_op->op_type - type::TB_REDUCTION_0_OP;
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        STensorMeta meta0 = stensor_metas.at(input.guid);
        STensorMeta meta1 = stensor_metas.at(output.guid);
        // FIXME: currently assume no change of partition dim in reduction
        assert(meta0.partition_dim == meta1.partition_dim);
        int num_dims = input.num_dims;
        assert(input.num_dims == output.num_dims);
        assert(0 <= reduc_dim && reduc_dim < num_dims);
        if (reduc_dim == num_dims - 2) {
          code.e("$ = nl.sum($, axis=0, keepdims=True)",
                 fmt("stensor$", output.guid),
                 fmt("stensor$", input.guid));
        } else if (reduc_dim == num_dims - 1) {
          code.e("$ = nl.sum($, axis=0, keepdims=True)",
                 fmt("stensor$", output.guid),
                 fmt("stensor$", input.guid));
        } else {
          assert(false && "Unsupported reduc_dim");
        }
        break;
      }
      default: {
        assert(false && "Unsupported op_type");
      }
    }
  }
  if (g.forloop_range > 1) {
    code.dec_indent();
  }
  // Perform transpose if the input and output of forloop accum does not
  // use the same partition_dim
  for (tb::TBOperator *tb_op : g.operators) {
    if (tb_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
      tb::STensor const &input = tb_op->input_tensors.at(0);
      tb::STensor const &output = tb_op->output_tensors.at(0);
      STensorMeta meta0 = stensor_metas.at(input.guid);
      STensorMeta meta1 = stensor_metas.at(output.guid);
      if (meta0.partition_dim != meta1.partition_dim) {
        code.e("$ = nl.transpose($)",
               fmt("stensor$", output.guid),
               fmt("stensor$", output.guid));
      }
    }
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
        STensorMeta meta = stensor_metas.at(stensor.guid);
        // Assert that only the last two dims can have size larger than 1
        for (int i = 0; i < stensor.num_dims - 2; i++) {
          assert(stensor.dim[i] == 1);
        }
        bool need_transpose = false;
        if (meta.partition_dim == stensor.num_dims - 1) {
          need_transpose = true;
        }
        int3 omap = output_op->output_map;
        std::string range = "";
        for (int i = 0; i < stensor.num_dims; i++) {
          if (omap.x == i) {
            range += fmt("$*$:$*$",
                         "nl.program_id(0)",
                         stensor.dim[i],
                         "(nl.program_id(0)+1)",
                         stensor.dim[i]);
          } else if (omap.y == i) {
            range += fmt("$*$:$*$",
                         "nl.program_id(1)",
                         stensor.dim[i],
                         "(nl.program_id(1)+1)",
                         stensor.dim[i]);
          } else if (omap.z == i) {
            range += fmt("$*$:$*$",
                         "nl.program_id(2)",
                         stensor.dim[i],
                         "(nl.program_id(2)+1)",
                         stensor.dim[i]);
          } else {
            range += fmt("$:$", 0, stensor.dim[i]);
          }
          if (i < stensor.num_dims - 1) {
            range += ",";
          }
        }
        // Generate code for TB Output
        std::string range_prefix = "";
        if (stensor.num_dims == 1 || stensor.num_dims == 2) {
          // Do nothing
        } else if (stensor.num_dims == 3) {
          range_prefix = "0,";
        } else if (stensor.num_dims == 4) {
          range_prefix = "0,0,";
        } else {
          assert(false && "Currently unsupported dim size");
        }
        code.e("nl.store($[$$], $)",
               fmt("dtensor$", dtensor.guid),
               range_prefix,
               range,
               need_transpose ? fmt("nl.transpose(stensor$)", stensor.guid)
                              : fmt("stensor$", stensor.guid));
        break;
      }
      case type::TB_EXP_OP:
      case type::TB_SILU_OP:
      case type::TB_SQUARE_OP:
      case type::TB_SQRT_OP:
      case type::TB_MUL_SCALAR_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        STensorMeta meta0 = stensor_metas.at(input.guid);
        STensorMeta meta1 = stensor_metas.at(output.guid);
        assert(input.num_dims == output.num_dims);
        string optional_second_operand = "";
        if (tb_op->op_type == type::TB_MUL_SCALAR_OP) {
          tb::TBElementUnaryOp *unary =
              static_cast<tb::TBElementUnaryOp *>(tb_op);
          optional_second_operand = fmt(", $", unary->scalar);
        }
        if (meta0.partition_dim != meta1.partition_dim) {
          // Need a transpose before elementwise
          code.e("$ = $(nl.transpose($)$)",
                 fmt("stensor$", output.guid),
                 tb_operator_type_to_nki(tb_op->op_type),
                 fmt("stensor$", input.guid),
                 optional_second_operand);
        } else {
          code.e("$ = $($$)",
                 fmt("stensor$", output.guid),
                 tb_operator_type_to_nki(tb_op->op_type),
                 fmt("stensor$", input.guid),
                 optional_second_operand);
        }
        break;
      }
      case type::TB_ADD_OP:
      case type::TB_MUL_OP:
      case type::TB_DIV_OP: {
        tb::STensor const &input0 = tb_op->input_tensors.at(0);
        tb::STensor const &input1 = tb_op->input_tensors.at(1);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        STensorMeta meta0 = stensor_metas.at(input0.guid);
        STensorMeta meta1 = stensor_metas.at(input1.guid);
        STensorMeta meta2 = stensor_metas.at(output.guid);
        assert(input0.num_dims == input1.num_dims);
        assert(input1.num_dims == output.num_dims);
        bool transpose0 = false, transpose1 = false;
        if (meta0.partition_dim != meta2.partition_dim) {
          transpose0 = true;
        }
        if (meta1.partition_dim != meta2.partition_dim) {
          transpose1 = true;
        }
        code.e("$ = $($, $)",
               fmt("stensor$", output.guid),
               tb_operator_type_to_nki(tb_op->op_type),
               transpose0 ? fmt("nl.transpose(stensor$)", input0.guid)
                          : fmt("stensor$", input0.guid),
               transpose1 ? fmt("nl.transpose(stensor$)", input1.guid)
                          : fmt("stensor$", input1.guid));
        break;
      }
      case type::TB_REDUCTION_0_OP:
      case type::TB_REDUCTION_1_OP:
      case type::TB_REDUCTION_2_OP: {
        // May need to recompute dim size since we may omit leading dims
        int reduc_dim = tb_op->op_type >= type::TB_REDUCTION_0_TO_DIMX_OP
                            ? tb_op->op_type - type::TB_REDUCTION_0_TO_DIMX_OP
                            : tb_op->op_type - type::TB_REDUCTION_0_OP;
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        STensorMeta meta0 = stensor_metas.at(input.guid);
        STensorMeta meta1 = stensor_metas.at(output.guid);
        // FIXME: currently assume no change of partition dim in reduction
        assert(meta0.partition_dim == meta1.partition_dim);
        int num_dims = input.num_dims;
        assert(input.num_dims == output.num_dims);
        // assert that reduc_dim is among the last two dimensions since
        // we omit all other leading dims (which must have a dim size of 1)
        assert(num_dims - 2 <= reduc_dim && reduc_dim < num_dims);
        // Cannot pick partition dim as the reduce_dim
        assert(reduc_dim != meta0.partition_dim);
        // reduction is perform on axis=1, since axis=0 maps to
        // the partition dim
        code.e("$ = nl.sum($, axis=1, keepdims=True)",
               fmt("stensor$", output.guid),
               fmt("stensor$", input.guid));
        break;
      }
      default: {
        assert(false && fmt("Unsupported op_type:$", tb_op->op_type).c_str());
      }
    }
  }
  return NKICustomOPTranspileResult{func_name, code.to_string()};
}

} // namespace nki_transpiler
} // namespace mirage
