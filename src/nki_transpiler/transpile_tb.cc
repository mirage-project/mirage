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
#include "mirage/type.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace mirage {
namespace nki_transpiler {

namespace kn = mirage::kernel;
namespace tb = mirage::threadblock;
namespace ty = mirage::type;

using mirage::transpiler::CodeKeeper;
using mirage::transpiler::fmt;
using mirage::transpiler::map;
using std::string;

namespace {
// Todo: Remove the code duplication.
string ugraph_tboperator_type_to_nki(ty::TBOperatorType const type) {
  switch (type) {
    case ty::TB_EXP_OP:
      return "nl.exp";
    case ty::TB_SILU_OP:
      return "nl.silu";
    case ty::TB_SQUARE_OP:
      return "nl.square";
    case ty::TB_SQRT_OP:
      return "nl.sqrt";
    case ty::TB_RELU_OP:
      return "nl.relu";
    case ty::TB_CLAMP_OP:
      return "nl.clamp";
    case ty::TB_MUL_SCALAR_OP:
      return "nl.multiply";
    case ty::TB_ADD_OP:
      return "nl.add";
    case ty::TB_MUL_OP:
      return "nl.multiply";
    case ty::TB_DIV_OP:
      return "nl.divide";
    case ty::TB_SUB_OP:
      return "nl.subtract";
    case ty::TB_POW_OP:
      return "nl.power";
    default:
      assert(false);
  }
}
string ugraph_knoperator_type_to_nki(ty::KNOperatorType const type) {
  switch (type) {
    case ty::KN_EXP_OP:
      return "nl.exp";
    case ty::KN_SILU_OP:
      return "nl.silu";
    case ty::KN_SQUARE_OP:
      return "nl.square";
    case ty::KN_SQRT_OP:
      return "nl.sqrt";
    case ty::KN_RELU_OP:
      return "nl.relu";
    case ty::KN_CLAMP_OP:
      return "nl.clamp";
    case ty::KN_MUL_SCALAR_OP:
      return "nl.multiply";
    case ty::KN_ADD_OP:
      return "nl.add";
    case ty::KN_MUL_OP:
      return "nl.multiply";
    case ty::KN_DIV_OP:
      return "nl.divide";
    case ty::KN_POW_OP:
      return "nl.power";
    default:
      assert(false);
  }
}

string mirage_dtype_to_nki(ty::DataType const dt) {
  string nki_type;
  switch (dt) {
    case ty::DataType::DT_INT4:
      assert(false && "4-bit integer not supported in nki");
      break;
    case ty::DataType::DT_INT8:
      nki_type = "nl.int8";
      break;
    case ty::DataType::DT_UINT16:
      nki_type = "nl.uint16";
      break;
    case ty::DataType::DT_FLOAT8:
      // todo: when we should use e5m2?
      nki_type = "nl.float8_e4m3";
      break;
    case ty::DataType::DT_FLOAT16:
      nki_type = "nl.float16";
      break;
    case ty::DataType::DT_BFLOAT16:
      nki_type = "nl.bfloat16";
      break;
    case ty::DataType::DT_FLOAT32:
      nki_type = "nl.float32";
      break;
    default:
      assert(false && "unsupported nki type in mirage");
      break;
  }
  return nki_type;
}
} // namespace
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
    // For numerical accuracies always prefer float32.
    if (tb_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
      // We determine whether we want to transpose the accum based on
      // tb_op's input. This is because, in the case tb_op's input
      // and output have different partition_dim, we defer the transpose
      // to be performed after forloop to reduce cost
      tb::STensor const &stensor = tb_op->output_tensors.at(0);
      tb::STensor const &before_accum_stensor = tb_op->input_tensors.at(0);
      // Assert that the two tensors have the same shape
      assert(stensor.num_dims == before_accum_stensor.num_dims);
      for (int i = 0; i < stensor.num_dims; i++) {
        assert(stensor.dim[i] == before_accum_stensor.dim[i]);
      }
      // We use before_accum_tensor's as the layout for the accumulator;
      // this can postpone necessary transpose to be perform after
      // for loop
      STensorMeta meta = stensor_metas.at(before_accum_stensor.guid);
      // Assert that only the last two dims can have size larger than 1
      for (int i = 0; i < stensor.num_dims - 2; i++) {
        assert(stensor.dim[i] == 1);
      }
      if (stensor.num_dims == 1) {
        // Create a 1D tile
        code.e("$ = nl.zeros(($), dtype=nl.float32, buffer=nl.sbuf)",
               fmt("stensor$", stensor.guid),
               stensor.dim[0]);
      } else {
        // Create a 2D tile
        if (meta.partition_dim == stensor.num_dims - 2) {
          code.e("$ = nl.zeros(($, $), dtype=nl.float32, buffer=nl.sbuf)",
                 fmt("stensor$", stensor.guid),
                 stensor.dim[stensor.num_dims - 2],
                 stensor.dim[stensor.num_dims - 1]);
        } else {
          assert(meta.partition_dim == stensor.num_dims - 1);
          // num_dims - 1 is the partition_dim
          code.e("$ = nl.zeros(($, $), dtype=nl.float32, buffer=nl.sbuf)",
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
        bool transposed = false;
        std::string range = "";
        std::string nki_dtype = mirage_dtype_to_nki(stensor.data_type);
        if (meta.partition_dim == stensor.num_dims - 2) {
          // Normal case
          code.e("$ = nl.ndarray(($, $), dtype=$, buffer=nl.sbuf)",
                 fmt("stensor$", stensor.guid),
                 stensor.dim[stensor.num_dims - 2],
                 stensor.dim[stensor.num_dims - 1],
                 nki_dtype);
        } else {
          // Tranposed case
          assert(meta.partition_dim == stensor.num_dims - 1);
          // partition dim is the innermost dimension so we need
          // to use load_transposed2d
          transposed = true;
          code.e("$ = nl.ndarray(($, $), dtype=$, buffer=nl.sbuf)",
                 fmt("stensor$", stensor.guid),
                 stensor.dim[stensor.num_dims - 1],
                 stensor.dim[stensor.num_dims - 2],
                 nki_dtype);
        }

        int3 imap = input_op->input_map;
        int forloop_dim = input_op->forloop_dim;
        for (int i = 0; i < stensor.num_dims; i++) {
          std::string index;
          if (imap.x == i) {
            int scale_factor = stensor.dim[i];
            if (forloop_dim == i) {
              scale_factor *= g.forloop_range;
            }
            index = fmt("nl.program_id(0) * $", scale_factor);
          } else if (imap.y == i) {
            int scale_factor = stensor.dim[i];
            if (forloop_dim == i) {
              scale_factor *= g.forloop_range;
            }
            index = fmt("nl.program_id(1) * $", scale_factor);
          } else if (imap.z == i) {
            int scale_factor = stensor.dim[i];
            if (forloop_dim == i) {
              scale_factor *= g.forloop_range;
            }
            index = fmt("nl.program_id(2) * $", scale_factor);
          }
          if (forloop_dim == i) {
            if (index == "") {
              index = fmt("i * $", stensor.dim[i]);
            } else {
              index = index + fmt("+ i * $", stensor.dim[i]);
            }
          }
          if (i == stensor.num_dims - 2) {
            if (index == "") {
              index = fmt("nl.arange($)[:, None]", stensor.dim[i]);
            } else {
              index = index + fmt(" + nl.arange($)[:, None]", stensor.dim[i]);
            }
          } else if (i == stensor.num_dims - 1) {
            if (index == "") {
              index = fmt("nl.arange($)[None, :]", stensor.dim[i]);
            } else {
              index = index + fmt(" + nl.arange($)[None, :]", stensor.dim[i]);
            }
          }
          if (index == "") {
            index = "0";
          }
          range += index;
          if (i < stensor.num_dims - 1) {
            range += ", ";
          }
        }

        code.e("$ = $($[$])",
               fmt("stensor$", stensor.guid),
               transposed ? "nl.load_transpose2d" : "nl.load",
               fmt("dtensor$", dtensor.guid),
               range);
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
        STensorMeta meta2 = stensor_metas.at(output.guid);
        std::string operand0 = fmt("stensor$", input0.guid);
        std::string operand1 = fmt("stensor$", input1.guid);
        int input0_par = 0, input0_contr = 1;
        int input1_par = 1, input1_contr = 0;

        // transpiler should be able to split bigger matmul in the forloop
        // accumulator, this assertion possibly mean a bug in nki transpiler.
        assert(
            !(input0.dim[input0_par] > 512 || input1.dim[input1_par] > 512) &&
            "attempting to compute matmul output shape bigger than "
            "largest possible legal shape(128,512) in nki");

        /* Special case:
           When both input0 and input1 have parallel axis mapped
           to partition dimension, but nki requires contraction
           axis mappend to partition dimension. Since contraction
           axis of both stensor is mapped to free dim, it may be
           possible that sizes are greater than 128, which is
           illegal in nki and we need to split the matmul op into
           (128, n) & (128, m) shapes.
        */
        auto splitMatmulAlongContractionAxis = [&]() {
          // split matmul into (128, n) & (128, m)
          std::string i0_paxis =
              fmt("nl.arange($)[:, None]", input0.dim[input0_par]);
          std::string contr_axis =
              fmt("accidx * 128 + nl.arange(128)[None, :]");
          std::string i1_paxis =
              fmt("nl.arange($)[:, None]", input1.dim[input1_par]);
          std::string op0 =
              fmt("stensor$[$, $]", input0.guid, i0_paxis, contr_axis);
          std::string op1 =
              fmt("stensor$[$, $]", input1.guid, i1_paxis, contr_axis);

          // emit a psum buffer for accum.
          int accum_psize = meta2.partition_dim == 0 ? input0.dim[input0_par]
                                                     : input1.dim[input1_par];
          int accum_fsize = meta2.partition_dim == 0 ? input1.dim[input1_par]
                                                     : input0.dim[input0_par];

          code.e("accum$ = nl.zeros(($, $), dtype=nl.float32, buffer=nl.psum)",
                 output.guid,
                 accum_psize,
                 accum_fsize);

          code.e("for accidx in range(($ + $ - 1) // $):",
                 input0.dim[input0_contr],
                 NeuronArch::pmax,
                 NeuronArch::pmax);
          code.inc_indent();

          if (meta2.partition_dim == 0) {
            code.e("accum$ += nisa.nc_matmul(nl.transpose($), nl.transpose($))",
                   output.guid,
                   op0,
                   op1);
          } else {
            code.e("accum$ += nisa.nc_matmul(nl.transpose($), nl.transpose($))",
                   output.guid,
                   op1,
                   op0);
          }
          code.dec_indent();
          // emit to copy from psum to sbuf, type cast to fp16
          // todo: we need to check whether output tensor is of fp16 or not.
          code.e("stensor$ = nl.copy(accum$, dtype=nl.float16)",
                 output.guid,
                 output.guid);
        };

        if (meta0.partition_dim == input0_par &&
            meta1.partition_dim == input1_par &&
            input0.dim[input0_contr] > 128) { // todo : change to arch value
          splitMatmulAlongContractionAxis();
        } else {
          // Add a nl.transpose if input0's partition dim is not
          // output.num_dims - 1
          if (meta0.partition_dim != output.num_dims - 1) {
            operand0 = fmt("nl.transpose($)", operand0);
          }
          // Add a nl.transpose if input1's partition dim is not
          // output.num_dims - 2
          if (meta1.partition_dim != output.num_dims - 2) {
            operand1 = fmt("nl.transpose($)", operand1);
          }
          if (meta2.partition_dim == output.num_dims - 2) {
            // First oprand: input0
            // Second operand: input1
            code.e("$ = nisa.nc_matmul($, $)",
                   fmt("stensor$", output.guid),
                   operand0,
                   operand1);
          } else {
            // First oprand: input1
            // Second operand: input0
            assert(meta2.partition_dim == output.num_dims - 1);
            code.e("$ = nisa.nc_matmul($, $)",
                   fmt("stensor$", output.guid),
                   operand1,
                   operand0);
          }
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
                 ugraph_tboperator_type_to_nki(tb_op->op_type),
                 fmt("stensor$", input.guid),
                 optional_second_operand);
        } else {
          code.e("$ = $($$)",
                 fmt("stensor$", output.guid),
                 ugraph_tboperator_type_to_nki(tb_op->op_type),
                 fmt("stensor$", input.guid),
                 optional_second_operand);
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
               ugraph_tboperator_type_to_nki(tb_op->op_type),
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
          std::string index;
          if (omap.x == i) {
            index = fmt("nl.program_id(0) * $", stensor.dim[i]);
          } else if (omap.y == i) {
            index = fmt("nl.program_id(1) * $", stensor.dim[i]);
          } else if (omap.z == i) {
            index = fmt("nl.program_id(2) * $", stensor.dim[i]);
          }
          if (i == stensor.num_dims - 2) {
            if (index == "") {
              index = fmt("nl.arange($)[:, None]", stensor.dim[i]);
            } else {
              index = index + fmt(" + nl.arange($)[:, None]", stensor.dim[i]);
            }
          } else if (i == stensor.num_dims - 1) {
            if (index == "") {
              index = fmt("nl.arange($)[None, :]", stensor.dim[i]);
            } else {
              index = index + fmt(" + nl.arange($)[None, :]", stensor.dim[i]);
            }
          }
          if (index == "") {
            index = "0";
          }
          range += index;
          if (i < stensor.num_dims - 1) {
            range += ", ";
          }
        }
        code.e("nl.store($[$], $)",
               fmt("dtensor$", dtensor.guid),
               range,
               need_transpose ? fmt("nl.transpose(stensor$)", stensor.guid)
                              : fmt("stensor$", stensor.guid));
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
                 ugraph_tboperator_type_to_nki(tb_op->op_type),
                 fmt("stensor$", input.guid),
                 optional_second_operand);
        } else {
          code.e("$ = $($$)",
                 fmt("stensor$", output.guid),
                 ugraph_tboperator_type_to_nki(tb_op->op_type),
                 fmt("stensor$", input.guid),
                 optional_second_operand);
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
               ugraph_tboperator_type_to_nki(tb_op->op_type),
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
      case type::TB_REDUCTION_0_TO_DIMX_OP:
      case type::TB_REDUCTION_1_TO_DIMX_OP:
      case type::TB_REDUCTION_2_TO_DIMX_OP: {
        // May need to recompute
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

// generate NKI kernels for supported binary operator at kernel level.
std::optional<NKICustomOPTranspileResult>
    NKITranspiler::transpile_kn_op(kn::KNOperator const *op) {
  kn::KNElementBinaryOp const *binary_op =
      dynamic_cast<kn::KNElementBinaryOp const *>(op);
  if (!binary_op || binary_op->input_tensors[0].num_dims > 2) {
    return std::nullopt;
  }

  // Transpile add,mul,div operators
  static int nki_block_kernel_counter = 0;
  int cur_block_kernel_idx = nki_block_kernel_counter++;
  string func_name = fmt("block_kernel_$", cur_block_kernel_idx);

  // partition size 0th axis - 128, 1th axis 512
  // ToDo: handle dimensions greater than 2
  std::vector<std::pair<string, int>> axis_span{{"ix", 128}, {"jx", 512}};
  auto const &inputs = op->input_tensors;
  auto const &output = op->output_tensors;
  int const num_dims = inputs[0].num_dims;

  // generate function signature
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

  auto emit_affineloops = [&]() {
    for (int i = 0; i < num_dims; i++) {
      code.e("for $ in nl.affine_range(($.shape[$] + $ - 1) // $):",
             axis_span[i].first[0],
             fmt("dtensor$", inputs[0].guid),
             i,
             axis_span[i].second,
             axis_span[i].second);
      code.inc_indent();
    }
  };
  emit_affineloops();

  // emit compute indices
  auto emit_indices = [&]() {
    for (int i = 0; i < num_dims; i++) {
      code.e("$ = $ * $ + nl.arange($)[$, $]",
             axis_span[i].first,
             axis_span[i].first[0],
             axis_span[i].second,
             axis_span[i].second,
             i == 0 ? ":" : "None",
             i == 0 ? "None" : ":");
    }
  };
  emit_indices();

  // generate mask
  std::string mask;
  for (int i = 0; i < num_dims; i++) {
    mask += fmt("($ < $.shape[$])",
                axis_span[i].first,
                fmt("dtensor$", inputs[0].guid),
                i);
    if (i != num_dims - 1) {
      mask += " & ";
    }
  }
  code.e("mask_ = ($)", mask);

  auto emit_load = [&](kn::DTensor const &tensor, std::string const &mk) {
    if (num_dims == 2) {
      code.e("$_tile = nl.load($[$, $], mask = $)",
             fmt("dtensor$", tensor.guid),
             fmt("dtensor$", tensor.guid),
             axis_span[0].first,
             axis_span[1].first,
             mk);
    } else {
      code.e("$_tile = nl.load($[$], mask = $)",
             fmt("dtensor$", tensor.guid),
             fmt("dtensor$", tensor.guid),
             axis_span[0].first,
             mk);
    }
  };
  emit_load(inputs[0], "mask_");
  emit_load(inputs[1], "mask_");

  // compute
  code.e("result_tile = $($, $)",
         ugraph_knoperator_type_to_nki(binary_op->op_type),
         fmt("dtensor$_tile", inputs[0].guid),
         fmt("dtensor$_tile", inputs[1].guid));

  auto emit_store = [&](kn::DTensor const &tensor,
                        std::string const &tile,
                        std::string const &mk) {
    if (num_dims == 2) {
      code.e("nl.store($[$, $], value = $, mask = $)",
             fmt("dtensor$", tensor.guid),
             axis_span[0].first,
             axis_span[1].first,
             tile,
             mk);
    } else {
      code.e("nl.store($[$], value = $, mask = $)",
             fmt("dtensor$", tensor.guid),
             axis_span[0].first,
             tile,
             mk);
    }
  };
  // masked store
  emit_store(output[0], "result_tile", "mask_");

  for (int i = 0; i < num_dims; i++) {
    code.dec_indent();
  }

  return NKICustomOPTranspileResult{func_name, code.to_string()};
}
} // namespace nki_transpiler
} // namespace mirage
