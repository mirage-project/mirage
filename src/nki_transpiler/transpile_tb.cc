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
#include "mirage/nki_transpiler/utils.h"
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

} // namespace
NKICustomOPTranspileResult
    NKITranspiler::transpile_kn_custom_op(kn::KNCustomizedOp const *op) {
  tb::Graph const &g = op->bgraph;
  int cur_custom_kernel_idx = nki_custom_kernel_idx_counter++;
  string func_name = fmt("custom_kernel_$", cur_custom_kernel_idx);

  int iterator_idx_counter = 0;

  // Generate code prologue
  CodeKeeper code;
  code.e("@nki.jit");
  code.e("def $($):",
         func_name,
         map<kn::DTensor, string>(op->input_tensors,
                                  [](kn::DTensor const &dtensor) -> string {
                                    return get_tensor_variable_name(dtensor);
                                  }));
  code.inc_indent();
  code.e("dtype = $.dtype", get_tensor_variable_name(op->input_tensors[0]));
  // Create output tensors
  std::vector<std::string> return_tensors;
  for (tb::TBOperator *tb_op : g.operators) {
    if (tb_op->op_type == type::TB_OUTPUT_OP) {
      tb::TBOutputOp const *output_op =
          static_cast<tb::TBOutputOp const *>(tb_op);
      std::string tensor_id = fmt("dtensor$", output_op->dtensor.guid);
      code.e("$ = $",
             tensor_id,
             allocate_nki_tensor(output_op->dtensor,
                                 -1,
                                 NKITensorInitializer::NONE,
                                 "nl.shared_hbm"));
      return_tensors.push_back(tensor_id);
    }
  }
  // Simulate SPMD by for-loops
  std::vector<std::string> SPMD_iterators;
  for (uint d : to_vector(g.grid_dim)) {
    std::string iterator_name = fmt("iter$", iterator_idx_counter++);
    code.e("for $ in nl.affine_range($):", iterator_name, d);
    SPMD_iterators.push_back(iterator_name);
    code.inc_indent();
  }
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
      code.e("$ = $",
             get_tensor_variable_name(stensor),
             allocate_nki_tensor(stensor,
                                 meta.partition_dim,
                                 NKITensorInitializer::ZERO,
                                 "nl.sbuf"));
    }
  }
  if (g.forloop_range > 1) {
    code.e("for i in nl.affine_range($):", g.forloop_range);
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
        bool transposed = meta.partition_dim == stensor.num_dims - 1;
        int partition_dim_degree =
            get_partition_dimension_degree(stensor, meta.partition_dim);
        int partition_dim_size =
            get_partition_dimension_size(stensor, meta.partition_dim);
        code.e("$ = $",
               get_tensor_variable_name(stensor),
               allocate_nki_tensor(stensor,
                                   meta.partition_dim,
                                   NKITensorInitializer::NONE,
                                   "nl.sbuf"));
        code.e("for idx in nl.affine_range($):", partition_dim_degree);
        code.inc_indent();
        std::vector<std::string> range_for_each_dim;

        int3 imap = input_op->input_map;
        int forloop_dim = input_op->forloop_dim;
        for (int i = 0; i < stensor.num_dims; i++) {
          std::vector<std::string> index_terms;
          for (int j = 0; j < 3; ++j) {
            if (to_vector(imap)[j] == i) {
              int scale_factor = stensor.dim[i];
              if (forloop_dim == i) {
                scale_factor *= g.forloop_range;
              }
              index_terms.push_back(
                  fmt("$ * $", SPMD_iterators[j], scale_factor));
            }
          }
          if (forloop_dim == i) {
            index_terms.push_back(fmt("i * $", stensor.dim[i]));
          }
          int range_size = stensor.dim[i];
          if (meta.partition_dim == i) {
            index_terms.push_back(fmt("idx * $", partition_dim_size));
            range_size = partition_dim_size;
          }
          // if (i == stensor.num_dims - 2) {
          //   index_terms.push_back(fmt("nl.arange($)[:, None]", range_size));
          // } else if (i == stensor.num_dims - 1) {
          //   index_terms.push_back(fmt("nl.arange($)[None, :]", range_size));
          // }
          std::string index = str_join(index_terms, "+", "0");
          if (i >= stensor.num_dims - 2) {
            index = get_dim_start_length(index, range_size);
          }
          range_for_each_dim.push_back(index);
        }
        std::string range = str_join(range_for_each_dim, ", ");

        code.e("$[:,idx,:] = $($[$])",
               get_tensor_variable_name(stensor),
               transposed ? "nl.load_transpose2d" : "nl.load",
               get_tensor_variable_name(dtensor),
               range);
        code.dec_indent();
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
        int num_dims = input0.num_dims;
        assert(num_dims == 2); // FIXME: currently only support 2D matmul
        std::string accumulator = [&](tb::STensor const &stensor) {
          for (tb::TBOperator *op : g.operators) {
            if (op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
              if (op->input_tensors[0].guid == stensor.guid) {
                return get_tensor_variable_name(op->output_tensors[0]);
              }
            }
          }
          return std::string("");
        }(output);
        if (accumulator.empty()) {
          code.e("$ = $",
                 fmt("stensor$", output.guid),
                 tiled_matmul_function().get_invocation({
                     operand0,
                     operand1,
                     get_python_literal(meta0.partition_dim == 1),
                     get_python_literal(meta1.partition_dim == 1),
                     "dtype",
                 }));
        } else {
          code.e(tiled_matmul_accum_function().get_invocation({
              operand0,
              operand1,
              accumulator,
              get_python_literal(meta0.partition_dim == 1),
          }));
        }
        break;
      }
      case type::TB_FORLOOP_ACCUM_NO_RED_OP: {
        tb::STensor const &input = tb_op->input_tensors.at(0);
        tb::STensor const &output = tb_op->output_tensors.at(0);
        STensorMeta meta0 = stensor_metas.at(input.guid);
        STensorMeta meta1 = stensor_metas.at(output.guid);
        assert(input.num_dims == output.num_dims);
        if (input.owner_op->op_type != type::TB_MATMUL_OP) {
          code.e("$ += $",
                 fmt("stensor$", output.guid),
                 fmt("stensor$", input.guid));
        }
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
        string first_operand = get_tensor_variable_name(input);
        if (meta0.partition_dim != meta1.partition_dim) {
          // Need a transpose before elementwise
          first_operand =
              tiled_transpose_function().get_invocation({first_operand});
        }
        code.e("$ = $($$)",
               fmt("stensor$", output.guid),
               ugraph_tboperator_type_to_nki(tb_op->op_type),
               first_operand,
               optional_second_operand);
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
        string first_operand = get_tensor_variable_name(input0);
        string second_operand = get_tensor_variable_name(input1);
        if (meta0.partition_dim != meta2.partition_dim) {
          first_operand =
              tiled_transpose_function().get_invocation({first_operand});
        }
        if (meta1.partition_dim != meta2.partition_dim) {
          second_operand =
              tiled_transpose_function().get_invocation({second_operand});
        }
        code.e("$ = $($, $)",
               get_tensor_variable_name(output),
               ugraph_tboperator_type_to_nki(tb_op->op_type),
               first_operand,
               second_operand);
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
        string var = get_tensor_variable_name(output);
        code.e("$ = $", var, tiled_transpose_function().get_invocation({var}));
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
        bool partition_dim = meta.partition_dim;
        if (meta.partition_dim == stensor.num_dims - 1) {
          need_transpose = true;
          partition_dim = stensor.num_dims - 2;
          code.e("$ = $",
                 get_tensor_variable_name(stensor),
                 tiled_transpose_function().get_invocation(
                     {get_tensor_variable_name(stensor)}));
        }
        int partition_dim_degree =
            get_partition_dimension_degree(stensor, partition_dim);
        int partition_dim_size =
            get_partition_dimension_size(stensor, partition_dim);
        code.e("for idx in nl.affine_range($):", partition_dim_degree);
        code.inc_indent();
        int3 omap = output_op->output_map;
        std::vector<std::string> range_for_each_dim;
        for (int i = 0; i < stensor.num_dims; i++) {
          std::vector<std::string> index_terms;
          for (int j = 0; j < 3; ++j) {
            if (to_vector(omap)[j] == i) {
              index_terms.push_back(
                  fmt("$ * $", SPMD_iterators[j], stensor.dim[i]));
            }
          }
          int range_size = stensor.dim[i];
          if (meta.partition_dim == i) {
            index_terms.push_back(fmt("idx * $", partition_dim_size));
            range_size = partition_dim_size;
          }
          // if (i == stensor.num_dims - 2) {
          //   index_terms.push_back(fmt("nl.arange($)[:, None]", range_size));
          // } else if (i == stensor.num_dims - 1) {
          //   index_terms.push_back(fmt("nl.arange($)[None, :]", range_size));
          // }
          std::string index = str_join(index_terms, "+", "0");
          if (i >= stensor.num_dims - 2) {
            index = get_dim_start_length(index, range_size);
          }
          range_for_each_dim.push_back(index);
        }
        std::string range = str_join(range_for_each_dim, ",");
        code.e("nl.store($[$], $[:, idx, :])",
               get_tensor_variable_name(dtensor),
               range,
               get_tensor_variable_name(stensor));
        code.dec_indent();
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
        string first_operand = get_tensor_variable_name(input);
        if (meta0.partition_dim != meta1.partition_dim) {
          first_operand =
              tiled_transpose_function().get_invocation({first_operand});
        }
        code.e("$ = $($$)",
               fmt("stensor$", output.guid),
               ugraph_tboperator_type_to_nki(tb_op->op_type),
               first_operand,
               optional_second_operand);
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
        string first_operand = get_tensor_variable_name(input0);
        string second_operand = get_tensor_variable_name(input1);
        if (meta0.partition_dim != meta2.partition_dim) {
          first_operand =
              tiled_transpose_function().get_invocation({first_operand});
        }
        if (meta1.partition_dim != meta2.partition_dim) {
          second_operand =
              tiled_transpose_function().get_invocation({second_operand});
        }
        code.e("$ = $($, $)",
               get_tensor_variable_name(output),
               ugraph_tboperator_type_to_nki(tb_op->op_type),
               first_operand,
               second_operand);
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
        code.e("$ = nl.sum($, axis=-1, keepdims=True)",
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
        code.e("$ = nl.sum($, axis=-11, keepdims=True)",
               fmt("stensor$", output.guid),
               fmt("stensor$", input.guid));
        break;
      }
      default: {
        assert(false && fmt("Unsupported op_type:$", tb_op->op_type).c_str());
      }
    }
  }
  // dec_indent for SPMD for-loops
  for (size_t i = 0; i < SPMD_iterators.size(); ++i) {
    code.dec_indent();
  }
  // Generate return statement
  code.e("return $", fmt("[$,]", transpiler::my_to_string(return_tensors)));
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
  code.e("def $($, dtype):",
         func_name,
         map<kn::DTensor, string>(op->input_tensors,
                                  [](kn::DTensor const &dtensor) -> string {
                                    return get_tensor_variable_name(dtensor);
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
