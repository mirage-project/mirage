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

#include "mirage/pallas_transpiler/transpile.h"

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#include "mirage/kernel/customized.h"
#include "mirage/kernel/element_unary.h"
#include "mirage/threadblock/element_unary.h"
#include "mirage/transpiler/graph_normalizer.h"
#include "mirage/transpiler/utils.h"

namespace mirage {
namespace pallas_transpiler {

namespace {

using mirage::transpiler::CodeKeeper;
using mirage::transpiler::Combine;
using mirage::transpiler::fmt;

struct LoweredBlockSpec {
  std::string name;
  std::vector<int> block_shape;
  std::vector<int> mapped_dims;
  int forloop_dim = -1;
  int forloop_range = 1;
  bool is_output = false;
};

std::string join_strings(std::vector<std::string> const &parts,
                         std::string const &sep,
                         bool trailing_comma_for_single = false) {
  std::ostringstream oss;
  for (size_t i = 0; i < parts.size(); ++i) {
    oss << parts[i];
    if (i + 1 < parts.size()) {
      oss << sep;
    }
  }
  if (trailing_comma_for_single && parts.size() == 1) {
    oss << ",";
  }
  return oss.str();
}

std::string get_dtensor_name(kn::DTensor const &tensor) {
  return fmt("dtensor$", tensor.guid);
}

std::string get_stensor_name(tb::STensor const &tensor) {
  return fmt("stensor$", tensor.guid);
}

std::string get_jnp_dtype(type::DataType dtype) {
  switch (dtype) {
    case type::DT_FLOAT16:
      return "jnp.float16";
    case type::DT_BFLOAT16:
      return "jnp.bfloat16";
    case type::DT_FLOAT32:
      return "jnp.float32";
    case type::DT_INT32:
      return "jnp.int32";
    default:
      return "";
  }
}

std::string get_shape_literal(std::vector<int> const &shape) {
  std::ostringstream oss;
  oss << "(";
  for (size_t i = 0; i < shape.size(); ++i) {
    oss << shape[i];
    if (shape.size() == 1 || i + 1 < shape.size()) {
      oss << ", ";
    }
  }
  oss << ")";
  return oss.str();
}

std::string get_shape_literal(kn::DTensor const &tensor) {
  std::vector<int> shape;
  for (int i = 0; i < tensor.num_dims; ++i) {
    shape.push_back(tensor.dim[i]);
  }
  return get_shape_literal(shape);
}

std::string get_shape_literal(tb::STensor const &tensor) {
  std::vector<int> shape;
  for (int i = 0; i < tensor.num_dims; ++i) {
    shape.push_back(tensor.dim[i]);
  }
  return get_shape_literal(shape);
}

std::string get_kernel_unary_expr(type::TBOperatorType type, std::string const &arg) {
  switch (type) {
    case type::TB_EXP_OP:
      return fmt("jnp.exp($)", arg);
    case type::TB_SQUARE_OP:
      return fmt("jnp.square($)", arg);
    case type::TB_SQRT_OP:
      return fmt("jnp.sqrt($)", arg);
    case type::TB_SILU_OP:
      return fmt("jax.nn.silu($)", arg);
    case type::TB_SIGMOID_OP:
      return fmt("jax.nn.sigmoid($)", arg);
    case type::TB_GELU_OP:
      return fmt("jax.nn.gelu($)", arg);
    case type::TB_RELU_OP:
      return fmt("jax.nn.relu($)", arg);
    case type::TB_LOG_OP:
      return fmt("jnp.log($)", arg);
    default:
      return "";
  }
}

std::string get_kernel_binary_expr(type::TBOperatorType type,
                                   std::string const &lhs,
                                   std::string const &rhs) {
  switch (type) {
    case type::TB_ADD_OP:
      return fmt("($) + ($)", lhs, rhs);
    case type::TB_MUL_OP:
      return fmt("($) * ($)", lhs, rhs);
    case type::TB_DIV_OP:
      return fmt("($) / ($)", lhs, rhs);
    case type::TB_SUB_OP:
      return fmt("($) - ($)", lhs, rhs);
    case type::TB_POW_OP:
      return fmt("jnp.power($, $)", lhs, rhs);
    default:
      return "";
  }
}

std::string get_kn_unary_expr(type::KNOperatorType type, std::string const &arg) {
  switch (type) {
    case type::KN_EXP_OP:
      return fmt("jnp.exp($)", arg);
    case type::KN_SQUARE_OP:
      return fmt("jnp.square($)", arg);
    case type::KN_SQRT_OP:
      return fmt("jnp.sqrt($)", arg);
    case type::KN_SILU_OP:
      return fmt("jax.nn.silu($)", arg);
    case type::KN_SIGMOID_OP:
      return fmt("jax.nn.sigmoid($)", arg);
    case type::KN_GELU_OP:
      return fmt("jax.nn.gelu($)", arg);
    case type::KN_RELU_OP:
      return fmt("jax.nn.relu($)", arg);
    case type::KN_LOG_OP:
      return fmt("jnp.log($)", arg);
    default:
      return "";
  }
}

std::string get_kn_binary_expr(type::KNOperatorType type,
                               std::string const &lhs,
                               std::string const &rhs) {
  switch (type) {
    case type::KN_ADD_OP:
      return fmt("($) + ($)", lhs, rhs);
    case type::KN_MUL_OP:
      return fmt("($) * ($)", lhs, rhs);
    case type::KN_DIV_OP:
      return fmt("($) / ($)", lhs, rhs);
    case type::KN_POW_OP:
      return fmt("jnp.power($, $)", lhs, rhs);
    default:
      return "";
  }
}

LoweredBlockSpec get_block_spec_for_input(tb::TBInputOp const *input_op, dim3 grid_dim,
                                          int forloop_range) {
  LoweredBlockSpec spec;
  spec.name = fmt("input_spec_$", input_op->output_tensors[0].guid);
  spec.forloop_dim = input_op->forloop_dim;
  spec.forloop_range = forloop_range;
  tb::STensor const &stensor = input_op->output_tensors[0];
  for (int i = 0; i < stensor.num_dims; ++i) {
    int dim = stensor.dim[i];
    if (input_op->forloop_dim == i) {
      dim *= forloop_range;
    }
    spec.block_shape.push_back(dim);
  }
  spec.mapped_dims = {input_op->input_map.x, input_op->input_map.y, input_op->input_map.z};
  return spec;
}

LoweredBlockSpec get_block_spec_for_output(tb::TBOutputOp const *output_op, dim3 grid_dim,
                                           int forloop_range) {
  LoweredBlockSpec spec;
  spec.name = fmt("output_spec_$", output_op->dtensor.guid);
  spec.forloop_dim = output_op->forloop_dim;
  spec.forloop_range = forloop_range;
  spec.is_output = true;
  tb::STensor const &stensor = output_op->input_tensors[0];
  for (int i = 0; i < stensor.num_dims; ++i) {
    int dim = stensor.dim[i];
    if (output_op->forloop_dim == i) {
      dim *= forloop_range;
    }
    spec.block_shape.push_back(dim);
  }
  spec.mapped_dims = {output_op->output_map.x, output_op->output_map.y, output_op->output_map.z};
  return spec;
}

std::string get_ref_slice(std::vector<int> const &shape, int forloop_dim, int loop_var = -1) {
  std::ostringstream oss;
  oss << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (static_cast<int>(i) == forloop_dim && loop_var >= 0) {
      oss << fmt("(loop_idx * $):((loop_idx + 1) * $)", shape[i], shape[i]);
    } else {
      oss << ":";
    }
    if (i + 1 < shape.size()) {
      oss << ", ";
    }
  }
  oss << "]";
  return oss.str();
}

std::string get_reduction_to_dimx_expr(std::string const &arg,
                                       tb::STensor const &input,
                                       tb::STensor const &output,
                                       int reduce_dim) {
  std::vector<int> reshaped_dims;
  for (int i = 0; i < input.num_dims; ++i) {
    if (i == reduce_dim) {
      reshaped_dims.push_back(output.dim[i]);
      reshaped_dims.push_back(input.dim[i] / output.dim[i]);
    } else {
      reshaped_dims.push_back(input.dim[i]);
    }
  }
  return fmt("jnp.sum(jnp.reshape($, $), axis=$)",
             arg,
             get_shape_literal(reshaped_dims),
             reduce_dim + 1);
}

bool is_supported_tb_op(type::TBOperatorType type) {
  switch (type) {
    case type::TB_INPUT_OP:
    case type::TB_OUTPUT_OP:
    case type::TB_MATMUL_OP:
    case type::TB_EXP_OP:
    case type::TB_SQUARE_OP:
    case type::TB_SQRT_OP:
    case type::TB_SILU_OP:
    case type::TB_SIGMOID_OP:
    case type::TB_GELU_OP:
    case type::TB_RELU_OP:
    case type::TB_CLAMP_OP:
    case type::TB_LOG_OP:
    case type::TB_MUL_SCALAR_OP:
    case type::TB_ADD_OP:
    case type::TB_MUL_OP:
    case type::TB_DIV_OP:
    case type::TB_SUB_OP:
    case type::TB_POW_OP:
    case type::TB_REDUCTION_0_OP:
    case type::TB_REDUCTION_1_OP:
    case type::TB_REDUCTION_2_OP:
    case type::TB_REDUCTION_0_TO_DIMX_OP:
    case type::TB_REDUCTION_1_TO_DIMX_OP:
    case type::TB_REDUCTION_2_TO_DIMX_OP:
    case type::TB_FORLOOP_ACCUM_NO_RED_OP:
      return true;
    default:
      return false;
  }
}

bool is_supported_kn_op(type::KNOperatorType type) {
  switch (type) {
    case type::KN_INPUT_OP:
    case type::KN_OUTPUT_OP:
    case type::KN_CUSTOMIZED_OP:
    case type::KN_MATMUL_OP:
    case type::KN_EXP_OP:
    case type::KN_SQUARE_OP:
    case type::KN_SQRT_OP:
    case type::KN_SILU_OP:
    case type::KN_SIGMOID_OP:
    case type::KN_GELU_OP:
    case type::KN_RELU_OP:
    case type::KN_CLAMP_OP:
    case type::KN_LOG_OP:
    case type::KN_ADD_OP:
    case type::KN_MUL_OP:
    case type::KN_DIV_OP:
    case type::KN_POW_OP:
    case type::KN_REDUCTION_0_OP:
    case type::KN_REDUCTION_1_OP:
    case type::KN_REDUCTION_2_OP:
      return true;
    default:
      return false;
  }
}

} // namespace

int PallasTranspiler::kernel_idx_counter = 0;

PallasTranspiler::PallasTranspiler(kernel::Graph const *graph,
                                   PallasTranspilerConfig const &config)
    : config(config) {
  for (kn::KNOperator const *op : graph->operators) {
    switch (op->op_type) {
      case type::KN_RMS_NORM_OP:
      case type::KN_ALLREDUCE_OP:
      case type::KN_CONCAT_0_OP:
      case type::KN_CONCAT_1_OP:
      case type::KN_CONCAT_2_OP:
      case type::KN_SPLIT_0_OP:
      case type::KN_SPLIT_1_OP:
      case type::KN_SPLIT_2_OP:
      case type::KN_CHUNK_0_OP:
      case type::KN_CHUNK_1_OP:
      case type::KN_CHUNK_2_OP:
        errors.push_back(
            fmt("Original graph contains an unsupported op for the Pallas v1 backend: $",
                static_cast<int>(op->op_type)));
        g = std::make_shared<kernel::Graph>();
        return;
      default:
        break;
    }

    if (op->op_type == type::KN_CUSTOMIZED_OP) {
      auto const *customized_op = static_cast<kn::KNCustomizedOp const *>(op);
      for (tb::TBOperator const *tb_op : customized_op->bgraph.operators) {
        switch (tb_op->op_type) {
          case type::TB_REDUCTION_0_MAX_OP:
          case type::TB_REDUCTION_1_MAX_OP:
          case type::TB_REDUCTION_2_MAX_OP:
          case type::TB_CONCAT_0_OP:
          case type::TB_CONCAT_1_OP:
          case type::TB_CONCAT_2_OP:
          case type::TB_CONCAT_THEN_MATMUL_OP:
          case type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP:
          case type::TB_FORLOOP_ACCUM_RED_LD_SUM_RESCALE_OP:
          case type::TB_FORLOOP_ACCUM_MAX_OP:
            errors.push_back(
                fmt("Original graph contains an unsupported threadblock op for the Pallas "
                    "v1 backend: $",
                    static_cast<int>(tb_op->op_type)));
            g = std::make_shared<kernel::Graph>();
            return;
          default:
            break;
        }
      }
    }
  }

  auto normalized = mirage::transpiler::normalize_graph(graph);
  g = std::move(normalized.graph);
  mugraph_output_tensors = std::move(normalized.mugraph_output_tensors);
}

std::optional<PallasErrorInfo> PallasTranspiler::validate_graph() {
  if (!errors.empty()) {
    return PallasErrorInfo(errors);
  }
  for (kn::KNOperator const *op : g->operators) {
    if (!is_supported_kn_op(op->op_type)) {
      errors.push_back(fmt("Unsupported kernel op for Pallas backend: $",
                           static_cast<int>(op->op_type)));
      continue;
    }
    for (kn::DTensor const &dtensor : Combine(op->input_tensors, op->output_tensors)) {
      if (get_jnp_dtype(dtensor.data_type).empty()) {
        errors.push_back(fmt("Unsupported dtype for Pallas backend on DTensor $",
                             dtensor.guid));
      }
    }
    if (op->op_type == type::KN_CUSTOMIZED_OP) {
      auto const *customized_op = static_cast<kn::KNCustomizedOp const *>(op);
      if (customized_op->bgraph.grid_dim.x <= 0 || customized_op->bgraph.grid_dim.y <= 0 ||
          customized_op->bgraph.grid_dim.z <= 0) {
        errors.push_back("Invalid threadblock grid dimensions for Pallas backend");
      }
      for (tb::TBOperator const *bop : customized_op->bgraph.operators) {
        if (!is_supported_tb_op(bop->op_type)) {
          errors.push_back(fmt("Unsupported threadblock op for Pallas backend: $",
                               static_cast<int>(bop->op_type)));
          continue;
        }
        for (tb::STensor const &stensor :
             Combine(bop->input_tensors, bop->output_tensors)) {
          if (get_jnp_dtype(stensor.data_type).empty()) {
            errors.push_back(
                fmt("Unsupported dtype for Pallas backend on STensor $", stensor.guid));
          }
        }
        if (bop->op_type == type::TB_MATMUL_OP) {
          tb::STensor const &lhs = bop->input_tensors[0];
          tb::STensor const &rhs = bop->input_tensors[1];
          if (lhs.num_dims != rhs.num_dims || lhs.num_dims < 2) {
            errors.push_back("Pallas backend only supports rank>=2 threadblock matmuls");
          }
        }
      }
    }
  }
  if (!errors.empty()) {
    return PallasErrorInfo(errors);
  }
  return std::nullopt;
}

PallasCustomOPTranspileResult
    PallasTranspiler::transpile_kn_custom_op(kn::KNCustomizedOp const *op) {
  tb::Graph const &tb_graph = op->bgraph;
  int kernel_idx = kernel_idx_counter++;
  std::string func_name = fmt("custom_kernel_$", kernel_idx);

  std::unordered_map<type::GuidType, LoweredBlockSpec> input_specs;
  std::unordered_map<type::GuidType, LoweredBlockSpec> output_specs;
  std::vector<tb::TBInputOp const *> input_ops;
  std::vector<tb::TBOutputOp const *> output_ops;

  for (tb::TBOperator const *tb_op : tb_graph.operators) {
    if (tb_op->op_type == type::TB_INPUT_OP) {
      auto const *input_op = static_cast<tb::TBInputOp const *>(tb_op);
      input_ops.push_back(input_op);
      input_specs[input_op->dtensor.guid] =
          get_block_spec_for_input(input_op, tb_graph.grid_dim, tb_graph.forloop_range);
    } else if (tb_op->op_type == type::TB_OUTPUT_OP) {
      auto const *output_op = static_cast<tb::TBOutputOp const *>(tb_op);
      output_ops.push_back(output_op);
      output_specs[output_op->dtensor.guid] =
          get_block_spec_for_output(output_op, tb_graph.grid_dim, tb_graph.forloop_range);
    }
  }

  CodeKeeper code;
  code.e("def $($):",
         func_name,
         mirage::transpiler::map<kn::DTensor, std::string>(
             op->input_tensors,
             [](kn::DTensor const &dtensor) { return get_dtensor_name(dtensor); }));
  code.inc_indent();
  code.e("grid = ($, $, $)",
         tb_graph.grid_dim.x,
         tb_graph.grid_dim.y,
         tb_graph.grid_dim.z);
  code.e("dimension_semantics = ('parallel', 'parallel', 'parallel')");
  for (tb::TBInputOp const *input_op : input_ops) {
    auto const &spec = input_specs.at(input_op->dtensor.guid);
    code.e("def $(_pid0, _pid1, _pid2):", spec.name);
    code.inc_indent();
    std::vector<std::string> starts;
    for (int dim_idx = 0; dim_idx < input_op->dtensor.num_dims; ++dim_idx) {
      std::vector<std::string> terms;
      for (int axis = 0; axis < 3; ++axis) {
        if (spec.mapped_dims[axis] == dim_idx) {
          terms.push_back(fmt("_pid$ * $", axis, spec.block_shape[dim_idx]));
        }
      }
      starts.push_back(terms.empty() ? "0" : join_strings(terms, " + "));
    }
    code.e("return ($)", join_strings(starts, ", ", true));
    code.dec_indent();
  }
  for (tb::TBOutputOp const *output_op : output_ops) {
    auto const &spec = output_specs.at(output_op->dtensor.guid);
    code.e("def $(_pid0, _pid1, _pid2):", spec.name);
    code.inc_indent();
    std::vector<std::string> starts;
    for (int dim_idx = 0; dim_idx < output_op->dtensor.num_dims; ++dim_idx) {
      std::vector<std::string> terms;
      for (int axis = 0; axis < 3; ++axis) {
        if (spec.mapped_dims[axis] == dim_idx) {
          terms.push_back(fmt("_pid$ * $", axis, spec.block_shape[dim_idx]));
        }
      }
      starts.push_back(terms.empty() ? "0" : join_strings(terms, " + "));
    }
    code.e("return ($)", join_strings(starts, ", ", true));
    code.dec_indent();
  }

  std::vector<std::string> kernel_args;
  for (tb::TBInputOp const *input_op : input_ops) {
    kernel_args.push_back(fmt("$_ref", get_dtensor_name(input_op->dtensor)));
  }
  for (tb::TBOutputOp const *output_op : output_ops) {
    kernel_args.push_back(fmt("$_ref", get_dtensor_name(output_op->dtensor)));
  }
  code.e("def _kernel($):", join_strings(kernel_args, ", "));
  code.inc_indent();

  for (tb::TBOperator const *tb_op : tb_graph.operators) {
    if (tb_op->op_type == type::TB_FORLOOP_ACCUM_NO_RED_OP) {
      tb::STensor const &output = tb_op->output_tensors[0];
      std::string acc_dtype = get_jnp_dtype(output.data_type);
      if (output.data_type != type::DT_FLOAT32) {
        acc_dtype = "jnp.float32";
      }
      code.e("$ = jnp.zeros($, dtype=$)",
             get_stensor_name(output),
             get_shape_literal(output),
             acc_dtype);
    }
  }

  if (tb_graph.forloop_range > 1) {
    code.e("for loop_idx in range($):", tb_graph.forloop_range);
    code.inc_indent();
  }

  for (tb::TBOperator const *tb_op : tb_graph.operators) {
    bool after_accum = false;
    for (tb::STensor const &input : tb_op->input_tensors) {
      after_accum = after_accum || input.after_accum;
    }
    if (after_accum) {
      continue;
    }

    switch (tb_op->op_type) {
      case type::TB_INPUT_OP: {
        auto const *input_op = static_cast<tb::TBInputOp const *>(tb_op);
        tb::STensor const &stensor = input_op->output_tensors[0];
        std::string ref_name = fmt("$_ref", get_dtensor_name(input_op->dtensor));
        std::string slice =
            input_op->forloop_dim >= 0 && tb_graph.forloop_range > 1
                ? get_ref_slice(
                      std::vector<int>(stensor.dim, stensor.dim + stensor.num_dims),
                      input_op->forloop_dim,
                      0)
                : "[...]";
        if (input_op->forloop_dim >= 0 && tb_graph.forloop_range > 1) {
          code.e("$ = jnp.asarray($$)",
                 get_stensor_name(stensor),
                 ref_name,
                 slice);
        } else {
          code.e("$ = jnp.asarray($[...])", get_stensor_name(stensor), ref_name);
        }
        break;
      }
      case type::TB_MATMUL_OP: {
        tb::STensor const &output = tb_op->output_tensors[0];
        code.e("$ = jnp.matmul($, $)",
               get_stensor_name(output),
               get_stensor_name(tb_op->input_tensors[0]),
               get_stensor_name(tb_op->input_tensors[1]));
        break;
      }
      case type::TB_EXP_OP:
      case type::TB_SQUARE_OP:
      case type::TB_SQRT_OP:
      case type::TB_SILU_OP:
      case type::TB_SIGMOID_OP:
      case type::TB_GELU_OP:
      case type::TB_RELU_OP:
      case type::TB_LOG_OP: {
        tb::STensor const &output = tb_op->output_tensors[0];
        code.e("$ = $",
               get_stensor_name(output),
               get_kernel_unary_expr(tb_op->op_type,
                                     get_stensor_name(tb_op->input_tensors[0])));
        break;
      }
      case type::TB_CLAMP_OP: {
        auto const *clamp_op = dynamic_cast<tb::TBClampUnaryOp const *>(tb_op);
        tb::STensor const &output = tb_op->output_tensors[0];
        code.e("$ = jnp.clip($, $, $)",
               get_stensor_name(output),
               get_stensor_name(tb_op->input_tensors[0]),
               clamp_op->min_val,
               clamp_op->max_val);
        break;
      }
      case type::TB_MUL_SCALAR_OP: {
        auto const *unary = dynamic_cast<tb::TBElementUnaryOp const *>(tb_op);
        tb::STensor const &output = tb_op->output_tensors[0];
        code.e("$ = ($) * ($)",
               get_stensor_name(output),
               get_stensor_name(tb_op->input_tensors[0]),
               unary->scalar);
        break;
      }
      case type::TB_ADD_OP:
      case type::TB_MUL_OP:
      case type::TB_DIV_OP:
      case type::TB_SUB_OP:
      case type::TB_POW_OP: {
        tb::STensor const &output = tb_op->output_tensors[0];
        code.e("$ = $",
               get_stensor_name(output),
               get_kernel_binary_expr(tb_op->op_type,
                                      get_stensor_name(tb_op->input_tensors[0]),
                                      get_stensor_name(tb_op->input_tensors[1])));
        break;
      }
      case type::TB_REDUCTION_0_OP:
      case type::TB_REDUCTION_1_OP:
      case type::TB_REDUCTION_2_OP: {
        tb::STensor const &input = tb_op->input_tensors[0];
        tb::STensor const &output = tb_op->output_tensors[0];
        int reduce_dim = tb_op->op_type - type::TB_REDUCTION_0_OP;
        code.e("$ = jnp.sum($, axis=$, keepdims=True)",
               get_stensor_name(output),
               get_stensor_name(input),
               reduce_dim);
        break;
      }
      case type::TB_REDUCTION_0_TO_DIMX_OP:
      case type::TB_REDUCTION_1_TO_DIMX_OP:
      case type::TB_REDUCTION_2_TO_DIMX_OP: {
        tb::STensor const &input = tb_op->input_tensors[0];
        tb::STensor const &output = tb_op->output_tensors[0];
        int reduce_dim = tb_op->op_type - type::TB_REDUCTION_0_TO_DIMX_OP;
        code.e("$ = $",
               get_stensor_name(output),
               get_reduction_to_dimx_expr(get_stensor_name(input), input, output, reduce_dim));
        break;
      }
      case type::TB_FORLOOP_ACCUM_NO_RED_OP: {
        tb::STensor const &output = tb_op->output_tensors[0];
        code.e("$ = $ + $.astype(jnp.float32)",
               get_stensor_name(output),
               get_stensor_name(output),
               get_stensor_name(tb_op->input_tensors[0]));
        break;
      }
      default:
        break;
    }
  }

  if (tb_graph.forloop_range > 1) {
    code.dec_indent();
  }

  for (tb::TBOperator const *tb_op : tb_graph.operators) {
    bool after_accum = false;
    for (tb::STensor const &input : tb_op->input_tensors) {
      after_accum = after_accum || input.after_accum;
    }
    if (!after_accum) {
      continue;
    }

    switch (tb_op->op_type) {
      case type::TB_OUTPUT_OP: {
        auto const *output_op = static_cast<tb::TBOutputOp const *>(tb_op);
        code.e("$_ref[...] = $.astype($_ref.dtype)",
               get_dtensor_name(output_op->dtensor),
               get_stensor_name(output_op->input_tensors[0]),
               get_dtensor_name(output_op->dtensor));
        break;
      }
      case type::TB_EXP_OP:
      case type::TB_SQUARE_OP:
      case type::TB_SQRT_OP:
      case type::TB_SILU_OP:
      case type::TB_SIGMOID_OP:
      case type::TB_GELU_OP:
      case type::TB_RELU_OP:
      case type::TB_LOG_OP: {
        tb::STensor const &output = tb_op->output_tensors[0];
        code.e("$ = $",
               get_stensor_name(output),
               get_kernel_unary_expr(tb_op->op_type,
                                     get_stensor_name(tb_op->input_tensors[0])));
        break;
      }
      case type::TB_CLAMP_OP: {
        auto const *clamp_op = dynamic_cast<tb::TBClampUnaryOp const *>(tb_op);
        tb::STensor const &output = tb_op->output_tensors[0];
        code.e("$ = jnp.clip($, $, $)",
               get_stensor_name(output),
               get_stensor_name(tb_op->input_tensors[0]),
               clamp_op->min_val,
               clamp_op->max_val);
        break;
      }
      case type::TB_MUL_SCALAR_OP: {
        auto const *unary = dynamic_cast<tb::TBElementUnaryOp const *>(tb_op);
        tb::STensor const &output = tb_op->output_tensors[0];
        code.e("$ = ($) * ($)",
               get_stensor_name(output),
               get_stensor_name(tb_op->input_tensors[0]),
               unary->scalar);
        break;
      }
      case type::TB_ADD_OP:
      case type::TB_MUL_OP:
      case type::TB_DIV_OP:
      case type::TB_SUB_OP:
      case type::TB_POW_OP: {
        tb::STensor const &output = tb_op->output_tensors[0];
        code.e("$ = $",
               get_stensor_name(output),
               get_kernel_binary_expr(tb_op->op_type,
                                      get_stensor_name(tb_op->input_tensors[0]),
                                      get_stensor_name(tb_op->input_tensors[1])));
        break;
      }
      case type::TB_REDUCTION_0_OP:
      case type::TB_REDUCTION_1_OP:
      case type::TB_REDUCTION_2_OP: {
        tb::STensor const &input = tb_op->input_tensors[0];
        tb::STensor const &output = tb_op->output_tensors[0];
        int reduce_dim = tb_op->op_type - type::TB_REDUCTION_0_OP;
        code.e("$ = jnp.sum($, axis=$, keepdims=True)",
               get_stensor_name(output),
               get_stensor_name(input),
               reduce_dim);
        break;
      }
      case type::TB_REDUCTION_0_TO_DIMX_OP:
      case type::TB_REDUCTION_1_TO_DIMX_OP:
      case type::TB_REDUCTION_2_TO_DIMX_OP: {
        tb::STensor const &input = tb_op->input_tensors[0];
        tb::STensor const &output = tb_op->output_tensors[0];
        int reduce_dim = tb_op->op_type - type::TB_REDUCTION_0_TO_DIMX_OP;
        code.e("$ = $",
               get_stensor_name(output),
               get_reduction_to_dimx_expr(get_stensor_name(input), input, output, reduce_dim));
        break;
      }
      default:
        break;
    }
  }
  code.dec_indent();

  std::vector<std::string> in_specs;
  for (tb::TBInputOp const *input_op : input_ops) {
    auto const &spec = input_specs.at(input_op->dtensor.guid);
    in_specs.push_back(
        fmt("pl.BlockSpec(block_shape=$, index_map=$, memory_space=pl.ANY)",
            get_shape_literal(spec.block_shape),
            spec.name));
  }
  std::vector<std::string> out_specs;
  std::vector<std::string> out_shapes;
  for (tb::TBOutputOp const *output_op : output_ops) {
    auto const &spec = output_specs.at(output_op->dtensor.guid);
    out_specs.push_back(
        fmt("pl.BlockSpec(block_shape=$, index_map=$, memory_space=pl.ANY)",
            get_shape_literal(spec.block_shape),
            spec.name));
    out_shapes.push_back(fmt("jax.ShapeDtypeStruct($, $)",
                             get_shape_literal(output_op->dtensor),
                             get_jnp_dtype(output_op->dtensor.data_type)));
  }

  code.e("return pl.pallas_call(");
  code.inc_indent();
  code.e("_kernel,");
  if (out_shapes.size() == 1) {
    code.e("out_shape=$,", out_shapes[0]);
  } else {
    code.e("out_shape=($),", join_strings(out_shapes, ", ", true));
  }
  code.e("grid=grid,");
  code.e("in_specs=[$],", join_strings(in_specs, ", "));
  if (out_specs.size() == 1) {
    code.e("out_specs=$,", out_specs[0]);
  } else {
    code.e("out_specs=($),", join_strings(out_specs, ", ", true));
  }
  code.e("interpret=DEBUG_MODE,");
  code.dec_indent();
  code.e(")($)",
         join_strings(mirage::transpiler::map<kn::DTensor, std::string>(
                          op->input_tensors,
                          [](kn::DTensor const &dtensor) {
                            return get_dtensor_name(dtensor);
                          }),
                      ", "));
  code.dec_indent();

  return {func_name, code.to_string()};
}

PallasTranspileResult PallasTranspiler::transpile_ugraph() {
  CodeKeeper code;
  code.e("import jax");
  code.e("import jax.numpy as jnp");
  code.e("from jax.experimental import pallas as pl");
  code.e("from jax.experimental.pallas import tpu as pltpu");
  code.e("");
  code.e("TARGET_CHIP = $", config.target_chip.empty()
                                   ? "None"
                                   : fmt("'$'", config.target_chip));
  code.e("DEBUG_MODE = $", config.debug ? "True" : "False");
  code.e("");

  std::unordered_map<kn::KNOperator const *, PallasCustomOPTranspileResult> custom_results;
  for (kn::KNOperator const *op : g->operators) {
    if (op->op_type == type::KN_CUSTOMIZED_OP) {
      custom_results.emplace(
          op, transpile_kn_custom_op(static_cast<kn::KNCustomizedOp const *>(op)));
      code.e("$", custom_results.at(op).code);
      code.e("");
    }
  }

  std::vector<std::vector<int>> output_shapes;
  std::vector<kn::DTensor> input_dtensors;
  for (kn::KNOperator const *op : g->operators) {
    if (op->op_type == type::KN_INPUT_OP) {
      input_dtensors.push_back(op->output_tensors[0]);
    }
  }
  code.e("def execute_mugraph($):",
         join_strings(mirage::transpiler::map<kn::DTensor, std::string>(
                          input_dtensors,
                          [](kn::DTensor const &dtensor) {
                            return get_dtensor_name(dtensor);
                          }),
                      ", "));
  code.inc_indent();
  code.e("outputs = []");

  for (kn::KNOperator const *op : g->operators) {
    switch (op->op_type) {
      case type::KN_INPUT_OP:
        break;
      case type::KN_CUSTOMIZED_OP: {
        auto const &result = custom_results.at(op);
        std::vector<std::string> out_names;
        for (kn::DTensor const &dtensor : op->output_tensors) {
          out_names.push_back(get_dtensor_name(dtensor));
        }
        if (out_names.size() == 1) {
          code.e("$ = $($)",
                 out_names[0],
                 result.func_name,
                 join_strings(mirage::transpiler::map<kn::DTensor, std::string>(
                                  op->input_tensors,
                                  [](kn::DTensor const &dtensor) {
                                    return get_dtensor_name(dtensor);
                                  }),
                              ", "));
        } else {
          code.e("$ = $($)",
                 join_strings(out_names, ", "),
                 result.func_name,
                 join_strings(mirage::transpiler::map<kn::DTensor, std::string>(
                                  op->input_tensors,
                                  [](kn::DTensor const &dtensor) {
                                    return get_dtensor_name(dtensor);
                                  }),
                              ", "));
        }
        break;
      }
      case type::KN_MATMUL_OP:
        code.e("$ = jnp.matmul($, $)",
               get_dtensor_name(op->output_tensors[0]),
               get_dtensor_name(op->input_tensors[0]),
               get_dtensor_name(op->input_tensors[1]));
        break;
      case type::KN_EXP_OP:
      case type::KN_SQUARE_OP:
      case type::KN_SQRT_OP:
      case type::KN_SILU_OP:
      case type::KN_SIGMOID_OP:
      case type::KN_GELU_OP:
      case type::KN_RELU_OP:
      case type::KN_LOG_OP:
        code.e("$ = $",
               get_dtensor_name(op->output_tensors[0]),
               get_kn_unary_expr(op->op_type, get_dtensor_name(op->input_tensors[0])));
        break;
      case type::KN_CLAMP_OP: {
        auto const *clamp_op = dynamic_cast<kernel::KNClampUnaryOp const *>(op);
        code.e("$ = jnp.clip($, $, $)",
               get_dtensor_name(op->output_tensors[0]),
               get_dtensor_name(op->input_tensors[0]),
               clamp_op->min_val,
               clamp_op->max_val);
        break;
      }
      case type::KN_ADD_OP:
      case type::KN_MUL_OP:
      case type::KN_DIV_OP:
      case type::KN_POW_OP:
        code.e("$ = $",
               get_dtensor_name(op->output_tensors[0]),
               get_kn_binary_expr(op->op_type,
                                  get_dtensor_name(op->input_tensors[0]),
                                  get_dtensor_name(op->input_tensors[1])));
        break;
      case type::KN_REDUCTION_0_OP:
      case type::KN_REDUCTION_1_OP:
      case type::KN_REDUCTION_2_OP:
        code.e("$ = jnp.sum($, axis=$, keepdims=True)",
               get_dtensor_name(op->output_tensors[0]),
               get_dtensor_name(op->input_tensors[0]),
               op->op_type - type::KN_REDUCTION_0_OP);
        break;
      case type::KN_OUTPUT_OP:
        code.e("outputs.append($)", get_dtensor_name(op->input_tensors[0]));
        {
          std::vector<int> shape;
          for (int i = 0; i < op->input_tensors[0].num_dims; ++i) {
            shape.push_back(op->input_tensors[0].dim[i]);
          }
          output_shapes.push_back(std::move(shape));
        }
        break;
      default:
        break;
    }
  }
  code.e("return tuple(outputs)");
  code.dec_indent();

  return PallasTranspileResult(code.to_string(), output_shapes);
}

PallasTranspileResult PallasTranspiler::generate_code() {
  if (auto validation_errors = validate_graph(); validation_errors.has_value()) {
    return PallasTranspileResult("", {}, validation_errors.value());
  }
  return transpile_ugraph();
}

PallasTranspileResult transpile(kernel::Graph const *g,
                                PallasTranspilerConfig const &config) {
  PallasTranspiler transpiler(g, config);
  return transpiler.generate_code();
}

} // namespace pallas_transpiler
} // namespace mirage
