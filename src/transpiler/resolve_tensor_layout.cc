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

#include "mirage/threadblock/smem_tensor.h"
#include "mirage/transpiler/transpiler.h"

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "mirage/type.h"
#include "z3++.h"

#include "mirage/kernel/customized.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/matmul.h"
#include "mirage/threadblock/reduction.h"
#include "mirage/transpiler/utils.h"

namespace mirage {
namespace transpiler {

// Helper functions
// Find the last dimension with stride 1. Return -1 if not found.
static int find_innermost_dim(size_t const strides[], int num_dims) {
  for (int i = num_dims - 1; i >= 0; i--) {
    if (strides[i] == 1) {
      return i;
    }
  }
  return -1;
}

static int find_innermost_dim(std::vector<size_t> const &strides) {
  return find_innermost_dim(strides.data(), strides.size());
}

// Pre-defined cost for every nonperfected operator
namespace cost {
using cost_t = int;

// In kernel-level reduction OP, the cost when innermost_dim == reduction_dim
cost_t KN_REDUCTION_INNERMOST_EQ_REDUC_DIM = 4000;

// In tb level input/output OP, the cost when the stensor do not have the same
// innermost dim as the source dtensor
cost_t TB_INPUT_NO_WIDE_COPY = 4000;
cost_t TB_OUTPUT_NO_WIDE_COPY = 4000;

// In tb level input/output op, the cost when ldmatrix is supported by hardware
// but we do not use it
cost_t TB_MATMUL_NO_LDMATRIX = 10000;

// In tb level input OP, the cost when cp.async is available but cannot be used
// since the innermost dim of the stensor is not the same as the source dtensor
cost_t TB_INPUT_NO_CP_ASYNC = 20000;

// Make a dim swizzled
cost_t SWIZZLE_DIM = 1000;

// The cost of allocate 1B of shared memory for stensor is 1
cost_t SMEM_FACTOR = 1;

} // namespace cost

// This function calculate strides and num_phy_elems
// When strides==nullptr, strides calculation is skipped
void calc_tensor_strides(size_t *strides,
                         size_t &num_phy_elems,
                         int num_dims,
                         int const dims[],
                         int innermost_dim,
                         int datatype_size) {
  // An order of dimensions. We layout elements according to this order
  vector<int> dim_order = {innermost_dim};
  for (int i = num_dims - 1; i >= 0; --i) {
    if (i != innermost_dim) {
      dim_order.push_back(i);
    }
  }
  size_t alignment = std::max(16 / datatype_size, 1);
  size_t cur_stride = 1;
  bool encountered_non1_dim = false;
  for (int dim_idx : dim_order) {
    int cur_dim = dims[dim_idx];
    if (strides != nullptr) {
      strides[dim_idx] = cur_stride;
    }
    if (cur_dim != 1) {
      if (!encountered_non1_dim) {
        cur_stride *= round_to_multiple((size_t)cur_dim, alignment);
        encountered_non1_dim = true;
      } else {
        cur_stride *= cur_dim;
      }
    }
  }
  if (cur_stride == 1) {
    // There is only one element in the tensor, we need to pad it to 16B
    cur_stride = alignment;
  }
  num_phy_elems = cur_stride;
}

// Resolve all tensor layouts
// Determine the innermost dimensions for all dtensors and stensors
// Determine the swizzled dimensions for all stensors
// Determine strides for all tensors
void Transpiler::resolve_tensor_layout() {
  using dguid_t = decltype(kn::DTensor::guid);
  using sguid_t = decltype(tb::STensor::guid);

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
  z3::expr_vector costs(
      ctx); // The optimize objective should be the sum of all costs

  // Create variables denoting whether a dimension is the innermost dimension
  // or a swizzled dimension
  // di_x_y denotes whether the y-th dimension of DTensor x is the innermost dim
  // si_x_y denotes whether the y-th dimension of STensor x is the innermost dim
  // sw_x_y denotes whether the y-th dimension of STensor x is swizzled
  std::unordered_map<dguid_t, std::vector<z3::expr>> d_is_innermost;
  std::unordered_map<sguid_t, std::vector<z3::expr>> s_is_innermost;
  std::unordered_map<sguid_t, std::vector<z3::expr>> s_is_swizzled;
  for (kn::DTensor const &dtensor : all_dtensors) {
    int num_dims = dtensor.num_dims;
    for (int i = 0; i < num_dims; ++i) {
      std::string var_name = fmt("di_$_$", dtensor.guid, i);
      d_is_innermost[dtensor.guid].push_back(ctx.bool_const(var_name.c_str()));
    }
  }
  for (tb::STensor const &stensor : all_stensors) {
    int num_dims = stensor.num_dims;
    for (int i = 0; i < num_dims; ++i) {
      std::string var_name = fmt("si_$_$", stensor.guid, i);
      s_is_innermost[stensor.guid].push_back(ctx.bool_const(var_name.c_str()));
    }
    for (int i = 0; i < num_dims; ++i) {
      std::string var_name = fmt("sw_$_$", stensor.guid, i);
      s_is_swizzled[stensor.guid].push_back(ctx.bool_const(var_name.c_str()));
    }
  }

  // Create equations that limits the number of innermost dimensions to 1,
  // and the limitation that the innermost dimension of a STensor cannot be
  // swizzled
  for (kn::DTensor const &dtensor : all_dtensors) {
    int num_dims = dtensor.num_dims;
    // Every DTensor can only have 1 innermost dim
    z3::expr_vector innermost_exprs(ctx);
    for (int i = 0; i < num_dims; ++i) {
      innermost_exprs.push_back(d_is_innermost[dtensor.guid][i]);
    }
    opt.add(z3::atmost(innermost_exprs, 1));
    opt.add(z3::atleast(innermost_exprs, 1));
  }
  for (tb::STensor const &stensor : all_stensors) {
    int num_dims = stensor.num_dims;
    // Every STensor can only have 1 innermost dim
    z3::expr_vector innermost_exprs(ctx);
    for (int i = 0; i < num_dims; ++i) {
      innermost_exprs.push_back(s_is_innermost[stensor.guid][i]);
    }
    opt.add(z3::atmost(innermost_exprs, 1));
    opt.add(z3::atleast(innermost_exprs, 1));
    // Every STensor can have at most 1 swizzle dim
    z3::expr_vector swizzled_exprs(ctx);
    for (int i = 0; i < num_dims; ++i) {
      swizzled_exprs.push_back(s_is_swizzled[stensor.guid][i]);
    }
    opt.add(z3::atmost(swizzled_exprs, 1));
    // The innermost dim of a STensor cannot be swizzled
    for (int i = 0; i < num_dims; ++i) {
      opt.add(!s_is_swizzled[stensor.guid][i] ||
              !s_is_innermost[stensor.guid][i]);
    }
    // Cost for swizzling a dimension
    for (int i = 0; i < num_dims; ++i) {
      costs.push_back(z3::ite(s_is_swizzled[stensor.guid][i],
                              ctx.int_val(cost::SWIZZLE_DIM),
                              ctx.int_val(0)));
    }
  }

  // Constraits & costs for every kernel-level operator
  int cur_input_idx = 0, cur_output_idx = 0;
  for (kn::KNOperator *const op : this->g->operators) {
    switch (op->op_type) {
      case type::KN_INPUT_OP: {
        // Input OP
        // The innermost dim of the input tensor must match the provided layout
        vector<size_t> const &cur_stride = this->input_strides[cur_input_idx];
        kn::DTensor const &tensor = op->output_tensors.at(0);
        if (tensor.num_dims != (int)cur_stride.size()) {
          throw std::runtime_error(
              fmt("The number of dimensions of the stride of the $th tensor "
                  "($) does not match the tensor's num_dims ($)",
                  cur_input_idx,
                  cur_stride.size(),
                  tensor.num_dims));
        }
        int innermost_dim = find_innermost_dim(cur_stride);
        if (innermost_dim == -1) {
          throw std::runtime_error(
              fmt("No innermost dim found for input tensor $", cur_input_idx));
        }
        opt.add(d_is_innermost[tensor.guid][innermost_dim]);
        cur_input_idx += 1;
        break;
      }
      case type::KN_OUTPUT_OP: {
        // if output strides provided
        if (this->output_strides.size() > cur_output_idx) {
          vector<size_t> const &cur_stride =
              this->output_strides[cur_output_idx];
          kn::DTensor const &tensor = op->input_tensors.at(0);
          if (tensor.num_dims != (int)cur_stride.size()) {
            throw std::runtime_error(
                fmt("The number of dimensions of the stride of the $th tensor "
                    "($) does not match the tensor's num_dims ($)",
                    cur_output_idx,
                    cur_stride.size(),
                    tensor.num_dims));
          }
          int innermost_dim = find_innermost_dim(cur_stride);
          if (innermost_dim == -1) {
            throw std::runtime_error(fmt(
                "No innermost dim found for input tensor $", cur_output_idx));
          }
          opt.add(d_is_innermost[tensor.guid][innermost_dim]);
        }
        cur_output_idx += 1;
        break;
      }
      case type::KN_MATMUL_OP: {
        // Matmul OP
        // The innermost dim of the input & output tensors must be within the
        // last two dims
        kn::DTensor const &lhs = op->input_tensors.at(0);
        kn::DTensor const &rhs = op->input_tensors.at(1);
        kn::DTensor const &output = op->output_tensors.at(0);
        for (kn::DTensor const &tensor : {lhs, rhs, output}) {
          int num_dims = tensor.num_dims;
          assert(num_dims >= 2);
          opt.add(d_is_innermost[tensor.guid][num_dims - 1] ||
                  d_is_innermost[tensor.guid][num_dims - 2]);
        }
        break;
      }
      case type::KN_REDUCTION_0_OP:
      case type::KN_REDUCTION_1_OP:
      case type::KN_REDUCTION_2_OP: {
        // Reduction OP
        int reduc_dim = op->op_type - type::KN_REDUCTION_0_OP;
        kn::DTensor const &input = op->input_tensors.at(0);
        kn::DTensor const &output = op->output_tensors.at(0);
        assert(input.num_dims == output.num_dims);
        // Currently the runtime requires that the input & output have the same
        // innermost dim
        for (int i = 0; i < input.num_dims; ++i) {
          opt.add(d_is_innermost[input.guid][i] ==
                  d_is_innermost[output.guid][i]);
        }
        // If the innermost dim == the reduction dim, add some extra cost
        assert(reduc_dim > 0 && reduc_dim < input.num_dims);
        costs.push_back(
            z3::ite(d_is_innermost[input.guid][reduc_dim],
                    ctx.int_val(cost::KN_REDUCTION_INNERMOST_EQ_REDUC_DIM),
                    ctx.int_val(0)));
        break;
      }
      case type::KN_EXP_OP:
      case type::KN_SQUARE_OP:
      case type::KN_SQRT_OP:
      case type::KN_SILU_OP:
      case type::KN_GELU_OP:
      case type::KN_RELU_OP:
      case type::KN_CLAMP_OP: {
        // Elementwise Unary OP
        kn::DTensor const &input = op->input_tensors.at(0);
        kn::DTensor const &output = op->output_tensors.at(0);
        // Currently the runtime requires that the input & output have the same
        // innermost dim
        assert(input.num_dims == output.num_dims);
        for (int i = 0; i < input.num_dims; ++i) {
          opt.add(d_is_innermost[input.guid][i] ==
                  d_is_innermost[output.guid][i]);
        }
        break;
      }
      case type::KN_ADD_OP:
      case type::KN_MUL_OP:
      case type::KN_DIV_OP:
      case type::KN_POW_OP: {
        // Elementwise Binary OP
        kn::DTensor const &lhs = op->input_tensors.at(0);
        kn::DTensor const &rhs = op->input_tensors.at(1);
        kn::DTensor const &output = op->output_tensors.at(0);
        // Currently the runtime requires that the input & output have the same
        // innermost dim
        assert(lhs.num_dims == rhs.num_dims && lhs.num_dims == output.num_dims);
        for (int i = 0; i < lhs.num_dims; ++i) {
          opt.add(d_is_innermost[lhs.guid][i] ==
                  d_is_innermost[output.guid][i]);
          opt.add(d_is_innermost[rhs.guid][i] ==
                  d_is_innermost[output.guid][i]);
        }
        break;
      }
      case type::KN_CUSTOMIZED_OP: {
        // Will be proceeded later, in the next loop
        break;
      }
      default: {
        assert("Unexpected kernel op type");
      }
    }
  }

  // Constraits & costs for every threadblock-level operator
  for (kn::KNOperator const *kn_op : this->g->operators) {
    if (kn_op->op_type == type::KN_CUSTOMIZED_OP) {
      kn::KNCustomizedOp const *kn_customized_op =
          dynamic_cast<kn::KNCustomizedOp const *>(kn_op);
      tb::Graph const &tb_graph = kn_customized_op->bgraph;
      for (tb::TBOperator const *tb_op : tb_graph.operators) {
        if (is_fused_with_prev[tb_op]) {
          continue;
        }
        // Now tb_op must be the "leader" of a chain of fused ops
        tb::TBOperator const *output_op = fusion_chain[tb_op].back();
        switch (tb_op->op_type) {
          case type::TB_INPUT_OP: {
            // TB input operator
            tb::TBInputOp const *tb_input_op =
                dynamic_cast<tb::TBInputOp const *>(tb_op);
            kn::DTensor const &input = tb_input_op->dtensor;
            tb::STensor const &output = output_op->output_tensors.at(0);
            assert(input.num_dims == output.num_dims);
            if (this->config.target_cc < GPU_CC::T4) {
              // Want to leverage wide copy (uint128_t), so need the innermost
              // dim to be the same
              for (int i = 0; i < input.num_dims; ++i) {
                costs.push_back(
                    z3::ite(d_is_innermost[input.guid][i] &&
                                !s_is_innermost[output.guid][i],
                            ctx.int_val(cost::TB_INPUT_NO_WIDE_COPY),
                            ctx.int_val(0)));
              }
            } else {
              // Want to leverage cp.async copying in uint128_t, so need the
              // innermost dim to be the same
              for (int i = 0; i < input.num_dims; ++i) {
                costs.push_back(z3::ite(d_is_innermost[input.guid][i] &&
                                            !s_is_innermost[output.guid][i],
                                        ctx.int_val(cost::TB_INPUT_NO_CP_ASYNC),
                                        ctx.int_val(0)));
              }
            }
            break;
          }
          case type::TB_OUTPUT_OP: {
            tb::TBOutputOp const *tb_output_op =
                dynamic_cast<tb::TBOutputOp const *>(tb_op);
            assert(tb_op == output_op);
            tb::STensor const &input = tb_output_op->input_tensors.at(0);
            kn::DTensor const &output = tb_output_op->dtensor;
            assert(input.num_dims == output.num_dims);
            if (this->config.target_cc < GPU_CC::H100) {
              // Want to leverage wide copy (uint128_t), so need the innermost
              // dim to be the same
              for (int i = 0; i < input.num_dims; ++i) {
                costs.push_back(
                    z3::ite(s_is_innermost[input.guid][i] &&
                                !d_is_innermost[output.guid][i],
                            ctx.int_val(cost::TB_OUTPUT_NO_WIDE_COPY),
                            ctx.int_val(0)));
              }
            } else {
              // Want to leverage cp.bulk.async (TMA instructions)
              for (int i = 0; i < input.num_dims; ++i) {
                costs.push_back(
                    z3::ite(s_is_innermost[input.guid][i] &&
                                !d_is_innermost[output.guid][i],
                            ctx.int_val(cost::TB_OUTPUT_NO_WIDE_COPY),
                            ctx.int_val(0)));
              }
              // assert(0 && "Not implemented");
            }
            break;
          }
          case type::TB_MATMUL_OP: {
            tb::STensor const &input0 = tb_op->input_tensors.at(0);
            tb::STensor const &input1 = tb_op->input_tensors.at(1);
            tb::STensor const &output = output_op->output_tensors.at(0);
            assert(input0.num_dims == input1.num_dims &&
                   input0.num_dims == output.num_dims);
            int num_dims = input0.num_dims;
            assert(num_dims >= 2);
            // Loading
            if (this->config.target_cc >= GPU_CC::T4 &&
                this->config.target_cc < GPU_CC::H100) {
              // Leverage ldmatrix copying on T4+
              for (tb::STensor const &input : {input0, input1}) {
                // If both dims are not the innermost one, cannot use ldmatrix
                costs.push_back(
                    z3::ite(!s_is_innermost[input.guid][num_dims - 1] &&
                                !s_is_innermost[input.guid][num_dims - 2],
                            ctx.int_val(cost::TB_MATMUL_NO_LDMATRIX),
                            ctx.int_val(0)));
                // Need to swizzle some dimensions
                opt.add(z3::implies(!s_is_innermost[input.guid][num_dims - 1],
                                    s_is_swizzled[input.guid][num_dims - 1]));
                opt.add(z3::implies(!s_is_innermost[input.guid][num_dims - 2],
                                    s_is_swizzled[input.guid][num_dims - 2]));
              }
            } else if (this->config.target_cc == GPU_CC::H100) {
              for (tb::STensor const &input : {input0, input1}) {
                // If both dims are not the innermost one, cannot use ldmatrix
                stensor_metas[input0.guid].m_input = true;
                costs.push_back(
                    z3::ite(!s_is_innermost[input.guid][num_dims - 1] &&
                                !s_is_innermost[input.guid][num_dims - 2],
                            ctx.int_val(cost::TB_MATMUL_NO_LDMATRIX),
                            ctx.int_val(0)));
                // Need to swizzle some dimensions
                opt.add(z3::implies(!s_is_innermost[input.guid][num_dims - 1],
                                    s_is_swizzled[input.guid][num_dims - 1]));
                opt.add(z3::implies(!s_is_innermost[input.guid][num_dims - 2],
                                    s_is_swizzled[input.guid][num_dims - 2]));
              }
            } else {
              // Use normal copying if ldmatrix is not supported by hardware
              assert(0 && "Not implemented");
            }
            // Storing
            if (this->config.target_cc >= GPU_CC::H100) {
              opt.add(z3::implies(!s_is_innermost[output.guid][num_dims - 1],
                                  s_is_swizzled[output.guid][num_dims - 1]));
              opt.add(z3::implies(!s_is_innermost[output.guid][num_dims - 2],
                                  s_is_swizzled[output.guid][num_dims - 2]));
              // no copy needed, for example: SM90_64x64x16_F16F16F16_SS
              // assert(0 && "Not implemented");
            } else {
              // Use normal copying. Need to swizzle some dimensions
              opt.add(z3::implies(!s_is_innermost[output.guid][num_dims - 1],
                                  s_is_swizzled[output.guid][num_dims - 1]));
              opt.add(z3::implies(!s_is_innermost[output.guid][num_dims - 2],
                                  s_is_swizzled[output.guid][num_dims - 2]));
            }
            break;
          }
          case type::TB_EXP_OP:
          case type::TB_SQUARE_OP:
          case type::TB_SQRT_OP:
          case type::TB_SILU_OP:
          case type::TB_GELU_OP:
          case type::TB_RELU_OP:
          case type::TB_CLAMP_OP:
          case type::TB_MUL_SCALAR_OP: {
            tb::STensor const &input = tb_op->input_tensors.at(0);
            tb::STensor const &output = output_op->output_tensors.at(0);
            assert(input.num_dims == output.num_dims);
            int num_dims = input.num_dims;
            // Enumerate the iteration dim (i.e. threads are laid out along that
            // dim) for the op The i-th variable means that the op is performed
            // along the i-th dim
            z3::expr_vector is_op_iter_dim(ctx);
            for (int i = 0; i < num_dims; ++i) {
              std::string var_name = fmt("op_iter_dim_$_$", output.guid, i);
              is_op_iter_dim.push_back(ctx.bool_const(var_name.c_str()));
            }
            opt.add(z3::atmost(is_op_iter_dim, 1));
            opt.add(z3::atleast(is_op_iter_dim, 1));
            // Need to swizzle one dimension if it is not the innermost dim
            for (int i = 0; i < num_dims; ++i) {
              opt.add(z3::implies(is_op_iter_dim[i] &&
                                      !s_is_innermost[input.guid][i],
                                  s_is_swizzled[input.guid][i]));
              opt.add(z3::implies(is_op_iter_dim[i] &&
                                      !s_is_innermost[output.guid][i],
                                  s_is_swizzled[output.guid][i]));
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
            tb::STensor const &output = output_op->output_tensors.at(0);
            assert(input0.num_dims == input1.num_dims &&
                   input0.num_dims == output.num_dims);
            int num_dims = input0.num_dims;
            // Enumerate the iteration dim (i.e. threads lay out along that dim)
            // for the op
            // The i-th variable means that the op is performed along the
            // i-th dim
            z3::expr_vector is_op_iter_dim(ctx);
            for (int i = 0; i < num_dims; ++i) {
              std::string var_name = fmt("op_iter_dim_$_$", output.guid, i);
              is_op_iter_dim.push_back(ctx.bool_const(var_name.c_str()));
            }
            opt.add(z3::atmost(is_op_iter_dim, 1));
            opt.add(z3::atleast(is_op_iter_dim, 1));
            // Need to swizzle one dimension if it is not the innermost dim
            for (int i = 0; i < num_dims; ++i) {
              opt.add(z3::implies(is_op_iter_dim[i] &&
                                      !s_is_innermost[input0.guid][i],
                                  s_is_swizzled[input0.guid][i]));
              opt.add(z3::implies(is_op_iter_dim[i] &&
                                      !s_is_innermost[input1.guid][i],
                                  s_is_swizzled[input1.guid][i]));
              opt.add(z3::implies(is_op_iter_dim[i] &&
                                      !s_is_innermost[output.guid][i],
                                  s_is_swizzled[output.guid][i]));
            }
            break;
          }
          case type::TB_FORLOOP_ACCUM_NO_RED_OP:
          case type::TB_FORLOOP_ACCUM_MAX_OP: {
            assert(tb_op == output_op);
            tb::STensor const &input = tb_op->input_tensors.at(0);
            tb::STensor const &output = tb_op->output_tensors.at(0);
            assert(input.num_dims == output.num_dims);
            int num_dims = input.num_dims;
            z3::expr_vector is_op_iter_dim(ctx);
            for (int i = 0; i < input.num_dims; ++i) {
              std::string var_name = fmt("op_iter_dim_$_$", output.guid, i);
              is_op_iter_dim.push_back(ctx.bool_const(var_name.c_str()));
            }
            opt.add(z3::atmost(is_op_iter_dim, 1));
            opt.add(z3::atleast(is_op_iter_dim, 1));
            for (int i = 0; i < num_dims; ++i) {
              opt.add(z3::implies(is_op_iter_dim[i] &&
                                      !s_is_innermost[input.guid][i],
                                  s_is_swizzled[input.guid][i]));
              opt.add(z3::implies(is_op_iter_dim[i] &&
                                      !s_is_innermost[output.guid][i],
                                  s_is_swizzled[output.guid][i]));
            }
            break;
          }
          case type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP: {
            assert(tb_op == output_op);
            tb::STensor const &input = tb_op->input_tensors.at(0);
            tb::STensor const &rescale = tb_op->input_tensors.at(1);
            tb::STensor const &output = tb_op->output_tensors.at(0);
            assert(input.num_dims == output.num_dims);
            int num_dims = input.num_dims;
            z3::expr_vector is_op_iter_dim(ctx);
            for (int i = 0; i < input.num_dims; ++i) {
              std::string var_name = fmt("op_iter_dim_$_$", output.guid, i);
              is_op_iter_dim.push_back(ctx.bool_const(var_name.c_str()));
            }
            opt.add(z3::atmost(is_op_iter_dim, 1));
            opt.add(z3::atleast(is_op_iter_dim, 1));
            for (int i = 0; i < num_dims; ++i) {
              opt.add(z3::implies(is_op_iter_dim[i] &&
                                      !s_is_innermost[input.guid][i],
                                  s_is_swizzled[input.guid][i]));
              opt.add(z3::implies(is_op_iter_dim[i] &&
                                      !s_is_innermost[output.guid][i],
                                  s_is_swizzled[output.guid][i]));
              opt.add(z3::implies(is_op_iter_dim[i] &&
                                      !s_is_innermost[rescale.guid][i],
                                  s_is_swizzled[rescale.guid][i]));
            }
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
            tb::STensor const &output = output_op->output_tensors.at(0);
            int num_dims = input.num_dims;
            assert(input.num_dims == output.num_dims);
            assert(0 <= reduc_dim && reduc_dim < num_dims);
            // Enumerate the iteration dim
            z3::expr_vector is_op_iter_dim(ctx);
            for (int i = 0; i < num_dims; ++i) {
              std::string var_name = fmt("op_iter_dim_$_$", output.guid, i);
              is_op_iter_dim.push_back(ctx.bool_const(var_name.c_str()));
            }
            opt.add(z3::atmost(is_op_iter_dim, 1));
            opt.add(z3::atleast(is_op_iter_dim, 1));
            // Currently, don't support the reduction dim as the iteration dim
            opt.add(!is_op_iter_dim[reduc_dim]);
            // Need to swizzle one dimension if it is not the innermost dim
            for (int i = 0; i < num_dims; ++i) {
              opt.add(z3::implies(is_op_iter_dim[i] &&
                                      !s_is_innermost[input.guid][i],
                                  s_is_swizzled[input.guid][i]));
              opt.add(z3::implies(is_op_iter_dim[i] &&
                                      !s_is_innermost[output.guid][i],
                                  s_is_swizzled[output.guid][i]));
            }
            break;
          }
          case type::TB_REDUCTION_0_MAX_OP:
          case type::TB_REDUCTION_1_MAX_OP:
          case type::TB_REDUCTION_2_MAX_OP: {
            int reduc_dim = tb_op->op_type - type::TB_REDUCTION_0_MAX_OP;
            tb::STensor const &input = tb_op->input_tensors.at(0);
            tb::STensor const &output = output_op->output_tensors.at(0);
            tb::STensor const &diff = tb_op->output_tensors.at(1);
            int num_dims = input.num_dims;
            assert(input.num_dims == output.num_dims);
            assert(input.num_dims == diff.num_dims);
            assert(0 <= reduc_dim && reduc_dim < num_dims);
            // Enumerate the iteration dim
            z3::expr_vector is_op_iter_dim(ctx);
            for (int i = 0; i < num_dims; ++i) {
              std::string var_name = fmt("op_iter_dim_$_$", output.guid, i);
              is_op_iter_dim.push_back(ctx.bool_const(var_name.c_str()));
            }
            opt.add(z3::atmost(is_op_iter_dim, 1));
            opt.add(z3::atleast(is_op_iter_dim, 1));
            // Currently, don't support the reduction dim as the iteration dim
            opt.add(!is_op_iter_dim[reduc_dim]);
            // Need to swizzle one dimension if it is not the innermost dim
            for (int i = 0; i < num_dims; ++i) {
              opt.add(z3::implies(is_op_iter_dim[i] &&
                                      !s_is_innermost[input.guid][i],
                                  s_is_swizzled[input.guid][i]));
              opt.add(z3::implies(is_op_iter_dim[i] &&
                                      !s_is_innermost[output.guid][i],
                                  s_is_swizzled[output.guid][i]));
              opt.add(z3::implies(is_op_iter_dim[i] &&
                                      !s_is_innermost[diff.guid][i],
                                  s_is_swizzled[diff.guid][i]));
            }
            break;
          }
          case type::TB_CONCAT_0_OP:
          case type::TB_CONCAT_1_OP:
          case type::TB_CONCAT_2_OP: {
            assert(0 && "Not implemented");
            break;
          }
          case type::TB_CONCAT_THEN_MATMUL_OP: {
            assert(0 && "Not implemented");
            break;
          }
          default:
            assert(fmt("Unknown TB op: $", tb_op->op_type).c_str());
        }
      }
    }
  }

  // Add stensors' cost related to shared memory usage
  for (tb::STensor const &stensor : all_stensors) {
    for (int i = 0; i < stensor.num_dims; ++i) {
      size_t num_phy_elems;
      calc_tensor_strides(nullptr,
                          num_phy_elems,
                          stensor.num_dims,
                          stensor.dim,
                          i,
                          type::get_datatype_size(stensor.data_type));
      int smem_usage_cost =
          num_phy_elems * type::get_datatype_size(stensor.data_type);
      costs.push_back(z3::ite(s_is_innermost[stensor.guid][i],
                              ctx.int_val(smem_usage_cost),
                              ctx.int_val(0)));
    }
  }

  // Optimize
  if (costs.empty()) {
    costs.push_back(ctx.int_val(0));
  }
  z3::expr objective = z3::sum(costs);
  opt.minimize(objective);
  z3::check_result check_result = opt.check();
  if (check_result == z3::unsat) {
    // No valid layout found
    throw std::runtime_error("Z3 returned unsat. No valid layout found.");
  } else if (check_result == z3::unknown) {
    // ???
    throw std::runtime_error("Z3 returned unknown.");
  }
  assert(check_result == z3::sat);

  // Retrieve the result
  z3::model m = opt.get_model();
  for (kn::DTensor const &dtensor : all_dtensors) {
    int num_dims = dtensor.num_dims;
    int innermost_dim = -1;
    for (int i = 0; i < num_dims; ++i) {
      z3::expr t = m.eval(d_is_innermost[dtensor.guid][i]);
      if (m.eval(d_is_innermost[dtensor.guid][i]).is_true()) {
        innermost_dim = i;
        break;
      }
    }
    assert(innermost_dim != -1);
    this->dtensor_metas[dtensor.guid].innermost_dim = innermost_dim;
  }
  for (tb::STensor const &stensor : all_stensors) {
    int num_dims = stensor.num_dims;
    int innermost_dim = -1;
    for (int i = 0; i < num_dims; ++i) {
      if (m.eval(s_is_innermost[stensor.guid][i]).is_true()) {
        innermost_dim = i;
        break;
      }
    }
    assert(innermost_dim != -1);
    this->stensor_metas[stensor.guid].innermost_dim = innermost_dim;

    this->stensor_metas[stensor.guid].swizzled_dim = -1;
    for (int i = 0; i < num_dims; ++i) {
      if (m.eval(s_is_swizzled[stensor.guid][i]).is_true()) {
        this->stensor_metas[stensor.guid].swizzled_dim = i;
        // Only swizzle the first met dim since an STensor can only have up to
        // 2 dimensions
        break;
      }
    }
  }

  // At this point we have resolved all innermost dimensions
  // Calculate strides for all tensors
  for (kn::DTensor const &dtensor : all_dtensors) {
    DTensorMeta &meta = this->dtensor_metas[dtensor.guid];
    int num_dims = dtensor.num_dims;
    int innermost_dim = meta.innermost_dim;
    if (meta.is_input) {
      // Input tensor
      // The strides are already provided
      assert(this->input_strides[meta.input_idx].size() == (size_t)num_dims);
      for (int i = 0; i < num_dims; ++i) {
        meta.strides[i] = this->input_strides[meta.input_idx][i];
      }
      assert(meta.strides[innermost_dim] == 1);
    } else if (meta.is_output && (meta.output_idx < output_strides.size())) {
      // with user provided output stride
      size_t total_ele = 1;
      for (int i = 0; i < num_dims; ++i) {
        meta.strides[i] = this->output_strides[meta.output_idx][i];
        total_ele *= dtensor.dim[i];
      }
      meta.num_phy_elems = total_ele;
      assert(meta.strides[innermost_dim] == 1);
    } else {
      // Intermediate tensor or output tensor
      calc_tensor_strides(meta.strides,
                          meta.num_phy_elems,
                          num_dims,
                          dtensor.dim,
                          innermost_dim,
                          type::get_datatype_size(dtensor.data_type));
    }
  }
  for (tb::STensor const &stensor : all_stensors) {
    STensorMeta &meta = this->stensor_metas[stensor.guid];
    int num_dims = stensor.num_dims;
    int innermost_dim = meta.innermost_dim;
    calc_tensor_strides(meta.strides,
                        meta.num_phy_elems,
                        num_dims,
                        stensor.dim,
                        innermost_dim,
                        type::get_datatype_size(stensor.data_type));
  }
}

} // namespace transpiler
} // namespace mirage
