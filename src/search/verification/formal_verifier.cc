#include "mirage/search/verification/formal_verifier.h"
#include "mirage/search/op_utils.h"

#include <iostream>

namespace mirage {
namespace search {

std::mutex FormalVerifier::formal_verifier_mutex;

FormalVerifier::FormalVerifier(kernel::Graph const &input_graph) {
  input_exprs = get_concrete_exprs(input_graph, _ctx, true, all_dims);
}

OutputMatch FormalVerifier::verify(kernel::Graph const &graph) {
  std::lock_guard<std::mutex> lock(formal_verifier_mutex);

  z3::context ctx;

  std::vector<z3::expr> input_exprs_in_current_ctx;
  for (auto const &expr : input_exprs) {
    z3::expr translated = z3::to_expr(ctx, Z3_translate(_ctx, expr, ctx));
    input_exprs_in_current_ctx.push_back(translated);
  }

  std::vector<z3::expr> graph_exprs =
      get_concrete_exprs(graph, ctx, false, all_dims);
  assert(input_exprs.size() == graph_exprs.size());

  auto verify_with_match = [&](OutputMatch const &match) {
    for (size_t i = 0; i < match.size(); i++) {
      if (!is_equivalent(input_exprs_in_current_ctx[i],
                         graph_exprs[match[i]],
                         ctx,
                         all_dims)) {
        return false;
      }
    }
    return true;
  };

  OutputMatch match(input_exprs.size());
  do {
    if (verify_with_match(match)) {
      return match;
    }
  } while (match.next());
  return OutputMatch::invalid_match();
}

std::vector<z3::expr>
    get_concrete_exprs(kernel::Graph const &graph,
                       z3::context &ctx,
                       bool with_output_ops,
                       std::unordered_set<std::string> &all_dims) {
  std::unordered_map<type::GuidType, z3::expr> tensor_exprs;

  z3::sort T = ctx.uninterpreted_sort("Tensor");
  z3::sort D = ctx.uninterpreted_sort("Dim");
  z3::sort I = ctx.int_sort();

  z3::expr data_dim0 = ctx.constant("data_dim0", D);
  z3::expr data_dim1 = ctx.constant("data_dim1", D);
  z3::expr data_dim2 = ctx.constant("data_dim2", D);

  all_dims.insert(data_dim0.to_string());
  all_dims.insert(data_dim1.to_string());
  all_dims.insert(data_dim2.to_string());

  z3::expr data_dim[3] = {data_dim0, data_dim1, data_dim2};

  z3::func_decl ew_add = ctx.function("ew_add", T, T, T);
  z3::func_decl ew_mul = ctx.function("ew_mul", T, T, T);
  z3::func_decl bc_div = ctx.function("bc_div", T, T, T);
  z3::func_decl bc_pow = ctx.function("bc_pow", T, T, T);
  z3::func_decl concat = ctx.function("concat", T, T, D, T);
  z3::func_decl ew_exp = ctx.function("ew_exp", T, T);
  z3::func_decl square = ctx.function("square", T, T);
  z3::func_decl sqrt = ctx.function("sqrt", T, T);
  z3::func_decl matmul = ctx.function("matmul", T, T, T);
  z3::func_decl sum = ctx.function("sum", T, D, T);
  z3::func_decl mean = ctx.function("mean", T, T);
  z3::func_decl rms = ctx.function("rms", T, T);
  z3::func_decl rms_norm = ctx.function("rms_norm", T, T);
  z3::func_decl silu = ctx.function("silu", T, T);
  z3::func_decl gelu = ctx.function("gelu", T, T);
  z3::func_decl relu = ctx.function("relu", T, T);
  z3::func_decl clamp = ctx.function("clamp", T, T);
  z3::func_decl partition = ctx.function("partition", T, D, D, I, T);
  z3::func_decl combine = ctx.function("combine", T, D, D, T);
  z3::func_decl replicate = ctx.function("replicate", T, D, I, T);
  z3::func_decl reduce = ctx.function("reduce", T, D, T);

  int custom_kernel_id = 0;
  int redtox_id = 0;

  auto calc_stensor_exprs = [&](threadblock::Graph const &graph) {
    z3::expr dx =
        ctx.constant(("dx" + std::to_string(custom_kernel_id)).data(), D);
    z3::expr dy =
        ctx.constant(("dy" + std::to_string(custom_kernel_id)).data(), D);
    z3::expr dz =
        ctx.constant(("dz" + std::to_string(custom_kernel_id)).data(), D);
    z3::expr df =
        ctx.constant(("df" + std::to_string(custom_kernel_id)).data(), D);
    all_dims.insert(dx.to_string());
    all_dims.insert(dy.to_string());
    all_dims.insert(dz.to_string());
    all_dims.insert(df.to_string());
    custom_kernel_id++;
    for (threadblock::TBOperator *op : graph.operators) {
      switch (op->op_type) {
        case type::TBOperatorType::TB_INPUT_OP: {
          threadblock::TBInputOp *input_op =
              static_cast<threadblock::TBInputOp *>(op);
          z3::expr a = tensor_exprs.at(input_op->dtensor.guid);
          if (graph.grid_dim.x > 1) {
            if (input_op->input_map.x >= 0) {
              size_t axis =
                  input_op->dtensor.num_dims - 1 - input_op->input_map.x;
              a = partition(
                  a, data_dim[axis], dx, ctx.int_val(graph.grid_dim.x));
            } else {
              a = replicate(a, dx, ctx.int_val(graph.grid_dim.x));
            }
          }
          if (graph.grid_dim.y > 1) {
            if (input_op->input_map.y >= 0) {
              size_t axis =
                  input_op->dtensor.num_dims - 1 - input_op->input_map.y;
              a = partition(
                  a, data_dim[axis], dy, ctx.int_val(graph.grid_dim.y));
            } else {
              a = replicate(a, dy, ctx.int_val(graph.grid_dim.y));
            }
          }
          if (graph.grid_dim.z > 1) {
            if (input_op->input_map.z >= 0) {
              size_t axis =
                  input_op->dtensor.num_dims - 1 - input_op->input_map.z;
              a = partition(
                  a, data_dim[axis], dz, ctx.int_val(graph.grid_dim.z));
            } else {
              a = replicate(a, dz, ctx.int_val(graph.grid_dim.z));
            }
          }
          if (graph.forloop_range > 1) {
            if (input_op->forloop_dim >= 0) {
              size_t axis =
                  input_op->dtensor.num_dims - 1 - input_op->forloop_dim;
              a = partition(
                  a, data_dim[axis], df, ctx.int_val(graph.forloop_range));
            } else {
              a = replicate(a, df, ctx.int_val(graph.forloop_range));
            }
          }
          tensor_exprs.emplace(op->output_tensors[0].guid, a);
          break;
        }
        case type::TBOperatorType::TB_OUTPUT_OP: {
          threadblock::TBOutputOp *output_op =
              static_cast<threadblock::TBOutputOp *>(op);
          z3::expr a = tensor_exprs.at(output_op->input_tensors[0].guid);
          if (graph.grid_dim.x > 1) {
            if (output_op->output_map.x >= 0) {
              size_t axis = output_op->input_tensors[0].num_dims - 1 -
                            output_op->output_map.x;
              a = combine(a, data_dim[axis], dx);
            } else {
              a = reduce(a, dx);
            }
          }
          if (graph.grid_dim.y > 1) {
            if (output_op->output_map.y >= 0) {
              size_t axis = output_op->input_tensors[0].num_dims - 1 -
                            output_op->output_map.y;
              a = combine(a, data_dim[axis], dy);
            } else {
              a = reduce(a, dy);
            }
          }
          if (graph.grid_dim.z > 1) {
            if (output_op->output_map.z >= 0) {
              size_t axis = output_op->input_tensors[0].num_dims - 1 -
                            output_op->output_map.z;
              a = combine(a, data_dim[axis], dz);
            } else {
              a = reduce(a, dz);
            }
          }
          tensor_exprs.emplace(output_op->dtensor.guid, a);
          break;
        }
        case type::TBOperatorType::TB_ADD_OP: {
          z3::expr lhs = tensor_exprs.at(op->input_tensors[0].guid);
          z3::expr rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, ew_add(lhs, rhs));
          break;
        }
        case type::TBOperatorType::TB_CONCAT_0_OP:
        case type::TBOperatorType::TB_CONCAT_1_OP:
        case type::TBOperatorType::TB_CONCAT_2_OP: {
          size_t axis = op->input_tensors[0].num_dims - 1 -
                        (op->op_type - type::TBOperatorType::TB_CONCAT_0_OP);
          z3::expr lhs = tensor_exprs.at(op->input_tensors[0].guid);
          z3::expr rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               concat(lhs, rhs, data_dim[axis]));
          break;
        }
        case type::TBOperatorType::TB_DIV_OP: {
          z3::expr lhs = tensor_exprs.at(op->input_tensors[0].guid);
          z3::expr rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, bc_div(lhs, rhs));
          break;
        }
        case type::TBOperatorType::TB_POW_OP: {
          z3::expr base = tensor_exprs.at(op->input_tensors[0].guid);
          z3::expr exponent = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               bc_pow(base, exponent));
          break;
        }
        case type::TBOperatorType::TB_EXP_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, ew_exp(a));
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          if (graph.forloop_range > 1) {
            a = reduce(a, df);
          }
          tensor_exprs.emplace(op->output_tensors[0].guid, a);
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_RMS_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          if (graph.forloop_range > 1) {
            a = reduce(a, df);
          }
          tensor_exprs.emplace(op->output_tensors[0].guid, rms(a));
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_SUM_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          if (graph.forloop_range > 1) {
            a = reduce(a, df);
          }
          tensor_exprs.emplace(op->output_tensors[0].guid, sum(a, data_dim[0]));
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          if (graph.forloop_range > 1) {
            a = reduce(a, df);
          }
          tensor_exprs.emplace(op->output_tensors[0].guid, mean(a));
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          if (graph.forloop_range > 1) {
            a = reduce(a, df);
          }
          z3::expr reddim =
              ctx.constant(("reddim" + std::to_string(redtox_id++)).data(), D);
          all_dims.insert(reddim.to_string());
          int reduce_degree =
              op->input_tensors[0].dim[op->input_tensors[0].num_dims - 1] /
              op->output_tensors[0].dim[op->output_tensors[0].num_dims - 1];
          a = partition(a, data_dim[0], reddim, ctx.int_val(reduce_degree));
          tensor_exprs.emplace(op->output_tensors[0].guid, reduce(a, reddim));
          break;
        }
        case type::TBOperatorType::TB_MATMUL_OP: {
          z3::expr lhs = tensor_exprs.at(op->input_tensors[0].guid);
          z3::expr rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, matmul(lhs, rhs));
          break;
        }
        case type::TBOperatorType::TB_MUL_OP: {
          z3::expr lhs = tensor_exprs.at(op->input_tensors[0].guid);
          z3::expr rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, ew_mul(lhs, rhs));
          break;
        }
        case type::TBOperatorType::TB_MUL_SCALAR_OP: {
          assert(false && "TODO");
          break;
        }
        case type::TBOperatorType::TB_REDUCTION_0_OP:
        case type::TBOperatorType::TB_REDUCTION_1_OP:
        case type::TBOperatorType::TB_REDUCTION_2_OP: {
          size_t axis = op->input_tensors[0].num_dims - 1 -
                        (op->op_type - type::TBOperatorType::TB_REDUCTION_0_OP);
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               sum(a, data_dim[axis]));
          break;
        }
        case type::TBOperatorType::TB_RMS_NORM_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, rms_norm(a));
          break;
        }
        case type::TBOperatorType::TB_SQUARE_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, square(a));
          break;
        }
        case type::TBOperatorType::TB_SQRT_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, sqrt(a));
          break;
        }
        case type::TBOperatorType::TB_SILU_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, silu(a));
          break;
        }
        case type::TBOperatorType::TB_GELU_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, gelu(a));
          break;
        }
        case type::TBOperatorType::TB_RELU_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, relu(a));
          break;
        }
        case type::TBOperatorType::TB_CLAMP_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, clamp(a));
          break;
        }
        default:
          assert(false && "Unsupported operator type");
      }
    }
  };

  auto calc_dtensor_exprs = [&](kernel::Graph const &graph) {
    int input_id = 0;
    for (kernel::KNOperator *op : graph.operators) {
      switch (op->op_type) {
        case type::KNOperatorType::KN_INPUT_OP: {
          tensor_exprs.emplace(
              op->output_tensors[0].guid,
              ctx.constant(("input" + std::to_string(input_id++)).data(), T));
          break;
        }
        case type::KNOperatorType::KN_OUTPUT_OP: {
          break;
        }
        case type::KNOperatorType::KN_ADD_OP: {
          z3::expr lhs = tensor_exprs.at(op->input_tensors[0].guid);
          z3::expr rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, ew_add(lhs, rhs));
          break;
        }
        case type::KNOperatorType::KN_DIV_OP: {
          z3::expr lhs = tensor_exprs.at(op->input_tensors[0].guid);
          z3::expr rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, bc_div(lhs, rhs));
          break;
        }
        case type::KNOperatorType::KN_POW_OP: {
          z3::expr base = tensor_exprs.at(op->input_tensors[0].guid);
          z3::expr exponent = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               bc_pow(base, exponent));
          break;
        }
        case type::KNOperatorType::KN_EXP_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, ew_exp(a));
          break;
        }
        case type::KNOperatorType::KN_SQUARE_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, square(a));
          break;
        }
        case type::KNOperatorType::KN_SQRT_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, sqrt(a));
          break;
        }
        case type::KNOperatorType::KN_MATMUL_OP: {
          z3::expr lhs = tensor_exprs.at(op->input_tensors[0].guid);
          z3::expr rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, matmul(lhs, rhs));
          break;
        }
        case type::KNOperatorType::KN_MUL_OP: {
          z3::expr lhs = tensor_exprs.at(op->input_tensors[0].guid);
          z3::expr rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, ew_mul(lhs, rhs));
          break;
        }
        case type::KNOperatorType::KN_REDUCTION_0_OP:
        case type::KNOperatorType::KN_REDUCTION_1_OP:
        case type::KNOperatorType::KN_REDUCTION_2_OP: {
          size_t axis = op->input_tensors[0].num_dims - 1 -
                        (op->op_type - type::KNOperatorType::KN_REDUCTION_0_OP);
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               sum(a, data_dim[axis]));
          break;
        }
        case type::KNOperatorType::KN_RMS_NORM_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, rms_norm(a));
          break;
        }
        case type::KNOperatorType::KN_SILU_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, silu(a));
          break;
        }
        case type::KNOperatorType::KN_GELU_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, gelu(a));
          break;
        }
        case type::KNOperatorType::KN_RELU_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, relu(a));
          break;
        }
        case type::KNOperatorType::KN_CLAMP_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, clamp(a));
          break;
        }
        case type::KNOperatorType::KN_CUSTOMIZED_OP: {
          calc_stensor_exprs(static_cast<kernel::KNCustomizedOp *>(op)->bgraph);
          break;
        }
        default:
          assert(false && "Unsupported operator type");
      }
    }
  };

  calc_dtensor_exprs(graph);

  std::vector<z3::expr> output_exprs;

  if (with_output_ops) {
    for (kernel::KNOperator *op : graph.operators) {
      if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
        output_exprs.push_back(tensor_exprs.at(op->input_tensors[0].guid));
      }
    }
  } else {
    for (kernel::KNOperator *op : graph.operators) {
      for (kernel::DTensor const &tensor : op->output_tensors) {
        if (get_num_consumers(graph, tensor) == 0) {
          output_exprs.push_back(tensor_exprs.at(tensor.guid));
        }
      }
    }
  }

  return output_exprs;
}

bool is_equivalent(z3::expr const &lhs,
                   z3::expr const &rhs,
                   z3::context &ctx,
                   std::unordered_set<std::string> const &all_dims) {
  auto to_expr_vector = [&](std::vector<z3::expr> const &vec) {
    z3::expr_vector expr_vec(ctx);
    for (auto const &e : vec) {
      expr_vec.push_back(e);
    }
    return expr_vec;
  };

  z3::solver slv(ctx);
  z3::params p(ctx);

  p.set("mbqi", true);
  p.set("timeout", 1000u);

  slv.set(p);

  z3::sort T = ctx.uninterpreted_sort("Tensor");
  z3::sort D = ctx.uninterpreted_sort("Dim");
  z3::sort I = ctx.int_sort();

  z3::expr data_dim0 = ctx.constant("data_dim0", D);
  z3::expr data_dim1 = ctx.constant("data_dim1", D);
  z3::expr data_dim2 = ctx.constant("data_dim2", D);

  z3::expr data_dim[3] = {data_dim0, data_dim1, data_dim2};

  z3::func_decl ew_add = ctx.function("ew_add", T, T, T);
  z3::func_decl ew_mul = ctx.function("ew_mul", T, T, T);
  z3::func_decl bc_div = ctx.function("bc_div", T, T, T);
  z3::func_decl bc_pow = ctx.function("bc_pow", T, T, T);
  z3::func_decl concat = ctx.function("concat", T, T, D, T);
  z3::func_decl ew_exp = ctx.function("ew_exp", T, T);
  z3::func_decl square = ctx.function("square", T, T);
  z3::func_decl sqrt = ctx.function("sqrt", T, T);
  z3::func_decl matmul = ctx.function("matmul", T, T, T);
  z3::func_decl sum = ctx.function("sum", T, D, T);
  z3::func_decl mean = ctx.function("mean", T, T);
  z3::func_decl rms = ctx.function("rms", T, T);
  z3::func_decl rms_norm = ctx.function("rms_norm", T, T);
  z3::func_decl silu = ctx.function("silu", T, T);
  z3::func_decl gelu = ctx.function("gelu", T, T);
  z3::func_decl relu = ctx.function("relu", T, T);
  z3::func_decl clamp = ctx.function("clamp", T, T);
  z3::func_decl partition = ctx.function("partition", T, D, D, I, T);
  z3::func_decl combine = ctx.function("combine", T, D, D, T);
  z3::func_decl replicate = ctx.function("replicate", T, D, I, T);
  z3::func_decl reduce = ctx.function("reduce", T, D, T);

  z3::expr t0 = ctx.constant("t0", T);
  z3::expr t1 = ctx.constant("t1", T);
  z3::expr t2 = ctx.constant("t2", T);
  z3::expr d0 = ctx.constant("d0", D);
  z3::expr d1 = ctx.constant("d1", D);
  z3::expr d2 = ctx.constant("d2", D);
  z3::expr d3 = ctx.constant("d3", D);
  z3::expr i0 = ctx.constant("i0", I);
  z3::expr i1 = ctx.constant("i1", I);

  for (auto i : all_dims) {
    for (auto j : all_dims) {
      if (i != j) {
        z3::expr dim_i = ctx.constant(i.data(), D);
        z3::expr dim_j = ctx.constant(j.data(), D);
        slv.add(dim_i != dim_j);
      }
    }
  }

  slv.add(forall(t0, t1, ew_add(t0, t1) == ew_add(t1, t0)));
  slv.add(forall(t0, t1, ew_mul(t0, t1) == ew_mul(t1, t0)));
  slv.add(forall(
      t0, t1, t2, ew_add(t0, ew_add(t1, t2)) == ew_add(ew_add(t0, t1), t2)));
  slv.add(forall(
      t0, t1, t2, ew_mul(t0, ew_mul(t1, t2)) == ew_mul(ew_mul(t0, t1), t2)));
  slv.add(forall(
      t0, t1, t2, matmul(t0, matmul(t1, t2)) == matmul(matmul(t0, t1), t2)));
  slv.add(forall(t0,
                 t1,
                 t2,
                 ew_mul(ew_add(t0, t1), t2) ==
                     ew_add(ew_mul(t0, t2), ew_mul(t1, t2))));
  slv.add(forall(t0,
                 t1,
                 t2,
                 bc_div(ew_add(t0, t1), t2) ==
                     ew_add(bc_div(t0, t2), bc_div(t1, t2))));
  slv.add(forall(t0,
                 t1,
                 t2,
                 matmul(ew_add(t0, t1), t2) ==
                     ew_add(matmul(t0, t2), matmul(t1, t2))));
  slv.add(forall(t0,
                 t1,
                 t2,
                 matmul(t0, ew_add(t1, t2)) ==
                     ew_add(matmul(t0, t1), matmul(t0, t2))));
  slv.add(forall(
      t0, t1, t2, matmul(bc_div(t0, t1), t2) == bc_div(matmul(t0, t2), t1)));

  slv.add(forall(t0,
                 t1,
                 t2,
                 bc_pow(ew_mul(t0, t1), t2) ==
                     ew_mul(bc_pow(t0, t2), bc_pow(t1, t2))));
  slv.add(forall(t0,
                 t1,
                 t2,
                 bc_pow(t0, ew_add(t1, t2)) ==
                     ew_mul(bc_pow(t0, t1), bc_pow(t0, t2))));

  slv.add(forall(t0, d0, reduce(sum(t0, d0), d0) == reduce(t0, d0)));

  slv.add(forall(to_expr_vector({t0, d0, d1, i0}),
                 combine(partition(t0, d0, d1, i0), d0, d1) == t0));
  slv.add(forall(to_expr_vector({t0, d0, d1, i0}),
                 partition(combine(t0, d0, d1), d0, d1, i0) == t0));

  {
    z3::expr_vector args = to_expr_vector({t0, d0, d1, d2, i0, i1});
    z3::expr pre_condition = (d0 != d1) && (d0 != d2) && (d1 != d2);

    std::vector<z3::expr> properties = {
        partition(reduce(t0, d0), d1, d2, i0) ==
            reduce(partition(t0, d1, d2, i0), d0),
        combine(reduce(t0, d0), d1, d2) == reduce(combine(t0, d1, d2), d0),
        partition(replicate(t0, d0, i0), d1, d2, i1) ==
            replicate(partition(t0, d1, d2, i1), d0, i0),
        combine(replicate(t0, d0, i0), d1, d2) ==
            replicate(combine(t0, d1, d2), d0, i0),
        partition(sum(t0, d0), d1, d2, i0) ==
            sum(partition(t0, d1, d2, i0), d0),
        combine(sum(t0, d0), d1, d2) == sum(combine(t0, d1, d2), d0),
    };

    for (z3::expr property : properties) {
      slv.add(forall(args, implies(pre_condition, property)));
    }
  }

  {
    z3::expr_vector args = to_expr_vector({t0, d0, d1, d2, d3, i0, i1});
    z3::expr pre_condition =
        (d0 != d2) && (d0 != d3) && (d1 != d2) && (d1 != d3);

    std::vector<z3::expr> properties = {
        partition(partition(t0, d0, d1, i0), d2, d3, i1) ==
            partition(partition(t0, d2, d3, i1), d0, d1, i0),
        combine(combine(t0, d0, d1), d2, d3) ==
            combine(combine(t0, d2, d3), d0, d1),
    };

    for (z3::expr property : properties) {
      slv.add(forall(args, implies(pre_condition, property)));
    }
  }

  {
    z3::expr partitioned_lhs = partition(t0, data_dim2, d0, i0);
    z3::expr partitioned_rhs = partition(t1, data_dim2, d0, i0);
    z3::expr partitioned_matmul = matmul(partitioned_lhs, partitioned_rhs);
    z3::expr combined = combine(partitioned_matmul, data_dim2, d0);
    slv.add(
        forall(to_expr_vector({t0, t1, d0, i0}), matmul(t0, t1) == combined));
  }

  {
    // lemma
    z3::expr partitioned_lhs = partition(t0, data_dim2, d0, i0);
    z3::expr partitioned_rhs = partition(t1, data_dim2, d0, i0);
    z3::expr partitioned_matmul = matmul(partitioned_lhs, partitioned_rhs);

    z3::expr normal_matmul = matmul(t0, t1);
    z3::expr partitioned_result = partition(normal_matmul, data_dim2, d0, i0);
    slv.add(forall(to_expr_vector({t0, t1, d0, i0}),
                   partitioned_matmul == partitioned_result));
  }

  {
    z3::expr row_partitioned_lhs = partition(t0, data_dim1, d0, i0);
    z3::expr replicated_rhs = replicate(t1, d0, i0);
    z3::expr row_partitioned_matmul =
        matmul(row_partitioned_lhs, replicated_rhs);
    z3::expr combined = combine(row_partitioned_matmul, data_dim1, d0);
    slv.add(
        forall(to_expr_vector({t0, t1, d0, i0}), matmul(t0, t1) == combined));
  }

  {
    // lemma
    z3::expr row_partitioned_lhs = partition(t0, data_dim1, d0, i0);
    z3::expr replicated_rhs = replicate(t1, d0, i0);
    z3::expr row_partitioned_matmul =
        matmul(row_partitioned_lhs, replicated_rhs);

    z3::expr normal_matmul = matmul(t0, t1);
    z3::expr partitioned_result = partition(normal_matmul, data_dim1, d0, i0);
    slv.add(forall(to_expr_vector({t0, t1, d0, i0}),
                   row_partitioned_matmul == partitioned_result));
  }

  {
    z3::expr replicated_lhs = replicate(t0, d0, i0);
    z3::expr col_partitioned_rhs = partition(t1, data_dim0, d0, i0);
    z3::expr col_partitioned_matmul =
        matmul(replicated_lhs, col_partitioned_rhs);
    z3::expr combined = combine(col_partitioned_matmul, data_dim0, d0);
    slv.add(
        forall(to_expr_vector({t0, t1, d0, i0}), matmul(t0, t1) == combined));
  }

  {
    // lemma
    z3::expr replicated_lhs = replicate(t0, d0, i0);
    z3::expr col_partitioned_rhs = partition(t1, data_dim0, d0, i0);
    z3::expr col_partitioned_matmul =
        matmul(replicated_lhs, col_partitioned_rhs);

    z3::expr normal_matmul = matmul(t0, t1);
    z3::expr partitioned_result = partition(normal_matmul, data_dim0, d0, i0);
    slv.add(forall(to_expr_vector({t0, t1, d0, i0}),
                   col_partitioned_matmul == partitioned_result));
  }

  {
    z3::expr col_partitioned_lhs = partition(t0, data_dim0, d0, i0);
    z3::expr row_partitioned_rhs = partition(t1, data_dim1, d0, i0);
    z3::expr partial_sum_matmul =
        matmul(col_partitioned_lhs, row_partitioned_rhs);
    z3::expr reduced = reduce(partial_sum_matmul, d0);
    slv.add(
        forall(to_expr_vector({t0, t1, d0, i0}), matmul(t0, t1) == reduced));
  }

  {
    // element-wise binary & bc_div
    std::vector<z3::func_decl> ops = {ew_add, ew_mul, bc_div, bc_pow};
    for (z3::func_decl op : ops) {
      for (z3::expr part_dim : {data_dim2, data_dim1}) {
        z3::expr partitioned_lhs = partition(t0, part_dim, d0, i0);
        z3::expr partitioned_rhs = partition(t1, part_dim, d0, i0);
        z3::expr partitioned_op = op(partitioned_lhs, partitioned_rhs);
        z3::expr combined = combine(partitioned_op, part_dim, d0);
        slv.add(
            forall(to_expr_vector({t0, t1, d0, i0}), op(t0, t1) == combined));
      }
    }

    for (z3::func_decl op : ops) {
      slv.add(
          forall(to_expr_vector({t0, t1, d0, d1, i0}),
                 partition(op(t0, t1), d0, d1, i0) ==
                     op(partition(t0, d0, d1, i0), partition(t1, d0, d1, i0))));
      slv.add(forall(to_expr_vector({t0, t1, d0, d1, i0}),
                     combine(op(t0, t1), d0, d1) ==
                         op(combine(t0, d0, d1), combine(t1, d0, d1))));
      slv.add(forall(t0,
                     d0,
                     i0,
                     replicate(op(t0, t1), d0, i0) ==
                         op(replicate(t0, d0, i0), replicate(t1, d0, i0))));
    }
  }

  {
    // element-wise unary
    std::vector<z3::func_decl> ops = {
        ew_exp, square, sqrt, silu, gelu, relu, clamp};
    for (z3::func_decl op : ops) {
      z3::expr partitioned = partition(t0, d0, d1, i0);
      z3::expr op_partitioned = op(partitioned);
      z3::expr sumed = sum(op_partitioned, d0);
      z3::expr reduced = reduce(sumed, d1);
      slv.add(forall(to_expr_vector({t0, d0, d1, i0}),
                     reduce(op(t0), d0) == reduced));
    }

    for (z3::func_decl op : ops) {
      slv.add(forall(to_expr_vector({t0, d0, d1, i0}),
                     partition(op(t0), d0, d1, i0) ==
                         op(partition(t0, d0, d1, i0))));
      slv.add(forall(to_expr_vector({t0, d0, d1, i0}),
                     combine(op(t0), d0, d1) == op(combine(t0, d0, d1))));
      slv.add(forall(
          t0, d0, i0, replicate(op(t0), d0, i0) == op(replicate(t0, d0, i0))));
    }
  }

  slv.add(lhs != rhs);

  return slv.check() == z3::unsat;
}

} // namespace search
} // namespace mirage
