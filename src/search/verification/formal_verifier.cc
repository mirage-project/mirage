#include "mirage/search/verification/formal_verifier.h"

namespace mirage {
namespace search {

FormalVerifier::FormalVerifier(kernel::Graph const &input_graph) {
  input_exprs = get_concrete_exprs(input_graph, ctx);
}

OutputMatch FormalVerifier::verify(kernel::Graph const &graph) {
  std::vector<z3::expr> graph_exprs = get_concrete_exprs(graph, ctx);
  assert(input_exprs.size() == graph_exprs.size());

  auto verify_with_match = [&](OutputMatch const &match) {
    for (size_t i = 0; i < match.size(); i++) {
      if (!is_equivalent(input_exprs[i], graph_exprs[match[i]], ctx)) {
        return false;
      }
    }
    return true;
  };

  OutputMatch match(input_exprs.size());
  while (match.next()) {
    if (verify_with_match(match)) {
      return match;
    }
  }
  return OutputMatch::invalid_match();
}

std::vector<z3::expr> get_concrete_exprs(kernel::Graph const &graph,
                                         z3::context &ctx) {
  std::unordered_map<type::GuidType, z3::expr> tensor_exprs;

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
  z3::func_decl concat = ctx.function("concat", T, T, D, T);
  z3::func_decl ew_exp = ctx.function("ew_exp", T, T);
  z3::func_decl matmul = ctx.function("matmul", T, T, T);
  z3::func_decl sum = ctx.function("sum", T, D, T);
  z3::func_decl mean = ctx.function("mean", T, T);
  z3::func_decl rms = ctx.function("rms", T, T);
  z3::func_decl rms_norm = ctx.function("rms_norm", T, T);
  z3::func_decl silu = ctx.function("silu", T, T);
  z3::func_decl partition = ctx.function("partition", T, D, D, I, T);
  z3::func_decl combine = ctx.function("combine", T, D, D, T);
  z3::func_decl replicate = ctx.function("replicate", T, D, I, T);
  z3::func_decl reduce = ctx.function("reduce", T, D, T);

  auto calc_stensor_exprs = [&](threadblock::Graph const &graph) {
    static int custom_kernel_id = 0;
    z3::expr dx = ctx.constant(("dx" + std::to_string(custom_kernel_id)).data(), D);
    z3::expr dy = ctx.constant(("dy" + std::to_string(custom_kernel_id)).data(), D);
    z3::expr dz = ctx.constant(("dz" + std::to_string(custom_kernel_id)).data(), D);
    z3::expr df = ctx.constant(("df" + std::to_string(custom_kernel_id)).data(), D);
    custom_kernel_id++;
    for (threadblock::TBOperator *op : graph.operators) {
      switch (op->op_type) {
        case type::TBOperatorType::TB_INPUT_OP: {
          threadblock::TBInputOp *input_op = static_cast<threadblock::TBInputOp *>(op);
          z3::expr a = tensor_exprs.at(input_op->dtensor.guid);
          if (graph.grid_dim.x > 1) {
            if (input_op->input_map.x >= 0) {
              size_t axis = input_op->dtensor.num_dims - 1 - input_op->input_map.x;
              a = partition(a, data_dim[axis], dx, ctx.int_val(graph.grid_dim.x));
            } else {
              a = replicate(a, dx, ctx.int_val(graph.grid_dim.x));
            }
          }
          if (graph.grid_dim.y > 1) {
            if (input_op->input_map.y >= 0) {
              size_t axis = input_op->dtensor.num_dims - 1 - input_op->input_map.y;
              a = partition(a, data_dim[axis], dy, ctx.int_val(graph.grid_dim.y));
            } else {
              a = replicate(a, dy, ctx.int_val(graph.grid_dim.y));
            }
          }
          if (graph.grid_dim.z > 1) {
            if (input_op->input_map.z >= 0) {
              size_t axis = input_op->dtensor.num_dims - 1 - input_op->input_map.z;
              a = partition(a, data_dim[axis], dz, ctx.int_val(graph.grid_dim.z));
            } else {
              a = replicate(a, dz, ctx.int_val(graph.grid_dim.z));
            }
          }
          if (graph.forloop_range > 1) {
            if (input_op->forloop_dim >= 0) {
              size_t axis = input_op->dtensor.num_dims - 1 - input_op->forloop_dim;
              a = partition(a, data_dim[axis], df, ctx.int_val(graph.forloop_range));
            } else {
              a = replicate(a, df, ctx.int_val(graph.forloop_range));
            }
          }
          tensor_exprs.emplace(op->output_tensors[0].guid, a);
          break;
        }
        case type::TBOperatorType::TB_OUTPUT_OP: {
          threadblock::TBOutputOp *output_op = static_cast<threadblock::TBOutputOp *>(op);
          z3::expr a = tensor_exprs.at(output_op->input_tensors[0].guid);
          if (graph.grid_dim.x > 1) {
            if (output_op->output_map.x >= 0) {
              size_t axis = output_op->input_tensors[0].num_dims - 1 - output_op->output_map.x;
              a = combine(a, data_dim[axis], dx);
            } else {
              a = reduce(a, dx);
            }
          }
          if (graph.grid_dim.y > 1) {
            if (output_op->output_map.y >= 0) {
              size_t axis = output_op->input_tensors[0].num_dims - 1 - output_op->output_map.y;
              a = combine(a, data_dim[axis], dy);
            } else {
              a = reduce(a, dy);
            }
          }
          if (graph.grid_dim.z > 1) {
            if (output_op->output_map.z >= 0) {
              size_t axis = output_op->input_tensors[0].num_dims - 1 - output_op->output_map.z;
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
          size_t axis = op->input_tensors[0].num_dims - 1 - (op->op_type - type::TBOperatorType::TB_CONCAT_0_OP);
          z3::expr lhs = tensor_exprs.at(op->input_tensors[0].guid);
          z3::expr rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, concat(lhs, rhs, data_dim[axis]));
          break;
        }
        case type::TBOperatorType::TB_DIV_OP: {
          z3::expr lhs = tensor_exprs.at(op->input_tensors[0].guid);
          z3::expr rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, bc_div(lhs, rhs));
          break;
        }
        case type::TBOperatorType::TB_EXP_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, ew_exp(a));
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, reduce(a, df));
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_RMS_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          a = reduce(a, df);
          tensor_exprs.emplace(op->output_tensors[0].guid, rms(a));
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_SUM_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          a = reduce(a, df);
          tensor_exprs.emplace(op->output_tensors[0].guid, sum(a, data_dim[0]));
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          a = reduce(a, df);
          tensor_exprs.emplace(op->output_tensors[0].guid, mean(a));
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          a = reduce(a, df);
          static int redtox_id = 0;
          z3::expr reddim = ctx.constant(("reddim" + std::to_string(redtox_id++)).data(), D);
          int reduce_degree = op->input_tensors[0].dim[op->input_tensors[0].num_dims - 1]
                            / op->output_tensors[0].dim[op->output_tensors[0].num_dims - 1];
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
          size_t axis = op->input_tensors[0].num_dims - 1 - (op->op_type - type::TBOperatorType::TB_REDUCTION_0_OP);
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, sum(a, data_dim[axis]));
          break;
        }
        case type::TBOperatorType::TB_RMS_NORM_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, rms_norm(a));
          break;
        }
        case type::TBOperatorType::TB_SILU_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, silu(a));
          break;
        }
        default:
          assert(false && "Unsupported operator type");
      }
    }
  };

  auto calc_dtensor_exprs = [&](kernel::Graph const &graph) {
    for (kernel::KNOperator *op : graph.operators) {
      switch (op->op_type) {
        case type::KNOperatorType::KN_INPUT_OP: {
          static int input_id = 0;
          tensor_exprs.emplace(op->output_tensors[0].guid, ctx.constant(("input" + std::to_string(input_id++)).data(), T));
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
        case type::KNOperatorType::KN_EXP_OP: {
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, ew_exp(a));
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
          size_t axis = op->input_tensors[0].num_dims - 1 - (op->op_type - type::KNOperatorType::KN_REDUCTION_0_OP);
          z3::expr a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, sum(a, data_dim[axis]));
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

  for (kernel::KNOperator *op : graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      output_exprs.push_back(tensor_exprs.at(op->input_tensors[0].guid));
    }
  }

  return output_exprs;
}

bool is_equivalent(z3::expr const &lhs, z3::expr const &rhs, z3::context &ctx) {
  auto to_expr_vector = [&](std::vector<z3::expr> const &vec) {
    z3::expr_vector expr_vec(ctx);
    for (auto const &e : vec) {
      expr_vec.push_back(e);
    }
    return expr_vec;
  };

  z3::solver slv(ctx);

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
  z3::func_decl concat = ctx.function("concat", T, T, D, T);
  z3::func_decl ew_exp = ctx.function("ew_exp", T, T);
  z3::func_decl matmul = ctx.function("matmul", T, T, T);
  z3::func_decl sum = ctx.function("sum", T, D, T);
  z3::func_decl mean = ctx.function("mean", T, T);
  z3::func_decl rms = ctx.function("rms", T, T);
  z3::func_decl rms_norm = ctx.function("rms_norm", T, T);
  z3::func_decl silu = ctx.function("silu", T, T);
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
  z3::expr i0 = ctx.constant("i0", I);
  z3::expr i1 = ctx.constant("i1", I);

  slv.add(forall(t0, t1, ew_add(t0, t1) == ew_add(t1, t0)));
  slv.add(forall(t0, t1, ew_mul(t0, t1) == ew_mul(t1, t0)));
  slv.add(forall(t0, t1, t2, ew_add(t0, ew_add(t1, t2)) == ew_add(ew_add(t0, t1), t2)));
  slv.add(forall(t0, t1, t2, ew_mul(t0, ew_mul(t1, t2)) == ew_mul(ew_mul(t0, t1), t2)));
  slv.add(forall(t0, t1, t2, matmul(t0, matmul(t1, t2)) == matmul(matmul(t0, t1), t2)));
  slv.add(forall(t0, t1, t2, ew_mul(ew_add(t0, t1), t2) == ew_add(ew_mul(t0, t2), ew_mul(t1, t2))));
  slv.add(forall(t0, t1, t2, bc_div(ew_add(t0, t1), t2) == ew_add(bc_div(t0, t2), bc_div(t1, t2))));
  slv.add(forall(t0, t1, t2, matmul(ew_add(t0, t1), t2) == ew_add(matmul(t0, t2), matmul(t1, t2))));
  slv.add(forall(t0, t1, t2, matmul(t0, ew_add(t1, t2)) == ew_add(matmul(t0, t1), matmul(t0, t2))));

  slv.add(forall(t0, d0, reduce(sum(t0, d0), d0) == reduce(t0, d0)));

  slv.add(forall(to_expr_vector({t0, d0, d1, i0}), combine(partition(t0, d0, d1, i0), d0, d1) == t0));
  slv.add(forall(to_expr_vector({t0, d0, d1, i0}), partition(combine(t0, d0, d1, i0), d0, d1) == t0));

  {
    z3::expr_vector args = to_expr_vector({t0, d0, d1, d2, i0, i1});
    z3::expr pre_condition = (d0 != d1) && (d0 != d2) && (d1 != d2);

    std::vector<z3::expr> properties = {
      partition(reduce(t0, d0), d1, d2, i0) == reduce(partition(t0, d1, d2, i0), d0),
      combine(reduce(t0, d0), d1, d2) == reduce(combine(t0, d1, d2), d0),
      partition(replicate(t0, d0, i0), d1, d2, i1) == replicate(partition(t0, d1, d2, i1), d0, i0),
      combine(replicate(t0, d0, i0), d1, d2) == replicate(combine(t0, d1, d2), d0, i0),
      partition(sum(t0, d0), d1, d2, i0) == sum(partition(t0, d1, d2, i0), d0),
      combine(sum(t0, d0), d1, d2) == sum(combine(t0, d1, d2), d0),
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
    slv.add(forall(to_expr_vector({t0, t1, d0, i0}), matmul(t0, t1) == combined));
  }

  {
    z3::expr row_partitioned_lhs = partition(t0, data_dim1, d0, i0);
    z3::expr replicated_rhs = replicate(t1, d0, i0);
    z3::expr row_partitioned_matmul = matmul(row_partitioned_lhs, replicated_rhs);
    z3::expr combined = combine(row_partitioned_matmul, data_dim1, d0);
    slv.add(forall(to_expr_vector({t0, t1, d0, i0}), matmul(t0, t1) == combined));
  }

  {
    z3::expr replicated_lhs = replicate(t0, d0, i0);
    z3::expr col_partitioned_rhs = partition(t1, data_dim0, d0, i0);
    z3::expr col_partitioned_matmul = matmul(replicated_lhs, col_partitioned_rhs);
    z3::expr combined = combine(col_partitioned_matmul, data_dim0, d0);
    slv.add(forall(to_expr_vector({t0, t1, d0, i0}), matmul(t0, t1) == combined));
  }

  {
    z3::expr col_partitioned_lhs = partition(t0, data_dim0, d0, i0);
    z3::expr row_partitioned_rhs = partition(t1, data_dim1, d0, i0);
    z3::expr partial_sum_matmul = matmul(col_partitioned_lhs, row_partitioned_rhs);
    z3::expr reduced = reduce(partial_sum_matmul, d0);
    slv.add(forall(to_expr_vector({t0, t1, d0, i0}), matmul(t0, t1) == reduced));
  }

  {
    // element-wise binary & bc_div
    std::vector<z3::func_decl> ops = {ew_add, ew_mul, bc_div};
    for (z3::func_decl op : ops) {
      z3::expr partitioned_lhs = partition(t0, data_dim2, d0, i0);
      z3::expr partitioned_rhs = partition(t1, data_dim2, d0, i0);
      z3::expr partitioned_op = op(partitioned_lhs, partitioned_rhs);
      z3::expr combined = combine(partitioned_op, data_dim2, d0);
      slv.add(forall(to_expr_vector({t0, t1, d0, i0}), op(t0, t1) == combined));
    }
  }

  {
    // element-wise unary
    std::vector<z3::func_decl> ops = {ew_exp, silu};
    for (z3::func_decl op : ops) {
      z3::expr partitioned = partition(t0, d0, d1, i0);
      z3::expr op_partitioned = op(partitioned);
      z3::expr sumed = sum(op_partitioned, d0);
      z3::expr reduced = reduce(sumed, d1);
      slv.add(forall(to_expr_vector({t0, d0, d1, i0}), reduce(op(t0)) == reduced));
    }

    for (z3::func_decl op : ops) {    
      slv.add(forall(to_expr_vector({t0, d0, d1, i0}), partition(op(t0), d0, d1, i0) == op(partition(t0, d0, d1, i0))));
      slv.add(forall(to_expr_vector({t0, d0, d1, i0}), combine(op(t0), d0, d1) == op(combine(t0, d0, d1))));
      slv.add(forall(t0, d0, i0, replicate(op(t0), d0, i0) == op(replicate(t0, d0, i0))));
    }
  }

  slv.add(lhs != rhs);
  return slv.check() != z3::sat;
}

} // namespace search
} // namespace mirage
