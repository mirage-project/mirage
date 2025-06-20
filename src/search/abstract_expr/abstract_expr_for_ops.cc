#include "mirage/search/abstract_expr/abstract_expr_for_ops.h"

namespace mirage {
namespace search {

std::shared_ptr<AbstractExpr const> get_abstract_expr(
    type::KNOperatorType op,
    std::vector<kernel::DTensor> const &tensors,
    std::vector<std::shared_ptr<AbstractExpr const>> const &opds) {
  for (auto const &expr : opds) {
    if (!expr) {
      return nullptr;
    }
  }
  switch (op) {
    case type::KNOperatorType::KN_REDUCTION_0_OP:
      return abstract_expr_make_red(tensors[0].dim[0], opds[0]);
    case type::KNOperatorType::KN_REDUCTION_1_OP:
      if (tensors[0].num_dims <= 1) {
        return nullptr;
      }
      return abstract_expr_make_red(tensors[0].dim[1], opds[0]);
    case type::KNOperatorType::KN_REDUCTION_2_OP:
      if (tensors[0].num_dims <= 2) {
        return nullptr;
      }
      return abstract_expr_make_red(tensors[0].dim[2], opds[0]);
    case type::KNOperatorType::KN_EXP_OP:
      return abstract_expr_make_exp(opds[0]);
    case type::KNOperatorType::KN_SILU_OP:
      return abstract_expr_make_silu(opds[0]);
    case type::KNOperatorType::KN_GELU_OP:
      return abstract_expr_make_gelu(opds[0]);
    case type::KNOperatorType::KN_RELU_OP:
      return abstract_expr_make_relu(opds[0]);
    case type::KNOperatorType::KN_CLAMP_OP:
      return abstract_expr_make_clamp(type::CLAMP_MIN_MAX["min_val"],
                                      type::CLAMP_MIN_MAX["max_val"],
                                      opds[0]);
    case type::KNOperatorType::KN_OUTPUT_OP:
      return opds[0];
    case type::KNOperatorType::KN_MATMUL_OP:
      return abstract_expr_make_red(tensors[0].dim[tensors[0].num_dims - 1],
                                    abstract_expr_make_mul(opds[0], opds[1]));
    case type::KNOperatorType::KN_ADD_OP:
      return abstract_expr_make_add(opds[0], opds[1]);
    case type::KNOperatorType::KN_DIV_OP:
      return abstract_expr_make_div(opds[0], opds[1]);
    case type::KNOperatorType::KN_MUL_OP:
      return abstract_expr_make_mul(opds[0], opds[1]);
    case type::KNOperatorType::KN_RMS_NORM_OP:
      return abstract_expr_make_rms(tensors[0].dim[tensors[0].num_dims - 1],
                                    opds[0]);
    case type::KNOperatorType::KN_SQUARE_OP:
      return abstract_expr_make_square(opds[0]);
    case type::KNOperatorType::KN_SQRT_OP:
      return abstract_expr_make_sqrt(opds[0]);
    case type::KNOperatorType::KN_POW_OP:
      return abstract_expr_make_pow(opds[0], opds[1]);
    default:
      printf("Operator type: %d\n", (int)op);
      assert(false && "Unsupported operator");
  }
}

std::shared_ptr<AbstractExpr const> get_abstract_expr(
    type::TBOperatorType op,
    std::vector<threadblock::STensor> const &tensors,
    std::vector<std::shared_ptr<AbstractExpr const>> const &opds) {
  for (auto const &expr : opds) {
    if (!expr) {
      return nullptr;
    }
  }
  int reduction_dimx = tensors[0].owner_op->bgraph->reduction_dimx;
  int forloop_range = tensors[0].owner_op->bgraph->forloop_range;
  switch (op) {
    case type::TBOperatorType::TB_EXP_OP:
      return abstract_expr_make_exp(opds[0]);
    case type::TBOperatorType::TB_SQUARE_OP:
      return abstract_expr_make_square(opds[0]);
    case type::TBOperatorType::TB_SQRT_OP:
      return abstract_expr_make_sqrt(opds[0]);
    case type::TBOperatorType::TB_SILU_OP:
      return abstract_expr_make_silu(opds[0]);
    case type::TBOperatorType::TB_GELU_OP:
      return abstract_expr_make_gelu(opds[0]);
    case type::TBOperatorType::TB_RELU_OP:
      return abstract_expr_make_relu(opds[0]);
    case type::TBOperatorType::TB_CLAMP_OP:
      return abstract_expr_make_clamp(type::CLAMP_MIN_MAX["min_val"],
                                      type::CLAMP_MIN_MAX["max_val"],
                                      opds[0]);
    case type::TBOperatorType::TB_RMS_NORM_OP: {
      return abstract_expr_make_div(
          opds[0],
          abstract_expr_make_rms(tensors[0].dim[tensors[0].num_dims - 1],
                                 opds[0]));
    }
    case type::TBOperatorType::TB_REDUCTION_0_OP:
      return abstract_expr_make_red(tensors[0].dim[0], opds[0]);
    case type::TBOperatorType::TB_REDUCTION_1_OP:
      if (tensors[0].num_dims <= 1) {
        return nullptr;
      }
      return abstract_expr_make_red(tensors[0].dim[1], opds[0]);
    case type::TBOperatorType::TB_REDUCTION_2_OP:
      if (tensors[0].num_dims <= 2) {
        return nullptr;
      }
      return abstract_expr_make_red(tensors[0].dim[2], opds[0]);
    case type::TBOperatorType::TB_REDUCTION_0_TO_DIMX_OP:
      if (tensors[0].dim[0] <= reduction_dimx) {
        return nullptr;
      }
      return abstract_expr_make_red(tensors[0].dim[0] / reduction_dimx,
                                    opds[0]);
    case type::TBOperatorType::TB_REDUCTION_1_TO_DIMX_OP:
      if (tensors[0].num_dims <= 1 || tensors[0].dim[1] <= reduction_dimx) {
        return nullptr;
      }
      return abstract_expr_make_red(tensors[0].dim[1] / reduction_dimx,
                                    opds[0]);
    case type::TBOperatorType::TB_REDUCTION_2_TO_DIMX_OP:
      if (tensors[0].num_dims <= 2 || tensors[0].dim[2] <= reduction_dimx) {
        return nullptr;
      }
      return abstract_expr_make_red(tensors[0].dim[2] / reduction_dimx,
                                    opds[0]);
    case type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_OP: {
      return abstract_expr_make_red(forloop_range, opds[0]);
    }
    case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP:
    case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_SUM_OP: {
      return abstract_expr_make_red(
          forloop_range * tensors[0].dim[tensors[0].num_dims - 1], opds[0]);
    }
    case type::TBOperatorType::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP: {
      if (tensors[0].dim[tensors[0].num_dims - 1] <= reduction_dimx) {
        return nullptr;
      }
      return abstract_expr_make_red(
          forloop_range * tensors[0].dim[tensors[0].num_dims - 1] /
              reduction_dimx,
          opds[0]);
    }
    case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_RMS_OP: {
      return abstract_expr_make_rms(
          forloop_range * tensors[0].dim[tensors[0].num_dims - 1], opds[0]);
    }
    case type::TBOperatorType::TB_MATMUL_OP:
      return abstract_expr_make_red(tensors[0].dim[tensors[0].num_dims - 1],
                                    abstract_expr_make_mul(opds[0], opds[1]));
    case type::TBOperatorType::TB_ADD_OP:
      return abstract_expr_make_add(opds[0], opds[1]);
    case type::TBOperatorType::TB_DIV_OP:
      return abstract_expr_make_div(opds[0], opds[1]);
    case type::TBOperatorType::TB_MUL_OP:
      return abstract_expr_make_mul(opds[0], opds[1]);
    case type::TBOperatorType::TB_CONCAT_THEN_MATMUL_OP: {
      assert(tensors.size() == 4);
      if (tensors[0].num_dims != tensors[1].num_dims ||
          tensors[0].num_dims != tensors[2].num_dims ||
          tensors[0].num_dims != tensors[3].num_dims) {
        return nullptr;
      }
      int num_dims = tensors[0].num_dims;
      int reduction_dim1 = tensors[0].dim[num_dims - 1],
          reduction_dim2 = tensors[1].dim[num_dims - 1];
      return abstract_expr_make_add(
          abstract_expr_make_red(reduction_dim1,
                                 abstract_expr_make_mul(opds[0], opds[2])),
          abstract_expr_make_red(reduction_dim2,
                                 abstract_expr_make_mul(opds[1], opds[3])));
    }
    case type::TBOperatorType::TB_POW_OP:
      return abstract_expr_make_pow(opds[0], opds[1]);
    default:
      assert(false && "Unsupported operator");
  }
}

std::shared_ptr<AbstractExpr const> get_abstract_expr(
    type::TBOperatorType op,
    std::vector<SymbolicSTensor> const &tensors,
    std::vector<std::shared_ptr<AbstractExpr const>> const &opds,
    SymbolicTBGraph const &g) {
  switch (op) {
    case type::TBOperatorType::TB_INPUT_OP:
    case type::TBOperatorType::TB_OUTPUT_OP: {
      assert(false && "Should not reach here");
    }
    case type::TBOperatorType::TB_CONCAT_0_OP:
    case type::TBOperatorType::TB_CONCAT_1_OP:
    case type::TBOperatorType::TB_CONCAT_2_OP: {
      assert(false && "Unsupported operator");
    }
    case type::TBOperatorType::TB_DIV_OP: {
      assert(opds.size() == 2);
      return abstract_expr_make_div(opds[0], opds[1]);
    }
    case type::TBOperatorType::TB_ADD_OP: {
      assert(opds.size() == 2);
      return abstract_expr_make_add(opds[0], opds[1]);
    }
    case type::TBOperatorType::TB_MUL_OP: {
      assert(opds.size() == 2);
      return abstract_expr_make_mul(opds[0], opds[1]);
    }
    case type::TBOperatorType::TB_EXP_OP: {
      assert(opds.size() == 1);
      return abstract_expr_make_exp(opds[0]);
    }
    case type::TBOperatorType::TB_MATMUL_OP: {
      assert(opds.size() == 2);
      return abstract_expr_make_red(tensors[0].dims[tensors[0].dims.size() - 1],
                                    abstract_expr_make_mul(opds[0], opds[1]));
    }
    case type::TBOperatorType::TB_SILU_OP: {
      assert(opds.size() == 1);
      return abstract_expr_make_silu(opds[0]);
    }
    case type::TBOperatorType::TB_RMS_NORM_OP: {
      assert(opds.size() == 1);
      return abstract_expr_make_div(
          opds[0],
          abstract_expr_make_rms(tensors[0].dims[tensors[0].dims.size() - 1],
                                 opds[0]));
    }
    case type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_OP: {
      assert(opds.size() == 1);
      return abstract_expr_make_red(g.forloop_range, opds[0]);
    }
    case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_RMS_OP: {
      assert(opds.size() == 1);
      std::shared_ptr<TensorDimExpr const> reduction_size_expr =
          dim_expr_make_mul(
              g.forloop_range.dim_expr,
              tensors[0].dims[tensors[0].dims.size() - 1].dim_expr);
      return abstract_expr_make_rms(reduction_size_expr, opds[0]);
    }
    case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_SUM_OP: {
      assert(opds.size() == 1);
      std::shared_ptr<TensorDimExpr const> reduction_size_expr =
          dim_expr_make_mul(
              g.forloop_range.dim_expr,
              tensors[0].dims[tensors[0].dims.size() - 1].dim_expr);
      return abstract_expr_make_red(reduction_size_expr, opds[0]);
    }
    case type::TBOperatorType::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP: {
      assert(opds.size() == 1);
      std::shared_ptr<TensorDimExpr const> reduction_size_expr =
          dim_expr_make_mul(
              g.forloop_range.dim_expr,
              tensors[0].dims[tensors[0].dims.size() - 1].dim_expr);
      reduction_size_expr = dim_expr_make_div(
          reduction_size_expr, dim_expr_make_const(g.reduction_dimx));
      return abstract_expr_make_red(reduction_size_expr, opds[0]);
    }
    default: {
      fprintf(stderr, "Unsupported operator: %d\n", (int)op);
      assert(false);
    }
  }
}

std::shared_ptr<AbstractExpr const> get_abstract_expr(
    type::KNOperatorType op,
    std::vector<SymbolicDTensor> const &tensors,
    std::vector<std::shared_ptr<AbstractExpr const>> const &opds,
    SymbolicKNGraph const &g) {
  switch (op) {
    case type::KNOperatorType::KN_MATMUL_OP: {
      assert(opds.size() == 2);
      return abstract_expr_make_red(tensors[0].dims[tensors[0].dims.size() - 1],
                                    abstract_expr_make_mul(opds[0], opds[1]));
    }
    case type::KNOperatorType::KN_ADD_OP: {
      assert(opds.size() == 2);
      return abstract_expr_make_add(opds[0], opds[1]);
    }
    case type::KNOperatorType::KN_EXP_OP: {
      assert(opds.size() == 1);
      return abstract_expr_make_exp(opds[0]);
    }
    case type::KNOperatorType::KN_DIV_OP: {
      assert(opds.size() == 2);
      return abstract_expr_make_div(opds[0], opds[1]);
    }
    case type::KNOperatorType::KN_MUL_OP: {
      assert(opds.size() == 2);
      return abstract_expr_make_mul(opds[0], opds[1]);
    }
    case type::KNOperatorType::KN_SILU_OP: {
      assert(opds.size() == 1);
      return abstract_expr_make_silu(opds[0]);
    }
    case type::KNOperatorType::KN_RMS_NORM_OP: {
      assert(opds.size() == 1);
      return abstract_expr_make_div(
          opds[0],
          abstract_expr_make_rms(tensors[0].dims[tensors[0].dims.size() - 1],
                                 opds[0]));
    }
    default: {
      fprintf(stderr, "Unsupported operator: %d\n", (int)op);
      assert(false);
    }
  }
}

} // namespace search
} // namespace mirage
