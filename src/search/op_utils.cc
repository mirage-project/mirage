#include "mirage/search/op_utils.h"
#include "mirage/utils/containers.h"

namespace mirage {
namespace search {

bool is_binary(type::TBOperatorType op) {
  std::unordered_set<type::TBOperatorType> true_values{
      type::TBOperatorType::TB_ADD_OP,
      type::TBOperatorType::TB_MUL_OP,
      type::TBOperatorType::TB_MATMUL_OP,
      type::TBOperatorType::TB_DIV_OP,
      type::TBOperatorType::TB_POW_OP,
      type::TBOperatorType::TB_MUL_OP};
  return contains(true_values, op);
}

bool is_unary(type::TBOperatorType op) {
  std::unordered_set<type::TBOperatorType> true_values{
      type::TBOperatorType::TB_EXP_OP,
      type::TBOperatorType::TB_SQUARE_OP,
      type::TBOperatorType::TB_SQRT_OP,
      type::TBOperatorType::TB_SILU_OP,
      type::TBOperatorType::TB_GELU_OP,
      type::TBOperatorType::TB_RELU_OP,
      type::TBOperatorType::TB_CLAMP_OP,
      type::TBOperatorType::TB_RMS_NORM_OP,
      type::TBOperatorType::TB_REDUCTION_0_OP,
      type::TBOperatorType::TB_REDUCTION_1_OP,
      type::TBOperatorType::TB_REDUCTION_2_OP,
      type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_OP,
      type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP,
      type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_SUM_OP,
      type::TBOperatorType::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP,
      type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_RMS_OP,
  };
  return contains(true_values, op);
}

bool is_binary(type::KNOperatorType op) {
  std::unordered_set<type::KNOperatorType> true_values{
      type::KNOperatorType::KN_ADD_OP,
      type::KNOperatorType::KN_MUL_OP,
      type::KNOperatorType::KN_MATMUL_OP,
      type::KNOperatorType::KN_DIV_OP,
      type::KNOperatorType::KN_POW_OP,
      type::KNOperatorType::KN_MUL_OP,
  };
  return contains(true_values, op);
}

bool is_unary(type::KNOperatorType op) {
  std::unordered_set<type::KNOperatorType> true_values{
      type::KNOperatorType::KN_REDUCTION_0_OP,
      type::KNOperatorType::KN_REDUCTION_1_OP,
      type::KNOperatorType::KN_REDUCTION_2_OP,
      type::KNOperatorType::KN_EXP_OP,
      type::KNOperatorType::KN_SQUARE_OP,
      type::KNOperatorType::KN_SQRT_OP,
      type::KNOperatorType::KN_SILU_OP,
      type::KNOperatorType::KN_GELU_OP,
      type::KNOperatorType::KN_RELU_OP,
      type::KNOperatorType::KN_CLAMP_OP,
      type::KNOperatorType::KN_RMS_NORM_OP,
      type::KNOperatorType::KN_OUTPUT_OP,
  };
  return contains(true_values, op);
}

int get_input_number(type::KNOperatorType op) {
  if (is_unary(op)) {
    return 1;
  }
  if (is_binary(op)) {
    return 2;
  }
  assert(false && "Unsupported operator");
}

int get_input_number(type::TBOperatorType op) {
  if (is_unary(op)) {
    return 1;
  }
  if (is_binary(op)) {
    return 2;
  }
  if (op == type::TBOperatorType::TB_CONCAT_THEN_MATMUL_OP) {
    return 4;
  }
  assert(false && "Unsupported operator");
}

std::shared_ptr<AbstractExpr>
    make_reduction_pattern(int dim, std::shared_ptr<AbstractExpr> opd) {
  if (dim == 1) {
    return opd;
  }
  return std::make_shared<Red>(dim, opd);
}

std::shared_ptr<AbstractExpr> get_pattern(type::KNOperatorType op,
                                          DTensor const &tensor,
                                          std::shared_ptr<AbstractExpr> opd) {
  assert(opd != nullptr);
  switch (op) {
    case type::KNOperatorType::KN_REDUCTION_0_OP:
      return make_reduction_pattern(tensor.dim[0], opd);
    case type::KNOperatorType::KN_REDUCTION_1_OP:
      if (tensor.num_dims <= 1) {
        return nullptr;
      }
      return make_reduction_pattern(tensor.dim[1], opd);
    case type::KNOperatorType::KN_REDUCTION_2_OP:
      if (tensor.num_dims <= 2) {
        return nullptr;
      }
      return make_reduction_pattern(tensor.dim[2], opd);
    case type::KNOperatorType::KN_EXP_OP:
      return std::make_shared<Exp>(opd);
    case type::KNOperatorType::KN_SILU_OP:
      return std::make_shared<Silu>(opd);
    case type::KNOperatorType::KN_SQUARE_OP:
      return std::make_shared<Square>(opd);
    case type::KNOperatorType::KN_SQRT_OP:
      return std::make_shared<Sqrt>(opd);
    case type::KNOperatorType::KN_GELU_OP:
      return std::make_shared<Gelu>(opd);
    case type::KNOperatorType::KN_RELU_OP:
      return std::make_shared<Relu>(opd);
    case type::KNOperatorType::KN_CLAMP_OP:
      return std::make_shared<Clamp>(
          type::CLAMP_MIN_MAX["min_val"], type::CLAMP_MIN_MAX["max_val"], opd);
    case type::KNOperatorType::KN_OUTPUT_OP:
      return opd;
    default:
      assert(false);
  }
}

std::shared_ptr<AbstractExpr> get_pattern(type::TBOperatorType op,
                                          STensor const &tensor,
                                          std::shared_ptr<AbstractExpr> opd) {
  assert(opd != nullptr);
  // Retrieve reduction_dimx and forloop_range from threadblock graph
  assert(tensor.owner_op != nullptr);
  assert(tensor.owner_op->bgraph != nullptr);
  int reduction_dimx = tensor.owner_op->bgraph->reduction_dimx;
  int forloop_range = tensor.owner_op->bgraph->forloop_range;
  switch (op) {
    case type::TBOperatorType::TB_EXP_OP:
      return std::make_shared<Exp>(opd);
    case type::TBOperatorType::TB_SQUARE_OP:
      return std::make_shared<Square>(opd);
    case type::TBOperatorType::TB_SQRT_OP:
      return std::make_shared<Sqrt>(opd);
    case type::TBOperatorType::TB_SILU_OP:
      return std::make_shared<Silu>(opd);
    case type::TBOperatorType::TB_GELU_OP:
      return std::make_shared<Gelu>(opd);
    case type::TBOperatorType::TB_RELU_OP:
      return std::make_shared<Relu>(opd);
    case type::TBOperatorType::TB_CLAMP_OP:
      return std::make_shared<Clamp>(
          type::CLAMP_MIN_MAX["min_val"], type::CLAMP_MIN_MAX["max_val"], opd);
    case type::TBOperatorType::TB_RMS_NORM_OP: {
      return std::make_shared<Div>(
          opd, std::make_shared<RMS>(tensor.dim[tensor.num_dims - 1], opd));
    }
    case type::TBOperatorType::TB_REDUCTION_0_OP:
      return make_reduction_pattern(tensor.dim[0], opd);
    case type::TBOperatorType::TB_REDUCTION_1_OP:
      if (tensor.num_dims <= 1) {
        return nullptr;
      }
      return make_reduction_pattern(tensor.dim[1], opd);
    case type::TBOperatorType::TB_REDUCTION_2_OP:
      if (tensor.num_dims <= 2) {
        return nullptr;
      }
      return make_reduction_pattern(tensor.dim[2], opd);
    case type::TBOperatorType::TB_REDUCTION_0_TO_DIMX_OP:
      if (tensor.dim[0] <= reduction_dimx) {
        return nullptr;
      }
      return make_reduction_pattern(tensor.dim[0] / reduction_dimx, opd);
    case type::TBOperatorType::TB_REDUCTION_1_TO_DIMX_OP:
      if (tensor.num_dims <= 1 || tensor.dim[1] <= reduction_dimx) {
        return nullptr;
      }
      return make_reduction_pattern(tensor.dim[1] / reduction_dimx, opd);
    case type::TBOperatorType::TB_REDUCTION_2_TO_DIMX_OP:
      if (tensor.num_dims <= 2 || tensor.dim[2] <= reduction_dimx) {
        return nullptr;
      }
      return make_reduction_pattern(tensor.dim[2] / reduction_dimx, opd);
    case type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_OP: {
      return make_reduction_pattern(forloop_range, opd);
    }
    case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP:
    case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_SUM_OP: {
      return make_reduction_pattern(
          forloop_range * tensor.dim[tensor.num_dims - 1], opd);
    }
    case type::TBOperatorType::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP: {
      if (tensor.dim[tensor.num_dims - 1] <= reduction_dimx) {
        return nullptr;
      }
      return make_reduction_pattern(
          forloop_range * tensor.dim[tensor.num_dims - 1] / reduction_dimx,
          opd);
    }
    case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_RMS_OP: {
      return std::make_shared<RMS>(
          forloop_range * tensor.dim[tensor.num_dims - 1], opd);
    }
    default:
      assert(false);
  }
}

std::shared_ptr<AbstractExpr> get_pattern(type::KNOperatorType op,
                                          DTensor const &tensor_l,
                                          DTensor const &tensor_r,
                                          std::shared_ptr<AbstractExpr> lhs,
                                          std::shared_ptr<AbstractExpr> rhs) {

  assert(lhs != nullptr);
  assert(rhs != nullptr);
  switch (op) {
    case type::KNOperatorType::KN_MATMUL_OP:
      return make_reduction_pattern(tensor_l.dim[tensor_l.num_dims - 1],
                                    std::make_shared<Mul>(lhs, rhs));
    case type::KNOperatorType::KN_ADD_OP:
      return std::make_shared<Add>(lhs, rhs);
    case type::KNOperatorType::KN_DIV_OP:
      return std::make_shared<Div>(lhs, rhs);
    case type::KNOperatorType::KN_MUL_OP:
      return std::make_shared<Mul>(lhs, rhs);
    case type::KNOperatorType::KN_POW_OP:
      return std::make_shared<Pow>(lhs, rhs);
    default:
      assert(false);
  }
}

std::shared_ptr<AbstractExpr> get_pattern(type::TBOperatorType op,
                                          STensor const &tensor_l,
                                          STensor const &tensor_r,
                                          std::shared_ptr<AbstractExpr> lhs,
                                          std::shared_ptr<AbstractExpr> rhs) {
  assert(lhs != nullptr);
  assert(rhs != nullptr);
  switch (op) {
    case type::TBOperatorType::TB_MATMUL_OP:
      return make_reduction_pattern(tensor_l.dim[tensor_l.num_dims - 1],
                                    std::make_shared<Mul>(lhs, rhs));
    case type::TBOperatorType::TB_ADD_OP:
      return std::make_shared<Add>(lhs, rhs);
    case type::TBOperatorType::TB_DIV_OP:
      return std::make_shared<Div>(lhs, rhs);
    case type::TBOperatorType::TB_MUL_OP:
      return std::make_shared<Mul>(lhs, rhs);
    case type::TBOperatorType::TB_POW_OP:
      return std::make_shared<Pow>(lhs, rhs);
    default:
      assert(false);
  }
}

std::shared_ptr<AbstractExpr>
    get_pattern(type::KNOperatorType op,
                std::vector<DTensor> const &tensors,
                std::vector<std::shared_ptr<AbstractExpr>> const &opds) {
  for (auto const &expr : opds) {
    if (!expr) {
      return nullptr;
    }
  }
  if (tensors.size() == 1) {
    return get_pattern(op, tensors[0], opds[0]);
  }
  if (tensors.size() == 2) {
    return get_pattern(op, tensors[0], tensors[1], opds[0], opds[1]);
  }
  assert(false && "Unsupported operator");
}

std::shared_ptr<AbstractExpr>
    get_pattern(type::TBOperatorType op,
                std::vector<STensor> const &tensors,
                std::vector<std::shared_ptr<AbstractExpr>> const &opds) {
  for (auto const &expr : opds) {
    if (!expr) {
      return nullptr;
    }
  }
  if (opds.size() == 1) {
    return get_pattern(op, tensors[0], opds[0]);
  }
  if (opds.size() == 2) {
    return get_pattern(op, tensors[0], tensors[1], opds[0], opds[1]);
  }

  if (op == type::TBOperatorType::TB_CONCAT_THEN_MATMUL_OP) {
    assert(tensors.size() == 4);
    if (tensors[0].num_dims != tensors[1].num_dims ||
        tensors[0].num_dims != tensors[2].num_dims ||
        tensors[0].num_dims != tensors[3].num_dims) {
      return nullptr;
    }
    int num_dims = tensors[0].num_dims;
    int reduction_dim1 = tensors[0].dim[num_dims - 1],
        reduction_dim2 = tensors[1].dim[num_dims - 1];
    return std::make_shared<Add>(
        make_reduction_pattern(reduction_dim1,
                               std::make_shared<Mul>(opds[0], opds[2])),
        make_reduction_pattern(reduction_dim2,
                               std::make_shared<Mul>(opds[1], opds[3])));
  }
  assert(false && "Unsupported operator");
}

KNOperator *create_op(kernel::Graph &g,
                      type::KNOperatorType type,
                      DTensor const &input) {
  switch (type) {
    case type::KNOperatorType::KN_REDUCTION_0_OP:
      return g.create_reduction_op(input, 0, 1);
    case type::KNOperatorType::KN_REDUCTION_1_OP:
      return g.create_reduction_op(input, 1, 1);
    case type::KNOperatorType::KN_REDUCTION_2_OP:
      return g.create_reduction_op(input, 2, 1);
    case type::KNOperatorType::KN_EXP_OP:
    case type::KNOperatorType::KN_SQUARE_OP:
    case type::KNOperatorType::KN_SQRT_OP:
    case type::KNOperatorType::KN_SILU_OP:
    case type::KNOperatorType::KN_GELU_OP:
    case type::KNOperatorType::KN_RELU_OP:
      return g.create_elementunary_op(input, type);
    case type::KNOperatorType::KN_CLAMP_OP:
      assert((!type::CLAMP_MIN_MAX.empty()) && "CLAMP_MIN_MAX not assigned");
      return g.create_elementunary_clamp_op(input,
                                            type::CLAMP_MIN_MAX["min_val"],
                                            type::CLAMP_MIN_MAX["max_val"]);
    default:
      assert(false && "Unsupported operator");
  }
}

KNOperator *create_op(kernel::Graph &g,
                      type::KNOperatorType type,
                      DTensor const &input1,
                      DTensor const &input2) {
  switch (type) {
    case type::KNOperatorType::KN_MATMUL_OP:
      return g.create_matmul_op(input1, input2);
    case type::KNOperatorType::KN_DIV_OP:
    case type::KNOperatorType::KN_ADD_OP:
    case type::KNOperatorType::KN_MUL_OP:
    case type::KNOperatorType::KN_POW_OP:
      return g.create_elementbinary_op(input1, input2, type);
    default:
      assert(false && "Unsupported operator");
  }
}

KNOperator *create_op(kernel::Graph &g,
                      type::KNOperatorType type,
                      std::vector<DTensor> const &inputs) {
  if (inputs.size() == 1) {
    return create_op(g, type, inputs[0]);
  }
  if (inputs.size() == 2) {
    return create_op(g, type, inputs[0], inputs[1]);
  }
  return nullptr;
}

TBOperator *create_op(threadblock::Graph &g,
                      type::TBOperatorType type,
                      STensor const &input) {
  switch (type) {
    case type::TBOperatorType::TB_EXP_OP:
    case type::TBOperatorType::TB_SQUARE_OP:
    case type::TBOperatorType::TB_SQRT_OP:
    case type::TBOperatorType::TB_SILU_OP:
    case type::TBOperatorType::TB_GELU_OP:
    case type::TBOperatorType::TB_RELU_OP:
      return g.create_elementunary_op(input, type);
    case type::TBOperatorType::TB_CLAMP_OP:
      assert((!type::CLAMP_MIN_MAX.empty()) && "CLAMP_MIN_MAX not assigned");
      return g.create_elementunary_clamp_op(input,
                                            type::CLAMP_MIN_MAX["min_val"],
                                            type::CLAMP_MIN_MAX["max_val"]);
    case type::TBOperatorType::TB_RMS_NORM_OP:
      return g.create_rms_norm_op(input);
    case type::TBOperatorType::TB_REDUCTION_0_OP:
    case type::TBOperatorType::TB_REDUCTION_1_OP:
    case type::TBOperatorType::TB_REDUCTION_2_OP: {
      int dim = (int)type - (int)type::TBOperatorType::TB_REDUCTION_0_OP;
      if (input.num_dims <= dim ||
          (input.num_dims > dim && input.dim[dim] == 1)) {
        return nullptr;
      }
      return g.create_reduction_op(input, dim);
    }
    case type::TBOperatorType::TB_REDUCTION_0_TO_DIMX_OP:
    case type::TBOperatorType::TB_REDUCTION_1_TO_DIMX_OP:
    case type::TBOperatorType::TB_REDUCTION_2_TO_DIMX_OP: {
      int dim =
          (int)type - (int)type::TBOperatorType::TB_REDUCTION_0_TO_DIMX_OP;
      if (input.num_dims <= dim) {
        return nullptr;
      }
      if ((input.dim[dim] <= g.reduction_dimx) ||
          (input.dim[dim] % g.reduction_dimx != 0)) {
        return nullptr;
      }
      return g.create_reduction_to_dimx_op(input, dim);
    }
    case type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_OP:
    case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP:
    case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_SUM_OP:
    case type::TBOperatorType::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP:
    case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_RMS_OP: {
      return g.create_forloop_accum_op(input, type);
    }
    default:
      assert(false && "Unsupported operator");
  }
}

TBOperator *create_op(threadblock::Graph &g,
                      type::TBOperatorType type,
                      STensor const &input1,
                      STensor const &input2) {
  switch (type) {
    case type::TBOperatorType::TB_MATMUL_OP:
      return g.create_matmul_op(input1, input2);
    case type::TBOperatorType::TB_DIV_OP:
    case type::TBOperatorType::TB_ADD_OP:
    case type::TBOperatorType::TB_MUL_OP:
    case type::TBOperatorType::TB_POW_OP:
      return g.create_elementbinary_op(input1, input2, type);
    default:
      assert(false && "Unsupported operator");
  }
}

TBOperator *create_op(threadblock::Graph &g,
                      type::TBOperatorType type,
                      std::vector<STensor> const &inputs) {
  if (inputs.size() == 1) {
    return create_op(g, type, inputs[0]);
  }
  if (inputs.size() == 2) {
    return create_op(g, type, inputs[0], inputs[1]);
  }
  if (type == type::TBOperatorType::TB_CONCAT_THEN_MATMUL_OP) {
    TBOperator *concat1 =
        g.create_concat_op(inputs[0], inputs[1], inputs[0].num_dims - 1);
    if (concat1 == nullptr) {
      return nullptr;
    }
    g.operators.push_back(concat1);
    TBOperator *concat2 =
        g.create_concat_op(inputs[2], inputs[3], inputs[2].num_dims - 2);
    if (concat2 == nullptr) {
      delete concat1;
      g.operators.pop_back();
      return nullptr;
    }
    g.operators.push_back(concat2);
    TBOperator *matmul = g.create_matmul_op(concat1->output_tensors[0],
                                            concat2->output_tensors[0]);
    if (matmul == nullptr) {
      delete concat2;
      delete concat1;
      g.operators.pop_back();
      g.operators.pop_back();
      return nullptr;
    }
    return matmul;
  }
  return nullptr;
}

size_t count_op_of_type(type::KNOperatorType op_type, kernel::Graph const &g) {
  int counter = 0;
  for (auto const &op : g.operators) {
    if (op->op_type == op_type) {
      ++counter;
    }
  }
  return counter;
}

size_t count_op_of_type(type::TBOperatorType op_type,
                        threadblock::Graph const &g) {
  int counter = 0;
  for (auto const &op : g.operators) {
    if (op->op_type == op_type) {
      ++counter;
    }
  }
  return counter;
}

} // namespace search
} // namespace mirage
