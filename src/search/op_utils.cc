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

size_t count_op_of_type(type::KNOperatorType op_type,
                        SymbolicKNGraph const &g) {
  int counter = 0;
  for (auto const &op : g.operators) {
    if (op.op_type == op_type) {
      ++counter;
    }
  }
  return counter;
}

size_t count_op_of_type(type::TBOperatorType op_type,
                        SymbolicTBGraph const &g) {
  int counter = 0;
  for (auto const &op : g.operators) {
    if (op.op_type == op_type) {
      ++counter;
    }
  }
  return counter;
}

} // namespace search
} // namespace mirage
