#pragma once

#include "mirage/search/symbolic_graph/symbolic_tensor_dim.h"

#include <vector>

namespace mirage {
namespace search {

enum class ConstraintType {
  EQUAL,
  EQUAL_OR_ONE,
};

class TensorDimConstraint {
public:
  TensorDimConstraint(ConstraintType type, SymbolicTensorDim lhs, SymbolicTensorDim rhs);
  ConstraintType type;
  SymbolicTensorDim lhs, rhs;
};

TensorDimConstraint make_equal_constraint(SymbolicTensorDim lhs, SymbolicTensorDim rhs);

TensorDimConstraint make_equal_or_one_constraint(SymbolicTensorDim lhs, SymbolicTensorDim rhs);

bool check_satisfiability(std::vector<TensorDimConstraint> const &pre_conds,
                          std::vector<TensorDimConstraint> const &constraints);

}
}
