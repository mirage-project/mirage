#include "mirage/search/graph_template/tensor_dim_template.h"

#include <vector>

namespace mirage {
namespace search {

enum class ConstraintType {
  EQUAL,
  EQUAL_OR_ONE,
};

class TensorDimConstraint {
public:
  TensorDimConstraint(ConstraintType type, TensorDimTemplate lhs, TensorDimTemplate rhs);
  ConstraintType type;
  TensorDimTemplate lhs, rhs;
};

TensorDimConstraint make_equal_constraint(TensorDimTemplate lhs, TensorDimTemplate rhs);

TensorDimConstraint make_equal_or_one_constraint(TensorDimTemplate lhs, TensorDimTemplate rhs);

bool check_satisfiability(std::vector<TensorDimConstraint> const &pre_conds,
                          std::vector<TensorDimConstraint> const &constraints);

}
}
