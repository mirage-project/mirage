#include "mirage/search/symbolic_graph/tensor_dim_expr.h"

namespace mirage {
namespace search {

int TensorDimVar::get_value(DimVarAssignments const &assignments) const {
  return assignments.get_value(index);
}

int TensorDimConst::get_value(DimVarAssignments const &assignments) const {
  return value;
}

int TensorDimAdd::get_value(DimVarAssignments const &assignments) const {
  return lhs->get_value(assignments) + rhs->get_value(assignments);
}

int TensorDimMul::get_value(DimVarAssignments const &assignments) const {
  return lhs->get_value(assignments) * rhs->get_value(assignments);
}

int TensorDimDiv::get_value(DimVarAssignments const &assignments) const {
  return lhs->get_value(assignments) / rhs->get_value(assignments);
}

}
}
