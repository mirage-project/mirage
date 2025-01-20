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

bool TensorDimExpr::is_var() const {
  return false;
}

bool TensorDimExpr::is_const() const {
  return false;
}

bool TensorDimExpr::is_add() const {
  return false;
}

bool TensorDimExpr::is_mul() const {
  return false;
}

bool TensorDimExpr::is_div() const {
  return false;
}

bool TensorDimVar::is_var() const {
  return true;
}

bool TensorDimConst::is_const() const {
  return true;
}

bool TensorDimAdd::is_add() const {
  return true;
}

bool TensorDimMul::is_mul() const {
  return true;
}

bool TensorDimDiv::is_div() const {
  return true;
}

}
}
