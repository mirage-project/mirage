#include "mirage/search/symbolic_graph/tensor_dim_expr.h"

namespace mirage {
namespace search {

TensorDimVar::TensorDimVar(tensor_dim_var_index_t index) : index(index) {}

TensorDimConst::TensorDimConst(int value) : value(value) {}

TensorDimAdd::TensorDimAdd(std::shared_ptr<TensorDimExpr> lhs, std::shared_ptr<TensorDimExpr> rhs) : lhs(lhs), rhs(rhs) {}

TensorDimMul::TensorDimMul(std::shared_ptr<TensorDimExpr> lhs, std::shared_ptr<TensorDimExpr> rhs) : lhs(lhs), rhs(rhs) {}

TensorDimDiv::TensorDimDiv(std::shared_ptr<TensorDimExpr> lhs, std::shared_ptr<TensorDimExpr> rhs) : lhs(lhs), rhs(rhs) {}

std::shared_ptr<TensorDimVar> dim_expr_make_var(tensor_dim_var_index_t index) {
  return std::make_shared<TensorDimVar>(index);
}

std::shared_ptr<TensorDimConst> dim_expr_make_const(int value) {
  return std::make_shared<TensorDimConst>(value);
}

std::shared_ptr<TensorDimAdd> dim_expr_make_add(std::shared_ptr<TensorDimExpr> lhs, std::shared_ptr<TensorDimExpr> rhs) {
  return std::make_shared<TensorDimAdd>(lhs, rhs);
}

std::shared_ptr<TensorDimMul> dim_expr_make_mul(std::shared_ptr<TensorDimExpr> lhs, std::shared_ptr<TensorDimExpr> rhs) {
  return std::make_shared<TensorDimMul>(lhs, rhs);
}

std::shared_ptr<TensorDimDiv> dim_expr_make_div(std::shared_ptr<TensorDimExpr> lhs, std::shared_ptr<TensorDimExpr> rhs) {
  return std::make_shared<TensorDimDiv>(lhs, rhs);
}

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

TensorDimVar::operator json() const {
  return json{{"opt", "var"}, {"index", index}};
}

TensorDimConst::operator json() const {
  return json{{"opt", "const"}, {"value", value}};
}

TensorDimAdd::operator json() const {
  return json{{"opt", "add"}, {"lhs", *lhs}, {"rhs", *rhs}};
}

TensorDimMul::operator json() const {
  return json{{"opt", "mul"}, {"lhs", *lhs}, {"rhs", *rhs}};
}

TensorDimDiv::operator json() const {
  return json{{"opt", "div"}, {"lhs", *lhs}, {"rhs", *rhs}};
}

}
}
