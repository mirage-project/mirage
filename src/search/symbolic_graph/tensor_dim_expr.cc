#include "mirage/search/symbolic_graph/tensor_dim_expr.h"
#include "mirage/utils/hash_utils.h"

namespace mirage {
namespace search {

TensorDimVar::TensorDimVar(tensor_dim_var_index_t index) : index(index) {}

TensorDimConst::TensorDimConst(int value) : value(value) {}

TensorDimAdd::TensorDimAdd(std::shared_ptr<TensorDimExpr const> lhs,
                           std::shared_ptr<TensorDimExpr const> rhs)
    : lhs(lhs), rhs(rhs) {}

TensorDimMul::TensorDimMul(std::shared_ptr<TensorDimExpr const> lhs,
                           std::shared_ptr<TensorDimExpr const> rhs)
    : lhs(lhs), rhs(rhs) {}

TensorDimDiv::TensorDimDiv(std::shared_ptr<TensorDimExpr const> lhs,
                           std::shared_ptr<TensorDimExpr const> rhs)
    : lhs(lhs), rhs(rhs) {}

std::shared_ptr<TensorDimVar const>
    dim_expr_make_var(tensor_dim_var_index_t index) {
  return std::make_shared<TensorDimVar const>(index);
}

std::shared_ptr<TensorDimConst const> dim_expr_make_const(int value) {
  return std::make_shared<TensorDimConst const>(value);
}

std::shared_ptr<TensorDimAdd const>
    dim_expr_make_add(std::shared_ptr<TensorDimExpr const> lhs,
                      std::shared_ptr<TensorDimExpr const> rhs) {
  return std::make_shared<TensorDimAdd const>(lhs, rhs);
}

std::shared_ptr<TensorDimMul const>
    dim_expr_make_mul(std::shared_ptr<TensorDimExpr const> lhs,
                      std::shared_ptr<TensorDimExpr const> rhs) {
  return std::make_shared<TensorDimMul const>(lhs, rhs);
}

std::shared_ptr<TensorDimDiv const>
    dim_expr_make_div(std::shared_ptr<TensorDimExpr const> lhs,
                      std::shared_ptr<TensorDimExpr const> rhs) {
  return std::make_shared<TensorDimDiv const>(lhs, rhs);
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

size_t TensorDimVar::hash() const {
  size_t h = 0;
  hash_combine(h, index);
  return h;
}

size_t TensorDimConst::hash() const {
  size_t h = 1;
  hash_combine(h, value);
  return h;
}

size_t TensorDimAdd::hash() const {
  size_t h = 2;
  hash_combine(h, lhs->hash());
  hash_combine(h, rhs->hash());
  return h;
}

size_t TensorDimMul::hash() const {
  size_t h = 3;
  hash_combine(h, lhs->hash());
  hash_combine(h, rhs->hash());
  return h;
}

size_t TensorDimDiv::hash() const {
  size_t h = 4;
  hash_combine(h, lhs->hash());
  hash_combine(h, rhs->hash());
  return h;
}

bool TensorDimVar::same_expr_as(
    std::shared_ptr<TensorDimExpr const> other) const {
  if (!other->is_var()) {
    return false;
  }
  return index == std::static_pointer_cast<TensorDimVar const>(other)->index;
}

bool TensorDimConst::same_expr_as(
    std::shared_ptr<TensorDimExpr const> other) const {
  if (!other->is_const()) {
    return false;
  }
  return value == std::static_pointer_cast<TensorDimConst const>(other)->value;
}

bool TensorDimAdd::same_expr_as(
    std::shared_ptr<TensorDimExpr const> other) const {
  if (!other->is_add()) {
    return false;
  }
  auto other_add = std::static_pointer_cast<TensorDimAdd const>(other);
  return lhs->same_expr_as(other_add->lhs) && rhs->same_expr_as(other_add->rhs);
}

bool TensorDimMul::same_expr_as(
    std::shared_ptr<TensorDimExpr const> other) const {
  if (!other->is_mul()) {
    return false;
  }
  auto other_mul = std::static_pointer_cast<TensorDimMul const>(other);
  return lhs->same_expr_as(other_mul->lhs) && rhs->same_expr_as(other_mul->rhs);
}

bool TensorDimDiv::same_expr_as(
    std::shared_ptr<TensorDimExpr const> other) const {
  if (!other->is_div()) {
    return false;
  }
  auto other_div = std::static_pointer_cast<TensorDimDiv const>(other);
  return lhs->same_expr_as(other_div->lhs) && rhs->same_expr_as(other_div->rhs);
}

z3::expr TensorDimVar::to_z3(z3::context &c, bool log_scaled) const {
  std::string z3_var_name = "tensor_dim_var_" + std::to_string(index);
  return c.int_const(z3_var_name.c_str());
}

z3::expr TensorDimConst::to_z3(z3::context &c, bool log_scaled) const {
  if (log_scaled) {
    int log_scaled_value = std::ceil(std::log2(value));
    return c.int_val(log_scaled_value);
  }
  return c.int_val(value);
}

z3::expr TensorDimAdd::to_z3(z3::context &c, bool log_scaled) const {
  if (log_scaled) {
    assert(false &&
           "Do not support addition between log scaled symbolic dimensions");
  }
  return lhs->to_z3(c, log_scaled) + rhs->to_z3(c, log_scaled);
}

z3::expr TensorDimMul::to_z3(z3::context &c, bool log_scaled) const {
  if (log_scaled) {
    return lhs->to_z3(c, log_scaled) + rhs->to_z3(c, log_scaled);
  }
  return lhs->to_z3(c, log_scaled) * rhs->to_z3(c, log_scaled);
}

z3::expr TensorDimDiv::to_z3(z3::context &c, bool log_scaled) const {
  if (log_scaled) {
    return lhs->to_z3(c, log_scaled) - rhs->to_z3(c, log_scaled);
  }
  return lhs->to_z3(c, log_scaled) / rhs->to_z3(c, log_scaled);
}

std::string TensorDimVar::to_string() const {
  return "var_" + std::to_string(index);
}

std::string TensorDimConst::to_string() const {
  return std::to_string(value);
}

std::string TensorDimAdd::to_string() const {
  return "(" + lhs->to_string() + " + " + rhs->to_string() + ")";
}

std::string TensorDimMul::to_string() const {
  return lhs->to_string() + " * " + rhs->to_string();
}

std::string TensorDimDiv::to_string() const {
  return lhs->to_string() + " / " + rhs->to_string();
}

} // namespace search
} // namespace mirage
