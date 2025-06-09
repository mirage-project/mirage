#include "mirage/search/symbolic_graph/tensor_dim_expr.h"
#include "mirage/utils/hash_utils.h"

namespace mirage {
namespace search {

TensorDimVar::TensorDimVar(tensor_dim_var_index_t index, TensorDimVarType type)
    : index(index), type(type) {}

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

TensorDimPow::TensorDimPow(std::shared_ptr<TensorDimExpr const> base,
                           std::shared_ptr<TensorDimExpr const> exp)
    : base(base), exp(exp) {}

TensorDimIte::TensorDimIte(std::shared_ptr<TensorDimExpr const> cond,
                           std::shared_ptr<TensorDimExpr const> true_case,
                           std::shared_ptr<TensorDimExpr const> false_case)
    : cond(cond), true_case(true_case), false_case(false_case) {}

std::shared_ptr<TensorDimVar const>
    dim_expr_make_var(tensor_dim_var_index_t index, TensorDimVarType type) {
  return std::make_shared<TensorDimVar const>(index, type);
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

std::shared_ptr<TensorDimPow const>
    dim_expr_make_pow(std::shared_ptr<TensorDimExpr const> base,
                      std::shared_ptr<TensorDimExpr const> exp) {
  return std::make_shared<TensorDimPow const>(base, exp);
}

std::shared_ptr<TensorDimIte const>
    dim_expr_make_ite(std::shared_ptr<TensorDimExpr const> cond,
                      std::shared_ptr<TensorDimExpr const> true_case,
                      std::shared_ptr<TensorDimExpr const> false_case) {
  return std::make_shared<TensorDimIte const>(cond, true_case, false_case);
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

int TensorDimPow::get_value(DimVarAssignments const &assignments) const {
  auto int_pow = [](int base, int exp) {
    int result = 1;
    for (int i = 0; i < exp; i++) {
      result *= base;
    }
    return result;
  };
  return int_pow(base->get_value(assignments), exp->get_value(assignments));
}

int TensorDimIte::get_value(DimVarAssignments const &assignments) const {
  if (cond->get_value(assignments) != 0) {
    return true_case->get_value(assignments);
  } else {
    return false_case->get_value(assignments);
  }
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

bool TensorDimExpr::is_pow() const {
  return false;
}

bool TensorDimExpr::is_ite() const {
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

bool TensorDimPow::is_pow() const {
  return true;
}

bool TensorDimIte::is_ite() const {
  return true;
}

TensorDimVar::operator json() const {
  return json{{"opt", "var"}, {"index", index}, {"type", type}};
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

TensorDimPow::operator json() const {
  return json{{"opt", "pow"}, {"base", *base}, {"exp", *exp}};
}

TensorDimIte::operator json() const {
  return json{{"opt", "ite"},
              {"cond", *cond},
              {"true_case", *true_case},
              {"false_case", *false_case}};
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

size_t TensorDimPow::hash() const {
  size_t h = 5;
  hash_combine(h, base->hash());
  hash_combine(h, exp->hash());
  return h;
}

size_t TensorDimIte::hash() const {
  size_t h = 6;
  hash_combine(h, cond->hash());
  hash_combine(h, true_case->hash());
  hash_combine(h, false_case->hash());
  return h;
}

bool TensorDimVar::same_expr_as(
    std::shared_ptr<TensorDimExpr const> other) const {
  if (!other->is_var()) {
    return false;
  }
  return index == std::static_pointer_cast<TensorDimVar const>(other)->index &&
         type == std::static_pointer_cast<TensorDimVar const>(other)->type;
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

bool TensorDimPow::same_expr_as(
    std::shared_ptr<TensorDimExpr const> other) const {
  if (!other->is_pow()) {
    return false;
  }
  auto other_pow = std::static_pointer_cast<TensorDimPow const>(other);
  return base->same_expr_as(other_pow->base) &&
         exp->same_expr_as(other_pow->exp);
}

bool TensorDimIte::same_expr_as(
    std::shared_ptr<TensorDimExpr const> other) const {
  if (!other->is_ite()) {
    return false;
  }
  auto other_ite = std::static_pointer_cast<TensorDimIte const>(other);
  return cond->same_expr_as(other_ite->cond) &&
         true_case->same_expr_as(other_ite->true_case) &&
         false_case->same_expr_as(other_ite->false_case);
}

z3::expr TensorDimVar::to_z3(z3::context &c,
                             DimVarAssignments const &assign,
                             bool log_scaled) const {
  if (assign.has_assignment(index)) {
    if (type == TensorDimVarType::BOOL) {
      return c.bool_val(assign.get_value(index) != 0);
    }
    if (log_scaled) {
      int log_scaled_value = std::ceil(std::log2(assign.get_value(index)));
      return c.int_val(log_scaled_value);
    }
    return c.int_val(assign.get_value(index));
  }
  if (type == TensorDimVarType::BOOL) {
    std::string z3_var_name = "map_var_" + std::to_string(index);
    return c.bool_const(z3_var_name.c_str());
  } else {
    assert(type == TensorDimVarType::INT);
    std::string z3_var_name = "tensor_dim_var_" + std::to_string(index);
    return c.int_const(z3_var_name.c_str());
  }
}

z3::expr TensorDimConst::to_z3(z3::context &c,
                               DimVarAssignments const &assign,
                               bool log_scaled) const {
  if (log_scaled) {
    int log_scaled_value = std::ceil(std::log2(value));
    return c.int_val(log_scaled_value);
  }
  return c.int_val(value);
}

z3::expr TensorDimAdd::to_z3(z3::context &c,
                             DimVarAssignments const &assign,
                             bool log_scaled) const {
  if (log_scaled) {
    assert(false &&
           "Do not support addition between log scaled symbolic dimensions");
  }
  return lhs->to_z3(c, assign, log_scaled) + rhs->to_z3(c, assign, log_scaled);
}

z3::expr TensorDimMul::to_z3(z3::context &c,
                             DimVarAssignments const &assign,
                             bool log_scaled) const {
  if (log_scaled) {
    return lhs->to_z3(c, assign, log_scaled) +
           rhs->to_z3(c, assign, log_scaled);
  }
  return lhs->to_z3(c, assign, log_scaled) * rhs->to_z3(c, assign, log_scaled);
}

z3::expr TensorDimDiv::to_z3(z3::context &c,
                             DimVarAssignments const &assign,
                             bool log_scaled) const {
  if (log_scaled) {
    return lhs->to_z3(c, assign, log_scaled) -
           rhs->to_z3(c, assign, log_scaled);
  }
  return lhs->to_z3(c, assign, log_scaled) / rhs->to_z3(c, assign, log_scaled);
}

z3::expr TensorDimPow::to_z3(z3::context &c,
                             DimVarAssignments const &assign,
                             bool log_scaled) const {
  if (log_scaled) {
    return base->to_z3(c, assign, log_scaled) * exp->to_z3(c, assign, false);
  }
  assert(false && "Power is only supported in log-scaled expr");
}

z3::expr TensorDimIte::to_z3(z3::context &c,
                             DimVarAssignments const &assign,
                             bool log_scaled) const {
  return z3::ite(cond->to_z3(c, assign, log_scaled) == 1,
                 true_case->to_z3(c, assign, log_scaled),
                 false_case->to_z3(c, assign, log_scaled));
}

std::string TensorDimVar::to_string() const {
  if (type == TensorDimVarType::BOOL) {
    return "bool_var_" + std::to_string(index);
  }
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

std::string TensorDimPow::to_string() const {
  return "pow(" + base->to_string() + ", " + exp->to_string() + ")";
}

std::string TensorDimIte::to_string() const {
  return "ite(" + cond->to_string() + ", " + true_case->to_string() + ", " +
         false_case->to_string() + ")";
}

} // namespace search
} // namespace mirage
