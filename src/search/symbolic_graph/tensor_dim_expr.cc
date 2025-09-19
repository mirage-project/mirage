#include "mirage/search/symbolic_graph/tensor_dim_expr.h"
#include "mirage/utils/containers.h"
#include "mirage/utils/hash_utils.h"
#include <unordered_set>

namespace mirage {
namespace search {

TensorDimVar::TensorDimVar(tensor_dim_var_index_t index)
    : index(index) {}

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

TensorDimGe::TensorDimGe(std::shared_ptr<TensorDimExpr const> lhs,
                           std::shared_ptr<TensorDimExpr const> rhs)
    : lhs(lhs), rhs(rhs) {}

TensorDimLe::TensorDimLe(std::shared_ptr<TensorDimExpr const> lhs,
                           std::shared_ptr<TensorDimExpr const> rhs)
    : lhs(lhs), rhs(rhs) {}

TensorDimGt::TensorDimGt(std::shared_ptr<TensorDimExpr const> lhs,
                           std::shared_ptr<TensorDimExpr const> rhs)
    : lhs(lhs), rhs(rhs) {}

TensorDimLt::TensorDimLt(std::shared_ptr<TensorDimExpr const> lhs,
                           std::shared_ptr<TensorDimExpr const> rhs)
    : lhs(lhs), rhs(rhs) {}

TensorDimEq::TensorDimEq(std::shared_ptr<TensorDimExpr const> lhs,
                           std::shared_ptr<TensorDimExpr const> rhs)
    : lhs(lhs), rhs(rhs) {}

TensorDimDisj::TensorDimDisj(std::vector<std::shared_ptr<TensorDimExpr const>> const &args)
    : args(args) {}

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

std::shared_ptr<TensorDimGe const>
    dim_expr_make_ge(std::shared_ptr<TensorDimExpr const> lhs,
                      std::shared_ptr<TensorDimExpr const> rhs) {
  return std::make_shared<TensorDimGe const>(lhs, rhs);
}

std::shared_ptr<TensorDimLe const>
    dim_expr_make_le(std::shared_ptr<TensorDimExpr const> lhs,
                      std::shared_ptr<TensorDimExpr const> rhs) {
  return std::make_shared<TensorDimLe const>(lhs, rhs);
}

std::shared_ptr<TensorDimGt const>
    dim_expr_make_gt(std::shared_ptr<TensorDimExpr const> lhs,
                      std::shared_ptr<TensorDimExpr const> rhs) {
  return std::make_shared<TensorDimGt const>(lhs, rhs);
}

std::shared_ptr<TensorDimLt const>
    dim_expr_make_lt(std::shared_ptr<TensorDimExpr const> lhs,
                      std::shared_ptr<TensorDimExpr const> rhs) {
  return std::make_shared<TensorDimLt const>(lhs, rhs);
}

std::shared_ptr<TensorDimEq const>
    dim_expr_make_eq(std::shared_ptr<TensorDimExpr const> lhs,
                      std::shared_ptr<TensorDimExpr const> rhs) {
  return std::make_shared<TensorDimEq const>(lhs, rhs);
}

std::shared_ptr<TensorDimDisj const>
    dim_expr_make_disj(std::vector<std::shared_ptr<TensorDimExpr const>> const &args) {
  return std::make_shared<TensorDimDisj const>(args);
}

std::shared_ptr<TensorDimAdd const> operator+(std::shared_ptr<TensorDimExpr const> lhs, std::shared_ptr<TensorDimExpr const> rhs) {
  return dim_expr_make_add(lhs, rhs);
}

std::shared_ptr<TensorDimMul const> operator*(std::shared_ptr<TensorDimExpr const> lhs, std::shared_ptr<TensorDimExpr const> rhs) {
  return dim_expr_make_mul(lhs, rhs);
}

std::shared_ptr<TensorDimDiv const> operator/(std::shared_ptr<TensorDimExpr const> lhs, std::shared_ptr<TensorDimExpr const> rhs) {
  return dim_expr_make_div(lhs, rhs);
}

std::shared_ptr<TensorDimPow const> operator^(std::shared_ptr<TensorDimExpr const> base, std::shared_ptr<TensorDimExpr const> exp) {
  return dim_expr_make_pow(base, exp);
}

std::shared_ptr<TensorDimGe const> operator>=(std::shared_ptr<TensorDimExpr const> lhs, std::shared_ptr<TensorDimExpr const> rhs) {
  return dim_expr_make_ge(lhs, rhs);
}

std::shared_ptr<TensorDimGe const> operator>=(std::shared_ptr<TensorDimExpr const> lhs, int rhs) {
  return dim_expr_make_ge(lhs, dim_expr_make_const(rhs));
}

std::shared_ptr<TensorDimLe const> operator<=(std::shared_ptr<TensorDimExpr const> lhs, std::shared_ptr<TensorDimExpr const> rhs) {
  return dim_expr_make_le(lhs, rhs);
}

std::shared_ptr<TensorDimLe const> operator<=(std::shared_ptr<TensorDimExpr const> lhs, int rhs) {
  return dim_expr_make_le(lhs, dim_expr_make_const(rhs));
}

std::shared_ptr<TensorDimLt const> operator<(std::shared_ptr<TensorDimExpr const> lhs, std::shared_ptr<TensorDimExpr const> rhs) {
  return dim_expr_make_lt(lhs, rhs);
}

std::shared_ptr<TensorDimLt const> operator<(std::shared_ptr<TensorDimExpr const> lhs, int rhs) {
  return dim_expr_make_lt(lhs, dim_expr_make_const(rhs));
}

std::shared_ptr<TensorDimGt const> operator>(std::shared_ptr<TensorDimExpr const> lhs, std::shared_ptr<TensorDimExpr const> rhs) {
  return dim_expr_make_gt(lhs, rhs);
}

std::shared_ptr<TensorDimGt const> operator>(std::shared_ptr<TensorDimExpr const> lhs, int rhs) {
  return dim_expr_make_gt(lhs, dim_expr_make_const(rhs));
}

std::shared_ptr<TensorDimEq const> operator==(std::shared_ptr<TensorDimExpr const> lhs, std::shared_ptr<TensorDimExpr const> rhs) {
  return dim_expr_make_eq(lhs, rhs);
}

std::shared_ptr<TensorDimEq const> operator==(std::shared_ptr<TensorDimExpr const> lhs, int rhs) {
  return dim_expr_make_eq(lhs, dim_expr_make_const(rhs));
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

int TensorDimGe::get_value(DimVarAssignments const &assignments) const {
  return lhs->get_value(assignments) >= rhs->get_value(assignments);
}

int TensorDimLe::get_value(DimVarAssignments const &assignments) const {
  return lhs->get_value(assignments) <= rhs->get_value(assignments);
}

int TensorDimGt::get_value(DimVarAssignments const &assignments) const {
  return lhs->get_value(assignments) > rhs->get_value(assignments);
}

int TensorDimLt::get_value(DimVarAssignments const &assignments) const {
  return lhs->get_value(assignments) < rhs->get_value(assignments);
}

int TensorDimEq::get_value(DimVarAssignments const &assignments) const {
  return lhs->get_value(assignments) == rhs->get_value(assignments);
}

int TensorDimDisj::get_value(DimVarAssignments const &assignments) const {
  for (auto const &arg : args) {
    if (arg->get_value(assignments) != 0) {
      return 1;
    }
  }
  return 0;
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

bool TensorDimExpr::is_ge() const {
  return false;
}

bool TensorDimExpr::is_le() const {
  return false;
}

bool TensorDimExpr::is_gt() const {
  return false;
}

bool TensorDimExpr::is_lt() const {
  return false;
}

bool TensorDimExpr::is_eq() const {
  return false;
}

bool TensorDimExpr::is_disj() const {
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

bool TensorDimGe::is_ge() const {
  return true;
}

bool TensorDimLe::is_le() const {
  return true;
}

bool TensorDimGt::is_gt() const {
  return true;
}

bool TensorDimLt::is_lt() const {
  return true;
}

bool TensorDimEq::is_eq() const {
  return true;
}

bool TensorDimDisj::is_disj() const {
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

TensorDimPow::operator json() const {
  return json{{"opt", "pow"}, {"base", *base}, {"exp", *exp}};
}

TensorDimIte::operator json() const {
  return json{{"opt", "ite"},
              {"cond", *cond},
              {"true_case", *true_case},
              {"false_case", *false_case}};
}

TensorDimGe::operator json() const {
  return json{{"opt", "ge"}, {"lhs", *lhs}, {"rhs", *rhs}};
}

TensorDimLe::operator json() const {
  return json{{"opt", "le"}, {"lhs", *lhs}, {"rhs", *rhs}};
}

TensorDimGt::operator json() const {
  return json{{"opt", "gt"}, {"lhs", *lhs}, {"rhs", *rhs}};
}

TensorDimLt::operator json() const {
  return json{{"opt", "lt"}, {"lhs", *lhs}, {"rhs", *rhs}};
}

TensorDimEq::operator json() const {
  return json{{"opt", "eq"}, {"lhs", *lhs}, {"rhs", *rhs}};
}

TensorDimDisj::operator json() const {
  std::vector<json> args_json;
  for (auto const &arg : args) {
    args_json.push_back(*arg);
  }
  return json{{"opt", "disj"}, {"args", args_json}};
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

size_t TensorDimGe::hash() const {
  size_t h = 7;
  hash_combine(h, lhs->hash());
  hash_combine(h, rhs->hash());
  return h;
}

size_t TensorDimLe::hash() const {
  size_t h = 8;
  hash_combine(h, lhs->hash());
  hash_combine(h, rhs->hash());
  return h;
}

size_t TensorDimGt::hash() const {
  size_t h = 9;
  hash_combine(h, lhs->hash());
  hash_combine(h, rhs->hash());
  return h;
}

size_t TensorDimLt::hash() const {
  size_t h = 10;
  hash_combine(h, lhs->hash());
  hash_combine(h, rhs->hash());
  return h;
}

size_t TensorDimEq::hash() const {
  size_t h = 11;
  hash_combine(h, lhs->hash());
  hash_combine(h, rhs->hash());
  return h;
}

size_t TensorDimDisj::hash() const {
  size_t h = 12;
  hash_combine(h, args);
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

bool TensorDimGe::same_expr_as(
    std::shared_ptr<TensorDimExpr const> other) const {
  if (!other->is_ge()) {
    return false;
  }
  auto other_ge = std::static_pointer_cast<TensorDimGe const>(other);
  return lhs->same_expr_as(other_ge->lhs) && rhs->same_expr_as(other_ge->rhs);
}

bool TensorDimLe::same_expr_as(
    std::shared_ptr<TensorDimExpr const> other) const {
  if (!other->is_le()) {
    return false;
  }
  auto other_le = std::static_pointer_cast<TensorDimLe const>(other);
  return lhs->same_expr_as(other_le->lhs) && rhs->same_expr_as(other_le->rhs);
}

bool TensorDimGt::same_expr_as(
    std::shared_ptr<TensorDimExpr const> other) const {
  if (!other->is_gt()) {
    return false;
  }
  auto other_gt = std::static_pointer_cast<TensorDimGt const>(other);
  return lhs->same_expr_as(other_gt->lhs) && rhs->same_expr_as(other_gt->rhs);
}

bool TensorDimLt::same_expr_as(
    std::shared_ptr<TensorDimExpr const> other) const {
  if (!other->is_lt()) {
    return false;
  }
  auto other_lt = std::static_pointer_cast<TensorDimLt const>(other);
  return lhs->same_expr_as(other_lt->lhs) && rhs->same_expr_as(other_lt->rhs);
}

bool TensorDimEq::same_expr_as(
    std::shared_ptr<TensorDimExpr const> other) const {
  if (!other->is_eq()) {
    return false;
  }
  auto other_eq = std::static_pointer_cast<TensorDimEq const>(other);
  return lhs->same_expr_as(other_eq->lhs) && rhs->same_expr_as(other_eq->rhs);
}

bool TensorDimDisj::same_expr_as(
    std::shared_ptr<TensorDimExpr const> other) const {
  if (!other->is_disj()) {
    return false;
  }
  auto other_disj = std::static_pointer_cast<TensorDimDisj const>(other);
  return args == other_disj->args;
}

z3::expr TensorDimVar::to_z3(z3::context &c,
                             DimVarAssignments const &assign,
                             bool log_scaled) const {
  if (assign.has_assignment(index)) {
    if (log_scaled) {
      int log_scaled_value = std::ceil(std::log2(assign.get_value(index)));
      return c.int_val(log_scaled_value);
    }
    return c.int_val(assign.get_value(index));
  }
  std::string z3_var_name = "var_" + std::to_string(index);
  return c.int_const(z3_var_name.c_str());
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
  return z3::ite(cond->to_z3(c, assign, log_scaled),
                 true_case->to_z3(c, assign, log_scaled),
                 false_case->to_z3(c, assign, log_scaled));
}

z3::expr TensorDimGe::to_z3(z3::context &c,
                             DimVarAssignments const &assign,
                             bool log_scaled) const {
  return lhs->to_z3(c, assign, log_scaled) >= rhs->to_z3(c, assign, log_scaled);
}

z3::expr TensorDimLe::to_z3(z3::context &c,
                             DimVarAssignments const &assign,
                             bool log_scaled) const {
  return lhs->to_z3(c, assign, log_scaled) <= rhs->to_z3(c, assign, log_scaled);
}

z3::expr TensorDimGt::to_z3(z3::context &c,
                             DimVarAssignments const &assign,
                             bool log_scaled) const {
  return lhs->to_z3(c, assign, log_scaled) > rhs->to_z3(c, assign, log_scaled);
}

z3::expr TensorDimLt::to_z3(z3::context &c,
                             DimVarAssignments const &assign,
                             bool log_scaled) const {
  return lhs->to_z3(c, assign, log_scaled) < rhs->to_z3(c, assign, log_scaled);
}

z3::expr TensorDimEq::to_z3(z3::context &c,
                             DimVarAssignments const &assign,
                             bool log_scaled) const {
  return lhs->to_z3(c, assign, log_scaled) == rhs->to_z3(c, assign, log_scaled);
}

z3::expr TensorDimDisj::to_z3(z3::context &c,
                             DimVarAssignments const &assign,
                             bool log_scaled) const {
  z3::expr_vector args_z3(c);
  for (auto const &arg : args) {
    args_z3.push_back(arg->to_z3(c, assign, log_scaled));
  }
  return z3::mk_or(args_z3);
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

std::string TensorDimPow::to_string() const {
  return "pow(" + base->to_string() + ", " + exp->to_string() + ")";
}

std::string TensorDimIte::to_string() const {
  return "ite(" + cond->to_string() + ", " + true_case->to_string() + ", " +
         false_case->to_string() + ")";
}

std::string TensorDimGe::to_string() const {
  return lhs->to_string() + " >= " + rhs->to_string();
}

std::string TensorDimLe::to_string() const {
  return lhs->to_string() + " <= " + rhs->to_string();
}

std::string TensorDimGt::to_string() const {
  return lhs->to_string() + " > " + rhs->to_string();
}

std::string TensorDimLt::to_string() const {
  return lhs->to_string() + " < " + rhs->to_string();
}

std::string TensorDimEq::to_string() const {
  return lhs->to_string() + " == " + rhs->to_string();
}

std::string TensorDimDisj::to_string() const {
  std::string result = "(";
  for (auto const &arg : args) {
    result += arg->to_string() + " || ";
  }
  result += ")";
  return result;
}

std::unordered_set<std::shared_ptr<TensorDimVar const>> TensorDimVar::get_all_vars() const {
  std::shared_ptr<TensorDimVar const> var = std::static_pointer_cast<TensorDimVar const>(shared_from_this());
  return {var};
}

std::unordered_set<std::shared_ptr<TensorDimVar const>> TensorDimConst::get_all_vars() const {
  return {};
}

std::unordered_set<std::shared_ptr<TensorDimVar const>> TensorDimAdd::get_all_vars() const {
  return set_union(lhs->get_all_vars(), rhs->get_all_vars());
}

std::unordered_set<std::shared_ptr<TensorDimVar const>> TensorDimMul::get_all_vars() const {
  return set_union(lhs->get_all_vars(), rhs->get_all_vars());
}

std::unordered_set<std::shared_ptr<TensorDimVar const>> TensorDimDiv::get_all_vars() const {
  return set_union(lhs->get_all_vars(), rhs->get_all_vars());
}

std::unordered_set<std::shared_ptr<TensorDimVar const>> TensorDimPow::get_all_vars() const {
  return set_union(base->get_all_vars(), exp->get_all_vars());
}

std::unordered_set<std::shared_ptr<TensorDimVar const>> TensorDimIte::get_all_vars() const {
  return set_union(cond->get_all_vars(), set_union(true_case->get_all_vars(), false_case->get_all_vars()));
}

std::unordered_set<std::shared_ptr<TensorDimVar const>> TensorDimGe::get_all_vars() const {
  return set_union(lhs->get_all_vars(), rhs->get_all_vars());
}

std::unordered_set<std::shared_ptr<TensorDimVar const>> TensorDimLe::get_all_vars() const {
  return set_union(lhs->get_all_vars(), rhs->get_all_vars());
}

std::unordered_set<std::shared_ptr<TensorDimVar const>> TensorDimGt::get_all_vars() const {
  return set_union(lhs->get_all_vars(), rhs->get_all_vars());
}

std::unordered_set<std::shared_ptr<TensorDimVar const>> TensorDimLt::get_all_vars() const {
  return set_union(lhs->get_all_vars(), rhs->get_all_vars());
}

std::unordered_set<std::shared_ptr<TensorDimVar const>> TensorDimEq::get_all_vars() const {
  return set_union(lhs->get_all_vars(), rhs->get_all_vars());
}

std::unordered_set<std::shared_ptr<TensorDimVar const>> TensorDimDisj::get_all_vars() const {
  return set_union(vector_map(this->args, [](auto const &arg) { return arg->get_all_vars(); }));
}

} // namespace search
} // namespace mirage

namespace std {

size_t hash<mirage::search::TensorDimExpr>::operator()(mirage::search::TensorDimExpr const &expr) const {
  return expr.hash();
}

size_t hash<std::shared_ptr<mirage::search::TensorDimExpr const>>::operator()(std::shared_ptr<mirage::search::TensorDimExpr const> const &expr) const {
  return expr->hash();
}

size_t hash<std::shared_ptr<mirage::search::TensorDimVar const>>::operator()(std::shared_ptr<mirage::search::TensorDimVar const> const &var) const {
  return var->hash();
}

} // namespace std
