#include "mirage/search/symbolic_graph/tensor_dim_expr.h"
#include "mirage/search/abstract_expr/abstract_expr.h"
#include "mirage/utils/containers.h"
#include "mirage/utils/hash_utils.h"
#include "mirage/utils/math_utils.h"
#include <unordered_set>

namespace mirage {
namespace search {

TensorDimVar::TensorDimVar(tensor_dim_var_index_t index, bool is_boolean)
    : index(index), is_boolean(is_boolean) {}

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
    dim_expr_make_var(tensor_dim_var_index_t index, bool is_boolean) {
  return std::make_shared<TensorDimVar const>(index, is_boolean);
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

std::shared_ptr<TensorDimAdd const>
    operator+(std::shared_ptr<TensorDimExpr const> lhs,
              std::shared_ptr<TensorDimExpr const> rhs) {
  return dim_expr_make_add(lhs, rhs);
}

std::shared_ptr<TensorDimMul const>
    operator*(std::shared_ptr<TensorDimExpr const> lhs,
              std::shared_ptr<TensorDimExpr const> rhs) {
  return dim_expr_make_mul(lhs, rhs);
}

std::shared_ptr<TensorDimDiv const>
    operator/(std::shared_ptr<TensorDimExpr const> lhs,
              std::shared_ptr<TensorDimExpr const> rhs) {
  return dim_expr_make_div(lhs, rhs);
}

std::shared_ptr<TensorDimExpr const>
    operator-(std::shared_ptr<TensorDimExpr const> lhs,
              std::shared_ptr<TensorDimExpr const> rhs) {
  return lhs + (dim_expr_make_const(-1) * rhs);
}

std::shared_ptr<TensorDimExpr const> TensorDimVar::with_partial_assignment(
    DimVarAssignment const &partial_assignment) const {
  if (partial_assignment.has_assignment(index)) {
    return dim_expr_make_const(partial_assignment.get_value(index));
  }
  return shared_from_this();
}

std::shared_ptr<TensorDimExpr const> TensorDimConst::with_partial_assignment(
    DimVarAssignment const &partial_assignment) const {
  return shared_from_this();
}

std::shared_ptr<TensorDimExpr const> TensorDimAdd::with_partial_assignment(
    DimVarAssignment const &partial_assignment) const {
  std::shared_ptr<TensorDimExpr const> lhs_expr =
      lhs->with_partial_assignment(partial_assignment);
  std::shared_ptr<TensorDimExpr const> rhs_expr =
      rhs->with_partial_assignment(partial_assignment);
  if (lhs_expr->is_const() && rhs_expr->is_const()) {
    std::shared_ptr<TensorDimConst const> lhs_const =
        std::static_pointer_cast<TensorDimConst const>(lhs_expr);
    std::shared_ptr<TensorDimConst const> rhs_const =
        std::static_pointer_cast<TensorDimConst const>(rhs_expr);
    return dim_expr_make_const(lhs_const->value + rhs_const->value);
  }
  return dim_expr_make_add(lhs_expr, rhs_expr);
}

std::shared_ptr<TensorDimExpr const> TensorDimMul::with_partial_assignment(
    DimVarAssignment const &partial_assignment) const {
  std::shared_ptr<TensorDimExpr const> lhs_expr =
      lhs->with_partial_assignment(partial_assignment);
  std::shared_ptr<TensorDimExpr const> rhs_expr =
      rhs->with_partial_assignment(partial_assignment);
  if (lhs_expr->is_const() && rhs_expr->is_const()) {
    std::shared_ptr<TensorDimConst const> lhs_const =
        std::static_pointer_cast<TensorDimConst const>(lhs_expr);
    std::shared_ptr<TensorDimConst const> rhs_const =
        std::static_pointer_cast<TensorDimConst const>(rhs_expr);
    return dim_expr_make_const(lhs_const->value * rhs_const->value);
  }
  return dim_expr_make_mul(lhs_expr, rhs_expr);
}

std::shared_ptr<TensorDimExpr const> TensorDimDiv::with_partial_assignment(
    DimVarAssignment const &partial_assignment) const {
  std::shared_ptr<TensorDimExpr const> lhs_expr =
      lhs->with_partial_assignment(partial_assignment);
  std::shared_ptr<TensorDimExpr const> rhs_expr =
      rhs->with_partial_assignment(partial_assignment);
  if (lhs_expr->is_const() && rhs_expr->is_const()) {
    std::shared_ptr<TensorDimConst const> lhs_const =
        std::static_pointer_cast<TensorDimConst const>(lhs_expr);
    std::shared_ptr<TensorDimConst const> rhs_const =
        std::static_pointer_cast<TensorDimConst const>(rhs_expr);
    return dim_expr_make_const(lhs_const->value / rhs_const->value);
  }
  return dim_expr_make_div(lhs_expr, rhs_expr);
}

std::optional<int>
    TensorDimExpr::maybe_get_value(DimVarAssignment const &assignments) const {
  std::shared_ptr<TensorDimExpr const> expr =
      with_partial_assignment(assignments);
  if (expr->is_const()) {
    std::shared_ptr<TensorDimConst const> const_expr =
        std::static_pointer_cast<TensorDimConst const>(expr);
    return const_expr->value;
  }
  return std::nullopt;
}

int TensorDimExpr::get_value(DimVarAssignment const &assignments) const {
  std::optional<int> value = maybe_get_value(assignments);
  assert(value);
  return *value;
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

bool TensorDimExpr::is_bool_var() const {
  return false;
}

bool TensorDimVar::is_var() const {
  return true;
}

bool TensorDimVar::is_bool_var() const {
  return is_boolean;
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

float TensorDimExpr::get_float_value(
    DimVarAssignment const &assignments) const {
  assert(false);
}

float TensorDimVar::get_float_value(DimVarAssignment const &assignments) const {
  return (float)get_value(assignments);
}

float TensorDimConst::get_float_value(
    DimVarAssignment const &assignments) const {
  return (float)value;
}

float TensorDimAdd::get_float_value(DimVarAssignment const &assignments) const {
  return lhs->get_float_value(assignments) + rhs->get_float_value(assignments);
}

float TensorDimMul::get_float_value(DimVarAssignment const &assignments) const {
  return lhs->get_float_value(assignments) * rhs->get_float_value(assignments);
}

float TensorDimDiv::get_float_value(DimVarAssignment const &assignments) const {
  return lhs->get_float_value(assignments) / rhs->get_float_value(assignments);
}

TensorDimVar::operator json() const {
  json j{{"opt", "var"}, {"index", index}};
  if (is_boolean) {
    j["is_boolean"] = true;
  }
  return j;
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

void from_json(json const &j, SymbolicTensorDim &dim) {
  if (j.is_null()) {
    dim = nullptr;
    return;
  }
  std::string opt = j.at("opt").get<std::string>();
  if (opt == "var") {
    bool is_boolean =
        j.contains("is_boolean") && j.at("is_boolean").get<bool>();
    dim = std::make_shared<TensorDimVar>(
        j.at("index").get<tensor_dim_var_index_t>(), is_boolean);
  } else if (opt == "const") {
    dim = dim_expr_make_const(j.at("value").get<int>());
  } else if (opt == "add") {
    SymbolicTensorDim lhs, rhs;
    from_json(j.at("lhs"), lhs);
    from_json(j.at("rhs"), rhs);
    dim = dim_expr_make_add(lhs, rhs);
  } else if (opt == "mul") {
    SymbolicTensorDim lhs, rhs;
    from_json(j.at("lhs"), lhs);
    from_json(j.at("rhs"), rhs);
    dim = dim_expr_make_mul(lhs, rhs);
  } else if (opt == "div") {
    SymbolicTensorDim lhs, rhs;
    from_json(j.at("lhs"), lhs);
    from_json(j.at("rhs"), rhs);
    dim = dim_expr_make_div(lhs, rhs);
  } else {
    assert(false && "unknown TensorDimExpr opt");
  }
}

size_t TensorDimVar::hash() const {
  size_t h = 0;
  hash_combine(h, index);
  hash_combine(h, is_boolean);
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
  auto other_var = std::static_pointer_cast<TensorDimVar const>(other);
  return index == other_var->index && is_boolean == other_var->is_boolean;
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

std::string TensorDimVar::to_egg() const {
  return this->to_string();
}

std::string TensorDimConst::to_egg() const {
  return std::to_string(value);
}

std::string TensorDimAdd::to_egg() const {
  return "(+ " + lhs->to_egg() + " " + rhs->to_egg() + ")";
}

std::string TensorDimMul::to_egg() const {
  return "(* " + lhs->to_egg() + " " + rhs->to_egg() + ")";
}

std::string TensorDimDiv::to_egg() const {
  return "(/ " + lhs->to_egg() + " " + rhs->to_egg() + ")";
}

std::unordered_set<std::shared_ptr<TensorDimVar const>>
    TensorDimVar::get_all_vars() const {
  std::shared_ptr<TensorDimVar const> var =
      std::static_pointer_cast<TensorDimVar const>(shared_from_this());
  return {var};
}

std::unordered_set<std::shared_ptr<TensorDimVar const>>
    TensorDimConst::get_all_vars() const {
  return {};
}

std::unordered_set<std::shared_ptr<TensorDimVar const>>
    TensorDimAdd::get_all_vars() const {
  return set_union(lhs->get_all_vars(), rhs->get_all_vars());
}

std::unordered_set<std::shared_ptr<TensorDimVar const>>
    TensorDimMul::get_all_vars() const {
  return set_union(lhs->get_all_vars(), rhs->get_all_vars());
}

std::unordered_set<std::shared_ptr<TensorDimVar const>>
    TensorDimDiv::get_all_vars() const {
  return set_union(lhs->get_all_vars(), rhs->get_all_vars());
}

bool TensorDimExpr::symbolically_equivalent_to(
    std::shared_ptr<TensorDimExpr const> other) const {
  std::string expr1 = this->to_egg();
  std::string expr2 = other->to_egg();
  return is_equiv(expr1.c_str(), expr2.c_str());
}

bool TensorDimExpr::can_be_symbolically_equivalent_to(
    std::shared_ptr<TensorDimExpr const> other) const {

  int value1 = get_value_with_all_vars_being(0, shared_from_this());
  int value2 = get_value_with_all_vars_being(0, other);
  if (value1 != value2) {
    return false;
  }

  return true;
}

bool TensorDimExpr::is_one() const {
  if (this->is_const()) {
    std::shared_ptr<TensorDimConst const> const_expr =
        std::static_pointer_cast<TensorDimConst const>(shared_from_this());
    return const_expr->value == 1;
  }
  return false;
}

int get_value_with_all_vars_being(int value,
                                  std::shared_ptr<TensorDimExpr const> expr) {
  std::unordered_set<std::shared_ptr<TensorDimVar const>> all_vars =
      expr->get_all_vars();
  DimVarAssignment assignment;
  for (auto const &var : all_vars) {
    assignment.assign(var->index, value);
  }
  return (int)expr->get_float_value(assignment);
}

int get_value_with_bool_vars_zero_others_one(
    std::shared_ptr<TensorDimExpr const> expr) {
  std::unordered_set<std::shared_ptr<TensorDimVar const>> all_vars =
      expr->get_all_vars();
  DimVarAssignment assignment;
  for (auto const &var : all_vars) {
    assignment.assign(var->index, var->is_boolean ? 0 : 1);
  }
  return (int)expr->get_float_value(assignment);
}

int get_value_with_bool_vars_zero_others_random(
    std::shared_ptr<TensorDimExpr const> expr) {
  std::unordered_set<std::shared_ptr<TensorDimVar const>> all_vars =
      expr->get_all_vars();
  DimVarAssignment assignment;
  for (auto const &var : all_vars) {
    assignment.assign(var->index, var->is_boolean ? 0 : (int)var->index + 2);
  }
  return (int)expr->get_float_value(assignment);
}

int get_value_with_all_vars_random(std::shared_ptr<TensorDimExpr const> expr) {
  std::unordered_set<std::shared_ptr<TensorDimVar const>> all_vars =
      expr->get_all_vars();
  DimVarAssignment assignment;
  for (auto const &var : all_vars) {
    assignment.assign(var->index, (int)var->index + 2);
  }
  return (int)expr->get_float_value(assignment);
}

} // namespace search
} // namespace mirage

namespace std {

size_t hash<mirage::search::TensorDimExpr>::operator()(
    mirage::search::TensorDimExpr const &expr) const {
  return expr.hash();
}

size_t hash<std::shared_ptr<mirage::search::TensorDimExpr const>>::operator()(
    std::shared_ptr<mirage::search::TensorDimExpr const> const &expr) const {
  return expr->hash();
}

size_t hash<std::shared_ptr<mirage::search::TensorDimVar const>>::operator()(
    std::shared_ptr<mirage::search::TensorDimVar const> const &var) const {
  return var->hash();
}

} // namespace std
