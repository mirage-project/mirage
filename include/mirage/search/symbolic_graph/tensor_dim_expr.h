#pragma once

#include "mirage/search/symbolic_graph/dim_var_assignment.h"
#include "mirage/utils/json_utils.h"

#include <memory>
#include <string>
#include <unordered_set>

namespace mirage {
namespace search {

class TensorDimVar;

class TensorDimExpr : public std::enable_shared_from_this<TensorDimExpr> {
public:
  TensorDimExpr() = default;
  virtual ~TensorDimExpr() = default;

  virtual int get_value(DimVarAssignment const &assignments) const;
  virtual float get_float_value(DimVarAssignment const &assignments) const;
  virtual std::optional<int> maybe_get_value(DimVarAssignment const &assignments) const;
  virtual std::shared_ptr<TensorDimExpr const> with_partial_assignment(DimVarAssignment const &partial_assignment) const = 0;
  virtual std::string to_egg() const = 0;
  virtual std::string to_string() const = 0;
  virtual std::unordered_set<std::shared_ptr<TensorDimVar const>> get_all_vars() const = 0;
  virtual bool is_var() const;
  virtual bool is_const() const;
  virtual bool is_add() const;
  virtual bool is_mul() const;
  virtual bool is_div() const;
  virtual bool is_bool_var() const;

  virtual size_t hash() const = 0;
  virtual operator json() const = 0;
  virtual bool same_expr_as(std::shared_ptr<TensorDimExpr const>) const = 0;
  bool symbolically_equivalent_to(std::shared_ptr<TensorDimExpr const> other) const;
  bool can_be_symbolically_equivalent_to(std::shared_ptr<TensorDimExpr const> other) const;
  bool is_one() const;
};

using SymbolicTensorDim = std::shared_ptr<TensorDimExpr const>;

} // namespace search
} // namespace mirage

namespace std {

template <>
struct hash<mirage::search::TensorDimExpr> {
  size_t operator()(mirage::search::TensorDimExpr const &expr) const;
};

template <>
struct hash<std::shared_ptr<mirage::search::TensorDimExpr const>> {
  size_t operator()(std::shared_ptr<mirage::search::TensorDimExpr const> const &expr) const;
};

template <>
struct hash<std::shared_ptr<mirage::search::TensorDimVar const>> {
  size_t operator()(std::shared_ptr<mirage::search::TensorDimVar const> const &var) const;
};

template <>
struct equal_to<std::shared_ptr<mirage::search::TensorDimExpr const>> {
  bool operator()(std::shared_ptr<mirage::search::TensorDimExpr const> const &lhs,
                 std::shared_ptr<mirage::search::TensorDimExpr const> const &rhs) const {
    if (!lhs && !rhs) return true;
    if (!lhs || !rhs) return false;
    return lhs->same_expr_as(rhs);
  }
};

} // namespace std

namespace mirage {
namespace search {

class TensorDimVar : public TensorDimExpr {
public:
  TensorDimVar(tensor_dim_var_index_t index, bool is_boolean = false);
  tensor_dim_var_index_t index;
  bool is_boolean;

  std::shared_ptr<TensorDimExpr const> with_partial_assignment(DimVarAssignment const &partial_assignment) const override;
  std::string to_egg() const override;
  std::string to_string() const override;
  std::unordered_set<std::shared_ptr<TensorDimVar const>> get_all_vars() const override;
  bool is_var() const override;
  bool is_bool_var() const override;
  float get_float_value(DimVarAssignment const &assignments) const override;

  size_t hash() const override;
  operator json() const override;
  bool same_expr_as(std::shared_ptr<TensorDimExpr const>) const override;
};

std::shared_ptr<TensorDimVar const>
    dim_expr_make_var(tensor_dim_var_index_t index, bool is_boolean = false);

class TensorDimConst : public TensorDimExpr {
public:
  TensorDimConst(int value);
  int value;

  std::shared_ptr<TensorDimExpr const> with_partial_assignment(DimVarAssignment const &partial_assignment) const override;
  std::string to_egg() const override;
  std::string to_string() const override;
  std::unordered_set<std::shared_ptr<TensorDimVar const>> get_all_vars() const override;
  bool is_const() const override;
  float get_float_value(DimVarAssignment const &assignments) const override;

  size_t hash() const override;
  operator json() const override;
  bool same_expr_as(std::shared_ptr<TensorDimExpr const>) const override;
};

std::shared_ptr<TensorDimConst const> dim_expr_make_const(int value);

class TensorDimAdd : public TensorDimExpr {
public:
  TensorDimAdd(std::shared_ptr<TensorDimExpr const> lhs,
               std::shared_ptr<TensorDimExpr const> rhs);
  std::shared_ptr<TensorDimExpr const> lhs, rhs;

  std::shared_ptr<TensorDimExpr const> with_partial_assignment(DimVarAssignment const &partial_assignment) const override;
  std::string to_egg() const override;
  std::string to_string() const override;
  std::unordered_set<std::shared_ptr<TensorDimVar const>> get_all_vars() const override;
  bool is_add() const override;
  float get_float_value(DimVarAssignment const &assignments) const override;

  size_t hash() const override;
  operator json() const override;
  bool same_expr_as(std::shared_ptr<TensorDimExpr const>) const override;
};

std::shared_ptr<TensorDimAdd const>
    dim_expr_make_add(std::shared_ptr<TensorDimExpr const> lhs,
                      std::shared_ptr<TensorDimExpr const> rhs);

std::shared_ptr<TensorDimAdd const> operator+(std::shared_ptr<TensorDimExpr const> lhs, std::shared_ptr<TensorDimExpr const> rhs);
std::shared_ptr<TensorDimExpr const> operator-(std::shared_ptr<TensorDimExpr const> lhs, std::shared_ptr<TensorDimExpr const> rhs);

class TensorDimMul : public TensorDimExpr {
public:
  TensorDimMul(std::shared_ptr<TensorDimExpr const> lhs,
               std::shared_ptr<TensorDimExpr const> rhs);
  std::shared_ptr<TensorDimExpr const> lhs, rhs;

  std::shared_ptr<TensorDimExpr const> with_partial_assignment(DimVarAssignment const &partial_assignment) const override;
  std::string to_egg() const override;
  std::string to_string() const override;
  std::unordered_set<std::shared_ptr<TensorDimVar const>> get_all_vars() const override;
  bool is_mul() const override;
  float get_float_value(DimVarAssignment const &assignments) const override;

  size_t hash() const override;
  operator json() const override;
  bool same_expr_as(std::shared_ptr<TensorDimExpr const>) const override;
};

std::shared_ptr<TensorDimMul const>
    dim_expr_make_mul(std::shared_ptr<TensorDimExpr const> lhs,
                      std::shared_ptr<TensorDimExpr const> rhs);
std::shared_ptr<TensorDimMul const> operator*(std::shared_ptr<TensorDimExpr const> lhs, std::shared_ptr<TensorDimExpr const> rhs);

class TensorDimDiv : public TensorDimExpr {
public:
  TensorDimDiv(std::shared_ptr<TensorDimExpr const> lhs,
               std::shared_ptr<TensorDimExpr const> rhs);
  std::shared_ptr<TensorDimExpr const> lhs, rhs;

  std::shared_ptr<TensorDimExpr const> with_partial_assignment(DimVarAssignment const &partial_assignment) const override;
  std::string to_egg() const override;
  std::string to_string() const override;
  std::unordered_set<std::shared_ptr<TensorDimVar const>> get_all_vars() const override;
  bool is_div() const override;
  float get_float_value(DimVarAssignment const &assignments) const override;

  size_t hash() const override;
  operator json() const override;
  bool same_expr_as(std::shared_ptr<TensorDimExpr const>) const override;
};

std::shared_ptr<TensorDimDiv const>
    dim_expr_make_div(std::shared_ptr<TensorDimExpr const> lhs,
                      std::shared_ptr<TensorDimExpr const> rhs);
std::shared_ptr<TensorDimDiv const> operator/(std::shared_ptr<TensorDimExpr const> lhs, std::shared_ptr<TensorDimExpr const> rhs);

int get_value_with_all_vars_being(int value, std::shared_ptr<TensorDimExpr const> expr);
int get_value_with_bool_vars_zero_others_one(std::shared_ptr<TensorDimExpr const> expr);
int get_value_with_bool_vars_zero_others_random(std::shared_ptr<TensorDimExpr const> expr);
int get_value_with_all_vars_random(std::shared_ptr<TensorDimExpr const> expr);

void from_json(json const &j, SymbolicTensorDim &dim);

} // namespace search
} // namespace mirage
