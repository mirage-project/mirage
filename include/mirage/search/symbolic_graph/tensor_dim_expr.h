#pragma once

#include "mirage/search/symbolic_graph/dim_var_assignments.h"
#include "mirage/utils/json_utils.h"
#include "z3++.h"

#include <memory>
#include <string>

namespace mirage {
namespace search {

class TensorDimExpr {
public:
  TensorDimExpr() = default;
  virtual ~TensorDimExpr() = default;

  virtual int get_value(DimVarAssignments const &assignments) const = 0;
  virtual z3::expr to_z3(z3::context &c,
                         DimVarAssignments const &assign,
                         bool log_scaled = false) const = 0;
  virtual std::string to_string() const = 0;
  virtual bool is_var() const;
  virtual bool is_const() const;
  virtual bool is_add() const;
  virtual bool is_mul() const;
  virtual bool is_div() const;
  virtual bool is_pow() const;
  virtual bool is_ite() const;

  virtual size_t hash() const = 0;
  virtual operator json() const = 0;
  virtual bool same_expr_as(std::shared_ptr<TensorDimExpr const>) const = 0;
};

enum class TensorDimVarType {
  INT,
  BOOL,
};

NLOHMANN_JSON_SERIALIZE_ENUM(TensorDimVarType,
                             {
                                 {TensorDimVarType::INT, "INT"},
                                 {TensorDimVarType::BOOL, "BOOL"},
                             })

class TensorDimVar : public TensorDimExpr {
public:
  TensorDimVar(tensor_dim_var_index_t index,
               TensorDimVarType type = TensorDimVarType::INT);
  tensor_dim_var_index_t index;
  TensorDimVarType type;

  int get_value(DimVarAssignments const &assignments) const override;
  z3::expr to_z3(z3::context &c,
                 DimVarAssignments const &assign,
                 bool log_scaled) const override;
  std::string to_string() const override;
  bool is_var() const override;

  size_t hash() const override;
  operator json() const override;
  bool same_expr_as(std::shared_ptr<TensorDimExpr const>) const override;
};

std::shared_ptr<TensorDimVar const>
    dim_expr_make_var(tensor_dim_var_index_t index,
                      TensorDimVarType type = TensorDimVarType::INT);

class TensorDimConst : public TensorDimExpr {
public:
  TensorDimConst(int value);
  int value;

  int get_value(DimVarAssignments const &assignments) const override;
  z3::expr to_z3(z3::context &c,
                 DimVarAssignments const &assign,
                 bool log_scaled) const override;
  std::string to_string() const override;
  bool is_const() const override;

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

  int get_value(DimVarAssignments const &assignments) const override;
  z3::expr to_z3(z3::context &c,
                 DimVarAssignments const &assign,
                 bool log_scaled) const override;
  std::string to_string() const override;
  bool is_add() const override;

  size_t hash() const override;
  operator json() const override;
  bool same_expr_as(std::shared_ptr<TensorDimExpr const>) const override;
};

std::shared_ptr<TensorDimAdd const>
    dim_expr_make_add(std::shared_ptr<TensorDimExpr const> lhs,
                      std::shared_ptr<TensorDimExpr const> rhs);

class TensorDimMul : public TensorDimExpr {
public:
  TensorDimMul(std::shared_ptr<TensorDimExpr const> lhs,
               std::shared_ptr<TensorDimExpr const> rhs);
  std::shared_ptr<TensorDimExpr const> lhs, rhs;

  int get_value(DimVarAssignments const &assignments) const override;
  z3::expr to_z3(z3::context &c,
                 DimVarAssignments const &assign,
                 bool log_scaled) const override;
  std::string to_string() const override;
  bool is_mul() const override;

  size_t hash() const override;
  operator json() const override;
  bool same_expr_as(std::shared_ptr<TensorDimExpr const>) const override;
};

std::shared_ptr<TensorDimMul const>
    dim_expr_make_mul(std::shared_ptr<TensorDimExpr const> lhs,
                      std::shared_ptr<TensorDimExpr const> rhs);

class TensorDimDiv : public TensorDimExpr {
public:
  TensorDimDiv(std::shared_ptr<TensorDimExpr const> lhs,
               std::shared_ptr<TensorDimExpr const> rhs);
  std::shared_ptr<TensorDimExpr const> lhs, rhs;

  int get_value(DimVarAssignments const &assignments) const override;
  z3::expr to_z3(z3::context &c,
                 DimVarAssignments const &assign,
                 bool log_scaled) const override;
  std::string to_string() const override;
  bool is_div() const override;

  size_t hash() const override;
  operator json() const override;
  bool same_expr_as(std::shared_ptr<TensorDimExpr const>) const override;
};

std::shared_ptr<TensorDimDiv const>
    dim_expr_make_div(std::shared_ptr<TensorDimExpr const> lhs,
                      std::shared_ptr<TensorDimExpr const> rhs);

// NOTE: only supported in log-scaled expr
class TensorDimPow : public TensorDimExpr {
public:
  TensorDimPow(std::shared_ptr<TensorDimExpr const> base,
               std::shared_ptr<TensorDimExpr const> exp);
  std::shared_ptr<TensorDimExpr const> base, exp;

  int get_value(DimVarAssignments const &assignments) const override;
  z3::expr to_z3(z3::context &c,
                 DimVarAssignments const &assign,
                 bool log_scaled) const override;
  std::string to_string() const override;
  bool is_pow() const override;
  size_t hash() const override;
  operator json() const override;
  bool same_expr_as(std::shared_ptr<TensorDimExpr const>) const override;
};

std::shared_ptr<TensorDimPow const>
    dim_expr_make_pow(std::shared_ptr<TensorDimExpr const> base,
                      std::shared_ptr<TensorDimExpr const> exp);

class TensorDimIte : public TensorDimExpr {
public:
  TensorDimIte(std::shared_ptr<TensorDimExpr const> cond,
               std::shared_ptr<TensorDimExpr const> true_case,
               std::shared_ptr<TensorDimExpr const> false_case);
  std::shared_ptr<TensorDimExpr const> cond, true_case, false_case;
  int get_value(DimVarAssignments const &assignments) const override;
  z3::expr to_z3(z3::context &c,
                 DimVarAssignments const &assign,
                 bool log_scaled) const override;
  std::string to_string() const override;
  bool is_ite() const override;
  size_t hash() const override;
  operator json() const override;
  bool same_expr_as(std::shared_ptr<TensorDimExpr const>) const override;
};

std::shared_ptr<TensorDimIte const>
    dim_expr_make_ite(std::shared_ptr<TensorDimExpr const> cond,
                      std::shared_ptr<TensorDimExpr const> true_case,
                      std::shared_ptr<TensorDimExpr const> false_case);

} // namespace search
} // namespace mirage
