#pragma once

#include "mirage/search/symbolic_graph/dim_var_assignments.h"
#include "mirage/search/symbolic_graph/tensor_dim_constraint.h"
#include "mirage/utils/hash_utils.h"
#include "mirage/utils/json_utils.h"
#include "z3++.h"
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

namespace mirage {
namespace search {

class AbstractExpr {
public:
  AbstractExpr() = default;
  virtual ~AbstractExpr() = default;

  virtual z3::expr to_z3(z3::context &c,
                         DimVarAssignments const &assign) const = 0;
  bool subexpr_to(
      AbstractExpr const &other,
      std::vector<DimVarAssignments> const &possible_assignments = {},
      std::unordered_set<TensorDimConstraint> const &constraints = {}) const;
  bool operator==(AbstractExpr const &other) const;
  virtual std::shared_ptr<AbstractExpr const> simplify() const = 0;
  virtual std::string to_string() const = 0;
};

class Var : public AbstractExpr {
public:
  Var(std::string const &name);
  z3::expr to_z3(z3::context &c,
                 DimVarAssignments const &assign) const override;
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr const> simplify() const override;
  std::string name;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_var(std::string const &name);

class Add : public AbstractExpr {
public:
  Add(std::shared_ptr<AbstractExpr const> lhs,
      std::shared_ptr<AbstractExpr const> rhs);
  z3::expr to_z3(z3::context &c,
                 DimVarAssignments const &assign) const override;
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr const> simplify() const override;
  std::shared_ptr<AbstractExpr const> lhs, rhs;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_add(std::shared_ptr<AbstractExpr const> lhs,
                           std::shared_ptr<AbstractExpr const> rhs);

class Mul : public AbstractExpr {
public:
  Mul(std::shared_ptr<AbstractExpr const> lhs,
      std::shared_ptr<AbstractExpr const> rhs);
  z3::expr to_z3(z3::context &c,
                 DimVarAssignments const &assign) const override;
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr const> simplify() const override;
  std::shared_ptr<AbstractExpr const> lhs, rhs;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_mul(std::shared_ptr<AbstractExpr const> lhs,
                           std::shared_ptr<AbstractExpr const> rhs);

class Div : public AbstractExpr {
public:
  Div(std::shared_ptr<AbstractExpr const> lhs,
      std::shared_ptr<AbstractExpr const> rhs);
  z3::expr to_z3(z3::context &c,
                 DimVarAssignments const &assign) const override;
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr const> simplify() const override;
  std::shared_ptr<AbstractExpr const> lhs, rhs;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_div(std::shared_ptr<AbstractExpr const> lhs,
                           std::shared_ptr<AbstractExpr const> rhs);

class Exp : public AbstractExpr {
public:
  Exp(std::shared_ptr<AbstractExpr const> exponent);
  z3::expr to_z3(z3::context &c,
                 DimVarAssignments const &assign) const override;
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr const> simplify() const override;
  std::shared_ptr<AbstractExpr const> exponent;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_exp(std::shared_ptr<AbstractExpr const> exponent);

class Silu : public AbstractExpr {
public:
  Silu(std::shared_ptr<AbstractExpr const> a);
  z3::expr to_z3(z3::context &c,
                 DimVarAssignments const &assign) const override;
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr const> simplify() const override;
  std::shared_ptr<AbstractExpr const> a;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_silu(std::shared_ptr<AbstractExpr const> a);

// Note(@Mengdi): Replace it with Sqr and Sqrt once we have related algebraic
// transformation
class RMS : public AbstractExpr {
public:
  RMS(std::shared_ptr<TensorDimExpr const> reduction_degree,
      std::shared_ptr<AbstractExpr const> elems);
  z3::expr to_z3(z3::context &c,
                 DimVarAssignments const &assign) const override;
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr const> simplify() const override;
  std::shared_ptr<TensorDimExpr const> reduction_degree;
  std::shared_ptr<AbstractExpr const> elems;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_rms(int reduction_degree,
                           std::shared_ptr<AbstractExpr const> elems);
std::shared_ptr<AbstractExpr const> abstract_expr_make_rms(
    std::shared_ptr<TensorDimExpr const> reduction_degree,
    std::shared_ptr<AbstractExpr const> elems);
std::shared_ptr<AbstractExpr const>
    abstract_expr_make_rms(SymbolicTensorDim const &reduction_dim,
                           std::shared_ptr<AbstractExpr const> elems);

class Red : public AbstractExpr {
public:
  Red(std::shared_ptr<TensorDimExpr const> reduction_degree,
      std::shared_ptr<AbstractExpr const> summand);
  z3::expr to_z3(z3::context &c,
                 DimVarAssignments const &assign) const override;
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr const> simplify() const override;
  std::shared_ptr<TensorDimExpr const> reduction_degree;
  std::shared_ptr<AbstractExpr const> summand;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_red(int reduction_degree,
                           std::shared_ptr<AbstractExpr const> summand);

std::shared_ptr<AbstractExpr const> abstract_expr_make_red(
    std::shared_ptr<TensorDimExpr const> reduction_degree,
    std::shared_ptr<AbstractExpr const> summand);

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_red(SymbolicTensorDim const &reduction_dim,
                           std::shared_ptr<AbstractExpr const> summand);

} // namespace search
} // namespace mirage