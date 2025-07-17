#pragma once

#include "mirage/search/symbolic_graph/dim_var_assignments.h"
#include "mirage/search/symbolic_graph/tensor_dim_constraints.h"
#include "mirage/utils/hash_utils.h"
#include "mirage/utils/json_utils.h"
#include "z3++.h"
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>

extern "C" {

struct KVPair {
  int key;
  bool value;
};

void get_egraph(char const *expr);
bool *egg_equiv(char const **inputs, int len);
}

namespace mirage {
namespace search {

class AbstractExpr {
public:
  AbstractExpr() = default;
  virtual ~AbstractExpr() = default;

  virtual std::string to_string() const = 0;
  virtual std::string to_egg() const = 0;
};

void initialize_final_expr(std::shared_ptr<AbstractExpr const> expr);
bool subexpr_to_final_expr(std::shared_ptr<AbstractExpr const> expr);
std::vector<bool> subexpr_to_final_expr(
    std::vector<std::shared_ptr<AbstractExpr const>> const &exprs);

class Var : public AbstractExpr {
public:
  Var(std::string const &name);
  std::string to_string() const override;
  std::string to_egg() const override;
  std::string name;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_var(std::string const &name);

class Add : public AbstractExpr {
public:
  Add(std::shared_ptr<AbstractExpr const> lhs,
      std::shared_ptr<AbstractExpr const> rhs);
  std::string to_egg() const override;
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr const> lhs, rhs;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_add(std::shared_ptr<AbstractExpr const> lhs,
                           std::shared_ptr<AbstractExpr const> rhs);

class Mul : public AbstractExpr {
public:
  Mul(std::shared_ptr<AbstractExpr const> lhs,
      std::shared_ptr<AbstractExpr const> rhs);
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr const> lhs, rhs;
  std::string to_egg() const override;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_mul(std::shared_ptr<AbstractExpr const> lhs,
                           std::shared_ptr<AbstractExpr const> rhs);

class Div : public AbstractExpr {
public:
  Div(std::shared_ptr<AbstractExpr const> lhs,
      std::shared_ptr<AbstractExpr const> rhs);
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr const> lhs, rhs;
  std::string to_egg() const override;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_div(std::shared_ptr<AbstractExpr const> lhs,
                           std::shared_ptr<AbstractExpr const> rhs);

class Pow : public AbstractExpr {
public:
  Pow(std::shared_ptr<AbstractExpr const> lhs,
      std::shared_ptr<AbstractExpr const> rhs);
  std::string to_string() const override;
  std::string to_egg() const override;
  std::shared_ptr<AbstractExpr const> lhs, rhs;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_pow(std::shared_ptr<AbstractExpr const> lhs,
                           std::shared_ptr<AbstractExpr const> rhs);

class Exp : public AbstractExpr {
public:
  Exp(std::shared_ptr<AbstractExpr const> exponent);
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr const> exponent;
  std::string to_egg() const override;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_exp(std::shared_ptr<AbstractExpr const> exponent);

class Square : public AbstractExpr {
public:
  Square(std::shared_ptr<AbstractExpr const> a);
  std::string to_string() const override;
  std::string to_egg() const override;
  std::shared_ptr<AbstractExpr const> a;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_square(std::shared_ptr<AbstractExpr const> a);

class Sqrt : public AbstractExpr {
public:
  Sqrt(std::shared_ptr<AbstractExpr const> a);
  std::string to_string() const override;
  std::string to_egg() const override;
  std::shared_ptr<AbstractExpr const> a;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_sqrt(std::shared_ptr<AbstractExpr const> a);

class Silu : public AbstractExpr {
public:
  Silu(std::shared_ptr<AbstractExpr const> a);
  std::string to_string() const override;
  std::shared_ptr<AbstractExpr const> a;
  std::string to_egg() const override;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_silu(std::shared_ptr<AbstractExpr const> a);

class Gelu : public AbstractExpr {
public:
  Gelu(std::shared_ptr<AbstractExpr const> a);
  std::string to_string() const override;
  std::string to_egg() const override;
  std::shared_ptr<AbstractExpr const> a;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_gelu(std::shared_ptr<AbstractExpr const> a);

class Relu : public AbstractExpr {
public:
  Relu(std::shared_ptr<AbstractExpr const> a);
  std::string to_string() const override;
  std::string to_egg() const override;
  std::shared_ptr<AbstractExpr const> a;
};

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_relu(std::shared_ptr<AbstractExpr const> a);

class Clamp : public AbstractExpr {
public:
  Clamp(float min_val,
        float max_val,
        std::shared_ptr<AbstractExpr const> elems);
  std::string to_string() const override;
  std::string to_egg() const override;
  float min_val;
  float max_val;
  std::shared_ptr<AbstractExpr const> elems;
};

std::shared_ptr<AbstractExpr const> abstract_expr_make_clamp(
    float min_val, float max_val, std::shared_ptr<AbstractExpr const> elems);

// Note(@Mengdi): Replace it with Sqr and Sqrt once we have related algebraic
// transformation
class RMS : public AbstractExpr {
public:
  RMS(std::shared_ptr<TensorDimExpr const> reduction_degree,
      std::shared_ptr<AbstractExpr const> elems);
  std::string to_string() const override;
  std::string to_egg() const override;
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
  std::string to_string() const override;
  std::string to_egg() const override;
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