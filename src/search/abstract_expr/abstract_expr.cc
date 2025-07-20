#include "mirage/search/abstract_expr/abstract_expr.h"
#include <atomic>
#include <cassert>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

#include "mirage/search/symbolic_graph/tensor_dim_expr.h"
#include "mirage/utils/containers.h"
#include "mirage/utils/z3_utils.h"

namespace mirage {
namespace search {

void initialize_final_expr(std::shared_ptr<AbstractExpr const> expr) {
  get_egraph(expr->to_egg().c_str());
}

bool subexpr_to_final_expr(std::shared_ptr<AbstractExpr const> expr) {
  return subexpr_to_final_expr(
      std::vector<std::shared_ptr<AbstractExpr const>>{expr})[0];
}

std::vector<bool> subexpr_to_final_expr(
    std::vector<std::shared_ptr<AbstractExpr const>> const &exprs) {
  std::vector<std::string> exprs_str =
      vector_map(exprs, [](std::shared_ptr<AbstractExpr const> const &expr) {
        return expr->to_egg();
      });

  char const **exprs_c_str = new char const *[exprs.size()];
  for (size_t i = 0; i < exprs.size(); ++i) {
    assert(exprs[i] != nullptr);
    exprs_c_str[i] = exprs_str[i].c_str();
  }

  bool *results_in_raw_array =
      egg_equiv(exprs_c_str, static_cast<int>(exprs.size()));

  std::vector<bool> results;
  for (size_t i = 0; i < exprs.size(); ++i) {
    results.push_back(results_in_raw_array[i]);
  }

  delete[] exprs_c_str;

  return results;
}

Var::Var(std::string const &name) : name(name) {}

std::string Var::to_string() const {
  return name;
}

std::string Var::to_egg() const {
  return name;
}

Add::Add(std::shared_ptr<AbstractExpr const> lhs,
         std::shared_ptr<AbstractExpr const> rhs)
    : lhs(lhs), rhs(rhs) {
  assert(lhs);
  assert(rhs);
}

std::string Add::to_string() const {
  return "(" + lhs->to_string() + "+" + rhs->to_string() + ")";
}

std::string Add::to_egg() const {
  return "(+ " + lhs->to_egg() + " " + rhs->to_egg() + ")";
}

Mul::Mul(std::shared_ptr<AbstractExpr const> lhs,
         std::shared_ptr<AbstractExpr const> rhs)
    : lhs(lhs), rhs(rhs) {
  assert(lhs);
  assert(rhs);
}

std::string Mul::to_string() const {
  return "(" + lhs->to_string() + rhs->to_string() + ")";
}

std::string Mul::to_egg() const {
  return "(* " + lhs->to_egg() + " " + rhs->to_egg() + ")";
}

Div::Div(std::shared_ptr<AbstractExpr const> lhs,
         std::shared_ptr<AbstractExpr const> rhs)
    : lhs(lhs), rhs(rhs) {
  assert(lhs);
  assert(rhs);
}

std::string Div::to_string() const {
  return "(" + lhs->to_string() + "/" + rhs->to_string() + ")";
}

std::string Div::to_egg() const {
  return "(/ " + lhs->to_egg() + " " + rhs->to_egg() + ")";
}

Pow::Pow(std::shared_ptr<AbstractExpr const> lhs,
         std::shared_ptr<AbstractExpr const> rhs)
    : lhs(lhs), rhs(rhs) {
  assert(lhs);
  assert(rhs);
}

std::string Pow::to_string() const {
  return "(" + lhs->to_string() + "^" + rhs->to_string() + ")";
}

std::string Pow::to_egg() const {
  return "(pow " + lhs->to_egg() + " " + rhs->to_egg() + ")";
}

Exp::Exp(std::shared_ptr<AbstractExpr const> exponent) : exponent(exponent) {
  assert(exponent);
}

std::string Exp::to_string() const {
  return "e^" + exponent->to_string();
}

std::string Exp::to_egg() const {
  return "(exp " + exponent->to_egg() + ")";
}

Square::Square(std::shared_ptr<AbstractExpr const> a) : a(a) {
  assert(a);
}

std::string Square::to_string() const {
  return "square(" + a->to_string() + ")";
}

std::string Square::to_egg() const {
  return "(square " + a->to_egg() + ")";
}

Sqrt::Sqrt(std::shared_ptr<AbstractExpr const> a) : a(a) {
  assert(a);
}

std::string Sqrt::to_string() const {
  return "sqrt(" + a->to_string() + ")";
}

std::string Sqrt::to_egg() const {
  return "(sqrt " + a->to_egg() + ")";
}

Silu::Silu(std::shared_ptr<AbstractExpr const> a) : a(a) {
  assert(a);
}

std::string Silu::to_string() const {
  return "silu(" + a->to_string() + ")";
}

std::string Silu::to_egg() const {
  return "(silu " + a->to_egg() + ")";
}

Gelu::Gelu(std::shared_ptr<AbstractExpr const> a) : a(a) {
  assert(a);
}

std::string Gelu::to_string() const {
  return "gelu(" + a->to_string() + ")";
}

std::string Gelu::to_egg() const {
  return "(gelu " + a->to_egg() + ")";
}

Relu::Relu(std::shared_ptr<AbstractExpr const> a) : a(a) {
  assert(a);
}

std::string Relu::to_string() const {
  return "relu(" + a->to_string() + ")";
}

std::string Relu::to_egg() const {
  return "(relu " + a->to_egg() + ")";
}

Clamp::Clamp(float min_val,
             float max_val,
             std::shared_ptr<AbstractExpr const> elems)
    : min_val(min_val), max_val(max_val), elems(elems) {
  assert(elems);
}

std::string Clamp::to_string() const {
  return "clamp(" + std::to_string(min_val) +
         " <= x <= " + std::to_string(max_val) + ", " + elems->to_string() +
         ")";
}

std::string Clamp::to_egg() const {
  return "(clamp " + std::to_string(min_val) + " " + std::to_string(max_val) +
         " " + elems->to_egg() + ")";
}

RMS::RMS(std::shared_ptr<TensorDimExpr const> reduction_degree,
         std::shared_ptr<AbstractExpr const> elems)
    : reduction_degree(reduction_degree), elems(elems) {
  assert(elems);
}

std::string RMS::to_string() const {
  return "rms(" + reduction_degree->to_string() + ", " + elems->to_string() +
         ")";
}

std::string RMS::to_egg() const {
  return "(rms " + reduction_degree->to_string() + " " + elems->to_egg() + ")";
}

Red::Red(std::shared_ptr<TensorDimExpr const> reduction_degree,
         std::shared_ptr<AbstractExpr const> summand)
    : reduction_degree(reduction_degree), summand(summand) {
  assert(reduction_degree);
  assert(summand);
}

std::string Red::to_string() const {
  return "r(" + reduction_degree->to_string() + ", " + summand->to_string() +
         ")";
}

std::string Red::to_egg() const {
  return "(sum " + reduction_degree->to_string() + " " + summand->to_egg() +
         ")";
}

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_var(std::string const &name) {
  return std::make_shared<Var>(name);
}

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_add(std::shared_ptr<AbstractExpr const> lhs,
                           std::shared_ptr<AbstractExpr const> rhs) {
  return std::make_shared<Add>(lhs, rhs);
}

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_mul(std::shared_ptr<AbstractExpr const> lhs,
                           std::shared_ptr<AbstractExpr const> rhs) {
  return std::make_shared<Mul>(lhs, rhs);
}

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_div(std::shared_ptr<AbstractExpr const> lhs,
                           std::shared_ptr<AbstractExpr const> rhs) {
  return std::make_shared<Div>(lhs, rhs);
}

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_pow(std::shared_ptr<AbstractExpr const> lhs,
                           std::shared_ptr<AbstractExpr const> rhs) {
  return std::make_shared<Pow>(lhs, rhs);
}

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_exp(std::shared_ptr<AbstractExpr const> exponent) {
  return std::make_shared<Exp>(exponent);
}

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_square(std::shared_ptr<AbstractExpr const> a) {
  return std::make_shared<Square>(a);
}

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_sqrt(std::shared_ptr<AbstractExpr const> a) {
  return std::make_shared<Sqrt>(a);
}

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_silu(std::shared_ptr<AbstractExpr const> a) {
  return std::make_shared<Silu>(a);
}

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_gelu(std::shared_ptr<AbstractExpr const> a) {
  return std::make_shared<Gelu>(a);
}

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_relu(std::shared_ptr<AbstractExpr const> a) {
  return std::make_shared<Relu>(a);
}

std::shared_ptr<AbstractExpr const> abstract_expr_make_clamp(
    float min_val, float max_val, std::shared_ptr<AbstractExpr const> elems) {
  return std::make_shared<Clamp>(min_val, max_val, elems);
}

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_rms(int reduction_degree,
                           std::shared_ptr<AbstractExpr const> elems) {
  return abstract_expr_make_rms(dim_expr_make_const(reduction_degree), elems);
}

std::shared_ptr<AbstractExpr const> abstract_expr_make_rms(
    std::shared_ptr<TensorDimExpr const> reduction_degree,
    std::shared_ptr<AbstractExpr const> elems) {
  return std::make_shared<RMS>(reduction_degree, elems);
}

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_rms(SymbolicTensorDim const &reduction_dim,
                           std::shared_ptr<AbstractExpr const> elems) {
  return abstract_expr_make_rms(reduction_dim.dim_expr, elems);
}

std::shared_ptr<AbstractExpr const>
    abstract_expr_make_red(int reduction_degree,
                           std::shared_ptr<AbstractExpr const> summand) {
  return abstract_expr_make_red(dim_expr_make_const(reduction_degree), summand);
}

std::shared_ptr<AbstractExpr const> abstract_expr_make_red(
    std::shared_ptr<TensorDimExpr const> reduction_degree,
    std::shared_ptr<AbstractExpr const> summand) {
  return std::make_shared<Red>(reduction_degree, summand);
}
std::shared_ptr<AbstractExpr const>
    abstract_expr_make_red(SymbolicTensorDim const &reduction_dim,
                           std::shared_ptr<AbstractExpr const> summand) {
  return abstract_expr_make_red(reduction_dim.dim_expr, summand);
}

} // namespace search
} // namespace mirage