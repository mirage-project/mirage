#include "mirage/search/abstract_expr/abstract_expr.h"
#include <atomic>
#include <cassert>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

#include "mirage/utils/containers.h"
#include "mirage/utils/z3_utils.h"
#include "mirage/search/symbolic_graph/tensor_dim_expr.h"

namespace mirage {
namespace search {

bool AbstractExpr::subexpr_to(AbstractExpr const &other, std::unordered_set<TensorDimConstraint> const &constraints) const {
  // Z3 context
  z3::context c;

  // Define the sorts
  z3::sort P = c.uninterpreted_sort("P");
  z3::sort I = c.int_sort();

  // Define the functions
  z3::func_decl add = c.function("add", P, P, P);
  z3::func_decl mul = c.function("mul", P, P, P);
  z3::func_decl div = c.function("div", P, P, P);
  z3::func_decl exp = c.function("exp", P, P);
  z3::func_decl silu = c.function("silu", P, P);
  z3::func_decl rms = c.function("rms", I, P, P);
  z3::func_decl red = c.function("red", I, P, P);

  // The subexpression relation
  z3::func_decl subexpr = c.function("subexpr", P, P, c.bool_sort());

  // Z3 solver
  z3::solver s(c);

  // Set solver parameters
  z3::params p(c);
  p.set("mbqi", true);
  p.set("rlimit", 3000000u);
  // p.set("timeout", 10u);
  s.set(p);

  // Define the variables used in the axioms
  z3::expr x = c.constant("x", P);
  z3::expr y = c.constant("y", P);
  z3::expr z = c.constant("z", P);
  z3::expr i = c.int_const("i");
  z3::expr i1 = c.int_const("i1");
  z3::expr i2 = c.int_const("i2");

  z3::expr expr1 = to_z3(c), expr2 = other.to_z3(c);

  // All inputs are different
  {
    z3::expr_vector input_consts = get_free_vars({expr1, expr2}, P);
    for (size_t i = 0; i < input_consts.size(); ++i) {
      for (size_t j = i + 1; j < input_consts.size(); ++j) {
        s.add(input_consts[i] != input_consts[j]);
        s.add(!subexpr(input_consts[i], input_consts[j]));
        s.add(!subexpr(input_consts[j], input_consts[i]));
      }
    }
  }

  // Subexpression axioms
  s.add(forall(x, subexpr(x, x)));
  s.add(
      forall(x,
             y,
             z,
             implies(subexpr(x, y) && subexpr(y, z), subexpr(x, z))));

  s.add(forall(x, y, subexpr(x, add(x, y))));
  s.add(forall(x, y, subexpr(x, mul(x, y))));
  s.add(forall(x, y, subexpr(x, div(x, y))));
  s.add(forall(x, y, subexpr(y, div(x, y))));
  s.add(forall(x, subexpr(x, exp(x))));
  s.add(forall(x, subexpr(x, silu(x))));
  s.add(forall(x, i, subexpr(x, rms(i, x))));
  s.add(forall(x, i, subexpr(x, red(i, x))));

  // Equivalence axioms
  s.add(forall(x, y, add(x, y) == add(y, x)));
  s.add(forall(x, y, mul(x, y) == mul(y, x)));
  s.add(forall(x, y, z, add(add(x, y), z) == add(x, add(y, z))));
  s.add(forall(x, y, z, mul(mul(x, y), z) == mul(x, mul(y, z))));
  s.add(forall(x, y, z, add(mul(x, z), mul(y, z)) == mul(add(x, y), z)));
  s.add(forall(x, y, z, add(div(x, z), div(y, z)) == div(add(x, y), z)));
  s.add(forall(x, y, z, mul(x, div(y, z)) == div(mul(x, y), z)));

  s.add(forall(x, x == red(c.int_val(0), x)));
  s.add(forall(x, i1, i2, red(i1, red(i2, x)) == red(i1 + i2, x)));
  s.add(forall(to_expr_vector({x, y, i, i1, i2}),
               red(i, add(red(i1, x), red(i2, y))) ==
                   add(red(i + i1, x), red(i + i2, y))));
  s.add(forall(x, y, i, red(i, mul(x, y)) == mul(red(i, x), y)));
  s.add(forall(x, y, i, red(i, div(x, y)) == div(red(i, x), y)));

  // Lemmas
  s.add(
      forall(x, i1, i2, implies(i1 <= i2, subexpr(red(i1, x), red(i2, x)))));

  // Theorem to prove
  {
    z3::expr dim_constraints = c.bool_val(true);

    // Add the constraints from tensor dimensions
    for (TensorDimConstraint const &constraint : constraints) {
      dim_constraints = dim_constraints && constraint.to_z3(c);
    }

    z3::expr_vector dim_vars = get_free_vars({dim_constraints, expr1, expr2}, I);

    for (z3::expr const &dim_var : dim_vars) {
      // NOTE: assume log-scaled
      s.add(dim_var >= 0);
    }

    if (dim_vars.size() > 0) {
      s.add(!z3::exists(dim_vars, dim_constraints && subexpr(expr1, expr2)));
    } else {
      s.add(!(dim_constraints && subexpr(expr1, expr2)));
    }
    // s.add(!z3::exists(dim_vars, dim_constraints && subexpr(expr1, expr2)));
    // s.add(z3::forall(dim_vars, !(dim_constraints && subexpr(expr1, expr2))));
  }

  bool result = s.check() == z3::unsat;
  return result;
}

bool AbstractExpr::operator==(AbstractExpr const &other) const {
  return subexpr_to(other) && other.subexpr_to(*this);
}

Var::Var(std::string const &name) : name(name) {}

z3::expr Var::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  return c.constant(name.data(), P);
}

std::string Var::to_string() const {
  return name;
}

Add::Add(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs)
    : lhs(lhs), rhs(rhs) {
  assert(lhs);
  assert(rhs);
}

z3::expr Add::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl add = c.function("add", P, P, P);
  return add(lhs->to_z3(c), rhs->to_z3(c));
}

std::string Add::to_string() const {
  return "(" + lhs->to_string() + "+" + rhs->to_string() + ")";
}

Mul::Mul(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs)
    : lhs(lhs), rhs(rhs) {
  assert(lhs);
  assert(rhs);
}

z3::expr Mul::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl mul = c.function("mul", P, P, P);
  return mul(lhs->to_z3(c), rhs->to_z3(c));
}

std::string Mul::to_string() const {
  return "(" + lhs->to_string() + rhs->to_string() + ")";
}

Div::Div(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs)
    : lhs(lhs), rhs(rhs) {
  assert(lhs);
  assert(rhs);
}

z3::expr Div::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl div = c.function("div", P, P, P);
  return div(lhs->to_z3(c), rhs->to_z3(c));
}

std::string Div::to_string() const {
  return "(" + lhs->to_string() + "/" + rhs->to_string() + ")";
}

Exp::Exp(std::shared_ptr<AbstractExpr> exponent) : exponent(exponent) {
  assert(exponent);
}

z3::expr Exp::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl exp = c.function("exp", P, P);
  return exp(exponent->to_z3(c));
}

std::string Exp::to_string() const {
  return "e^" + exponent->to_string();
}

Silu::Silu(std::shared_ptr<AbstractExpr> a) : a(a) {
  assert(a);
}

z3::expr Silu::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl silu = c.function("silu", P, P);
  return silu(a->to_z3(c));
}

std::string Silu::to_string() const {
  return "silu(" + a->to_string() + ")";
}

RMS::RMS(std::shared_ptr<TensorDimExpr> reduction_degree, std::shared_ptr<AbstractExpr> elems)
    : reduction_degree(reduction_degree), elems(elems) {
  assert(elems);
}

z3::expr RMS::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl rms = c.function("rms", c.int_sort(), P, P);
  return rms(reduction_degree->to_z3(c), elems->to_z3(c));
}

std::string RMS::to_string() const {
  return "rms(" + reduction_degree->to_string() + ", " + elems->to_string() + ")";
}

Red::Red(std::shared_ptr<TensorDimExpr> reduction_degree, std::shared_ptr<AbstractExpr> summand)
    : reduction_degree(reduction_degree), summand(summand) {
  assert(summand);
}

z3::expr Red::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl red = c.function("red", c.int_sort(), P, P);
  return red(reduction_degree->to_z3(c), summand->to_z3(c));
}

std::string Red::to_string() const {
  return "r(" + reduction_degree->to_string() + ", " + summand->to_string() + ")";
}

std::shared_ptr<AbstractExpr> abstract_expr_make_var(std::string const &name) {
  return std::make_shared<Var>(name);
}

std::shared_ptr<AbstractExpr>
    abstract_expr_make_add(std::shared_ptr<AbstractExpr> lhs,
                           std::shared_ptr<AbstractExpr> rhs) {
  return std::make_shared<Add>(lhs, rhs);
}

std::shared_ptr<AbstractExpr>
    abstract_expr_make_mul(std::shared_ptr<AbstractExpr> lhs,
                           std::shared_ptr<AbstractExpr> rhs) {
  return std::make_shared<Mul>(lhs, rhs);
}

std::shared_ptr<AbstractExpr>
    abstract_expr_make_div(std::shared_ptr<AbstractExpr> lhs,
                           std::shared_ptr<AbstractExpr> rhs) {
  return std::make_shared<Div>(lhs, rhs);
}

std::shared_ptr<AbstractExpr>
    abstract_expr_make_exp(std::shared_ptr<AbstractExpr> exponent) {
  return std::make_shared<Exp>(exponent);
}

std::shared_ptr<AbstractExpr>
    abstract_expr_make_silu(std::shared_ptr<AbstractExpr> a) {
  return std::make_shared<Silu>(a);
}
std::shared_ptr<AbstractExpr>
    abstract_expr_make_rms(int reduction_degree, std::shared_ptr<AbstractExpr> elems) {
  return abstract_expr_make_rms(dim_expr_make_const(reduction_degree), elems);
}

std::shared_ptr<AbstractExpr>
    abstract_expr_make_rms(std::shared_ptr<TensorDimExpr> reduction_degree, std::shared_ptr<AbstractExpr> elems) {
  return std::make_shared<RMS>(reduction_degree, elems);
}

std::shared_ptr<AbstractExpr>
    abstract_expr_make_rms(SymbolicTensorDim const &reduction_dim, std::shared_ptr<AbstractExpr> elems) {
  return abstract_expr_make_rms(reduction_dim.dim_expr, elems);
}

std::shared_ptr<AbstractExpr>
    abstract_expr_make_red(int reduction_degree, std::shared_ptr<AbstractExpr> summand) {
  return abstract_expr_make_red(dim_expr_make_const(reduction_degree), summand);
}

std::shared_ptr<AbstractExpr>
    abstract_expr_make_red(std::shared_ptr<TensorDimExpr> reduction_degree, std::shared_ptr<AbstractExpr> summand) {
  return std::make_shared<Red>(reduction_degree, summand);
}
std::shared_ptr<AbstractExpr>
    abstract_expr_make_red(SymbolicTensorDim const &reduction_dim, std::shared_ptr<AbstractExpr> summand) {
  return abstract_expr_make_red(reduction_dim.dim_expr, summand);
}

std::shared_ptr<AbstractExpr> Var::simplify() const {
  return abstract_expr_make_var(name);
}

std::shared_ptr<AbstractExpr> Add::simplify() const {
  return abstract_expr_make_add(lhs->simplify(), rhs->simplify());
}

std::shared_ptr<AbstractExpr> Mul::simplify() const {
  return abstract_expr_make_mul(lhs->simplify(), rhs->simplify());
}

std::shared_ptr<AbstractExpr> Div::simplify() const {
  std::shared_ptr<AbstractExpr> lhs_simplified = lhs->simplify(), rhs_simplified = rhs->simplify();
  if (std::dynamic_pointer_cast<Div>(lhs_simplified)) {
    std::shared_ptr<Div> div = std::static_pointer_cast<Div>(lhs_simplified);
    return abstract_expr_make_div(div->lhs, abstract_expr_make_mul(div->rhs, rhs_simplified));
  }
  if (std::dynamic_pointer_cast<Div>(rhs_simplified)) {
    std::shared_ptr<Div> div = std::static_pointer_cast<Div>(rhs_simplified);
    return abstract_expr_make_div(abstract_expr_make_mul(lhs_simplified, div->rhs), div->lhs);
  }
  return abstract_expr_make_div(lhs_simplified, rhs_simplified);
}

std::shared_ptr<AbstractExpr> Exp::simplify() const {
  return abstract_expr_make_exp(exponent->simplify());
}

std::shared_ptr<AbstractExpr> Silu::simplify() const {
  return abstract_expr_make_silu(a->simplify());
}

std::shared_ptr<AbstractExpr> RMS::simplify() const {
  return abstract_expr_make_rms(reduction_degree, elems->simplify());
}

std::shared_ptr<AbstractExpr> Red::simplify() const {
  std::shared_ptr<AbstractExpr> summand_simplified = summand->simplify();
  if (std::dynamic_pointer_cast<Red>(summand_simplified)) {
    std::shared_ptr<Red> red = std::static_pointer_cast<Red>(summand_simplified);
    return abstract_expr_make_red(dim_expr_make_mul(reduction_degree, red->reduction_degree), red->summand);
  }
  return abstract_expr_make_red(reduction_degree, summand_simplified);
}

} // namespace search
} // namespace mirage
