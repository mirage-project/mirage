#include "mirage/search/abstract_expr/abstract_expr.h"
#include <atomic>
#include <cassert>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

#include "mirage/utils/containers.h"

namespace mirage {
namespace search {

z3::expr_vector to_expr_vector(z3::context &c,
                               std::vector<z3::expr> const &_vec) {
  z3::expr_vector vec(c);
  for (auto const &e : _vec) {
    vec.push_back(e);
  }
  return vec;
}

bool AbstractExpr::subpattern_to(AbstractExpr const &other) const {
  z3::context c;

  z3::sort P = c.uninterpreted_sort("P");
  z3::sort I = c.int_sort();
  z3::sort F = c.fpa_sort(8, 24);

  z3::func_decl add = c.function("add", P, P, P);
  z3::func_decl mul = c.function("mul", P, P, P);
  z3::func_decl div = c.function("div", P, P, P);
  z3::func_decl exp = c.function("exp", P, P);
  z3::func_decl silu = c.function("silu", P, P);
  z3::func_decl gelu = c.function("gelu", P, P);
  z3::func_decl relu = c.function("relu", P, P);
  z3::func_decl clamp = c.function("clamp", F, F, P, P);
  z3::func_decl rms = c.function("rms", I, P, P);
  z3::func_decl red = c.function("red", I, P, P);

  z3::func_decl subpattern = c.function("subpattern", P, P, c.bool_sort());

  z3::solver s(c);

  z3::params p(c);
  p.set("mbqi", true);
  p.set("rlimit", 80000u);
  // p.set("timeout", 10u);
  s.set(p);

  z3::expr x = c.constant("x", P);
  z3::expr y = c.constant("y", P);
  z3::expr z = c.constant("z", P);
  z3::expr i = c.int_const("i");
  z3::expr i1 = c.int_const("i1");
  z3::expr i2 = c.int_const("i2");
  z3::expr f = c.fpa_const("f", 8, 24);
  z3::expr f1 = c.fpa_const("f", 8, 24);

  std::unordered_set<std::string> all_variables;
  z3::expr pattern1 = to_z3(c, all_variables),
           pattern2 = other.to_z3(c, all_variables);

  for (std::string const &name1 : all_variables) {
    for (std::string const &name2 : all_variables) {
      if (name1 < name2) {
        z3::expr v1 = c.constant(name1.data(), P);
        z3::expr v2 = c.constant(name2.data(), P);
        s.add(v1 != v2);
        s.add(!subpattern(v1, v2));
        s.add(!subpattern(v2, v1));
      }
    }
  }

  s.add(forall(x, subpattern(x, x)));
  s.add(
      forall(x,
             y,
             z,
             implies(subpattern(x, y) && subpattern(y, z), subpattern(x, z))));

  s.add(forall(x, y, add(x, y) == add(y, x)));
  s.add(forall(x, y, mul(x, y) == mul(y, x)));
  s.add(forall(x, y, z, add(add(x, y), z) == add(x, add(y, z))));
  s.add(forall(x, y, z, mul(mul(x, y), z) == mul(x, mul(y, z))));
  s.add(forall(x, y, z, add(mul(x, z), mul(y, z)) == mul(add(x, y), z)));
  s.add(forall(x, y, z, add(div(x, z), div(y, z)) == div(add(x, y), z)));
  s.add(forall(x, y, z, mul(x, div(y, z)) == div(mul(x, y), z)));

  s.add(forall(x, y, subpattern(x, add(x, y))));
  s.add(forall(x, y, subpattern(x, mul(x, y))));
  s.add(forall(x, y, subpattern(x, div(x, y))));
  s.add(forall(x, y, subpattern(y, div(x, y))));
  s.add(forall(x, subpattern(x, exp(x))));
  s.add(forall(x, subpattern(x, silu(x))));
  s.add(forall(x, subpattern(x, gelu(x))));
  s.add(forall(x, subpattern(x, relu(x))));
  s.add(forall(x, f, f1, subpattern(x, clamp(f, f1, x))));
  s.add(forall(x, i, subpattern(x, rms(i, x))));
  s.add(forall(x, i, subpattern(x, red(i, x))));

  s.add(forall(x, x == red(c.int_val(0), x)));
  s.add(forall(x, i1, i2, red(i1, red(i2, x)) == red(i1 + i2, x)));
  s.add(forall(to_expr_vector(c, {x, y, i, i1, i2}),
               red(i, add(red(i1, x), red(i2, y))) ==
                   add(red(i + i1, x), red(i + i2, y))));
  s.add(forall(x, y, i, red(i, mul(x, y)) == mul(red(i, x), y)));
  s.add(forall(x, y, i, red(i, div(x, y)) == div(red(i, x), y)));

  // Lemmas
  s.add(
      forall(x, i1, i2, implies(i1 <= i2, subpattern(red(i1, x), red(i2, x)))));

  // Theorem to prove
  s.add(!subpattern(pattern1, pattern2));

  bool result = s.check() == z3::unsat;
  return result;
}

bool AbstractExpr::operator==(AbstractExpr const &other) const {
  return subpattern_to(other) && other.subpattern_to(*this);
}

Var::Var(std::string const &name) : name(name) {}

z3::expr Var::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  all_variables.insert(name);
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

z3::expr Add::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl add = c.function("add", P, P, P);
  return add(lhs->to_z3(c, all_variables), rhs->to_z3(c, all_variables));
}

std::string Add::to_string() const {
  return "(" + lhs->to_string() + "+" + rhs->to_string() + ")";
}

Mul::Mul(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs)
    : lhs(lhs), rhs(rhs) {
  assert(lhs);
  assert(rhs);
}

z3::expr Mul::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl mul = c.function("mul", P, P, P);
  return mul(lhs->to_z3(c, all_variables), rhs->to_z3(c, all_variables));
}

std::string Mul::to_string() const {
  return "(" + lhs->to_string() + rhs->to_string() + ")";
}

Div::Div(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs)
    : lhs(lhs), rhs(rhs) {
  assert(lhs);
  assert(rhs);
}

z3::expr Div::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl div = c.function("div", P, P, P);
  return div(lhs->to_z3(c, all_variables), rhs->to_z3(c, all_variables));
}

std::string Div::to_string() const {
  return "(" + lhs->to_string() + "/" + rhs->to_string() + ")";
}

Exp::Exp(std::shared_ptr<AbstractExpr> exponent) : exponent(exponent) {
  assert(exponent);
}

z3::expr Exp::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl exp = c.function("exp", P, P);
  return exp(exponent->to_z3(c, all_variables));
}

std::string Exp::to_string() const {
  return "e^" + exponent->to_string();
}

Silu::Silu(std::shared_ptr<AbstractExpr> a) : a(a) {
  assert(a);
}

z3::expr Silu::to_z3(z3::context &c,
                     std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl silu = c.function("silu", P, P);
  return silu(a->to_z3(c, all_variables));
}

std::string Silu::to_string() const {
  return "silu(" + a->to_string() + ")";
}

Gelu::Gelu(std::shared_ptr<AbstractExpr> a) : a(a) {
  assert(a);
}

z3::expr Gelu::to_z3(z3::context &c,
                     std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl gelu = c.function("gelu", P, P);
  return gelu(a->to_z3(c, all_variables));
}

std::string Gelu::to_string() const {
  return "gelu(" + a->to_string() + ")";
}

Relu::Relu(std::shared_ptr<AbstractExpr> a) : a(a) {
  assert(a);
}

z3::expr Relu::to_z3(z3::context &c,
                     std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl relu = c.function("relu", P, P);
  return relu(a->to_z3(c, all_variables));
}

std::string Relu::to_string() const {
  return "relu(" + a->to_string() + ")";
}

Clamp::Clamp(float min_val, float max_val, std::shared_ptr<AbstractExpr> elems)
    : min_val(min_val), max_val(max_val), elems(elems) {
  assert(elems);
}

z3::expr Clamp::to_z3(z3::context &c,
                      std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl clamp =
      c.function("clamp", c.fpa_sort(8, 24), c.fpa_sort(8, 24), P, P);
  return clamp(
      c.fpa_val(min_val), c.fpa_val(max_val), elems->to_z3(c, all_variables));
}

std::string Clamp::to_string() const {
  return "clamp(" + std::to_string(min_val) +
         " <= x <= " + std::to_string(max_val) + ", " + elems->to_string() +
         ")";
}

RMS::RMS(int red_deg, std::shared_ptr<AbstractExpr> elems)
    : red_deg(red_deg), elems(elems) {
  assert(elems);
}

z3::expr RMS::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl rms = c.function("rms", c.int_sort(), P, P);
  return rms(c.int_val(red_deg), elems->to_z3(c, all_variables));
}

std::string RMS::to_string() const {
  return "rms(" + std::to_string(red_deg) + ", " + elems->to_string() + ")";
}

Red::Red(int red_deg, std::shared_ptr<AbstractExpr> summand)
    : red_deg_log(std::ceil(std::log2(red_deg))), summand(summand) {
  assert(red_deg > 1);
  assert(summand);
}

z3::expr Red::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl red = c.function("red", c.int_sort(), P, P);
  return red(c.int_val(red_deg_log), summand->to_z3(c, all_variables));
}

std::string Red::to_string() const {
  return "r(" + std::to_string(red_deg_log) + ", " + summand->to_string() + ")";
}

} // namespace search
} // namespace mirage
