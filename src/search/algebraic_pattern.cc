#include "mirage/search/algebraic_pattern.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include "mirage/utils/containers.h"

namespace mirage {
namespace search {

std::unordered_set<std::string> AlgebraicPattern::all_variables;
std::unordered_map<std::pair<std::string, std::string>, bool>
    AlgebraicPattern::cached_results;

z3::expr_vector to_expr_vector(z3::context &c,
                               std::vector<z3::expr> const &_vec) {
  z3::expr_vector vec(c);
  for (auto const &e : _vec) {
    vec.push_back(e);
  }
  return vec;
}

bool AlgebraicPattern::subpattern_to(AlgebraicPattern const &other) const {
  std::pair<std::string, std::string> str_pair =
      std::make_pair(to_string(), other.to_string());
  if (contains_key(cached_results, str_pair)) {
    return cached_results.at(str_pair);
  }

  // clock_t start_time = clock();
  z3::context c;

  z3::sort P = c.uninterpreted_sort("P");
  z3::sort I = c.int_sort();

  z3::func_decl add = z3::function("add", P, P, P);
  z3::func_decl mul = z3::function("mul", P, P, P);
  z3::func_decl div = z3::function("div", P, P, P);
  z3::func_decl exp = z3::function("exp", P, P);
  z3::func_decl red = z3::function("red", I, P, P);

  z3::func_decl subpattern = z3::partial_order(P, 0);

  z3::solver s(c);

  z3::params p(c);
  p.set("mbqi", true);
  p.set("timeout", 10u);
  s.set(p);

  z3::expr x = c.constant("x", P);
  z3::expr y = c.constant("y", P);
  z3::expr z = c.constant("z", P);
  z3::expr i = c.int_const("i");
  z3::expr i1 = c.int_const("i1");
  z3::expr i2 = c.int_const("i2");

  z3::expr pattern1 = to_z3(c), pattern2 = other.to_z3(c);

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
  s.add(forall(x, i, subpattern(x, red(i, x))));

  s.add(forall(x, x == red(0, x)));
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
  cached_results[str_pair] = result;
  // clock_t end_time = clock();
  // std::cerr << "solver time:" << ((double)end_time - start_time) /
  // CLOCKS_PER_SEC << std::endl; std::cerr << "result: " << result <<
  // std::endl;
  return result;
}

bool AlgebraicPattern::operator==(AlgebraicPattern const &other) const {
  return subpattern_to(other) && other.subpattern_to(*this);
}

Var::Var(std::string const &name) : name(name) {}

z3::expr Var::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  all_variables.insert(name);
  return c.constant(name.data(), P);
}

std::string Var::to_string() const {
  return name;
}

Add::Add(std::shared_ptr<AlgebraicPattern> lhs,
         std::shared_ptr<AlgebraicPattern> rhs)
    : lhs(lhs), rhs(rhs) {}

z3::expr Add::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl add = z3::function("add", P, P, P);
  return add(lhs->to_z3(c), rhs->to_z3(c));
}

std::string Add::to_string() const {
  return "(" + lhs->to_string() + "+" + rhs->to_string() + ")";
}

Mul::Mul(std::shared_ptr<AlgebraicPattern> lhs,
         std::shared_ptr<AlgebraicPattern> rhs)
    : lhs(lhs), rhs(rhs) {}

z3::expr Mul::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl mul = z3::function("mul", P, P, P);
  return mul(lhs->to_z3(c), rhs->to_z3(c));
}

std::string Mul::to_string() const {
  return "(" + lhs->to_string() + rhs->to_string() + ")";
}

Div::Div(std::shared_ptr<AlgebraicPattern> lhs,
         std::shared_ptr<AlgebraicPattern> rhs)
    : lhs(lhs), rhs(rhs) {}

z3::expr Div::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl div = z3::function("div", P, P, P);
  return div(lhs->to_z3(c), rhs->to_z3(c));
}

std::string Div::to_string() const {
  return "(" + lhs->to_string() + "/" + rhs->to_string() + ")";
}

Exp::Exp(std::shared_ptr<AlgebraicPattern> exponent) : exponent(exponent) {}

z3::expr Exp::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl exp = z3::function("exp", P, P);
  return exp(exponent->to_z3(c));
}

std::string Exp::to_string() const {
  return "e^" + exponent->to_string();
}

Red::Red(int red_deg, std::shared_ptr<AlgebraicPattern> summand)
    : red_deg_log(std::ceil(std::log2(red_deg))), summand(summand) {}

z3::expr Red::to_z3(z3::context &c) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl red = z3::function("red", c.int_sort(), P, P);
  return red(red_deg_log, summand->to_z3(c));
}

std::string Red::to_string() const {
  return "r(" + std::to_string(red_deg_log) + ", " + summand->to_string() + ")";
}

} // namespace search
} // namespace mirage
