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

std::vector<bool> AbstractExpr::subpattern_to(
    std::vector<std::shared_ptr<AbstractExpr>> const &input_patterns) const {
  std::vector<std::string> subexpr;
  std::vector<bool> is_valid;
  for (auto const &input : input_patterns) {
    if (input == nullptr) {
      subexpr.push_back("null");
      is_valid.push_back(false);
    } else {
      subexpr.push_back(input->to_egg().c_str());
      is_valid.push_back(true);
    }
  }

  std::vector<char const *> c_subexpr;
  c_subexpr.reserve(subexpr.size());
  for (auto const &s : subexpr) {
    c_subexpr.push_back(s.c_str());
  }

  KVPair *datas =
      egg_equiv(c_subexpr.data(), static_cast<int>(c_subexpr.size()));

  std::unordered_map<int, bool> result;
  size_t len = c_subexpr.size();
  for (size_t i = 0; i < len; ++i) {
    if (is_valid[i]) {
      result[datas[i].key] = datas[i].value;
    }
  }
  std::vector<int> keys;
  for (auto const &kv : result) {
    keys.push_back(kv.first);
  }

  std::sort(keys.begin(), keys.end());
  std::vector<bool> ordered_results;
  for (int key : keys) {
    ordered_results.push_back(result[key]);
  }
  return ordered_results;
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

std::string Var::to_egg() const {
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

std::string Add::to_egg() const {
  return "(+ " + lhs->to_egg() + " " + rhs->to_egg() + ")";
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

std::string Mul::to_egg() const {
  return "(* " + lhs->to_egg() + " " + rhs->to_egg() + ")";
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

std::string Div::to_egg() const {
  return "(/ " + lhs->to_egg() + " " + rhs->to_egg() + ")";
}

Pow::Pow(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs)
    : lhs(lhs), rhs(rhs) {
  assert(lhs);
  assert(rhs);
}

z3::expr Pow::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl pow = c.function("pow", P, P, P);
  return pow(lhs->to_z3(c, all_variables), rhs->to_z3(c, all_variables));
}

std::string Pow::to_string() const {
  return "(" + lhs->to_string() + "^" + rhs->to_string() + ")";
}

std::string Pow::to_egg() const {
  return "(pow " + lhs->to_egg() + " " + rhs->to_egg() + ")";
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

std::string Exp::to_egg() const {
  return "(exp " + exponent->to_egg() + ")";
}

Square::Square(std::shared_ptr<AbstractExpr> a) : a(a) {
  assert(a);
}

z3::expr Square::to_z3(z3::context &c,
                       std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl square = c.function("square", P, P);
  return square(a->to_z3(c, all_variables));
}

std::string Square::to_string() const {
  return "square(" + a->to_string() + ")";
}

std::string Square::to_egg() const {
  return "(square " + a->to_egg() + ")";
}

Sqrt::Sqrt(std::shared_ptr<AbstractExpr> a) : a(a) {
  assert(a);
}

z3::expr Sqrt::to_z3(z3::context &c,
                     std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl sqrt = c.function("sqrt", P, P);
  return sqrt(a->to_z3(c, all_variables));
}

std::string Sqrt::to_string() const {
  return "sqrt(" + a->to_string() + ")";
}

std::string Sqrt::to_egg() const {
  return "(sqrt " + a->to_egg() + ")";
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

std::string Silu::to_egg() const {
  return "(silu " + a->to_egg() + ")";
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

std::string Gelu::to_egg() const {
  return "(gelu " + a->to_egg() + ")";
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

std::string Relu::to_egg() const {
  return "(relu " + a->to_egg() + ")";
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

std::string Clamp::to_egg() const {
  return "(clamp " + std::to_string(min_val) + " " + std::to_string(max_val) +
         " " + elems->to_egg() + ")";
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

std::string RMS::to_egg() const {
  return "(rms " + std::to_string(red_deg) + " " + elems->to_egg() + ")";
}

Red::Red(int red_deg, std::shared_ptr<AbstractExpr> summand)
    : red_deg(red_deg), summand(summand) {
  assert(red_deg > 1);
  assert(summand);
}

z3::expr Red::to_z3(z3::context &c,
                    std::unordered_set<std::string> &all_variables) const {
  z3::sort P = c.uninterpreted_sort("P");
  z3::func_decl red = c.function("red", c.int_sort(), P, P);
  return red(c.int_val(static_cast<int>(std::ceil(std::log2(red_deg)))),
             summand->to_z3(c, all_variables));
}

std::string Red::to_string() const {
  return "r(" + std::to_string(std::ceil(std::log2(red_deg))) + ", " +
         summand->to_string() + ")";
}

std::string Red::to_egg() const {
  return "(sum " + std::to_string(red_deg) + " " + summand->to_egg() + ")";
}

} // namespace search
} // namespace mirage
