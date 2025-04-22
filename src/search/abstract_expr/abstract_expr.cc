#include "mirage/search/abstract_expr/abstract_expr.h"
#include <atomic>
#include <cassert>
#include <cmath>
#include <iostream>
#include <thread>
#include <vector>

#include "mirage/utils/containers.h"

extern "C" {
    bool egg_equiv(const char* expr1, const char* expr2);
}


namespace mirage {
namespace search {

bool AbstractExpr::subpattern_to(AbstractExpr const &other) const {
  std::string string1 = to_string();
  std::string string2 = other.to_string();
  bool result = egg_equiv(string1.to_cstr(), string2.to_cstr());

  return result;
}

bool AbstractExpr::operator==(AbstractExpr const &other) const {
  return egg_subpattern_to(other) && other.egg_subpattern_to(*this);
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
  return "(+ " + lhs->to_string() + " " + rhs->to_string() + ")";
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
  return "(* " + lhs->to_string() + " " + rhs->to_string() + ")";
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
  return "(/ " + lhs->to_string() + " " + rhs->to_string() + ")";
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
  return "(exp " + exponent->to_string() + ")";
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
  return "(silu " + a->to_string() + ")";
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
  return "(gelu " + a->to_string() + ")";
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
  return "(relu " + a->to_string() + ")";
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
  return "(clamp " + std::to_string(min_val) +
         " <= x <= " + std::to_string(max_val) + " " + elems->to_string() +
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
  return "(rms " + std::to_string(red_deg) + " " + elems->to_string() + ")";
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
  return "(sum (factor " + std::to_string(red_deg_log) + ") " + summand->to_string() + ")";
}

} // namespace search
} // namespace mirage
