#pragma once

#include "mirage/utils/hash_utils.h"
#include "mirage/utils/json_utils.h"
#include "z3++.h"
#include <memory>
#include <unordered_map>
#include <unordered_set>

namespace mirage {
namespace search {

class AlgebraicPattern {
public:
  AlgebraicPattern() = default;
  virtual ~AlgebraicPattern() = default;

  virtual z3::expr to_z3(z3::context &c) const = 0;
  bool subpattern_to(AlgebraicPattern const &other) const;
  bool operator==(AlgebraicPattern const &other) const;
  virtual std::string to_string() const = 0;

  static std::unordered_set<std::string> all_variables;
  static std::unordered_map<std::pair<std::string, std::string>, bool>
      cached_results;
};

class Var : public AlgebraicPattern {
public:
  Var(std::string const &name);
  z3::expr to_z3(z3::context &c) const override;
  std::string to_string() const override;
  std::string name;
};

class Add : public AlgebraicPattern {
public:
  Add(std::shared_ptr<AlgebraicPattern> lhs,
      std::shared_ptr<AlgebraicPattern> rhs);
  z3::expr to_z3(z3::context &c) const override;
  std::string to_string() const override;
  std::shared_ptr<AlgebraicPattern> lhs, rhs;
};

class Mul : public AlgebraicPattern {
public:
  Mul(std::shared_ptr<AlgebraicPattern> lhs,
      std::shared_ptr<AlgebraicPattern> rhs);
  z3::expr to_z3(z3::context &c) const override;
  std::string to_string() const override;
  std::shared_ptr<AlgebraicPattern> lhs, rhs;
};

class Div : public AlgebraicPattern {
public:
  Div(std::shared_ptr<AlgebraicPattern> lhs,
      std::shared_ptr<AlgebraicPattern> rhs);
  z3::expr to_z3(z3::context &c) const override;
  std::string to_string() const override;
  std::shared_ptr<AlgebraicPattern> lhs, rhs;
};

class Exp : public AlgebraicPattern {
public:
  Exp(std::shared_ptr<AlgebraicPattern> exponent);
  z3::expr to_z3(z3::context &c) const override;
  std::string to_string() const override;
  std::shared_ptr<AlgebraicPattern> exponent;
};

class Red : public AlgebraicPattern {
public:
  Red(int red_deg, std::shared_ptr<AlgebraicPattern> summand);
  z3::expr to_z3(z3::context &c) const override;
  std::string to_string() const override;
  int red_deg_log;
  std::shared_ptr<AlgebraicPattern> summand;
};

} // namespace search
} // namespace mirage