#pragma once

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

KVPair *egg_equiv(char const **inputs, int len);
}

namespace mirage {
namespace search {

class AbstractExpr {
public:
  AbstractExpr() = default;
  virtual ~AbstractExpr() = default;

  virtual z3::expr
      to_z3(z3::context &c,
            std::unordered_set<std::string> &all_variables) const = 0;
  std::vector<bool> subpattern_to(
      std::vector<std::shared_ptr<AbstractExpr>> const &input_patterns) const;
  virtual std::string to_string() const = 0;
  virtual std::string to_egg() const = 0;
};

class Var : public AbstractExpr {
public:
  Var(std::string const &name);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::string to_egg() const override;
  std::string name;
};

class Add : public AbstractExpr {
public:
  Add(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::string to_egg() const override;
  std::shared_ptr<AbstractExpr> lhs, rhs;
};

class Mul : public AbstractExpr {
public:
  Mul(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::string to_egg() const override;
  std::shared_ptr<AbstractExpr> lhs, rhs;
};

class Div : public AbstractExpr {
public:
  Div(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::string to_egg() const override;
  std::shared_ptr<AbstractExpr> lhs, rhs;
};

class Pow : public AbstractExpr {
public:
  Pow(std::shared_ptr<AbstractExpr> lhs, std::shared_ptr<AbstractExpr> rhs);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::string to_egg() const override;
  std::shared_ptr<AbstractExpr> lhs, rhs;
};

class Exp : public AbstractExpr {
public:
  Exp(std::shared_ptr<AbstractExpr> exponent);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::string to_egg() const override;
  std::shared_ptr<AbstractExpr> exponent;
};

class Square : public AbstractExpr {
public:
  Square(std::shared_ptr<AbstractExpr> a);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::string to_egg() const override;
  std::shared_ptr<AbstractExpr> a;
};

class Sqrt : public AbstractExpr {
public:
  Sqrt(std::shared_ptr<AbstractExpr> a);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::string to_egg() const override;
  std::shared_ptr<AbstractExpr> a;
};

class Silu : public AbstractExpr {
public:
  Silu(std::shared_ptr<AbstractExpr> a);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::string to_egg() const override;
  std::shared_ptr<AbstractExpr> a;
};

class Gelu : public AbstractExpr {
public:
  Gelu(std::shared_ptr<AbstractExpr> a);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::string to_egg() const override;
  std::shared_ptr<AbstractExpr> a;
};

class Relu : public AbstractExpr {
public:
  Relu(std::shared_ptr<AbstractExpr> a);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::string to_egg() const override;
  std::shared_ptr<AbstractExpr> a;
};

class Clamp : public AbstractExpr {
public:
  Clamp(float min_val, float max_val, std::shared_ptr<AbstractExpr> elems);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::string to_egg() const override;
  float min_val;
  float max_val;
  std::shared_ptr<AbstractExpr> elems;
};

// Note(@Mengdi): Replace it with Sqr and Sqrt once we have related algebraic
// transformation
class RMS : public AbstractExpr {
public:
  RMS(int red_deg, std::shared_ptr<AbstractExpr> elems);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::string to_egg() const override;
  int red_deg;
  std::shared_ptr<AbstractExpr> elems;
};

class Red : public AbstractExpr {
public:
  Red(int red_deg, std::shared_ptr<AbstractExpr> summand);
  z3::expr to_z3(z3::context &c,
                 std::unordered_set<std::string> &all_variables) const override;
  std::string to_string() const override;
  std::string to_egg() const override;
  int red_deg;
  std::shared_ptr<AbstractExpr> summand;
};

} // namespace search
} // namespace mirage
