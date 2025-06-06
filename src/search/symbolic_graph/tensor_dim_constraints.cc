#include "mirage/search/symbolic_graph/tensor_dim_constraints.h"
#include "mirage/search/symbolic_graph/dim_var_assignments.h"

#include <iostream>

namespace mirage {
namespace search {

bool TensorDimConstraints::revert() {
  return constraints.revert();
}

bool TensorDimConstraints::add_constraint(
    TensorDimConstraint const &constraint) {
  return this->add_constraints({constraint});
}

template <typename Iter>
bool TensorDimConstraints::add_constraints(Iter begin, Iter end) {
  constraints.insert(begin, end);
  if (!this->satisfiable()) {
    constraints.revert();
    return false;
  }
  return true;
}

bool TensorDimConstraints::add_constraints(
    std::initializer_list<TensorDimConstraint> const &constraints) {
  return this->add_constraints(constraints.begin(), constraints.end());
}

bool TensorDimConstraints::add_constraints(
    std::unordered_set<TensorDimConstraint> const &constraints) {
  return this->add_constraints(constraints.begin(), constraints.end());
}

bool TensorDimConstraints::satisfiable() const {
  z3::context c;
  z3::solver s(c);
  DimVarAssignments empty_assign;
  for (auto it = constraints.begin(); it != constraints.end(); ++it) {
    std::cerr << json(*it) << std::endl;
    s.add(it->to_z3(c, empty_assign));
  }
  return s.check() != z3::unsat;
}

TensorDimConstraints::operator json() const {
  return json{
      {"constraints", constraints},
  };
}

} // namespace search
} // namespace mirage
