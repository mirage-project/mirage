#include "mirage/search/symbolic_graph/tensor_dim_constraints.h"
#include "mirage/search/symbolic_graph/dim_var_assignment.h"
#include "mirage/search/symbolic_graph/tensor_dim_expr.h"

#include <iostream>
#include <unordered_set>
#include <atomic>

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
  // if (begin == end) {
  //   return true;
  // }
  // if (!this->satisfiable()) {
  //   constraints.revert();
  //   return false;
  // }
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
  return satisfiable_with_partial_assignment(DimVarAssignment());
}

bool TensorDimConstraints::satisfiable_with_partial_assignment(DimVarAssignment const &partial_assignment) const {
  return maybe_get_a_satisfying_assignment_with_partial_assignment(partial_assignment) != std::nullopt;
}

std::unordered_set<TensorDimConstraint> TensorDimConstraints::get_all_constraints() const {
  return std::unordered_set<TensorDimConstraint>(constraints.begin(), constraints.end());
}

DimVarAssignment TensorDimConstraints::get_a_satisfying_assignment() const {
  std::optional<DimVarAssignment> assignment = maybe_get_a_satisfying_assignment();
  assert(assignment);
  return *assignment;
}

DimVarAssignment TensorDimConstraints::get_a_satisfying_assignment_with_partial_assignment(DimVarAssignment const &partial_assignment) const {
  std::optional<DimVarAssignment> assignment = maybe_get_a_satisfying_assignment_with_partial_assignment(partial_assignment);
  assert(assignment);
  return *assignment;
}

std::optional<DimVarAssignment> TensorDimConstraints::maybe_get_a_satisfying_assignment() const {
  return maybe_get_a_satisfying_assignment_with_partial_assignment(DimVarAssignment());
}

std::optional<DimVarAssignment> TensorDimConstraints::maybe_get_a_satisfying_assignment_with_partial_assignment(DimVarAssignment const &partial_assignment) const {
  z3::context c;
  z3::solver s(c);
  std::unordered_set<std::shared_ptr<TensorDimVar const>> all_vars;
  for (auto it = constraints.begin(); it != constraints.end(); ++it) {
    s.add((*it)->to_z3(c, partial_assignment));
    auto vars = (*it)->get_all_vars();
    all_vars.insert(vars.begin(), vars.end());
  }
  bool result = s.check() == z3::sat;
  if (result) {
    z3::model m = s.get_model();
    DimVarAssignment assignment;
    for (auto const &var : all_vars) {
      if (partial_assignment.has_assignment(var->index)) {
        assignment.assign(var->index, partial_assignment.get_value(var->index));
      } else {
        z3::expr z3_var = var->to_z3(c, partial_assignment, false);
        assignment.assign(var->index, m.eval(z3_var).get_numeral_int());  
      }
    }
    return assignment;
  }
  return std::nullopt;
}

TensorDimConstraints TensorDimConstraints::with_partial_assignment(DimVarAssignment const &partial_assignment) const {
  std::unordered_set<TensorDimConstraint> new_constraints_set;
  for (TensorDimConstraint const &constraint : constraints) {
    new_constraints_set.insert(constraint->with_partial_assignment(partial_assignment));
  }
  TensorDimConstraints new_constraints;
  new_constraints.add_constraints(new_constraints_set);
  return new_constraints;
}

TensorDimConstraints::operator json() const {
  std::vector<json> constraints_json;
  for (auto it = constraints.begin(); it != constraints.end(); ++it) {
    constraints_json.push_back((*it)->to_string());
    // constraints_json.push_back(**it);
  }
  return json{
      {"constraints", constraints_json},
  };
}

} // namespace search
} // namespace mirage
