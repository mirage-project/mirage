#include "mirage/search/symbolic_graph/tensor_dim_constraints.h"
#include "mirage/search/symbolic_graph/dim_var_assignments.h"
#include "mirage/search/symbolic_graph/tensor_dim_expr.h"

#include <iostream>
#include <unordered_set>

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
  z3::context c;
  z3::solver s(c);
  DimVarAssignments empty_assign;
  for (auto it = constraints.begin(); it != constraints.end(); ++it) {
    s.add((*it)->to_z3(c, empty_assign));
  }
  // timing
  // std::cerr << "Checking satisfiability" << std::endl;
  // auto start_time = std::chrono::high_resolution_clock::now();
  // bool result = s.check() == z3::sat;
  // auto end_time = std::chrono::high_resolution_clock::now();
  // std::cerr << "Result: " << result << std::endl;
  // std::chrono::duration<double> duration = end_time - start_time;
  // std::cerr << "Time taken: " << duration.count() << " seconds" << std::endl;

  // return result;
  return s.check() == z3::sat;
}

std::unordered_set<TensorDimConstraint> TensorDimConstraints::get_all_constraints() const {
  return std::unordered_set<TensorDimConstraint>(constraints.begin(), constraints.end());
}

DimVarAssignments TensorDimConstraints::get_a_satisfying_assignment() const {
  z3::context c;
  z3::solver s(c);
  DimVarAssignments empty_assign;
  std::unordered_set<std::shared_ptr<TensorDimVar const>> all_vars;
  for (auto it = constraints.begin(); it != constraints.end(); ++it) {
    s.add((*it)->to_z3(c, empty_assign));
    auto vars = (*it)->get_all_vars();
    all_vars.insert(vars.begin(), vars.end());
  }
  if (s.check() == z3::sat) {
    z3::model m = s.get_model();
    DimVarAssignments assignment;
    for (auto const &var : all_vars) {
      z3::expr z3_var = var->to_z3(c, empty_assign, false);
      assignment.assign(var->index, m.eval(z3_var).get_numeral_int());
    }
    return assignment;
  }
  assert(false);
  return DimVarAssignments();
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
