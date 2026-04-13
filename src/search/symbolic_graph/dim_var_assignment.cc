#include "mirage/search/symbolic_graph/dim_var_assignment.h"
#include "mirage/search/symbolic_graph/tensor_dim_expr.h"

namespace mirage {
namespace search {

DimVarAssignment::DimVarAssignment(
    std::unordered_map<tensor_dim_var_index_t, int> const &assignments)
    : assignments(assignments) {}

void DimVarAssignment::assign(tensor_dim_var_index_t dim_var_index, int value) {
  assignments[dim_var_index] = value;
}

int DimVarAssignment::get_value(SymbolicTensorDim const &dim_template) const {
  return dim_template->get_value(*this);
}

int DimVarAssignment::get_value(tensor_dim_var_index_t dim_var_index) const {
  return assignments.at(dim_var_index);
}

bool DimVarAssignment::has_assignment(
    tensor_dim_var_index_t dim_var_index) const {
  return assignments.find(dim_var_index) != assignments.end();
}

std::unordered_map<tensor_dim_var_index_t, int> const &
    DimVarAssignment::get_assignments() const {
  return assignments;
}

bool DimVarAssignment::extend(DimVarAssignment const &rhs) {
  for (auto const &kv : rhs.assignments) {
    if (contains_key(assignments, kv.first)) {
      return false;
    }
    assignments[kv.first] = kv.second;
  }
  return true;
}

std::optional<DimVarAssignment>
    DimVarAssignment::combine(DimVarAssignment const &lhs,
                              DimVarAssignment const &rhs) {
  DimVarAssignment combined_assignment = lhs;
  if (!combined_assignment.extend(rhs)) {
    return std::nullopt;
  }
  return combined_assignment;
}

} // namespace search
} // namespace mirage
