#include "mirage/search/symbolic_graph/dim_var_assignments.h"
#include "mirage/search/symbolic_graph/symbolic_tensor_dim.h"

namespace mirage {
namespace search {

DimVarAssignments::DimVarAssignments(
    std::unordered_map<tensor_dim_var_index_t, int> const &assignments)
    : assignments(assignments) {}

void DimVarAssignments::assign(tensor_dim_var_index_t dim_var_index,
                               int value) {
  assignments[dim_var_index] = value;
}

int DimVarAssignments::get_value(SymbolicTensorDim const &dim_template) const {
  return dim_template.dim_expr->get_value(*this);
}

int DimVarAssignments::get_value(tensor_dim_var_index_t dim_var_index) const {
  return assignments.at(dim_var_index);
}

bool DimVarAssignments::has_assignment(
    tensor_dim_var_index_t dim_var_index) const {
  return assignments.find(dim_var_index) != assignments.end();
}

DimVarAssignments
    DimVarAssignments::combine(DimVarAssignments const &rhs) const {
  DimVarAssignments combined_assignments;
  for (auto const &kv : assignments) {
    combined_assignments.assign(kv.first, kv.second);
  }
  for (auto const &kv : rhs.assignments) {
    combined_assignments.assign(kv.first, kv.second);
  }
  return combined_assignments;
}

DimVarAssignments combine_assignments(DimVarAssignments const &lhs,
                                      DimVarAssignments const &rhs) {
  return lhs.combine(rhs);
}

} // namespace search
} // namespace mirage
