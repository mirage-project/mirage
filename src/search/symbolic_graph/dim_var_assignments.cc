#include "mirage/search/symbolic_graph/dim_var_assignments.h"
#include "mirage/search/symbolic_graph/dim_var_assignment.h"

#include <cassert>

namespace mirage {
namespace search {

DimVarAssignments::DimVarAssignments(std::vector<DimVarAssignment> const &assignments) : assignments(assignments) {}

void DimVarAssignments::append(DimVarAssignment const &assignment) {
  assignments.push_back(assignment);
}

size_t DimVarAssignments::size() const {
  return assignments.size();
}

std::vector<DimVarAssignment>::const_iterator DimVarAssignments::begin() const {
  return assignments.begin();
}

std::vector<DimVarAssignment>::const_iterator DimVarAssignments::end() const {
  return assignments.end();
}

std::optional<DimVarAssignments> DimVarAssignments::maybe_cartesian_product(DimVarAssignments const &lhs, DimVarAssignments const &rhs) {
  std::vector<DimVarAssignment> new_assignments;
  for (DimVarAssignment const &assignment : lhs.assignments) {
    for (DimVarAssignment const &rhs_assignment : rhs.assignments) {
      std::optional<DimVarAssignment> combined_assignment = DimVarAssignment::combine(assignment, rhs_assignment);
      if (!combined_assignment) {
        return std::nullopt;
      }
      new_assignments.push_back(*combined_assignment);
    }
  }
  return DimVarAssignments(new_assignments);
}

std::optional<DimVarAssignments> DimVarAssignments::maybe_cartesian_product(std::vector<DimVarAssignments> const &assignments) {
  DimVarAssignments product;
  for (DimVarAssignments const &assignment : assignments) {
    std::optional<DimVarAssignments> new_product = maybe_cartesian_product(product, assignment);
    if (!new_product) {
      return std::nullopt;
    }
    product = *new_product;
  }
  return product;
}

DimVarAssignments DimVarAssignments::cartesian_product(DimVarAssignments const &lhs, DimVarAssignments const &rhs) {
  std::optional<DimVarAssignments> new_product = maybe_cartesian_product(lhs, rhs);
  assert(new_product);
  return *new_product;
}

DimVarAssignments DimVarAssignments::cartesian_product(std::vector<DimVarAssignments> const &assignments) {
  std::optional<DimVarAssignments> new_product = maybe_cartesian_product(assignments);
  assert(new_product);
  return *new_product;
}

} // namespace search
} // namespace mirage
