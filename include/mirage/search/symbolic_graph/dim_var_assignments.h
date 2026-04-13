#pragma once

#include "mirage/search/symbolic_graph/dim_var_assignment.h"
#include <optional>
#include <vector>

namespace mirage {
namespace search {

class DimVarAssignments {
public:
  DimVarAssignments() = default;
  DimVarAssignments(std::vector<DimVarAssignment> const &assignments);

  void append(DimVarAssignment const &assignment);
  size_t size() const;

  std::vector<DimVarAssignment>::const_iterator begin() const;
  std::vector<DimVarAssignment>::const_iterator end() const;

  static std::optional<DimVarAssignments>
      maybe_cartesian_product(DimVarAssignments const &lhs,
                              DimVarAssignments const &rhs);
  static std::optional<DimVarAssignments> maybe_cartesian_product(
      std::vector<DimVarAssignments> const &assignments);
  static DimVarAssignments cartesian_product(DimVarAssignments const &lhs,
                                             DimVarAssignments const &rhs);
  static DimVarAssignments
      cartesian_product(std::vector<DimVarAssignments> const &assignments);

private:
  std::vector<DimVarAssignment> assignments;
};

} // namespace search
} // namespace mirage
