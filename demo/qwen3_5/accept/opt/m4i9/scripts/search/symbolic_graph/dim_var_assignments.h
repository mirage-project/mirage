#pragma once

#include "mirage/search/symbolic_graph/types.h"
#include <unordered_map>

namespace mirage {
namespace search {

class SymbolicTensorDim;

class DimVarAssignments {
public:
  DimVarAssignments() = default;
  DimVarAssignments(
      std::unordered_map<tensor_dim_var_index_t, int> const &assignments);

  void assign(tensor_dim_var_index_t dim_var_index, int value);
  int get_value(SymbolicTensorDim const &dim_expr) const;
  int get_value(tensor_dim_var_index_t dim_var_index) const;
  bool has_assignment(tensor_dim_var_index_t dim_var_index) const;

  DimVarAssignments combine(DimVarAssignments const &) const;

private:
  std::unordered_map<tensor_dim_var_index_t, int> assignments;
};

DimVarAssignments combine_assignments(DimVarAssignments const &lhs,
                                      DimVarAssignments const &rhs);

} // namespace search
} // namespace mirage
