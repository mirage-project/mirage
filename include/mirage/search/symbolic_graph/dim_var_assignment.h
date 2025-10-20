#pragma once

#include "mirage/search/symbolic_graph/types.h"

#include <memory>
#include <optional>
#include <unordered_map>

namespace mirage {
namespace search {

class TensorDimExpr;

class DimVarAssignment {
public:
  DimVarAssignment() = default;
  DimVarAssignment(
      std::unordered_map<tensor_dim_var_index_t, int> const &assignments);

  void assign(tensor_dim_var_index_t dim_var_index, int value);
  int get_value(std::shared_ptr<TensorDimExpr const> const &dim_expr) const;
  int get_value(tensor_dim_var_index_t dim_var_index) const;
  bool has_assignment(tensor_dim_var_index_t dim_var_index) const;
  std::unordered_map<tensor_dim_var_index_t, int> const &get_assignments() const;

  bool extend(DimVarAssignment const &rhs);

  static std::optional<DimVarAssignment> combine(DimVarAssignment const &lhs, DimVarAssignment const &rhs);

private:
  std::unordered_map<tensor_dim_var_index_t, int> assignments;
};

} // namespace search
} // namespace mirage
