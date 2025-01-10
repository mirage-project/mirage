#pragma once

#include <unordered_map>

namespace mirage {
namespace search {

class TensorDimTemplate;

using tensor_dim_var_index_t = uint32_t;

class DimVarAssignments {
public:
  DimVarAssignments() = default;

  void assign(tensor_dim_var_index_t dim_var_index, int value);
  int get_value(TensorDimTemplate const &dim_expr) const;
  int get_value(tensor_dim_var_index_t dim_var_index) const;

private:
  std::unordered_map<tensor_dim_var_index_t, int> assignments;
};

}
}