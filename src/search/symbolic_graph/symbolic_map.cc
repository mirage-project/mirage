#include "mirage/search/symbolic_graph/symbolic_map.h"

namespace mirage {
namespace search {

SymbolicMap::SymbolicMap(std::vector<SymbolicTensorDim> const &device_dims,
                         size_t num_tensor_dims,
                         tensor_dim_var_index_t &index_counter)
    : device_dims(device_dims), num_tensor_dims(num_tensor_dims) {
  for (SymbolicTensorDim const &dim : device_dims) {
    for (size_t j = 0; j < num_tensor_dims; ++j) {
      map_mat[{dim, j}] =
          dim_expr_make_var(index_counter++, TensorDimVarType::INT);
    }
  }
}

SymbolicMap::SymbolicMap(
    std::vector<SymbolicTensorDim> const &device_dims,
    size_t num_tensor_dims,
    std::unordered_map<SymbolicTensorDim, int> const &mapped_dims)
    : device_dims(device_dims), num_tensor_dims(num_tensor_dims) {
  for (SymbolicTensorDim const &dim : device_dims) {
    for (size_t j = 0; j < num_tensor_dims; ++j) {
      if (mapped_dims.at(dim) == j) {
        map_mat[{dim, j}] = dim_expr_make_const(1);
      } else {
        map_mat[{dim, j}] = dim_expr_make_const(0);
      }
    }
  }
}

SymbolicMap::operator json() const {
  // TODO
  return json();
}

} // namespace search
} // namespace mirage
