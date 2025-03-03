#pragma once

#include "mirage/search/symbolic_graph/symbolic_tensor_dim.h"
#include <vector>
#include <vector_types.h>
#include <unordered_map>

namespace mirage {
namespace search {

class SymbolicMap {
public:
  std::vector<SymbolicTensorDim> device_dims;
  size_t num_tensor_dims;
  std::unordered_map<std::pair<SymbolicTensorDim, size_t>, TensorDimExpr> map_mat;

  SymbolicMap(std::vector<SymbolicTensorDim> const &device_dims, size_t num_tensor_dims, SymbolicTensorDim &index_counter);
  SymbolicMap(std::vector<SymbolicTensorDim> const &device_dims, size_t num_tensor_dims,
    std::unordered_map<SymbolicTensorDim, int> const &mapped_dims);
};

}
}
