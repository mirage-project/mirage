#pragma once

#include "mirage/search/symbolic_graph/tensor_dim_expr.h"
#include "mirage/utils/json_utils.h"

#include <vector>

namespace mirage {
namespace search {

class SymbolicDTensor {
public:
  SymbolicDTensor(std::vector<SymbolicTensorDim> dim_templates);

  std::vector<SymbolicTensorDim> dims;

  operator json() const;
};

class SymbolicSTensor {
public:
  SymbolicSTensor(std::vector<SymbolicTensorDim> dim_templates,
                  bool after_accum);

  std::vector<SymbolicTensorDim> dims;
  bool after_accum;

  operator json() const;
};

void from_json(json const &j, SymbolicDTensor &tensor);
void from_json(json const &j, SymbolicSTensor &tensor);

template <typename TensorType>
SymbolicTensorDim get_tensor_size(TensorType const &tensor) {
  SymbolicTensorDim size = tensor.dims[0];
  for (size_t i = 1; i < tensor.dims.size(); i++) {
    size = size * tensor.dims[i];
  }
  return size;
}

} // namespace search
} // namespace mirage
