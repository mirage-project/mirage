#pragma once

#include "mirage/search/symbolic_graph/tensor_dim_expr.h"
#include "mirage/utils/json_utils.h"
#include "mirage/utils/hash_utils.h"

#include <unordered_map>
#include <vector>

namespace mirage {
namespace search {

template <typename ParallelDimType, typename DataDimType>
class SymbolicMap {
public:
  std::vector<ParallelDimType> parallel_dims;
  std::vector<DataDimType> data_dims;
  std::unordered_map<std::pair<ParallelDimType, DataDimType>,
                     std::shared_ptr<TensorDimExpr const>> mat;
  
  SymbolicMap(std::vector<ParallelDimType> const &parallel_dims, std::vector<DataDimType> const &data_dims, tensor_dim_var_index_t &index_counter)
    : parallel_dims(parallel_dims), data_dims(data_dims) {
    for (ParallelDimType const &parallel_dim : parallel_dims) {
      for (DataDimType const &data_dim : data_dims) {
        mat[{parallel_dim, data_dim}] = dim_expr_make_var(index_counter++);
      }
    }
  }
  // From existing map
  SymbolicMap(std::vector<ParallelDimType> const &parallel_dims, std::vector<DataDimType> const &data_dims, std::unordered_map<ParallelDimType, DataDimType> const &existing_map) 
    : parallel_dims(parallel_dims), data_dims(data_dims) {
    for (ParallelDimType const &parallel_dim : parallel_dims) {
      for (DataDimType const &data_dim : data_dims) {
        if (existing_map.at(parallel_dim) == data_dim) {
          mat[{parallel_dim, data_dim}] = dim_expr_make_const(1);
        } else {
          mat[{parallel_dim, data_dim}] = dim_expr_make_const(0);
        }
      }
    }
  }

  operator json() const {
    // TODO: Implement
    return json();
    // return json{{"parallel_dims", parallel_dims}, {"data_dims", data_dims}, {"mat", mat}};
  }
};

using SymbolicIMap = SymbolicMap<SymbolicTensorDim, size_t>;
using SymbolicOmap = SymbolicMap<SymbolicTensorDim, size_t>;

} // namespace search
} // namespace mirage