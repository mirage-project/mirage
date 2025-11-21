#pragma once

#include "mirage/search/symbolic_graph/dim_var_assignment.h"
#include "mirage/search/symbolic_graph/tensor_dim_expr.h"
#include "mirage/utils/containers.h"
#include "mirage/utils/json_utils.h"
#include "mirage/utils/hash_utils.h"
#include "mirage/vector_types.h"
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

  std::vector<std::shared_ptr<TensorDimExpr const>> get_all_variables_in_default_order() const {
    std::vector<std::shared_ptr<TensorDimExpr const>> variables;
    for (ParallelDimType const &parallel_dim : parallel_dims) {
      for (DataDimType const &data_dim : data_dims) {
        variables.push_back(mat.at({parallel_dim, data_dim}));
      }
    }
    return variables;
  }

  operator json() const {
    // TODO: Implement
    return json();
    // return json{{"parallel_dims", parallel_dims}, {"data_dims", data_dims}, {"mat", mat}};
  }
};

using SymbolicIMap = SymbolicMap<SymbolicTensorDim, size_t>;
using SymbolicOmap = SymbolicMap<SymbolicTensorDim, size_t>;

template <typename ParallelDimType, typename DataDimType>
int3 get_int3(SymbolicMap<ParallelDimType, DataDimType> const &map, DimVarAssignment const &assignment, int num_parallel_dims = 3) {
  assert(map.parallel_dims.size() >= num_parallel_dims);
  std::vector<int> ret(num_parallel_dims, -1);
  for (int i = 0; i < num_parallel_dims; i++) {
    for (size_t j = 0; j < map.data_dims.size(); j++) {
      if (map.mat.at({map.parallel_dims[i], map.data_dims[j]})->get_value(assignment) == 1) {
        ret[i] = j;
        break;
      }
    }
  }
  return vec_to_int3(pad_vector(ret, 3, -1));
}

template <typename ParallelDimType, typename DataDimType>
int get_forloop_dim(SymbolicMap<ParallelDimType, DataDimType> const &map, DimVarAssignment const &assignment) {
  int forloop_dim_id = map.parallel_dims.size() - 1;
  for (size_t i = 0; i < map.data_dims.size(); i++) {
    if (map.mat.at({map.parallel_dims[forloop_dim_id], i})->get_value(assignment) == 1) {
      return i;
    }
  }
  return -1;
}

} // namespace search
} // namespace mirage