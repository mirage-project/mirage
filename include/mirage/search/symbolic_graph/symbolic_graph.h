#pragma once

#include "mirage/search/symbolic_graph/symbolic_tensor.h"
#include "mirage/search/symbolic_graph/symbolic_op.h"
#include "mirage/search/symbolic_graph/tensor_dim_constraint.h"
#include "mirage/search/symbolic_graph/dim_var_assignments.h"
#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"

#include <vector_types.h>
#include <unordered_map>

namespace mirage {
namespace search {

class SymbolicTBGraph {
public:
  SymbolicTBGraph();

  threadblock::Graph *to_threadblock_graph(DimVarAssignments const &assignments, std::vector<kernel::DTensor> const &inputs) const;
  bool add_operator(type::TBOperatorType op_type, std::vector<int> input_indices);
  bool add_input(SymbolicDTensor dtensor, int3 input_map, int forloop_dim);
  bool add_output(int input_index, int3 output_map, int forloop_dim, mirage::type::TBEpilogueType epilogue_type);
  bool remove_last_operator();

  std::vector<SymbolicTensorDim> grid_dim, block_dim;
  SymbolicTensorDim forloop_range;
  int reduction_dimx;
  std::vector<SymbolicTBOp> operators;
  std::vector<SymbolicSTensor> tensors;
  std::vector<std::vector<int>> input_indices;
  std::vector<std::vector<int>> output_indices;

  std::vector<TensorDimConstraint> conds;

  static tensor_dim_var_index_t next_dim_variable_index;

  operator json() const;
};

class SymbolicKNGraph {
public:
  SymbolicKNGraph() = default;

  kernel::Graph *to_kernel_graph(DimVarAssignments const &assignments) const;
  bool add_operator(type::KNOperatorType op_type, std::vector<int> input_indices);
  bool add_customized_operator(SymbolicTBGraph tb_graph, std::vector<int> input_indices);
  bool add_input(std::vector<int> input_dims, std::vector<size_t> input_strides, int3 input_map = {-1, -1, -1});
  bool add_output(int input_index, std::vector<size_t> output_strides, int3 output_map);
  bool remove_last_operator();

  std::vector<SymbolicKNOp> operators;
  std::vector<SymbolicDTensor> tensors;
  std::vector<std::vector<int>> input_indices;
  std::vector<std::vector<int>> output_indices;

  std::vector<TensorDimConstraint> conds;

  operator json() const;
};

}
}