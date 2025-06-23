#pragma once

#include "mirage/kernel/graph.h"
#include "mirage/search/symbolic_graph/dim_var_assignments.h"
#include "mirage/search/symbolic_graph/symbolic_map.h"
#include "mirage/search/symbolic_graph/symbolic_op.h"
#include "mirage/search/symbolic_graph/symbolic_tensor.h"
#include "mirage/search/symbolic_graph/tensor_dim_constraints.h"
#include "mirage/threadblock/graph.h"

#include <unordered_map>
#include <unordered_set>
#include <vector_types.h>

namespace mirage {
namespace search {

class SymbolicTBGraph {
public:
  SymbolicTBGraph(tensor_dim_var_index_t dim_variable_index_base);
  std::vector<DimVarAssignments> enumerate_assignments() const;

  threadblock::Graph *
      to_threadblock_graph(DimVarAssignments const &assignments,
                           std::vector<kernel::DTensor> const &inputs) const;
  bool add_operator(type::TBOperatorType op_type,
                    std::vector<int> input_indices);
  bool add_input(SymbolicDTensor dtensor, SymbolicMap const &imap);
  bool add_input(SymbolicDTensor dtensor);
  bool add_input(SymbolicDTensor dtensor, int3 input_map, int forloop_dim);
  // bool add_output(int input_index,
  //                 SymbolicMap const &omap,
  //                 mirage::type::TBEpilogueType epilogue_type);
  // bool add_output(int input_index, mirage::type::TBEpilogueType
  // epilogue_type);
  bool add_output(int input_index,
                  int3 output_map,
                  int forloop_dim,
                  mirage::type::TBEpilogueType epilogue_type);
  bool remove_last_operator();

  tensor_dim_var_index_t dim_variable_index_base;
  tensor_dim_var_index_t next_dim_variable_index;

  std::vector<SymbolicTensorDim> grid_dim, block_dim;
  SymbolicTensorDim forloop_range;
  int reduction_dimx;
  std::vector<SymbolicTBOp> operators;
  std::vector<SymbolicSTensor> tensors;
  std::vector<std::vector<int>> input_indices;
  std::vector<std::vector<int>> output_indices;

  TensorDimConstraints conds;

  operator json() const;
};

class SymbolicKNGraph {
public:
  SymbolicKNGraph() = default;

  kernel::Graph *to_kernel_graph(DimVarAssignments const &assignments) const;
  // std::vector<DimVarAssignments> enumerate_assignments() const;
  bool add_operator(type::KNOperatorType op_type,
                    std::vector<int> input_indices);
  bool add_customized_operator(SymbolicTBGraph tb_graph,
                               std::vector<int> input_indices);
  bool add_input(std::vector<int> input_dims,
                 std::vector<size_t> input_strides,
                 int3 input_map = {-1, -1, -1});
  bool add_output(int input_index,
                  std::vector<size_t> output_strides,
                  int3 output_map);
  bool remove_last_operator();

  tensor_dim_var_index_t next_dim_variable_index;

  std::vector<SymbolicKNOp> operators;
  std::vector<SymbolicDTensor> tensors;
  std::vector<std::vector<int>> input_indices;
  std::vector<std::vector<int>> output_indices;

  std::vector<std::unordered_set<TensorDimConstraint>> conds_from_op;
  std::unordered_set<TensorDimConstraint> conds;

  operator json() const;

private:
  bool add_conds_for_new_op(
      std::unordered_set<TensorDimConstraint> const &new_conds);
};

} // namespace search
} // namespace mirage