#pragma once

#include "mirage/kernel/graph.h"
#include "mirage/search/symbolic_graph/dim_var_assignment.h"
#include "mirage/search/symbolic_graph/symbolic_map.h"
#include "mirage/search/symbolic_graph/symbolic_op.h"
#include "mirage/search/symbolic_graph/symbolic_tensor.h"
#include "mirage/search/symbolic_graph/tensor_dim_constraints.h"
#include "mirage/search/symbolic_graph/tensor_dim_expr.h"
#include "mirage/threadblock/graph.h"
#include "mirage/type.h"
#include "mirage/vector_types.h"
#include "mirage/utils/containers.h"

#include <unordered_map>
#include <unordered_set>

namespace mirage {
namespace search {

class SymbolicTBGraph {
public:
  SymbolicTBGraph(tensor_dim_var_index_t dim_variable_index_base, int num_parallel_dims);

  threadblock::Graph *
      to_threadblock_graph(DimVarAssignment const &assignment,
                           std::vector<kernel::DTensor> const &inputs) const;
  bool add_operator(type::TBOperatorType op_type,
                    std::vector<int> input_indices);
  // bool add_input(SymbolicDTensor dtensor);
  bool add_input(SymbolicDTensor dtensor, std::vector<int> input_map, int forloop_dim);
  // bool add_output(int input_index, mirage::type::TBEpilogueType epilogue_type);
  bool add_output(int input_index, std::vector<int> output_map, type::TBEpilogueType epilogue_type);
  bool remove_last_operator();
  TensorDimConstraint get_memory_usage_constraint() const;
  bool check_memory_usage();

  tensor_dim_var_index_t dim_variable_index_base;
  tensor_dim_var_index_t next_dim_variable_index;

  std::vector<SymbolicTensorDim> grid_dim, block_dim;
  SymbolicTensorDim forloop_range;
  SymbolicTensorDim reduction_degree;
  std::vector<SymbolicTBOp> operators;
  std::vector<SymbolicSTensor> tensors;
  std::vector<std::vector<int>> input_indices;
  std::vector<std::vector<int>> output_indices;

  operator json() const;
};

class SymbolicKNGraph {
public:
  SymbolicKNGraph() = default;

  kernel::Graph *to_kernel_graph(DimVarAssignment const &assignment) const;
  bool add_operator(type::KNOperatorType op_type,
                    std::vector<int> input_indices);
  bool add_customized_operator(SymbolicTBGraph tb_graph,
                               std::vector<int> input_indices);
  bool add_input(std::vector<int> input_dims,
                 std::vector<size_t> input_strides,
                 mirage::type::DataType data_type,
                 mirage::layout::DmemLayout layout,
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

  operator json() const;
};

struct SymbolicKNGraphWithPartialAssignment {
  SymbolicKNGraph kn_graph;
  DimVarAssignment partial_assignment;
};

} // namespace search
} // namespace mirage