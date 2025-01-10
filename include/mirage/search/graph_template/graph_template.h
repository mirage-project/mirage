#pragma once

#include "mirage/search/graph_template/tensor_template.h"
#include "mirage/search/graph_template/op_template.h"
#include "mirage/search/graph_template/tensor_dim_constraint.h"
#include "mirage/search/graph_template/dim_var_assignments.h"
#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"

#include <vector_types.h>
#include <unordered_map>

namespace mirage {
namespace search {

class TBGraphTemplate {
public:
  TBGraphTemplate();

  threadblock::Graph *to_threadblock_graph(DimVarAssignments const &assignments, std::vector<kernel::DTensor> const &inputs);
  bool add_operator(type::TBOperatorType op_type, std::vector<int> input_indices);
  bool add_input(DTensorTemplate dtensor, int3 input_map, int forloop_dim);
  bool add_output(int input_index, int3 output_map, int forloop_dim, mirage::type::TBEpilogueType epilogue_type);

  std::vector<TensorDimTemplate> grid_dim, block_dim;
  TensorDimTemplate forloop_range;
  int reduction_dimx;
  std::vector<TBOpTemplate> operators;
  std::vector<STensorTemplate> tensors;
  std::vector<std::vector<int>> input_indices;
  std::vector<std::vector<int>> output_indices;

  std::vector<TensorDimConstraint> conds;

  static tensor_dim_var_index_t next_dim_variable_index;
};

class KNGraphTemplate {
public:
  KNGraphTemplate() = default;

  kernel::Graph *to_kernel_graph(DimVarAssignments const &assignments);
  bool add_operator(type::KNOperatorType op_type, std::vector<int> input_indices);
  bool add_customized_operator(TBGraphTemplate tb_graph, std::vector<int> input_indices);
  bool add_input(std::vector<int> input_dims, std::vector<size_t> input_strides, int3 input_map);
  bool add_output(int input_index, std::vector<size_t> output_strides, int3 output_map);

  std::vector<KNOpTemplate> operators;
  std::vector<DTensorTemplate> tensors;
  std::vector<std::vector<int>> input_indices;
  std::vector<std::vector<int>> output_indices;

  std::vector<TensorDimConstraint> conds;
};

}
}