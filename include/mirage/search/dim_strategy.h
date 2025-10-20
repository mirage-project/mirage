#pragma once

#include "mirage/search/config.h"
#include "mirage/search/op_utils.h"
#include "mirage/search/symbolic_graph/symbolic_tensor.h"

namespace mirage {
namespace search {

struct DimStrategy {
  DimStrategy() = default;
  DimStrategy(GeneratorConfig const &config);

  std::vector<type::KNOperatorType> get_knop_cand();
  std::vector<type::TBOperatorType> get_tbop_cand();
  std::vector<std::vector<int3>>
      get_input_map_cand(std::vector<DTensor> const &tensors, dim3 grid_dim);
  std::vector<std::vector<std::vector<int>>>
      get_input_map_cand(std::vector<SymbolicDTensor> const &tensors, size_t num_parallel_dims);
  std::vector<int3> get_output_map_cand(std::vector<STensor> const &tensors,
                                        dim3 grid_dim);
  std::vector<std::vector<std::vector<int>>> get_output_map_cand(std::vector<SymbolicSTensor> const &tensors, size_t num_parallel_dims);
  std::vector<dim3> get_grid_dim_cand(std::vector<DTensor> const &tensors);
  std::vector<dim3> get_block_dim_cand(std::vector<DTensor> const &tensors,
                                       dim3 grid_dim);
  std::vector<std::vector<int>>
      get_forloop_dim_cand(std::vector<DTensor> const &input_tensers);
  std::vector<std::vector<int>>
      get_forloop_dim_cand(std::vector<SymbolicDTensor> const &input_tensers);
  std::vector<int>
      get_forloop_range_cand(std::vector<DTensor> const &input_tensors,
                             std::vector<int3> const &input_map,
                             dim3 grid_dim,
                             dim3 block_dim,
                             std::vector<int> const &forloop_dim);
  std::vector<std::vector<int>> get_unary_input(int num_tensors);
  std::vector<std::vector<int>> get_binary_input(int num_tensors);
  std::vector<std::vector<int>> get_nary_input(int num_tensors, int n);

  std::vector<size_t> get_num_parallel_dims_cand(std::vector<SymbolicDTensor> const &tensors);

  template <typename OpType, typename TensorType>
  std::vector<std::vector<int>>
      get_input_cand_idx(OpType op_type,
                         std::vector<TensorType> const &all_inputs) {
    if (is_unary(op_type)) {
      return get_unary_input(all_inputs.size());
    }
    if (is_binary(op_type)) {
      return get_binary_input(all_inputs.size());
    }
    return get_nary_input(all_inputs.size(), get_input_number(op_type));
    assert(false && "Unsupported operator");
  }
  std::vector<std::vector<int>>
      get_customized_input_cand_idx(std::vector<DTensor> const &all_inputs);
  std::vector<std::vector<int>> get_customized_input_cand_idx(
      std::vector<SymbolicDTensor> const &all_inputs);

  GeneratorConfig config;
};

} // namespace search
} // namespace mirage