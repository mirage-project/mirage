#include "mirage/search/auto_tuner/auto_tuner.h"
#include "mirage/search/symbolic_graph/dim_var_assignment.h"
#include "mirage/search/symbolic_graph/tensor_dim_expr.h"
#include "mirage/search/auto_tuner/simulated_annealing.h"
#include "mirage/search/profile.h"
#include "mirage/search/symbolic_graph/op_args.h"

namespace mirage {
namespace search {

AutoTuner::AutoTuner(AutoTunerConfig const &config)
  : config(config) {}

DimVarAssignment AutoTuner::tune(SymbolicTBGraph const &symbolic_tb_graph) {
  auto get_parallel_dim_vars = [&] {
    std::vector<SymbolicTensorDim> dims_to_tune;
    for (size_t i = 0; i < symbolic_tb_graph.grid_dim.size(); ++i) {
      dims_to_tune.push_back(symbolic_tb_graph.grid_dim[i]);
    }
    dims_to_tune.push_back(symbolic_tb_graph.forloop_range);
    std::vector<tensor_dim_var_index_t> dim_indices_to_tune;
    for (size_t i = 0; i < dims_to_tune.size(); ++i) {
      assert(dims_to_tune[i]->is_var());
      dim_indices_to_tune.push_back(std::dynamic_pointer_cast<TensorDimVar const>(dims_to_tune[i])->index);
    }
    return dim_indices_to_tune;
  };

  std::vector<tensor_dim_var_index_t> dim_indices_to_tune = get_parallel_dim_vars();

  auto initial_state_func = [&]() -> std::vector<int> {
    std::vector<int> values;
    for (size_t i = 0; i < dim_indices_to_tune.size(); ++i) {
      // randomly select a power of 2
      int power = rand() % 10;
      values.push_back(rand() % (1 << power));
    }
    return values;
  };

  auto neighbor_func = [&](std::vector<int> const &values) -> std::vector<int> {
    std::vector<int> neighbor_values = values;
    size_t index_to_change = rand() % values.size();
    if (rand() % 2 == 0) {
      neighbor_values[index_to_change] /= 2;
    } else {
      neighbor_values[index_to_change] *= 2;
    }
    return neighbor_values;
  };

  auto energy_func = [&](std::vector<int> const &values) -> float {
    // Create DimVarAssignment from values
    DimVarAssignment assignment;
    for (size_t i = 0; i < dim_indices_to_tune.size(); ++i) {
      assignment.assign(dim_indices_to_tune[i], values[i]);
    }
    
    // Collect input DTensors needed by the threadblock graph
    std::vector<kernel::DTensor> input_dtensors;
    for (size_t i = 0; i < symbolic_tb_graph.operators.size(); ++i) {
      if (symbolic_tb_graph.operators[i].op_type == type::TBOperatorType::TB_INPUT_OP) {
        TBInputOpArgs const *args = static_cast<TBInputOpArgs const *>(symbolic_tb_graph.operators[i].args.get());
        SymbolicDTensor const &sym_dtensor = args->dtensor;
        
        // Evaluate symbolic dimensions to get concrete dimensions
        std::vector<int> concrete_dims;
        for (auto const &sym_dim : sym_dtensor.dims) {
          int dim_value = assignment.get_value(sym_dim);
          // Use a reasonable default if dimension is not assigned
          if (dim_value <= 0) {
            dim_value = 128; // Default size
          }
          concrete_dims.push_back(dim_value);
        }
        
        // Create default strides (row-major)
        std::vector<size_t> strides;
        size_t stride = 1;
        for (int i = concrete_dims.size() - 1; i >= 0; --i) {
          strides.insert(strides.begin(), stride);
          stride *= concrete_dims[i];
        }
        
        // Create kernel graph with input DTensor
        kernel::Graph temp_kn_graph;
        kernel::DTensor dtensor = temp_kn_graph.new_input(
            concrete_dims, strides, type::DT_FLOAT16, layout::DmemRowMajor);
        input_dtensors.push_back(dtensor);
      }
    }
    
    // Create threadblock graph from symbolic graph
    threadblock::Graph *tb_graph = symbolic_tb_graph.to_threadblock_graph(assignment, input_dtensors);
    if (tb_graph == nullptr) {
      // Failed to create threadblock graph - return high energy (bad performance)
      return 1e9f;
    }
    
    // Create kernel graph with customized op wrapping the threadblock graph
    kernel::Graph *kn_graph = new kernel::Graph();
    
    // Create input DTensors in the kernel graph
    std::vector<kernel::DTensor> kn_input_dtensors;
    for (auto const &dtensor : input_dtensors) {
      // Recreate in the new kernel graph
      std::vector<int> dims;
      for (int i = 0; i < dtensor.num_dims; ++i) {
        dims.push_back(dtensor.dim[i]);
      }
      std::vector<size_t> strides_vec;
      // Note: DTensor doesn't store strides directly, use default row-major
      size_t stride = 1;
      for (int i = dims.size() - 1; i >= 0; --i) {
        strides_vec.insert(strides_vec.begin(), stride);
        stride *= dims[i];
      }
      kernel::DTensor kn_dtensor = kn_graph->new_input(
          dims, strides_vec, dtensor.data_type, dtensor.layout);
      kn_input_dtensors.push_back(kn_dtensor);
    }
    
    // Create customized op with the threadblock graph
    kernel::KNOperator *customized_op = kn_graph->create_customized_op(kn_input_dtensors, *tb_graph);
    if (customized_op == nullptr) {
      delete tb_graph;
      delete kn_graph;
      return 1e9f; // Failed to create customized op
    }
    kn_graph->operators.push_back(customized_op);
    
    // Mark outputs
    for (auto const &output_tensor : customized_op->output_tensors) {
      kn_graph->mark_output(output_tensor);
    }
    
    // Profile the kernel graph
    ProfileResult result = profile(kn_graph);
    
    // Cleanup
    delete tb_graph;
    delete kn_graph;
    
    // Return runtime as energy (higher runtime = higher energy = worse)
    if (!result.is_success) {
      return 1e9f; // Failed to profile - return high energy
    }
    return result.run_time;
  };

  SimulatedAnnealing<std::vector<int>, float> simulated_annealing(SimulatedAnnealingConfig(), initial_state_func, neighbor_func, energy_func);
  std::vector<int> best_values = simulated_annealing.optimize();

  DimVarAssignment assignment;
  for (size_t i = 0; i < dim_indices_to_tune.size(); ++i) {
    assignment.assign(dim_indices_to_tune[i], best_values[i]);
  }
  return assignment;
}

DimVarAssignment AutoTuner::tune(SymbolicKNGraph const &symbolic_kn_graph) {
  DimVarAssignment assignment;
  for (size_t i = 0; i < symbolic_kn_graph.operators.size(); ++i) {
    if (symbolic_kn_graph.operators[i].op_type == type::KNOperatorType::KN_CUSTOMIZED_OP) {
      std::shared_ptr<KNCustomizedOpArgs const> args =
          std::static_pointer_cast<KNCustomizedOpArgs const>(symbolic_kn_graph.operators[i].args);
      DimVarAssignment tb_assignment = tune(args->tb_graph_template);
      assert(assignment.extend(tb_assignment));
    }
  }
  return assignment;
}

} // namespace search
} // namespace mirage
