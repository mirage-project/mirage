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

  auto check_grid_dim = [&](std::vector<int> const &values) -> bool {
    int num_threadblocks = 1;
    for (size_t i = 0; i < values.size(); ++i) {
      num_threadblocks *= values[i];
    }
    return 0 < num_threadblocks && num_threadblocks <= config::MAX_NUM_THREADBLOCKS_PER_KERNEL;
  };

  std::vector<tensor_dim_var_index_t> dim_indices_to_tune = get_parallel_dim_vars();

  auto initial_state_sampling = [&]() -> std::vector<int> {
    std::vector<int> values;
    for (size_t i = 0; i < dim_indices_to_tune.size(); ++i) {
    // randomly select a power of 2
      int power = rand() % 3 + 2;
      values.push_back(1 << power);
    }
    std::cerr << "initial_state: " << json(values).dump() << std::endl;
    return values;
  };

  auto initial_state_func = [&]() -> std::vector<int> {
    while (true) {
      std::vector<int> values = initial_state_sampling();
      if (check_grid_dim(values)) {
        return values;
      }
    }
  };

  auto neighbor_sampling = [&](std::vector<int> const &values) -> std::vector<int> {
    std::vector<int> neighbor_values = values;
    size_t index_to_change = rand() % values.size();
    if (rand() % 2 == 0 && values[index_to_change] > 1) {
      neighbor_values[index_to_change] /= 2;
    } else {
      neighbor_values[index_to_change] *= 2;
    }
    return neighbor_values;
  };

  auto neighbor_func = [&](std::vector<int> const &values) -> std::vector<int> {
    while (true) {
      std::vector<int> neighbor_values = neighbor_sampling(values);
      if (check_grid_dim(neighbor_values)) {
        return neighbor_values;
      }
    }
  };

  auto energy_func = [&](std::vector<int> const &values) -> float {
    DimVarAssignment assignment;
    for (size_t i = 0; i < dim_indices_to_tune.size(); ++i) {
      assignment.assign(dim_indices_to_tune[i], values[i]);
    }
    
    kernel::Graph kn_graph;
    
    std::vector<kernel::DTensor> input_dtensors;
    for (size_t i = 0; i < symbolic_tb_graph.operators.size(); ++i) {
      if (symbolic_tb_graph.operators[i].op_type == type::TBOperatorType::TB_INPUT_OP) {
        TBInputOpArgs const *args = static_cast<TBInputOpArgs const *>(symbolic_tb_graph.operators[i].args.get());
        SymbolicDTensor const &sym_dtensor = args->dtensor;
        
        // Evaluate symbolic dimensions to get concrete dimensions
        std::vector<int> concrete_dims;
        for (auto const &sym_dim : sym_dtensor.dims) {
          int dim_value = assignment.get_value(sym_dim);
          assert(dim_value > 0);
          concrete_dims.push_back(dim_value);
        }
        
        // Create default strides (row-major)
        std::vector<size_t> strides;
        size_t stride = 1;
        for (int i = concrete_dims.size() - 1; i >= 0; --i) {
          strides.insert(strides.begin(), stride);
          stride *= concrete_dims[i];
        }
        
        kernel::DTensor dtensor = kn_graph.new_input(
            concrete_dims, strides, type::DT_FLOAT16, layout::DmemRowMajor);
        input_dtensors.push_back(dtensor);
      }
    }
    
    // Create threadblock graph from symbolic graph
    threadblock::Graph *tb_graph = symbolic_tb_graph.to_threadblock_graph(assignment, input_dtensors);
    if (tb_graph == nullptr) {
      std::cerr << "failed to create threadblock graph" << std::endl;
      return 1e9f;
    }
    
    // Create customized op with the threadblock graph
    kernel::KNOperator *customized_op = kn_graph.create_customized_op(input_dtensors, *tb_graph);
    if (customized_op == nullptr) {
      delete tb_graph;
      std::cerr << "failed to create customized op" << std::endl;
      return 1e9f;
    }
    kn_graph.operators.push_back(customized_op);
    
    // Mark outputs
    for (auto const &output_tensor : customized_op->output_tensors) {
      kn_graph.mark_output(output_tensor);
    }
    
    // Profile the kernel graph
    ProfileResult result = profile(&kn_graph);
    
    // Cleanup
    delete tb_graph;
    
    // Return runtime as energy (higher runtime = higher energy = worse)
    if (!result.is_success) {
      std::cerr << "failed to profile the kernel graph" << std::endl;
      return 1e9f;
    }
    return result.run_time;
  };

  SimulatedAnnealingConfig simulated_annealing_config;
  simulated_annealing_config.time_limit_seconds = 60.0;

  SimulatedAnnealing<std::vector<int>, float> simulated_annealing(simulated_annealing_config, initial_state_func, neighbor_func, energy_func);
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
