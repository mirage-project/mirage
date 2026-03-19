#include "mirage/search/auto_tuner/auto_tuner.h"
#include "mirage/search/symbolic_graph/dim_var_assignment.h"
#include "mirage/search/symbolic_graph/tensor_dim_expr.h"
#include "mirage/search/auto_tuner/simulated_annealing.h"
#include "mirage/search/profile.h"
#include "mirage/search/symbolic_graph/op_args.h"

#include <limits>
#include <map>
#include <string>
#include <thread>

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
    for (size_t i = 0; i < symbolic_tb_graph.grid_dim.size(); ++i) {
      num_threadblocks *= values[i];
    }
    return 0 < num_threadblocks && num_threadblocks <= config::MAX_NUM_THREADBLOCKS_PER_KERNEL;
  };

  std::vector<tensor_dim_var_index_t> dim_indices_to_tune = get_parallel_dim_vars();

  // Build a minimal kernel::Graph wrapping the TB graph for a given assignment.
  // Returns a heap-allocated Graph (caller must delete), or nullptr on failure.
  auto build_kn_graph_for_tb = [&](DimVarAssignment const &a) -> kernel::Graph * {
    kernel::Graph *kn_graph = new kernel::Graph();

    std::vector<kernel::DTensor> input_dtensors;
    for (size_t i = 0; i < symbolic_tb_graph.operators.size(); ++i) {
      if (symbolic_tb_graph.operators[i].op_type == type::TBOperatorType::TB_INPUT_OP) {
        TBInputOpArgs const *args =
            static_cast<TBInputOpArgs const *>(symbolic_tb_graph.operators[i].args.get());
        SymbolicDTensor const &sym_dtensor = args->dtensor;

        std::vector<int> concrete_dims;
        for (auto const &sym_dim : sym_dtensor.dims) {
          int dim_value = a.get_value(sym_dim);
          assert(dim_value > 0);
          concrete_dims.push_back(dim_value);
        }

        std::vector<size_t> strides;
        size_t stride = 1;
        for (int j = (int)concrete_dims.size() - 1; j >= 0; --j) {
          strides.insert(strides.begin(), stride);
          stride *= concrete_dims[j];
        }

        kernel::DTensor dtensor = kn_graph->new_input(
            concrete_dims, strides, type::DT_FLOAT16, layout::DmemRowMajor);
        input_dtensors.push_back(dtensor);
      }
    }

    threadblock::Graph *tb_graph =
        symbolic_tb_graph.to_threadblock_graph(a, input_dtensors);
    if (tb_graph == nullptr) {
      delete kn_graph;
      return nullptr;
    }

    kernel::KNOperator *customized_op =
        kn_graph->create_customized_op(input_dtensors, *tb_graph);
    delete tb_graph;
    if (customized_op == nullptr) {
      delete kn_graph;
      return nullptr;
    }
    kn_graph->operators.push_back(customized_op);

    for (auto const &output_tensor : customized_op->output_tensors) {
      kn_graph->mark_output(output_tensor);
    }

    return kn_graph;
  };

  // Energy cache: avoids redundant compile+run for the same tile assignment.
  // Used by both grid search initialization and SA refinement.
  std::map<std::vector<int>, float> energy_cache;

  auto cached_energy = [&](std::vector<int> const &values) -> float {
    auto it = energy_cache.find(values);
    if (it != energy_cache.end()) return it->second;

    DimVarAssignment assignment;
    for (size_t i = 0; i < dim_indices_to_tune.size(); ++i)
      assignment.assign(dim_indices_to_tune[i], values[i]);

    kernel::Graph *kg = build_kn_graph_for_tb(assignment);
    if (!kg) { energy_cache[values] = 1e9f; return 1e9f; }
    ProfileCompileResult compiled = profile_compile(kg);
    delete kg;
    if (!compiled.is_success) { energy_cache[values] = 1e9f; return 1e9f; }
    ProfileResult result = profile_run(compiled);
    float e = result.is_success ? result.run_time : 1e9f;
    energy_cache[values] = e;
    return e;
  };

  bool grid_found_valid = false;
  auto initial_state_func = [&]() -> std::vector<int> {
    // Grid search: enumerate all valid power-of-2 assignments and profile each.
    // This replaces BFS initialization with exhaustive search over a small space.

    // Step 1: Generate candidate values per variable.
    // For each var, collect divisor constants (C where C/var appears in dims),
    // then generate all powers of 2 from 1..max(C) that divide every constant.
    size_t num_grid_dims = symbolic_tb_graph.grid_dim.size();
    std::vector<std::vector<int>> per_var_candidates;
    for (size_t vi = 0; vi < dim_indices_to_tune.size(); ++vi) {
      auto idx = dim_indices_to_tune[vi];
      int max_val = symbolic_tb_graph.get_initial_value_for_var(idx);
      // For grid dimensions, cap at MAX_NUM_THREADBLOCKS since higher
      // values are wasteful; SA can still refine within this range.
      if (vi < num_grid_dims) {
        max_val = std::min(max_val,
                           (int)config::MAX_NUM_THREADBLOCKS_PER_KERNEL);
      }
      std::vector<int> candidates;
      for (int v = 4; v <= max_val; v *= 4) {
        candidates.push_back(v);
      }
      // Also include max_val itself if it's not already in the list
      if (candidates.empty() || candidates.back() != max_val) {
        candidates.push_back(max_val);
      }
      per_var_candidates.push_back(std::move(candidates));
    }

    std::cerr << "[GridSearch tid=" << std::this_thread::get_id() << "] candidates per var:";
    for (size_t i = 0; i < dim_indices_to_tune.size(); ++i) {
      std::cerr << " var" << dim_indices_to_tune[i] << "={";
      for (size_t j = 0; j < per_var_candidates[i].size(); ++j) {
        if (j > 0) std::cerr << ",";
        std::cerr << per_var_candidates[i][j];
      }
      std::cerr << "}";
    }
    std::cerr << std::endl;

    // Step 2: Enumerate Cartesian product, filter by validity, profile each.
    std::vector<int> best_values;
    float best_time = std::numeric_limits<float>::max();
    int total_candidates = 0, valid_candidates = 0, profiled_candidates = 0;

    // Recursive enumeration via stack-based iteration.
    size_t num_vars = dim_indices_to_tune.size();
    std::vector<int> current(num_vars, 0);  // indices into per_var_candidates
    std::vector<int> values(num_vars);

    auto enumerate_done = [&]() -> bool {
      // Increment the multi-index; return true if exhausted.
      for (int i = (int)num_vars - 1; i >= 0; --i) {
        ++current[i];
        if (current[i] < (int)per_var_candidates[i].size()) return false;
        current[i] = 0;
      }
      return true;
    };

    do {
      ++total_candidates;
      for (size_t i = 0; i < num_vars; ++i)
        values[i] = per_var_candidates[i][current[i]];

      // Cheap validity check first.
      DimVarAssignment a;
      for (size_t i = 0; i < num_vars; ++i)
        a.assign(dim_indices_to_tune[i], values[i]);
      if (!check_grid_dim(values) || !symbolic_tb_graph.is_valid_assignment(a))
        continue;
      ++valid_candidates;

      // Profile via cached energy function.
      float e = cached_energy(values);
      std::cerr << "[GridSearch] ";
      for (size_t i = 0; i < num_vars; ++i)
        std::cerr << "var" << dim_indices_to_tune[i] << "=" << values[i] << " ";
      std::cerr << "-> " << (e >= 1e9f ? "FAIL" : std::to_string(e) + "ms") << std::endl;
      if (e >= 1e9f) continue;
      ++profiled_candidates;

      if (e < best_time) {
        best_time = e;
        best_values = values;
      }
    } while (!enumerate_done());

    std::cerr << "[GridSearch tid=" << std::this_thread::get_id()
              << "] total=" << total_candidates
              << " valid=" << valid_candidates
              << " profiled=" << profiled_candidates;
    if (!best_values.empty()) {
      std::cerr << " best_time=" << best_time << "ms best:";
      for (size_t i = 0; i < num_vars; ++i)
        std::cerr << " var" << dim_indices_to_tune[i] << "=" << best_values[i];
      grid_found_valid = true;
    } else {
      std::cerr << " NO VALID ASSIGNMENT FOUND";
    }
    std::cerr << std::endl;

    if (best_values.empty()) {
      return std::vector<int>(num_vars, 1);  // fallback
    }
    return best_values;
  };

  std::mt19937 local_rng(std::random_device{}());
  auto neighbor_sampling = [&](std::vector<int> const &values) -> std::vector<int> {
    std::vector<int> neighbor_values = values;
    size_t index_to_change = local_rng() % values.size();
    if (local_rng() % 2 == 0 && values[index_to_change] > 2) {
      neighbor_values[index_to_change] /= 2;
    } else {
      neighbor_values[index_to_change] *= 2;
    }
    return neighbor_values;
  };

  auto neighbor_func = [&](std::vector<int> const &values) -> std::vector<int> {
    for (int retry = 0; retry < 1000; ++retry) {
      std::vector<int> candidate = neighbor_sampling(values);
      DimVarAssignment a;
      for (size_t i = 0; i < dim_indices_to_tune.size(); ++i)
        a.assign(dim_indices_to_tune[i], candidate[i]);
      if (check_grid_dim(candidate) && symbolic_tb_graph.is_valid_assignment(a))
        return candidate;
    }
    return values;  // fallback: stay in place
  };

  std::vector<int> initial_state = initial_state_func();
  if (!grid_found_valid) {
    std::cerr << "[SA tid=" << std::this_thread::get_id() << "] grid search failed to find valid initial state; skipping tuning" << std::endl;
    DimVarAssignment assignment;
    for (size_t i = 0; i < dim_indices_to_tune.size(); ++i)
      assignment.assign(dim_indices_to_tune[i], initial_state[i]);
    return assignment;
  }

  SimulatedAnnealingConfig simulated_annealing_config;
  simulated_annealing_config.time_limit_seconds = 60.0;

  auto cached_initial = [&initial_state]() { return initial_state; };
  SimulatedAnnealing<std::vector<int>, float> simulated_annealing(simulated_annealing_config, cached_initial, neighbor_func, cached_energy);
  simulated_annealing.set_state_to_string_func([&](std::vector<int> const &vals) -> std::string {
    std::string s;
    for (size_t i = 0; i < dim_indices_to_tune.size(); ++i) {
      if (i > 0) s += " ";
      s += "var" + std::to_string(dim_indices_to_tune[i]) + "=" + std::to_string(vals[i]);
    }
    return s;
  });
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
      bool extended = assignment.extend(tb_assignment);
      assert(extended);
    }
  }
  return assignment;
}

kernel::Graph *AutoTuner::tune(std::vector<SymbolicKNGraph> const &symbolic_kn_graphs) {
  if (symbolic_kn_graphs.empty()) {
    return nullptr;
  }

  // One "group" per TB graph (customized op): we will pick a group and perturb its dim assignment.
  struct Group {
    size_t kn_graph_idx;
    SymbolicTBGraph const *tb_graph;
    std::vector<tensor_dim_var_index_t> dim_indices_to_tune;
  };
  std::vector<Group> groups;

  auto get_parallel_dim_vars = [](SymbolicTBGraph const &symbolic_tb_graph) {
    std::vector<SymbolicTensorDim> dims_to_tune;
    for (size_t i = 0; i < symbolic_tb_graph.grid_dim.size(); ++i) {
      dims_to_tune.push_back(symbolic_tb_graph.grid_dim[i]);
    }
    dims_to_tune.push_back(symbolic_tb_graph.forloop_range);
    std::vector<tensor_dim_var_index_t> dim_indices;
    for (size_t i = 0; i < dims_to_tune.size(); ++i) {
      assert(dims_to_tune[i]->is_var());
      dim_indices.push_back(std::dynamic_pointer_cast<TensorDimVar const>(dims_to_tune[i])->index);
    }
    return dim_indices;
  };

  auto check_grid_dim = [](std::vector<int> const &values, size_t num_grid_dims) -> bool {
    int num_threadblocks = 1;
    for (size_t i = 0; i < num_grid_dims; ++i) {
      num_threadblocks *= values[i];
    }
    return 0 < num_threadblocks && num_threadblocks <= config::MAX_NUM_THREADBLOCKS_PER_KERNEL;
  };

  for (size_t kg = 0; kg < symbolic_kn_graphs.size(); ++kg) {
    SymbolicKNGraph const &symbolic_kn_graph = symbolic_kn_graphs[kg]; 
    for (size_t i = 0; i < symbolic_kn_graph.operators.size(); ++i) {
      if (symbolic_kn_graph.operators[i].op_type == type::KNOperatorType::KN_CUSTOMIZED_OP) {
        std::shared_ptr<KNCustomizedOpArgs const> args =
            std::static_pointer_cast<KNCustomizedOpArgs const>(symbolic_kn_graph.operators[i].args);
        std::vector<tensor_dim_var_index_t> dim_indices = get_parallel_dim_vars(args->tb_graph_template);
        groups.push_back({kg, &args->tb_graph_template, std::move(dim_indices)});
      }
    }
  }

  if (groups.empty()) {
    return nullptr;
  }

  // State: one vector of dim values per group. Each SA step we pick a group and perturb that group's vector.
  using StateType = std::vector<std::vector<int>>;

  auto make_group_assignment = [&](size_t g, std::vector<int> const &vals) {
    DimVarAssignment a;
    for (size_t i = 0; i < groups[g].dim_indices_to_tune.size(); ++i)
      a.assign(groups[g].dim_indices_to_tune[i], vals[i]);
    return a;
  };

  auto initial_state_func = [&]() -> StateType {
    StateType state;
    state.reserve(groups.size());
    for (size_t g = 0; g < groups.size(); ++g) {
      // Shape-derived starting values.
      std::vector<int> values;
      for (auto idx : groups[g].dim_indices_to_tune)
        values.push_back(groups[g].tb_graph->get_initial_value_for_var(idx));

      std::mt19937 init_rng(std::random_device{}());
      for (int retry = 0; retry < 1000; ++retry) {
        DimVarAssignment a = make_group_assignment(g, values);
        if (check_grid_dim(values, groups[g].tb_graph->grid_dim.size()) && groups[g].tb_graph->is_valid_assignment(a))
          break;
        size_t idx = init_rng() % values.size();
        if (values[idx] > 1) values[idx] /= 2;
      }
      state.push_back(std::move(values));
    }
    return state;
  };

  std::mt19937 local_rng_multi(std::random_device{}());
  auto neighbor_sampling_for_group = [&](std::vector<int> const &values) -> std::vector<int> {
    std::vector<int> neighbor_values = values;
    size_t index_to_change = local_rng_multi() % values.size();
    if (local_rng_multi() % 2 == 0 && values[index_to_change] > 1) {
      neighbor_values[index_to_change] /= 2;
    } else {
      neighbor_values[index_to_change] *= 2;
    }
    return neighbor_values;
  };

  auto neighbor_func = [&](StateType const &state) -> StateType {
    StateType new_state = state;
    size_t g = local_rng_multi() % groups.size();
    for (int retry = 0; retry < 1000; ++retry) {
      std::vector<int> candidate = neighbor_sampling_for_group(state[g]);
      DimVarAssignment a = make_group_assignment(g, candidate);
      if (check_grid_dim(candidate, groups[g].tb_graph->grid_dim.size()) && groups[g].tb_graph->is_valid_assignment(a)) {
        new_state[g] = std::move(candidate);
        return new_state;
      }
    }
    return new_state;  // fallback: unchanged
  };

  auto state_to_assignment = [&](StateType const &state) -> DimVarAssignment {
    DimVarAssignment assignment;
    for (size_t g = 0; g < groups.size(); ++g) {
      for (size_t i = 0; i < groups[g].dim_indices_to_tune.size(); ++i) {
        assignment.assign(groups[g].dim_indices_to_tune[i], state[g][i]);
      }
    }
    return assignment;
  };

  auto energy_func = [&](StateType const &state) -> float {
    DimVarAssignment assignment = state_to_assignment(state);
    float total_time = 0.0f;
    for (size_t kg = 0; kg < symbolic_kn_graphs.size(); ++kg) {
      kernel::Graph *kn_graph = symbolic_kn_graphs[kg].to_kernel_graph(assignment);
      if (kn_graph == nullptr) {
        return 1e9f;
      }
      ProfileCompileResult compiled = profile_compile(kn_graph);
      delete kn_graph;
      if (!compiled.is_success) {
        return 1e9f;
      }
      ProfileResult result = profile_run(compiled);
      if (!result.is_success) {
        return 1e9f;
      }
      total_time += result.run_time;
    }
    return total_time;
  };

  SimulatedAnnealingConfig simulated_annealing_config;
  simulated_annealing_config.time_limit_seconds = 60.0;

  SimulatedAnnealing<StateType, float> simulated_annealing(
      simulated_annealing_config, initial_state_func, neighbor_func, energy_func);
  simulated_annealing.set_state_to_string_func([&](StateType const &state) -> std::string {
    std::string s;
    for (size_t g = 0; g < groups.size(); ++g) {
      if (g > 0) s += " | ";
      s += "g" + std::to_string(g) + ":[";
      for (size_t i = 0; i < groups[g].dim_indices_to_tune.size(); ++i) {
        if (i > 0) s += " ";
        s += "var" + std::to_string(groups[g].dim_indices_to_tune[i]) + "=" + std::to_string(state[g][i]);
      }
      s += "]";
    }
    return s;
  });
  StateType best_state = simulated_annealing.optimize();

  DimVarAssignment best_assignment = state_to_assignment(best_state);
  return symbolic_kn_graphs[0].to_kernel_graph(best_assignment);
}

kernel::Graph *AutoTuner::tune_multi_threaded(std::vector<SymbolicKNGraph> const &symbolic_kn_graphs) {
  if (symbolic_kn_graphs.empty()) {
    return nullptr;
  }
  std::vector<DimVarAssignment> assignments(symbolic_kn_graphs.size());
  std::vector<std::thread> threads;
  threads.reserve(symbolic_kn_graphs.size());
  for (size_t i = 0; i < symbolic_kn_graphs.size(); ++i) {
    threads.emplace_back([this, &symbolic_kn_graphs, &assignments, i]() {
      assignments[i] = tune(symbolic_kn_graphs[i]);
    });
  }
  for (auto &t : threads) {
    t.join();
  }
  // Try each graph's assignment; return the best (fastest) kernel graph.
  kernel::Graph *best_kg = nullptr;
  float best_time = std::numeric_limits<float>::max();
  for (size_t i = 0; i < symbolic_kn_graphs.size(); ++i) {
    // Print assignment values for debugging
    std::cerr << "[tune_mt] graph[" << i << "] assignment:";
    for (auto const &kv : assignments[i].get_assignments()) {
      std::cerr << " var" << kv.first << "=" << kv.second;
    }
    std::cerr << std::endl;
    kernel::Graph *kg = symbolic_kn_graphs[i].to_kernel_graph(assignments[i]);
    std::cerr << "[tune_mt] graph[" << i << "] to_kernel_graph=" << (kg ? "ok" : "null") << std::endl;
    if (kg == nullptr) {
      continue;
    }
    ProfileCompileResult probe = profile_compile(kg);
    std::cerr << "[tune_mt] graph[" << i << "] compile=" << (probe.is_success ? "ok" : "fail:" + probe.error_message) << std::endl;
    if (!probe.is_success) {
      delete kg;
      continue;
    }
    ProfileResult run_result = profile_run(probe);
    std::cerr << "[tune_mt] graph[" << i << "] run_time="
              << (run_result.is_success ? std::to_string(run_result.run_time) + " ms" : "fail:" + run_result.error_message)
              << std::endl;
    if (run_result.is_success && run_result.run_time < best_time) {
      delete best_kg;
      best_kg = kg;
      best_time = run_result.run_time;
    } else {
      delete kg;
    }
  }
  return best_kg;
}

} // namespace search
} // namespace mirage
