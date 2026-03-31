#include "mirage/search/auto_tuner/auto_tuner.h"
#include "mirage/search/symbolic_graph/dim_var_assignment.h"
#include "mirage/search/symbolic_graph/tensor_dim_expr.h"
#include "mirage/search/auto_tuner/simulated_annealing.h"
#include "mirage/search/profile.h"
#include "mirage/search/symbolic_graph/op_args.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <random>
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
    // Use input tensor shapes to determine appropriate ranges:
    // - Grid dims produce tile = data_dim / grid; tile should be in [8, 512]
    // - Forloop produces K-tile = data_dim / fl; K-tile should be in [16, 256]
    // - Small dims (<=64): all powers of 2 up to data_dim
    size_t num_grid_dims = symbolic_tb_graph.grid_dim.size();
    std::vector<std::vector<int>> per_var_candidates;

    for (size_t vi = 0; vi < dim_indices_to_tune.size(); ++vi) {
      auto idx = dim_indices_to_tune[vi];
      int data_dim = symbolic_tb_graph.get_initial_value_for_var(idx);
      bool is_forloop = (vi == dim_indices_to_tune.size() - 1);
      std::vector<int> candidates;

      if (data_dim <= 64) {
        // Small dim: all powers of 2 up to data_dim
        for (int v = 1; v <= data_dim; v *= 2) {
          if (data_dim % v == 0) candidates.push_back(v);
        }
      } else if (is_forloop) {
        // Forloop: tile = data_dim/fl should be in [16, 256]
        // So fl in [data_dim/256, data_dim/16]
        int fl_min = std::max(1, data_dim / 256);
        int fl_max = data_dim / 16;
        for (int v = fl_min; v <= fl_max; v *= 2) {
          if (data_dim % v == 0) candidates.push_back(v);
        }
      } else {
        // Grid dim: tile = data_dim/grid should be in [8, 512]
        // So grid in [data_dim/512, data_dim/8]
        int g_min = std::max(1, data_dim / 512);
        int g_max = data_dim / 8;
        for (int v = g_min; v <= g_max; v *= 2) {
          if (data_dim % v == 0) candidates.push_back(v);
        }
      }
      // Include 2 when the data dimension is exactly 2 (e.g., batch=2)
      if (data_dim == 2 && std::find(candidates.begin(), candidates.end(), 2) == candidates.end()) {
        candidates.push_back(2);
      }
      std::sort(candidates.begin(), candidates.end());
      if (candidates.empty()) candidates.push_back(1);
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

    // Step 2a: Collect all valid candidates.
    std::vector<std::vector<int>> all_valid;
    do {
      ++total_candidates;
      for (size_t i = 0; i < num_vars; ++i)
        values[i] = per_var_candidates[i][current[i]];

      DimVarAssignment a;
      for (size_t i = 0; i < num_vars; ++i)
        a.assign(dim_indices_to_tune[i], values[i]);
      if (!check_grid_dim(values) || !symbolic_tb_graph.is_valid_assignment(a))
        continue;
      ++valid_candidates;
      all_valid.push_back(values);
    } while (!enumerate_done());

    // Subsample up to 15 evenly distributed candidates.
    std::vector<std::vector<int>> valid_value_sets;
    // Adaptive cap: more candidates for fewer variables
    size_t const max_grid_candidates = 20;
    if (all_valid.size() <= max_grid_candidates) {
      valid_value_sets = all_valid;
    } else {
      for (size_t i = 0; i < max_grid_candidates; ++i) {
        size_t idx = i * all_valid.size() / max_grid_candidates;
        valid_value_sets.push_back(all_valid[idx]);
      }
    }

    // Step 2b: Build kernel graphs and compile in parallel.
    size_t nv = valid_value_sets.size();
    std::vector<kernel::Graph*> kg_vec(nv, nullptr);
    std::vector<ProfileCompileResult> compiled_vec(nv);
    for (size_t i = 0; i < nv; ++i) {
      DimVarAssignment a;
      for (size_t j = 0; j < num_vars; ++j)
        a.assign(dim_indices_to_tune[j], valid_value_sets[i][j]);
      kg_vec[i] = build_kn_graph_for_tb(a);
      if (kg_vec[i]) {
        std::cerr << "[GridSearch] built graph for";
        for (size_t j = 0; j < num_vars; ++j)
          std::cerr << " var" << dim_indices_to_tune[j] << "=" << valid_value_sets[i][j];
        std::cerr << ": " << json(*kg_vec[i]) << std::endl;
      } else {
        std::cerr << "[GridSearch] build_kn_graph FAILED for";
        for (size_t j = 0; j < num_vars; ++j)
          std::cerr << " var" << dim_indices_to_tune[j] << "=" << valid_value_sets[i][j];
        std::cerr << std::endl;
      }
    }
    {
      unsigned hw = std::thread::hardware_concurrency();
      size_t num_threads = std::min(nv, static_cast<size_t>(hw > 0 ? hw : 8));
      std::atomic<size_t> next_idx{0};
      std::vector<std::thread> threads;
      for (size_t t = 0; t < num_threads; ++t) {
        threads.emplace_back([&]() {
          while (true) {
            size_t i = next_idx.fetch_add(1, std::memory_order_relaxed);
            if (i >= nv) break;
            if (kg_vec[i]) {
              compiled_vec[i] = profile_compile(kg_vec[i]);
            }
          }
        });
      }
      for (auto &t : threads) t.join();
    }
    for (size_t i = 0; i < nv; ++i) {
      delete kg_vec[i];
    }

    // Step 2c: Sequential GPU profiling.
    for (size_t i = 0; i < nv; ++i) {
      float e = 1e9f;
      if (compiled_vec[i].is_success) {
        ProfileResult result = profile_run(compiled_vec[i]);
        if (result.is_success) e = result.run_time;
      }
      energy_cache[valid_value_sets[i]] = e;
      std::cerr << "[GridSearch] ";
      for (size_t j = 0; j < num_vars; ++j)
        std::cerr << "var" << dim_indices_to_tune[j] << "=" << valid_value_sets[i][j] << " ";
      std::cerr << "-> " << (e >= 1e9f ? "FAIL" : std::to_string(e) + "ms") << std::endl;
      if (e >= 1e9f) continue;
      ++profiled_candidates;
      if (e < best_time) {
        best_time = e;
        best_values = valid_value_sets[i];
      }
    }

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

  std::vector<int> best_values;

  if (!config.use_sa && !config.use_evolutionary) {
    // --- Grid search only, no refinement ---
    best_values = initial_state;

  } else if (config.use_evolutionary) {
    // --- Evolutionary search ---
    // Collect grid search population from energy cache (already profiled).
    struct Individual { std::vector<int> values; float energy; };
    std::vector<Individual> population;
    for (auto const &kv : energy_cache) {
      if (kv.second < 1e8f) {
        population.push_back({kv.first, kv.second});
      }
    }
    std::sort(population.begin(), population.end(),
              [](auto const &a, auto const &b) { return a.energy < b.energy; });

    auto evo_start = std::chrono::steady_clock::now();
    double evo_time_limit = 15.0;
    int generation = 0;

    std::cerr << "[Evo tid=" << std::this_thread::get_id()
              << "] initial population: " << population.size()
              << " best=" << (population.empty() ? 0.f : population[0].energy) << "ms" << std::endl;

    while (!population.empty()) {
      auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration<double>(now - evo_start).count() >= evo_time_limit) break;

      std::vector<Individual> offspring;

      // Mutation: mutate top half (up to 5)
      for (size_t i = 0; i < population.size() / 2 && i < 5; ++i) {
        auto child_vals = neighbor_func(population[i].values);
        float e = cached_energy(child_vals);
        offspring.push_back({child_vals, e});
        now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - evo_start).count() >= evo_time_limit) break;
      }

      // Crossover: pair top individuals (up to 2 pairs)
      for (size_t i = 0; i + 1 < population.size() && i < 4; i += 2) {
        now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - evo_start).count() >= evo_time_limit) break;

        auto const &p1 = population[i].values;
        auto const &p2 = population[i + 1].values;
        std::vector<int> child(p1.size());
        for (size_t d = 0; d < p1.size(); ++d) {
          child[d] = (local_rng() % 2 == 0) ? p1[d] : p2[d];
        }
        DimVarAssignment a;
        for (size_t d = 0; d < dim_indices_to_tune.size(); ++d)
          a.assign(dim_indices_to_tune[d], child[d]);
        if (check_grid_dim(child) && symbolic_tb_graph.is_valid_assignment(a)) {
          float e = cached_energy(child);
          offspring.push_back({child, e});
        }
      }

      // Merge and select best
      population.insert(population.end(), offspring.begin(), offspring.end());
      std::sort(population.begin(), population.end(),
                [](auto const &a, auto const &b) { return a.energy < b.energy; });
      if (population.size() > 15) population.resize(15);

      ++generation;
    }

    best_values = population.empty() ? initial_state : population[0].values;
    std::cerr << "[Evo tid=" << std::this_thread::get_id()
              << "] done: generations=" << generation
              << " best=" << (population.empty() ? 0.f : population[0].energy) << "ms" << std::endl;

  } else {
    // --- Simulated Annealing (original) ---
    SimulatedAnnealingConfig simulated_annealing_config;
    simulated_annealing_config.time_limit_seconds = 15.0;

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
    best_values = simulated_annealing.optimize();
  }

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
  simulated_annealing_config.time_limit_seconds = 15.0;

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

  size_t N = symbolic_kn_graphs.size();

  // Phase 1: Per-graph tuning in parallel threads.
  // Each thread calls tune(SymbolicKNGraph) which internally does grid search
  // with parallel compilation and sequential profiling.
  // GPU profiling within each tune() is serialized by the CUDA runtime.
  std::vector<DimVarAssignment> assignments(N);
  std::vector<bool> tune_success(N, false);
  {
    std::atomic<size_t> next_idx{0};
    unsigned hw = std::thread::hardware_concurrency();
    // Use fewer threads than HW concurrency to avoid GPU contention during profiling
    size_t num_threads = std::min(N, static_cast<size_t>(std::max(hw / 4, 2u)));
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
      threads.emplace_back([&]() {
        while (true) {
          size_t gi = next_idx.fetch_add(1, std::memory_order_relaxed);
          if (gi >= N) break;
          assignments[gi] = tune(symbolic_kn_graphs[gi]);
          tune_success[gi] = true;
        }
      });
    }
    for (auto &t : threads) t.join();
  }

  // Phase 2: Build kernel graphs and compile in parallel.
  std::vector<kernel::Graph*> kg_vec(N, nullptr);
  std::vector<ProfileCompileResult> compiled_vec(N);
  for (size_t gi = 0; gi < N; ++gi) {
    if (!tune_success[gi]) continue;
    kg_vec[gi] = symbolic_kn_graphs[gi].to_kernel_graph(assignments[gi]);
  }
  {
    unsigned hw = std::thread::hardware_concurrency();
    size_t num_threads = std::min(N, static_cast<size_t>(hw > 0 ? hw : 8));
    std::atomic<size_t> next_idx{0};
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
      threads.emplace_back([&]() {
        while (true) {
          size_t i = next_idx.fetch_add(1, std::memory_order_relaxed);
          if (i >= N) break;
          if (kg_vec[i]) compiled_vec[i] = profile_compile(kg_vec[i]);
        }
      });
    }
    for (auto &t : threads) t.join();
  }

  // Phase 3: Profile sequentially, pick best.
  kernel::Graph *best_kg = nullptr;
  float best_time = std::numeric_limits<float>::max();
  for (size_t gi = 0; gi < N; ++gi) {
    if (!kg_vec[gi]) continue;
    if (!compiled_vec[gi].is_success) { delete kg_vec[gi]; kg_vec[gi] = nullptr; continue; }
    ProfileResult result = profile_run(compiled_vec[gi]);
    std::cerr << "[tune_mt] graph[" << gi << "] -> "
              << (result.is_success ? std::to_string(result.run_time) + "ms" : "FAIL") << std::endl;
    if (result.is_success && result.run_time < best_time) {
      delete best_kg;
      best_kg = kg_vec[gi]; kg_vec[gi] = nullptr;
      best_time = result.run_time;
    } else {
      delete kg_vec[gi]; kg_vec[gi] = nullptr;
    }
  }

  std::cerr << "[tune_mt] best: " << best_time << "ms" << std::endl;
  return best_kg;
}

} // namespace search
} // namespace mirage
