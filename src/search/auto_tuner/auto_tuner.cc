#include "mirage/search/auto_tuner/auto_tuner.h"
#include "mirage/search/profile.h"
#include "mirage/search/symbolic_graph/dim_var_assignment.h"
#include "mirage/search/symbolic_graph/op_args.h"
#include "mirage/search/symbolic_graph/tensor_dim_expr.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <thread>

namespace mirage {
namespace search {

DimVarAssignment AutoTuner::tune_tb(SymbolicTBGraph const &symbolic_tb_graph) {
  auto get_parallel_dim_vars = [&] {
    std::vector<SymbolicTensorDim> dims_to_tune;
    for (size_t i = 0; i < symbolic_tb_graph.grid_dim.size(); ++i) {
      dims_to_tune.push_back(symbolic_tb_graph.grid_dim[i]);
    }
    dims_to_tune.push_back(symbolic_tb_graph.forloop_range);
    std::vector<tensor_dim_var_index_t> dim_indices_to_tune;
    for (size_t i = 0; i < dims_to_tune.size(); ++i) {
      assert(dims_to_tune[i]->is_var());
      dim_indices_to_tune.push_back(
          std::dynamic_pointer_cast<TensorDimVar const>(dims_to_tune[i])
              ->index);
    }
    return dim_indices_to_tune;
  };

  auto check_grid_dim = [&](std::vector<int> const &values) -> bool {
    int num_threadblocks = 1;
    for (size_t i = 0; i < symbolic_tb_graph.grid_dim.size(); ++i) {
      num_threadblocks *= values[i];
    }
    return 0 < num_threadblocks &&
           num_threadblocks <= config::MAX_NUM_THREADBLOCKS_PER_KERNEL;
  };

  std::vector<tensor_dim_var_index_t> dim_indices_to_tune =
      get_parallel_dim_vars();

  // Build a minimal kernel::Graph wrapping the TB graph for a given assignment.
  auto build_kn_graph_for_tb =
      [&](DimVarAssignment const &a) -> kernel::Graph * {
    kernel::Graph *kn_graph = new kernel::Graph();

    std::vector<kernel::DTensor> input_dtensors;
    for (size_t i = 0; i < symbolic_tb_graph.operators.size(); ++i) {
      if (symbolic_tb_graph.operators[i].op_type ==
          type::TBOperatorType::TB_INPUT_OP) {
        TBInputOpArgs const *args = static_cast<TBInputOpArgs const *>(
            symbolic_tb_graph.operators[i].args.get());
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

  // --- Generate shape-aware candidates per variable ---
  size_t num_vars = dim_indices_to_tune.size();
  std::vector<std::vector<int>> per_var_candidates;

  for (size_t vi = 0; vi < num_vars; ++vi) {
    auto idx = dim_indices_to_tune[vi];
    int data_dim = symbolic_tb_graph.get_initial_value_for_var(idx);
    bool is_forloop = (vi == num_vars - 1);
    std::vector<int> candidates;

    if (data_dim <= 64) {
      for (int v = 1; v <= data_dim; v *= 2) {
        if (data_dim % v == 0) {
          candidates.push_back(v);
        }
      }
    } else if (is_forloop) {
      int fl_min = std::max(1, data_dim / 256);
      int fl_max = data_dim / 16;
      for (int v = fl_min; v <= fl_max; v *= 2) {
        if (data_dim % v == 0) {
          candidates.push_back(v);
        }
      }
    } else {
      int g_min = std::max(1, data_dim / 512);
      int g_max = data_dim / 8;
      for (int v = g_min; v <= g_max; v *= 2) {
        if (data_dim % v == 0) {
          candidates.push_back(v);
        }
      }
    }
    if (data_dim == 2 && std::find(candidates.begin(), candidates.end(), 2) ==
                             candidates.end()) {
      candidates.push_back(2);
    }
    std::sort(candidates.begin(), candidates.end());
    if (candidates.empty()) {
      candidates.push_back(1);
    }
    per_var_candidates.push_back(std::move(candidates));
  }

  std::cerr << "[tune_tb tid=" << std::this_thread::get_id() << "] candidates:";
  for (size_t i = 0; i < num_vars; ++i) {
    std::cerr << " var" << dim_indices_to_tune[i] << "={";
    for (size_t j = 0; j < per_var_candidates[i].size(); ++j) {
      if (j > 0) {
        std::cerr << ",";
      }
      std::cerr << per_var_candidates[i][j];
    }
    std::cerr << "}";
  }
  std::cerr << std::endl;

  // --- Enumerate Cartesian product, filter by validity ---
  std::vector<int> current(num_vars, 0);
  std::vector<int> values(num_vars);
  auto enumerate_done = [&]() -> bool {
    for (int i = (int)num_vars - 1; i >= 0; --i) {
      ++current[i];
      if (current[i] < (int)per_var_candidates[i].size()) {
        return false;
      }
      current[i] = 0;
    }
    return true;
  };

  std::vector<std::vector<int>> all_valid;
  do {
    for (size_t i = 0; i < num_vars; ++i) {
      values[i] = per_var_candidates[i][current[i]];
    }
    DimVarAssignment a;
    for (size_t i = 0; i < num_vars; ++i) {
      a.assign(dim_indices_to_tune[i], values[i]);
    }
    if (!check_grid_dim(values) || !symbolic_tb_graph.is_valid_assignment(a)) {
      continue;
    }
    all_valid.push_back(values);
  } while (!enumerate_done());

  // --- Subsample up to 20 evenly distributed candidates ---
  std::vector<std::vector<int>> valid_value_sets;
  size_t const max_candidates = 20;
  if (all_valid.size() <= max_candidates) {
    valid_value_sets = all_valid;
  } else {
    for (size_t i = 0; i < max_candidates; ++i) {
      size_t idx = i * all_valid.size() / max_candidates;
      valid_value_sets.push_back(all_valid[idx]);
    }
  }

  // --- Build kernel graphs and compile in parallel ---
  size_t nv = valid_value_sets.size();
  std::vector<kernel::Graph *> kg_vec(nv, nullptr);
  std::vector<ProfileCompileResult> compiled_vec(nv);
  for (size_t i = 0; i < nv; ++i) {
    DimVarAssignment a;
    for (size_t j = 0; j < num_vars; ++j) {
      a.assign(dim_indices_to_tune[j], valid_value_sets[i][j]);
    }
    kg_vec[i] = build_kn_graph_for_tb(a);
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
          if (i >= nv) {
            break;
          }
          if (kg_vec[i]) {
            compiled_vec[i] = profile_compile(kg_vec[i]);
          }
        }
      });
    }
    for (auto &t : threads) {
      t.join();
    }
  }
  for (size_t i = 0; i < nv; ++i) {
    delete kg_vec[i];
  }

  // --- Sequential GPU profiling ---
  std::vector<int> best_values;
  float best_time = std::numeric_limits<float>::max();
  int profiled = 0;
  for (size_t i = 0; i < nv; ++i) {
    float e = 1e9f;
    if (compiled_vec[i].is_success) {
      ProfileResult result = profile_run(compiled_vec[i]);
      if (result.is_success) {
        e = result.run_time;
      }
    }
    std::cerr << "[tune_tb] ";
    for (size_t j = 0; j < num_vars; ++j) {
      std::cerr << "var" << dim_indices_to_tune[j] << "="
                << valid_value_sets[i][j] << " ";
    }
    std::cerr << "-> " << (e >= 1e9f ? "FAIL" : std::to_string(e) + "ms")
              << std::endl;
    if (e >= 1e9f) {
      continue;
    }
    ++profiled;
    if (e < best_time) {
      best_time = e;
      best_values = valid_value_sets[i];
    }
  }

  std::cerr << "[tune_tb tid=" << std::this_thread::get_id()
            << "] valid=" << all_valid.size() << " sampled=" << nv
            << " profiled=" << profiled;
  if (!best_values.empty()) {
    std::cerr << " best=" << best_time << "ms";
  } else {
    std::cerr << " NO VALID ASSIGNMENT";
  }
  std::cerr << std::endl;

  DimVarAssignment assignment;
  if (best_values.empty()) {
    // Fallback: assign 1 to all variables
    for (size_t i = 0; i < num_vars; ++i) {
      assignment.assign(dim_indices_to_tune[i], 1);
    }
  } else {
    for (size_t i = 0; i < num_vars; ++i) {
      assignment.assign(dim_indices_to_tune[i], best_values[i]);
    }
  }
  return assignment;
}

DimVarAssignment AutoTuner::tune_kn(SymbolicKNGraph const &symbolic_kn_graph) {
  DimVarAssignment assignment;
  for (size_t i = 0; i < symbolic_kn_graph.operators.size(); ++i) {
    if (symbolic_kn_graph.operators[i].op_type ==
        type::KNOperatorType::KN_CUSTOMIZED_OP) {
      std::shared_ptr<KNCustomizedOpArgs const> args =
          std::static_pointer_cast<KNCustomizedOpArgs const>(
              symbolic_kn_graph.operators[i].args);
      DimVarAssignment tb_assignment = tune_tb(args->tb_graph_template);
      bool extended = assignment.extend(tb_assignment);
      assert(extended);
    }
  }
  return assignment;
}

kernel::Graph *
    AutoTuner::tune(std::vector<SymbolicKNGraph> const &symbolic_kn_graphs) {
  if (symbolic_kn_graphs.empty()) {
    return nullptr;
  }

  size_t N = symbolic_kn_graphs.size();

  // Phase 1: Per-graph tuning in parallel threads.
  std::vector<DimVarAssignment> assignments(N);
  std::vector<bool> tune_success(N, false);
  {
    std::atomic<size_t> next_idx{0};
    unsigned hw = std::thread::hardware_concurrency();
    size_t num_threads = std::min(N, static_cast<size_t>(std::max(hw / 4, 2u)));
    std::vector<std::thread> threads;
    for (size_t t = 0; t < num_threads; ++t) {
      threads.emplace_back([&]() {
        while (true) {
          size_t gi = next_idx.fetch_add(1, std::memory_order_relaxed);
          if (gi >= N) {
            break;
          }
          assignments[gi] = tune_kn(symbolic_kn_graphs[gi]);
          tune_success[gi] = true;
        }
      });
    }
    for (auto &t : threads) {
      t.join();
    }
  }

  // Phase 2: Build kernel graphs and compile in parallel.
  std::vector<kernel::Graph *> kg_vec(N, nullptr);
  std::vector<ProfileCompileResult> compiled_vec(N);
  for (size_t gi = 0; gi < N; ++gi) {
    if (!tune_success[gi]) {
      continue;
    }
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
          if (i >= N) {
            break;
          }
          if (kg_vec[i]) {
            compiled_vec[i] = profile_compile(kg_vec[i]);
          }
        }
      });
    }
    for (auto &t : threads) {
      t.join();
    }
  }

  // Phase 3: Profile sequentially, pick best.
  kernel::Graph *best_kg = nullptr;
  float best_time = std::numeric_limits<float>::max();
  for (size_t gi = 0; gi < N; ++gi) {
    if (!kg_vec[gi]) {
      continue;
    }
    if (!compiled_vec[gi].is_success) {
      delete kg_vec[gi];
      kg_vec[gi] = nullptr;
      continue;
    }
    ProfileResult result = profile_run(compiled_vec[gi]);
    std::cerr << "[tune] graph[" << gi << "] -> "
              << (result.is_success ? std::to_string(result.run_time) + "ms"
                                    : "FAIL")
              << std::endl;
    if (result.is_success && result.run_time < best_time) {
      delete best_kg;
      best_kg = kg_vec[gi];
      kg_vec[gi] = nullptr;
      best_time = result.run_time;
    } else {
      delete kg_vec[gi];
      kg_vec[gi] = nullptr;
    }
  }

  std::cerr << "[tune] best: " << best_time << "ms" << std::endl;
  return best_kg;
}

} // namespace search
} // namespace mirage
