#include "mirage/search/search.h"
#include "mirage/kernel/customized.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/search/abstract_expr/abstract_expr_eval.h"
#include "mirage/search/dim_strategy.h"
#include "mirage/search/op_utils.h"
#include "mirage/search/verification/formal_verifier.h"
#include "mirage/search/verification/probabilistic_verifier.h"
#include "mirage/utils/containers.h"
#include "mirage/utils/json_utils.h"

#include <fstream>
#include <iostream>
#include <thread>

namespace mirage {
namespace search {

KernelGraphGenerator::KernelGraphGenerator(
    kernel::Graph const &computation_graph,
    GeneratorConfig const &config,
    char const *filename,
    bool verbose)
    : config(config), dim_strategy(DimStrategy(config)), filename(filename),
      verbose(verbose), num_total_random_tests(0), num_valid_kernel_graphs(0),
      num_total_states(0), num_tasks(0), max_depth(5) {
  // setting num_thread
  unsigned int max_num_threads = std::thread::hardware_concurrency();
  if (config.search_thread > max_num_threads) {
    printf("Config number of threads (%d) too high, setting num_thread to %d",
           config.search_thread,
           max_num_threads);
    num_thread = max_num_threads;
  } else {
    num_thread = config.search_thread;
  }

  preprocess(computation_graph);
}

template <typename GraphType>
std::vector<typename GraphType::TensorType>
    get_all_tensors(GraphType const &g) {
  std::vector<typename GraphType::TensorType> tensors;
  for (auto const &op : g.operators) {
    for (auto const &tensor : op->output_tensors) {
      tensors.push_back(tensor);
    }
  }
  return tensors;
}

template <typename GraphType>
std::vector<typename GraphType::TensorType>
    get_tensors_from_idx(GraphType const &g, std::vector<int> idx) {
  std::vector<typename GraphType::TensorType> tensors,
      all_tensors = get_all_tensors(g);
  for (auto i : idx) {
    tensors.push_back(all_tensors[i]);
  }
  return tensors;
}

template <typename GraphType>
Order get_max_op_order(GraphType const &g) {
  std::vector<typename GraphType::TensorType> all_tensors = get_all_tensors(g);
  std::unordered_map<int, int> guid2index;
  for (size_t i = 0; i < all_tensors.size(); ++i) {
    guid2index[all_tensors[i].guid] = i;
  }
  std::vector<int> input_idx;
  for (auto const &input : g.operators.back()->input_tensors) {
    assert(contains_key(guid2index, input.guid));
    input_idx.push_back(guid2index.at(input.guid));
  }
  return Order(input_idx, static_cast<int>(g.operators.back()->op_type));
}

std::vector<DTensor> get_input_tensors(threadblock::Graph const &g) {
  std::vector<DTensor> input_tensors;
  for (auto const &op : g.operators) {
    if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
      input_tensors.push_back(
          static_cast<threadblock::TBInputOp *>(op)->dtensor);
    }
  }
  return input_tensors;
}

template <typename GraphType>
std::vector<typename GraphType::TensorType>
    get_output_tensors(GraphType const &g) {
  std::vector<typename GraphType::TensorType> output_tensors;
  for (auto const &op : g.operators) {
    for (auto const &tensor : op->output_tensors) {
      if (get_num_consumers(g, tensor) == 0) {
        output_tensors.push_back(tensor);
      }
    }
  }
  return output_tensors;
}

void KernelGraphGenerator::generate_next_operator(
    SearchContext &c,
    std::function<bool(SearchContext const &)> const &verify,
    std::vector<SerializedSearchContext> &verified,
    size_t depth) {
  ++num_total_states;
  if (num_total_states % 100 == 1) {
    show_statistics();
  }
  if (verify(c)) {
    verified.push_back(SerializedSearchContext(c));
    return;
  }

  std::unordered_map<int64_t, std::shared_ptr<AbstractExpr>> algebraic_pattern;
  abstract_expr_eval(*c.kn_graph, algebraic_pattern);
  if (c.tb_graph) {
    abstract_expr_eval(*c.tb_graph, algebraic_pattern);
  }
  if (c.level == SearchLevel::LV_KERNEL) {
    assert(c.tb_graph == nullptr);
    // Case K1: finish and verify the current graph
    if (c.kn_graph->operators.size() >= config.max_num_kernel_graph_op) {
      return;
    }
    std::vector<DTensor> all_tensors = get_all_tensors(*c.kn_graph);
    for (type::KNOperatorType op_type : dim_strategy.get_knop_cand()) {
      if (op_type != type::KNOperatorType::KN_CUSTOMIZED_OP) {
        // Case K2: generate a pre-defined kernel operator
        std::vector<std::shared_ptr<AbstractExpr>> inputs;
        std::vector<std::vector<DTensor>> tensors;
        for (auto const &input_idx :
             dim_strategy.get_input_cand_idx(op_type, all_tensors)) {
          Order order(input_idx, static_cast<int>(op_type));
          if (order <= get_max_op_order(*c.kn_graph)) {
            continue;
          }
          std::vector<DTensor> input_tensors =
              get_tensors_from_idx(*c.kn_graph, input_idx);
          std::vector<std::shared_ptr<AbstractExpr>> input_patterns;
          for (auto const &t : input_tensors) {
            assert(contains_key(algebraic_pattern, t.guid));
            input_patterns.push_back(algebraic_pattern.at(t.guid));
          }
          std::shared_ptr<AbstractExpr> pattern =
              get_pattern(op_type, input_tensors, input_patterns);
          if (!pattern) {
            continue;
          } else {
            tensors.push_back(input_tensors);
            inputs.push_back(pattern);
          }
        }
        std::vector<bool> results = check_pattern(inputs);
        // filter input_tensors of 'true'
        for (int i = 0; i < results.size(); i++) {
          if (results[i]) {
            KNOperator *new_op = create_op(*c.kn_graph, op_type, tensors[i]);
            if (new_op) {
              c.kn_graph->operators.push_back(new_op);
              if (check_range(init_ranges, target_ranges, *c.kn_graph)) {
                if (depth < max_depth) {
                  num_tasks++;
                  SearchContext c_tmp =
                      SerializedSearchContext(c).deserialize();
#pragma omp task
                  {
                    generate_next_operator(c_tmp, verify, verified, depth + 1);
                  }
                } else {
                  generate_next_operator(c, verify, verified, depth + 1);
                }
              }
              delete c.kn_graph->operators.back();
              c.kn_graph->operators.pop_back();
            }
          }
        }
        results.clear();
        inputs.clear();
        tensors.clear();
      } else {
        // Case K3: generate a graph-def kernel operator
        if (count_op_of_type(type::KNOperatorType::KN_CUSTOMIZED_OP,
                             *c.kn_graph) >=
            config.max_num_threadblock_graphs) {
          continue;
        }
        static std::unordered_set<TBGraphConfig> displayed_tbgraph_configs;
        for (auto const &input_tensor_idx :
             dim_strategy.get_customized_input_cand_idx(all_tensors)) {
          Order order(input_tensor_idx, static_cast<int>(op_type));
          if (order <= get_max_op_order(*c.kn_graph)) {
            continue;
          }
          std::vector<DTensor> input_tensors;
          for (int i : input_tensor_idx) {
            input_tensors.push_back(all_tensors[i]);
          }
          for (dim3 grid_dim : dim_strategy.get_grid_dim_cand(input_tensors)) {
            for (dim3 block_dim :
                 dim_strategy.get_block_dim_cand(input_tensors, grid_dim)) {
              for (std::vector<int3> const &input_map :
                   dim_strategy.get_input_map_cand(input_tensors, grid_dim)) {
                for (std::vector<int> const &forloop_dim :
                     dim_strategy.get_forloop_dim_cand(input_tensors)) {
                  for (int forloop_range :
                       dim_strategy.get_forloop_range_cand(input_tensors,
                                                           input_map,
                                                           grid_dim,
                                                           block_dim,
                                                           forloop_dim)) {
                    if (verbose) {
                      TBGraphConfig cfg{grid_dim,
                                        block_dim,
                                        input_map,
                                        forloop_dim,
                                        forloop_range};
                      if (!contains(displayed_tbgraph_configs, cfg)) {
                        cfg.show();
                        displayed_tbgraph_configs.insert(cfg);
                      }
                    }
                    c.tb_graph = std::make_shared<threadblock::Graph>(
                        grid_dim,
                        block_dim,
                        forloop_range,
                        config.reduction_dimx);
                    bool input_created = true;
                    for (size_t i = 0; i < input_tensors.size(); ++i) {
                      DTensor dtensor = input_tensors[i];
                      TBOperator *input_op =
                          c.tb_graph->create_input_op(dtensor,
                                                      input_map[i],
                                                      forloop_dim[i],
                                                      layout::SmemRowMajor);
                      if (input_op == nullptr) {
                        input_created = false;
                        break;
                      }
                      c.tb_graph->operators.push_back(input_op);
                    }
                    if (input_created) {
                      c.level = SearchLevel::LV_THREADBLOCK;

                      if (depth < max_depth) {
                        num_tasks++;
                        SearchContext c_tmp =
                            SerializedSearchContext(c).deserialize();
#pragma omp task
                        {
                          generate_next_operator(
                              c_tmp, verify, verified, depth + 1);
                        }
                      } else {
                        generate_next_operator(c, verify, verified, depth + 1);
                      }
                      c.level = SearchLevel::LV_KERNEL;
                    }
                    c.tb_graph = nullptr;
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    // threadblock-level search
    assert(c.tb_graph != nullptr);

    auto create_threadblock_outputs = [&](int3 output_map) {
      std::vector<STensor> output_tensors;
      for (auto const &op : c.tb_graph->operators) {
        for (auto const &tensor : op->output_tensors) {
          if (get_num_consumers(*c.tb_graph, tensor) == 0) {
            if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
              return false;
            }
            if (!tensor.after_accum) {
              return false;
            }
            output_tensors.push_back(tensor);
          }
        }
      }

      if (output_tensors.size() > config.max_num_threadblock_graph_outputs) {
        return false;
      }

      for (STensor const &stensor : output_tensors) {
        assert(stensor.after_accum);
        assert(contains_key(algebraic_pattern, stensor.guid));
        TBOperator *new_op =
            c.tb_graph->create_output_op(stensor,
                                         output_map,
                                         -1 /*forloop_dim*/,
                                         mirage::type::TB_EPILOGUE_NONE);
        if (!new_op) {
          return false;
        }
        c.tb_graph->operators.push_back(new_op);
      }

      return true;
    };

    // Case B1. Finish and return to kernel-level search
    for (int3 output_map :
         dim_strategy.get_output_map_cand(c.tb_graph->grid_dim)) {
      if (create_threadblock_outputs(output_map)) {
        KNOperator *new_op = c.kn_graph->create_customized_op(
            get_input_tensors(*c.tb_graph), *c.tb_graph);
        if (!new_op) {
          continue;
        }
        c.kn_graph->operators.push_back(new_op);
        c.level = SearchLevel::LV_KERNEL;
        std::shared_ptr<threadblock::Graph> tb_graph = c.tb_graph;
        c.tb_graph = nullptr;
        if (check_range(init_ranges, target_ranges, *c.kn_graph)) {
          if (depth < max_depth) {
            num_tasks++;
            SearchContext c_tmp = SerializedSearchContext(c).deserialize();
#pragma omp task
            { generate_next_operator(c_tmp, verify, verified, depth + 1); }
          } else {
            generate_next_operator(c, verify, verified, depth + 1);
          }
        }
        c.tb_graph = tb_graph;
        c.level = SearchLevel::LV_THREADBLOCK;
        delete c.kn_graph->operators.back();
        c.kn_graph->operators.pop_back();
      }
      while (c.tb_graph->operators.back()->op_type ==
             type::TBOperatorType::TB_OUTPUT_OP) {
        c.tb_graph->operators.pop_back();
      }
    }

    if (c.tb_graph->operators.size() >= config.max_num_threadblock_graph_op) {
      return;
    }

    // Case B2: Generate pre-defined threadblock operator
    std::vector<STensor> all_tensors = get_all_tensors(*c.tb_graph);
    for (type::TBOperatorType op_type : dim_strategy.get_tbop_cand()) {
      if (count_op_of_type(type::TBOperatorType::TB_CONCAT_0_OP, *c.tb_graph) >=
              1 &&
          op_type == type::TBOperatorType::TB_CONCAT_THEN_MATMUL_OP) {
        continue;
      }
      std::vector<std::shared_ptr<AbstractExpr>> inputs;
      std::vector<std::vector<STensor>> tensors;
      for (auto const &input_idx :
           dim_strategy.get_input_cand_idx(op_type, all_tensors)) {
        Order order(input_idx, static_cast<int>(op_type));
        if (order <= get_max_op_order(*c.tb_graph)) {
          continue;
        }
        std::vector<STensor> input_tensors =
            get_tensors_from_idx(*c.tb_graph, input_idx);
        std::vector<std::shared_ptr<AbstractExpr>> input_patterns;
        for (auto const &t : input_tensors) {
          assert(contains_key(algebraic_pattern, t.guid));
          input_patterns.push_back(algebraic_pattern.at(t.guid));
        }
        std::shared_ptr<AbstractExpr> pattern =
            get_pattern(op_type, input_tensors, input_patterns);
        if (!pattern) {
          continue;
        } else {
          tensors.push_back(input_tensors);
          inputs.push_back(pattern);
        }
      }
      std::vector<bool> results = check_pattern(inputs);
      // filter input_tensors of 'true'
      for (int i = 0; i < results.size(); i++) {
        if (results[i]) {
          std::vector<STensor> input_tensors = tensors[i];
          TBOperator *last_op = c.tb_graph->operators.back();
          TBOperator *new_op = create_op(*c.tb_graph, op_type, input_tensors);

          if (new_op) {
            c.tb_graph->operators.push_back(new_op);
            if (depth < max_depth) {
              num_tasks++;
              SearchContext c_tmp = SerializedSearchContext(c).deserialize();
#pragma omp task
              { generate_next_operator(c_tmp, verify, verified, depth + 1); }
            } else {
              generate_next_operator(c, verify, verified, depth + 1);
            }
            while (c.tb_graph->operators.back() != last_op) {
              delete c.tb_graph->operators.back();
              c.tb_graph->operators.pop_back();
            }
          }
        }
      }
      results.clear();
      inputs.clear();
      tensors.clear();
    }
  }
}

void KernelGraphGenerator::generate_kernel_graphs() {
  start_time = std::chrono::steady_clock::now();
  SearchContext c;
  c.level = SearchLevel::LV_KERNEL;
  c.kn_graph = std::make_shared<kernel::Graph>();

  for (auto const &input_attr : computation_graph_input_attrs) {
    auto [dim, data_type, layout, strides] = input_attr;
    // FIXME: remove the layout attr since we use the strides
    // to describe the layout
    c.kn_graph->new_input(dim, strides, data_type, layout);
  }

  std::vector<SerializedSearchContext> verified;

  printf("num_thread = %d\n", num_thread);
#pragma omp parallel num_threads(num_thread)
  {
#pragma omp single
    {
      generate_next_operator(
          c,
          [this](SearchContext const &c) {
            return c.level == SearchLevel::LV_KERNEL &&
                   this->verify(*c.kn_graph);
          },
          verified,
          0);
    }
  }

  printf("num_tasks = %d tasks\n", num_tasks.load());

  save_results();

  printf("\n");
  printf("[Search] Second step finished. Time elapsed: %fsec\n",
         std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       start_time)
             .count());
  printf("[Search] Total states explored: %d\n", num_total_states.load());
  printf("[Search] Random tests performed: %d\n",
         num_total_random_tests.load());
  printf("[Serach] Valid kernel graphs explored: %d\n",
         num_valid_kernel_graphs.load());
}

void KernelGraphGenerator::preprocess(kernel::Graph const &computation_graph) {
  for (kernel::KNOperator *op : computation_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      computation_graph_input_attrs.push_back(
          {to_vector(op->output_tensors[0].num_dims, op->output_tensors[0].dim),
           op->output_tensors[0].data_type,
           op->output_tensors[0].layout,
           static_cast<kernel::KNInputOp *>(op)->input_strides});
    }
  }

  std::unordered_map<int64_t, std::shared_ptr<AbstractExpr>>
      computation_graph_patterns;
  abstract_expr_eval(computation_graph, computation_graph_patterns);

  init_ranges = get_init_ranges(computation_graph);
  target_ranges = get_interact_ranges(init_ranges, computation_graph);
  assert(init_ranges.size() == target_ranges.size());

  for (kernel::KNOperator *op : computation_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      computation_graph_output_patterns.push_back(
          computation_graph_patterns.at(op->input_tensors[0].guid));
    }
  }

  for (auto const &final_pattern : computation_graph_output_patterns) {
    std::string expr = final_pattern->to_egg();
    get_egraph(expr.c_str());
  }

  if (config.verifier_type == VerifierType::PROBABILISTIC_VERIFIER) {
    this->verifier = std::make_shared<ProbabilisticVerifier>(computation_graph);
  } else {
    this->verifier = std::make_shared<FormalVerifier>(computation_graph);
  }
}

std::vector<bool> KernelGraphGenerator::check_pattern(
    std::vector<std::shared_ptr<AbstractExpr>> &inputs) {
  std::unordered_map<int, bool> results;
  std::vector<int> keys;
  for (int i = 0; i < inputs.size(); i++) {
    auto input = inputs[i];
    if (seen_patterns.find(input->to_string()) != seen_patterns.end()) {
      results[i] = seen_patterns[input->to_string()];
      inputs[i] = nullptr;
    } else {
      keys.push_back(i);
    }
  }

  bool all_null = std::all_of(
      inputs.begin(),
      inputs.end(),
      [](std::shared_ptr<AbstractExpr> const &ptr) { return ptr == nullptr; });
  if (all_null) {
    std::vector<bool> ordered_results;
    for (int i = 0; i < results.size(); i++) {
      ordered_results.push_back(results[i]);
    }
    return ordered_results;
  }

  for (auto const &final_pattern : computation_graph_output_patterns) {
    std::vector<bool> tmp = final_pattern->subpattern_to(inputs);
    for (int i = 0; i < keys.size(); i++) {
      if (tmp[i] == true) {
        results[keys[i]] = true;
        auto input = inputs[keys[i]];
#pragma omp critical
        { seen_patterns[input->to_string()] = true; }

        inputs[keys[i]] = nullptr;
      }
    }
  }
  for (int i = 0; i < inputs.size(); ++i) {
    auto input = inputs[i];
    if (input != nullptr) {
      results[i] = false;
#pragma omp critical
      { seen_patterns[input->to_string()] = false; }
    }
  }
  std::vector<bool> ordered_results;
  for (int i = 0; i < results.size(); i++) {
    ordered_results.push_back(results[i]);
  }
  return ordered_results;
}

bool KernelGraphGenerator::verify(kernel::Graph &g) {
  std::vector<DTensor> outputs = get_output_tensors(g);

  if (outputs.size() != computation_graph_output_patterns.size()) {
    return false;
  }

  {
    ++num_total_random_tests;
    auto mark_outputs = [&](OutputMatch const &match) {
      for (size_t i = 0; i < outputs.size(); ++i) {
        g.mark_output(outputs[match[i]]);
      }
    };

    auto unmark_outputs = [&]() {
      while (g.operators.back()->op_type ==
             type::KNOperatorType::KN_OUTPUT_OP) {
        delete g.operators.back();
        g.operators.pop_back();
      }
    };

    auto save_graph = [&]() {
#pragma omp critical
      { generated_graphs.push_back(json(g)); }
    };

    OutputMatch match = verifier->verify(g);
    if (match.is_valid()) {
      ++num_valid_kernel_graphs;
      mark_outputs(match);
      save_graph();
      unmark_outputs();
      return true;
    }
  }

  return false;
}

void KernelGraphGenerator::save_results() const {
  std::ofstream ofs(filename);
  ofs << json(generated_graphs);
}

double KernelGraphGenerator::get_elapsed_time_in_sec() const {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       start_time)
      .count();
}

void KernelGraphGenerator::show_statistics() const {
  printf(
      "[Search] States: %d, Random tests: %d, Valid mugraphs: %d, Time: %lf\r",
      num_total_states.load(),
      num_total_random_tests.load(),
      num_valid_kernel_graphs.load(),
      get_elapsed_time_in_sec());
}

} // namespace search
} // namespace mirage