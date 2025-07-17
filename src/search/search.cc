#include "mirage/search/search.h"
#include "mirage/kernel/customized.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/search/abstract_expr/abstract_expr_eval.h"
#include "mirage/search/dim_strategy.h"
#include "mirage/search/op_utils.h"
#include "mirage/search/symbolic_graph/op_args.h"
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
    char const *checkpoint_filename,
    bool verbose)
    : config(config), dim_strategy(DimStrategy(config)),
      checkpoint_filename(checkpoint_filename), verbose(verbose),
      num_total_random_tests(0), num_valid_kernel_graphs(0),
      num_total_states(0), num_tasks(0), multithread_threshold_depth(5) {
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

template <typename GraphType, typename OpType>
bool check_order(std::vector<int> input_idx,
                 OpType op_type,
                 GraphType const &g) {
  Order order(input_idx, static_cast<int>(op_type));
  if (order <= get_max_op_order(g)) {
    return false;
  }
  return true;
}

template <>
Order get_max_op_order(SymbolicKNGraph const &g) {
  return Order(g.input_indices.back(),
               static_cast<int>(g.operators.back().op_type));
}

template <>
Order get_max_op_order(SymbolicTBGraph const &g) {
  return Order(g.input_indices.back(),
               static_cast<int>(g.operators.back().op_type));
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
    std::vector<SerializedSearchContext> &verified_graphs,
    size_t search_depth,
    bool is_a_new_thread_start) {
  ++num_total_states;
  if (num_total_states % 100 == 1) {
    show_statistics();
  }
  if (verify(c)) {
    verified_graphs.push_back(SerializedSearchContext(c));
    return;
  }

  if (!is_a_new_thread_start && search_depth <= multithread_threshold_depth) {
    SearchContext c_copied = SerializedSearchContext(c).deserialize();
#pragma omp task
    {
      generate_next_operator(
          c_copied, verify, verified_graphs, search_depth, true);
    }
    return;
  }

  std::unordered_map<type::GuidType, std::shared_ptr<AbstractExpr const>>
      algebraic_expr;
  abstract_expr_eval(*c.kn_graph, algebraic_expr);
  if (c.tb_graph) {
    abstract_expr_eval(*c.tb_graph, algebraic_expr);
  }

  auto infer_and_check_abstract_expr = [&](auto const &input_tensors,
                                           auto op_type) {
    std::vector<std::shared_ptr<AbstractExpr const>> input_exprs =
        vector_map(input_tensors,
                   [&](auto const &t) { return algebraic_expr.at(t.guid); });
    std::shared_ptr<AbstractExpr const> expr =
        get_abstract_expr(op_type, input_tensors, input_exprs);
    return check_abstract_expr(expr);
  };

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
        for (auto const &input_idx :
             dim_strategy.get_input_cand_idx(op_type, all_tensors)) {
          if (!check_order(input_idx, op_type, *c.kn_graph)) {
            continue;
          }
          std::vector<DTensor> input_tensors = vector_map(
              input_idx, [&](int index) { return all_tensors[index]; });
          if (!infer_and_check_abstract_expr(input_tensors, op_type)) {
            continue;
          }

          KNOperator *old_last_op = c.kn_graph->operators.back();
          KNOperator *new_op = create_op(*c.kn_graph, op_type, input_tensors);
          if (new_op) {
            c.kn_graph->operators.push_back(new_op);
            generate_next_operator(
                c, verify, verified_graphs, search_depth + 1);
          }
          while (c.kn_graph->operators.back() != old_last_op) {
            delete c.kn_graph->operators.back();
            c.kn_graph->operators.pop_back();
          }
        }
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
          if (!check_order(input_tensor_idx, op_type, *c.kn_graph)) {
            continue;
          }
          std::vector<DTensor> input_tensors = vector_map(
              input_tensor_idx, [&](int index) { return all_tensors[index]; });
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
                                                      layout::SmemRowMajor,
                                                      false /*store_in_dmem*/);
                      if (input_op == nullptr) {
                        input_created = false;
                        break;
                      }
                      c.tb_graph->operators.push_back(input_op);
                    }
                    if (input_created) {
                      c.level = SearchLevel::LV_THREADBLOCK;
                      generate_next_operator(
                          c, verify, verified_graphs, search_depth + 1);
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
        assert(contains_key(algebraic_expr, stensor.guid));
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
          generate_next_operator(c, verify, verified_graphs, search_depth + 1);
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
      for (auto const &input_idx :
           dim_strategy.get_input_cand_idx(op_type, all_tensors)) {
        if (!check_order(input_idx, op_type, *c.tb_graph)) {
          continue;
        }
        std::vector<STensor> input_tensors = vector_map(
            input_idx, [&](int index) { return all_tensors[index]; });
        if (!infer_and_check_abstract_expr(input_tensors, op_type)) {
          continue;
        }

        TBOperator *old_last_op = c.tb_graph->operators.back();
        TBOperator *new_op = create_op(*c.tb_graph, op_type, input_tensors);

        if (new_op) {
          c.tb_graph->operators.push_back(new_op);
          generate_next_operator(c, verify, verified_graphs, search_depth + 1);
        }

        while (c.tb_graph->operators.back() != old_last_op) {
          delete c.tb_graph->operators.back();
          c.tb_graph->operators.pop_back();
        }
      }
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

  std::vector<SerializedSearchContext> verified_graphs;

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
          verified_graphs,
          /*search_depth=*/0,
          /*is_a_new_thread_start=*/true);
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

void KernelGraphGenerator::generate_kernel_graphs_symbolic() {
  start_time = std::chrono::steady_clock::now();
  std::shared_ptr<SymbolicKNGraph> kn_graph =
      std::make_shared<SymbolicKNGraph>();

  for (auto const &input_attr : computation_graph_input_attrs) {
    auto [dim, data_type, layout, strides] = input_attr;
    // FIXME: remove the layout attr since we use the strides
    // to describe the layout
    kn_graph->add_input(dim, strides);
  }

  generate_next_symbolic_operator(
      kn_graph, nullptr, {}, SearchLevel::LV_KERNEL, 0);
  std::cerr << num_symbolic_graphs << std::endl;
  printf("[Search] Symbolic search finished. Time elapsed: %fsec\n",
         std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       start_time)
             .count());
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

  std::unordered_map<type::GuidType, std::shared_ptr<AbstractExpr const>>
      computation_graph_exprs;
  abstract_expr_eval(computation_graph, computation_graph_exprs);

  init_ranges = get_init_ranges(computation_graph);
  target_ranges = get_interact_ranges(init_ranges, computation_graph);
  assert(init_ranges.size() == target_ranges.size());

  for (kernel::KNOperator *op : computation_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      computation_graph_output_exprs.push_back(
          computation_graph_exprs.at(op->input_tensors[0].guid));
    }
  }

  for (auto const &final_expr : computation_graph_output_exprs) {
    initialize_final_expr(final_expr);
  }

  if (config.verifier_type == VerifierType::PROBABILISTIC_VERIFIER) {
    this->verifier = std::make_shared<ProbabilisticVerifier>(computation_graph);
  } else {
    this->verifier = std::make_shared<FormalVerifier>(computation_graph);
  }
}

bool KernelGraphGenerator::check_abstract_expr(
    std::shared_ptr<AbstractExpr const> expr,
    TensorDimConstraints const &constraints) {

  if (!expr) {
    return false;
  }

  for (auto const &final_expr : computation_graph_output_exprs) {
    if (subexpr_to_final_expr(expr)) {
      return true;
    }
  }

  return false;
}

bool KernelGraphGenerator::verify(kernel::Graph &g) {
  std::vector<DTensor> outputs = get_output_tensors(g);

  if (outputs.size() != computation_graph_output_exprs.size()) {
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
  std::ofstream ofs(checkpoint_filename);
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

void KernelGraphGenerator::generate_next_symbolic_operator(
    std::shared_ptr<SymbolicKNGraph> kn_graph,
    std::shared_ptr<SymbolicTBGraph> tb_graph,
    std::vector<int> input_dtensor_indices_for_tb_graph,
    SearchLevel level,
    int search_depth) {

  if (0 < search_depth && search_depth <= multithread_threshold_depth) {
    std::shared_ptr<SymbolicKNGraph> kn_graph_copy =
        std::make_shared<SymbolicKNGraph>(*kn_graph);
    std::shared_ptr<SymbolicTBGraph> tb_graph_copy;
    if (tb_graph) {
      tb_graph_copy = std::make_shared<SymbolicTBGraph>(*tb_graph);
    }
    // #pragma omp task
    generate_next_symbolic_operator(kn_graph_copy,
                                    tb_graph_copy,
                                    input_dtensor_indices_for_tb_graph,
                                    level,
                                    -search_depth);
    return;
  }

  if (search_depth < 0) {
    search_depth = -search_depth;
  }

  if (level == SearchLevel::LV_KERNEL) {
    // In kernel-level search, tb_graph is always empty
    assert(tb_graph == nullptr);
    assert(input_dtensor_indices_for_tb_graph.empty());

    // Case K1: finish and verify the current graph
    if (instantiate_symbolic_graph(*kn_graph)) {
      return;
    }

    // Obtain abstract expressions
    std::vector<std::shared_ptr<AbstractExpr const>> abs_exprs;
    abstract_expr_eval(*kn_graph, abs_exprs);

    // Upper bound of the number of operators
    if (kn_graph->operators.size() >= config.max_num_kernel_graph_op) {
      return;
    }

    for (type::KNOperatorType op_type : dim_strategy.get_knop_cand()) {
      if (op_type != type::KNOperatorType::KN_CUSTOMIZED_OP) {
        // Case K2: generate a pre-defined kernel operator
        for (auto const &input_idx :
             dim_strategy.get_input_cand_idx(op_type, kn_graph->tensors)) {
          // Only generate operators in the increasing order of the order
          Order order(input_idx, static_cast<int>(op_type));
          if (order <= get_max_op_order(*kn_graph)) {
            continue;
          }

          // Input symbolic tensors
          std::vector<SymbolicDTensor> input_tensors = vector_map(
              input_idx, [&](int i) { return kn_graph->tensors[i]; });
          // Abstract expressions of input tensors
          std::vector<std::shared_ptr<AbstractExpr const>> input_exprs =
              vector_map(input_idx, [&](int i) { return abs_exprs[i]; });
          // Obtain the abstract expression of the output tensor
          std::shared_ptr<AbstractExpr const> expr =
              get_abstract_expr(op_type, input_tensors, input_exprs, *kn_graph);
          // Check if the abstract expression is a subexpression of the final
          // output
          if (!check_abstract_expr(expr)) {
            continue;
          }

          // Now we pass all the checks, create the operator
          if (kn_graph->add_operator(op_type, input_idx)) {
            // Recursively generate the next operator
            generate_next_symbolic_operator(kn_graph,
                                            tb_graph,
                                            input_dtensor_indices_for_tb_graph,
                                            level,
                                            search_depth + 1);
            // Revert the changes
            kn_graph->remove_last_operator();
          }
        }
      } else {
        // Case K3: generate a graph-def kernel operator
        if (count_op_of_type(type::KNOperatorType::KN_CUSTOMIZED_OP,
                             *kn_graph) >= config.max_num_threadblock_graphs) {
          continue;
        }

        for (auto const &input_tensor_idx :
             dim_strategy.get_customized_input_cand_idx(kn_graph->tensors)) {
          // Only generate operators in the increasing order of the order
          Order order(input_tensor_idx, static_cast<int>(op_type));
          if (order <= get_max_op_order(*kn_graph)) {
            continue;
          }

          // Input symbolic tensors
          std::vector<SymbolicDTensor> input_tensors = vector_map(
              input_tensor_idx, [&](int i) { return kn_graph->tensors[i]; });

          std::shared_ptr<SymbolicTBGraph> new_tb_graph =
              std::make_shared<SymbolicTBGraph>(
                  kn_graph->next_dim_variable_index);
          new_tb_graph->reduction_dimx = config.reduction_dimx;

          // Try to create input operators
          bool input_created = true;
          for (size_t i = 0; i < input_tensors.size(); ++i) {
            SymbolicDTensor dtensor = input_tensors[i];
            if (!new_tb_graph->add_input(dtensor)) {
              input_created = false;
              break;
            }
          }

          if (input_created) {
            // Recursively generate the next operator
            generate_next_symbolic_operator(kn_graph,
                                            new_tb_graph,
                                            input_tensor_idx,
                                            SearchLevel::LV_THREADBLOCK,
                                            search_depth + 1);
          }
        }
      }
    }
  } else {
    // threadblock-level search
    assert(tb_graph != nullptr);

    // Case B1: Create the customized operator and return to kernel-level search
    auto create_threadblock_outputs = [&](int3 output_map) {
      // Compute the nubmer of consumers for each tensor
      std::vector<int> num_consumers(tb_graph->tensors.size(), 0);
      for (auto const input_indices : tb_graph->input_indices) {
        for (int input_index : input_indices) {
          num_consumers[input_index]++;
        }
      }
      // Output tensors are the tensors without consumers
      std::vector<int> output_tensor_indices;
      for (size_t i = 0; i < tb_graph->tensors.size(); ++i) {
        if (num_consumers[i] == 0) {
          if (!tb_graph->tensors[i].after_accum) {
            return false;
          }
          output_tensor_indices.push_back(i);
        }
      }
      // Upper bound of the number of output tensors
      if (output_tensor_indices.size() >
          config.max_num_threadblock_graph_outputs) {
        return false;
      }

      // Add output operators
      for (int output_index : output_tensor_indices) {
        if (!tb_graph->add_output(
                output_index,
                output_map,
                -1,
                mirage::type::TBEpilogueType::TB_EPILOGUE_NONE)) {
          return false;
        }
      }

      return true;
    };

    for (int3 output_map : dim_strategy.get_output_map_cand(*tb_graph)) {
      if (create_threadblock_outputs(output_map)) {
        // Create the customized operator
        if (kn_graph->add_customized_operator(
                *tb_graph, input_dtensor_indices_for_tb_graph)) {
          // Recursively generate the next operator
          generate_next_symbolic_operator(
              kn_graph, nullptr, {}, SearchLevel::LV_KERNEL, search_depth + 1);
          // Revert the changes
          kn_graph->remove_last_operator();
        }
      }
      // Revert the changes
      while (tb_graph->operators.back().op_type ==
             type::TBOperatorType::TB_OUTPUT_OP) {
        tb_graph->remove_last_operator();
      }
    }

    // Upper bound of the number of operators
    if (tb_graph->operators.size() >= config.max_num_threadblock_graph_op) {
      return;
    }

    // Evaluate the abstract expressions
    std::vector<std::shared_ptr<AbstractExpr const>> abs_exprs;
    {
      std::vector<std::shared_ptr<AbstractExpr const>> kn_abs_exprs,
          input_exprs, output_exprs;
      abstract_expr_eval(*kn_graph, kn_abs_exprs);
      input_exprs = vector_map(input_dtensor_indices_for_tb_graph,
                               [&](int i) { return kn_abs_exprs[i]; });
      abstract_expr_eval(*tb_graph, input_exprs, abs_exprs, output_exprs);
    }

    // Case B2: Generate pre-defined threadblock operator
    for (type::TBOperatorType op_type : dim_strategy.get_tbop_cand()) {
      for (auto const &input_idx :
           dim_strategy.get_input_cand_idx(op_type, tb_graph->tensors)) {
        // Only generate operators in the increasing order of the order
        Order order(input_idx, static_cast<int>(op_type));
        if (order <= get_max_op_order(*tb_graph)) {
          continue;
        }

        // Input symbolic tensors
        std::vector<SymbolicSTensor> input_tensors =
            vector_map(input_idx, [&](int i) { return tb_graph->tensors[i]; });
        // Abstract expressions of input tensors
        std::vector<std::shared_ptr<AbstractExpr const>> input_exprs =
            vector_map(input_idx, [&](int i) { return abs_exprs[i]; });

        // Obtain the abstract expression of the output tensor
        std::shared_ptr<AbstractExpr const> expr =
            get_abstract_expr(op_type, input_tensors, input_exprs, *tb_graph);
        std::cerr << expr->to_string() << std::endl;
        // Check if the abstract expression is a subexpression of the final
        // output
        if (!check_abstract_expr(expr)) {
          continue;
        }

        // Now we pass all the checks, create the operator
        if (tb_graph->add_operator(op_type, input_idx)) {
          // Recursively generate the next operator
          generate_next_symbolic_operator(kn_graph,
                                          tb_graph,
                                          input_dtensor_indices_for_tb_graph,
                                          level,
                                          search_depth + 1);
          // Revert the changes
          tb_graph->remove_last_operator();
        }
      }
    }
  }
}

bool KernelGraphGenerator::instantiate_symbolic_graph(
    SymbolicKNGraph const &symbolic_graph) {
  {
    ++num_symbolic_graphs;
    std::cerr << "symbolic graph: " << json(symbolic_graph) << std::endl;
    return false;
  }

  // TODO: move to SearchConfig
  std::vector<int> grid_dim_cands = {1, 4, 8, 16, 32, 64, 128};
  std::vector<int> frange_cands = {1, 2, 4, 8, 16, 32, 64};

  auto get_var_index = [](SymbolicTensorDim const &dim) {
    assert(dim.dim_expr->is_var());
    std::shared_ptr<TensorDimVar const> var =
        std::static_pointer_cast<TensorDimVar const>(dim.dim_expr);
    return var->index;
  };

  auto combine_candidate_assignments =
      [](std::vector<std::vector<DimVarAssignments>> const &assignment_sets) {
        std::vector<DimVarAssignments> combined_assignments =
            assignment_sets[0];
        for (size_t i = 1; i < assignment_sets.size(); ++i) {
          std::vector<DimVarAssignments> new_combined_assignments;
          for (DimVarAssignments const &assignments : combined_assignments) {
            for (DimVarAssignments const &new_assignments :
                 assignment_sets[i]) {
              new_combined_assignments.push_back(
                  combine_assignments(assignments, new_assignments));
            }
          }
          combined_assignments = new_combined_assignments;
        }
        return combined_assignments;
      };

  auto get_tb_graph_assignments = [&](SymbolicTBGraph const &tb_graph) {
    std::vector<DimVarAssignments> candidate_assignments;
    for (int grid_dim_x : grid_dim_cands) {
      for (int grid_dim_y : grid_dim_cands) {
        for (int grid_dim_z : grid_dim_cands) {
          for (int frange : frange_cands) {
            DimVarAssignments assignments;
            assignments.assign(get_var_index(tb_graph.grid_dim[0]), grid_dim_x);
            assignments.assign(get_var_index(tb_graph.grid_dim[1]), grid_dim_y);
            assignments.assign(get_var_index(tb_graph.grid_dim[2]), grid_dim_z);
            assignments.assign(get_var_index(tb_graph.forloop_range), frange);
            candidate_assignments.push_back(assignments);
          }
        }
      }
    }
    return candidate_assignments;
  };

  auto get_kn_graph_assignments = [&](SymbolicKNGraph const &kn_graph) {
    std::vector<std::vector<DimVarAssignments>> assignment_sets;
    for (SymbolicKNOp const &op : kn_graph.operators) {
      if (op.op_type == type::KNOperatorType::KN_CUSTOMIZED_OP) {
        std::shared_ptr<KNCustomizedOpArgs const> args =
            std::static_pointer_cast<KNCustomizedOpArgs const>(op.args);
        std::vector<DimVarAssignments> tb_graph_assignments =
            get_tb_graph_assignments(args->tb_graph_template);
        assignment_sets.push_back(tb_graph_assignments);
      }
    }
    return combine_candidate_assignments(assignment_sets);
  };

  bool at_least_one_valid = false;
  for (DimVarAssignments const &assignments :
       get_kn_graph_assignments(symbolic_graph)) {
    kernel::Graph *instantiated_graph =
        symbolic_graph.to_kernel_graph(assignments);
    if (verify(*instantiated_graph)) {
      at_least_one_valid = true;
      this->generated_graphs.push_back(json(*instantiated_graph));
    }
  }

  return at_least_one_valid;
}

} // namespace search
} // namespace mirage
