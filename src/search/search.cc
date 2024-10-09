#include "mirage/search/search.h"
#include "mirage/kernel/customized.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/search/dim_strategy.h"
#include "mirage/search/op_utils.h"
#include "mirage/search/pattern_eval.h"
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
      num_thread(config.search_thread), verbose(verbose),
      num_total_random_tests(0), num_valid_kernel_graphs(0),
      num_total_states(0) {
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

template <typename GraphType>
int get_num_consumers(GraphType const &g,
                      typename GraphType::TensorType const &tensor) {
  int num_consumers = 0;
  for (auto const &op : g.operators) {
    for (auto const &t : op->input_tensors) {
      if (t.guid == tensor.guid) {
        num_consumers++;
      }
    }
  }
  return num_consumers;
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
    std::vector<SerializedSearchContext> &verified) {
  ++num_total_states;
  if (num_total_states % 100 == 1) {
    show_statistics();
  }
  if (verify(c)) {
    verified.push_back(SerializedSearchContext(c));
    return;
  }

  std::unordered_map<int64_t, std::shared_ptr<AlgebraicPattern>>
      algebraic_pattern;
  pattern_eval(*c.kn_graph, algebraic_pattern);
  if (c.tb_graph) {
    pattern_eval(*c.tb_graph, algebraic_pattern);
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
        for (auto const &input_idx :
             dim_strategy.get_input_cand_idx(op_type, all_tensors)) {
          Order order(input_idx, static_cast<int>(op_type));
          if (order <= get_max_op_order(*c.kn_graph)) {
            continue;
          }
          std::vector<DTensor> input_tensors =
              get_tensors_from_idx(*c.kn_graph, input_idx);
          std::vector<std::shared_ptr<AlgebraicPattern>> input_patterns;
          for (auto const &t : input_tensors) {
            assert(contains_key(algebraic_pattern, t.guid));
            input_patterns.push_back(algebraic_pattern.at(t.guid));
          }
          std::shared_ptr<AlgebraicPattern> pattern =
              get_pattern(op_type, input_tensors, input_patterns);
          if (!check_pattern(pattern)) {
            continue;
          }

          KNOperator *new_op = create_op(*c.kn_graph, op_type, input_tensors);
          if (new_op) {
            c.kn_graph->operators.push_back(new_op);
            if (check_range(init_ranges, target_ranges, *c.kn_graph)) {
              generate_next_operator(c, verify, verified);
            }
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
                      generate_next_operator(c, verify, verified);
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

    // Case B1. Finish and return to kernel-level search
    for (int3 output_map :
         dim_strategy.get_output_map_cand(c.tb_graph->grid_dim)) {
      if (create_threadblock_outputs(c, algebraic_pattern, output_map)) {
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
          generate_next_operator(c, verify, verified);
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
      for (auto const &input_idx :
           dim_strategy.get_input_cand_idx(op_type, all_tensors)) {
        Order order(input_idx, static_cast<int>(op_type));
        if (order <= get_max_op_order(*c.tb_graph)) {
          continue;
        }
        std::vector<STensor> input_tensors =
            get_tensors_from_idx(*c.tb_graph, input_idx);
        std::vector<std::shared_ptr<AlgebraicPattern>> input_patterns;
        for (auto const &t : input_tensors) {
          assert(contains_key(algebraic_pattern, t.guid));
          input_patterns.push_back(algebraic_pattern.at(t.guid));
        }
        std::shared_ptr<AlgebraicPattern> pattern =
            get_pattern(op_type, input_tensors, input_patterns);
        if (!check_pattern(pattern)) {
          continue;
        }

        TBOperator *new_op = create_op(*c.tb_graph, op_type, input_tensors);
        if (!new_op) {
          continue;
        };
        c.tb_graph->operators.push_back(new_op);
        if (check_range(init_ranges, target_ranges, *c.kn_graph, c.tb_graph)) {
          generate_next_operator(c, verify, verified);
        }
        delete c.tb_graph->operators.back();
        c.tb_graph->operators.pop_back();
      }
    }
  }
}

bool KernelGraphGenerator::create_threadblock_outputs(
    SearchContext &c,
    std::unordered_map<int64_t, std::shared_ptr<AlgebraicPattern>> const
        &algebraic_pattern,
    int3 output_map) {
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
}

void KernelGraphGenerator::search_from(
    std::vector<SerializedSearchContext> const &contexts) {
  for (auto const &sc : contexts) {
    SearchContext c = sc.deserialize();
    std::vector<SerializedSearchContext> verified;
    generate_next_operator(
        c,
        [this](SearchContext const &c) {
          return c.level == SearchLevel::LV_KERNEL && this->verify(*c.kn_graph);
        },
        verified);
    {
      std::lock_guard<std::mutex> lock(generated_graphs_mutex);
      for (auto const &v : verified) {
        generated_graphs.push_back(json(*v.deserialize().kn_graph));
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
    auto [dim, data_type, layout] = input_attr;
    // FIXME: strides should be an additional attr for KNInputOp
    // Currently use the default strided layout
    // FIXME: remove the layout attr since we use the strides
    // to describe the layout
    std::vector<size_t> strides;
    int num_elements = 1;
    for (size_t i = 0; i < dim.size(); i++) {
      strides.push_back(num_elements);
      num_elements *= dim[dim.size() - 1 - i];
    }
    std::reverse(strides.begin(), strides.end());
    c.kn_graph->new_input(dim, strides, data_type, layout);
  }

  std::vector<SerializedSearchContext> middle_states;
  generate_next_operator(
      c,
      [this](SearchContext const &c) {
        return c.tb_graph != nullptr || this->verify(*c.kn_graph);
      },
      middle_states);
  printf("\n");
  printf("[Search] First step finished. Time elapsed: %lfsec\n",
         get_elapsed_time_in_sec());
  std::vector<std::vector<SerializedSearchContext>> split_middle_states(
      num_thread);
  for (size_t i = 0; i < middle_states.size(); ++i) {
    split_middle_states[i % num_thread].push_back(middle_states[i]);
  }
  std::vector<std::thread> threads;
  for (int i = 0; i < num_thread; ++i) {
    threads.push_back(std::thread(
        &KernelGraphGenerator::search_from, this, split_middle_states[i]));
  }
  for (auto &thread : threads) {
    thread.join();
  }

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
           op->output_tensors[0].layout});
    }
  }

  std::unordered_map<int64_t, std::shared_ptr<AlgebraicPattern>>
      computation_graph_patterns;
  pattern_eval(computation_graph, computation_graph_patterns);

  init_ranges = get_init_ranges(computation_graph);
  target_ranges = get_interact_ranges(init_ranges, computation_graph);
  assert(init_ranges.size() == target_ranges.size());

  for (auto const &op : computation_graph.operators) {
    op->fingerprint();
  }

  std::unordered_map<DTensor, int> num_consumers;
  for (kernel::KNOperator *op : computation_graph.operators) {
    for (DTensor const &input : op->input_tensors) {
      num_consumers[input]++;
    }
  }

  for (kernel::KNOperator *op : computation_graph.operators) {
    for (DTensor const &output : op->output_tensors) {
      if (num_consumers[output] == 0) {
        computation_graph_output_tensors.push_back(
            output.copy_fingerprint_to_ctensor());
        computation_graph_output_patterns.push_back(
            computation_graph_patterns.at(output.guid));
      }
    }
  }
}

bool KernelGraphGenerator::check_pattern(
    std::shared_ptr<AlgebraicPattern> pattern) {
  if (!pattern) {
    return false;
  }
  for (auto const &final_pattern : computation_graph_output_patterns) {
    if (pattern->subpattern_to(*final_pattern)) {
      return true;
    }
  }
  return false;
}

bool KernelGraphGenerator::verify(kernel::Graph const &g) {
  std::vector<DTensor> outputs = get_output_tensors(g);

  if (outputs.size() != computation_graph_output_patterns.size()) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(fp_mutex);

    ++num_total_random_tests;

    for (auto const &op : g.operators) {
      op->fingerprint();
    }

    auto get_matches = [](int num_outputs) {
      std::vector<std::vector<int>> results;
      std::vector<int> perm;
      for (int i = 0; i < num_outputs; ++i) {
        perm.push_back(i);
      }
      do {
        results.push_back(perm);
      } while (std::next_permutation(perm.begin(), perm.end()));
      return results;
    };

    for (auto const &match : get_matches(outputs.size())) {
      if (have_same_fingerprint(outputs, match)) {
        ++num_valid_kernel_graphs;
        return true;
      }
    }
  }

  return false;
}

bool KernelGraphGenerator::have_same_fingerprint(
    std::vector<DTensor> const &outputs, std::vector<int> const &match) const {
  assert(outputs.size() == match.size());
  for (int i = 0; i < static_cast<int>(match.size()); ++i) {
    if (!outputs[match[i]].has_same_fingerprint(
            computation_graph_output_tensors[i])) {
      return false;
    }
  }
  return true;
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
