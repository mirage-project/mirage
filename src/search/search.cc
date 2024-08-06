#include "mirage/search/search.h"
#include "mirage/kernel/customized.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/search/dim_strategy.h"
#include "mirage/search/op_utils.h"
#include "mirage/utils/containers.h"
#include "mirage/utils/json_utils.h"

#include <fstream>
#include <iostream>
#include <thread>

namespace mirage {
namespace search {

void pattern_eval(
    threadblock::Graph const &g,
    std::unordered_map<int64_t, std::shared_ptr<AlgebraicPattern>> &patterns) {
  for (TBOperator *op : g.operators) {
    if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
      patterns.insert(
          {op->output_tensors[0].guid,
           patterns.at(
               static_cast<threadblock::TBInputOp *>(op)->dtensor.guid)});
    } else if (op->op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      patterns.insert(
          {static_cast<threadblock::TBOutputOp *>(op)->dtensor.guid,
           g.forloop_range > 1
               ? std::make_shared<Red>(g.forloop_range,
                                       patterns.at(op->input_tensors[0].guid))
               : patterns.at(op->input_tensors[0].guid)});
    } else {
      std::vector<std::shared_ptr<AlgebraicPattern>> input_patterns;
      for (STensor const &input_tensor : op->input_tensors) {
        input_patterns.push_back(patterns.at(input_tensor.guid));
      }
      patterns.insert(
          {op->output_tensors[0].guid,
           get_pattern(op->op_type, op->input_tensors, input_patterns)});
    }
  }
}

void pattern_eval(
    kernel::Graph const &g,
    std::unordered_map<int64_t, std::shared_ptr<AlgebraicPattern>> &patterns) {
  int input_id = 0;
  for (KNOperator *op : g.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      patterns.insert({op->output_tensors[0].guid,
                       std::make_shared<Var>("v_" + std::to_string(input_id))});
      input_id++;
    } else if (op->op_type != type::KNOperatorType::KN_CUSTOMIZED_OP) {
      std::vector<std::shared_ptr<AlgebraicPattern>> input_patterns;
      for (DTensor const &input_tensor : op->input_tensors) {
        assert(contains_key(patterns, input_tensor.guid));
        input_patterns.push_back(patterns.at(input_tensor.guid));
      }
      patterns.insert(
          {op->output_tensors[0].guid,
           get_pattern(op->op_type, op->input_tensors, input_patterns)});
    } else {
      assert(op->op_type == type::KNOperatorType::KN_CUSTOMIZED_OP);
      pattern_eval(static_cast<kernel::KNCustomizedOp *>(op)->bgraph, patterns);
    }
  }
}

SearchContext::SearchContext()
    : kn_graph(nullptr), tb_graph(nullptr), level(SearchLevel::LV_KERNEL) {}

SearchContext::~SearchContext() {}

SearchContext SearchContext::copy() const {
  SearchContext c;
  from_json(json(*this), c);
  return c;
}

void to_json(json &j, SearchContext const &c) {
  j["kn_graph"] = json(*c.kn_graph);
  std::vector<std::pair<size_t, size_t>> inputs;
  if (c.tb_graph) {
    threadblock::ExecutionPlan plan = c.tb_graph->get_plan();
    j["tb_plan"] = json(plan);
    for (auto const &op : c.tb_graph->operators) {
      if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
        for (size_t i = 0; i < c.kn_graph->operators.size(); ++i) {
          for (size_t j = 0;
               j < c.kn_graph->operators[i]->output_tensors.size();
               ++j) {
            if (c.kn_graph->operators[i]->output_tensors[j].guid ==
                static_cast<threadblock::TBInputOp *>(op)->dtensor.guid) {
              inputs.push_back({i, j});
              break;
            }
          }
        }
      }
    }
    assert(plan.input_map.size() == inputs.size());
  }
  j["inputs"] = inputs;
  j["level"] = c.level;
}

void from_json(json const &j, SearchContext &c) {
  c.kn_graph = std::make_shared<kernel::Graph>();
  from_json(j["kn_graph"], *c.kn_graph);
  std::vector<std::pair<size_t, size_t>> inputs;
  from_json(j["inputs"], inputs);
  if (inputs.size()) {
    std::vector<DTensor> input_tensors;
    threadblock::ExecutionPlan plan;
    from_json(j["tb_plan"], plan);
    for (auto const &id : inputs) {
      input_tensors.push_back(
          c.kn_graph->operators[id.first]->output_tensors[id.second]);
    }
    c.tb_graph = std::make_shared<threadblock::Graph>(input_tensors, plan);
  }
  c.level = j["level"];
}

KernelGraphGenerator::KernelGraphGenerator(
    kernel::Graph const &computation_graph,
    GeneratorConfig const &config,
    char const *filename)
    : best_profile_result(ProfileResult::infinity()), config(config),
      dim_strategy(DimStrategy(config)), filename(filename),
      num_thread(std::min((int)std::thread::hardware_concurrency(),
                          MAX_SEARCH_THREAD)),
      timeout(1000), num_total_kernel_graphs(0), num_total_random_tests(0),
      num_valid_kernel_graphs(0), num_total_states(0) {
  preprocess(computation_graph);
}

int count_op(type::KNOperatorType op_type, kernel::Graph const &g) {
  int counter = 0;
  for (auto const &op : g.operators) {
    if (op->op_type == op_type) {
      ++counter;
    }
  }
  return counter;
}

std::vector<std::vector<int>> get_matches(int num_outputs) {
  std::vector<std::vector<int>> results;
  std::vector<int> perm;
  for (int i = 0; i < num_outputs; ++i) {
    perm.push_back(i);
  }
  do {
    results.push_back(perm);
  } while (std::next_permutation(perm.begin(), perm.end()));
  return results;
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

void KernelGraphGenerator::enqueue(SearchContext const &c) {
  std::lock_guard<std::mutex> lock(queue_mutex);
  search_queue.push(c);
  queue_cv.notify_one();
}

bool KernelGraphGenerator::dequeue(SearchContext &c) {
  std::unique_lock<std::mutex> lock(queue_mutex);
  if (!queue_cv.wait_for(
          lock, timeout, [&] { return !search_queue.empty(); })) {
    return false;
  }
  c = search_queue.front();
  search_queue.pop();
  return true;
}

void KernelGraphGenerator::generate_next_operator(SearchContext const &c) {
  ++num_total_states;
  if (num_total_states % 1000 == 1) {
    printf("Total states explored: %d.\n", num_total_states.load());
    printf("Search queue size: %lu\n", search_queue.size());
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
    if (verify(c)) {
      std::lock_guard<std::mutex> lock(generated_graphs_mutex);
      generated_graphs.push_back(json(*c.kn_graph));
      return;
    }
    if (c.kn_graph->operators.size() >= MAX_NUM_KERNEL_GRAPH_OP) {
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
          {
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
          }

          SearchContext nc = c.copy();
          KNOperator *new_op =
              create_op(*nc.kn_graph,
                        op_type,
                        get_tensors_from_idx(*nc.kn_graph, input_idx));
          if (new_op) {
            nc.kn_graph->operators.push_back(new_op);
            enqueue(nc);
          }
        }
      } else {
        // Case K3: generate a graph-def kernel operator
        if (count_op(type::KNOperatorType::KN_CUSTOMIZED_OP, *c.kn_graph) >=
            MAX_NUM_THREADBLOCK) {
          continue;
        }
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
                    SearchContext nc = c.copy();
                    assert(nc.tb_graph == nullptr);
                    nc.tb_graph = std::make_shared<threadblock::Graph>(
                        grid_dim,
                        block_dim,
                        forloop_range,
                        config.reduction_dimx);
                    bool input_created = true;
                    std::vector<DTensor> all_new_tensors =
                        get_all_tensors(*nc.kn_graph);
                    for (size_t i = 0; i < input_tensor_idx.size(); ++i) {
                      DTensor dtensor = all_new_tensors[input_tensor_idx[i]];
                      TBOperator *input_op =
                          nc.tb_graph->create_input_op(dtensor,
                                                       input_map[i],
                                                       forloop_dim[i],
                                                       layout::SmemRowMajor);
                      if (input_op == nullptr) {
                        input_created = false;
                        break;
                      }
                      nc.tb_graph->operators.push_back(input_op);
                    }
                    if (input_created) {
                      nc.level = SearchLevel::LV_THREADBLOCK;
                      enqueue(nc);
                    }
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
      SearchContext nc = c.copy();
      std::unordered_map<int64_t, std::shared_ptr<AlgebraicPattern>>
          new_algebraic_pattern;
      pattern_eval(*nc.kn_graph, new_algebraic_pattern);
      pattern_eval(*nc.tb_graph, new_algebraic_pattern);
      if (create_threadblock_outputs(nc, new_algebraic_pattern, output_map)) {
        KNOperator *new_op = nc.kn_graph->create_customized_op(
            get_input_tensors(*nc.tb_graph), *nc.tb_graph);
        if (!new_op) {
          continue;
        }
        nc.kn_graph->operators.push_back(new_op);
        assert(nc.kn_graph->operators.back()->kgraph->gpu_dim.x == 1);
        nc.level = SearchLevel::LV_KERNEL;
        nc.tb_graph = nullptr;
        enqueue(nc);
      }
    }

    if (c.tb_graph->operators.size() >= MAX_NUM_THREADBLOCK_GRAPH_OP) {
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
        {
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
        }

        SearchContext nc = c.copy();

        TBOperator *new_op =
            create_op(*nc.tb_graph,
                      op_type,
                      get_tensors_from_idx(*nc.tb_graph, input_idx));
        if (!new_op) {
          continue;
        }
        nc.tb_graph->operators.push_back(new_op);
        enqueue(nc);
      }
    }
  }
}

void KernelGraphGenerator::launch_thread() {
  SearchContext c;
  while (dequeue(c)) {
    generate_next_operator(c);
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
        output_tensors.push_back(tensor);
      }
    }
  }

  if (output_tensors.size() > MAX_NUM_THREADBLOCK_OUTPUT) {
    return false;
  }

  for (STensor const &stensor : output_tensors) {
    assert(contains_key(algebraic_pattern, stensor.guid));
    std::shared_ptr<AlgebraicPattern> pattern =
        c.tb_graph->forloop_range > 1
            ? std::make_shared<Red>(c.tb_graph->forloop_range,
                                    algebraic_pattern.at(stensor.guid))
            : algebraic_pattern.at(stensor.guid);
    if (!check_pattern(pattern)) {
      return false;
    }
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

void KernelGraphGenerator::generate_kernel_graphs() {
  printf("num_thread: %d\n", num_thread);

  SearchContext c;
  c.kn_graph = std::make_shared<kernel::Graph>();

  for (auto const &input_attr : computation_graph_input_attrs) {
    auto [dim, data_type, layout] = input_attr;
    c.kn_graph->new_input(dim, data_type, layout);
  }

  enqueue(c);

  std::vector<std::thread> threads;
  for (int i = 0; i < num_thread; ++i) {
    threads.push_back(std::thread(&KernelGraphGenerator::launch_thread, this));
  }
  for (int i = 0; i < num_thread; ++i) {
    threads[i].join();
  }

  save_results();

  printf("Total kernel graphs explored: %d\n", num_total_kernel_graphs.load());
  printf("Random tests performed: %d\n", num_total_random_tests.load());
  printf("Valid kernel graphs explored: %d\n", num_valid_kernel_graphs.load());
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

bool KernelGraphGenerator::verify(SearchContext c) {
  ++num_total_kernel_graphs;
  if (num_total_kernel_graphs % 1000 == 1) {
    printf("Total kernel graphs explored: %d.\n",
           num_total_kernel_graphs.load());
    // if (num_total_kernel_graphs % 10000 == 1) {
    //   save_checkpoint();
    // }
  }

  std::vector<DTensor> outputs = get_output_tensors(*c.kn_graph);

  if (outputs.size() != computation_graph_output_patterns.size()) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(fp_mutex);

    ++num_total_random_tests;
    assert(c.kn_graph->gpu_dim.x == 1);

    for (auto const &op : c.kn_graph->operators) {
      op->fingerprint();
    }

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

} // namespace search
} // namespace mirage
