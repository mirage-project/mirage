#include "mirage/search/search.h"
#include "mirage/kernel/customized.h"
#include "mirage/search/dim_strategy.h"
#include "mirage/search/op_utils.h"
#include "mirage/utils/containers.h"

#include <fstream>
#include <iostream>

namespace mirage {
namespace search {

KernelGraphGenerator::KernelGraphGenerator(
    kernel::Graph const &computation_graph,
    GeneratorConfig const &config,
    char const *filename)
    : computation_graph(computation_graph),
      best_profile_result(ProfileResult::infinity()), config(config),
      dim_strategy(DimStrategy(config)), filename(filename),
      num_total_kernel_graphs(0), num_total_random_tests(0),
      num_valid_kernel_graphs(0) {}

KernelGraphGenerator::KernelGraphGenerator(char const *filename)
    : filename(filename) {
  std::ifstream ifs(filename);
  json j;
  ifs >> j;
  Checkpoint checkpoint;
  from_json(j, checkpoint);
  recovery_from_checkpoint(checkpoint);
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

std::vector<int> get_open_tensor_idx(SearchContext<DTensor> &c,
                                     kernel::Graph const &g) {
  std::vector<int> result;
  for (size_t i = 0; i < c.all_tensors.size(); ++i) {
    if (c.num_consumers[i] == 0) {
      result.push_back(i);
    }
  }
  return result;
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

bool KernelGraphGenerator::create_tb_outputs(SearchContext<STensor> &c,
                                             threadblock::Graph &g,
                                             int3 output_map) {

  assert(c.output_pattern.empty());

  for (size_t i = 0; i < c.all_tensors.size(); ++i) {
    if (c.num_consumers[i] == 0) {
      if (c.all_tensors[i].owner_op->op_type ==
          type::TBOperatorType::TB_INPUT_OP) {
        return false;
      }
      STensor input = c.all_tensors[i];
      std::shared_ptr<AlgebraicPattern> pattern =
          g.forloop_range > 1
              ? std::make_shared<Red>(g.forloop_range, c.algebraic_pattern[i])
              : c.algebraic_pattern[i];
      if (!check_pattern(pattern)) {
        return false;
      }
      TBOperator *new_op = g.create_output_op(input, output_map);
      if (!new_op) {
        return false;
      }
      g.operators.push_back(new_op);
      c.output_pattern.push_back(pattern);
    }
  }
  return true;
}

void KernelGraphGenerator::generate_next_tb_operator(
    SearchContext<STensor> &c,
    threadblock::Graph &g,
    std::function<void(int)> const &create_customized_then_next_kn,
    int depth) {

  if (depth >= (int)callstack.size()) {
    callstack.push_back(LayerCheckpoint{});
  }
  assert(!callstack.empty());

  // Finish threadblock graph search and continue to search the next kernel
  // operator
  for (int3 output_map : dim_strategy.get_output_map_cand(g.grid_dim)) {
    if (contains(callstack.back().output_map_explored, output_map)) {
      continue;
    }
    if (create_tb_outputs(c, g, output_map)) {
      create_customized_then_next_kn(depth);
    }
    while (!c.output_pattern.empty()) {
      c.output_pattern.pop_back();
      delete g.operators.back();
      g.operators.pop_back();
    }
    callstack.back().output_map_explored.insert(output_map);
  }

  if (g.operators.size() >= MAX_NUM_THREADBLOCK_GRAPH_OP) {
    callstack.pop_back();
    return;
  }

  for (type::TBOperatorType op_type : config.tbop_to_explore) {
    if (contains(callstack.back().tbop_explored, op_type)) {
      continue;
    }
    callstack.back().input_idx_explored.clear();
    for (auto const &input_idx :
         dim_strategy.get_input_cand_idx(op_type, c.all_tensors)) {
      if (contains(callstack.back().input_idx_explored, input_idx)) {
        continue;
      }
      Order order(input_idx, static_cast<int>(op_type));
      if (order <= c.op_order.back()) {
        callstack.back().input_idx_explored.insert(input_idx);
        continue;
      }
      std::vector<STensor> input_tensors;
      std::vector<std::shared_ptr<AlgebraicPattern>> input_patterns;
      for (int i : input_idx) {
        input_tensors.push_back(c.all_tensors[i]);
        input_patterns.push_back(c.algebraic_pattern[i]);
      }
      std::shared_ptr<AlgebraicPattern> pattern =
          get_pattern(op_type, input_tensors, input_patterns);
      if (!check_pattern(pattern)) {
        callstack.back().input_idx_explored.insert(input_idx);
        continue;
      }

      threadblock::TBOperator *new_op = create_op(g, op_type, input_tensors);

      if (!new_op) {
        callstack.back().input_idx_explored.insert(input_idx);
        continue;
      }

      STensor output = new_op->output_tensors[0];

      g.operators.push_back(new_op);
      c.all_tensors.push_back(output);
      c.algebraic_pattern.push_back(pattern);
      c.num_consumers.push_back(0);
      for (int i : input_idx) {
        c.num_consumers[i]++;
      }
      c.op_order.push_back(order);
      generate_next_tb_operator(
          c, g, create_customized_then_next_kn, depth + 1);
      c.op_order.pop_back();
      for (int i : input_idx) {
        c.num_consumers[i]--;
      }
      c.num_consumers.pop_back();
      c.algebraic_pattern.pop_back();
      c.all_tensors.pop_back();
      assert(g.operators.back() == new_op);
      delete g.operators.back();
      g.operators.pop_back();
      if (op_type == type::TBOperatorType::TB_CONCAT_THEN_MATMUL_OP) {
        assert(g.operators.back()->op_type ==
               type::TB_CONCAT_FIRST_OP_ID + output.num_dims - 2);
        delete g.operators.back();
        g.operators.pop_back();
        assert(g.operators.back()->op_type ==
               type::TB_CONCAT_FIRST_OP_ID + output.num_dims - 1);
        delete g.operators.back();
        g.operators.pop_back();
      }
      callstack.back().input_idx_explored.insert(input_idx);
    }
    callstack.back().tbop_explored.insert(op_type);
  }

  callstack.pop_back();
}

void KernelGraphGenerator::generate_next_kn_operator(SearchContext<DTensor> &c,
                                                     kernel::Graph &g,
                                                     int depth) {
  if (depth >= (int)callstack.size()) {
    callstack.push_back(LayerCheckpoint{});
  }

  if (verify(c, g)) {
    ++num_valid_kernel_graphs;
    // std::ofstream ofs;
    // ofs.open("generated_graphs.txt", std::ofstream::out |
    // std::ofstream::app); ofs << json(g) << std::endl; ofs.close();
    generated_graphs.push_back(json(g));
    // update_best_graph(g);
    std::cerr << "kernel graph candidate: " << json(g) << std::endl;
    callstack.pop_back();
    return;
  }

  if (g.operators.size() >= MAX_NUM_KERNEL_GRAPH_OP) {
    callstack.pop_back();
    return;
  }

  for (type::KNOperatorType op_type : config.knop_to_explore) {
    if (contains(callstack.back().knop_explored, op_type)) {
      continue;
    }
    callstack.back().input_idx_explored.clear();
    if (op_type != type::KNOperatorType::KN_CUSTOMIZED_OP) {
      for (auto const &input_idx :
           dim_strategy.get_input_cand_idx(op_type, c.all_tensors)) {
        if (contains(callstack.back().input_idx_explored, input_idx)) {
          continue;
        }
        Order order(input_idx, static_cast<int>(op_type));
        if (order <= c.op_order.back()) {
          callstack.back().input_idx_explored.insert(input_idx);
          continue;
        }
        std::vector<DTensor> input_tensors;
        std::vector<std::shared_ptr<AlgebraicPattern>> input_patterns;
        for (int i : input_idx) {
          input_tensors.push_back(c.all_tensors[i]);
          input_patterns.push_back(c.algebraic_pattern[i]);
        }
        std::shared_ptr<AlgebraicPattern> pattern =
            get_pattern(op_type, input_tensors, input_patterns);
        if (!check_pattern(pattern)) {
          callstack.back().input_idx_explored.insert(input_idx);
          continue;
        }
        KNOperator *new_op = create_op(g, op_type, input_tensors);
        if (!new_op) {
          callstack.back().input_idx_explored.insert(input_idx);
          continue;
        }
        DTensor output = new_op->output_tensors[0];

        g.operators.push_back(new_op);
        c.all_tensors.push_back(output);
        c.algebraic_pattern.push_back(pattern);
        c.num_consumers.push_back(0);
        for (int i : input_idx) {
          c.num_consumers[i]++;
        }
        c.op_order.push_back(order);
        generate_next_kn_operator(c, g, depth + 1);
        c.op_order.pop_back();
        for (int i : input_idx) {
          c.num_consumers[i]--;
        }
        c.num_consumers.pop_back();
        c.algebraic_pattern.pop_back();
        c.all_tensors.pop_back();
        assert(g.operators.back() == new_op);
        delete g.operators.back();
        g.operators.pop_back();
        callstack.back().input_idx_explored.insert(input_idx);
      }
    } else {
      if (count_op(type::KNOperatorType::KN_CUSTOMIZED_OP, g) >=
          MAX_NUM_THREADBLOCK) {
        callstack.back().knop_explored.insert(op_type);
        continue;
      }

      for (auto const &input_tensor_idx :
           dim_strategy.get_customized_input_cand_idx(
               c.all_tensors, get_open_tensor_idx(c, g))) {
        if (contains(callstack.back().input_idx_explored, input_tensor_idx)) {
          continue;
        }
        std::vector<DTensor> input_tensors;
        for (int i : input_tensor_idx) {
          input_tensors.push_back(c.all_tensors[i]);
        }
        Order order(input_tensor_idx, static_cast<int>(op_type));
        if (order <= c.op_order.back()) {
          callstack.back().input_idx_explored.insert(input_tensor_idx);
          continue;
        }
        callstack.back().grid_dim_explored.clear();
        for (dim3 grid_dim : dim_strategy.get_grid_dim_cand(input_tensors)) {
          if (contains(callstack.back().grid_dim_explored, grid_dim)) {
            continue;
          }
          callstack.back().block_dim_explored.clear();
          for (dim3 block_dim :
               dim_strategy.get_block_dim_cand(input_tensors, grid_dim)) {
            if (contains(callstack.back().block_dim_explored, block_dim)) {
              continue;
            }
            callstack.back().input_map_explored.clear();
            for (std::vector<int3> const &input_map :
                 dim_strategy.get_input_map_cand(input_tensors, grid_dim)) {
              if (contains(callstack.back().input_map_explored, input_map)) {
                continue;
              }
              callstack.back().forloop_dim_explored.clear();
              for (std::vector<int> const &forloop_dim :
                   dim_strategy.get_forloop_dim_cand(input_tensors)) {
                if (contains(callstack.back().forloop_dim_explored,
                             forloop_dim)) {
                  continue;
                }
                callstack.back().forloop_range_explored.clear();
                for (int forloop_range :
                     dim_strategy.get_forloop_range_cand(input_tensors,
                                                         input_map,
                                                         grid_dim,
                                                         block_dim,
                                                         forloop_dim)) {
                  if (contains(callstack.back().forloop_range_explored,
                               forloop_range)) {
                    continue;
                  }
                  SearchContext<STensor> tb_context;
                  threadblock::Graph tb_graph(grid_dim,
                                              block_dim,
                                              forloop_range,
                                              config.reduction_dimx);

                  bool input_created = true;
                  for (size_t i = 0; i < input_tensors.size(); ++i) {
                    DTensor tensor = input_tensors[i];
                    TBOperator *input_op =
                        tb_graph.create_input_op(tensor,
                                                 input_map[i],
                                                 forloop_dim[i],
                                                 layout::SmemRowMajor);
                    if (input_op == nullptr) {
                      input_created = false;
                      break;
                    }
                    tb_graph.operators.push_back(input_op);
                    STensor output = input_op->output_tensors[0];
                    tb_context.all_tensors.push_back(output);
                    tb_context.algebraic_pattern.push_back(
                        c.algebraic_pattern[input_tensor_idx[i]]);
                    tb_context.num_consumers.push_back(0);
                    tb_context.op_order.push_back(Order(
                        {},
                        static_cast<int>(type::TBOperatorType::TB_INPUT_OP)));
                  }

                  if (input_created) {
                    generate_next_tb_operator(
                        tb_context,
                        tb_graph,
                        [&](int depth) {
                          KNOperator *new_op =
                              g.create_customized_op(input_tensors, tb_graph);
                          if (!new_op) {
                            return;
                          }
                          g.operators.push_back(new_op);
                          assert(new_op->output_tensors.size() ==
                                 tb_context.output_pattern.size());
                          for (size_t i = 0; i < new_op->output_tensors.size();
                               ++i) {
                            c.all_tensors.push_back(new_op->output_tensors[i]);
                            c.algebraic_pattern.push_back(
                                tb_context.output_pattern[i]);
                            c.num_consumers.push_back(0);
                          }
                          for (int input_idx : input_tensor_idx) {
                            c.num_consumers[input_idx]++;
                          }
                          c.op_order.push_back(order);
                          generate_next_kn_operator(c, g, depth + 1);
                          c.op_order.pop_back();
                          for (int input_idx : input_tensor_idx) {
                            c.num_consumers[input_idx]--;
                          }
                          for (size_t j = 0; j < new_op->output_tensors.size();
                               ++j) {
                            c.all_tensors.pop_back();
                            c.algebraic_pattern.pop_back();
                            c.num_consumers.pop_back();
                          }
                          assert(g.operators.back() == new_op);
                          delete g.operators.back();
                          g.operators.pop_back();
                        },
                        depth + 1);
                  }

                  while (!tb_graph.operators.empty()) {
                    delete tb_graph.operators.back();
                    tb_graph.operators.pop_back();
                  }
                  callstack.back().forloop_range_explored.insert(forloop_range);
                }
                callstack.back().forloop_dim_explored.insert(forloop_dim);
              }
              callstack.back().input_map_explored.insert(input_map);
            }
            callstack.back().block_dim_explored.insert(block_dim);
          }
          callstack.back().grid_dim_explored.insert(grid_dim);
        }
        callstack.back().input_idx_explored.insert(input_tensor_idx);
      }
    }
    callstack.back().knop_explored.insert(op_type);
  }

  callstack.pop_back();
}

void KernelGraphGenerator::generate_kernel_graphs() {
  pattern_eval();
  fingerprint_eval();
  generated_graphs.push_back(json(computation_graph));

  kernel::Graph g;
  SearchContext<DTensor> c;

  for (kernel::KNOperator *op : computation_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      g.operators.push_back(op);
      DTensor output_tensor = op->output_tensors[0];
      assert(contains_key(computation_graph_patterns, output_tensor));
      c.all_tensors.push_back(output_tensor);
      c.algebraic_pattern.push_back(
          computation_graph_patterns.at(output_tensor));
      c.num_consumers.push_back(0);
      c.op_order.push_back(
          Order({}, static_cast<int>(type::KNOperatorType::KN_INPUT_OP)));
    }
  }

  process_outputs();
  generate_next_kn_operator(c, g, 0);

  save_checkpoint();

  printf("Total kernel graphs explored: %d\n", num_total_kernel_graphs);
  printf("Random tests performed: %d\n", num_total_random_tests);
  printf("Valid kernel graphs explored: %d\n", num_valid_kernel_graphs);
}

void KernelGraphGenerator::fingerprint_eval() {
  for (auto const &op : computation_graph.operators) {
    op->fingerprint();
  }
}

void KernelGraphGenerator::process_outputs() {
  std::unordered_map<DTensor, int> num_consumers;
  for (kernel::KNOperator *op : computation_graph.operators) {
    for (DTensor const &input : op->input_tensors) {
      num_consumers[input]++;
    }
  }

  for (kernel::KNOperator *op : computation_graph.operators) {
    for (DTensor const &output : op->output_tensors) {
      if (num_consumers[output] == 0) {
        output_tensors.push_back(output);
        final_patterns.push_back(computation_graph_patterns.at(output));
      }
    }
  }
}

bool KernelGraphGenerator::check_pattern(
    std::shared_ptr<AlgebraicPattern> pattern) {
  if (!pattern) {
    return false;
  }
  for (auto const &final_pattern : final_patterns) {
    if (pattern->subpattern_to(*final_pattern)) {
      return true;
    }
  }
  return false;
}

void KernelGraphGenerator::pattern_eval() {
  // Assume operators are in topological order
  int input_id = 0;
  for (KNOperator *op : computation_graph.operators) {
    switch (op->op_type) {
      case type::KNOperatorType::KN_INPUT_OP:
        computation_graph_patterns.insert(
            {op->output_tensors[0],
             std::make_shared<Var>("v_" + std::to_string(input_id))});
        input_id++;
        break;
      case type::KNOperatorType::KN_MATMUL_OP:
        computation_graph_patterns.insert(
            {op->output_tensors[0],
             std::make_shared<Red>(
                 op->input_tensors[0].dim[op->input_tensors[0].num_dims - 1],
                 std::make_shared<Mul>(
                     computation_graph_patterns.at(op->input_tensors[0]),
                     computation_graph_patterns.at(op->input_tensors[1])))});
        break;
      case type::KNOperatorType::KN_ADD_OP:
        computation_graph_patterns.insert(
            {op->output_tensors[0],
             std::make_shared<Add>(
                 computation_graph_patterns.at(op->input_tensors[0]),
                 computation_graph_patterns.at(op->input_tensors[1]))});
      case type::KNOperatorType::KN_REDUCTION_0_OP:
        computation_graph_patterns.insert(
            {op->output_tensors[0],
             std::make_shared<Red>(
                 op->input_tensors[0].dim[0],
                 computation_graph_patterns.at(op->input_tensors[0]))});
      case type::KNOperatorType::KN_REDUCTION_1_OP:
        computation_graph_patterns.insert(
            {op->output_tensors[0],
             std::make_shared<Red>(
                 op->input_tensors[0].dim[1],
                 computation_graph_patterns.at(op->input_tensors[0]))});
      case type::KNOperatorType::KN_REDUCTION_2_OP:
        computation_graph_patterns.insert(
            {op->output_tensors[0],
             std::make_shared<Red>(
                 op->input_tensors[0].dim[2],
                 computation_graph_patterns.at(op->input_tensors[0]))});
        break;
      case type::KNOperatorType::KN_DIV_OP:
        computation_graph_patterns.insert(
            {op->output_tensors[0],
             std::make_shared<Div>(
                 computation_graph_patterns.at(op->input_tensors[0]),
                 computation_graph_patterns.at(op->input_tensors[1]))});
        break;
      case type::KNOperatorType::KN_EXP_OP:
        computation_graph_patterns.insert(
            {op->output_tensors[0],
             std::make_shared<Exp>(
                 computation_graph_patterns.at(op->input_tensors[0]))});
        break;
      default:
        assert(false && "Unsupported computation graph operator");
    }
  }
}

bool KernelGraphGenerator::verify(SearchContext<DTensor> &c,
                                  kernel::Graph const &g) {
  ++num_total_kernel_graphs;
  if (num_total_kernel_graphs % 100 == 1) {
    printf("Total kernel graphs explored: %d.\n",
           num_total_kernel_graphs);
    if (num_total_kernel_graphs % 10000 == 1) {
      save_checkpoint();
    }
  }

  size_t num_outputs = 0;
  for (size_t i = 0; i < c.all_tensors.size(); ++i) {
    if (c.num_consumers[i] == 0) {
      ++num_outputs;
    }
  }
  if (num_outputs != final_patterns.size()) {
    return false;
  }

  std::vector<DTensor> outputs;
  for (size_t i = 0; i < c.all_tensors.size(); ++i) {
    if (c.num_consumers[i] == 0) {
      outputs.push_back(c.all_tensors[i]);
    }
  }

  ++num_total_random_tests;

  std::cout << "random testing: " << json(g) << std::endl;

  for (auto const &op : g.operators) {
    op->fingerprint();
  }

  for (auto const &match : get_matches(outputs.size())) {
    if (have_same_fingerprint(outputs, match)) {
      return true;
    }
  }

  return false;
}

bool KernelGraphGenerator::have_same_fingerprint(
    std::vector<DTensor> const &outputs, std::vector<int> const &match) const {
  assert(outputs.size() == match.size());
  for (int i = 0; i < static_cast<int>(match.size()); ++i) {
    if (!output_tensors[i].has_same_fingerprint(outputs[match[i]])) {
      return false;
    }
  }
  return true;
}

std::vector<layout::SmemLayout> KernelGraphGenerator::get_valid_output_layout(
    threadblock::TBOperator const *op, int idx) {
  assert(idx == 0);
  config.smem_layout_to_explore = {
      layout::SmemRowMajor,
      layout::SmemColumnMajor,
      // layout::SmemRowMajorTensorOpMultiplicand_Crosswise16,
      layout::SmemRowMajorTensorOpMultiplicand_Crosswise32,
      layout::SmemRowMajorTensorOpMultiplicand_Crosswise64,
      // layout::SmemColumnMajorTensorOpMultiplicand_Crosswise16,
      layout::SmemColumnMajorTensorOpMultiplicand_Crosswise32,
      layout::SmemColumnMajorTensorOpMultiplicand_Crosswise64,
  };
  switch (op->op_type) {
    case type::TBOperatorType::TB_INPUT_OP:
      return config.smem_layout_to_explore;
    case type::TBOperatorType::TB_MATMUL_OP: {
      layout::SmemLayout layout1 = op->input_tensors[0].layout,
                         layout2 = op->input_tensors[1].layout;
      if ((layout1 == layout::SmemRowMajor &&
           layout2 == layout::SmemColumnMajor) ||
          (layout1 == layout::SmemRowMajorTensorOpMultiplicand_Crosswise32 &&
           layout2 ==
               layout::SmemColumnMajorTensorOpMultiplicand_Crosswise32) ||
          (layout1 == layout::SmemRowMajorTensorOpMultiplicand_Crosswise64 &&
           layout2 ==
               layout::SmemColumnMajorTensorOpMultiplicand_Crosswise64)) {
        return config.smem_layout_to_explore;
      } else {
        return {};
      }
    }
    case type::TBOperatorType::TB_OUTPUT_OP: {
      if (op->input_tensors[0].layout == layout::SmemRowMajor ||
          op->input_tensors[0].layout ==
              layout::SmemRowMajorTensorOpMultiplicand_Crosswise16 ||
          op->input_tensors[0].layout ==
              layout::SmemRowMajorTensorOpMultiplicand_Crosswise32 ||
          op->input_tensors[0].layout ==
              layout::SmemRowMajorTensorOpMultiplicand_Crosswise64) {
        return {op->input_tensors[0].layout};
      } else {
        return {};
      }
    }
    case type::TBOperatorType::TB_EXP_OP: {
      return {op->input_tensors[0].layout};
    }
    case type::TBOperatorType::TB_DIV_OP:
    case type::TBOperatorType::TB_ADD_OP:
    case type::TBOperatorType::TB_CONCAT_0_OP:
    case type::TBOperatorType::TB_CONCAT_1_OP:
    case type::TBOperatorType::TB_CONCAT_2_OP: {
      return {op->input_tensors[0].layout};
    }
    case type::TBOperatorType::TB_REDUCTION_0_OP:
    case type::TBOperatorType::TB_REDUCTION_1_OP:
    case type::TBOperatorType::TB_REDUCTION_2_OP:
    case type::TBOperatorType::TB_REDUCTION_0_TO_DIMX_OP:
    case type::TBOperatorType::TB_REDUCTION_1_TO_DIMX_OP:
    case type::TBOperatorType::TB_REDUCTION_2_TO_DIMX_OP: {
      if (op->input_tensors[0].layout == layout::SmemRowMajor ||
          op->input_tensors[0].layout == layout::SmemColumnMajor) {
        return {op->input_tensors[0].layout};
      } else {
        return {};
      }
    }
    default:
      assert("Unsupported op type");
      return {};
  }
}

void KernelGraphGenerator::optimize_layout(kernel::Graph &g) {
  optimize_layout(g, 0, 0, -1, -1);
}

void KernelGraphGenerator::optimize_layout(
    kernel::Graph &g, int op_idx, int ts_idx, int bop_idx, int bts_idx) {
  if (bop_idx != -1) {
    kernel::KNCustomizedOp *op =
        dynamic_cast<kernel::KNCustomizedOp *>(g.operators[op_idx]);
    assert(op != nullptr);
    if (bop_idx >= (int)op->bgraph.operators.size()) {
      optimize_layout(g, op_idx + 1, 0, -1, -1);
      return;
    }
    if (bts_idx >= (int)op->bgraph.operators[bop_idx]->output_tensors.size()) {
      optimize_layout(g, op_idx, ts_idx, bop_idx + 1, 0);
      return;
    }
    for (layout::SmemLayout layout :
         get_valid_output_layout(op->bgraph.operators[bop_idx], bts_idx)) {
      op->bgraph.operators[bop_idx]->output_tensors[bts_idx].layout = layout;
      for (TBOperator *bop : op->bgraph.operators) {
        for (STensor &stensor : bop->input_tensors) {
          if (stensor.guid ==
              op->bgraph.operators[bop_idx]->output_tensors[bts_idx].guid) {
            stensor.layout = layout;
          }
        }
      }
      optimize_layout(g, op_idx, ts_idx, bop_idx, bts_idx + 1);
    }
    return;
  }

  if (op_idx >= (int)g.operators.size()) {
    update_best_graph(g);
    return;
  }
  if (ts_idx >= (int)g.operators[op_idx]->output_tensors.size()) {
    if (g.operators[op_idx]->op_type ==
        type::KNOperatorType::KN_CUSTOMIZED_OP) {
      assert(bop_idx == -1);
      optimize_layout(g, op_idx, ts_idx, 0, 0);
    } else {
      optimize_layout(g, op_idx + 1, 0, bop_idx, bts_idx);
    }
    return;
  }

  for (layout::DmemLayout layout :
       {layout::DmemRowMajor, layout::DmemColumnMajor}) {
    if (g.operators[op_idx]->op_type == type::KN_MATMUL_OP && layout != layout::DmemRowMajor) {
      continue;
    }
    g.operators[op_idx]->output_tensors[ts_idx].layout = layout;
    for (KNOperator *op : g.operators) {
      for (DTensor &dtensor : op->input_tensors) {
        if (dtensor.guid == g.operators[op_idx]->output_tensors[ts_idx].guid) {
          dtensor.layout = layout;
        }
      }
    }
    optimize_layout(g, op_idx, ts_idx + 1, bop_idx, bts_idx);
  }
}

void KernelGraphGenerator::update_best_graph(kernel::Graph &g) {
  // std::cerr << "kernel graph candidate: " << json(g) << std::endl;
  ProfileResult result{0};
  for (auto op : g.operators) {
    ProfileResult op_result;
    op->profile(op_result);
    result.run_time += op_result.run_time;
  }
  if (result.run_time < best_profile_result.run_time) {
    best_graph = json(g);
    best_profile_result = result;
    save_checkpoint();
  }
  return;
}

void KernelGraphGenerator::save_checkpoint() const {
  Checkpoint checkpoint{computation_graph,
                        best_graph,
                        best_profile_result,
                        config,
                        callstack,
                        generated_graphs,
                        num_total_kernel_graphs,
                        num_total_random_tests,
                        num_valid_kernel_graphs};
  std::ofstream ofs(filename);
  ofs << json(checkpoint);
}

void KernelGraphGenerator::recovery_from_checkpoint(
    Checkpoint const &checkpoint) {
  computation_graph = checkpoint.computation_graph;
  best_graph = checkpoint.best_graph;
  best_profile_result = checkpoint.best_profile_result;
  config = checkpoint.config;
  dim_strategy = DimStrategy(config);
  callstack = checkpoint.callstack;
  generated_graphs = checkpoint.generated_graphs;

  num_total_kernel_graphs = checkpoint.num_total_kernel_graphs;
  num_total_random_tests = checkpoint.num_total_random_tests;
  num_valid_kernel_graphs = checkpoint.num_valid_kernel_graphs;
}
} // namespace search
} // namespace mirage
