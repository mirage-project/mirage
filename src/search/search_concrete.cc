#include "mirage/search/search.h"
#include "search_helpers.h"

#include "mirage/kernel/customized.h"
#include "mirage/search/abstract_expr/abstract_expr.h"
#include "mirage/search/abstract_expr/abstract_expr_eval.h"
#include "mirage/search/dim_strategy.h"
#include "mirage/search/op_utils.h"
#include "mirage/search/verification/verifier.h"
#include "mirage/type.h"
#include "mirage/utils/containers.h"

#include <iostream>
#include <memory>
#include <omp.h>

namespace mirage {
namespace search {

namespace {

using AbsExprMap = std::unordered_map<type::GuidType,
                                      std::shared_ptr<AbstractExpr const>>;

AbsExprMap compute_concrete_abs_exprs(kernel::Graph const &kn,
                                      threadblock::Graph const *tb) {
  AbsExprMap result;
  abstract_expr_eval(kn, result);
  if (tb != nullptr) {
    abstract_expr_eval(*tb, result);
  }
  return result;
}

template <typename TensorT, typename OpType>
std::shared_ptr<AbstractExpr const> concrete_infer_expr(
    std::vector<TensorT> const &input_tensors,
    OpType op_type,
    AbsExprMap const &exprs) {
  std::vector<std::shared_ptr<AbstractExpr const>> input_exprs =
      vector_map(input_tensors,
                 [&](auto const &t) { return exprs.at(t.guid); });
  return get_abstract_expr(op_type, input_tensors, input_exprs);
}

template <typename Graph, typename Tensor, typename OpType, typename RecurseFn>
void try_enumerate_concrete_op(
    Graph &g,
    OpType op_type,
    std::vector<int> const &input_idx,
    std::vector<Tensor> const &all_tensors,
    AbsExprMap const &exprs,
    KernelGraphGenerator &gen,
    RecurseFn &&recurse) {
  if (!check_order(input_idx, op_type, g)) {
    return;
  }
  std::vector<Tensor> input_tensors = vector_map(
      input_idx, [&](int index) { return all_tensors[index]; });
  if (!gen.check_abstract_expr(
          concrete_infer_expr(input_tensors, op_type, exprs))) {
    return;
  }
  auto *old_last_op = g.operators.back();
  auto *new_op = create_op(g, op_type, input_tensors);
  if (new_op) {
    g.operators.push_back(new_op);
    recurse();
  }
  while (g.operators.back() != old_last_op) {
    delete g.operators.back();
    g.operators.pop_back();
  }
}

} // namespace

void KernelGraphGenerator::enumerate_ops(
    SearchContext &c,
    std::function<bool(SearchContext const &)> const &verify,
    std::vector<SerializedSearchContext> &verified_graphs,
    size_t search_depth,
    bool is_a_new_thread_start) {
  ++num_total_states;
  if (num_total_states % 100000 == 1) {
    show_statistics();
  }
  if (config.search_time_limit_sec > 0 &&
      get_elapsed_time_in_sec() > config.search_time_limit_sec) {
    return;
  }
  if (verify(c)) {
    verified_graphs.push_back(SerializedSearchContext(c));
    return;
  }

  if (!is_a_new_thread_start && search_depth <= multithread_threshold_depth) {
    SearchContext c_copied = SerializedSearchContext(c).deserialize();
#if !defined(MIRAGE_FINGERPRINT_USE_CPU) || defined(MIRAGE_USE_FORMAL_VERIFIER)
#pragma omp task
#endif
    {
      enumerate_ops(
          c_copied, verify, verified_graphs, search_depth, true);
    }
    return;
  }

  AbsExprMap algebraic_expr =
      compute_concrete_abs_exprs(*c.kn_graph, c.tb_graph.get());

  if (c.level == SearchLevel::LV_KERNEL) {
    assert(c.tb_graph == nullptr);
    if (c.kn_graph->operators.size() >= config.max_num_kernel_graph_op) {
      return;
    }
    std::vector<DTensor> all_tensors = get_all_tensors(*c.kn_graph);
    for (type::KNOperatorType op_type : dim_strategy.get_knop_cand()) {
      if (op_type != type::KNOperatorType::KN_CUSTOMIZED_OP) {
        for (auto const &input_idx :
             dim_strategy.get_input_cand_idx(op_type, all_tensors)) {
          try_enumerate_concrete_op(
              *c.kn_graph, op_type, input_idx, all_tensors, algebraic_expr,
              *this,
              [&]() {
                enumerate_ops(c, verify, verified_graphs, search_depth + 1);
              });
        }
      } else {
        if (count_op_of_type(type::KNOperatorType::KN_CUSTOMIZED_OP,
                             *c.kn_graph) >=
            config.max_num_threadblock_graphs) {
          continue;
        }
        for (auto const &input_tensor_idx :
             dim_strategy.get_customized_input_cand_idx(all_tensors)) {
          if (!check_order(input_tensor_idx, op_type, *c.kn_graph)) {
            continue;
          }
          enumerate_tb_config(c, verify, verified_graphs, search_depth,
                              input_tensor_idx);
        }
      }
    }
  } else {
    assert(c.tb_graph != nullptr);

    emit_tb_graph(c, verify, verified_graphs, search_depth, {});

    if (c.tb_graph->operators.size() >= config.max_num_threadblock_graph_op) {
      return;
    }

    std::vector<STensor> all_tensors = get_all_tensors(*c.tb_graph);
    for (type::TBOperatorType op_type : dim_strategy.get_tbop_cand()) {
      if (count_op_of_type(type::TBOperatorType::TB_CONCAT_0_OP, *c.tb_graph) >=
              1 &&
          op_type == type::TBOperatorType::TB_CONCAT_THEN_MATMUL_OP) {
        continue;
      }
      for (auto const &input_idx :
           dim_strategy.get_input_cand_idx(op_type, all_tensors)) {
        try_enumerate_concrete_op(
            *c.tb_graph, op_type, input_idx, all_tensors, algebraic_expr,
            *this,
            [&]() {
              enumerate_ops(c, verify, verified_graphs, search_depth + 1);
            });
      }
    }
  }
}

void KernelGraphGenerator::enumerate_tb_config(
    SearchContext &c,
    std::function<bool(SearchContext const &)> const &verify,
    std::vector<SerializedSearchContext> &verified_graphs,
    size_t search_depth,
    std::vector<int> const &input_tensor_idx) {

  std::vector<DTensor> all_tensors = get_all_tensors(*c.kn_graph);
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
            c.tb_graph = std::make_shared<threadblock::Graph>(
                grid_dim, block_dim, forloop_range, config.reduction_dimx);
            bool input_created = true;
            for (size_t i = 0; i < input_tensors.size(); ++i) {
              TBOperator *input_op =
                  c.tb_graph->create_input_op(input_tensors[i],
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
              enumerate_ops(c, verify, verified_graphs, search_depth + 1);
              c.level = SearchLevel::LV_KERNEL;
            }
            c.tb_graph = nullptr;
          }
        }
      }
    }
  }
}

void KernelGraphGenerator::emit_tb_graph(
    SearchContext &c,
    std::function<bool(SearchContext const &)> const &verify,
    std::vector<SerializedSearchContext> &verified_graphs,
    size_t search_depth,
    std::vector<int> const &) {

  std::vector<STensor> output_tensors;
  for (auto const &op : c.tb_graph->operators) {
    for (auto const &tensor : op->output_tensors) {
      if (get_num_consumers(*c.tb_graph, tensor) == 0) {
        if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
          return;
        }
        if (!tensor.after_accum) {
          return;
        }
        output_tensors.push_back(tensor);
      }
    }
  }
  if (output_tensors.empty()) return;
  if (output_tensors.size() > config.max_num_threadblock_graph_outputs) return;

  for (int3 output_map : dim_strategy.get_output_map_cand(
           output_tensors, c.tb_graph->grid_dim)) {
    bool ok = true;
    for (STensor const &stensor : output_tensors) {
      TBOperator *new_op =
          c.tb_graph->create_output_op(stensor, output_map,
                                        -1 /*forloop_dim*/,
                                        mirage::type::TB_EPILOGUE_NONE);
      if (!new_op) { ok = false; break; }
      c.tb_graph->operators.push_back(new_op);
    }
    if (ok) {
      KNOperator *new_op = c.kn_graph->create_customized_op(
          get_input_tensors(*c.tb_graph), *c.tb_graph);
      if (new_op) {
        c.kn_graph->operators.push_back(new_op);
        c.level = SearchLevel::LV_KERNEL;
        std::shared_ptr<threadblock::Graph> tb_graph = c.tb_graph;
        c.tb_graph = nullptr;
        if (check_range(init_ranges, target_ranges, *c.kn_graph)) {
          enumerate_ops(c, verify, verified_graphs, search_depth + 1);
        }
        c.tb_graph = tb_graph;
        c.level = SearchLevel::LV_THREADBLOCK;
        delete c.kn_graph->operators.back();
        c.kn_graph->operators.pop_back();
      }
    }
    while (c.tb_graph->operators.back()->op_type ==
           type::TBOperatorType::TB_OUTPUT_OP) {
      c.tb_graph->operators.pop_back();
    }
  }
}

bool KernelGraphGenerator::verify_concrete_graph(kernel::Graph &g) {
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
#if !defined(MIRAGE_FINGERPRINT_USE_CPU) || defined(MIRAGE_USE_FORMAL_VERIFIER)
#pragma omp critical
#endif
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

} // namespace search
} // namespace mirage
