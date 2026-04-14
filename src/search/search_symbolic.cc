#include "mirage/search/search.h"
#include "search_helpers.h"

#include "mirage/search/abstract_expr/abstract_expr.h"
#include "mirage/search/abstract_expr/abstract_expr_eval.h"
#include "mirage/search/dim_strategy.h"
#include "mirage/search/op_utils.h"
#include "mirage/search/symbolic_graph/op_args.h"
#include "mirage/search/symbolic_graph/symbolic_graph.h"
#include "mirage/search/symbolic_graph/symbolic_tensor.h"
#include "mirage/search/verification/formal_verifier.h"
#include "mirage/type.h"
#include "mirage/utils/containers.h"
#include "mirage/utils/json_utils.h"

#include <iostream>
#include <memory>
#include <omp.h>

namespace mirage {
namespace search {

namespace {

std::vector<std::shared_ptr<AbstractExpr const>>
    compute_symbolic_kn_abs_exprs(SymbolicKNGraph const &kn) {
  std::vector<std::shared_ptr<AbstractExpr const>> result;
  abstract_expr_eval(kn, result);
  return result;
}

std::vector<std::shared_ptr<AbstractExpr const>> compute_symbolic_tb_abs_exprs(
    SymbolicKNGraph const &kn,
    SymbolicTBGraph const &tb,
    std::vector<int> const &input_dtensor_indices_for_tb) {
  std::vector<std::shared_ptr<AbstractExpr const>> kn_abs_exprs, output_exprs,
      result;
  abstract_expr_eval(kn, kn_abs_exprs);
  std::vector<std::shared_ptr<AbstractExpr const>> input_exprs = vector_map(
      input_dtensor_indices_for_tb, [&](int i) { return kn_abs_exprs[i]; });
  abstract_expr_eval(tb, input_exprs, result, output_exprs);
  return result;
}

template <typename SymGraph, typename TensorT, typename OpType>
std::shared_ptr<AbstractExpr const> symbolic_infer_expr(
    std::vector<int> const &input_idx,
    OpType op_type,
    std::vector<TensorT> const &all_tensors,
    std::vector<std::shared_ptr<AbstractExpr const>> const &abs_exprs,
    SymGraph const &graph) {
  std::vector<TensorT> input_tensors =
      vector_map(input_idx, [&](int i) { return all_tensors[i]; });
  std::vector<std::shared_ptr<AbstractExpr const>> input_exprs =
      vector_map(input_idx, [&](int i) { return abs_exprs[i]; });
  return get_abstract_expr(op_type, input_tensors, input_exprs, graph);
}

bool is_reduction_op(type::TBOperatorType op_type) {
  return op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_OP ||
         op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_SUM_OP ||
         op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP ||
         op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_RMS_OP ||
         op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP ||
         op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP ||
         op_type ==
             type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_SUM_RESCALE_OP ||
         op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_MAX_OP;
}

bool symbolic_tb_reduction_constraints_ok(SymbolicTBGraph const &tb) {
  int reduction_count = 0;
  for (auto const &op : tb.operators) {
    if (is_reduction_op(op.op_type)) {
      ++reduction_count;
    }
  }
  if (reduction_count > 3) {
    return false;
  }
  for (size_t i = 0; i < tb.operators.size(); ++i) {
    for (size_t j = i + 1; j < tb.operators.size(); ++j) {
      if (is_reduction_op(tb.operators[i].op_type) &&
          is_reduction_op(tb.operators[j].op_type) &&
          tb.input_indices[i][0] == tb.input_indices[j][0]) {
        return false;
      }
    }
  }
  return true;
}

template <typename SymGraph,
          typename TensorT,
          typename OpType,
          typename RecurseFn>
void try_enumerate_symbolic_op(
    SymGraph &g,
    OpType op_type,
    std::vector<int> const &input_idx,
    std::vector<TensorT> const &all_tensors,
    std::vector<std::shared_ptr<AbstractExpr const>> const &abs_exprs,
    KernelGraphGenerator &gen,
    RecurseFn &&recurse) {
  Order order(input_idx, static_cast<int>(op_type));
  if (order <= get_max_op_order(g)) {
    return;
  }
  if (!gen.check_abstract_expr(
          symbolic_infer_expr(input_idx, op_type, all_tensors, abs_exprs, g))) {
    return;
  }
  if (g.add_operator(op_type, input_idx)) {
    recurse();
    g.remove_last_operator();
  }
}

} // namespace

void KernelGraphGenerator::enumerate_symbolic_ops(
    std::shared_ptr<SymbolicKNGraph> kn_graph,
    std::shared_ptr<SymbolicTBGraph> tb_graph,
    std::vector<int> input_dtensor_indices_for_tb_graph,
    SearchLevel level,
    int search_depth,
    bool is_a_new_thread_start) {

  if (!is_a_new_thread_start &&
      search_depth <= (int)multithread_threshold_depth) {
    std::shared_ptr<SymbolicKNGraph> kn_graph_copy =
        std::make_shared<SymbolicKNGraph>(*kn_graph);
    std::shared_ptr<SymbolicTBGraph> tb_graph_copy;
    if (tb_graph) {
      tb_graph_copy = std::make_shared<SymbolicTBGraph>(*tb_graph);
    }
    {
#pragma omp task
      enumerate_symbolic_ops(kn_graph_copy,
                             tb_graph_copy,
                             input_dtensor_indices_for_tb_graph,
                             level,
                             search_depth,
                             true);
    }
    return;
  }

  ++num_total_states;
  if (num_total_states % 100000 == 1) {
    show_statistics();
  }
  if (config.search_time_limit_sec > 0 &&
      get_elapsed_time_in_sec() > config.search_time_limit_sec) {
    return;
  }

  if (level == SearchLevel::LV_KERNEL) {
    assert(tb_graph == nullptr);
    assert(input_dtensor_indices_for_tb_graph.empty());

    if (verify_symbolic_graph(*kn_graph)) {
      return;
    }

    std::vector<std::shared_ptr<AbstractExpr const>> abs_exprs =
        compute_symbolic_kn_abs_exprs(*kn_graph);

    if (kn_graph->operators.size() >= config.max_num_kernel_graph_op) {
      return;
    }

    for (type::KNOperatorType op_type : dim_strategy.get_knop_cand()) {
      if (op_type != type::KNOperatorType::KN_CUSTOMIZED_OP) {
        for (auto const &input_idx :
             dim_strategy.get_input_cand_idx(op_type, kn_graph->tensors)) {
          try_enumerate_symbolic_op(*kn_graph,
                                    op_type,
                                    input_idx,
                                    kn_graph->tensors,
                                    abs_exprs,
                                    *this,
                                    [&]() {
                                      enumerate_symbolic_ops(
                                          kn_graph,
                                          tb_graph,
                                          input_dtensor_indices_for_tb_graph,
                                          level,
                                          search_depth + 1,
                                          false);
                                    });
        }
      } else {
        if (count_symbolic_op_of_type(type::KNOperatorType::KN_CUSTOMIZED_OP,
                                      *kn_graph) >=
            config.max_num_threadblock_graphs) {
          continue;
        }

        for (auto const &input_tensor_idx :
             dim_strategy.get_customized_input_cand_idx(kn_graph->tensors)) {
          Order order(input_tensor_idx, static_cast<int>(op_type));
          if (order <= get_max_op_order(*kn_graph)) {
            continue;
          }
          enumerate_symbolic_tb_config(
              kn_graph, input_tensor_idx, search_depth);
        }
      }
    }
  } else {
    assert(tb_graph != nullptr);

    emit_symbolic_tb_graph(
        kn_graph, tb_graph, input_dtensor_indices_for_tb_graph, search_depth);

    if (tb_graph->operators.size() >= config.max_num_threadblock_graph_op) {
      return;
    }

    if (!symbolic_tb_reduction_constraints_ok(*tb_graph)) {
      return;
    }

    std::vector<std::shared_ptr<AbstractExpr const>> abs_exprs =
        compute_symbolic_tb_abs_exprs(
            *kn_graph, *tb_graph, input_dtensor_indices_for_tb_graph);

    for (type::TBOperatorType op_type : dim_strategy.get_tbop_cand()) {
      for (auto const &input_idx :
           dim_strategy.get_input_cand_idx(op_type, tb_graph->tensors)) {
        try_enumerate_symbolic_op(*tb_graph,
                                  op_type,
                                  input_idx,
                                  tb_graph->tensors,
                                  abs_exprs,
                                  *this,
                                  [&]() {
                                    enumerate_symbolic_ops(
                                        kn_graph,
                                        tb_graph,
                                        input_dtensor_indices_for_tb_graph,
                                        level,
                                        search_depth + 1,
                                        false);
                                  });
      }
    }
  }
}

void KernelGraphGenerator::enumerate_symbolic_tb_config(
    std::shared_ptr<SymbolicKNGraph> kn_graph,
    std::vector<int> const &input_tensor_idx,
    int search_depth) {

  std::vector<SymbolicDTensor> input_tensors =
      vector_map(input_tensor_idx, [&](int i) { return kn_graph->tensors[i]; });

  for (size_t num_parallel_dims :
       dim_strategy.get_num_parallel_dims_cand(input_tensors)) {

    auto imap_cands =
        config.sym_imap
            ? std::vector<std::vector<std::vector<int>>>{std::vector<
                  std::vector<int>>(input_tensors.size(),
                                    std::vector<int>(num_parallel_dims, -1))}
            : dim_strategy.get_input_map_cand(input_tensors, num_parallel_dims);
    auto fmap_cands = config.sym_fmap
                          ? std::vector<std::vector<int>>{std::vector<int>(
                                input_tensors.size(), -1)}
                          : dim_strategy.get_forloop_dim_cand(input_tensors);

    for (auto const &input_maps : imap_cands) {
      for (auto const &forloop_dims : fmap_cands) {
        for (auto const &reduction_degree :
             dim_strategy.get_reduction_degree_cand(*kn_graph)) {

          auto new_tb_graph = std::make_shared<SymbolicTBGraph>(
              kn_graph->next_dim_variable_index, num_parallel_dims);
          new_tb_graph->reduction_degree = reduction_degree;

          bool input_created = true;
          for (size_t i = 0; i < input_tensors.size(); ++i) {
            bool ok;
            if (config.sym_imap && config.sym_fmap) {
              ok = new_tb_graph->add_input(input_tensors[i]);
            } else if (!config.sym_imap && !config.sym_fmap) {
              ok = new_tb_graph->add_input(
                  input_tensors[i], input_maps[i], forloop_dims[i]);
            } else {
              ok = new_tb_graph->add_input(input_tensors[i],
                                           input_maps[i],
                                           forloop_dims[i],
                                           config.sym_imap,
                                           config.sym_fmap);
            }
            if (!ok) {
              input_created = false;
              break;
            }
          }

          if (input_created) {
            enumerate_symbolic_ops(kn_graph,
                                   new_tb_graph,
                                   input_tensor_idx,
                                   SearchLevel::LV_THREADBLOCK,
                                   search_depth + 1,
                                   false);
          }
        }
      }
    }
  }
}

void KernelGraphGenerator::emit_symbolic_tb_graph(
    std::shared_ptr<SymbolicKNGraph> kn_graph,
    std::shared_ptr<SymbolicTBGraph> tb_graph,
    std::vector<int> const &input_dtensor_indices_for_tb_graph,
    int search_depth) {

  std::vector<int> output_tensor_indices;
  {
    std::vector<int> num_consumers(tb_graph->tensors.size(), 0);
    for (auto const &input_indices : tb_graph->input_indices) {
      for (int idx : input_indices) {
        num_consumers[idx]++;
      }
    }
    for (size_t i = 0; i < tb_graph->tensors.size(); ++i) {
      if (num_consumers[i] == 0) {
        output_tensor_indices.push_back(i);
      }
    }
  }

  std::vector<SymbolicSTensor> output_tensors = vector_map(
      output_tensor_indices, [&](int i) { return tb_graph->tensors[i]; });

  if (output_tensor_indices.size() > config.max_num_threadblock_graph_outputs) {
    return;
  }
  if (!all_of(output_tensors,
              [](SymbolicSTensor const &t) { return t.after_accum; })) {
    return;
  }

  auto omap_cands =
      config.sym_omap
          ? std::vector<
                std::vector<std::vector<int>>>{std::vector<std::vector<int>>(
                output_tensor_indices.size(),
                std::vector<int>(tb_graph->grid_dim.size(), -1))}
          : dim_strategy.get_output_map_cand(output_tensors,
                                             tb_graph->grid_dim.size());

  for (auto const &output_maps : omap_cands) {
    bool outputs_created = true;
    if (config.sym_omap) {
      for (int output_index : output_tensor_indices) {
        if (!tb_graph->add_output(output_index, type::TB_EPILOGUE_NONE)) {
          outputs_created = false;
          break;
        }
      }
    } else {
      for (size_t i = 0; i < output_tensor_indices.size(); ++i) {
        if (!tb_graph->add_output(output_tensor_indices[i],
                                  output_maps[i],
                                  type::TB_EPILOGUE_NONE)) {
          outputs_created = false;
          break;
        }
      }
    }
    if (outputs_created) {
      if (kn_graph->add_customized_operator(
              *tb_graph, input_dtensor_indices_for_tb_graph)) {
        enumerate_symbolic_ops(kn_graph,
                               nullptr,
                               {},
                               SearchLevel::LV_KERNEL,
                               search_depth + 1,
                               false);
        kn_graph->remove_last_operator();
      }
    }
    while (tb_graph->operators.back().op_type ==
           type::TBOperatorType::TB_OUTPUT_OP) {
      tb_graph->remove_last_operator();
    }
  }
}

bool KernelGraphGenerator::verify_symbolic_graph(
    SymbolicKNGraph const &symbolic_graph) {
  ++num_total_random_tests;

  std::shared_ptr<FormalVerifier> verifier =
      std::dynamic_pointer_cast<FormalVerifier>(this->verifier);
  assert(verifier);

  bool has_symbolic_maps = false;
  for (auto const &op : symbolic_graph.operators) {
    if (op.op_type == type::KNOperatorType::KN_CUSTOMIZED_OP) {
      auto args = std::static_pointer_cast<KNCustomizedOpArgs const>(op.args);
      for (auto const &tb_op : args->tb_graph_template.operators) {
        if (tb_op.op_type == type::TBOperatorType::TB_INPUT_OP) {
          auto a = static_cast<TBInputOpArgs const *>(tb_op.args.get());
          if (!a->input_map.is_concrete()) {
            has_symbolic_maps = true;
            break;
          }
        } else if (tb_op.op_type == type::TBOperatorType::TB_OUTPUT_OP) {
          auto a = static_cast<TBOutputOpArgs const *>(tb_op.args.get());
          if (!a->output_map.is_concrete()) {
            has_symbolic_maps = true;
            break;
          }
        }
      }
      if (has_symbolic_maps) {
        break;
      }
    }
  }

  if (has_symbolic_maps) {
    auto verified =
        verifier->verify_symbolic_graph_with_unknown_maps(symbolic_graph);
    if (!verified.empty()) {
      for (auto const &[concrete_graph, match] : verified) {
        ++num_symbolic_graphs;
        ++num_valid_kernel_graphs;
#pragma omp critical
        {
          std::cerr << "verified symbolic graph (from unknown maps): "
                    << json(concrete_graph) << std::endl;
          generated_graphs.push_back(json(concrete_graph));
        }
      }
      return true;
    }
    return false;
  }

  OutputMatch match = verifier->verify_symbolic_graph(symbolic_graph);

  if (match.is_valid()) {
    ++num_symbolic_graphs;
    ++num_valid_kernel_graphs;
#pragma omp critical
    {
      std::cerr << "verified symbolic graph: " << json(symbolic_graph)
                << std::endl;
      generated_graphs.push_back(json(symbolic_graph));
    }
    return true;
  } else {
    return false;
  }
}

} // namespace search
} // namespace mirage
