#include "mirage/search/abstract_expr/abstract_expr_eval.h"
#include "mirage/kernel/rms_norm.h"
#include "mirage/search/op_utils.h"
#include "mirage/search/symbolic_graph/op_args.h"

namespace mirage {
namespace search {

void abstract_expr_eval(
    threadblock::Graph const &g,
    std::unordered_map<type::GuidType, std::shared_ptr<AbstractExpr const>>
        &exprs) {
  for (size_t i = 0; i < g.operators.size(); ++i) {
    auto const &op = g.operators[i];
    if (op->output_tensors.size() > 0 &&
        contains_key(exprs, op->output_tensors[0].guid)) {
      continue;
    }
    if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
      exprs.insert(
          {op->output_tensors[0].guid,
           exprs.at(static_cast<threadblock::TBInputOp *>(op)->dtensor.guid)});
    } else if (op->op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      exprs.insert({static_cast<threadblock::TBOutputOp *>(op)->dtensor.guid,
                    exprs.at(op->input_tensors[0].guid)});
    } else if (op->op_type == type::TBOperatorType::TB_CONCAT_1_OP) {
      assert(g.operators[i + 1]->op_type ==
             type::TBOperatorType::TB_CONCAT_0_OP);
      assert(g.operators[i + 2]->op_type == type::TBOperatorType::TB_MATMUL_OP);
      std::vector<threadblock::STensor> input_tensors;
      input_tensors.push_back(op->input_tensors[0]);
      input_tensors.push_back(op->input_tensors[1]);
      input_tensors.push_back(g.operators[i + 1]->input_tensors[0]);
      input_tensors.push_back(g.operators[i + 1]->input_tensors[1]);
      std::vector<std::shared_ptr<AbstractExpr const>> input_exprs;
      for (auto const &input_tensor : input_tensors) {
        assert(contains_key(exprs, input_tensor.guid));
        input_exprs.push_back(exprs.at(input_tensor.guid));
      }
      exprs.insert({op->output_tensors[0].guid, nullptr});
      exprs.insert({g.operators[i + 1]->output_tensors[0].guid, nullptr});
      exprs.insert(
          {g.operators[i + 2]->output_tensors[0].guid,
           get_abstract_expr(type::TBOperatorType::TB_CONCAT_THEN_MATMUL_OP,
                             input_tensors,
                             input_exprs)});
    } else {
      std::vector<std::shared_ptr<AbstractExpr const>> input_exprs;
      for (auto const &input_tensor : op->input_tensors) {
        input_exprs.push_back(exprs.at(input_tensor.guid));
      }
      exprs.insert(
          {op->output_tensors[0].guid,
           get_abstract_expr(op->op_type, op->input_tensors, input_exprs)});
    }
  }
}

void abstract_expr_eval(
    kernel::Graph const &g,
    std::unordered_map<type::GuidType, std::shared_ptr<AbstractExpr const>>
        &exprs) {
  int input_id = 0;
  for (auto const &op : g.operators) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      continue;
    } else if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      exprs.insert({op->output_tensors[0].guid,
                    std::make_shared<Var>("v_" + std::to_string(input_id))});
      input_id++;
    } else if (op->op_type == type::KNOperatorType::KN_RMS_NORM_OP) {
      std::shared_ptr<AbstractExpr const> input_expr =
          exprs.at(op->input_tensors[0].guid);
      std::shared_ptr<AbstractExpr const> denominator_expr =
          abstract_expr_make_rms(
              static_cast<kernel::KNRMSNormOp *>(op)->normalized_size,
              input_expr);
      std::shared_ptr<AbstractExpr const> output_expr =
          abstract_expr_make_div(input_expr, denominator_expr);
      exprs.insert({op->output_tensors[0].guid, output_expr});
    } else if (op->op_type != type::KNOperatorType::KN_CUSTOMIZED_OP) {
      std::vector<std::shared_ptr<AbstractExpr const>> input_exprs;
      for (auto const &input_tensor : op->input_tensors) {
        assert(contains_key(exprs, input_tensor.guid));
        input_exprs.push_back(exprs.at(input_tensor.guid));
      }
      exprs.insert(
          {op->output_tensors[0].guid,
           get_abstract_expr(op->op_type, op->input_tensors, input_exprs)});
    } else {
      assert(op->op_type == type::KNOperatorType::KN_CUSTOMIZED_OP);
      abstract_expr_eval(static_cast<kernel::KNCustomizedOp *>(op)->bgraph,
                         exprs);
    }
  }
}

void abstract_expr_eval(
    SymbolicKNGraph const &kn_graph,
    std::vector<std::shared_ptr<AbstractExpr const>> &exprs) {
  int input_id = 0;
  for (size_t i = 0; i < kn_graph.operators.size(); ++i) {
    if (kn_graph.operators[i].op_type == type::KN_OUTPUT_OP) {
      // Skip the output operator
      continue;
    } else if (kn_graph.operators[i].op_type == type::KN_INPUT_OP) {
      // Create a new variable for each input operator
      exprs.push_back(std::make_shared<Var>("v_" + std::to_string(input_id)));
      input_id++;
    } else if (kn_graph.operators[i].op_type != type::KN_CUSTOMIZED_OP) {
      // Evaluate the expression for pre-defined operators
      std::vector<SymbolicDTensor> input_tensors =
          vector_map(kn_graph.input_indices[i],
                     [&](int i) { return kn_graph.tensors[i]; });
      std::vector<std::shared_ptr<AbstractExpr const>> input_exprs = vector_map(
          kn_graph.input_indices[i], [&](int i) { return exprs[i]; });
      exprs.push_back(get_abstract_expr(
          kn_graph.operators[i].op_type, input_tensors, input_exprs, kn_graph));
    } else {
      // Evaluate the expression for customized operators
      assert(kn_graph.operators[i].op_type == type::KN_CUSTOMIZED_OP);
      std::vector<std::shared_ptr<AbstractExpr const>> input_exprs = vector_map(
          kn_graph.input_indices[i], [&](int i) { return exprs[i]; });
      std::vector<std::shared_ptr<AbstractExpr const>> tb_graph_exprs,
          output_exprs;
      SymbolicTBGraph const &tb_graph =
          std::static_pointer_cast<KNCustomizedOpArgs const>(
              kn_graph.operators[i].args)
              ->tb_graph_template;
      abstract_expr_eval(tb_graph, input_exprs, tb_graph_exprs, output_exprs);
      exprs.insert(exprs.end(), output_exprs.begin(), output_exprs.end());
    }
  }
}

void abstract_expr_eval(
    SymbolicTBGraph const &tb_graph,
    std::vector<std::shared_ptr<AbstractExpr const>> const &input_exprs,
    std::vector<std::shared_ptr<AbstractExpr const>> &exprs,
    std::vector<std::shared_ptr<AbstractExpr const>> &output_exprs) {
  for (size_t i = 0; i < tb_graph.operators.size(); ++i) {
    if (tb_graph.operators[i].op_type == type::TBOperatorType::TB_INPUT_OP) {
      exprs.push_back(input_exprs[i]);
    } else if (tb_graph.operators[i].op_type ==
               type::TBOperatorType::TB_OUTPUT_OP) {
      output_exprs.push_back(exprs[tb_graph.input_indices[i][0]]);
    } else {
      std::vector<SymbolicSTensor> input_tensors =
          vector_map(tb_graph.input_indices[i],
                     [&](int i) { return tb_graph.tensors[i]; });
      std::vector<std::shared_ptr<AbstractExpr const>> input_exprs = vector_map(
          tb_graph.input_indices[i], [&](int i) { return exprs[i]; });
      exprs.push_back(get_abstract_expr(
          tb_graph.operators[i].op_type, input_tensors, input_exprs, tb_graph));
    }
  }
}

} // namespace search
} // namespace mirage
