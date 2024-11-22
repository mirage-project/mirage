#include "mirage/search/abstract_expr/abstract_expr_eval.h"
#include "mirage/kernel/rms_norm.h"
#include "mirage/search/op_utils.h"

namespace mirage {
namespace search {

void abstract_expr_eval(
    threadblock::Graph const &g,
    std::unordered_map<int64_t, std::shared_ptr<AbstractExpr>> &patterns) {
  for (size_t i = 0; i < g.operators.size(); ++i) {
    auto const &op = g.operators[i];
    if (op->output_tensors.size() > 0 &&
        contains_key(patterns, op->output_tensors[0].guid)) {
      continue;
    }
    if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
      patterns.insert(
          {op->output_tensors[0].guid,
           patterns.at(
               static_cast<threadblock::TBInputOp *>(op)->dtensor.guid)});
    } else if (op->op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      patterns.insert({static_cast<threadblock::TBOutputOp *>(op)->dtensor.guid,
                       patterns.at(op->input_tensors[0].guid)});
    } else if (op->op_type == type::TBOperatorType::TB_CONCAT_1_OP) {
      assert(g.operators[i + 1]->op_type ==
             type::TBOperatorType::TB_CONCAT_0_OP);
      assert(g.operators[i + 2]->op_type == type::TBOperatorType::TB_MATMUL_OP);
      std::vector<threadblock::STensor> input_tensors;
      input_tensors.push_back(op->input_tensors[0]);
      input_tensors.push_back(op->input_tensors[1]);
      input_tensors.push_back(g.operators[i + 1]->input_tensors[0]);
      input_tensors.push_back(g.operators[i + 1]->input_tensors[1]);
      std::vector<std::shared_ptr<AbstractExpr>> input_patterns;
      for (auto const &input_tensor : input_tensors) {
        assert(contains_key(patterns, input_tensor.guid));
        input_patterns.push_back(patterns.at(input_tensor.guid));
      }
      patterns.insert({op->output_tensors[0].guid, nullptr});
      patterns.insert({g.operators[i + 1]->output_tensors[0].guid, nullptr});
      patterns.insert(
          {g.operators[i + 2]->output_tensors[0].guid,
           get_pattern(type::TBOperatorType::TB_CONCAT_THEN_MATMUL_OP,
                       input_tensors,
                       input_patterns)});
    } else {
      std::vector<std::shared_ptr<AbstractExpr>> input_patterns;
      for (auto const &input_tensor : op->input_tensors) {
        input_patterns.push_back(patterns.at(input_tensor.guid));
      }
      patterns.insert(
          {op->output_tensors[0].guid,
           get_pattern(op->op_type, op->input_tensors, input_patterns)});
    }
  }
}

void abstract_expr_eval(
    kernel::Graph const &g,
    std::unordered_map<int64_t, std::shared_ptr<AbstractExpr>> &patterns) {
  int input_id = 0;
  for (auto const &op : g.operators) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      continue;
    } else if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      patterns.insert({op->output_tensors[0].guid,
                       std::make_shared<Var>("v_" + std::to_string(input_id))});
      input_id++;
    } else if (op->op_type == type::KNOperatorType::KN_RMS_NORM_OP) {
      std::shared_ptr<AbstractExpr> input_pattern =
          patterns.at(op->input_tensors[0].guid);
      std::shared_ptr<AbstractExpr> denominator_pattern = std::make_shared<RMS>(
          static_cast<kernel::KNRMSNormOp *>(op)->normalized_size,
          input_pattern);
      std::shared_ptr<AbstractExpr> output_pattern =
          std::make_shared<Div>(input_pattern, denominator_pattern);
      patterns.insert({op->output_tensors[0].guid, output_pattern});
    } else if (op->op_type != type::KNOperatorType::KN_CUSTOMIZED_OP) {
      std::vector<std::shared_ptr<AbstractExpr>> input_patterns;
      for (auto const &input_tensor : op->input_tensors) {
        assert(contains_key(patterns, input_tensor.guid));
        input_patterns.push_back(patterns.at(input_tensor.guid));
      }
      patterns.insert(
          {op->output_tensors[0].guid,
           get_pattern(op->op_type, op->input_tensors, input_patterns)});
    } else {
      assert(op->op_type == type::KNOperatorType::KN_CUSTOMIZED_OP);
      abstract_expr_eval(static_cast<kernel::KNCustomizedOp *>(op)->bgraph,
                         patterns);
    }
  }
}

} // namespace search
} // namespace mirage