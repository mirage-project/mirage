#include "mirage/search/pattern_eval.h"
#include "mirage/kernel/rms_norm.h"
#include "mirage/search/op_utils.h"

namespace mirage {
namespace search {

void pattern_eval(
    threadblock::Graph const &g,
    std::unordered_map<int64_t, std::shared_ptr<AlgebraicPattern>> &patterns) {
  for (auto const &op : g.operators) {
    if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
      patterns.insert(
          {op->output_tensors[0].guid,
           patterns.at(
               static_cast<threadblock::TBInputOp *>(op)->dtensor.guid)});
    } else if (op->op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      patterns.insert({static_cast<threadblock::TBOutputOp *>(op)->dtensor.guid,
                       patterns.at(op->input_tensors[0].guid)});
    } else {
      std::vector<std::shared_ptr<AlgebraicPattern>> input_patterns;
      for (auto const &input_tensor : op->input_tensors) {
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
  for (auto const &op : g.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      patterns.insert({op->output_tensors[0].guid,
                       std::make_shared<Var>("v_" + std::to_string(input_id))});
      input_id++;
    } else if (op->op_type == type::KNOperatorType::KN_RMS_NORM_OP) {
      std::shared_ptr<AlgebraicPattern> input_pattern =
          patterns.at(op->input_tensors[0].guid);
      std::shared_ptr<AlgebraicPattern> denominator_pattern =
          std::make_shared<RMS>(
              static_cast<kernel::KNRMSNormOp *>(op)->normalized_size,
              input_pattern);
      std::shared_ptr<AlgebraicPattern> output_pattern =
          std::make_shared<Div>(input_pattern, denominator_pattern);
      patterns.insert({op->output_tensors[0].guid, output_pattern});
    } else if (op->op_type != type::KNOperatorType::KN_CUSTOMIZED_OP) {
      std::vector<std::shared_ptr<AlgebraicPattern>> input_patterns;
      for (auto const &input_tensor : op->input_tensors) {
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

} // namespace search
} // namespace mirage