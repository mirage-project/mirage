#pragma once

#include "mirage/search/abstract_expr/abstract_expr.h"
#include "mirage/search/config.h"
#include "mirage/utils/hash_utils.h"

namespace mirage {
namespace search {

template <typename Op, typename OpType>
size_t get_operator_hash(Op i, OpType op) {
  size_t h = 0;
  hash_combine(h, i);
  hash_combine(h, op);
  return h;
}

template <typename Op, typename OpType>
size_t get_operator_hash(Op i, Op j, OpType op) {
  size_t h = 0;
  hash_combine(h, i);
  hash_combine(h, j);
  hash_combine(h, op);
  return h;
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

bool is_binary(type::TBOperatorType op);
bool is_unary(type::TBOperatorType op);
bool is_binary(type::KNOperatorType op);
bool is_unary(type::KNOperatorType op);

int get_input_number(type::KNOperatorType);
int get_input_number(type::TBOperatorType);

std::shared_ptr<AbstractExpr> get_pattern(type::KNOperatorType op,
                                          DTensor const &tensor,
                                          std::shared_ptr<AbstractExpr> opd);
std::shared_ptr<AbstractExpr> get_pattern(type::TBOperatorType op,
                                          STensor const &tensor,
                                          std::shared_ptr<AbstractExpr> opd);
std::shared_ptr<AbstractExpr>
    get_pattern(type::KNOperatorType op,
                std::vector<DTensor> const &tensors,
                std::vector<std::shared_ptr<AbstractExpr>> const &opds);
std::shared_ptr<AbstractExpr> get_pattern(type::KNOperatorType op,
                                          DTensor const &input1,
                                          DTensor const &input2,
                                          std::shared_ptr<AbstractExpr> lhs,
                                          std::shared_ptr<AbstractExpr> rhs);
std::shared_ptr<AbstractExpr> get_pattern(type::TBOperatorType op,
                                          STensor const &input1,
                                          STensor const &input2,
                                          std::shared_ptr<AbstractExpr> lhs,
                                          std::shared_ptr<AbstractExpr> rhs);
std::shared_ptr<AbstractExpr>
    get_pattern(type::TBOperatorType op,
                std::vector<STensor> const &tensors,
                std::vector<std::shared_ptr<AbstractExpr>> const &opds);

KNOperator *create_op(kernel::Graph &g,
                      type::KNOperatorType type,
                      DTensor const &input);
KNOperator *create_op(kernel::Graph &g,
                      type::KNOperatorType type,
                      DTensor const &input1,
                      DTensor const &input2);
KNOperator *create_op(kernel::Graph &g,
                      type::KNOperatorType type,
                      std::vector<DTensor> const &inputs);
TBOperator *create_op(threadblock::Graph &g,
                      type::TBOperatorType type,
                      STensor const &input);
TBOperator *create_op(threadblock::Graph &g,
                      type::TBOperatorType type,
                      STensor const &input1,
                      STensor const &input2);
TBOperator *create_op(threadblock::Graph &g,
                      type::TBOperatorType type,
                      std::vector<STensor> const &inputs);
size_t count_op_of_type(type::KNOperatorType op_type, kernel::Graph const &g);
size_t count_op_of_type(type::TBOperatorType op_type,
                        threadblock::Graph const &g);

} // namespace search
} // namespace mirage
