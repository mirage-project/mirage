#pragma once

#include "mirage/search/abstract_expr/abstract_expr.h"
#include "mirage/search/config.h"
#include "mirage/search/symbolic_graph/symbolic_graph.h"
#include "mirage/utils/containers.h"
#include "mirage/utils/hash_utils.h"

namespace mirage {
namespace search {

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
bool is_commutative(type::TBOperatorType op);
bool is_binary(type::KNOperatorType op);
bool is_unary(type::KNOperatorType op);
bool is_commutative(type::KNOperatorType op);

int get_input_number(type::KNOperatorType);
int get_input_number(type::TBOperatorType);

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

template <typename GraphType, typename OperatorType>
size_t count_op_of_types(std::unordered_set<OperatorType> const &op_types, GraphType const &g) {
  return filter(g.operators,[&](auto const &op) {
    return contains(op_types, op->op_type);
  }).size();
}

template <typename GraphType, typename OperatorType>
size_t count_op_of_type(OperatorType op_type, GraphType const &g) {
  return count_op_of_types(std::unordered_set<OperatorType>{op_type}, g);
}

template <typename GraphType, typename OperatorType>
size_t count_symbolic_op_of_types(std::unordered_set<OperatorType> const &op_types, GraphType const &g) {
  return filter(g.operators,[&](auto const &op) {
    return contains(op_types, op.op_type);
  }).size();
}

template <typename GraphType, typename OperatorType>
size_t count_symbolic_op_of_type(OperatorType op_type, GraphType const &g) {
  return count_symbolic_op_of_types(std::unordered_set<OperatorType>{op_type}, g);
}

} // namespace search
} // namespace mirage
