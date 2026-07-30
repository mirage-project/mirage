#pragma once

#include "mirage/search/abstract_expr/abstract_expr.h"
#include "mirage/search/config.h"
#include "mirage/search/symbolic_graph/symbolic_graph.h"
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
bool is_binary(type::KNOperatorType op);
bool is_unary(type::KNOperatorType op);

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
size_t count_op_of_type(type::KNOperatorType op_type, kernel::Graph const &g);
size_t count_op_of_type(type::TBOperatorType op_type,
                        threadblock::Graph const &g);
size_t count_op_of_type(type::KNOperatorType op_type, SymbolicKNGraph const &g);
size_t count_op_of_type(type::TBOperatorType op_type, SymbolicTBGraph const &g);

} // namespace search
} // namespace mirage
