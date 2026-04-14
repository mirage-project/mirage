#pragma once

#include "mirage/search/order.h"
#include "mirage/search/symbolic_graph/symbolic_graph.h"
#include "mirage/threadblock/graph.h"
#include "mirage/utils/containers.h"

#include <cassert>
#include <unordered_map>
#include <vector>

namespace mirage {
namespace search {

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

template <>
inline Order get_max_op_order(SymbolicKNGraph const &g) {
  return Order(g.input_indices.back(),
               static_cast<int>(g.operators.back().op_type));
}

template <>
inline Order get_max_op_order(SymbolicTBGraph const &g) {
  return Order(g.input_indices.back(),
               static_cast<int>(g.operators.back().op_type));
}

template <typename GraphType, typename OpType>
bool check_order(std::vector<int> input_idx,
                 OpType op_type,
                 GraphType const &g) {
  Order order(input_idx, static_cast<int>(op_type));
  if (order <= get_max_op_order(g)) {
    return false;
  }
  return true;
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

inline std::vector<kernel::DTensor>
    get_input_tensors(threadblock::Graph const &g) {
  std::vector<kernel::DTensor> input_tensors;
  for (auto const &op : g.operators) {
    if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
      input_tensors.push_back(
          static_cast<threadblock::TBInputOp *>(op)->dtensor);
    }
  }
  return input_tensors;
}

} // namespace search
} // namespace mirage
