#pragma once

#include "mirage/search/algebraic_pattern.h"
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

bool is_binary(type::TBOperatorType op);
bool is_unary(type::TBOperatorType op);
bool is_binary(type::KNOperatorType op);
bool is_unary(type::KNOperatorType op);
std::shared_ptr<AlgebraicPattern>
    get_pattern(type::KNOperatorType op,
                DTensor const &tensor,
                std::shared_ptr<AlgebraicPattern> opd);
std::shared_ptr<AlgebraicPattern>
    get_pattern(type::TBOperatorType op,
                STensor const &tensor,
                std::shared_ptr<AlgebraicPattern> opd);
std::shared_ptr<AlgebraicPattern>
    get_pattern(type::KNOperatorType op,
                std::vector<DTensor> const &tensors,
                std::vector<std::shared_ptr<AlgebraicPattern>> const &opds);
std::shared_ptr<AlgebraicPattern>
    get_pattern(type::KNOperatorType op,
                DTensor const &input1,
                DTensor const &input2,
                std::shared_ptr<AlgebraicPattern> lhs,
                std::shared_ptr<AlgebraicPattern> rhs);
std::shared_ptr<AlgebraicPattern>
    get_pattern(type::TBOperatorType op,
                STensor const &input1,
                STensor const &input2,
                std::shared_ptr<AlgebraicPattern> lhs,
                std::shared_ptr<AlgebraicPattern> rhs);
std::shared_ptr<AlgebraicPattern>
    get_pattern(type::TBOperatorType op,
                std::vector<STensor> const &tensors,
                std::vector<std::shared_ptr<AlgebraicPattern>> const &opds);

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

} // namespace search
} // namespace mirage
