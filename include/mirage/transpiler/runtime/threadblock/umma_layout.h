
#pragma once

#include "cutlass/gemm/gemm.h"
#include <cute/layout.hpp>
#include <type_traits>

namespace tb {

constexpr bool umma_operand_k_major(bool is_a_role, bool swap_ab) {
  return is_a_role != swap_ab;
}

constexpr cute::UMMA::Major umma_operand_major(bool is_a_role, bool swap_ab) {
  return umma_operand_k_major(is_a_role, swap_ab) ? cute::UMMA::Major::K
                                                  : cute::UMMA::Major::MN;
}

template <bool IS_A_ROLE, bool SWAP_AB>
using UmmaOperandStep =
    std::conditional_t<umma_operand_k_major(IS_A_ROLE, SWAP_AB),
                       cute::Step<cute::_1, cute::_2, cute::_3>,
                       cute::Step<cute::_2, cute::_1, cute::_3>>;

} // namespace tb
