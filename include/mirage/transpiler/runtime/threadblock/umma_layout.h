// umma_layout.h - the one place the UMMA operand majorness rule lives.
//
// ROLE vs MAJORNESS, the invariant every Blackwell operand path must agree
// on: which operand slot a tensor feeds (the MMA role, A or B) and how it
// sits in memory (its majorness) are different things, and swapAB flips the
// role without moving the bytes. An A-role operand is K-major and a B-role
// operand is MN-major -- UNLESS the matmul is swapped, in which case each
// role carries the other's majorness, because swapAB computes C^T = B^T A^T
// over the same physical tiles.
//
// Three things read smem through layouts derived from this rule: the TMA
// input atom (write side), the wide-operand sync copy (write side), and
// Blackwell_Matmul's DstPipeLayout_A/B (read side). They must agree by
// construction, so none of them spells the rule out itself.

#pragma once

#include "cutlass/gemm/gemm.h"
#include <cute/layout.hpp>
#include <type_traits>

namespace tb {

// True when the operand is K-major from the UMMA's point of view:
// role A unswapped, or role B under swapAB.
constexpr bool umma_operand_k_major(bool is_a_role, bool swap_ab) {
  return is_a_role != swap_ab;
}

constexpr cute::UMMA::Major umma_operand_major(bool is_a_role, bool swap_ab) {
  return umma_operand_k_major(is_a_role, swap_ab) ? cute::UMMA::Major::K
                                                  : cute::UMMA::Major::MN;
}

// The tile_to_mma_shape step order follows majorness: K-major iterates
// K-innermost (<1,2,3>), MN-major the other way (<2,1,3>). Matches what
// CUTLASS's sm100_mma_warpspecialized selects.
template <bool IS_A_ROLE, bool SWAP_AB>
using UmmaOperandStep =
    std::conditional_t<umma_operand_k_major(IS_A_ROLE, SWAP_AB),
                       cute::Step<cute::_1, cute::_2, cute::_3>,
                       cute::Step<cute::_2, cute::_1, cute::_3>>;

} // namespace tb
