#pragma once

#include "linear_fp8_sm100_types.hpp"
#include "linear_fp8_sm100_utils.cuh"

namespace mirage::blackwell::linear_fp8_sm100 {

struct EpilogueIdentity {
  template <uint32_t STORE_BLOCK_N>
  __device__ __forceinline__ static uint32_t
      apply_index_n(uint32_t const &n_idx) {
    return n_idx;
  }
};

template <uint32_t kLeft, uint32_t kMid, uint32_t kRight>
struct EpilogueHeadSplits : EpilogueIdentity {
  template <uint32_t STORE_BLOCK_N>
  __device__ __forceinline__ static uint32_t
      apply_index_n(uint32_t const &n_idx) {
    DG_STATIC_ASSERT(kLeft % STORE_BLOCK_N == 0 && kMid % STORE_BLOCK_N == 0 &&
                         kRight % STORE_BLOCK_N == 0,
                     "Invalid head splits config");
    return n_idx + (n_idx + kRight) / (kLeft + kRight) * kMid;
  }
};

#pragma clang diagnostic pop

} // namespace mirage::blackwell::linear_fp8_sm100
