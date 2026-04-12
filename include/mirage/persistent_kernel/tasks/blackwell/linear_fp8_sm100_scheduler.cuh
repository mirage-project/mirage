#pragma once

#include "linear_fp8_sm100_types.hpp"
#include "linear_fp8_sm100_utils.cuh"

namespace mirage::blackwell::linear_fp8_sm100 {

enum class IndexType {
  MN,
  K,
  SF_K,
};

template <GemmType kGemmType,
          uint32_t BLOCK_M,
          uint32_t BLOCK_N,
          uint32_t kNumSMs,
          bool kIsMulticastOnA>
static constexpr uint32_t get_num_1d_blocks_per_group() {
  uint32_t num_best_blocks = 0,
           min_usage = cute::numeric_limits<uint32_t>::max();
  for (auto const &candidate : {8u, 16u}) {
    auto const &usage =
        kIsMulticastOnA ? candidate * BLOCK_N +
                              constexpr_ceil_div(kNumSMs, candidate) * BLOCK_M
                        : candidate * BLOCK_M +
                              constexpr_ceil_div(kNumSMs, candidate) * BLOCK_N;
    if (usage < min_usage) {
      min_usage = usage;
      num_best_blocks = candidate;
    }
  }
  return num_best_blocks;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "cppcoreguidelines-pro-type-member-init"
template <GemmType kGemmType,
          uint32_t BLOCK_M,
          uint32_t BLOCK_N,
          uint32_t kNumGroups,
          uint32_t kNumMulticast,
          bool kIsMulticastOnA,
          uint32_t kNumSMs,
          uint32_t SF_K_ALIGNMENT = 512u,
          uint32_t kNum1DBlocksPerGroup =
              get_num_1d_blocks_per_group<kGemmType,
                                          BLOCK_M,
                                          BLOCK_N,
                                          kNumSMs,
                                          kIsMulticastOnA>()>
struct Scheduler {
  int current_iter = -1;
  uint32_t num_blocks;
  uint32_t num_m_blocks;
  uint32_t num_n_blocks;
  uint32_t num_blocks_in_group;
  bool is_peer_cta_alive = true;
  int *grouped_layout;
  uint32_t current_group_idx = 0;
  uint32_t current_m_cumsum = 0;
  uint32_t last_psum_m = 0, current_psum_m, current_m_block_cumsum = 0;
  uint32_t current_shape_k, current_num_valid_groups = 0, current_k_cumsum = 0,
                            current_sf_k_cumsum = 0;
  uint32_t next_group_idx, next_shape_k;

  __device__ __forceinline__ void get_next_k_group(uint32_t &group_idx,
                                                   uint32_t &shape_k) const {
    for (; group_idx < kNumGroups; ++group_idx) {
      shape_k = __ldg(grouped_layout + group_idx);
      if (shape_k > 0) {
        break;
      }
    }
  }

  __device__ __forceinline__ explicit Scheduler(uint32_t const &shape_m,
                                                uint32_t const &shape_n,
                                                uint32_t const &shape_k,
                                                int *grouped_layout = nullptr) {
    num_m_blocks = ceil_div(shape_m, BLOCK_M);
    num_n_blocks = ceil_div(shape_n, BLOCK_N);
    current_shape_k = shape_k;
    if constexpr (kGemmType == GemmType::Normal ||
                  kGemmType == GemmType::Batched) {
      num_blocks = num_m_blocks * num_n_blocks;
    } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
      num_blocks = num_m_blocks * num_n_blocks;
      this->grouped_layout = grouped_layout;
    } else if constexpr (kGemmType == GemmType::MGroupedMasked) {
      this->grouped_layout = grouped_layout;
    } else if constexpr (kGemmType ==
                         GemmType::MGroupedContiguousWithPsumLayout) {
      this->grouped_layout = grouped_layout;
      current_psum_m = __ldg(grouped_layout);
      num_m_blocks = ceil_div(current_psum_m, BLOCK_M);
    } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
      this->grouped_layout = grouped_layout;
      get_next_k_group(current_group_idx, current_shape_k);
      next_group_idx = current_group_idx + 1;
      get_next_k_group(next_group_idx, next_shape_k);
    }
  }

  __device__ __forceinline__ void get_swizzled_block_idx(
      uint32_t const &block_idx, uint32_t &m_block_idx, uint32_t &n_block_idx) {
    DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumMulticast == 0,
                     "Invalid group size");

    auto const &primary_num_blocks =
        kIsMulticastOnA ? num_n_blocks : num_m_blocks;
    auto const &secondary_num_blocks =
        kIsMulticastOnA ? num_m_blocks : num_n_blocks;
    auto const &num_blocks_per_group =
        secondary_num_blocks * kNum1DBlocksPerGroup;
    auto const &group_idx = block_idx / num_blocks_per_group;
    auto first_block_idx = group_idx * kNum1DBlocksPerGroup;
    auto in_group_idx = block_idx % num_blocks_per_group;
    num_blocks_in_group =
        min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx);

    if constexpr (kIsMulticastOnA) {
      m_block_idx = in_group_idx / num_blocks_in_group;
      n_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
    } else {
      m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group;
      n_block_idx = in_group_idx / num_blocks_in_group;
    }
  }

  template <bool kWithGroupOffset, IndexType kIndexType = IndexType::MN>
  __device__ __forceinline__ uint32_t
      get_global_idx(uint32_t shape_dim,
                     uint32_t block_size,
                     uint32_t const &block_idx,
                     uint32_t const &m_block_idx = 0) {
    if constexpr (kGemmType == GemmType::Normal) {
      return block_idx * block_size;
    } else if constexpr (kGemmType == GemmType::MGroupedContiguous) {
      auto const offset =
          kWithGroupOffset
              ? cute::max(0, __ldg(grouped_layout + m_block_idx * BLOCK_M))
              : 0;
      return offset * shape_dim + block_idx * block_size;
    } else if constexpr (kGemmType == GemmType::MGroupedMasked ||
                         kGemmType ==
                             GemmType::MGroupedContiguousWithPsumLayout) {
      auto const offset = kWithGroupOffset ? current_group_idx : 0;
      return offset * shape_dim + block_idx * block_size;
    } else if constexpr (kGemmType == GemmType::KGroupedContiguous) {
      auto offset = 0;
      if constexpr (kWithGroupOffset) {
        if constexpr (kIndexType == IndexType::MN) {
          offset = current_group_idx * shape_dim;
        } else if constexpr (kIndexType == IndexType::K) {
          offset = current_k_cumsum;
        } else if constexpr (kIndexType == IndexType::SF_K) {
          offset = current_sf_k_cumsum;
        }
      }
      return offset + block_idx * block_size;
    } else {
      auto const offset = kIndexType == IndexType::SF_K ? current_group_idx : 0;
      return offset * shape_dim + block_idx * block_size;
    }
  }

  // In persistent kernel mode, each CTA is a single worker that processes
  // ALL tiles sequentially (kNumSMs=1, block_offset=0).
  uint32_t block_offset = 0;  // persistent kernel: each CTA handles its own 128-row slice

  __device__ __forceinline__ bool get_next_block(uint32_t &m_block_idx,
                                                 uint32_t &n_block_idx) {
    auto const next_block_idx = (++current_iter) * kNumSMs + block_offset;
    // DEBUG: only process first tile to test kernel exit path
    if (current_iter > 0) return false;
    if constexpr (kGemmType == GemmType::Normal) {
      if (next_block_idx >= num_blocks) {
        return false;
      }
      get_swizzled_block_idx(next_block_idx, m_block_idx, n_block_idx);
      return true;
    } else if constexpr (kGemmType == GemmType::Batched) {
      if (next_block_idx >= num_blocks * kNumGroups) {
        return false;
      }
      current_group_idx = next_block_idx / num_blocks;
      auto const &block_idx = next_block_idx - current_group_idx * num_blocks;
      if constexpr (kIsMulticastOnA) {
        m_block_idx = block_idx / num_n_blocks;
        n_block_idx = block_idx % num_n_blocks;
      } else {
        m_block_idx = block_idx % num_m_blocks;
        n_block_idx = block_idx / num_m_blocks;
      }
      return true;
    } else {
      return false;
    }
  }
};

} // namespace mirage::blackwell::linear_fp8_sm100
