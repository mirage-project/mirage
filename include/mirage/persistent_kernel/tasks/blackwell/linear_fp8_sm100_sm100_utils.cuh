#pragma once

#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cutlass/arch/barrier.h>

#include "linear_fp8_sm100_tma_utils.cuh"
#include "linear_fp8_sm100_utils.cuh"

namespace mirage::blackwell::linear_fp8_sm100::sm100 {

__device__ __forceinline__ cute::UMMA::SmemDescriptor
    make_smem_desc(cute::UMMA::LayoutType layout,
                   void *smem_ptr,
                   uint32_t stride_byte_offset,
                   uint32_t leading_byte_offset) {
  cute::UMMA::SmemDescriptor desc;
  desc.version_ = 1;
  desc.lbo_mode_ = 0;
  desc.layout_type_ = static_cast<uint8_t>(layout);
  auto const uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
  desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
  desc.base_offset_ = 0;
  desc.stride_byte_offset_ = stride_byte_offset >> 4;
  desc.leading_byte_offset_ = leading_byte_offset >> 4;
  return desc;
}

__device__ __forceinline__ cute::UMMA::SmemDescriptor
    make_sf_desc(void *smem_ptr) {
  return make_smem_desc(
      cute::UMMA::LayoutType::SWIZZLE_NONE, smem_ptr, 8 * 16, 0);
}

__device__ __forceinline__ void
    replace_smem_desc_addr(cute::UMMA::SmemDescriptor &desc,
                           void const *smem_ptr) {
  auto const uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
  desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
}

__device__ __forceinline__ static uint32_t
    get_atom_base(cute::UMMA::LayoutType const &layout_type) {
  return layout_type == cute::UMMA::LayoutType::SWIZZLE_128B_BASE32B ? 32 : 16;
}

template <cute::UMMA::Major kMajorMode,
          uint32_t kSwizzleMode,
          bool kUseBase32,
          typename dtype_t>
constexpr static cute::UMMA::LayoutType to_umma_layout_type() {
  DG_STATIC_ASSERT(kSwizzleMode == 0 || kSwizzleMode == 16 ||
                       kSwizzleMode == 32 || kSwizzleMode == 64 ||
                       kSwizzleMode == 128,
                   "Invalid swizzling mode");
  if constexpr ((cute::is_same_v<dtype_t, float> &&
                 kMajorMode == cute::UMMA::Major::MN) ||
                kUseBase32) {
    DG_STATIC_ASSERT(kUseBase32, "Invalid swizzling base");
    return cute::UMMA::LayoutType::SWIZZLE_128B_BASE32B;
  }

  if constexpr (kSwizzleMode == 0) {
    return cute::UMMA::LayoutType::SWIZZLE_NONE;
  }
  if constexpr (kSwizzleMode == 16) {
    return cute::UMMA::LayoutType::SWIZZLE_NONE;
  }
  if constexpr (kSwizzleMode == 32) {
    return cute::UMMA::LayoutType::SWIZZLE_32B;
  }
  if constexpr (kSwizzleMode == 64) {
    return cute::UMMA::LayoutType::SWIZZLE_64B;
  }
  if constexpr (kSwizzleMode == 128) {
    return cute::UMMA::LayoutType::SWIZZLE_128B;
  }
}

template <cute::UMMA::Major kMajorMode,
          uint32_t BLOCK_MN,
          uint32_t kSwizzleMode,
          typename dtype_t>
__device__ __forceinline__ constexpr uint32_t get_umma_desc_stride_k() {
  return kMajorMode == cute::UMMA::Major::K
             ? 1
             : get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();
}

template <cute::UMMA::Major kMajorMode,
          uint32_t BLOCK_MN,
          uint32_t kSwizzleMode,
          typename dtype_t>
__device__ __forceinline__ uint32_t advance_umma_desc_lo(
    uint32_t const &base, uint32_t const &offset, uint32_t const &k_idx) {
  return base + (((offset + k_idx * get_umma_desc_stride_k<kMajorMode,
                                                           BLOCK_MN,
                                                           kSwizzleMode,
                                                           dtype_t>()) *
                  static_cast<uint32_t>(sizeof(dtype_t))) >>
                 4u);
}

template <cute::UMMA::Major kMajorMode,
          uint32_t BLOCK_MN,
          uint32_t BLOCK_K,
          uint32_t kSwizzleMode,
          bool kUseBase32 = false,
          typename dtype_t>
__device__ __forceinline__ cute::UMMA::SmemDescriptor
    make_umma_desc(dtype_t *base_smem_ptr, uint32_t mn_idx, uint32_t k_idx) {
  uint32_t const stride_k =
      get_umma_desc_stride_k<kMajorMode, BLOCK_MN, kSwizzleMode, dtype_t>();
  auto const &layout_type =
      to_umma_layout_type<kMajorMode, kSwizzleMode, kUseBase32, dtype_t>();
  auto const &num_non_contiguous = 128 / get_atom_base(layout_type);
  if constexpr (kMajorMode == cute::UMMA::Major::K) {
    DG_STATIC_ASSERT(kSwizzleMode == BLOCK_K * sizeof(dtype_t),
                     "Unexpected value");
    uint32_t const stride_byte_offset =
        num_non_contiguous * BLOCK_K * sizeof(dtype_t);
    uint32_t const leading_byte_offset = 0;
    return make_smem_desc(layout_type,
                          base_smem_ptr + mn_idx * BLOCK_K + k_idx * stride_k,
                          stride_byte_offset,
                          leading_byte_offset);
  } else {
    constexpr uint32_t BLOCK_MN_ATOM =
        get_inner_block_atom_size<BLOCK_MN, kSwizzleMode, dtype_t>();
    DG_DEVICE_ASSERT(mn_idx % BLOCK_MN_ATOM == 0);
    DG_STATIC_ASSERT(kSwizzleMode > 0, "Invalid swizzling");

    uint32_t stride_byte_offset =
        num_non_contiguous * BLOCK_MN_ATOM * sizeof(dtype_t);
    uint32_t leading_byte_offset = BLOCK_K * BLOCK_MN_ATOM * sizeof(dtype_t);
    if constexpr (kSwizzleMode == 16) {
      swap(stride_byte_offset, leading_byte_offset);
    }
    return make_smem_desc(layout_type,
                          base_smem_ptr + mn_idx * BLOCK_K + k_idx * stride_k,
                          stride_byte_offset,
                          leading_byte_offset);
  }
}

__device__ __forceinline__ uint64_t make_runtime_instr_desc_with_sf_id(
    cute::UMMA::InstrDescriptorBlockScaled desc,
    uint32_t const &sfa_id,
    uint32_t const &sfb_id) {
  desc.a_sf_id_ = sfa_id;
  desc.b_sf_id_ = sfb_id;
  return static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32;
}

template <uint32_t kNumCols>
__device__ constexpr uint32_t get_num_aligned_tmem_cols() {
  DG_STATIC_ASSERT(kNumCols <= 512, "Too many tensor memory columns");
  if (kNumCols <= 32) {
    return 32;
  }
  if (kNumCols <= 64) {
    return 64;
  }
  if (kNumCols <= 128) {
    return 128;
  }
  if (kNumCols <= 256) {
    return 256;
  }
  return 512;
}

__device__ __forceinline__ void tcgen05_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ __forceinline__ void tcgen05_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

struct SM100_MMA_MXF8F6F4_SS {
  __device__ static void fma(uint64_t const &desc_a,
                             uint64_t const &desc_b,
                             uint32_t const &tmem_c,
                             uint32_t const &scale_c,
                             uint64_t const &desc,
                             uint32_t const &tmem_sfa,
                             uint32_t const &tmem_sfb) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], "
                 "%1, %2, %3, [%5], [%6], p; \n\t"
                 "}\n"
                 :
                 : "r"(tmem_c),
                   "l"(desc_a),
                   "l"(desc_b),
                   "r"(static_cast<uint32_t>(desc >> 32)),
                   "r"(scale_c),
                   "r"(tmem_sfa),
                   "r"(tmem_sfb));
  }
};

struct SM100_MMA_MXF8F6F4_2x1SM_SS {
  __device__ static void fma(uint64_t const &desc_a,
                             uint64_t const &desc_b,
                             uint32_t const &tmem_c,
                             uint32_t const &scale_c,
                             uint64_t const &desc,
                             uint32_t const &tmem_sfa,
                             uint32_t const &tmem_sfb) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], "
                 "%1, %2, %3, [%5], [%6], p; \n\t"
                 "}\n"
                 :
                 : "r"(tmem_c),
                   "l"(desc_a),
                   "l"(desc_b),
                   "r"(static_cast<uint32_t>(desc >> 32)),
                   "r"(scale_c),
                   "r"(tmem_sfa),
                   "r"(tmem_sfb));
  }
};

} // namespace mirage::blackwell::linear_fp8_sm100::sm100
