#pragma once

#include <cstdint>
#include <cute/arch/mma_sm100_umma.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace kernel::sm100 {
// adapted from the DeepGEMM SM100 Blackwell path: provide the minimum
// descriptor, TMEM, block-scaled MMA helpers needed by Mirage's block-scaled
// FP8 linear kernel

__device__ __forceinline__
    // Build a generic UMMA shared-memory descriptor
    cute::UMMA::SmemDescriptor
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

__device__ __forceinline__
    // Build the shared-memory descriptor used by the scale-factor UTCCP path

    // In the current Mirage block-scaled FP8 implementation, scale factors are
    // first staged in shared memory, then copied into TMEM scale columns
    // through UTCCP This descriptor uses a non-swizzled layout because the
    // required reordering is handled explicitly before UTCCP
    cute::UMMA::SmemDescriptor
    make_sf_desc(void *smem_ptr) {
  // UTCCP scale path: 8 x 128b atom, K-major-like staging
  return make_smem_desc(
      cute::UMMA::LayoutType::SWIZZLE_NONE, smem_ptr, 8 * 16, 0);
}

__device__ __forceinline__
    // Update only the address field of an existing UMMA shared-memory
    // descriptor
    void
    replace_smem_desc_addr(cute::UMMA::SmemDescriptor &desc,
                           void const *smem_ptr) {
  auto const uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
  desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
}

template <cute::UMMA::Major kMajorMode,
          uint32_t BLOCK_MN,
          uint32_t kSwizzleMode,
          typename dtype_t>
__device__ __forceinline__ uint32_t advance_umma_desc_lo(
    uint32_t const &base, uint32_t const &offset, uint32_t const &k_idx) {
  static_assert(kMajorMode == cute::UMMA::Major::K,
                "Current Mirage helper only supports K-major path.");
  // For K-major SW128 path, descriptor increments linearly along K.
  return base +
         (((offset + k_idx) * static_cast<uint32_t>(sizeof(dtype_t))) >> 4u);
}

template <cute::UMMA::Major kMajorMode,
          uint32_t BLOCK_MN,
          uint32_t BLOCK_K,
          uint32_t kSwizzleMode,
          bool kUseBase32 = false,
          typename dtype_t>
__device__ __forceinline__ cute::UMMA::SmemDescriptor
    make_umma_desc(dtype_t *base_smem_ptr, uint32_t mn_idx, uint32_t k_idx) {
  static_assert(kMajorMode == cute::UMMA::Major::K,
                "Current Mirage helper only supports K-major path.");
  static_assert(
      kSwizzleMode == BLOCK_K * sizeof(dtype_t),
      "For K-major path, swizzle mode must equal BLOCK_K * sizeof(dtype_t).");
  constexpr auto layout_type = cute::UMMA::LayoutType::SWIZZLE_128B;

  // K-major:
  // atom size = 8 x 128B on K axis
  // SBO = one atom step across MN
  // LBO = 0 because one swizzle atom on K axis inside a block
  const uint32_t stride_byte_offset = 8 * BLOCK_K * sizeof(dtype_t);
  const uint32_t leading_byte_offset = 0;

  return make_smem_desc(layout_type,
                        base_smem_ptr + mn_idx * BLOCK_K + k_idx,
                        stride_byte_offset,
                        leading_byte_offset);
}

__device__ __forceinline__ uint64_t make_runtime_instr_desc_with_sf_id(
    cute::UMMA::InstrDescriptorBlockScaled desc,
    uint32_t const &sfa_id,
    uint32_t const &sfb_id) {
  desc.a_sf_id_ = sfa_id;
  desc.b_sf_id_ = sfb_id;
  return static_cast<uint64_t>(static_cast<uint32_t>(desc)) << 32;
}

// Thin inline-PTX wrapper for the Blackwell tcgen05 block-scaled MMA
// instruction
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
                 "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale "
                 "[%0], %1, %2, %3, [%5], [%6], p;\n\t"
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

} // namespace kernel::sm100
