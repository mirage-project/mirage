#pragma once

#include <cutlass/arch/barrier.h>
#include <cute/arch/copy_sm100_tma.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

namespace deep_gemm {

template <uint32_t BLOCK_INNER, uint32_t kSwizzleMode, typename dtype_t>
constexpr uint32_t get_inner_block_atom_size() {
  return kSwizzleMode == 0 ? BLOCK_INNER : kSwizzleMode / sizeof(dtype_t);
}

template <uint32_t BLOCK_INNER,
          uint32_t BLOCK_OUTER,
          uint32_t kSwizzleMode,
          typename dtype_t,
          bool kIs3DTMA = false>
__device__ __forceinline__ void
tma_copy(void const *desc_ptr,
         cutlass::arch::ClusterTransactionBarrier *barrier_ptr,
         dtype_t *smem_ptr,
         const uint32_t &inner_idx,
         const uint32_t &outer_idx,
         const uint32_t &num_tma_multicast = 1,
         const uint32_t &batch_idx = 0) {
  DG_STATIC_ASSERT(
      static_cast<uint64_t>(cute::TMA::CacheHintSm90::EVICT_NORMAL) ==
          static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
      "Invalid cache hint");
  constexpr uint32_t BLOCK_INNER_ATOM =
      get_inner_block_atom_size<BLOCK_INNER, kSwizzleMode, dtype_t>();

  if constexpr (!kIs3DTMA) {
    if (num_tma_multicast == 1) {
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++i) {
        cute::SM90_TMA_LOAD_2D::copy(
            desc_ptr,
            reinterpret_cast<uint64_t *>(barrier_ptr),
            static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
            smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
            inner_idx + i * BLOCK_INNER_ATOM,
            outer_idx);
      }
    } else {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++i) {
        cute::SM100_TMA_2SM_LOAD_2D::copy(
            desc_ptr,
            reinterpret_cast<uint64_t *>(barrier_ptr),
            static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
            smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
            inner_idx + i * BLOCK_INNER_ATOM,
            outer_idx);
      }
#endif
    }
  } else {
    if (num_tma_multicast == 1) {
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++i) {
        cute::SM90_TMA_LOAD_3D::copy(
            desc_ptr,
            reinterpret_cast<uint64_t *>(barrier_ptr),
            static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
            smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
            inner_idx + i * BLOCK_INNER_ATOM,
            outer_idx,
            batch_idx);
      }
    } else {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000))
#pragma unroll
      for (uint32_t i = 0; i < BLOCK_INNER / BLOCK_INNER_ATOM; ++i) {
        cute::SM100_TMA_2SM_LOAD_3D::copy(
            desc_ptr,
            reinterpret_cast<uint64_t *>(barrier_ptr),
            static_cast<uint64_t>(cute::TMA::CacheHintSm100::EVICT_NORMAL),
            smem_ptr + i * BLOCK_OUTER * BLOCK_INNER_ATOM,
            inner_idx + i * BLOCK_INNER_ATOM,
            outer_idx,
            batch_idx);
      }
#endif
    }
  }
}

__device__ __forceinline__ void tensor_map_release_cta() {
  asm volatile("fence.proxy.tensormap::generic.release.cta;");
}

__device__ __forceinline__ void
tensor_map_acquire_cta(const cute::TmaDescriptor *gmem_desc_ptr) {
  auto gmem_int_desc = reinterpret_cast<uint64_t>(gmem_desc_ptr);
  asm volatile("fence.proxy.tensormap::generic.acquire.cta [%0], 128;"
               :
               : "l"(gmem_int_desc)
               : "memory");
}

__device__ __forceinline__ void
tensor_map_replace_global_addr_in_smem(cute::TmaDescriptor *smem_desc,
                                       const void *new_addr) {
  auto smem_int_desc = static_cast<uint32_t>(__cvta_generic_to_shared(smem_desc));
  const auto new_int64_addr = reinterpret_cast<uint64_t>(new_addr);
  asm volatile(
      "tensormap.replace.tile.global_address.shared::cta.b1024.b64 [%0], %1;"
      :
      : "r"(smem_int_desc), "l"(new_int64_addr));
}

} // namespace deep_gemm
