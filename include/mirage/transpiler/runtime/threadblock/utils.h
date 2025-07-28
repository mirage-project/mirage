// utils.h - Implementation of warpgroup level helper functions

#include "../../../config.h"
#pragma once

namespace tb {

static __device__ __forceinline__ bool block0() {
  return (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0);
}

static __device__ __forceinline__ int lane_id() {
  return threadIdx.x & 0x1f;
}

static __device__ __forceinline__ int warp_id_in_wg() {
  return __shfl_sync(0xffffffff,
                     (threadIdx.x / mirage::config::NUM_THREADS_PER_WARP) %
                         mirage::config::NUM_WARPS_PER_GROUP,
                     0);
}

static __device__ __forceinline__ int warp_id() {
  return __shfl_sync(
      0xffffffff, threadIdx.x / mirage::config::NUM_THREADS_PER_WARP, 0);
}

static __device__ __forceinline__ int warpgroup_id() {
  return __shfl_sync(
      0xffffffff, threadIdx.x / mirage::config::NUM_THREADS_PER_GROUP, 0);
}

// decrease register files in a wg
template <uint32_t RegCount>
static __device__ __forceinline__ void wg_decrease_regs() {
#ifdef MIRAGE_GRACE_HOPPER
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

// increase register files in a wg
template <uint32_t RegCount>
static __device__ __forceinline__ void wg_increase_regs() {
#ifdef MIRAGE_GRACE_HOPPER
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

// sync inside a warp group
template <int GROUP_THREADS>
static __device__ __forceinline__ void wg_sync(uint32_t barrier_id) {
#if defined(MIRAGE_GRACE_HOPPER) || defined(MIRAGE_BLACKWELL)
  asm volatile("bar.sync %0, %1;\n" ::"r"(barrier_id), "n"(GROUP_THREADS));
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

template <int GROUP_THREADS>
static __device__ __forceinline__ void wg_arrive(uint32_t barrier_id) {
#ifdef MIRAGE_GRACE_HOPPER
  // cutlass::arch::synclog_emit_named_barrier_arrive(__LINE__, num_threads,
  // barrier_id);
  asm volatile("bar.arrive %0, %1;" : : "r"(barrier_id), "r"(GROUP_THREADS));
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

// Barrier wait
static __device__ __forceinline__ void
    wait_barrier(uint64_t &smem_barrier, // 64 bits user-manged barrier in smem
                 int phase_bit) // Current phase bit the barrier waiting to flip
{
#if defined(MIRAGE_GRACE_HOPPER) || defined(MIRAGE_BLACKWELL)
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  asm volatile("{\n"
               ".reg .pred                P1;\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@P1                       bra DONE;\n"
               "bra                   LAB_WAIT;\n"
               "DONE:\n"
               "}\n" ::"r"(smem_int_ptr),
               "r"(phase_bit));

#endif
}

template <typename TiledMMA, typename ClusterShape_MNK>
static __device__ __forceinline__ auto get_cluster_layout() {
  return tiled_divide(make_layout(ClusterShape_MNK{}),
                      make_tile(typename TiledMMA::AtomThrID{}));
}

template <typename TiledMMA, typename ClusterShape_MNK>
static __device__ __forceinline__ auto get_mma_coord_vmnk(int blockIdx_x,
                                                          int blockIdx_y) {
  auto cluster_layout = get_cluster_layout<TiledMMA, ClusterShape_MNK>();
  return make_coord(blockIdx_x % size<0>(cluster_layout),
                    blockIdx_x / size<0>(cluster_layout),
                    blockIdx_y,
                    _);
}

template <typename TiledMMA, typename ClusterShape_MNK>
static __device__ __forceinline__ auto get_cta_mma(int blockIdx_x,
                                                   int blockIdx_y) {
  TiledMMA tiled_mma;
  auto mma_coord_vmnk =
      get_mma_coord_vmnk<TiledMMA, ClusterShape_MNK>(blockIdx_x, blockIdx_y);
  auto mma_v = get<0>(mma_coord_vmnk);
  return tiled_mma.get_slice(mma_v);
}

} // namespace tb
