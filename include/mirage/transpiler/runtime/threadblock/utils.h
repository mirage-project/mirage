// utils.h - Implementation of warpgroup level helper functions

#include "../../../config.h"
#pragma once

namespace tb {

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
static __device__ __forceinline__ void warpgroup_sync(uint32_t barrier_id,
                                                      int x) {

#ifdef MIRAGE_GRACE_HOPPER
  asm volatile("bar.sync %0, %1;\n" ::"r"(barrier_id),
               "n"(mirage::config::NUM_THREADS_PER_GROUP));
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

} // namespace tb
