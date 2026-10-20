// tasks/sm90_common.cuh -- what the sm90_* task types share: each runs an MPK
// Hopper task body (include/mirage/persistent_kernel/tasks/hopper/) as a task
// of the static megakernel. The body reads and writes its tensors by pointer
// (graph tensors: the node's buffer slots) and takes the dynamic shared memory
// from its start, over the ring.
#pragma once
#include "../core.cuh"
#include <cuda_fp16.h>
#include <cute/numeric/numeric_types.hpp> // cute::bfloat16_t, the bodies' bf16

namespace static_mk {

// Before the body: every producer node of IN... (a graph input's StaticNode<0,
// 0, 0> has x == 0) has counted all its tasks, and this SM's earlier tasks are
// done with the shared memory (the ring's jobs end with their last warp, so the
// first barrier waits for them too).
template <class... IN>
__device__ __forceinline__ void sm90_task_begin(G const &g) {
  __syncthreads();
  if (threadIdx.x == 0) {
    (
        [&] {
          if constexpr (IN::x > 0 && IN::counter >= 0) {
            cnt_wait(g.cnt + IN::counter, IN::x * IN::y * IN::z);
          }
        }(),
        ...);
  }
  __syncthreads();
}

// Whether the producers of the inputs in MASK (bit i: IN's i-th) have counted
// all their tasks; sm90_wait_inputs: wait until they have (thread 0 spins,
// nothing synchronizes)
template <uint32_t MASK, class... IN>
__device__ __forceinline__ bool sm90_inputs_ready(G const &g) {
  bool ready = true;
  int i = 0;
  (
      [&] {
        if constexpr (IN::x > 0 && IN::counter >= 0) {
          if ((MASK >> i) & 1u) {
            ready =
                ready && cnt_ld(g.cnt + IN::counter) >= IN::x * IN::y * IN::z;
          }
        }
        i++;
      }(),
      ...);
  return ready;
}
template <uint32_t MASK, class... IN>
__device__ __forceinline__ void sm90_wait_inputs(G const &g) {
  while (!sm90_inputs_ready<MASK, IN...>(g)) {
    __nanosleep(200);
  }
}

// After the body: its outputs are written; count the task in its node's
// counter (release: the CTA's stores, ordered by the barrier, before it).
template <class SELF>
__device__ __forceinline__ void sm90_task_end(G const &g) {
  __syncthreads();
  if constexpr (SELF::counter >= 0) {
    if (threadIdx.x == 0) {
      cnt_add(g.cnt + SELF::counter, 1);
    }
  }
}

// 8 bf16 or f16 <-> fp32 (one 16-B vector)
__device__ __forceinline__ void bf16x8_to_f32(uint4 v, float (&f)[8]) {
  __nv_bfloat162 const *h = reinterpret_cast<__nv_bfloat162 const *>(&v);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 const t = __bfloat1622float2(h[i]);
    f[2 * i] = t.x, f[2 * i + 1] = t.y;
  }
}
__device__ __forceinline__ void f16x8_to_f32(uint4 v, float (&f)[8]) {
  __half2 const *h = reinterpret_cast<__half2 const *>(&v);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float2 const t = __half22float2(h[i]);
    f[2 * i] = t.x, f[2 * i + 1] = t.y;
  }
}
__device__ __forceinline__ uint4 f32_to_f16x8(float const (&f)[8]) {
  uint4 v;
  __half2 *h = reinterpret_cast<__half2 *>(&v);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    h[i] = __floats2half2_rn(f[2 * i], f[2 * i + 1]);
  }
  return v;
}
__device__ __forceinline__ uint4 f32_to_bf16x8(float const (&f)[8]) {
  uint4 v;
  __nv_bfloat162 *h = reinterpret_cast<__nv_bfloat162 *>(&v);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    h[i] = __floats2bfloat162_rn(f[2 * i], f[2 * i + 1]);
  }
  return v;
}

// the dynamic shared memory an MPK Hopper body may take
constexpr int SM90_BODY_SMEM = 200 * 1024;

} // namespace static_mk
