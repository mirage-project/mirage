/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#include "common.h"
namespace kernel {
using bfloat16 = type::bfloat16_t;

constexpr int log2_constexpr(int n, int p = 0) {
  return (n <= 1) ? p : log2_constexpr(n >> 1, p + 1);
}

constexpr int max_power_of_two_le(int x) {
  if (x <= 0) {
    return 0;
  }
  int result = 1;
  while ((result << 1) <= x) {
    result <<= 1;
  }
  return result;
}

__device__ __forceinline__ void
    convert_32_f32_to_16_bf16_uint32(float const (&s_frag)[32],
                                     uint32_t (&a_frag)[16]) {
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    bfloat16 low = bfloat16(s_frag[2 * i]);
    bfloat16 high = bfloat16(s_frag[2 * i + 1]);
    a_frag[i] = (static_cast<uint32_t>(high.storage) << 16) | low.storage;
  }
}

__device__ __forceinline__ void
    convert_f32_to_bf16_uint32(float const (&s_frag)[8],
                               uint32_t (&a_frag)[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    bfloat16 low = bfloat16(s_frag[2 * i]);
    bfloat16 high = bfloat16(s_frag[2 * i + 1]);
    a_frag[i] = (static_cast<uint32_t>(high.storage) << 16) | low.storage;
  }
}

__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  float y;
  asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
               : "=f"(y)
               : "f"(x), "r"(lane_mask));
  return y;
}

/*!
 * \brief Wrapper of PTX ex2.approx instruction, which computes 2^x
 * \param x input
 */
__device__ __forceinline__ float ptx_exp2(float x) {
  float y;
  asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

static __device__ __forceinline__ int lane_id() {
  return threadIdx.x & 0x1f;
}

static __device__ __forceinline__ int warp_id() {
  return __shfl_sync(0xffffffff, threadIdx.x / NUM_THREADS_PER_WARP, 0);
}

template <typename T, int NUM_ELEMENTS>
__device__ __forceinline__ void clear_smem_buffer(T *buffer) {
  constexpr int total_bytes = NUM_ELEMENTS * sizeof(T);
  constexpr int num_128bit_writes = total_bytes / 16;
  constexpr int remaining_elements_offset =
      num_128bit_writes * (16 / sizeof(T));

  // Clear the bulk of the buffer using 128-bit writes
  for (int i = threadIdx.x; i < num_128bit_writes; i += NUM_THREADS) {
    ((__uint128_t *)buffer)[i] = 0ul;
  }

  // Handle the tail if the total size is not a multiple of 16 bytes
  if constexpr ((total_bytes % 16) != 0) {
    for (int i = remaining_elements_offset + threadIdx.x; i < NUM_ELEMENTS;
         i += NUM_THREADS) {
      buffer[i] = T(0.0f);
    }
  }
}

static __device__ __forceinline__ void clear_8_floats(float *buffer) {
  *((__uint128_t *)(buffer)) = 0ul;
  *((__uint128_t *)(buffer + 4)) = 0ul;
}

// Vectorized zero initialization struct
template <typename T, int N>
struct vec_zero_t {
  static __device__ __forceinline__ void fill_zero(T *ptr) {
    // Ensure sizeof(T) * N is a multiple of 16 bytes
    static_assert((sizeof(T) * N) % 16 == 0,
                  "sizeof(T) * N must be a multiple of 16 bytes for proper "
                  "vectorized operations");

    constexpr int total_bytes = sizeof(T) * N;
    constexpr int num_chunks = total_bytes / sizeof(__uint128_t);
    __uint128_t *vec_ptr = reinterpret_cast<__uint128_t *>(ptr);
    constexpr int max_iters = (num_chunks + NUM_THREADS - 1) / NUM_THREADS;

#pragma unroll
    for (int i = 0; i < max_iters; ++i) {
      int idx = i * blockDim.x + threadIdx.x;
      if (idx < num_chunks) {
        vec_ptr[idx] = 0ul;
      }
    }
  }
};

} // namespace kernel
