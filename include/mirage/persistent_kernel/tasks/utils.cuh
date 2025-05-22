

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

__device__ __forceinline__ void
    convert_f32_to_bf16_uint32(float const (&s_frag)[8],
                               uint32_t (&a_frag)[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    type::bfloat16_t low = bfloat16(s_frag[2 * i]);
    type::bfloat16_t high = bfloat16(s_frag[2 * i + 1]);
    a_frag[i] = (static_cast<uint32_t>(high.storage) << 16) | low.storage;
  }
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

// #define DISPATCH_SEQ_LEN(SEQ_LEN, FUNC, ...)                     \
//   [&] {                                                          \
//     if ((SEQ_LEN) <= 8) { return FUNC<8>(__VA_ARGS__); }       \
//     else if ((SEQ_LEN) <= 16>) { return FUNC<16>(__VA_ARGS__); }  \
//     else if ((SEQ_LEN) <= 24) { return FUNC<24>(__VA_ARGS__); }  \
//     else if ((SEQ_LEN) <= 32) { return FUNC<32>(__VA_ARGS__); }\
//     else if ((SEQ_LEN) <= 40>) { return FUNC<40>(__VA_ARGS__); }  \
//     else if ((SEQ_LEN) <= 48) { return FUNC<48>(__VA_ARGS__); }  \
//     else if ((SEQ_LEN) <= 56) { return FUNC<56>(__VA_ARGS__); }\
//     else if ((SEQ_LEN) <= 64>) { return FUNC<64>(__VA_ARGS__); }  \
//     else if ((SEQ_LEN) <= 72) { return FUNC<72>(__VA_ARGS__); }  \
//     else if ((SEQ_LEN) <= 80) { return FUNC<80>(__VA_ARGS__); }\
//     else if ((SEQ_LEN) <= 88) { return FUNC<88>(__VA_ARGS__); }  \
//     else if ((SEQ_LEN) <= 96) { return FUNC<96>(__VA_ARGS__); }  \
//     else if ((SEQ_LEN) <= 104) { return FUNC<104>(__VA_ARGS__); }\
//     else if ((SEQ_LEN) <= 112) { return FUNC<112>(__VA_ARGS__); }  \
//     else if ((SEQ_LEN) <= 120) { return FUNC<120>(__VA_ARGS__); }  \
//     else if ((SEQ_LEN) <= 128) { return FUNC<128>(__VA_ARGS__); }\
//     else {                                                       \
//       printf("Unsupported seq_len: %zu\n", SEQ_LEN);             \
//       return;                                                    \
//     }                                                            \
// }()

} // namespace kernel