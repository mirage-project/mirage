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
#include <cassert>
#include <cstdint>
#include <cuda_bf16.h>
constexpr int NUM_THREADS = 128;
constexpr int NUM_THREADS_PER_WARP = 32;

inline __nv_bfloat16 float2bfloat16(float x) {
  uint16_t storage;

  // Use CUDA intrinsic for conversion if available
  // This is only available in CUDA 11.0 and later, and for architectures
  // with compute capability >= 8.0 (Ampere and later)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) &&                        \
    (__CUDACC_VER_MAJOR__ >= 11)

  asm("cvt.rn.bf16.f32 %0, %1;\n" : "=h"(storage) : "f"(x));

#else
  uint32_t bits;

#if defined(__CUDA_ARCH__)
  bits = reinterpret_cast<uint32_t &>(x);
#else
  std::memcpy(&bits, &x, sizeof(bits));
#endif

  if ((bits & 0x7f800000) != 0x7f800000) {

    bool mantissa_bit = ((bits & (1 << 16)) != 0);
    bool round_bit = ((bits & (1 << 15)) != 0);
    bool sticky_bit = ((bits & ((1 << 15) - 1)) != 0);

    if ((round_bit && sticky_bit) || (round_bit && mantissa_bit)) {
      bits += uint32_t(1 << 16);
    }
  } else if (bits & ~0xff800000) {
    bits = 0x7fffffff;
  }

  storage = uint16_t((bits >> 16) & 0xffff);
#endif
  return reinterpret_cast<__nv_bfloat16 &>(storage);
}

inline float bfloat162float(__nv_bfloat16 x) {
  uint32_t bits = (reinterpret_cast<uint32_t &>(x) << 16);
#if defined(__CUDA_ARCH__)
  return reinterpret_cast<float const &>(bits);
#else
  float flt;
  std::memcpy(&flt, &bits, sizeof(flt));
  return flt;
#endif
}