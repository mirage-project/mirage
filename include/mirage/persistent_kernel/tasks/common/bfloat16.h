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
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <limits>

namespace type {

struct alignas(2) bfloat16_t {
  uint16_t storage;

  __host__ __device__ inline static bfloat16_t bitcast(uint16_t x) {
    bfloat16_t h;
    h.storage = x;
    return h;
  }

public:
  bfloat16_t() = default;

  // Reinterpret cast from CUDA's __nv_bfloat16 type
  __host__ __device__ inline explicit bfloat16_t(__nv_bfloat16 const &x) {
#if defined(__CUDA_ARCH__)
    storage = reinterpret_cast<uint16_t const &>(x);
#else
    __nv_bfloat16_raw raw(x);
    std::memcpy(&storage, &raw.x, sizeof(storage));
#endif
  }

  // Floating-point conversion - round toward nearest
  __host__ __device__ inline explicit bfloat16_t(float x) {
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
  }

  // Converts to float
  __host__ __device__ inline operator float() const {
    unsigned bits = (unsigned(storage) << 16);
#if defined(__CUDA_ARCH__)
    return reinterpret_cast<float const &>(bits);
#else
    float flt;
    std::memcpy(&flt, &bits, sizeof(flt));
    return flt;
#endif
  }

  // Bitcasts to CUDA's bf16 type
  __host__ __device__ inline __nv_bfloat16 to_nv_bfloat16() const {
    return reinterpret_cast<__nv_bfloat16 const &>(storage);
  }
};

// Overload operators
__host__ __device__ inline bool operator==(bfloat16_t const &lhs,
                                           bfloat16_t const &rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __heq(lhs.to_nv_bfloat16(), rhs.to_nv_bfloat16());
#else
  return float(lhs) == float(rhs);
#endif
}

__host__ __device__ inline bool operator!=(bfloat16_t const &lhs,
                                           bfloat16_t const &rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __hne(lhs.to_nv_bfloat16(), rhs.to_nv_bfloat16());
#else
  return float(lhs) != float(rhs);
#endif
}

__host__ __device__ inline bool operator<(bfloat16_t const &lhs,
                                          bfloat16_t const &rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __hlt(lhs.to_nv_bfloat16(), rhs.to_nv_bfloat16());
#else
  return float(lhs) < float(rhs);
#endif
}

__host__ __device__ inline bool operator<=(bfloat16_t const &lhs,
                                           bfloat16_t const &rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __hle(lhs.to_nv_bfloat16(), rhs.to_nv_bfloat16());
#else
  return float(lhs) <= float(rhs);
#endif
}

__host__ __device__ inline bool operator>(bfloat16_t const &lhs,
                                          bfloat16_t const &rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __hgt(lhs.to_nv_bfloat16(), rhs.to_nv_bfloat16());
#else
  return float(lhs) > float(rhs);
#endif
}

__host__ __device__ inline bool operator>=(bfloat16_t const &lhs,
                                           bfloat16_t const &rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return __hge(lhs.to_nv_bfloat16(), rhs.to_nv_bfloat16());
#else
  return float(lhs) >= float(rhs);
#endif
}

__host__ __device__ inline bfloat16_t operator+(bfloat16_t const &lhs,
                                                bfloat16_t const &rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return bfloat16_t(__hadd(lhs.to_nv_bfloat16(), rhs.to_nv_bfloat16()));
#else
  return bfloat16_t(float(lhs) + float(rhs));
#endif
}

__host__ __device__ inline bfloat16_t operator-(bfloat16_t const &lhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return bfloat16_t(__hneg(lhs.to_nv_bfloat16()));
#else
  return bfloat16_t(-float(lhs));
#endif
}

__host__ __device__ inline bfloat16_t operator-(bfloat16_t const &lhs,
                                                bfloat16_t const &rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return bfloat16_t(__hsub(lhs.to_nv_bfloat16(), rhs.to_nv_bfloat16()));
#else
  return bfloat16_t(float(lhs) - float(rhs));
#endif
}

__host__ __device__ inline bfloat16_t operator*(bfloat16_t const &lhs,
                                                bfloat16_t const &rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return bfloat16_t(__hmul(lhs.to_nv_bfloat16(), rhs.to_nv_bfloat16()));
#else
  return bfloat16_t(float(lhs) * float(rhs));
#endif
}

__host__ __device__ inline bfloat16_t operator/(bfloat16_t const &lhs,
                                                bfloat16_t const &rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  return bfloat16_t(__hdiv(lhs.to_nv_bfloat16(), rhs.to_nv_bfloat16()));
#else
  return bfloat16_t(float(lhs) / float(rhs));
#endif
}

__host__ __device__ inline bfloat16_t &operator+=(bfloat16_t &lhs,
                                                  bfloat16_t const &rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  lhs = bfloat16_t(__hadd(lhs.to_nv_bfloat16(), rhs.to_nv_bfloat16()));
#else
  lhs = bfloat16_t(float(lhs) + float(rhs));
#endif
  return lhs;
}

__host__ __device__ inline bfloat16_t &operator-=(bfloat16_t &lhs,
                                                  bfloat16_t const &rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  lhs = bfloat16_t(__hsub(lhs.to_nv_bfloat16(), rhs.to_nv_bfloat16()));
#else
  lhs = bfloat16_t(float(lhs) - float(rhs));
#endif
  return lhs;
}

__host__ __device__ inline bfloat16_t &operator*=(bfloat16_t &lhs,
                                                  bfloat16_t const &rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  lhs = bfloat16_t(__hmul(lhs.to_nv_bfloat16(), rhs.to_nv_bfloat16()));
#else
  lhs = bfloat16_t(float(lhs) * float(rhs));
#endif
  return lhs;
}

__host__ __device__ inline bfloat16_t &operator/=(bfloat16_t &lhs,
                                                  bfloat16_t const &rhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  lhs = bfloat16_t(__hdiv(lhs.to_nv_bfloat16(), rhs.to_nv_bfloat16()));
#else
  lhs = bfloat16_t(float(lhs) / float(rhs));
#endif
  return lhs;
}

__host__ __device__ inline bfloat16_t &operator++(bfloat16_t &lhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  lhs = bfloat16_t(
      __hadd(lhs.to_nv_bfloat16(), bfloat16_t(1.0f).to_nv_bfloat16()));
#else
  float tmp(lhs);
  ++tmp;
  lhs = bfloat16_t(tmp);
#endif
  return lhs;
}

__host__ __device__ inline bfloat16_t &operator--(bfloat16_t &lhs) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  lhs = bfloat16_t(
      __hsub(lhs.to_nv_bfloat16(), bfloat16_t(1.0f).to_nv_bfloat16()));
#else
  float tmp(lhs);
  --tmp;
  lhs = bfloat16_t(tmp);
#endif
  return lhs;
}

__host__ __device__ inline bfloat16_t operator++(bfloat16_t &lhs, int) {
  bfloat16_t ret(lhs);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  lhs = bfloat16_t(
      __hadd(lhs.to_nv_bfloat16(), bfloat16_t(1.0f).to_nv_bfloat16()));
#else
  float tmp(lhs);
  tmp++;
  lhs = bfloat16_t(tmp);
#endif
  return ret;
}

__host__ __device__ inline bfloat16_t operator--(bfloat16_t &lhs, int) {
  bfloat16_t ret(lhs);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  lhs = bfloat16_t(
      __hsub(lhs.to_nv_bfloat16(), bfloat16_t(1.0f).to_nv_bfloat16()));
#else
  float tmp(lhs);
  tmp--;
  lhs = bfloat16_t(tmp);
#endif
  return ret;
}

} // namespace type