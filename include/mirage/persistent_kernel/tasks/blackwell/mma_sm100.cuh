/*  Copyright 2025 CMU
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once
#include "common.h"
#include "copy_sm100.cuh"
#include "dmem_layout.cuh"
#include "element_binary.cuh"
#include "element_unary.cuh"
#include "mma.cuh"
#include "reduction.cuh"
#include "smem_layout.cuh"
#include "utils.cuh"

namespace kernel {
namespace blackwell {

using bfloat16 = type::bfloat16_t;

// borrowed from
// Megakernels/ThunderKittens/include/ops/thread/mma/tensor/tensor.cuh
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#instruction-descriptor
template <typename D,
          typename AB,
          int M,
          int N,
          bool trans_a,
          bool trans_b,
          bool neg = false>
__device__ static inline uint32_t mma_instruction_descriptor() {
  uint32_t desc = 0;
  if constexpr (sizeof(AB) == 2) { // kind::f16
    // either accumulate to float, or the input is half and the output is half
    static_assert(std::is_same_v<D, float> || std::is_same_v<AB, half>);
    desc |= 0b00 << 0; // sparsity bits unneeded
    desc |= 0b0 << 2;  // dense
    desc |= 0b0 << 3;  // no saturate on fp types
    if constexpr (std::is_same_v<D, float>) {
      desc |= 0b01 << 4; // D matrix is FP32
    } else {
      desc |= 0b00 << 4; // D matrix is FP16
    }
    desc |= 0b0 << 6; // reserved
    if constexpr (std::is_same_v<AB, half>) {
      desc |= 0b000 << 7;  // 16-bit A input type as FP16
      desc |= 0b000 << 10; // 16-bit B input type as FP16
    } else if constexpr (std::is_same_v<AB, bfloat16>) {
      desc |= 0b001 << 7;  // 16-bit A input type as BF16
      desc |= 0b001 << 10; // 16-bit B input type as BF16
    }
    /* fp8
    else if constexpr (std::is_same_v<AB, fp8e4m3>) {
        desc |= 0b000 << 7;  // 8-bit A input type as FP8 e4m3
        desc |= 0b000 << 10; // 8-bit B input type as FP8 e4m3
    } else if constexpr (std::is_same_v<AB, fp8e5m2>) {
        desc |= 0b001 << 7;  // 8-bit A input type as FP8 e5m2
        desc |= 0b001 << 10; // 8-bit B input type as FP8 e5m2
    }
    */
    /* fp6 and fp4
    else if constexpr (std::is_same_v<AB, fp6e2m3>) {
        desc |= 0b011 << 7;  // 6-bit A input type as FP6 e2m3
        desc |= 0b011 << 10; // 6-bit B input type as FP6 e2m3
    }
    else if constexpr (std::is_same_v<AB, fp4e2m3>) {
        desc |= 0b100 << 7;  // 6-bit A input type as FP6 e3m2
        desc |= 0b100 << 10; // 6-bit B input type as FP6 e3m2
    }
    else if constexpr (std::is_same_v<AB, fp4e3m1>) {
        desc |= 0b101 << 7;  // 4-bit A input type as FP4 e3m1
        desc |= 0b101 << 10; // 4-bit B input type as FP4 e3m1
    }
    */
    if constexpr (neg) {
      desc |= 0b1 << 13; // Do negate A matrix
    } else {
      desc |= 0b0 << 13; // Don't negate A matrix
    }
    desc |= 0b0 << 14; // Don't negate B matrix (in all cases)
    if constexpr (trans_a) {
      desc |= 0b1 << 15; // Transpose A matrix
    } else {
      desc |= 0b0 << 15; // Don't transpose A matrix
    }
    if constexpr (trans_b) {
      desc |= 0b1 << 16; // Transpose B matrix
    } else {
      desc |= 0b0 << 16; // Don't transpose B matrix
    }
    desc |= (N >> 3) << 17;               // B matrix has dimension N, encoded
    desc |= 0b0 << 23;                    // reserved
    desc |= (M >> 4) << 24;               // A matrix has dimension M, encoded
    desc |= 0b0 << 29;                    // reserved
    desc |= 0b00 << 30;                   // no shift for B-matrix reuse
  } else if constexpr (sizeof(AB) == 1) { // kind::f8f6f4
    static_assert(
        std::is_same_v<D, float> ||
        std::is_same_v<D, half>); // FP8/6/4 has to accumulate to float or half
    desc |= 0b00 << 0;            // sparsity bits unneeded
    desc |= 0b0 << 2;             // dense
    desc |= 0b0 << 3;             // no saturate on fp types
    if constexpr (std::is_same_v<D, float>) {
      desc |= 0b01 << 4; // D matrix is FP32
    } else {
      desc |= 0b00 << 4; // D matrix is FP16
    }
    desc |= 0b0 << 6; // reserved
    /* fp8
    if constexpr (std::is_same_v<AB, fp8e4m3>) {
        desc |= 0b000 << 7;  // 8-bit A input type as FP8 e4m3
        desc |= 0b000 << 10; // 8-bit B input type as FP8 e4m3
    } else if constexpr (std::is_same_v<AB, fp8e5m2>) {
        desc |= 0b001 << 7;  // 8-bit A input type as FP8 e5m2
        desc |= 0b001 << 10; // 8-bit B input type as FP8 e5m2
    }
    */
    /* fp6 and fp4
    else if constexpr (std::is_same_v<AB, fp6e2m3>) {
        desc |= 0b011 << 7;  // 6-bit A input type as FP6 e2m3
        desc |= 0b011 << 10; // 6-bit B input type as FP6 e2m3
    }
    else if constexpr (std::is_same_v<AB, fp4e2m3>) {
        desc |= 0b100 << 7;  // 6-bit A input type as FP6 e3m2
        desc |= 0b100 << 10; // 6-bit B input type as FP6 e3m2
    }
    else if constexpr (std::is_same_v<AB, fp4e3m1>) {
        desc |= 0b101 << 7;  // 4-bit A input type as FP4 e3m1
        desc |= 0b101 << 10; // 4-bit B input type as FP4 e3m1
    }
    */
    if constexpr (neg) {
      desc |= 0b1 << 13; // Do negate A matrix
    } else {
      desc |= 0b0 << 13; // Don't negate A matrix
    }
    desc |= 0b0 << 14; // Don't negate B matrix (in all cases)
    if constexpr (trans_a) {
      desc |= 0b1 << 15; // Transpose A matrix
    } else {
      desc |= 0b0 << 15; // Don't transpose A matrix
    }
    if constexpr (trans_b) {
      desc |= 0b1 << 16; // Transpose B matrix
    } else {
      desc |= 0b0 << 16; // Don't transpose B matrix
    }
    desc |= (N >> 3) << 17; // B matrix has dimension N, encoded
    desc |= 0b0 << 23;      // reserved
    desc |= (M >> 4) << 24; // A matrix has dimension M, encoded
    desc |= 0b0 << 29;      // reserved
    desc |= 0b00 << 30;     // no shift for B-matrix reuse
  } else {
    static_assert(sizeof(AB) == 999,
                  "Invalid AB type size; not implemented yet.");
  }
  return desc;
};

template <typename T>
__device__ __forceinline__ uint64_t make_smem_desc(T *smem_ptr,
                                                   uint32_t lbo_bytes,
                                                   uint32_t sbo_bytes,
                                                   uint32_t swizzle = 0,
                                                   uint32_t mbase_off = 0,
                                                   bool lbo_mode = false) {
  uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  auto pack = [](uint32_t v) { return (uint64_t)(v >> 4) & 0x3FFFULL; };
  uint64_t desc = pack(smem_addr)                     // bits 0-13
                  | (pack(lbo_bytes) << 16)           // bits 16-29
                  | (pack(sbo_bytes) << 32)           // bits 32-45
                  | ((uint64_t)mbase_off & 0x7) << 49 // bits 49-51
                  | ((uint64_t)lbo_mode & 0x1) << 52  // bit 52
                  | ((uint64_t)swizzle & 0x7) << 61;  // bits 61-63
  return desc;
}

__device__ inline static void
    call_mma_m64n64k16_bf16bf16f32(uint32_t *d_addr,
                                   uint64_t const &a_desc,
                                   uint64_t const &b_desc,
                                   uint32_t idesc,
                                   bool accumulate = false) {

  if (threadIdx.x % 32 == 0) {
    auto uint_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(d_addr));
    uint32_t mask[4] = {0, 0, 0, 0};
    uint32_t scaleC = 1;
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3,{%5, %6, "
                 "%7, %8}, p; \n\t"
                 "}\n"
                 :
                 : "r"(uint_ptr),
                   "l"(a_desc),
                   "l"(b_desc),
                   "r"(idesc),
                   "r"(scaleC),
                   "r"(mask[0]),
                   "r"(mask[1]),
                   "r"(mask[2]),
                   "r"(mask[3]));
  }
}

__device__ inline static void
    mma_m64n64k16_bf16bf16f32(bfloat16 *A_ptr, bfloat16 *B_ptr, uint32_t *D) {
  uint32_t idesc =
      mma_instruction_descriptor<float, bfloat16, 16, 16, false, true, false>();
  uint64_t a_desc = make_smem_desc(A_ptr, 1, 0, 0, 0, false);
  uint64_t b_desc = make_smem_desc(B_ptr, 1, 1, 0, 0, false);
  call_mma_m64n64k16_bf16bf16f32(D, a_desc, b_desc, idesc);
}
} // namespace blackwell
} // namespace kernel
