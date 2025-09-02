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
// A reg
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-a
// reference
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
namespace kernel {

namespace wgmma {

template <typename T, typename = void>
struct has_inner_col : ::std::false_type {};

template <typename T>
struct has_inner_col<T, ::std::void_t<decltype(T::INNER_COL)>>
    : ::std::true_type {};

// choose inner col if has, otherwise use col
template <typename SMEM>
constexpr size_t get_col_param() {
  if constexpr (has_inner_col<SMEM>::value) {
    return SMEM::INNER_COL;
  } else {
    return SMEM::COL;
  }
}

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
  return (((x)&0x3FFFF) >> 0x4);
}
// from  kitten include/ops/group/wgmma/base/base.cuh
template <typename SMEM, bool MNmajor = false>
struct mma_descriptor {
  uint64_t base_desc;

  __device__ mma_descriptor(SMEM smem) {
    base_desc = matrix_descriptor_encode((uint64_t)(smem.base_ptr));
    if constexpr (MNmajor == false) {
      // swizzle mode
      if constexpr (SMEM::b == 3) {
        base_desc |= matrix_descriptor_encode((uint64_t)16) << 16;
        base_desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
        base_desc |= 1llu << 62;
      } else if constexpr (SMEM::b == 2) {
        base_desc |= matrix_descriptor_encode((uint64_t)16) << 16;
        base_desc |= matrix_descriptor_encode((uint64_t)512) << 32;
        base_desc |= 2llu << 62; // set wgmma_swizzle mode
      } else if constexpr (SMEM::b == 1) {
        base_desc |= matrix_descriptor_encode((uint64_t)16) << 16;
        base_desc |= matrix_descriptor_encode((uint64_t)256) << 32;
        base_desc |= 3llu << 62; // set wgmma_swizzle mode
      } else {
        base_desc |= matrix_descriptor_encode((uint64_t)128) << 16;
        base_desc |= matrix_descriptor_encode((uint64_t)256) << 32;
        base_desc |= 0llu << 62; // set wgmma_swizzle mode
      }
    } else {
      if constexpr (SMEM::b == 3) {
        base_desc |= matrix_descriptor_encode((uint64_t)2048) << 16;
        base_desc |= matrix_descriptor_encode((uint64_t)1024) << 32;
        base_desc |= 1llu << 62; // set wgmma_swizzle mode
      } else if constexpr (SMEM::b == 2) {
        base_desc |= matrix_descriptor_encode((uint64_t)1024) << 16;
        base_desc |= matrix_descriptor_encode((uint64_t)512) << 32;
        base_desc |= 2llu << 62; // set wgmma_swizzle mode
      } else {
        base_desc |= matrix_descriptor_encode((uint64_t)512) << 16;
        base_desc |= matrix_descriptor_encode((uint64_t)256) << 32;
        base_desc |= 3llu << 62; // set wgmma_swizzle mode
      }
    }
  }

  __device__ inline uint64_t at(size_t offset) {
    return base_desc + matrix_descriptor_encode(offset);
  }
};

__device__ static inline void warpgroup_commit_batch() {
#ifdef MIRAGE_GRACE_HOPPER
  // cutlass::arch::synclog_emit_warpgroup_commit_batch(__LINE__);
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

template <int tnspA, int tnspB>
__device__ static inline void
    wgmma_m64n16k16_bf16bf16bf32(uint64_t const &desc_a,
                                 uint64_t const &desc_b,
                                 float &d00,
                                 float &d01,
                                 float &d02,
                                 float &d03,
                                 float &d04,
                                 float &d05,
                                 float &d06,
                                 float &d07) {
#ifdef MIRAGE_GRACE_HOPPER
  asm volatile("{\n"
               ".reg .pred p;\n"
               "setp.ne.b32 p, %10, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
               "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
               " %8,"
               " %9,"
               " p,   %11, %12, %13, %14;\n"
               "}\n"
               : "+f"(d00),
                 "+f"(d01),
                 "+f"(d02),
                 "+f"(d03),
                 "+f"(d04),
                 "+f"(d05),
                 "+f"(d06),
                 "+f"(d07)
               : "l"(desc_a),
                 "l"(desc_b),
                 "r"(int32_t(1)),
                 "n"(int32_t(1)),
                 "n"(int32_t(1)),
                 "n"(int32_t(tnspA)),
                 "n"(int32_t(tnspB)));
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

template <int tnspA, int tnspB>
__device__ static inline void
    wgmma_m64n32k16_bf16bf16bf32(uint64_t const &desc_a,
                                 uint64_t const &desc_b,
                                 float &d00,
                                 float &d01,
                                 float &d02,
                                 float &d03,
                                 float &d04,
                                 float &d05,
                                 float &d06,
                                 float &d07,
                                 float &d08,
                                 float &d09,
                                 float &d10,
                                 float &d11,
                                 float &d12,
                                 float &d13,
                                 float &d14,
                                 float &d15) {
#ifdef MIRAGE_GRACE_HOPPER
  asm volatile("{\n"
               ".reg .pred p;\n"
               "setp.ne.b32 p, %18, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "
               "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
               " %8,  %9,  %10, %11, %12, %13, %14, %15},"
               " %16,"
               " %17,"
               " p,   %19, %20, %21, %22;\n"
               "}\n"
               : "+f"(d00),
                 "+f"(d01),
                 "+f"(d02),
                 "+f"(d03),
                 "+f"(d04),
                 "+f"(d05),
                 "+f"(d06),
                 "+f"(d07),
                 "+f"(d08),
                 "+f"(d09),
                 "+f"(d10),
                 "+f"(d11),
                 "+f"(d12),
                 "+f"(d13),
                 "+f"(d14),
                 "+f"(d15)
               : "l"(desc_a),
                 "l"(desc_b),
                 "r"(int32_t(1)),
                 "n"(int32_t(1)),
                 "n"(int32_t(1)),
                 "n"(int32_t(tnspA)),
                 "n"(int32_t(tnspB)));
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

template <int tnspA, int tnspB>
__device__ static inline void
    wgmma_m64n64k16_bf16bf16bf32(uint64_t const &desc_a,
                                 uint64_t const &desc_b,
                                 float &d00,
                                 float &d01,
                                 float &d02,
                                 float &d03,
                                 float &d04,
                                 float &d05,
                                 float &d06,
                                 float &d07,
                                 float &d08,
                                 float &d09,
                                 float &d10,
                                 float &d11,
                                 float &d12,
                                 float &d13,
                                 float &d14,
                                 float &d15,
                                 float &d16,
                                 float &d17,
                                 float &d18,
                                 float &d19,
                                 float &d20,
                                 float &d21,
                                 float &d22,
                                 float &d23,
                                 float &d24,
                                 float &d25,
                                 float &d26,
                                 float &d27,
                                 float &d28,
                                 float &d29,
                                 float &d30,
                                 float &d31) {
#ifdef MIRAGE_GRACE_HOPPER
  asm volatile("{\n"
               ".reg .pred p;\n"
               "setp.ne.b32 p, %34, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
               "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
               " %8,  %9,  %10, %11, %12, %13, %14, %15, "
               " %16, %17, %18, %19, %20, %21, %22, %23, "
               " %24, %25, %26, %27, %28, %29, %30, %31},"
               " %32,"
               " %33,"
               " p,   %35, %36, %37, %38;\n"
               "}\n"
               : "+f"(d00),
                 "+f"(d01),
                 "+f"(d02),
                 "+f"(d03),
                 "+f"(d04),
                 "+f"(d05),
                 "+f"(d06),
                 "+f"(d07),
                 "+f"(d08),
                 "+f"(d09),
                 "+f"(d10),
                 "+f"(d11),
                 "+f"(d12),
                 "+f"(d13),
                 "+f"(d14),
                 "+f"(d15),
                 "+f"(d16),
                 "+f"(d17),
                 "+f"(d18),
                 "+f"(d19),
                 "+f"(d20),
                 "+f"(d21),
                 "+f"(d22),
                 "+f"(d23),
                 "+f"(d24),
                 "+f"(d25),
                 "+f"(d26),
                 "+f"(d27),
                 "+f"(d28),
                 "+f"(d29),
                 "+f"(d30),
                 "+f"(d31)
               : "l"(desc_a),
                 "l"(desc_b),
                 "r"(int32_t(1)),
                 "n"(int32_t(1)),
                 "n"(int32_t(1)),
                 "n"(int32_t(tnspA)),
                 "n"(int32_t(tnspB)));
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

template <typename T,
          int M,
          int N,
          int K,
          typename SMEM_A,
          typename SMEM_B,
          typename A_DESC,
          typename B_DESC,
          bool tnspA,
          bool tnspB>
__device__ static inline void mma(float *frag, A_DESC a_desc, B_DESC b_desc) {
  // static_assert(SMEM_A::ROW == 64);
  // static_assert(SMEM_B::COL == 64);
  if constexpr (M == 64 && K == 16 && std::is_same<T, bfloat16>::value) {
    for (int k = 0; k < (SMEM_A::COL / K); k++) {
      constexpr size_t a_col_param = get_col_param<SMEM_A>();
      constexpr size_t b_col_param = get_col_param<SMEM_B>();

      size_t a_offset = (k % 4) * 32 + (k / 4) * 2 * SMEM_A::ROW * a_col_param;
      size_t b_offset = (k % 4) * 32 + (k / 4) * 2 * SMEM_B::ROW * b_col_param;
      if constexpr (N == 16) {
        wgmma_m64n16k16_bf16bf16bf32<tnspA, tnspB>(a_desc.at(a_offset),
                                                   b_desc.at(b_offset),
                                                   frag[0],
                                                   frag[1],
                                                   frag[2],
                                                   frag[3],
                                                   frag[4],
                                                   frag[5],
                                                   frag[6],
                                                   frag[7]);
      } else if constexpr (N == 32) {
        wgmma_m64n32k16_bf16bf16bf32<tnspA, tnspB>(a_desc.at(a_offset),
                                                   b_desc.at(b_offset),
                                                   frag[0],
                                                   frag[1],
                                                   frag[2],
                                                   frag[3],
                                                   frag[4],
                                                   frag[5],
                                                   frag[6],
                                                   frag[7],
                                                   frag[8],
                                                   frag[9],
                                                   frag[10],
                                                   frag[11],
                                                   frag[12],
                                                   frag[13],
                                                   frag[14],
                                                   frag[15]);
      } else if constexpr (N == 64) {
        wgmma_m64n64k16_bf16bf16bf32<tnspA, tnspB>(a_desc.at(a_offset),
                                                   b_desc.at(b_offset),
                                                   frag[0],
                                                   frag[1],
                                                   frag[2],
                                                   frag[3],
                                                   frag[4],
                                                   frag[5],
                                                   frag[6],
                                                   frag[7],
                                                   frag[8],
                                                   frag[9],
                                                   frag[10],
                                                   frag[11],
                                                   frag[12],
                                                   frag[13],
                                                   frag[14],
                                                   frag[15],
                                                   frag[16],
                                                   frag[17],
                                                   frag[18],
                                                   frag[19],
                                                   frag[20],
                                                   frag[21],
                                                   frag[22],
                                                   frag[23],
                                                   frag[24],
                                                   frag[25],
                                                   frag[26],
                                                   frag[27],
                                                   frag[28],
                                                   frag[29],
                                                   frag[30],
                                                   frag[31]);
      } else {
        assert(false);
      }
    }
  } else {
    assert(false);
  }
}

template <int tnspB>
__device__ static inline void
    wgmma_m64n16k16_bf16bf16bf32_rs(uint32_t const &a0,
                                    uint32_t const &a1,
                                    uint32_t const &a2,
                                    uint32_t const &a3,
                                    uint64_t const &desc_b,
                                    float &d0,
                                    float &d1,
                                    float &d2,
                                    float &d3,
                                    float &d4,
                                    float &d5,
                                    float &d6,
                                    float &d7) {
#ifdef MIRAGE_GRACE_HOPPER
  asm volatile("{\n"
               ".reg .pred p;\n"
               "setp.ne.b32 p, %13, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "
               "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7},"
               "{%8,  %9,  %10, %11},"
               " %12,"
               " p,   %14, %15, %16;\n"
               "}\n"
               : "+f"(d0),
                 "+f"(d1),
                 "+f"(d2),
                 "+f"(d3),
                 "+f"(d4),
                 "+f"(d5),
                 "+f"(d6),
                 "+f"(d7)
               : "r"(a0),
                 "r"(a1),
                 "r"(a2),
                 "r"(a3),
                 "l"(desc_b),
                 "r"(int32_t(1)),
                 "n"(int32_t(1)),
                 "n"(int32_t(1)),
                 "n"(int32_t(tnspB)));
#else
  asm volatile("brkpt;\n" ::);
#endif
}

template <int tnspB>
__device__ static inline void
    wgmma_m64n64k16_bf16bf16bf32_rs(uint32_t const &a00,
                                    uint32_t const &a01,
                                    uint32_t const &a02,
                                    uint32_t const &a03,
                                    uint64_t const &desc_b,
                                    float &d00,
                                    float &d01,
                                    float &d02,
                                    float &d03,
                                    float &d04,
                                    float &d05,
                                    float &d06,
                                    float &d07,
                                    float &d08,
                                    float &d09,
                                    float &d10,
                                    float &d11,
                                    float &d12,
                                    float &d13,
                                    float &d14,
                                    float &d15,
                                    float &d16,
                                    float &d17,
                                    float &d18,
                                    float &d19,
                                    float &d20,
                                    float &d21,
                                    float &d22,
                                    float &d23,
                                    float &d24,
                                    float &d25,
                                    float &d26,
                                    float &d27,
                                    float &d28,
                                    float &d29,
                                    float &d30,
                                    float &d31) {
#ifdef MIRAGE_GRACE_HOPPER
  asm volatile("{\n"
               ".reg .pred p;\n"
               "setp.ne.b32 p, %37, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
               "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
               " %8,  %9,  %10, %11, %12, %13, %14, %15, "
               " %16, %17, %18, %19, %20, %21, %22, %23, "
               " %24, %25, %26, %27, %28, %29, %30, %31},"
               "{%32, %33, %34, %35},"
               " %36,"
               " p,   %38, %39, %40;\n"
               "}\n"
               : "+f"(d00),
                 "+f"(d01),
                 "+f"(d02),
                 "+f"(d03),
                 "+f"(d04),
                 "+f"(d05),
                 "+f"(d06),
                 "+f"(d07),
                 "+f"(d08),
                 "+f"(d09),
                 "+f"(d10),
                 "+f"(d11),
                 "+f"(d12),
                 "+f"(d13),
                 "+f"(d14),
                 "+f"(d15),
                 "+f"(d16),
                 "+f"(d17),
                 "+f"(d18),
                 "+f"(d19),
                 "+f"(d20),
                 "+f"(d21),
                 "+f"(d22),
                 "+f"(d23),
                 "+f"(d24),
                 "+f"(d25),
                 "+f"(d26),
                 "+f"(d27),
                 "+f"(d28),
                 "+f"(d29),
                 "+f"(d30),
                 "+f"(d31)
               : "r"(a00),
                 "r"(a01),
                 "r"(a02),
                 "r"(a03),
                 "l"(desc_b),
                 "r"(int32_t(1)),
                 "n"(int32_t(1)),
                 "n"(int32_t(1)),
                 "n"(int32_t(tnspB)));
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

template <int tnspB>
__device__ static inline void
    wgmma_m64n128k16_bf16bf16bf32_rs(uint32_t const &a00,
                                     uint32_t const &a01,
                                     uint32_t const &a02,
                                     uint32_t const &a03,
                                     uint64_t const &desc_b,
                                     float &d00,
                                     float &d01,
                                     float &d02,
                                     float &d03,
                                     float &d04,
                                     float &d05,
                                     float &d06,
                                     float &d07,
                                     float &d08,
                                     float &d09,
                                     float &d10,
                                     float &d11,
                                     float &d12,
                                     float &d13,
                                     float &d14,
                                     float &d15,
                                     float &d16,
                                     float &d17,
                                     float &d18,
                                     float &d19,
                                     float &d20,
                                     float &d21,
                                     float &d22,
                                     float &d23,
                                     float &d24,
                                     float &d25,
                                     float &d26,
                                     float &d27,
                                     float &d28,
                                     float &d29,
                                     float &d30,
                                     float &d31,
                                     float &d32,
                                     float &d33,
                                     float &d34,
                                     float &d35,
                                     float &d36,
                                     float &d37,
                                     float &d38,
                                     float &d39,
                                     float &d40,
                                     float &d41,
                                     float &d42,
                                     float &d43,
                                     float &d44,
                                     float &d45,
                                     float &d46,
                                     float &d47,
                                     float &d48,
                                     float &d49,
                                     float &d50,
                                     float &d51,
                                     float &d52,
                                     float &d53,
                                     float &d54,
                                     float &d55,
                                     float &d56,
                                     float &d57,
                                     float &d58,
                                     float &d59,
                                     float &d60,
                                     float &d61,
                                     float &d62,
                                     float &d63) {
#ifdef MIRAGE_GRACE_HOPPER
  asm volatile("{\n"
               ".reg .pred p;\n"
               "setp.ne.b32 p, %69, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
               "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
               " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
               " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
               " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
               " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
               " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
               " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
               " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
               "{%64,  %65,  %66,  %67},"
               " %68,"
               " p,    %70,  %71,  %72;\n"
               "}\n"
               : "+f"(d00),
                 "+f"(d01),
                 "+f"(d02),
                 "+f"(d03),
                 "+f"(d04),
                 "+f"(d05),
                 "+f"(d06),
                 "+f"(d07),
                 "+f"(d08),
                 "+f"(d09),
                 "+f"(d10),
                 "+f"(d11),
                 "+f"(d12),
                 "+f"(d13),
                 "+f"(d14),
                 "+f"(d15),
                 "+f"(d16),
                 "+f"(d17),
                 "+f"(d18),
                 "+f"(d19),
                 "+f"(d20),
                 "+f"(d21),
                 "+f"(d22),
                 "+f"(d23),
                 "+f"(d24),
                 "+f"(d25),
                 "+f"(d26),
                 "+f"(d27),
                 "+f"(d28),
                 "+f"(d29),
                 "+f"(d30),
                 "+f"(d31),
                 "+f"(d32),
                 "+f"(d33),
                 "+f"(d34),
                 "+f"(d35),
                 "+f"(d36),
                 "+f"(d37),
                 "+f"(d38),
                 "+f"(d39),
                 "+f"(d40),
                 "+f"(d41),
                 "+f"(d42),
                 "+f"(d43),
                 "+f"(d44),
                 "+f"(d45),
                 "+f"(d46),
                 "+f"(d47),
                 "+f"(d48),
                 "+f"(d49),
                 "+f"(d50),
                 "+f"(d51),
                 "+f"(d52),
                 "+f"(d53),
                 "+f"(d54),
                 "+f"(d55),
                 "+f"(d56),
                 "+f"(d57),
                 "+f"(d58),
                 "+f"(d59),
                 "+f"(d60),
                 "+f"(d61),
                 "+f"(d62),
                 "+f"(d63)
               : "r"(a00),
                 "r"(a01),
                 "r"(a02),
                 "r"(a03),
                 "l"(desc_b),
                 "r"(int32_t(1)),
                 "n"(int32_t(1)),
                 "n"(int32_t(1)),
                 "n"(int32_t(tnspB)));
#else
  asm volatile("brkpt;\n" ::);
#endif
}

template <typename T,
          int M,
          int N,
          int K,
          typename SMEM_B,
          typename B_DESC,
          bool tnspB>
__device__ static inline void
    mma_rs(float *frag, uint32_t *a_frag, B_DESC b_desc) {
  if constexpr (M == 64 && K == 16 && std::is_same<T, bfloat16>::value) {
    // tnspB=true, i.e. B is of shape KxN, otherwise B is of shape NxK
    constexpr int NUM_K_ITERS =
        (tnspB ? (SMEM_B::ROW + K - 1) / K : (SMEM_B::COL + K - 1) / K);
    for (int k_iter = 0; k_iter < NUM_K_ITERS; k_iter++) {
      uint32_t *a_frag_k = a_frag + k_iter * 4;

      size_t b_offset;
      if constexpr (tnspB) {
        b_offset = k_iter * 2048;
      } else {
        constexpr size_t b_col_param = get_col_param<SMEM_B>();
        b_offset =
            (k_iter % 4) * 32 + (k_iter / 4) * 2 * SMEM_B::ROW * b_col_param;
      }

      float *frag_k;
      if constexpr (N == 16) {
        frag_k = frag + k_iter * 8;
        wgmma_m64n16k16_bf16bf16bf32_rs<tnspB>(a_frag_k[0],
                                               a_frag_k[1],
                                               a_frag_k[2],
                                               a_frag_k[3],
                                               b_desc.at(b_offset),
                                               frag_k[0],
                                               frag_k[1],
                                               frag_k[2],
                                               frag_k[3],
                                               frag_k[4],
                                               frag_k[5],
                                               frag_k[6],
                                               frag_k[7]);
      } else if constexpr (N == 64) {
        frag_k = frag + k_iter * 32;
        wgmma_m64n64k16_bf16bf16bf32_rs<tnspB>(a_frag_k[0],
                                               a_frag_k[1],
                                               a_frag_k[2],
                                               a_frag_k[3],
                                               b_desc.at(b_offset),
                                               frag_k[0],
                                               frag_k[1],
                                               frag_k[2],
                                               frag_k[3],
                                               frag_k[4],
                                               frag_k[5],
                                               frag_k[6],
                                               frag_k[7],
                                               frag_k[8],
                                               frag_k[9],
                                               frag_k[10],
                                               frag_k[11],
                                               frag_k[12],
                                               frag_k[13],
                                               frag_k[14],
                                               frag_k[15],
                                               frag_k[16],
                                               frag_k[17],
                                               frag_k[18],
                                               frag_k[19],
                                               frag_k[20],
                                               frag_k[21],
                                               frag_k[22],
                                               frag_k[23],
                                               frag_k[24],
                                               frag_k[25],
                                               frag_k[26],
                                               frag_k[27],
                                               frag_k[28],
                                               frag_k[29],
                                               frag_k[30],
                                               frag_k[31]);
      } else if constexpr (N == 128) {
        frag_k = frag + k_iter * 64;
        wgmma_m64n128k16_bf16bf16bf32_rs<tnspB>(a_frag_k[0],
                                                a_frag_k[1],
                                                a_frag_k[2],
                                                a_frag_k[3],
                                                b_desc.at(b_offset),
                                                frag_k[0],
                                                frag_k[1],
                                                frag_k[2],
                                                frag_k[3],
                                                frag_k[4],
                                                frag_k[5],
                                                frag_k[6],
                                                frag_k[7],
                                                frag_k[8],
                                                frag_k[9],
                                                frag_k[10],
                                                frag_k[11],
                                                frag_k[12],
                                                frag_k[13],
                                                frag_k[14],
                                                frag_k[15],
                                                frag_k[16],
                                                frag_k[17],
                                                frag_k[18],
                                                frag_k[19],
                                                frag_k[20],
                                                frag_k[21],
                                                frag_k[22],
                                                frag_k[23],
                                                frag_k[24],
                                                frag_k[25],
                                                frag_k[26],
                                                frag_k[27],
                                                frag_k[28],
                                                frag_k[29],
                                                frag_k[30],
                                                frag_k[31],
                                                frag_k[32],
                                                frag_k[33],
                                                frag_k[34],
                                                frag_k[35],
                                                frag_k[36],
                                                frag_k[37],
                                                frag_k[38],
                                                frag_k[39],
                                                frag_k[40],
                                                frag_k[41],
                                                frag_k[42],
                                                frag_k[43],
                                                frag_k[44],
                                                frag_k[45],
                                                frag_k[46],
                                                frag_k[47],
                                                frag_k[48],
                                                frag_k[49],
                                                frag_k[50],
                                                frag_k[51],
                                                frag_k[52],
                                                frag_k[53],
                                                frag_k[54],
                                                frag_k[55],
                                                frag_k[56],
                                                frag_k[57],
                                                frag_k[58],
                                                frag_k[59],
                                                frag_k[60],
                                                frag_k[61],
                                                frag_k[62],
                                                frag_k[63]);
      } else {
        assert(false);
      }
    }
  } else {
    assert(false);
  }
}

__device__ static inline void warpgroup_arrive() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ static inline void mma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ static inline void mma_async_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" : : "n"(0) : "memory");
}

template <int N>
__device__ inline void warpgroup_fence_fragment(float (&frag)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    asm volatile("" : "+f"(frag[i])::"memory");
  }
}

} // namespace wgmma
} // namespace kernel
