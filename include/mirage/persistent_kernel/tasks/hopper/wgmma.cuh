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

// A reg
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#wgmma-64n16-a
// reference
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
namespace kernel {

namespace wgmma {
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
        base_desc |= matrix_descriptor_encode((uint64_t)16) << 16;
        base_desc |= matrix_descriptor_encode((uint64_t)256) << 32;
        base_desc |= 0llu << 62; // set wgmma_swizzle mode
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
  static_assert(SMEM_A::ROW == M);
  static_assert(SMEM_A::COL == K);
  static_assert(SMEM_B::ROW == K);
  static_assert(SMEM_B::COL == N);
  if constexpr (M == 64 && N == 64 && K == 16 &&
                std::is_same<T, bfloat16>::value && tnspA == false &&
                tnspB == false) {
    for (int k = 0; k < (SMEM_A::COL / K); k++) {
      wgmma_m64n64k16_bf16bf16bf32<tnspA, tnspB>(a_desc.at(k * 16 * sizeof(T)),
                                                 b_desc.at(k * 16 * sizeof(T)),
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

} // namespace wgmma
} // namespace kernel
