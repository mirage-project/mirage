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
    wgmma_m64n8k16_bf16bf16bf32(uint64_t const &desc_a,
                                uint64_t const &desc_b,
                                float &d00,
                                float &d01,
                                float &d02,
                                float &d03) {
#ifdef MIRAGE_GRACE_HOPPER
  asm volatile("{\n"
               ".reg .pred p;\n"
               "setp.ne.b32 p, %6, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
               "{%0,  %1,  %2,  %3},"
               " %4,"
               " %5,"
               " p,   %7,  %8,  %9,  %10;\n"
               "}\n"
               : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03)
               : "l"(desc_a),
                 "l"(desc_b),
                 "r"(int32_t(1)),
                 "n"(int32_t(1)),
                 "n"(int32_t(1)),
                 "n"(int32_t(tnspA)),
                 "n"(int32_t(tnspB)));
#else
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

template <int tnspA, int tnspB>
__device__ static inline void
    wgmma_m64n128k16_bf16bf16bf32(uint64_t const &desc_a,
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
               "setp.ne.b32 p, %66, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "
               "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
               " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
               " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
               " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
               " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
               " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
               " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
               " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63},"
               " %64,"
               " %65,"
               " p,    %67,  %68,  %69,  %70;\n"
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
    wgmma_m64n256k16_bf16bf16bf32(uint64_t const &desc_a,
                                  uint64_t const &desc_b,
                                  float &d000,
                                  float &d001,
                                  float &d002,
                                  float &d003,
                                  float &d004,
                                  float &d005,
                                  float &d006,
                                  float &d007,
                                  float &d008,
                                  float &d009,
                                  float &d010,
                                  float &d011,
                                  float &d012,
                                  float &d013,
                                  float &d014,
                                  float &d015,
                                  float &d016,
                                  float &d017,
                                  float &d018,
                                  float &d019,
                                  float &d020,
                                  float &d021,
                                  float &d022,
                                  float &d023,
                                  float &d024,
                                  float &d025,
                                  float &d026,
                                  float &d027,
                                  float &d028,
                                  float &d029,
                                  float &d030,
                                  float &d031,
                                  float &d032,
                                  float &d033,
                                  float &d034,
                                  float &d035,
                                  float &d036,
                                  float &d037,
                                  float &d038,
                                  float &d039,
                                  float &d040,
                                  float &d041,
                                  float &d042,
                                  float &d043,
                                  float &d044,
                                  float &d045,
                                  float &d046,
                                  float &d047,
                                  float &d048,
                                  float &d049,
                                  float &d050,
                                  float &d051,
                                  float &d052,
                                  float &d053,
                                  float &d054,
                                  float &d055,
                                  float &d056,
                                  float &d057,
                                  float &d058,
                                  float &d059,
                                  float &d060,
                                  float &d061,
                                  float &d062,
                                  float &d063,
                                  float &d064,
                                  float &d065,
                                  float &d066,
                                  float &d067,
                                  float &d068,
                                  float &d069,
                                  float &d070,
                                  float &d071,
                                  float &d072,
                                  float &d073,
                                  float &d074,
                                  float &d075,
                                  float &d076,
                                  float &d077,
                                  float &d078,
                                  float &d079,
                                  float &d080,
                                  float &d081,
                                  float &d082,
                                  float &d083,
                                  float &d084,
                                  float &d085,
                                  float &d086,
                                  float &d087,
                                  float &d088,
                                  float &d089,
                                  float &d090,
                                  float &d091,
                                  float &d092,
                                  float &d093,
                                  float &d094,
                                  float &d095,
                                  float &d096,
                                  float &d097,
                                  float &d098,
                                  float &d099,
                                  float &d100,
                                  float &d101,
                                  float &d102,
                                  float &d103,
                                  float &d104,
                                  float &d105,
                                  float &d106,
                                  float &d107,
                                  float &d108,
                                  float &d109,
                                  float &d110,
                                  float &d111,
                                  float &d112,
                                  float &d113,
                                  float &d114,
                                  float &d115,
                                  float &d116,
                                  float &d117,
                                  float &d118,
                                  float &d119,
                                  float &d120,
                                  float &d121,
                                  float &d122,
                                  float &d123,
                                  float &d124,
                                  float &d125,
                                  float &d126,
                                  float &d127) {
#ifdef MIRAGE_GRACE_HOPPER
  asm volatile("{\n"
               ".reg .pred p;\n"
               "setp.ne.b32 p, %130, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16 "
               "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
               " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
               " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
               " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
               " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
               " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
               " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
               " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
               " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
               " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
               " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
               " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
               " %96,  %97,  %98,  %99,  %100, %101, %102, %103, "
               " %104, %105, %106, %107, %108, %109, %110, %111, "
               " %112, %113, %114, %115, %116, %117, %118, %119, "
               " %120, %121, %122, %123, %124, %125, %126, %127},"
               " %128,"
               " %129,"
               " p,    %131, %132, %133, %134;\n"
               "}\n"
               : "+f"(d000),
                 "+f"(d001),
                 "+f"(d002),
                 "+f"(d003),
                 "+f"(d004),
                 "+f"(d005),
                 "+f"(d006),
                 "+f"(d007),
                 "+f"(d008),
                 "+f"(d009),
                 "+f"(d010),
                 "+f"(d011),
                 "+f"(d012),
                 "+f"(d013),
                 "+f"(d014),
                 "+f"(d015),
                 "+f"(d016),
                 "+f"(d017),
                 "+f"(d018),
                 "+f"(d019),
                 "+f"(d020),
                 "+f"(d021),
                 "+f"(d022),
                 "+f"(d023),
                 "+f"(d024),
                 "+f"(d025),
                 "+f"(d026),
                 "+f"(d027),
                 "+f"(d028),
                 "+f"(d029),
                 "+f"(d030),
                 "+f"(d031),
                 "+f"(d032),
                 "+f"(d033),
                 "+f"(d034),
                 "+f"(d035),
                 "+f"(d036),
                 "+f"(d037),
                 "+f"(d038),
                 "+f"(d039),
                 "+f"(d040),
                 "+f"(d041),
                 "+f"(d042),
                 "+f"(d043),
                 "+f"(d044),
                 "+f"(d045),
                 "+f"(d046),
                 "+f"(d047),
                 "+f"(d048),
                 "+f"(d049),
                 "+f"(d050),
                 "+f"(d051),
                 "+f"(d052),
                 "+f"(d053),
                 "+f"(d054),
                 "+f"(d055),
                 "+f"(d056),
                 "+f"(d057),
                 "+f"(d058),
                 "+f"(d059),
                 "+f"(d060),
                 "+f"(d061),
                 "+f"(d062),
                 "+f"(d063),
                 "+f"(d064),
                 "+f"(d065),
                 "+f"(d066),
                 "+f"(d067),
                 "+f"(d068),
                 "+f"(d069),
                 "+f"(d070),
                 "+f"(d071),
                 "+f"(d072),
                 "+f"(d073),
                 "+f"(d074),
                 "+f"(d075),
                 "+f"(d076),
                 "+f"(d077),
                 "+f"(d078),
                 "+f"(d079),
                 "+f"(d080),
                 "+f"(d081),
                 "+f"(d082),
                 "+f"(d083),
                 "+f"(d084),
                 "+f"(d085),
                 "+f"(d086),
                 "+f"(d087),
                 "+f"(d088),
                 "+f"(d089),
                 "+f"(d090),
                 "+f"(d091),
                 "+f"(d092),
                 "+f"(d093),
                 "+f"(d094),
                 "+f"(d095),
                 "+f"(d096),
                 "+f"(d097),
                 "+f"(d098),
                 "+f"(d099),
                 "+f"(d100),
                 "+f"(d101),
                 "+f"(d102),
                 "+f"(d103),
                 "+f"(d104),
                 "+f"(d105),
                 "+f"(d106),
                 "+f"(d107),
                 "+f"(d108),
                 "+f"(d109),
                 "+f"(d110),
                 "+f"(d111),
                 "+f"(d112),
                 "+f"(d113),
                 "+f"(d114),
                 "+f"(d115),
                 "+f"(d116),
                 "+f"(d117),
                 "+f"(d118),
                 "+f"(d119),
                 "+f"(d120),
                 "+f"(d121),
                 "+f"(d122),
                 "+f"(d123),
                 "+f"(d124),
                 "+f"(d125),
                 "+f"(d126),
                 "+f"(d127)
               : "l"(desc_a),
                 "l"(desc_b),
                 "r"(int32_t(1)),
                 "n"(int32_t(1)),
                 "n"(int32_t(1)),
                 "n"(int32_t(tnspA)),
                 "n"(int32_t(tnspB)));
#else
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
#pragma unroll
    for (int k = 0; k < (SMEM_A::COL / K); k++) {
      constexpr size_t a_col_param = get_col_param<SMEM_A>();
      constexpr size_t b_col_param = get_col_param<SMEM_B>();

      size_t a_offset = (k % 4) * 32 + (k / 4) * 2 * SMEM_A::ROW * a_col_param;
      size_t b_offset = (k % 4) * 32 + (k / 4) * 2 * SMEM_B::ROW * b_col_param;
      if constexpr (N == 8) {
        wgmma_m64n8k16_bf16bf16bf32<tnspA, tnspB>(a_desc.at(a_offset),
                                                  b_desc.at(b_offset),
                                                  frag[0],
                                                  frag[1],
                                                  frag[2],
                                                  frag[3]);
      } else if constexpr (N == 16) {
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
      } else if constexpr (N == 128) {
        wgmma_m64n128k16_bf16bf16bf32<tnspA, tnspB>(a_desc.at(a_offset),
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
                                                    frag[31],
                                                    frag[32],
                                                    frag[33],
                                                    frag[34],
                                                    frag[35],
                                                    frag[36],
                                                    frag[37],
                                                    frag[38],
                                                    frag[39],
                                                    frag[40],
                                                    frag[41],
                                                    frag[42],
                                                    frag[43],
                                                    frag[44],
                                                    frag[45],
                                                    frag[46],
                                                    frag[47],
                                                    frag[48],
                                                    frag[49],
                                                    frag[50],
                                                    frag[51],
                                                    frag[52],
                                                    frag[53],
                                                    frag[54],
                                                    frag[55],
                                                    frag[56],
                                                    frag[57],
                                                    frag[58],
                                                    frag[59],
                                                    frag[60],
                                                    frag[61],
                                                    frag[62],
                                                    frag[63]);
      } else if constexpr (N == 256) {
        wgmma_m64n256k16_bf16bf16bf32<tnspA, tnspB>(a_desc.at(a_offset),
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
                                                    frag[31],
                                                    frag[32],
                                                    frag[33],
                                                    frag[34],
                                                    frag[35],
                                                    frag[36],
                                                    frag[37],
                                                    frag[38],
                                                    frag[39],
                                                    frag[40],
                                                    frag[41],
                                                    frag[42],
                                                    frag[43],
                                                    frag[44],
                                                    frag[45],
                                                    frag[46],
                                                    frag[47],
                                                    frag[48],
                                                    frag[49],
                                                    frag[50],
                                                    frag[51],
                                                    frag[52],
                                                    frag[53],
                                                    frag[54],
                                                    frag[55],
                                                    frag[56],
                                                    frag[57],
                                                    frag[58],
                                                    frag[59],
                                                    frag[60],
                                                    frag[61],
                                                    frag[62],
                                                    frag[63],
                                                    frag[64],
                                                    frag[65],
                                                    frag[66],
                                                    frag[67],
                                                    frag[68],
                                                    frag[69],
                                                    frag[70],
                                                    frag[71],
                                                    frag[72],
                                                    frag[73],
                                                    frag[74],
                                                    frag[75],
                                                    frag[76],
                                                    frag[77],
                                                    frag[78],
                                                    frag[79],
                                                    frag[80],
                                                    frag[81],
                                                    frag[82],
                                                    frag[83],
                                                    frag[84],
                                                    frag[85],
                                                    frag[86],
                                                    frag[87],
                                                    frag[88],
                                                    frag[89],
                                                    frag[90],
                                                    frag[91],
                                                    frag[92],
                                                    frag[93],
                                                    frag[94],
                                                    frag[95],
                                                    frag[96],
                                                    frag[97],
                                                    frag[98],
                                                    frag[99],
                                                    frag[100],
                                                    frag[101],
                                                    frag[102],
                                                    frag[103],
                                                    frag[104],
                                                    frag[105],
                                                    frag[106],
                                                    frag[107],
                                                    frag[108],
                                                    frag[109],
                                                    frag[110],
                                                    frag[111],
                                                    frag[112],
                                                    frag[113],
                                                    frag[114],
                                                    frag[115],
                                                    frag[116],
                                                    frag[117],
                                                    frag[118],
                                                    frag[119],
                                                    frag[120],
                                                    frag[121],
                                                    frag[122],
                                                    frag[123],
                                                    frag[124],
                                                    frag[125],
                                                    frag[126],
                                                    frag[127]);
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
    // If tnspB=true, then B is assumed to be of shape KxN, otherwise B is of
    // shape NxK so the computation of NUM_K_ITERS is different for tnspB=true
    // and tnspB=false
    constexpr int NUM_K_ITERS =
        (tnspB ? (SMEM_B::ROW + K - 1) / K : (SMEM_B::COL + K - 1) / K);

#pragma unroll
    for (int k_iter = 0; k_iter < NUM_K_ITERS; k_iter++) {
      uint32_t *a_frag_k = a_frag + k_iter * 4;

      size_t b_offset;
      if constexpr (tnspB) {
        b_offset = k_iter * 2048;
      } else {
        constexpr size_t b_col_param = get_col_param<SMEM_B>();
        // (k_iter % 4) * 32 is the offset within each 64 cols
        // (k_iter / 4) * 2 * SMEM_B::ROW * b_col_param; is the offset from each
        // 64 cols to the next 64 cols
        b_offset =
            (k_iter % 4) * 32 + (k_iter / 4) * 2 * SMEM_B::ROW * b_col_param;
      }

      if constexpr (N == 16) {
        wgmma_m64n16k16_bf16bf16bf32_rs<tnspB>(a_frag_k[0],
                                               a_frag_k[1],
                                               a_frag_k[2],
                                               a_frag_k[3],
                                               b_desc.at(b_offset),
                                               frag[0],
                                               frag[1],
                                               frag[2],
                                               frag[3],
                                               frag[4],
                                               frag[5],
                                               frag[6],
                                               frag[7]);
      } else if constexpr (N == 64) {
        wgmma_m64n64k16_bf16bf16bf32_rs<tnspB>(a_frag_k[0],
                                               a_frag_k[1],
                                               a_frag_k[2],
                                               a_frag_k[3],
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
      } else if constexpr (N == 128) {
        wgmma_m64n128k16_bf16bf16bf32_rs<tnspB>(a_frag_k[0],
                                                a_frag_k[1],
                                                a_frag_k[2],
                                                a_frag_k[3],
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
                                                frag[31],
                                                frag[32],
                                                frag[33],
                                                frag[34],
                                                frag[35],
                                                frag[36],
                                                frag[37],
                                                frag[38],
                                                frag[39],
                                                frag[40],
                                                frag[41],
                                                frag[42],
                                                frag[43],
                                                frag[44],
                                                frag[45],
                                                frag[46],
                                                frag[47],
                                                frag[48],
                                                frag[49],
                                                frag[50],
                                                frag[51],
                                                frag[52],
                                                frag[53],
                                                frag[54],
                                                frag[55],
                                                frag[56],
                                                frag[57],
                                                frag[58],
                                                frag[59],
                                                frag[60],
                                                frag[61],
                                                frag[62],
                                                frag[63]);
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
