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
namespace kernel {
namespace blackwell {

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
#define CP_ASYNC_SM100_ENABLED
#endif

__device__ inline static void load_wait() {
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}
__device__ inline static void store_wait() {
  asm volatile("tcgen05.wait::st.sync.aligned;");
}

__device__ inline static void load_tmem_to_reg(uint32_t const &tmem_ptr,
                                               uint32_t *R) {
  //  uint32_t s_tmem_ptr =
  //  static_cast<uint32_t>(__cvta_generic_to_shared(tmem_ptr));
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32"
               "{%0, %1, %2, %3,"
               "%4, %5, %6, %7},"
               "[%8];\n"
               : "=r"(R[0]),
                 "=r"(R[1]),
                 "=r"(R[2]),
                 "=r"(R[3]),
                 "=r"(R[4]),
                 "=r"(R[5]),
                 "=r"(R[6]),
                 "=r"(R[7])
               : "r"(tmem_ptr));
}

__device__ inline static void cp_smem_to_tmem(uint32_t const &tmem_ptr,
                                              uint32_t &smem_desc) {
  asm volatile("tcgen05.cp.cta_group::1.4x256b [%0], %1;"
               :
               : "r"(tmem_ptr), "l"(uint64_t(smem_desc)));
}

} // namespace blackwell
} // namespace kernel