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

// SM100 (Blackwell) PTX helpers: tcgen05 MMA, TMA mbarrier, TMEM, warp
// election. Shared by any SM100 kernel using tensor cores (MLA decode, MTP,
// future kernels).
#pragma once

#include <cuda.h>
#include <stdint.h>

namespace kernel {
namespace sm100_ptx {

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t p = 0;
  asm volatile("{\n\t.reg .pred %%px;\n\t"
               "elect.sync _|%%px, 0xFFFFFFFF;\n\t"
               "@%%px mov.s32 %0, 1;\n\t}"
               : "+r"(p));
  return p;
}

__device__ __forceinline__ void mbar_init(int addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr),
               "r"(count));
}

__device__ __forceinline__ void mbar_wait(int addr, int phase) {
  asm volatile("{\n\t.reg .pred P;\n\t"
               "WAIT: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, "
               "[%0], %1, 0x989680;\n\t"
               "@P bra DONE;\n\t"
               "bra WAIT;\n\t"
               "DONE:\n\t}" ::"r"(addr),
               "r"(phase));
}

__device__ __forceinline__ void mbar_arrive(int addr) {
  asm volatile(
      "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(addr)
      : "memory");
}

__device__ __forceinline__ void named_barrier_sync(int bar, int count) {
  asm volatile("bar.sync %0, %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void tma_load_2d_cta(
    int smem_addr, void const *desc, int c0, int c1, int mbar_addr) {
  uint64_t desc_i = reinterpret_cast<uint64_t>(desc);
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::"
      "bytes [%0], [%1, {%3, %4, %5, %6, %7}], [%2];" ::"r"(smem_addr),
      "l"(desc_i),
      "r"(mbar_addr),
      "r"(c0),
      "r"(c1),
      "r"(0),
      "r"(0),
      "r"(0)
      : "memory");
}

__device__ __forceinline__ void mbar_tx(int addr, int bytes) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::
          "r"(addr),
      "r"(bytes)
      : "memory");
}

__device__ __forceinline__ constexpr uint64_t desc_enc(uint64_t x) {
  return (x & 0x3FFFFULL) >> 4;
}

__device__ __forceinline__ uint64_t make_desc(int smem_addr) {
  constexpr uint64_t SBO = 8ULL * 128;
  return desc_enc(smem_addr) | (desc_enc(SBO) << 32) | (1ULL << 46) |
         (2ULL << 61);
}

__device__ __forceinline__ void tcgen05_mma(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, int acc) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::"r"(
          taddr),
      "l"(a_desc),
      "l"(b_desc),
      "r"(idesc),
      "r"(acc));
}

__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
               "cluster.b64 [%0];" ::"r"(mbar_addr)
               : "memory");
}

__device__ __forceinline__ void tcgen05_alloc(int addr_smem, int num_cols) {
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(
          addr_smem),
      "r"(num_cols));
}

__device__ __forceinline__ void tcgen05_dealloc(int taddr, int num_cols) {
  asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(taddr),
      "r"(num_cols));
}

__device__ __forceinline__ void tcgen05_relinquish_alloc_permit() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                   : "memory");
}

__device__ __forceinline__ void tcgen05_fence_before() {
  asm volatile("tcgen05.fence::before_thread_sync;");
}

__device__ __forceinline__ void tcgen05_fence_after() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

__device__ __forceinline__ void tcgen05_ld_x16(int taddr, float (&out)[16]) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=f"(out[0]),
                 "=f"(out[1]),
                 "=f"(out[2]),
                 "=f"(out[3]),
                 "=f"(out[4]),
                 "=f"(out[5]),
                 "=f"(out[6]),
                 "=f"(out[7]),
                 "=f"(out[8]),
                 "=f"(out[9]),
                 "=f"(out[10]),
                 "=f"(out[11]),
                 "=f"(out[12]),
                 "=f"(out[13]),
                 "=f"(out[14]),
                 "=f"(out[15])
               : "r"(taddr));
}

template <int N>
__device__ __forceinline__ void tcgen05_ld_cols(int taddr, float (&out)[N]) {
  static_assert(N % 16 == 0, "tcgen05_ld_cols supports multiples of 16");
#pragma unroll
  for (int c = 0; c < N / 16; c++) {
    float chunk[16];
    tcgen05_ld_x16(taddr + c * 16, chunk);
#pragma unroll
    for (int i = 0; i < 16; i++) {
      out[c * 16 + i] = chunk[i];
    }
  }
}

__device__ __forceinline__ void tcgen05_ld_wait() {
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}

__device__ __forceinline__ constexpr uint32_t make_idesc_f16(int mma_m,
                                                             int mma_n) {
  return (1U << 4) | (1U << 7) | (1U << 10) | ((uint32_t)(mma_n >> 3) << 17) |
         ((uint32_t)(mma_m >> 4) << 24);
}

__device__ __forceinline__ void supergroup_tile_coord(int linear_idx,
                                                      int num_m_tiles,
                                                      int num_n_tiles,
                                                      int group_m,
                                                      int &m_tile,
                                                      int &n_tile) {
  int tiles_per_group = group_m * num_n_tiles;
  int group_id = linear_idx / tiles_per_group;
  int idx_in_group = linear_idx - group_id * tiles_per_group;
  int first_m = group_id * group_m;
  int rows_in_group = num_m_tiles - first_m;
  if (rows_in_group > group_m) {
    rows_in_group = group_m;
  }
  m_tile = first_m + (idx_in_group % rows_in_group);
  n_tile = idx_in_group / rows_in_group;
}

} // namespace sm100_ptx
} // namespace kernel
