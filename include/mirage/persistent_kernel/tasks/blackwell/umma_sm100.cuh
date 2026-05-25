/* Copyright 2026 CMU
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

// Pure-PTX building blocks for SM100 (Blackwell) UMMA / tcgen05 used by the
// hand-written linear kernels in this directory. The intent is to mirror the
// role hopper/wgmma.cuh plays for SM90: provide just enough wrappers that the
// kernels themselves contain no CuTe / Cutlass MMA dependency.
//
// Bit layouts and PTX strings here are cross-checked against
// deps/cutlass/include/cute/arch/mma_sm100_desc.hpp, mma_sm100_umma.hpp,
// copy_sm100.hpp and tmem_allocator_sm100.hpp.

#pragma once

#include <cstdint>

#include "../hopper/barrier.cuh"

namespace kernel {
namespace umma {

// ---------------------------------------------------------------------------
// SMEM matrix descriptor for tcgen05.mma SS variants.
//
// Layout (cute::UMMA::SmemDescriptor):
//   bits [ 0,14)  start_address >> 4
//   bits [16,30)  leading_byte_offset >> 4
//   bits [32,46)  stride_byte_offset >> 4
//   bit  [46,48)  version (=1 for SM100)
//   bit  [48,49)  unused
//   bits [49,52)  base_offset (3 bits)
//   bit  [52,53)  lbo_mode
//   bits [53,61)  unused
//   bits [61,64)  layout_type (SWIZZLE_NONE=0, SWIZZLE_128B=2,
//                              SWIZZLE_64B=4, SWIZZLE_32B=6, SW128_BASE32B=1)
// ---------------------------------------------------------------------------

enum class Swizzle : uint8_t {
  None       = 0,
  SW128_B32B = 1,
  SW128      = 2,
  SW64       = 4,
  SW32       = 6,
};

__device__ static inline uint64_t encode14(uint64_t x) {
  return (x & 0x3FFFFull) >> 4;
}

// LBO/SBO are byte counts; PTX expects them already shifted right by 4.
__device__ static inline uint64_t make_smem_desc(void const *smem_ptr,
                                                 uint32_t leading_byte_offset,
                                                 uint32_t stride_byte_offset,
                                                 Swizzle swizzle = Swizzle::SW128,
                                                 uint8_t base_offset = 0,
                                                 uint8_t lbo_mode = 0) {
  uint32_t start = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  uint64_t desc = 0;
  desc |= encode14(static_cast<uint64_t>(start));
  desc |= encode14(static_cast<uint64_t>(leading_byte_offset)) << 16;
  desc |= encode14(static_cast<uint64_t>(stride_byte_offset)) << 32;
  desc |= (uint64_t{1} & 0x3ull) << 46;                 // version = 1
  desc |= (uint64_t{base_offset} & 0x7ull) << 49;
  desc |= (uint64_t{lbo_mode} & 0x1ull) << 52;
  desc |= (uint64_t{static_cast<uint8_t>(swizzle)} & 0x7ull) << 61;
  return desc;
}

// Update the start_address field of an existing descriptor in place.
__device__ static inline uint64_t set_smem_desc_addr(uint64_t desc,
                                                     void const *smem_ptr) {
  uint32_t start = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  desc &= ~uint64_t{0x3FFFull};
  desc |= encode14(static_cast<uint64_t>(start));
  return desc;
}

// ---------------------------------------------------------------------------
// Instruction descriptor for block-scaled mxf4nvf4 MMA.
//
// Layout (cute::UMMA::InstrDescriptorBlockScaled):
//   bits [ 0, 2)  sparse_id2 (=0 for dense)
//   bit  [ 2, 3)  sparse_flag (=0)
//   bit  [ 3, 4)  unused
//   bits [ 4, 6)  b_sf_id   (top 2 bits of TMEM SFB column)
//   bit  [ 6, 7)  unused
//   bits [ 7,10)  a_format  (E2M1 = 5)
//   bits [10,13)  b_format  (E2M1 = 5)
//   bit  [13,14)  a_negate  (=0)
//   bit  [14,15)  b_negate  (=0)
//   bit  [15,16)  a_major   (K=0, MN=1)
//   bit  [16,17)  b_major   (K=0, MN=1)
//   bits [17,23)  n_dim     (N >> 3)
//   bit  [23,24)  scale_format (E4M3 = 0, E8M0 = 1)
//   bits [24,29)  m_dim     (M >> 4)
//   bits [29,31)  a_sf_id   (top 2 bits of TMEM SFA column)
//   bit  [31,32)  k_size    (MXF4: 0 = K64, 1 = K96; for nvfp4-vec16 use 0)
//
// CuTe wraps the 32-bit instruction descriptor in a uint64_t with the descriptor
// in the upper 32 bits (lower 32 bits are the sparse-metadata TMEM addr, =0 here).
// We mirror that convention so the PTX issuer can use the same 64-bit operand.
// ---------------------------------------------------------------------------

enum class CFormat : uint8_t { F16 = 0, F32 = 1, S32 = 2 };
enum class ScaleFormat : uint8_t { UE4M3 = 0, UE8M0 = 1 };
enum class Major : uint8_t { K = 0, MN = 1 };

__device__ static inline uint64_t make_inst_desc_mxf4(int M,
                                                      int N,
                                                      ScaleFormat sf,
                                                      Major a_major = Major::K,
                                                      Major b_major = Major::K,
                                                      uint32_t tmem_sfa_addr = 0,
                                                      uint32_t tmem_sfb_addr = 0) {
  // a/b format = E2M1 (5)
  constexpr uint32_t a_fmt = 5;
  constexpr uint32_t b_fmt = 5;
  uint32_t desc = 0;
  desc |= (a_fmt   & 0x7u)    << 7;
  desc |= (b_fmt   & 0x7u)    << 10;
  desc |= ((static_cast<uint32_t>(a_major) & 0x1u)) << 15;
  desc |= ((static_cast<uint32_t>(b_major) & 0x1u)) << 16;
  desc |= ((static_cast<uint32_t>(N >> 3)  & 0x3Fu)) << 17;
  desc |= ((static_cast<uint32_t>(sf)      & 0x1u)) << 23;
  desc |= ((static_cast<uint32_t>(M >> 4)  & 0x1Fu)) << 24;
  // a_sf_id / b_sf_id: top 2 bits of the TMEM column address.
  desc |= ((tmem_sfa_addr & 0xC0000000u) >> 30) << 29;
  desc |= ((tmem_sfb_addr & 0xC0000000u) >> 30) << 4;
  // k_size = 0 (K=64 for MXF4 dense)
  return (static_cast<uint64_t>(desc) << 32);
}

// ---------------------------------------------------------------------------
// tcgen05.mma issuer for kind::mxf4nvf4 block_scale, single CTA, scale_vec=1X.
//
// SCALE_VEC_SIZE = 16 -> scale_vec::4X (PTX <12.9) / block16 (PTX >=12.9)
// SCALE_VEC_SIZE = 32 -> scale_vec::2X (PTX <12.9) / block32 (PTX >=12.9)
// We use the kind::mxf4nvf4.* form for both VS=16 (NVFP4 with E4M3 scales) and
// VS=32 (MXFP4 with E8M0 scales) because mxf4nvf4 is the strict superset.
//
// Operands:
//   tmem_d   : u32  TMEM column index of accumulator
//   desc_a   : u64  SMEM descriptor for A
//   desc_b   : u64  SMEM descriptor for B
//   inst_desc: u64  instruction descriptor (upper 32 bits set)
//   scale_d  : u32  0 -> overwrite, !=0 -> accumulate
//   sfa_addr : u32  TMEM column index of SFA fragment
//   sfb_addr : u32  TMEM column index of SFB fragment
// ---------------------------------------------------------------------------

template <int SCALE_VEC_SIZE>
__device__ static inline void mma_mxf4_ss(uint32_t tmem_d,
                                          uint64_t desc_a,
                                          uint64_t desc_b,
                                          uint64_t inst_desc,
                                          uint32_t scale_d,
                                          uint32_t sfa_addr,
                                          uint32_t sfb_addr) {
  static_assert(SCALE_VEC_SIZE == 16 || SCALE_VEC_SIZE == 32,
                "SCALE_VEC_SIZE must be 16 (NVFP4) or 32 (MXFP4)");
  uint32_t idesc_hi = static_cast<uint32_t>(inst_desc >> 32);
  if constexpr (SCALE_VEC_SIZE == 16) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
        "}\n"
        :
        : "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc_hi), "r"(scale_d),
          "r"(sfa_addr), "r"(sfb_addr));
  } else {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
#if (__CUDACC_VER_MAJOR__ > 12) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#else
        "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X [%0], %1, %2, %3, [%5], [%6], p; \n\t"
#endif
        "}\n"
        :
        : "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(idesc_hi), "r"(scale_d),
          "r"(sfa_addr), "r"(sfb_addr));
  }
}

// ---------------------------------------------------------------------------
// tcgen05.cp 32x128b warpx4 — broadcast-style SMEM->TMEM copy used to stage
// scale-factor fragments before MMA. Issued by a single warp (4-warp internal
// fan-out), source described by a SMEM descriptor with the same encoding as
// the MMA SS variant.
// ---------------------------------------------------------------------------

__device__ static inline void utccp_4x32x128b(uint32_t tmem_dst,
                                              uint64_t smem_src_desc) {
  asm volatile(
      "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
      :
      : "r"(tmem_dst), "l"(smem_src_desc));
}

// ---------------------------------------------------------------------------
// tcgen05.ld.sync.aligned.32x32b.x2.b32 — TMEM->register load. Each lane in
// the issuing 128-thread group reads two 32-bit values out of TMEM.
// dst[0] receives column tmem_addr, dst[1] receives the next column.
// ---------------------------------------------------------------------------

__device__ static inline void tmem_ld_32dp32b2x(uint32_t tmem_addr,
                                                uint32_t &dst0,
                                                uint32_t &dst1) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];\n"
      : "=r"(dst0), "=r"(dst1)
      : "r"(tmem_addr));
}

// Wait for outstanding tcgen05.ld to complete (reg is now live).
__device__ static inline void tmem_ld_wait() {
  asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::);
}

// Wait for outstanding tcgen05.st to drain.
__device__ static inline void tmem_st_wait() {
  asm volatile("tcgen05.wait::st.sync.aligned;\n" ::);
}

// ---------------------------------------------------------------------------
// TMEM allocator. Mirrors cute::TMEM::Allocator1Sm.
// Must be issued by a single fully-active warp.
// ---------------------------------------------------------------------------

static constexpr int kTmemColumnsPerSlice = 32;
static constexpr int kTmemTotalColumns    = 512;

__device__ static inline void tmem_alloc(int num_columns,
                                         uint32_t *dst_smem_u32) {
  uint32_t dst_intptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem_u32));
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
      :
      : "r"(dst_intptr), "r"(num_columns));
}

__device__ static inline void tmem_dealloc(uint32_t tmem_ptr, int num_columns) {
  asm volatile(
      "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
      :
      : "r"(tmem_ptr), "r"(num_columns));
}

__device__ static inline void tmem_relinquish_alloc_permit() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::);
}

// ---------------------------------------------------------------------------
// tcgen05.commit — signals an mbarrier that all preceding tcgen05.mma issued
// by this CTA in cta_group::1 have completed. Replaces
// cutlass::arch::umma_arrive.
// ---------------------------------------------------------------------------

__device__ static inline void umma_arrive(uint64_t *smem_mbar_ptr) {
  uint32_t bar_intptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_mbar_ptr));
  // Caller is expected to be inside an elect_one_sync() guard, matching the
  // CuTe convention.
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
      :
      : "r"(bar_intptr));
}

// ---------------------------------------------------------------------------
// elect.sync — pick a single active thread of a warp to issue side-effecting
// PTX. Returns 1 on the elected lane, 0 otherwise.
// ---------------------------------------------------------------------------

__device__ static inline bool elect_one_sync() {
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
      "{\n\t"
      ".reg .b32 rx;\n\t"
      ".reg .pred px;\n\t"
      "    elect.sync rx|px, %2;\n\t"
      "@px mov.s32 %1, 1;\n\t"
      "    mov.s32 %0, rx;\n\t"
      "}"
      : "+r"(laneid), "+r"(pred)
      : "r"(0xFFFFFFFFu));
  return pred != 0;
}

// ---------------------------------------------------------------------------
// Cluster-sync fences. In single-CTA kernels we only need the basic forms.
// ---------------------------------------------------------------------------

__device__ static inline void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" ::);
}

// Bulk-async (TMA) write fences/arrives needed by the epilogue.
__device__ static inline void tma_store_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::);
}

__device__ static inline void tma_store_arrive() {
  asm volatile("cp.async.bulk.commit_group;" ::);
}

template <int N>
__device__ static inline void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N));
}

// ---------------------------------------------------------------------------
// mbarrier helpers used by the kernel scheduler. Some of these duplicate
// logic in hopper/barrier.cuh but accept a raw u64* (the form most natural
// for the new raw shared-storage struct).
// ---------------------------------------------------------------------------

__device__ static inline void init_barrier(uint64_t *bar, int arrive_count) {
  uint32_t intptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
               :
               : "r"(intptr), "r"(arrive_count));
}

template <int N>
__device__ static inline void init_barrier_array(uint64_t *bar_arr,
                                                 int arrive_count) {
#pragma unroll
  for (int i = 0; i < N; ++i) {
    init_barrier(&bar_arr[i], arrive_count);
  }
}

__device__ static inline void wait_barrier(uint64_t &bar, uint32_t phase) {
  uint32_t intptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
  asm volatile("{\n"
               ".reg .pred P1;\n"
               "LAB_WAIT_UMMA:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@P1 bra.uni DONE_UMMA;\n"
               "bra.uni LAB_WAIT_UMMA;\n"
               "DONE_UMMA:\n"
               "}\n" ::"r"(intptr),
               "r"(phase));
}

__device__ static inline void arrive_barrier(uint64_t &bar,
                                             uint32_t count = 1) {
  uint32_t intptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(intptr), "r"(count)
               : "memory");
}

__device__ static inline void
set_barrier_transaction_bytes(uint64_t &bar, uint32_t bytes) {
  uint32_t intptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
      :
      : "r"(intptr), "r"(bytes));
}

// ---------------------------------------------------------------------------
// Named barrier — simple wrapper over PTX bar.sync. Lets us reproduce
// cutlass::arch::NamedBarrier behaviour for the small set of named IDs the
// kernel uses (TmemAllocBarrier, EpilogueBarrier).
// ---------------------------------------------------------------------------

class NamedBarrier {
public:
  __device__ NamedBarrier(int num_threads, int barrier_id)
      : num_threads_(num_threads), barrier_id_(barrier_id) {}

  __device__ inline void arrive_and_wait() const {
    asm volatile("bar.sync %0, %1;" ::"r"(barrier_id_), "r"(num_threads_));
  }

  __device__ inline void arrive() const {
    asm volatile("bar.arrive %0, %1;" ::"r"(barrier_id_), "r"(num_threads_));
  }

private:
  int num_threads_;
  int barrier_id_;
};

// Reserved named-barrier IDs we use, picked to avoid clashing with the
// hard-wired Cutlass reserved range (0..7 are reserved-ish; we use 8 and 9).
static constexpr int kTmemAllocBarrierId = 8;
static constexpr int kEpilogueBarrierId  = 9;

} // namespace umma
} // namespace kernel
