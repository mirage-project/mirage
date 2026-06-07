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

constexpr int WARP_SIZE = 32;
constexpr int MMA_K = 64;
constexpr uint64_t EVICT_FIRST = 0x12F0000000000000ULL;
constexpr uint64_t EVICT_LAST = 0x14F0000000000000ULL;

__device__ __forceinline__ uint32_t cluster_ctaid_x() {
  uint32_t x;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(x));
  return x;
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

__device__ __forceinline__ uint32_t map_shared_to_cta(int smem_addr,
                                                      int dst_cta) {
  uint32_t mapped;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(mapped)
               : "r"(smem_addr), "r"(dst_cta));
  return mapped;
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr,
                                                      int phase) {
  asm volatile("{\n\t"
               ".reg .pred P1;\n\t"
               "LAB_WAIT:\n\t"
               "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, "
               "[%0], %1;\n\t"
               "@P1 bra.uni DONE;\n\t"
               "bra.uni LAB_WAIT;\n\t"
               "DONE:\n\t"
               "}"
               :
               : "r"(mbar_addr), "r"(phase)
               : "memory");
}

__device__ __forceinline__ void tma_load_bulk(
    int dst, void const *src, int size, int mbar_addr, uint64_t cache_policy) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes."
      "L2::cache_hint [%0], [%1], %2, [%3], %4;"
      :
      : "r"(dst), "l"(src), "r"(size), "r"(mbar_addr), "l"(cache_policy));
}

__device__ __forceinline__ void
    tma_store_2d(int smem_int_ptr, void const *tmap_ptr, int x, int y) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
               "[%0, {%2, %3}], [%1];"
               :
               : "l"(tmap_ptr), "r"(smem_int_ptr), "r"(x), "r"(y)
               : "memory");
}

__device__ __forceinline__ void tma_store_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tma_store_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

template <int N>
__device__ __forceinline__ void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group %0;" ::"n"(N) : "memory");
}

__device__ __forceinline__ void tma_load_3d_multicast(int dst,
                                                      void const *tmap_ptr,
                                                      int x,
                                                      int y,
                                                      int z,
                                                      int mbar_addr,
                                                      uint16_t cta_mask,
                                                      uint64_t cache_policy) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.tile."
               "mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint "
               "[%0], [%1, {%3, %4, %5}], [%2], %6, %7;"
               :
               : "r"(dst),
                 "l"(tmap_ptr),
                 "r"(mbar_addr),
                 "r"(x),
                 "r"(y),
                 "r"(z),
                 "h"(cta_mask),
                 "l"(cache_policy)
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_to_cta0(int mbar_addr) {
  const uint32_t mbar_cta0 = map_shared_to_cta(mbar_addr, 0);
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
               :
               : "r"(mbar_cta0)
               : "memory");
}

struct PersistentTile {
  int row_block;
  int col_block;
};

__device__ __forceinline__ PersistentTile
    map_supergroup_tile(int block_idx,
                        int num_row_blocks,
                        int num_col_blocks,
                        int supergroup_size) {
  int const num_blocks_per_supergroup = supergroup_size * num_col_blocks;
  int const supergroup_idx = block_idx / num_blocks_per_supergroup;
  int const idx_within_supergroup = block_idx % num_blocks_per_supergroup;
  int const first_row = supergroup_idx * supergroup_size;
  int const rows_in_supergroup =
      min(supergroup_size, num_row_blocks - first_row);
  int const row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
  int const col_block = idx_within_supergroup / rows_in_supergroup;
  return {first_row + row_within_supergroup, col_block};
}

template <int RANK, int SM>
__device__ __forceinline__ void tma_load(int dst,
                                         void const *tmap_ptr,
                                         int x,
                                         int y,
                                         int z,
                                         int mbar_addr,
                                         uint64_t cache_policy) {
  static_assert(RANK == 3, "tma_load currently only supports RANK=3");
  if constexpr (SM == 1) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::"
        "bytes.cta_group::1.L2::cache_hint "
        "[%0], [%1, {%2, %3, %4}], [%5], %6;"
        :
        : "r"(dst),
          "l"(tmap_ptr),
          "r"(x),
          "r"(y),
          "r"(z),
          "r"(mbar_addr),
          "l"(cache_policy)
        : "memory");
  } else {
    static_assert(SM == 2, "SM must be 1 or 2");
    const uint32_t mbar_cta0 = map_shared_to_cta(mbar_addr, 0);
    asm volatile("cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global."
                 "mbarrier::complete_tx::bytes.L2::cache_hint "
                 "[%0], [%1, {%2, %3, %4}], [%5], %6;"
                 :
                 : "r"(dst),
                   "l"(tmap_ptr),
                   "r"(x),
                   "r"(y),
                   "r"(z),
                   "r"(mbar_cta0),
                   "l"(cache_policy)
                 : "memory");
  }
}

__device__ __forceinline__ void
    mbarrier_arrive_expect_tx_tile_local(int mbar_addr, int expected_tx) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 "
               "_, [%0], %1;"
               :
               : "r"(mbar_addr), "r"(expected_tx)
               : "memory");
}

__device__ __forceinline__ void
    mbarrier_arrive_expect_tx_tile_cluster(int mbar_addr, int expected_tx) {
  const uint32_t mapped_mbar = map_shared_to_cta(mbar_addr, 0);
  asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 "
               "_, [%0], %1;"
               :
               : "r"(mapped_mbar), "r"(expected_tx)
               : "memory");
}

__device__ __forceinline__ void
    mbarrier_arrive_expect_tx_local(int mbar_addr, int expected_tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :
               : "r"(mbar_addr), "r"(expected_tx)
               : "memory");
}

__device__ __forceinline__ void swapab_arrive_local(int mbar_addr) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
               :
               : "r"(mbar_addr)
               : "memory");
}

template <int SM>
__device__ __forceinline__ void tcgen05_cp_fp4(int taddr, uint64_t s_desc) {
  if constexpr (SM == 1) {
    asm volatile(
        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" ::"r"(taddr),
        "l"(s_desc));
  } else {
    static_assert(SM == 2, "SM must be 1 or 2");
    asm volatile(
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" ::"r"(taddr),
        "l"(s_desc));
  }
}

template <int SM>
__device__ __forceinline__ void tcgen05_mma_nvfp4(uint64_t a_desc,
                                                  uint64_t b_desc,
                                                  uint32_t i_desc,
                                                  int scale_A_tmem,
                                                  int scale_B_tmem,
                                                  int enable_input_d,
                                                  int d_tmem = 0) {
  if constexpr (SM == 1) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}"
        :
        : "r"(d_tmem),
          "l"(a_desc),
          "l"(b_desc),
          "r"(i_desc),
          "r"(scale_A_tmem),
          "r"(scale_B_tmem),
          "r"(enable_input_d));
  } else {
    static_assert(SM == 2, "SM must be 1 or 2");
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X "
        "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}"
        :
        : "r"(d_tmem),
          "l"(a_desc),
          "l"(b_desc),
          "r"(i_desc),
          "r"(scale_A_tmem),
          "r"(scale_B_tmem),
          "r"(enable_input_d));
  }
}

template <int SM>
__device__ __forceinline__ void tcgen05_mma_mxfp4(uint64_t a_desc,
                                                  uint64_t b_desc,
                                                  uint32_t i_desc,
                                                  int scale_A_tmem,
                                                  int scale_B_tmem,
                                                  int enable_input_d,
                                                  int d_tmem = 0) {
  if constexpr (SM == 1) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %6, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 "
                 "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
                 "}"
                 :
                 : "r"(d_tmem),
                   "l"(a_desc),
                   "l"(b_desc),
                   "r"(i_desc),
                   "r"(scale_A_tmem),
                   "r"(scale_B_tmem),
                   "r"(enable_input_d));
  } else {
    static_assert(SM == 2, "SM must be 1 or 2");
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %6, 0;\n\t"
                 "tcgen05.mma.cta_group::2.kind::mxf4.block_scale.block32 "
                 "[%0], %1, %2, %3, [%4], [%5], p;\n\t"
                 "}"
                 :
                 : "r"(d_tmem),
                   "l"(a_desc),
                   "l"(b_desc),
                   "r"(i_desc),
                   "r"(scale_A_tmem),
                   "r"(scale_B_tmem),
                   "r"(enable_input_d));
  }
}

template <int SM>
__device__ __forceinline__ void tcgen05_commit_arrive(int mbar_addr) {
  if constexpr (SM == 1) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
        "[%0];"
        :
        : "r"(mbar_addr)
        : "memory");
  } else {
    static_assert(SM == 2, "SM must be 1 or 2");
    constexpr uint16_t CTA_MASK_2SM = 0x3;
    asm volatile(
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
        "multicast::cluster.b64 [%0], %1;"
        :
        : "r"(mbar_addr), "h"(CTA_MASK_2SM)
        : "memory");
  }
}

template <int SM, int COLS>
__device__ __forceinline__ void tmem_alloc(int smem_addr) {
  if constexpr (SM == 1) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(smem_addr), "r"(COLS));
  } else {
    static_assert(SM == 2, "SM must be 1 or 2");
    asm volatile(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
        :
        : "r"(smem_addr), "r"(COLS));
  }
}

template <int SM, int COLS>
__device__ __forceinline__ void tmem_dealloc(int base_col) {
  if constexpr (SM == 1) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :
                 : "r"(base_col), "r"(COLS));
  } else {
    static_assert(SM == 2, "SM must be 1 or 2");
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :
                 : "r"(base_col), "r"(COLS));
  }
}

struct LD_SHAPE {
  static constexpr char _32x32b[] = ".32x32b";
  static constexpr char _16x256b[] = ".16x256b";
};

struct LD_NUM {
  static constexpr char x4[] = ".x4";
  static constexpr char x8[] = ".x8";
  static constexpr char x16[] = ".x16";
  static constexpr char x32[] = ".x32";
  static constexpr char x64[] = ".x64";
  static constexpr char x128[] = ".x128";
};

template <char const *SHAPE_NAME, char const *NUM_NAME>
__device__ __forceinline__ void
    tcgen05_ld_16regs(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned%17%18.b32 "
               "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
               "  %8,  %9, %10, %11, %12, %13, %14, %15}, [%16];"
               : "=f"(tmp[0]),
                 "=f"(tmp[1]),
                 "=f"(tmp[2]),
                 "=f"(tmp[3]),
                 "=f"(tmp[4]),
                 "=f"(tmp[5]),
                 "=f"(tmp[6]),
                 "=f"(tmp[7]),
                 "=f"(tmp[8]),
                 "=f"(tmp[9]),
                 "=f"(tmp[10]),
                 "=f"(tmp[11]),
                 "=f"(tmp[12]),
                 "=f"(tmp[13]),
                 "=f"(tmp[14]),
                 "=f"(tmp[15])
               : "r"((row << 16) | col), "C"(SHAPE_NAME), "C"(NUM_NAME));
}

template <char const *SHAPE_NAME, char const *NUM_NAME>
__device__ __forceinline__ void
    tcgen05_ld_32regs(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned%33%34.b32 "
               "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
               "  %8,  %9, %10, %11, %12, %13, %14, %15, "
               " %16, %17, %18, %19, %20, %21, %22, %23, "
               " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
               : "=f"(tmp[0]),
                 "=f"(tmp[1]),
                 "=f"(tmp[2]),
                 "=f"(tmp[3]),
                 "=f"(tmp[4]),
                 "=f"(tmp[5]),
                 "=f"(tmp[6]),
                 "=f"(tmp[7]),
                 "=f"(tmp[8]),
                 "=f"(tmp[9]),
                 "=f"(tmp[10]),
                 "=f"(tmp[11]),
                 "=f"(tmp[12]),
                 "=f"(tmp[13]),
                 "=f"(tmp[14]),
                 "=f"(tmp[15]),
                 "=f"(tmp[16]),
                 "=f"(tmp[17]),
                 "=f"(tmp[18]),
                 "=f"(tmp[19]),
                 "=f"(tmp[20]),
                 "=f"(tmp[21]),
                 "=f"(tmp[22]),
                 "=f"(tmp[23]),
                 "=f"(tmp[24]),
                 "=f"(tmp[25]),
                 "=f"(tmp[26]),
                 "=f"(tmp[27]),
                 "=f"(tmp[28]),
                 "=f"(tmp[29]),
                 "=f"(tmp[30]),
                 "=f"(tmp[31])
               : "r"((row << 16) | col), "C"(SHAPE_NAME), "C"(NUM_NAME));
}

template <char const *SHAPE_NAME, char const *NUM_NAME>
__device__ __forceinline__ void
    tcgen05_ld_64regs(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned%65%66.b32 "
               "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
               "  %8,  %9, %10, %11, %12, %13, %14, %15, "
               " %16, %17, %18, %19, %20, %21, %22, %23, "
               " %24, %25, %26, %27, %28, %29, %30, %31, "
               " %32, %33, %34, %35, %36, %37, %38, %39, "
               " %40, %41, %42, %43, %44, %45, %46, %47, "
               " %48, %49, %50, %51, %52, %53, %54, %55, "
               " %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
               : "=f"(tmp[0]),
                 "=f"(tmp[1]),
                 "=f"(tmp[2]),
                 "=f"(tmp[3]),
                 "=f"(tmp[4]),
                 "=f"(tmp[5]),
                 "=f"(tmp[6]),
                 "=f"(tmp[7]),
                 "=f"(tmp[8]),
                 "=f"(tmp[9]),
                 "=f"(tmp[10]),
                 "=f"(tmp[11]),
                 "=f"(tmp[12]),
                 "=f"(tmp[13]),
                 "=f"(tmp[14]),
                 "=f"(tmp[15]),
                 "=f"(tmp[16]),
                 "=f"(tmp[17]),
                 "=f"(tmp[18]),
                 "=f"(tmp[19]),
                 "=f"(tmp[20]),
                 "=f"(tmp[21]),
                 "=f"(tmp[22]),
                 "=f"(tmp[23]),
                 "=f"(tmp[24]),
                 "=f"(tmp[25]),
                 "=f"(tmp[26]),
                 "=f"(tmp[27]),
                 "=f"(tmp[28]),
                 "=f"(tmp[29]),
                 "=f"(tmp[30]),
                 "=f"(tmp[31]),
                 "=f"(tmp[32]),
                 "=f"(tmp[33]),
                 "=f"(tmp[34]),
                 "=f"(tmp[35]),
                 "=f"(tmp[36]),
                 "=f"(tmp[37]),
                 "=f"(tmp[38]),
                 "=f"(tmp[39]),
                 "=f"(tmp[40]),
                 "=f"(tmp[41]),
                 "=f"(tmp[42]),
                 "=f"(tmp[43]),
                 "=f"(tmp[44]),
                 "=f"(tmp[45]),
                 "=f"(tmp[46]),
                 "=f"(tmp[47]),
                 "=f"(tmp[48]),
                 "=f"(tmp[49]),
                 "=f"(tmp[50]),
                 "=f"(tmp[51]),
                 "=f"(tmp[52]),
                 "=f"(tmp[53]),
                 "=f"(tmp[54]),
                 "=f"(tmp[55]),
                 "=f"(tmp[56]),
                 "=f"(tmp[57]),
                 "=f"(tmp[58]),
                 "=f"(tmp[59]),
                 "=f"(tmp[60]),
                 "=f"(tmp[61]),
                 "=f"(tmp[62]),
                 "=f"(tmp[63])
               : "r"((row << 16) | col), "C"(SHAPE_NAME), "C"(NUM_NAME));
}

template <char const *SHAPE_NAME, char const *NUM_NAME>
__device__ __forceinline__ void
    tcgen05_ld_128regs(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned%129%130.b32 "
               "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
               "  %8,  %9, %10, %11, %12, %13, %14, %15, "
               " %16, %17, %18, %19, %20, %21, %22, %23, "
               " %24, %25, %26, %27, %28, %29, %30, %31, "
               " %32, %33, %34, %35, %36, %37, %38, %39, "
               " %40, %41, %42, %43, %44, %45, %46, %47, "
               " %48, %49, %50, %51, %52, %53, %54, %55, "
               " %56, %57, %58, %59, %60, %61, %62, %63, "
               " %64, %65, %66, %67, %68, %69, %70, %71, "
               " %72, %73, %74, %75, %76, %77, %78, %79, "
               " %80, %81, %82, %83, %84, %85, %86, %87, "
               " %88, %89, %90, %91, %92, %93, %94, %95, "
               " %96, %97, %98, %99,%100,%101,%102,%103, "
               "%104,%105,%106,%107,%108,%109,%110,%111, "
               "%112,%113,%114,%115,%116,%117,%118,%119, "
               "%120,%121,%122,%123,%124,%125,%126,%127}, [%128];"
               : "=f"(tmp[0]),
                 "=f"(tmp[1]),
                 "=f"(tmp[2]),
                 "=f"(tmp[3]),
                 "=f"(tmp[4]),
                 "=f"(tmp[5]),
                 "=f"(tmp[6]),
                 "=f"(tmp[7]),
                 "=f"(tmp[8]),
                 "=f"(tmp[9]),
                 "=f"(tmp[10]),
                 "=f"(tmp[11]),
                 "=f"(tmp[12]),
                 "=f"(tmp[13]),
                 "=f"(tmp[14]),
                 "=f"(tmp[15]),
                 "=f"(tmp[16]),
                 "=f"(tmp[17]),
                 "=f"(tmp[18]),
                 "=f"(tmp[19]),
                 "=f"(tmp[20]),
                 "=f"(tmp[21]),
                 "=f"(tmp[22]),
                 "=f"(tmp[23]),
                 "=f"(tmp[24]),
                 "=f"(tmp[25]),
                 "=f"(tmp[26]),
                 "=f"(tmp[27]),
                 "=f"(tmp[28]),
                 "=f"(tmp[29]),
                 "=f"(tmp[30]),
                 "=f"(tmp[31]),
                 "=f"(tmp[32]),
                 "=f"(tmp[33]),
                 "=f"(tmp[34]),
                 "=f"(tmp[35]),
                 "=f"(tmp[36]),
                 "=f"(tmp[37]),
                 "=f"(tmp[38]),
                 "=f"(tmp[39]),
                 "=f"(tmp[40]),
                 "=f"(tmp[41]),
                 "=f"(tmp[42]),
                 "=f"(tmp[43]),
                 "=f"(tmp[44]),
                 "=f"(tmp[45]),
                 "=f"(tmp[46]),
                 "=f"(tmp[47]),
                 "=f"(tmp[48]),
                 "=f"(tmp[49]),
                 "=f"(tmp[50]),
                 "=f"(tmp[51]),
                 "=f"(tmp[52]),
                 "=f"(tmp[53]),
                 "=f"(tmp[54]),
                 "=f"(tmp[55]),
                 "=f"(tmp[56]),
                 "=f"(tmp[57]),
                 "=f"(tmp[58]),
                 "=f"(tmp[59]),
                 "=f"(tmp[60]),
                 "=f"(tmp[61]),
                 "=f"(tmp[62]),
                 "=f"(tmp[63]),
                 "=f"(tmp[64]),
                 "=f"(tmp[65]),
                 "=f"(tmp[66]),
                 "=f"(tmp[67]),
                 "=f"(tmp[68]),
                 "=f"(tmp[69]),
                 "=f"(tmp[70]),
                 "=f"(tmp[71]),
                 "=f"(tmp[72]),
                 "=f"(tmp[73]),
                 "=f"(tmp[74]),
                 "=f"(tmp[75]),
                 "=f"(tmp[76]),
                 "=f"(tmp[77]),
                 "=f"(tmp[78]),
                 "=f"(tmp[79]),
                 "=f"(tmp[80]),
                 "=f"(tmp[81]),
                 "=f"(tmp[82]),
                 "=f"(tmp[83]),
                 "=f"(tmp[84]),
                 "=f"(tmp[85]),
                 "=f"(tmp[86]),
                 "=f"(tmp[87]),
                 "=f"(tmp[88]),
                 "=f"(tmp[89]),
                 "=f"(tmp[90]),
                 "=f"(tmp[91]),
                 "=f"(tmp[92]),
                 "=f"(tmp[93]),
                 "=f"(tmp[94]),
                 "=f"(tmp[95]),
                 "=f"(tmp[96]),
                 "=f"(tmp[97]),
                 "=f"(tmp[98]),
                 "=f"(tmp[99]),
                 "=f"(tmp[100]),
                 "=f"(tmp[101]),
                 "=f"(tmp[102]),
                 "=f"(tmp[103]),
                 "=f"(tmp[104]),
                 "=f"(tmp[105]),
                 "=f"(tmp[106]),
                 "=f"(tmp[107]),
                 "=f"(tmp[108]),
                 "=f"(tmp[109]),
                 "=f"(tmp[110]),
                 "=f"(tmp[111]),
                 "=f"(tmp[112]),
                 "=f"(tmp[113]),
                 "=f"(tmp[114]),
                 "=f"(tmp[115]),
                 "=f"(tmp[116]),
                 "=f"(tmp[117]),
                 "=f"(tmp[118]),
                 "=f"(tmp[119]),
                 "=f"(tmp[120]),
                 "=f"(tmp[121]),
                 "=f"(tmp[122]),
                 "=f"(tmp[123]),
                 "=f"(tmp[124]),
                 "=f"(tmp[125]),
                 "=f"(tmp[126]),
                 "=f"(tmp[127])
               : "r"((row << 16) | col), "C"(SHAPE_NAME), "C"(NUM_NAME));
}

__device__ __forceinline__ void
    tcgen05_ld_32x32bx32(float *tmp, int row, int col) {
  tcgen05_ld_32regs<LD_SHAPE::_32x32b, LD_NUM::x32>(tmp, row, col);
}
__device__ __forceinline__ void
    tcgen05_ld_32x32bx64(float *tmp, int row, int col) {
  tcgen05_ld_64regs<LD_SHAPE::_32x32b, LD_NUM::x64>(tmp, row, col);
}
__device__ __forceinline__ void
    tcgen05_ld_32x32bx128(float *tmp, int row, int col) {
  tcgen05_ld_128regs<LD_SHAPE::_32x32b, LD_NUM::x128>(tmp, row, col);
}

__device__ __forceinline__ void
    tcgen05_ld_16x256bx4(float *tmp, int row, int col) {
  tcgen05_ld_16regs<LD_SHAPE::_16x256b, LD_NUM::x4>(tmp, row, col);
}
__device__ __forceinline__ void
    tcgen05_ld_16x256bx8(float *tmp, int row, int col) {
  tcgen05_ld_32regs<LD_SHAPE::_16x256b, LD_NUM::x8>(tmp, row, col);
}
__device__ __forceinline__ void
    tcgen05_ld_16x256bx16(float *tmp, int row, int col) {
  tcgen05_ld_64regs<LD_SHAPE::_16x256b, LD_NUM::x16>(tmp, row, col);
}

} // namespace sm100_ptx
} // namespace kernel
