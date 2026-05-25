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

#pragma once

#include <cstdint>

#include "common/bfloat16.h"

#include <c10/util/Exception.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

inline void check_cu_2sm(CUresult err) {
  if (err == CUDA_SUCCESS) {
    return;
  }
  const char *error_msg_ptr = nullptr;
  if (cuGetErrorString(err, &error_msg_ptr) != CUDA_SUCCESS) {
    error_msg_ptr = "unable to get error string";
  }
  TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", error_msg_ptr);
}

inline void check_cuda_2sm(cudaError_t err) {
  if (err == cudaSuccess) {
    return;
  }
  TORCH_CHECK(false, cudaGetErrorString(err));
}

inline void init_AB_tmap_2sm(CUtensorMap *tmap,
                         const char *ptr,
                         uint64_t global_height,
                         uint64_t global_width,
                         uint32_t shared_height,
                         uint32_t shared_width) {
  constexpr uint32_t rank = 3;
  uint64_t globalDim[rank]          = {256, global_height, global_width / 256};
  uint64_t globalStrides[rank - 1]  = {global_width / 2, 128};
  uint32_t boxDim[rank]             = {256, shared_height, shared_width / 256};
  uint32_t elementStrides[rank]     = {1, 1, 1};

  CUresult err = cuTensorMapEncodeTiled(
      tmap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
      rank,
      const_cast<char *>(ptr),
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  check_cu_2sm(err);
}

// Tensor map for the NVFP4 scale-factor tensor.
//
// The scale data in gmem is laid out as `[rows/128, K/64, 512]` ue4m3 bytes —
// a contiguous 512-byte tile per (row_block, k_block) pair, addressable as
// `base + ((row_block * (K/64)) + k_block) * 512`. cuTensorMapEncodeTiled
// requires `boxDim[0] * elementSize <= 256` for `SWIZZLE_NONE`, so we split
// each 512-byte tile into two 256-byte halves: the inner-most dim indexes
// half-tiles (2 per (row_block, k_block) pair) of 256 bytes each.
// Final layout: rank-3 UINT8 tensor `{256 bytes, 2*K/64 half-tiles,
// rows/128 row_blocks}`. A `boxDim={256, 2*BLOCK_K/64, 1}` invocation reads
// one row_block × BLOCK_K bytes per CTA. Both CTAs issue the cta_group::2
// TMA; the single cluster completion lands on CTA0's mbarrier.
inline void init_SF_tmap_2sm(CUtensorMap *tmap,
                             const char *ptr,
                             uint64_t rows,
                             uint64_t reduction_size,
                             uint32_t shared_k_blocks) {
  constexpr uint32_t rank = 3;
  // Strides (in bytes): half-tile (256B) → next half-tile = 256 (contiguous).
  // half-tile → row_block = 256 * 2 * (K/64) = 512 * (K/64).
  uint64_t globalDim[rank]         = {256, 2 * (reduction_size / 64), rows / 128};
  uint64_t globalStrides[rank - 1] = {256, 512 * (reduction_size / 64)};
  uint32_t boxDim[rank]            = {256, 2 * shared_k_blocks, 1};
  uint32_t elementStrides[rank]    = {1, 1, 1};

  CUresult err = cuTensorMapEncodeTiled(
      tmap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8,
      rank,
      const_cast<char *>(ptr),
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  check_cu_2sm(err);
}

inline void init_C_tmap_2sm(CUtensorMap *tmap,
                            void *ptr,
                            uint64_t batch_size,
                            uint64_t output_size,
                            uint32_t tile_rows,
                            uint32_t tile_cols) {
  constexpr uint32_t rank = 2;
  uint64_t globalDim[rank] = {output_size, batch_size};
  uint64_t globalStrides[rank - 1] = {output_size * sizeof(type::bfloat16_t)};
  uint32_t boxDim[rank] = {tile_cols, tile_rows};
  uint32_t elementStrides[rank] = {1, 1};

  CUresult err = cuTensorMapEncodeTiled(
      tmap,
      CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
      rank,
      ptr,
      globalDim,
      globalStrides,
      boxDim,
      elementStrides,
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  check_cu_2sm(err);
}

namespace kernel {
namespace nvfp4_1d2d_2sm_detail {

constexpr int WARP_SIZE = 32;
constexpr int MMA_K = 64;
constexpr uint64_t EVICT_FIRST = 0x12F0000000000000ULL;
constexpr uint64_t EVICT_LAST = 0x14F0000000000000ULL;

__device__ __forceinline__ constexpr uint64_t desc_encode(uint64_t x) {
  return (x & 0x3FFFFULL) >> 4ULL;
}

__device__ __forceinline__ uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
      "{\n\t"
      ".reg .pred %%px;\n\t"
      "elect.sync _|%%px, %1;\n\t"
      "@%%px mov.s32 %0, 1;\n\t"
      "}"
      : "+r"(pred)
      : "r"(0xFFFFFFFF));
  return pred;
}

__device__ __forceinline__ uint32_t cluster_ctaid_x() {
  uint32_t x;
  asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(x));
  return x;
}

__device__ __forceinline__ uint32_t cluster_ctaid_y() {
  uint32_t y;
  asm volatile("mov.u32 %0, %%cluster_ctaid.y;" : "=r"(y));
  return y;
}

__device__ __forceinline__ uint32_t cluster_ctaid_z() {
  uint32_t z;
  asm volatile("mov.u32 %0, %%cluster_ctaid.z;" : "=r"(z));
  return z;
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
      "@P1 bra.uni DONE;\n\t"
      "bra.uni LAB_WAIT;\n\t"
      "DONE:\n\t"
      "}"
      :
      : "r"(mbar_addr), "r"(phase), "r"(ticks));
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
  asm volatile(
      "{\n\t"
      ".reg .pred P1;\n\t"
      "LAB_WAIT:\n\t"
      "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra.uni DONE;\n\t"
      "bra.uni LAB_WAIT;\n\t"
      "DONE:\n\t"
      "}"
      :
      : "r"(mbar_addr), "r"(phase)
      : "memory");
}

__device__ __forceinline__ uint32_t map_shared_to_cta(int smem_addr, int dst_cta) {
  uint32_t mapped;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(mapped)
               : "r"(smem_addr), "r"(dst_cta));
  return mapped;
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx_cluster(int mbar_addr,
                                                                  int expected_tx,
                                                                  int dst_cta) {
  const uint32_t mapped_mbar = map_shared_to_cta(mbar_addr, dst_cta);
  asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
               :
               : "r"(mapped_mbar), "r"(expected_tx)
               : "memory");
}

// CTA-local arrive+expect_tx — arms this CTA's own mbar with the bytes that
// will be delivered to it. Used by the consumer-side arming pattern (TK's
// `tma::expect_bytes` equivalent): the MMA leader arms its local scale mbar
// for per-CTA bytes (own SFA share + multicast SFB share) right before
// waiting on it.
__device__ __forceinline__ void mbarrier_arrive_expect_tx_local(int mbar_addr,
                                                                int expected_tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
               :
               : "r"(mbar_addr), "r"(expected_tx)
               : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_2sm_sm0(int mbar_addr) {
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

__device__ __forceinline__ PersistentTile map_supergroup_tile(int block_idx, int num_row_blocks, int num_col_blocks, int supergroup_size) {
  const int num_blocks_per_supergroup = supergroup_size * num_col_blocks;
  const int supergroup_idx = block_idx / num_blocks_per_supergroup;
  const int idx_within_supergroup = block_idx % num_blocks_per_supergroup;
  const int first_row = supergroup_idx * supergroup_size;
  const int rows_in_supergroup = min(supergroup_size, num_row_blocks - first_row);
  const int row_within_supergroup = idx_within_supergroup % rows_in_supergroup;
  const int col_block = idx_within_supergroup / rows_in_supergroup;
  return {first_row + row_within_supergroup, col_block};
}

__device__ __forceinline__ void
tma_gmem2smem(int dst, const void *src, int size, int mbar_addr, uint64_t cache_policy) {
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint "
      "[%0], [%1], %2, [%3], %4;"
      :
      : "r"(dst), "l"(src), "r"(size), "r"(mbar_addr), "l"(cache_policy));
}

__device__ __forceinline__ void tma_3d_gmem2smem(int dst,
                                                 const void *tmap_ptr,
                                                 int x,
                                                 int y,
                                                 int z,
                                                 int mbar_addr,
                                                 uint64_t cache_policy) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes."
      "cta_group::1.L2::cache_hint [%0], [%1, {%2, %3, %4}], [%5], %6;"
      :
      : "r"(dst),
        "l"(tmap_ptr),
        "r"(x),
        "r"(y),
        "r"(z),
        "r"(mbar_addr),
        "l"(cache_policy)
	      : "memory");
}

__device__ __forceinline__ void tma_3d_gmem2smem_2sm(int dst,
                                                     const void *tmap_ptr,
                                                     int x,
                                                     int y,
                                                     int z,
                                                     int mbar_addr,
                                                     uint64_t cache_policy) {
  // For cta_group::2 TMA both CTAs execute the instruction, but transaction
  // completion is reported to CTA0's mbarrier.
  const uint32_t mbar_cta0 = map_shared_to_cta(mbar_addr, 0);
  asm volatile(
      "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global."
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

// Cluster multicast TMA (cta_group::2 form): both CTAs in the cluster issue
// this instruction (required by cta_group::2 semantics) with identical coords
// and cta_mask. The HW performs ONE global fetch and multicasts identical
// bytes into each CTA's smem at `[dst]` whose bit is set in `cta_mask`. The
// "peer bit" (bit 24) is cleared in the mbar address so that BOTH CTAs'
// tx-byte increments route to CTA0's mbarrier — matching the non-multicast
// 2sm path (see CUTLASS SM100_TMA_2SM_LOAD_MULTICAST_3D). This halves DRAM
// bytes for SFB vs. the prior cta_group::2 (non-multicast) SFB path where
// each CTA's TMA engine performed an independent global fetch of the same
// data.
// Cluster multicast TMA (cta_group::2 + multicast::cluster). Issued by BOTH
// CTAs in the cluster (cta_group::2 invariant) with identical coords and the
// same cta_mask covering the multicast destinations. The HW dedups across
// the cluster issuers and performs ONE global fetch, then fans identical
// bytes into each receiver CTA's smem at `[dst]`. tx-byte accounting is
// routed to CTA0's mbar via `map_shared_to_cta(mbar_addr, 0)` (same trick
// the non-multicast 2sm helper uses); each receiver's TMA engine increments
// CTA0's mbar by its own receiver-share (= 1 × the bytes-per-receiver). For
// a 2-CTA cluster with both CTAs in the mask, CTA0's mbar therefore sees
// 2 × bytes-per-receiver in total, identical to the non-multicast accounting.
//
// No L2::cache_hint qualifier — empirically the multicast variant deadlocks
// when armed against the existing scale_mbar with EVICT_FIRST L2 policy, and
// scales are accessed too few times to benefit from explicit eviction hints
// anyway. Matches the TK pattern (cluster::load_async, default cache policy).
__device__ __forceinline__ void tma_3d_gmem2smem_2sm_multicast(int dst,
                                                               const void *tmap_ptr,
                                                               int x,
                                                               int y,
                                                               int z,
                                                               int mbar_addr,
                                                               uint16_t cta_mask,
                                                               uint64_t cache_policy) {
  // Mirrors CUTLASS SM100_TMA_2SM_LOAD_MULTICAST_3D: clear peer bit (0xFEFFFFFF)
  // so tx-byte accounting routes to CTA0's mbar; PTX qualifier order matches.
  const uint32_t mbar_peer = static_cast<uint32_t>(mbar_addr) & 0xFEFFFFFFu;
  asm volatile(
      "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global."
      "mbarrier::complete_tx::bytes.multicast::cluster.L2::cache_hint "
      "[%0], [%1, {%3, %4, %5}], [%2], %6, %7;"
      :
      : "r"(dst),
        "l"(tmap_ptr),
        "r"(mbar_peer),
        "r"(x),
        "r"(y),
        "r"(z),
        "h"(cta_mask),
        "l"(cache_policy)
      : "memory");
}

// TK-pattern cluster multicast TMA, cta_group::1 form. Each CTA independently
// issues with its own coords and a cta_mask; the issuing CTA's TMA engine
// fetches gmem once and writes to every receiver's smem named in cta_mask.
// Each receiver's local mbar is incremented by bytes-per-receiver. Unlike the
// cta_group::2 multicast variant, both CTAs need NOT issue the same instruction
// — they may issue different coords (used for CTA-partitioned SFB loads).
__device__ __forceinline__ void tma_3d_gmem2smem_multicast_g1(int dst,
                                                              const void *tmap_ptr,
                                                              int x,
                                                              int y,
                                                              int z,
                                                              int mbar_addr,
                                                              uint16_t cta_mask,
                                                              uint64_t cache_policy) {
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
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

__device__ __forceinline__ void tma_2d_smem2gmem(int smem_int_ptr,
                                                 const void *tmap_ptr,
                                                 int x,
                                                 int y) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%2, %3}], [%1];"
      :
      : "l"(tmap_ptr), "r"(smem_int_ptr), "r"(x), "r"(y)
      : "memory");
}

__device__ __forceinline__ void tma_store_fence_2sm() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

__device__ __forceinline__ void tma_store_arrive_2sm() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

template <int N>
__device__ __forceinline__ void tma_store_wait_2sm() {
  asm volatile("cp.async.bulk.wait_group %0;" :: "n"(N) : "memory");
}

__device__ __forceinline__ void tcgen05_cp_nvfp4(int taddr, uint64_t s_desc) {
  asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
}

__device__ __forceinline__ void tcgen05_mma_nvfp4(uint64_t a_desc,
                                                  uint64_t b_desc,
                                                  uint32_t i_desc,
                                                  int scale_A_tmem,
                                                  int scale_B_tmem,
                                                  int enable_input_d) {
  const int d_tmem = 0;
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

__device__ __forceinline__ void umma_arrive_multicast_2sm(int mbar_addr) {
  constexpr uint16_t CTA_MASK_2SM = 0x3;
  asm volatile(
      "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
      "multicast::cluster.b64 [%0], %1;"
      :
      : "r"(mbar_addr), "h"(CTA_MASK_2SM)
      : "memory");
}

struct SHAPE {
  static constexpr char _32x32b[] = ".32x32b";
};

struct NUM {
  static constexpr char x32[] = ".x32";
  static constexpr char x64[] = ".x64";
  static constexpr char x128[] = ".x128";
};

template <const char *SHAPE_NAME, const char *NUM_NAME>
__device__ __forceinline__ void tcgen05_ld_32regs(float *tmp, int row, int col) {
  asm volatile(
      "tcgen05.ld.sync.aligned%33%34.b32 "
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

template <const char *SHAPE_NAME, const char *NUM_NAME>
__device__ __forceinline__ void tcgen05_ld_64regs(float *tmp, int row, int col) {
  asm volatile(
      "tcgen05.ld.sync.aligned%65%66.b32 "
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

template <const char *SHAPE_NAME, const char *NUM_NAME>
__device__ __forceinline__ void tcgen05_ld_128regs(float *tmp, int row, int col) {
  asm volatile(
      "tcgen05.ld.sync.aligned%129%130.b32 "
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

__device__ __forceinline__ void tcgen05_ld_32x32bx32(float *tmp, int row, int col) {
  tcgen05_ld_32regs<SHAPE::_32x32b, NUM::x32>(tmp, row, col);
}
__device__ __forceinline__ void tcgen05_ld_32x32bx64(float *tmp, int row, int col) {
  tcgen05_ld_64regs<SHAPE::_32x32b, NUM::x64>(tmp, row, col);
}
__device__ __forceinline__ void tcgen05_ld_32x32bx128(float *tmp, int row, int col) {
  tcgen05_ld_128regs<SHAPE::_32x32b, NUM::x128>(tmp, row, col);
}

}  // namespace nvfp4_1d2d_2sm_detail

template <int REDUCTION_SIZE,
          int BLOCK_M,
          int BLOCK_N,
          int BLOCK_K,
          int NUM_STAGES,
          int SUPERGROUP_SIZE,
          int EPI_TILE_N,
          int EPI_NUM_D_TILES,
          bool EPI_BATCHED,
          int EPI_BATCH_LA,
          bool OVERLAP_OUTPUT_MBAR,
          bool HAS_BIAS>
__global__ __launch_bounds__(BLOCK_M + 4 * 32) void
linear_nvfp4_1d2d_2sm_sm100_kernel(const __grid_constant__ CUtensorMap A_tmap,
                                   const __grid_constant__ CUtensorMap B_tmap,
                                   const __grid_constant__ CUtensorMap C_tmap,
                                   const __grid_constant__ CUtensorMap SFA_tmap,
                                   const __grid_constant__ CUtensorMap SFB_tmap,
                                   const type::bfloat16_t *bias_ptr,
                                   int M,
                                   int N) {
  using namespace nvfp4_1d2d_2sm_detail;

  // Experimental 2SM variant. Launch this kernel with cooperative clusters
  // containing two CTAs in M, e.g. cluster_dim=(2, 1, 1). Each CTA still loads
  // and stores a local 128-row A slice; B is split across the peer CTAs as
  // [2, N/2, K], matching tcgen05.cta_group::2 operand partitioning.
  static_assert(BLOCK_M == 128, "2SM SM100 NVFP4 uses 128 rows per CTA");
  static_assert(BLOCK_N == 256, "2SM kernel requires BLOCK_N == 256");
  static_assert(BLOCK_K % MMA_K == 0, "BLOCK_K must be divisible by MMA_K");
  static_assert(REDUCTION_SIZE % BLOCK_K == 0, "K must be divisible by BLOCK_K");
  static_assert(BLOCK_N == 32 || BLOCK_N == 64 || BLOCK_N == 128 || BLOCK_N == 256,
                "BLOCK_N must be 32, 64, 128, or 256");

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int cta_group_m = static_cast<int>(cluster_ctaid_x());
  const int cluster_idx = static_cast<int>(blockIdx.x) / 2;
  const int num_clusters = static_cast<int>(gridDim.x) / 2;
  const int num_m_tiles = M / (2 * BLOCK_M);
  const int num_n_tiles = N / BLOCK_N;
  const int num_output_tiles = num_m_tiles * num_n_tiles;

  constexpr int EPILOGUE_WARPS = BLOCK_M / WARP_SIZE;
  constexpr int MMA_WARP = EPILOGUE_WARPS;
  constexpr int SCALE_TMA_WARP = EPILOGUE_WARPS + 2;
  constexpr int TILE_TMA_WARP = EPILOGUE_WARPS + 3;
  static_assert(SUPERGROUP_SIZE > 0, "SUPERGROUP_SIZE must be positive");
  static_assert(EPI_TILE_N == 32 || EPI_TILE_N == 64 || EPI_TILE_N == 128, "EPI_TILE_N must be 32, 64, or 128");
  static_assert(EPI_NUM_D_TILES > 0, "EPI_NUM_D_TILES must be positive");
  static_assert(BLOCK_N % EPI_TILE_N == 0, "BLOCK_N must be divisible by EPI_TILE_N");
  constexpr int EPI_PIPE_DEPTH = BLOCK_N / EPI_TILE_N;
  constexpr int EPI_TILE_BYTES = EPI_TILE_N * BLOCK_M * sizeof(type::bfloat16_t);
  static_assert(EPI_BATCH_LA >= 1, "EPI_BATCH_LA must be >= 1");
  static_assert(EPI_BATCH_LA <= EPI_PIPE_DEPTH, "EPI_BATCH_LA must not exceed EPI_PIPE_DEPTH");
  static_assert(EPI_PIPE_DEPTH % EPI_BATCH_LA == 0, "EPI_PIPE_DEPTH must be divisible by EPI_BATCH_LA");

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int B_LOCAL_N = BLOCK_N / 2;
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = B_LOCAL_N * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;
  constexpr int SFB_SCALE_TILES = (BLOCK_N + 127) / 128;
  constexpr int SFB_TILE_BYTES = 128 * BLOCK_K / 16;
  constexpr int SFB_size = SFB_SCALE_TILES * SFB_TILE_BYTES;  // per-CTA smem
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;
  constexpr int SCALE_EXPECTED_TX = SFA_size + SFB_size;

#pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 3 + 2];
  const int tile_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int scale_mbar_addr = tile_mbar_addr + NUM_STAGES * 8;
  const int mma_mbar_addr = scale_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  const int output_mbar_addr = mainloop_mbar_addr + 8;

  constexpr int MMA_PER_TILE = BLOCK_K / MMA_K;
  constexpr int SCALE_ADDR_DIV = 4;
  constexpr int SFA_STAGE_STRIDE = (16 * MMA_PER_TILE) / SCALE_ADDR_DIV;
  constexpr int SFB_STAGE_STRIDE = (16 * SFB_SCALE_TILES * MMA_PER_TILE) / SCALE_ADDR_DIV;
  constexpr int SFA_K_STRIDE = 16 / SCALE_ADDR_DIV;
  constexpr int SFB_K_STRIDE = (16 * SFB_SCALE_TILES) / SCALE_ADDR_DIV;
  constexpr int SFB_N_TILE_STRIDE = 16 / SCALE_ADDR_DIV;
  constexpr int SFA_tmem = BLOCK_N;
  constexpr int SFB_tmem = SFA_tmem + SFA_STAGE_STRIDE * NUM_STAGES;
  constexpr int TMEM_ALLOC_COLS = BLOCK_N * 2;
  constexpr int TILE_EXPECTED_TX = 2 * A_size + 2 * B_size;

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++) {
      mbarrier_init(tile_mbar_addr + i * 8, 1);
      mbarrier_init(scale_mbar_addr + i * 8, 1);
      mbarrier_init(mma_mbar_addr + i * 8, 1);
    }
    mbarrier_init(mainloop_mbar_addr, 1);
    mbarrier_init(output_mbar_addr, 2);
    asm volatile("fence.mbarrier_init.release.cluster;");
  } else if (warp_id == 1) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :
                 : "r"(smem), "r"(TMEM_ALLOC_COLS));
  }
  __syncthreads();
  cluster_sync();

  constexpr int NUM_ITERS = REDUCTION_SIZE / BLOCK_K;
  const uint64_t cache_A = (M > N) ? EVICT_FIRST : EVICT_LAST;
  const uint64_t cache_B = (M > N) ? EVICT_LAST : EVICT_FIRST;

  auto make_desc_AB = [](int addr) -> uint64_t {
    const int SBO = 8 * 128;
    return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
  };
  auto make_desc_SF = [](int addr) -> uint64_t {
    const int SBO = 8 * 16;
    return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
  };

  if (warp_id == TILE_TMA_WARP && elect_sync()) {
    
    for (int output_tile = cluster_idx, work_idx = 0; output_tile < num_output_tiles; output_tile += num_clusters, work_idx++) {
      const PersistentTile tile = map_supergroup_tile(output_tile, num_m_tiles, num_n_tiles, SUPERGROUP_SIZE);
      const int off_m = (tile.row_block * 2 + cta_group_m) * BLOCK_M;
      const int off_n = tile.col_block * BLOCK_N;
      for (int iter_k = 0; iter_k < NUM_ITERS; iter_k++) {
        const int pipeline_iter = work_idx * NUM_ITERS + iter_k;
        const int stage_id = pipeline_iter % NUM_STAGES;
        const int mma_phase = ((pipeline_iter - NUM_STAGES) / NUM_STAGES) % 2;

        const int mbar_addr = tile_mbar_addr + stage_id * 8;
        const int A_smem = smem + stage_id * STAGE_SIZE;
        const int B_smem = A_smem + A_size;
        const int off_k = iter_k * BLOCK_K;

        if (pipeline_iter >= NUM_STAGES) mbarrier_wait(mma_mbar_addr + stage_id * 8, mma_phase);
        if (cta_group_m == 0) mbarrier_arrive_expect_tx_cluster(mbar_addr, TILE_EXPECTED_TX, 0);
        tma_3d_gmem2smem_2sm(A_smem, &A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_A);
        tma_3d_gmem2smem_2sm(B_smem, &B_tmap, 0, off_n + cta_group_m * B_LOCAL_N, off_k / 256, mbar_addr, cache_B);
      }
    }
  } else if (warp_id == SCALE_TMA_WARP && elect_sync()) {
    for (int output_tile = cluster_idx, work_idx = 0; output_tile < num_output_tiles; output_tile += num_clusters, work_idx++) {
      const PersistentTile tile = map_supergroup_tile(output_tile, num_m_tiles, num_n_tiles, SUPERGROUP_SIZE);
      const int off_m = (tile.row_block * 2 + cta_group_m) * BLOCK_M;
      const int off_n = tile.col_block * BLOCK_N;
      for (int iter_k = 0; iter_k < NUM_ITERS; iter_k++) {
        const int pipeline_iter = work_idx * NUM_ITERS + iter_k;
        const int stage_id  = pipeline_iter % NUM_STAGES;
        const int mma_phase = ((pipeline_iter - NUM_STAGES) / NUM_STAGES) % 2;

        const int mbar_addr = scale_mbar_addr + stage_id * 8;
        const int A_smem = smem + stage_id * STAGE_SIZE;
        const int SFA_smem = A_smem + A_size + B_size;
        const int SFB_smem = SFA_smem + SFA_size;
        const int off_k = iter_k * BLOCK_K;
        const uint16_t self_mask = static_cast<uint16_t>(1u << cta_group_m);

        if (pipeline_iter >= NUM_STAGES) mbarrier_wait(mma_mbar_addr + stage_id * 8, mma_phase);
        tma_3d_gmem2smem_multicast_g1(SFA_smem, &SFA_tmap, 0, 2 * (off_k / 64), off_m / 128, mbar_addr, self_mask, cache_A);
        tma_3d_gmem2smem_multicast_g1(SFB_smem + cta_group_m * SFB_TILE_BYTES, &SFB_tmap, 0, 2 * (off_k / 64), (off_n / 128) + cta_group_m, mbar_addr, 0b11, cache_B);
      }
    }
  } else if (warp_id == MMA_WARP) {
    constexpr int MMA_M = 256;
    constexpr int MMA_N = BLOCK_N;
    constexpr uint32_t i_desc = (1U << 7U) | (1U << 10U) | ((uint32_t) MMA_N >> 3U << 17U) | ((uint32_t) MMA_M >> 7U << 27U);
    const bool mma_leader = (cta_group_m == 0) && elect_sync();

    if (mma_leader) {
      for (int output_tile = cluster_idx, work_idx = 0; output_tile < num_output_tiles; output_tile += num_clusters, work_idx++) {
        
        const PersistentTile tile = map_supergroup_tile(output_tile, num_m_tiles, num_n_tiles, SUPERGROUP_SIZE);
        const int tile_n = tile.col_block;
        mbarrier_wait(output_mbar_addr, (work_idx - 1) % 2);
        // TK pattern: fence after tmem is released by epilogue, before scale-cp begins.
        asm volatile("tcgen05.fence::after_thread_sync;");

        for (int iter_k = 0; iter_k < NUM_ITERS; iter_k++) {
          const int pipeline_iter = work_idx * NUM_ITERS + iter_k;
          const int stage_id = pipeline_iter % NUM_STAGES;
          const int tma_phase = (pipeline_iter / NUM_STAGES) % 2;
          const int A_smem   = smem + stage_id * STAGE_SIZE;
          const int B_smem   = A_smem + A_size;
          const int SFA_smem = B_smem + B_size;
          const int SFB_smem = SFA_smem + SFA_size;
          const uint64_t SFA_desc = make_desc_SF(SFA_smem);
          const uint64_t SFB_desc = make_desc_SF(SFB_smem);

          auto copy_scale_k = [&](int k) {
            uint64_t sfa_desc = SFA_desc + static_cast<uint64_t>(k) * (512ULL >> 4ULL);
            tcgen05_cp_nvfp4(SFA_tmem + stage_id * SFA_STAGE_STRIDE + k * SFA_K_STRIDE, sfa_desc);
            #pragma unroll
            for (int n_tile = 0; n_tile < SFB_SCALE_TILES; n_tile++) {
              uint64_t sfb_desc = SFB_desc + static_cast<uint64_t>(n_tile) * (SFB_TILE_BYTES >> 4ULL) + static_cast<uint64_t>(k) * (512ULL >> 4ULL);
              tcgen05_cp_nvfp4(SFB_tmem + stage_id * SFB_STAGE_STRIDE + k * SFB_K_STRIDE + n_tile * SFB_N_TILE_STRIDE, sfb_desc);
            }
          };

          mbarrier_arrive_expect_tx_local(scale_mbar_addr + stage_id * 8, SCALE_EXPECTED_TX);
          mbarrier_wait(scale_mbar_addr + stage_id * 8, tma_phase);

          #pragma unroll
          for (int k_sf = 0; k_sf < BLOCK_K / MMA_K; k_sf++) {
            copy_scale_k(k_sf);
          }

          mbarrier_wait_cluster(tile_mbar_addr + stage_id * 8, tma_phase);

          #pragma unroll
          for (int k = 0; k < BLOCK_K / MMA_K; k++) {
            const int k1 = k / 4;
            const int k2 = k % 4;

            uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
            uint64_t b_desc = make_desc_AB(B_smem + k1 * B_LOCAL_N * 128 + k2 * 32);

            const int scale_A_tmem = SFA_tmem + stage_id * SFA_STAGE_STRIDE + k * SFA_K_STRIDE;
            const int scale_B_tmem = SFB_tmem + stage_id * SFB_STAGE_STRIDE + k * SFB_K_STRIDE + ((BLOCK_N < 128) ? (tile_n % (128 / BLOCK_N)) * (BLOCK_N / 32) : 0);
            const int enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;

            tcgen05_mma_nvfp4(a_desc, b_desc, i_desc, scale_A_tmem, scale_B_tmem, enable_input_d);
          }

          umma_arrive_multicast_2sm(mma_mbar_addr + stage_id * 8);
        }
        umma_arrive_multicast_2sm(mainloop_mbar_addr);
      }
    }
  } else if (warp_id < EPILOGUE_WARPS) {
    for (int output_tile = cluster_idx, work_idx = 0; output_tile < num_output_tiles; output_tile += num_clusters, work_idx++) {
      const PersistentTile tile = map_supergroup_tile(output_tile, num_m_tiles, num_n_tiles, SUPERGROUP_SIZE);
      const int off_m = (tile.row_block * 2 + cta_group_m) * BLOCK_M;
      const int off_n = tile.col_block * BLOCK_N;

      mbarrier_wait(mainloop_mbar_addr, work_idx % 2);
      asm volatile("tcgen05.fence::after_thread_sync;");

      auto epilogue_M_major = [&]() {
        const int tmem_row_base = cta_group_m * BLOCK_M;
        const int out_smem_addr = smem + STAGE_SIZE * NUM_STAGES;
        type::bfloat16_t *out_smem = reinterpret_cast<type::bfloat16_t *>(smem_ptr + STAGE_SIZE * NUM_STAGES);

        auto load_subtile = [&](float *dst, int n) {
          if constexpr (EPI_TILE_N == 128) {
            tcgen05_ld_32x32bx128(dst, tmem_row_base + warp_id * 32, n * EPI_TILE_N);
          }
          if constexpr (EPI_TILE_N == 64) {
            tcgen05_ld_32x32bx64(dst, tmem_row_base + warp_id * 32, n * EPI_TILE_N);
          }
          if constexpr (EPI_TILE_N == 32) {
            tcgen05_ld_32x32bx32(dst, tmem_row_base + warp_id * 32, n * EPI_TILE_N);
          }
        };

        auto store_subtile = [&](const float *src, int n) {

          // Wait for SMEM buffer
          const int buffer_id = n % EPI_NUM_D_TILES;
          if (n >= EPI_NUM_D_TILES) {
            if (warp_id == 0 && elect_sync()) {
              tma_store_wait_2sm<EPI_NUM_D_TILES - 1>();
            }
            asm volatile("bar.sync 1, %0;" : : "r"(BLOCK_M) : "memory");
          }
    
          // Store into SMEM (and add bias)
          type::bfloat16_t *out_smem_tile = out_smem + buffer_id * EPI_TILE_N * BLOCK_M;
          if constexpr (HAS_BIAS) {
            const type::bfloat16_t *bias_row = bias_ptr + (off_n + n * EPI_TILE_N) * M + off_m + tid;
            for (int i = 0; i < EPI_TILE_N; i++) {
              type::bfloat16_t acc_bf16(src[i]);
              out_smem_tile[i * BLOCK_M + tid] = acc_bf16 + bias_row[i * M];
            }
          } else {
            for (int i = 0; i < EPI_TILE_N; i++) {
              out_smem_tile[i * BLOCK_M + tid] = type::bfloat16_t(src[i]);
            }
          }
          tma_store_fence_2sm();
          asm volatile("bar.sync 1, %0;" : : "r"(BLOCK_M) : "memory");

          // Move from SMEM to GMEM
          if (warp_id == 0 && elect_sync()) {
            tma_2d_smem2gmem(out_smem_addr + buffer_id * EPI_TILE_BYTES, &C_tmap, off_m, off_n + n * EPI_TILE_N);
            tma_store_arrive_2sm();
          }
        };

        // Distributed-register early-release (TK pattern, register-neutral):
        // process one subtile at a time (so peak registers stay at one
        // EPI_TILE_N-wide slice per thread, NOT the whole tile), but issue the
        // last subtile's TMEM read FIRST and, once all reads have retired,
        // signal output_mbar before the (slow) convert+store of the remaining
        // subtiles. This frees the accumulator as soon as the final TMEM read
        // completes — overlapping this tile's epilogue stores with the next
        // tile's MMA — without ever holding the full tile in registers.
        //
        // Read order: load subtile (EPI_PIPE_DEPTH-1) first (its TMEM read is
        // what we gate the release on), then 0..EPI_PIPE_DEPTH-2. After the
        // last of these loads + wait, every accumulator column has been read,
        // so we can release. We keep one slice in flight: load(k), wait,
        // [if k is the final read: release], store(k).
        if constexpr (OVERLAP_OUTPUT_MBAR) {
          #pragma unroll
          for (int j = 0; j < EPI_PIPE_DEPTH; j++) {
            float tmp[EPI_TILE_N];
            load_subtile(tmp, j);
            if (j == EPI_PIPE_DEPTH - 1) {
              // All accumulator columns read — release for the next MMA.
              asm volatile("tcgen05.wait::ld.sync.aligned;");
              asm volatile("tcgen05.fence::before_thread_sync;\n");
              asm volatile("bar.sync 1, %0;" : : "r"(BLOCK_M) : "memory");
              if (warp_id == 0 && elect_sync()) {
                mbarrier_arrive_2sm_sm0(output_mbar_addr);
              }
            }
            store_subtile(tmp, j);
          }
          if (warp_id == 0 && elect_sync()) {
            tma_store_wait_2sm<0>();
          }
        } 
        
        else if constexpr (!EPI_BATCHED) {
          for (int n = 0; n < EPI_PIPE_DEPTH; n++) {
            float tmp[EPI_TILE_N];
            load_subtile(tmp, n);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            store_subtile(tmp, n);
          }
          if (warp_id == 0 && elect_sync()) {
            tma_store_wait_2sm<0>();
          }
        } 
        
        else {
          // Batched: issue EPI_BATCH_LA TMEM loads back-to-back so their
          // latencies overlap under a single wait, then run the store loop.
          for (int g = 0; g < EPI_PIPE_DEPTH; g += EPI_BATCH_LA) {
            float tmp_batch[EPI_BATCH_LA][EPI_TILE_N];
            #pragma unroll
            for (int b = 0; b < EPI_BATCH_LA; b++) {
              load_subtile(tmp_batch[b], g + b);
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            #pragma unroll
            for (int b = 0; b < EPI_BATCH_LA; b++) {
              store_subtile(tmp_batch[b], g + b);
            }
          }
          if (warp_id == 0 && elect_sync()) {
            tma_store_wait_2sm<0>();
          }
        }
      };

      epilogue_M_major();
      asm volatile("bar.sync 1, %0;" : : "r"(BLOCK_M) : "memory");
      // Non-early-release paths signal output_mbar after the full epilogue. The
      // early-release (OVERLAP_OUTPUT_MBAR) path already signaled it mid-loop,
      // right after the final TMEM read.
      if constexpr (!OVERLAP_OUTPUT_MBAR) {
        if (warp_id == 0 && elect_sync()) {
          mbarrier_arrive_2sm_sm0(output_mbar_addr);
        }
      }
    }
  }

  __syncthreads();
  cluster_sync();

  if (warp_id == 0) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :
                 : "r"(0), "r"(TMEM_ALLOC_COLS));
  }
}

}  // namespace kernel
