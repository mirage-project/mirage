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

// Raw shared-memory storage for the pure-CUDA Blackwell linear kernel.
// Mirrors PipedScaledEpilogueSharedStorage from storage.cuh but without any
// CuTe / cutlass dependency: SMEM is just byte arrays with explicit stage
// strides and the kernel computes pointers directly. The swizzle pattern is
// enforced by the TMA descriptor on write and by the UMMA SMEM descriptor on
// read; the storage itself only needs to provide the correct byte layout.

#pragma once

#include <cstdint>

namespace kernel {
namespace umma {

// Layout assumptions (1d2d kernel, NVFP4 single-CTA):
//
//   A: [MMA_M (=128), bK (=256)] FP4 packed -> bytes_per_stage = MMA_M*bK/2
//   B: [MMA_N,        bK]        FP4 packed -> bytes_per_stage = MMA_N*bK/2
//   C: [MMA_M,        EPI_N] fp32 staged    -> bytes_per_stage = MMA_M*EPI_N*4
//   SFA: 256 bytes per K-tile per (MMA_M/128) row block
//        -> bytes_per_stage = (MMA_M/128) * (bK/64) * 256
//   SFB: 256 bytes per K-tile per (MMA_N_SFB/128) col block
//        -> bytes_per_stage = (MMA_N_SFB/128) * (bK/64) * 256
//
// The SF tiles use the standard Sm1xxBlockScaledConfig packing with BLK_MN=128
// in the SF M/N axis: 32 lanes x 4 inner x 4 K = 256 bytes per (128, 16) SF
// block. (Sourced from the TMA SFA/SFB descriptor shape in the wrapper.)
//
// All buffers are 128-byte aligned for tcgen05.cp / TMA / UMMA descriptor
// requirements. Stage strides are returned in bytes for direct pointer
// arithmetic.

template <int MMA_M_,
          int MMA_N_,
          int MMA_K_,
          int NUM_MMA_K_,
          int EPI_N_,
          int NUM_AB_STAGE_,
          int NUM_ACC_STAGE_,
          int NUM_C_STAGE_,
          int MMA_N_SFB_ = ((MMA_N_ + 127) / 128) * 128>
struct PipedScaledEpilogueRawStorage {
  static constexpr int MMA_M         = MMA_M_;
  static constexpr int MMA_N         = MMA_N_;
  static constexpr int MMA_K         = MMA_K_;
  static constexpr int NUM_MMA_K     = NUM_MMA_K_;
  static constexpr int EPI_N         = EPI_N_;
  static constexpr int NUM_AB_STAGE  = NUM_AB_STAGE_;
  static constexpr int NUM_ACC_STAGE = NUM_ACC_STAGE_;
  static constexpr int NUM_C_STAGE   = NUM_C_STAGE_;
  static constexpr int MMA_N_SFB     = MMA_N_SFB_;

  static constexpr int bK            = MMA_K * NUM_MMA_K;

  // Per-stage byte sizes.
  static constexpr int A_STAGE_BYTES   = MMA_M     * bK / 2;          // FP4
  static constexpr int B_STAGE_BYTES   = MMA_N     * bK / 2;          // FP4
  static constexpr int C_STAGE_BYTES   = MMA_M     * EPI_N * 4;       // fp32
  static constexpr int SFA_STAGE_BYTES =
      ((MMA_M + 127) / 128) * NUM_MMA_K * 256;
  static constexpr int SFB_STAGE_BYTES =
      ((MMA_N_SFB + 127) / 128) * NUM_MMA_K * 256;

  // Per-MMA (single 128xMMA_K SF) block byte size = 256 bytes.
  static constexpr int SF_BLOCK_BYTES = 256;

  // Total bytes for each multi-stage buffer.
  static constexpr int A_BYTES   = A_STAGE_BYTES   * NUM_AB_STAGE;
  static constexpr int B_BYTES   = B_STAGE_BYTES   * NUM_AB_STAGE;
  static constexpr int C_BYTES   = C_STAGE_BYTES   * NUM_C_STAGE;
  static constexpr int SFA_BYTES = SFA_STAGE_BYTES * NUM_AB_STAGE;
  static constexpr int SFB_BYTES = SFB_STAGE_BYTES * NUM_AB_STAGE;

  // Buffers.
  alignas(128) uint8_t A  [A_BYTES];
  alignas(128) uint8_t B  [B_BYTES];
  alignas(128) uint8_t C  [C_BYTES];
  alignas(128) uint8_t SFA[SFA_BYTES];
  alignas(128) uint8_t SFB[SFB_BYTES];

  // Pipeline mbarriers.
  alignas(16) uint64_t ab_full_mbar [NUM_AB_STAGE];
  alignas(16) uint64_t ab_empty_mbar[NUM_AB_STAGE];
  alignas(16) uint64_t sf_full_mbar [NUM_AB_STAGE];
  alignas(16) uint64_t sf_empty_mbar[NUM_AB_STAGE];
  alignas(16) uint64_t acc_full_mbar [NUM_ACC_STAGE];
  alignas(16) uint64_t acc_empty_mbar[NUM_ACC_STAGE];

  // TMEM column pointers, populated by the allocator warp.
  alignas(16) uint32_t tmem_acc_ptr;
  alignas(16) uint32_t tmem_sfa_ptr;
  alignas(16) uint32_t tmem_sfb_ptr;

  // Stage pointer accessors.
  __device__ inline uint8_t *a_stage(int s) {
    return A + s * A_STAGE_BYTES;
  }
  __device__ inline uint8_t *b_stage(int s) {
    return B + s * B_STAGE_BYTES;
  }
  __device__ inline uint8_t *c_stage(int s) {
    return C + s * C_STAGE_BYTES;
  }
  __device__ inline uint8_t *sfa_stage(int s) {
    return SFA + s * SFA_STAGE_BYTES;
  }
  __device__ inline uint8_t *sfb_stage(int s) {
    return SFB + s * SFB_STAGE_BYTES;
  }
};

} // namespace umma
} // namespace kernel
