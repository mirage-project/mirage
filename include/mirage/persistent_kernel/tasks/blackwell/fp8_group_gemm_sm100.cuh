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

#include "sm100_utils.cuh"
#include <cstdio>
#include <iostream>

// Cutlass includes
#include <cutlass/arch/barrier.h>     // ClusterTransactionBarrier, NamedBarrier
#include <cutlass/cluster_launch.hpp> // Cluster launch utilities
#include <cutlass/cutlass.h>          // Core CUTLASS types
#include <cutlass/numeric_conversion.h> // NumericConverter (FP32→BF16)
#include <cutlass/numeric_types.h> // float_e4m3_t, float_ue8m0_t, bfloat16_t

// CuTe includes
#include <cute/algorithm/cooperative_copy.hpp> // Cooperative copy algorithms
#include <cute/arch/cluster_sm90.hpp> // Cluster synchronization primitives
#include <cute/arch/copy_sm100.hpp> // SM100_UTCCP_4x32dp128bit_1cta (scale copy to TMEM)
#include <cute/arch/mma_sm100_desc.hpp> // UMMA::SmemDescriptor, make_instr_desc_block_scaled
#include <cute/arch/mma_sm100_umma.hpp> // SM100_MMA_MXF8F6F4_SS (block-scaled FP8 UMMA)
#include <cute/arch/tmem_allocator_sm100.hpp> // TMEM::Allocator1Sm (tensor memory allocator)
#include <cute/atom/mma_traits_sm100.hpp> // Layout_K_SW128_Atom, UMMA namespace
#include <cute/numeric/integral_constant.hpp> // Compile-time integer constants
#include <cute/tensor.hpp>                    // CuTe tensor abstractions

#include "../common/dmem_layout.cuh" // Device memory layout helpers
#include "../common/worker_config.h" // MPK worker configuration
#include "../hopper/barrier.cuh" // Barrier helper functions (try_wait_barrier)
#include "../hopper/smem_layout_tma.cuh" // smem_tma layout descriptor
#include "../hopper/tma.cuh"             // TMA copy async wrapper
#include "storage.cuh"                   // Shared storage base types

namespace kernel {

// ===================================================================
// FP8 Block-Scaled MoE Group GEMM for SM100 (Blackwell B200)
// ===================================================================
// MATHEMATICAL OPERATION
// ===================================================================
// For each expert `e` assigned to this task, the kernel computes:
//   output[token, topk_slot, :] = input_fp8[token, :] @ weight_fp8[e, :, :].T
//
// where:
//   - input_fp8:  [batch_size, K]            FP8 E4M3 activations
//   - weight_fp8: [num_experts, N, K]        FP8 E4M3 weights (row-major)
//   - output:     [batch_size, num_topk, N]  BF16 results
//   - Routing indices determine which tokens go to which experts
// ===================================================================
// OPERAND MAPPING (swapAB Convention)
// ===================================================================
// The UMMA hardware computes C[M,N] += A[M,K] * B[N,K] (both K-major).
// We map the operands as follows ("swapAB"):
//   UMMA operand A ← weight_fp8[e] shape [N_weight,K] (M dimension = N_weight)
//   UMMA operand B ← input_fp8     shape [B_tokens,K] (N dimension = B_tokens)
//   UMMA operand C ← accumulator   shape [N_weight,B_tokens]
// This means:
//   - MMA_M corresponds to the OUTPUT dimension (weight rows, N_weight)
//   - MMA_N corresponds to the BATCH dimension (tokens, B_tokens)
// ===================================================================
// EXPECTED SCALE FACTOR FORMAT
// ===================================================================
// This kernel expects FP8 E4M3 data with per-block float32 scale factors.
// The scale factors are converted to UE8M0 (8-bit exponent-only) in-kernel.
//
// ---- Weight scales (SFA) ----
//   Input tensor:  weight_scale [num_experts, N, K/128]
//   Dtype:       float32
//   Granularity:   Per-row, per-128-K-block
// ---- Input/activation scales (SFB) ----
//   Input tensor:  input_scale [batch, K/128]              (W13_LINEAR=true)
//                  input_scale [batch, num_topk, K/128]    (W13_LINEAR=false)
//   Dtype:       float32
//   Granularity:   Per-token, per-128-K-block
// ===================================================================
// WARP ROLE ASSIGNMENT (256 threads = 8 warps × 32 threads)
// ===================================================================
//   Warps 0-3 (threads 0-127):   EPILOGUE
//     - Copy FP32 accumulators from TMEM → registers via TMEM_LOAD
//     - Convert FP32 → BF16
//     - Scatter-write results to global memory using routing indices
//   Warp 4 (threads 128-159):    MMA (UTCCP + UMMA only)
//     - Wait for DMA to fill A/B smem tiles (a_full, b_full barriers)
//     - Wait for scale warp to finish transpose (sf_ready barrier)
//     - Issue UTCCP to copy transposed scales from smem → TMEM
//     - Issue 4× FP8 UMMA instructions per K-tile (bK/UMMA_K = 128/32 = 4)
//   Warp 5 (threads 160-191):    DMA (Data Movement)
//     - TMA load: FP8 weight tile [MMA_M × bK] from global → smem A
//     - cp.async load: FP8 input tile [MMA_N × bK] from global → smem B
//       with SWIZZLE_128B and per-token routing checks
//   Warp 6 (threads 192-223):    SCALE (float32 → UE8M0 → transpose)
//     - Load float32 weight/input scale factors from global memory
//     - Convert float32 → UE8M0, pack 4 copies into uint32
//     - Warp-transpose packed scales for UTCCP column-major layout
//   Warp 7 (threads 224-255):    IDLE
// ===================================================================
// 2D WORK DISTRIBUTION ACROSS CTAs
// ===================================================================
// With grid_dim=(X, Y, 1), work is distributed in 2 dimensions:
//
//   grid_dim.x (X CTAs): Expert distribution
//     - EXPERT_STRIDE = X (set to grid_dim.x by task_register)
//     - CTA i processes experts at indices i, i+X, i+2X, ...
//   grid_dim.y (Y CTAs): N-dimension (output row) splitting
//     - OUTPUT_SIZE = ORIG_OUTPUT_SIZE / Y (per-CTA slice)
// ===================================================================
// SMEM LAYOUT AND SWIZZLING
// ===================================================================
// Both A (weight) and B (input) tiles are stored in shared memory with
// SWIZZLE_128B layout (CuTe Swizzle<3,4,3> at byte level):
//
// A (weight): loaded via TMA which handles swizzling automatically.
// B (input):  loaded via manual cp.async with explicit swizzle computation
//             in the DMA warp (since we need per-token routing checks).
//
// Scale factor buffers use SWIZZLE_NONE layout (no swizzle needed for UTCCP).
// ===================================================================

// Load a 32-bit value from shared memory using its smem address.
// Used by fp8_utccp_warp_transpose to read scale factor entries.
__device__ __forceinline__ uint32_t fp8_ld_shared(uint32_t const *ptr) {
  uint32_t val;
  asm volatile("ld.shared.u32 %0, [%1];"
               : "=r"(val)
               : "r"(cute::cast_smem_ptr_to_uint(ptr)));
  return val;
}

// Store a 32-bit value to shared memory using its smem address.
// Used by fp8_utccp_warp_transpose to write transposed scale entries.
__device__ __forceinline__ void fp8_st_shared(uint32_t *ptr, uint32_t val) {
  asm volatile("st.shared.u32 [%0], %1;"
               :
               : "r"(cute::cast_smem_ptr_to_uint(ptr)), "r"(val));
}

// Issue a TCGEN05 fence that ensures all prior thread-level shared memory
// writes are visible to subsequent TCGEN05 (UMMA/UTCCP) operations.
// Must be called after writing scale factors to smem and before UTCCP.
__device__ __forceinline__ void fp8_tcgen05_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

// ===================================================================
// Shared Memory Storage Layout
// ===================================================================
//
// This struct defines the shared memory layout for the FP8 MoE GEMM kernel.
//
// IMPORTANT: We use raw uint8_t arrays for FP8 data tiles (A and B) rather
// than CuTe layout-wrapped tensors. This is because CuTe's smem_ptr_flag_bits
// mechanism asserts that the element size matches the swizzle atom size, and
// uint8_t (1 byte) elements with SWIZZLE_128B would trigger that assertion.
// Instead, we handle swizzling manually in the DMA warp (for B/input) and
// rely on TMA's built-in swizzling (for A/weight).
template <
    class BF16_ASmemLayout, // Placeholder type (used only for template
                            // deduction)
    class BF16_BSmemLayout, // Placeholder type (used only for template
                            // deduction)
    class BF16_BCpLayout, // Placeholder type (used only for template deduction)
    int Num_Experts,      // Number of experts (for expert_mask array sizing)
    int Num_AB_Stage,     // Number of pipeline stages for A/B smem buffers
    int Num_ACC_Stage,    // Number of pipeline stages for accumulator buffers
    int MMA_M_val,        // UMMA M dimension (weight/output rows per tile)
    int MMA_N_val,        // UMMA N dimension (tokens per tile)
    int bK_val>           // K-tile size (always 128 to match scale block size)
struct MoEFP8SharedStorage {
  // --- FP8 Data Tiles ---
  // A: Weight tile buffer. One tile = [MMA_M × bK] bytes of FP8 E4M3 data.
  //    Loaded by TMA (warp 5) with automatic SWIZZLE_128B layout.
  //    The UMMA descriptor for A points into this buffer.
  alignas(128) uint8_t A[Num_AB_Stage][MMA_M_val * bK_val];

  // B: Input/activation tile buffer. One tile = [MMA_N × bK] bytes of FP8 E4M3
  // data.
  //    Loaded by manual cp.async (warp 5) with explicit SWIZZLE_128B
  //    computation. The UMMA descriptor for B points into this buffer.
  alignas(128) uint8_t B[Num_AB_Stage][MMA_N_val * bK_val];

  // --- Scale Factor Buffers (for UTCCP → TMEM) ---
  // Each buffer holds 128 packed uint32_t entries. Each uint32 contains 4 UE8M0
  // bytes, one per 32-element K-subtile within the 128-K block:
  // Since our scale granularity is per-128-K (one scale per row per K-tile),
  // all 4 bytes in each uint32 are identical replicas of the same UE8M0 value.
  //
  // The 128 entries correspond to 128 rows (MMA_M rows for SFA, MMA_N rows for
  // SFB). After warp-transpose, these are in the column-major layout that UTCCP
  // expects.
  //
  // SFA: Weight scale factors (one per weight row in the current M-tile)
  alignas(128) uint32_t sfa_smem[Num_AB_Stage][128];
  // SFB: Input scale factors (one per token in the current N-tile)
  alignas(128) uint32_t sfb_smem[Num_AB_Stage][128];

  // --- Pipeline Barriers ---
  // These barriers synchronize the 3-stage producer-consumer pipeline:
  //   DMA warp (producer) → MMA warp (consumer/producer) → Epilogue warps
  //   (consumer)
  //
  // a_full:    Signaled by DMA warp when TMA weight load completes.
  //            Expected by MMA warp (1 transaction per signal).
  //            Uses ClusterTransactionBarrier (tracks TMA byte count).
  alignas(16) cute::uint64_t a_full_mbar_ptr[Num_AB_Stage];
  // b_full:    Signaled by DMA warp when cp.async input load completes.
  //            Expected by MMA warp (32 threads arrive via
  //            cpasync_barrier_arrive_noinc). Uses ClusterBarrier (thread-count
  //            based).
  alignas(16) cute::uint64_t b_full_mbar_ptr[Num_AB_Stage];
  // ab_empty:  Signaled by MMA warp (via umma_arrive) when it finishes reading
  // A/B.
  //            Expected by DMA warp before overwriting the smem stage.
  //            Uses ClusterBarrier (1 thread arrives via umma_arrive).
  alignas(16) cute::uint64_t ab_empty_mbar_ptr[Num_AB_Stage];
  // acc_full:  Signaled by MMA warp (via umma_arrive) when all K-tiles for one
  //            (expert, m_tile, n_tile) work unit are accumulated.
  //            Expected by epilogue warps before reading TMEM.
  alignas(16) cute::uint64_t acc_full_mbar_ptr[Num_ACC_Stage];
  // acc_empty: Signaled by epilogue warps when they finish reading the
  // accumulator.
  //            Expected by MMA warp before reusing the ACC buffer for the next
  //            tile. Uses ClusterBarrier (4 warps × 1 elected thread = 4
  //            arrivals).
  alignas(16) cute::uint64_t acc_empty_mbar_ptr[Num_ACC_Stage];
  // sf_ready:  Signaled by scale warp (warp 6) when scale factors have been
  // loaded,
  //            packed, and warp-transposed in smem. Expected by MMA warp (warp
  //            4) before issuing UTCCP to copy scales to TMEM. Uses
  //            ClusterBarrier (1 thread arrival from warp 6 via
  //            elect_one_sync).
  alignas(16) cute::uint64_t sf_ready_mbar_ptr[Num_AB_Stage];

  // --- Expert Mask and TMEM Base ---
  // expert_mask: cached copy of activated expert indices (unused in current
  // impl,
  //              experts are read directly from mMask global tensor).
  alignas(16) cute::uint32_t expert_mask[Num_Experts];
  // tmem_base_ptr: base column address in TMEM, allocated by warp 0 of the
  //                epilogue group and broadcast to all warps via this shared
  //                variable. All TMEM accesses (accumulators + scale columns)
  //                are offset from this.
  alignas(16) cute::uint32_t tmem_base_ptr;
};

// Advance the low 32 bits of a K-major UMMA descriptor to point at a
// different K-subtile within the same smem buffer.
//
// For K-major layout, the K dimension is contiguous, so advancing by
// k_idx elements adds (offset + k_idx) * sizeof(element) / 16 to the
// descriptor's low word (which encodes the start address).
//
// This is called 4 times per K-tile (k_sub = 0..3) to step through
// the 32-element UMMA sub-tiles within the 128-element K-tile.
__device__ __forceinline__ uint32_t fp8_advance_umma_desc_lo(uint32_t base_lo,
                                                             uint32_t offset,
                                                             uint32_t k_idx) {
  return base_lo + (((offset + k_idx * 1) * sizeof(uint8_t)) >> 4u);
}

// Build a shared memory descriptor for UTCCP (Unified Tensor Core CoPy)
// scale factor source.
//
// UTCCP copies UE8M0 scale factors from shared memory into TMEM (Tensor
// Memory), where the UMMA instruction reads them during block-scaled FP8
// computation. The descriptor uses:
//   - SWIZZLE_NONE: scale factors don't need swizzling
//   - SBO = 8 (in 16-byte units = 128 bytes): stride between 32-element groups
//
// This matches DeepGEMM's make_sf_desc function.
__device__ __forceinline__ cute::UMMA::SmemDescriptor
    fp8_make_sf_smem_desc(uint32_t const *smem_ptr) {
  cute::UMMA::SmemDescriptor desc;
  desc.desc_ = 0;
  desc.version_ = 1;
  desc.lbo_mode_ = 0;
  desc.layout_type_ = 0; // SWIZZLE_NONE
  const uint32_t uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
  desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
  desc.base_offset_ = 0;
  desc.stride_byte_offset_ = 8; // 8 * 16 = 128 bytes between 32-element groups
  desc.leading_byte_offset_ = 0;
  return desc;
}

// ===================================================================
// UTCCP Warp Transpose
// ===================================================================
//
// In-place warp-transpose of 128 uint32_t elements in shared memory.
// This is required because UTCCP (SM100_UTCCP_4x32dp128bit_1cta) expects
// scale factors in a specific column-major layout, but our scale loading
// code writes them in row-major order (one uint32 per row, sequentially).
//
// The transpose reorganizes 128 elements from:
//   Row-major: elements[0..127] stored linearly
// To:
//   Column-major (4×32 blocked): the 128 elements are viewed as a 4×32
//   matrix and transposed so that UTCCP can copy them into the correct
//   TMEM columns (4 columns × 32 entries each for the 4 K-subtiles).
//
// Algorithm (identical to DeepGEMM's utccp_required_smem_warp_transpose):
//   1. Each of 32 lanes reads 4 elements from non-contiguous positions
//      using an XOR-based index pattern: (i ^ (lane_idx >> 3)) * 32 + lane_idx
//   2. After a warp sync, each lane writes its 4 values to contiguous
//      positions: lane_idx * 4 + (i ^ (lane_idx >> 3))
//   This effectively performs a 4×32 → 32×4 transpose within the warp.
//
// Must be called by all 32 threads of the warp simultaneously.
__device__ __forceinline__ void fp8_utccp_warp_transpose(uint32_t *smem_ptr) {
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  uint32_t values[4];
// Phase 1: Read — each lane reads 4 non-contiguous elements
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    values[i] = fp8_ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
  }
  __syncwarp();
// Phase 2: Write — each lane writes 4 contiguous elements (transposed)
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    fp8_st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
  }
}

// ===================================================================
// MAIN KERNEL: FP8 Block-Scaled MoE Group GEMM (Per-Expert Task)
// ===================================================================
// The expert_offset parameter indexes into the activated experts list
// (stored in mMask). With EXPERT_STRIDE > 1, multiple CTAs can process
// different experts in parallel — CTA i handles experts at indices
// expert_offset, expert_offset + EXPERT_STRIDE, expert_offset +
// 2*EXPERT_STRIDE, ...
//
// For each assigned expert, the kernel iterates over all (m_tile, n_tile)
// output tiles, performing a full K-reduction for each tile via the
// DMA→MMA→Epilogue pipeline described in the file header.
//
// Template Parameters:
//   TMA_Weight:       TMA descriptor type for FP8 weight tensor
//                     (tma_2d<uint8_t, ...>).
//                     Encodes the global memory layout, swizzle mode, and tile
//                     shape for hardware-accelerated tensor memory access.
//   InputTensor:      CuTe tensor type for FP8 input activations.
//                     W13_LINEAR=true:  [batch_size, K] (flat input before MoE
//                     routing) W13_LINEAR=false: [batch_size, num_topk, K]
//                     (after first MoE layer) Element type is uint8_t* (FP8
//                     E4M3 stored as raw bytes to avoid CuTe type assertion
//                     issues with gmem_ptr<float_e4m3_t>).
//   InputScaleTensor: CuTe tensor type for float32 input scale factors.
//                     Shape [batch_size, K/128] or [batch_size, num_topk,
//                     K/128]. One scale per 128 consecutive K-elements per
//                     token.
//   WeightScaleTensor: CuTe tensor type for float32 weight scale factors.
//                      Shape [num_experts * N, K/128] (flattened expert
//                      dimension). One scale per 128 consecutive K-elements per
//                      weight row.
//   IndicesTensor:    CuTe tensor type for int32 routing indices.
//                     Shape [num_experts, batch_size].
//                     routing_indices[e, token] = topk_slot (1-indexed) if
//                     token is routed to expert e, else 0 (not routed).
//   MaskTensor:       CuTe tensor type for int32 expert activation mask.
//                     Shape [num_experts + 1].
//                     mask[0..N-1] = expert IDs of the N activated experts.
//                     mask[num_experts] = N (number of activated experts).
//   OutputTensor:     CuTe tensor type for BF16 output.
//                     Shape [batch_size, num_topk, N] with strides [topk*N, N,
//                     1]. output[token, topk_slot, output_row] stores the GEMM
//                     result.
//   MMA_M:            UMMA tile M dimension (weight/output rows).
//   MMA_N:            UMMA tile N dimension (tokens). Typically 128.
//   BATCH_SIZE:       Padded batch size (padded to MMA_N).
//   OUTPUT_SIZE:      Per-CTA output dimension (N / grid_dim.y). This is
//                     the number
//                     of weight rows each CTA processes. When grid_dim.y=1 (no
//                     N-split), OUTPUT_SIZE = ORIG_OUTPUT_SIZE. May not be a
//                     multiple of MMA_M.
//   ORIG_OUTPUT_SIZE: Full output dimension N before splitting across
//                     grid_dim.y.
//                     Used for expert stride in TMA coordinates and weight
//                     scale indexing: expert e's rows start at e *
//                     ORIG_OUTPUT_SIZE in the global weight tensor.
//   REDUCTION_SIZE:   K dimension (hidden size). Must be a multiple of bK=128.
//   NUM_EXPERTS:      Total number of experts in the model.
//   NUM_TOPK:         Number of experts each token is routed to (top-k).
//   EXPERT_STRIDE:    Stride for multi-CTA expert parallelism. Set to
//                     grid_dim.x
//                     by the task register so each CTA in the x-dimension
//                     handles a disjoint set of experts.
//   W13_LINEAR:       true for W1/W3 (gate/up) projection (input is flat
//                     [batch, K]),
//                     false for W2 (down) projection (input is [batch, topk,
//                     K]).
//   NUM_AB_STAGE:     Number of pipeline stages for A/B smem buffers.
//   NUM_ACC_STAGE:    Number of accumulator double-buffer stages in TMEM.
//   NUM_C_STAGE:      Number of output staging buffers.
//
template <typename TMA_Weight,
          class InputTensor,
          class InputScaleTensor,
          class WeightScaleTensor,
          class IndicesTensor,
          class MaskTensor,
          class OutputTensor,
          int MMA_M,
          int MMA_N,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int ORIG_OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int NUM_EXPERTS,
          int NUM_TOPK,
          int EXPERT_STRIDE,
          bool W13_LINEAR,
          int NUM_AB_STAGE = 4,
          int NUM_ACC_STAGE = 2,
          int NUM_C_STAGE = 4>
__device__ __forceinline__ void
    fp8_moe_group_gemm_sm100_task_impl(TMA_Weight const &tma_weight,
                                       InputTensor mInput,
                                       InputScaleTensor mInputScale,
                                       WeightScaleTensor mWeightScale,
                                       IndicesTensor mRoutingIndices,
                                       MaskTensor mMask,
                                       OutputTensor mOutput,
                                       int const expert_offset) {
  using namespace cute;

  using bf16_t = cute::bfloat16_t;
  using AccType = float;                  // FP32 accumulators in TMEM
  using ue8m0_t = cutlass::float_ue8m0_t; // 8-bit exponent-only scale type

  // ----------------------------------------------------------------
  // COMPILE-TIME CONSTANTS
  // ----------------------------------------------------------------

  // bK = 128: the K-tile size, chosen to match the FP8 block quantization
  // granularity (one scale per 128 K-elements). This means each K-tile
  // corresponds to exactly one scale factor per row.
  constexpr int bK = 128;

  // UMMA_K = 32: the SM100 FP8 UMMA instruction processes 32 K-elements
  // at a time. Each bK=128 K-tile requires 4 UMMA calls (128/32 = 4).
  // Each UMMA sub-tile selects one byte from the packed 4-byte UE8M0 scale
  // via the sfa_id/sfb_id field in the runtime instruction descriptor.
  constexpr int UMMA_K = 32;

  // ----------------------------------------------------------------
  // TMEM (Tensor Memory) COLUMN LAYOUT
  // ----------------------------------------------------------------
  // SM100 introduces TMEM, a 128-row x N-column on-chip memory dedicated
  // to the tensor core. The UMMA instruction reads/writes TMEM directly.
  //
  // Our TMEM allocation contains three regions:
  //   1. Accumulators: MMA_N columns x NUM_ACC_STAGE stages for
  //   double-buffering
  //      the FP32 accumulator matrix. Each accumulator holds one (m_tile,
  //      n_tile) result of shape [MMA_M=128 rows, MMA_N columns].
  //   2. SFA (Scale Factor A): 4 columns for weight UE8M0 scales.
  //      The 4 columns correspond to the 4 K-subtiles (k_sub=0..3).
  //   3. SFB (Scale Factor B): 4 columns for input UE8M0 scales.
  //      Same structure as SFA.
  //
  // Layout: [ACC_0 | ACC_1 | ... | SFA_0..3 | SFB_0..3]
  //         |--- num_tmem_acc_cols ---||-- 4 --||-- 4 --|
  //
  // Each UMMA tile requires Mx1 and 1xN scale factors from A and B,
  // respectively, so SFA=SFB=4 columns (one per k_sub).
  constexpr int num_tmem_acc_cols = MMA_N * NUM_ACC_STAGE; // e.g., 128*2 = 256
  constexpr int kNumSFATmemCols =
      4; // 4 columns for weight scales (one per k_sub)
  constexpr int kNumSFBTmemCols =
      4; // 4 columns for input scales (one per k_sub)
  constexpr int kTmemStartColOfSFA =
      num_tmem_acc_cols; // SFA starts right after accumulators
  constexpr int kTmemStartColOfSFB =
      num_tmem_acc_cols + kNumSFATmemCols; // SFB after SFA

  // TMEM allocation sizes must be powers of 2 (hardware constraint).
  // Round up to the nearest valid size: 64, 128, 256, or 512 columns.
  constexpr int kNumTmemColsRaw =
      num_tmem_acc_cols + kNumSFATmemCols + kNumSFBTmemCols;
  constexpr int kNumTmemColsTotal = kNumTmemColsRaw <= 64    ? 64
                                    : kNumTmemColsRaw <= 128 ? 128
                                    : kNumTmemColsRaw <= 256 ? 256
                                                             : 512;

  // ----------------------------------------------------------------
  // BF16 TiledMMA PROXY (for accumulator structure only)
  // ----------------------------------------------------------------
  // We use a BF16 TiledMMA object as a "proxy" to derive the correct TMEM
  // accumulator layout and fragment shapes. This works because:
  //   - FP8 and BF16 UMMAs with the same (MMA_M, MMA_N) produce identical
  //     FP32 accumulator layouts in TMEM (the accumulator format depends
  //     only on the output tile shape, not the input data type).
  //   - CuTe's FP8 MMA traits are not fully available in our CUTLASS version,
  //     so we piggyback on the well-tested BF16 infrastructure for the
  //     accumulator/epilogue data path.
  //   - The actual FP8 UMMA is issued via manual PTX (see MMA warp section).
  cute::TiledMMA tiled_mma_bf16 =
      cute::make_tiled_mma(cute::SM100_MMA_F16BF16_SS<bf16_t,
                                                      bf16_t,
                                                      AccType,
                                                      MMA_M,
                                                      MMA_N,
                                                      UMMA::Major::K,
                                                      UMMA::Major::K>{});

  // mma_coord_vmnk = (V=0, M=_, N=_, K=_): selects the first "value group"
  // (V=0) and leaves M, N, K as free tiling dimensions for partitioning.
  auto mma_coord_vmnk = cute::make_coord(0, cute::_, cute::_, cute::_);
  auto mma_v = cute::get<0>(mma_coord_vmnk);

  // ----------------------------------------------------------------
  // GLOBAL MEMORY TENSOR VIEWS
  // ----------------------------------------------------------------
  // Coordinate tensor for weight global memory layout.
  // Shape: [OUTPUT_SIZE, REDUCTION_SIZE, NUM_EXPERTS]
  //   - dim 0 = per-CTA N-slice rows (OUTPUT_SIZE, not ORIG_OUTPUT_SIZE)
  //   - dim 1 = reduction dimension (K)
  //   - dim 2 = expert dimension
  //
  // When grid_dim.y > 1, the N dimension is split across CTAs:
  //   OUTPUT_SIZE = ORIG_OUTPUT_SIZE / grid_dim.y  (per-CTA slice)
  //   ORIG_OUTPUT_SIZE = full N dimension (used for expert stride in TMA)
  //
  // The runtime offsets each CTA's weight base pointer to point at its
  // N-slice, so the kernel only sees OUTPUT_SIZE rows per expert.
  // Within this view, expert e's rows start at offset e * OUTPUT_SIZE.
  //
  // NOTE: This is a *coordinate* tensor (no actual data pointer) — it's
  // used only by CuTe's tiling infrastructure to compute TMA coordinates
  // for the DMA warp.
  cute::Tensor mA = cute::make_coord_tensor(cute::make_layout(
      cute::make_shape(OUTPUT_SIZE, REDUCTION_SIZE, NUM_EXPERTS),
      cute::make_stride(cute::E<1>{},
                        cute::E<0>{},
                        cute::E<1>{} * cute::Int<OUTPUT_SIZE>{})));

  // MMA tiler: defines the tile shape for each UMMA invocation.
  //   dim 0 = MMA_M (weight rows per tile, maps to UMMA M dimension)
  //   dim 1 = MMA_N (tokens per tile, maps to UMMA N dimension)
  //   dim 2 = bK    (reduction elements per tile)
  auto mma_tiler =
      cute::make_shape(cute::Int<MMA_M>{}, cute::Int<MMA_N>{}, cute::Int<bK>{});
  auto mma_coord = cute::select<1, 2, 3>(mma_coord_vmnk);
  auto cd_tiler =
      cute::make_shape(cute::Int<MMA_N>{}, cute::Int<MMA_M>{}, cute::Int<bK>{});

  // Tile the weight coordinate tensor into per-tile views.
  // gA(m_tile, k_tile) gives the coordinate pair for TMA address generation.
  // Step<_1, X, _1> selects M and K dimensions, skipping N (tokens aren't in
  // A).
  cute::Tensor gA = cute::local_tile(
      mA, mma_tiler, mma_coord, cute::Step<cute::_1, cute::X, cute::_1>{});
  // Note: gB (input tensor tiling) is NOT created here because input loading
  // is done with manual cp.async PTX in the DMA warp. The per-token routing
  // logic (checking topk_idx > 0) requires manual address computation that
  // CuTe's tiled copy infrastructure doesn't support.

  // ----------------------------------------------------------------
  // ACCUMULATOR SHAPE DERIVATION (via BF16 proxy)
  // ----------------------------------------------------------------
  // Use the BF16 TiledMMA to derive the accumulator partition shape.
  // partition_shape_A computes how the A operand tiles are distributed,
  // which indirectly determines the shared storage template parameter shape.
  // The bK/2 division accounts for BF16 elements being 2 bytes each —
  // the BF16 proxy sees half as many "elements" in K as FP8 would.
  auto mma_shape_A_acc = cute::partition_shape_A(
      tiled_mma_bf16,
      cute::make_shape(cute::Int<MMA_M>{},
                       cute::size<2>(mma_tiler) / 2, // bK/2: BF16 element count
                       cute::Int<NUM_AB_STAGE>{}));

  // ----------------------------------------------------------------
  // SHARED MEMORY SETUP
  // ----------------------------------------------------------------
  // Cast the dynamically-allocated shared memory (extern __shared__) to our
  // MoEFP8SharedStorage struct. Align to 128 bytes for TMA and UMMA
  // requirements. The first three template parameters are placeholders —
  // MoEFP8SharedStorage doesn't actually use them (it sizes A/B arrays from
  // MMA_M, MMA_N, bK directly).
  using SharedStorage =
      MoEFP8SharedStorage<decltype(mma_shape_A_acc), // placeholder
                          decltype(mma_shape_A_acc), // placeholder
                          decltype(mma_shape_A_acc), // placeholder
                          NUM_EXPERTS,
                          NUM_AB_STAGE,
                          NUM_ACC_STAGE,
                          MMA_M,
                          MMA_N,
                          bK>;
  extern __shared__ char shared_memory[];
  // Align to 1024 bytes: SM100 UMMA with SWIZZLE_128B requires the smem
  // buffer base to be aligned to at least 1024 bytes (the swizzle pattern
  // repeats at 8 rows * 128B = 1024B). Without this, the UMMA descriptor's
  // start_address interacts with absolute smem address bits, causing the
  // hardware to misinterpret the swizzled layout when the dynamic smem base
  // has non-zero bits in positions [7:9].
  uintptr_t aligned_smem =
      (reinterpret_cast<uintptr_t>(shared_memory) + 1023) / 1024 * 1024;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(aligned_smem);

  // Identify which warp this thread belongs to (0-7 for 256-thread block).
  // canonical_warp_idx_sync uses __syncwarp to ensure consistent results.
  int warp_idx = cutlass::canonical_warp_idx_sync();

  // ----------------------------------------------------------------
  // BARRIER INITIALIZATION (warp 0 only)
  // ----------------------------------------------------------------
  // Initialize all pipeline barriers before any warp can use them.
  // Each barrier type tracks a different synchronization event:
  //   a_full (TransactionBarrier, 1): TMA weight load complete (byte-count
  //   tracking) b_full (ClusterBarrier, 32):    cp.async input load complete
  //   (32 lane arrivals) ab_empty (ClusterBarrier, 1):   MMA done reading A/B
  //   (1 umma_arrive) acc_full (ClusterBarrier, 1):   MMA accumulated all
  //   K-tiles (1 umma_arrive) acc_empty (ClusterBarrier, 4):  Epilogue done
  //   reading accumulator (4 warp arrivals)
  if (warp_idx == 0) {
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        NUM_AB_STAGE>(shared_storage.a_full_mbar_ptr, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_AB_STAGE>(shared_storage.b_full_mbar_ptr, 32);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_AB_STAGE>(shared_storage.ab_empty_mbar_ptr, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_ACC_STAGE>(shared_storage.acc_full_mbar_ptr, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_ACC_STAGE>(shared_storage.acc_empty_mbar_ptr, 4);
    // sf_ready: scale warp (warp 6) signals 1 elected thread per stage
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_AB_STAGE>(shared_storage.sf_ready_mbar_ptr, 1);
  }

  // ----------------------------------------------------------------
  // NAMED BARRIERS (cross-warp synchronization)
  // ----------------------------------------------------------------
  // tmem_allocation_result_barrier: synchronizes TMEM allocation (done by
  // epilogue warp 0) with all consumers (MMA warp 4 + epilogue warps 0-3).
  // Thread count = 32 (allocator warp) + 128 (4 epilogue warps) = 160.
  // After this barrier fires, shared_storage.tmem_base_ptr is valid.
  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
      32 + 128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
  // epilogue_wg_barrier: synchronizes the 4 epilogue warps (128 threads)
  // to ensure all threads finish reading TMEM before signaling acc_empty.
  cutlass::arch::NamedBarrier epilogue_wg_barrier(
      128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

  // ----------------------------------------------------------------
  // CuTe MMA SLICING (for accumulator fragment structure)
  // ----------------------------------------------------------------
  // Get the per-V-group MMA view. Used only for:
  //   1. Partitioning the weight coordinate tensor (tCgA) — determines TMA
  //      tile coordinates (though we compute them manually in the DMA warp).
  //   2. Determining the TMEM accumulator fragment layout for the epilogue.
  cute::ThrMMA cta_mma = tiled_mma_bf16.get_slice(mma_v);
  cute::Tensor tCgA = cta_mma.partition_A(gA);

  // ----------------------------------------------------------------
  // TILE GEOMETRY
  // ----------------------------------------------------------------

  // TMA transaction bytes: number of bytes in one weight tile load.
  // The ClusterTransactionBarrier tracks this byte count to determine when
  // the TMA transfer is complete. For FP8: MMA_M * bK * 1 byte = 16384 bytes.
  int tma_transaction_bytes_A = MMA_M * bK * sizeof(uint8_t);

  // Tile counts derived from compile-time dimensions.
  // Computed directly from integer constants rather than from CuTe tensor
  // shapes to avoid confusion with the BF16 proxy (which sees different element
  // counts).
  //
  // fp8_num_m_tiles uses OUTPUT_SIZE (per-CTA slice), NOT ORIG_OUTPUT_SIZE.
  // When grid_dim.y > 1, each CTA only processes its own N-slice.
  // Ceiling division handles cases where OUTPUT_SIZE is not a multiple of MMA_M
  // (e.g., OUTPUT_SIZE=448 with MMA_M=128 → 4 tiles, last tile partially
  // valid).
  constexpr int fp8_k_tile_count = REDUCTION_SIZE / bK; // K-tiles per expert
  constexpr int fp8_num_m_tiles =
      (OUTPUT_SIZE + MMA_M - 1) / MMA_M; // output-dim tiles (per-CTA)
  constexpr int fp8_num_n_tiles =
      (BATCH_SIZE + MMA_N - 1) / MMA_N; // token-dim tiles (ceil)

  // ----------------------------------------------------------------
  // TMA AND SMEM INFRASTRUCTURE FOR WEIGHT (A) LOADING
  // ----------------------------------------------------------------
  // Constants for the TMA weight tile.
  //   TILE_SIZE = bK = 128: K-dimension of each tile (contiguous in memory)
  //   OUTPUT_ATOM_SIZE = MMA_M = 128: M-dimension of each tile (weight rows)
  //   B_PARAM, M_PARAM, S_PARAM = 3,3,3: CuTe Swizzle<3,3,3> parameters
  //   encoding
  //     SWIZZLE_128B mode. The swizzle XORs row*16 with the byte offset within
  //     each 128-byte line to eliminate shared memory bank conflicts.
  constexpr int TILE_SIZE = bK;
  constexpr int WEIGHT_TMA_TILE_SIZE = bK;
  constexpr int OUTPUT_ATOM_SIZE = MMA_M;
  constexpr int B_PARAM = 3; // log2(8 rows in XOR group)
  constexpr int M_PARAM = 3; // log2(8-byte shift amount)
  constexpr int S_PARAM = 3; // log2(8-byte XOR base stride)
  uint8_t *shared_weight = &shared_storage.A[0][0];

  // Barrier for TMA weight loads: ClusterTransactionBarrier tracks the byte
  // count of in-flight TMA transfers (set via set_barrier_transaction_bytes).
  using Barrier = cutlass::arch::ClusterTransactionBarrier;
  Barrier *a_full_mbar_ptr =
      reinterpret_cast<Barrier *>(shared_storage.a_full_mbar_ptr);

  // WeightSmem: CuTe smem_tma wrapper that encodes the shared memory tile
  // layout for TMA. The TMA hardware uses this to map global memory addresses
  // to swizzled shared memory addresses automatically.
  using WeightSmem = smem_tma<uint8_t,
                              B_PARAM,
                              M_PARAM,
                              S_PARAM,
                              OUTPUT_ATOM_SIZE,     // SMEM_ROW = MMA_M
                              WEIGHT_TMA_TILE_SIZE, // SMEM_COL = bK
                              1>;                   // repeat count = 1
  WeightSmem weight_smem(shared_weight);
  // Note: Input (B) loading does NOT use TMA — see DMA warp section below.

  // ----------------------------------------------------------------
  // ACCUMULATOR TMEM FRAGMENT
  // ----------------------------------------------------------------
  // Create a CuTe TMEM fragment handle for the FP32 accumulators.
  // This does NOT allocate TMEM; it just computes the fragment shape/layout.
  // The actual TMEM allocation happens later via tmem_allocator.allocate()
  // in the epilogue warp. The base column address is stored in
  // shared_storage.tmem_base_ptr and assigned to tCtAcc.data() after
  // allocation.
  auto acc_shape = cute::partition_shape_C(
      tiled_mma_bf16,
      cute::make_shape(cute::size<0>(mma_tiler),     // MMA_M
                       cute::size<1>(mma_tiler),     // MMA_N
                       cute::Int<NUM_ACC_STAGE>{})); // double-buffering
  auto tCtAcc = tiled_mma_bf16.make_fragment_C(acc_shape);

  // ----------------------------------------------------------------
  // SYNCHRONIZE BEFORE WARP-DIVERGENT CODE
  // ----------------------------------------------------------------
  // fence_barrier_init ensures all barrier initializations by warp 0 are
  // visible to all other warps. __syncthreads provides a full CTA barrier.
  cutlass::arch::fence_barrier_init();

  // Prefetch TMA descriptor into L1 cache to hide first-access latency.
  // Only one thread needs to issue this (warp 0, lane 0).
  if (warp_idx == 0 && cutlass::canonical_lane_idx() == 0) {
    cute::prefetch_tma_descriptor(tma_weight.desc_ptr);
  }

  __syncthreads();

  // Runtime loop bounds used by all warps.
  int k_tile_count = fp8_k_tile_count; // K-tiles per expert
  int num_activated_experts =
      mMask(NUM_EXPERTS); // read from last element of mask

  // TMEM allocator for single-SM mode (no multi-SM cluster in this kernel).
  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  __syncthreads();

  // ================================================================
  // WARP 5 -- DMA WARP: TMA Weight Load + CP_ASYNC Input Load
  // ================================================================
  //
  // This warp is responsible for all data movement from global memory to
  // shared memory. It runs concurrently with the MMA warp (warp 4) and
  // epilogue warps (warps 0-3) via the barrier-based pipeline.
  //
  // For each (expert, m_tile, n_tile) work unit, it loads k_tile_count
  // K-tiles of both weight (A) and input (B) data into rotating smem stages.
  //
  // Weight loading uses TMA (Tensor Memory Access):
  //   - Hardware-accelerated bulk copy with automatic swizzling
  //   - Only one lane (elect_one_sync) issues the TMA instruction
  //   - Completion tracked by ClusterTransactionBarrier (byte count)
  //
  // Input loading uses manual cp.async:
  //   - All 32 lanes cooperate to copy the [MMA_N x bK] tile
  //   - Each lane handles (MMA_N * bK / 32) bytes with 8-byte cp.async ops
  //   - Explicit SWIZZLE_128B address computation per element
  //   - Per-token routing check: only load if token is routed to this expert
  //     (topk_idx > 0), otherwise write zeros
  //   - Completion signaled by cpasync_barrier_arrive_noinc on b_full barrier
  //
  if (warp_idx == 5) {
    const uint32_t lane_idx = cutlass::canonical_lane_idx();

    // total_k_tile_count tracks the cumulative K-tile index across ALL
    // work units, used to compute the rotating smem buffer index and
    // barrier phase. This ensures correct pipeline stage assignment even
    // when processing multiple (expert, m_tile, n_tile) work units.
    int total_k_tile_count = 0;

    // Outer loop: iterate over activated experts assigned to this CTA.
    // EXPERT_STRIDE > 1 allows multiple CTAs to process different experts.
    for (int activated_expert_offset = expert_offset;
         activated_expert_offset < num_activated_experts;
         activated_expert_offset += EXPERT_STRIDE) {
      int32_t expert_idx = mMask[activated_expert_offset];
      // tRoutingIndex(token_idx) returns the top-k slot (1-indexed) if token
      // is routed to this expert, or 0 if not routed.
      cute::Tensor tRoutingIndex = mRoutingIndices(expert_idx, cute::_);
      for (int m_tile = 0; m_tile < fp8_num_m_tiles; ++m_tile) {
        for (int n_tile = 0; n_tile < fp8_num_n_tiles; ++n_tile) {
          int num_prev_k_blk = total_k_tile_count;
          total_k_tile_count += k_tile_count;

          // Pipeline stage management: compute which smem buffer to write into
          // and what barrier phase to wait on. The phase alternates every
          // NUM_AB_STAGE K-tiles (XOR with 1 to get initial DMA phase).
          int tma_wr_k_tile = 0;
          int smem_wr_buffer = (num_prev_k_blk + tma_wr_k_tile) % NUM_AB_STAGE;
          int tma_wr_ab_empty_phase =
              (num_prev_k_blk + tma_wr_k_tile) / NUM_AB_STAGE % 2 ^ 1;

          // Optimistic peek: try to check if the MMA warp has already consumed
          // this buffer (ab_empty). If yes, we can skip the blocking wait
          // below.
          bool peek_ab_empty_status =
              ::kernel::try_wait_barrier(shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                               tma_wr_ab_empty_phase);

          // Inner loop: iterate over K-tiles for this (expert, m_tile, n_tile)
          for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
            // Pre-compute next iteration's pipeline state for lookahead peeks
            int tma_wr_k_tile_next = tma_wr_k_tile + 1;
            int smem_wr_buffer_next =
                (num_prev_k_blk + tma_wr_k_tile_next) % NUM_AB_STAGE;
            int tma_wr_ab_empty_phase_next = smem_wr_buffer_next == 0
                                                 ? tma_wr_ab_empty_phase ^ 1
                                                 : tma_wr_ab_empty_phase;

            // Wait for MMA warp to finish reading this smem buffer (if peek
            // failed)
            if (!peek_ab_empty_status) {
              cute::wait_barrier(
                  shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                  tma_wr_ab_empty_phase);
            }

            // ---- TMA WEIGHT LOAD (operand A) ----
            // Issue a TMA (Tensor Memory Access) instruction to copy one
            // weight tile [MMA_M x bK] = [128 x 128] bytes from global to smem.
            //
            // TMA is a hardware unit that performs bulk tensor copies with:
            //   - Automatic address generation from (row, col) coordinates
            //   - Built-in swizzling (SWIZZLE_128B) for bank conflict avoidance
            //   - Async completion tracking via ClusterTransactionBarrier
            //
            // Only ONE lane issues the TMA instruction (elect_one_sync).
            // TMA coordinates: [K_offset, M_offset_in_global_weight_tensor]
            //   K_offset = k_tile * bK (which K-tile along the reduction dim)
            //   M_offset = expert_idx * ORIG_OUTPUT_SIZE + m_tile * MMA_M
            //              (which weight rows, accounting for expert offset)
            if (cute::elect_one_sync()) {
              int tma_coords_A[2] = {k_tile * TILE_SIZE,
                                     m_tile * OUTPUT_ATOM_SIZE +
                                         expert_idx * ORIG_OUTPUT_SIZE};
              // Point smem destination to the correct pipeline stage buffer
              weight_smem.set_ptr(shared_weight + smem_wr_buffer *
                                                      OUTPUT_ATOM_SIZE *
                                                      TILE_SIZE);
              // Tell the transaction barrier how many bytes to expect
              cute::set_barrier_transaction_bytes(
                  shared_storage.a_full_mbar_ptr[smem_wr_buffer],
                  tma_transaction_bytes_A);
              // Fire the async TMA copy (hardware handles the rest)
              tma_weight.tma_cp_async(a_full_mbar_ptr[smem_wr_buffer],
                                      weight_smem.base_ptr,
                                      tma_coords_A);
            }

            // ---- MANUAL CP.ASYNC INPUT LOAD (operand B) ----
            // Copy one input tile [MMA_N x bK] from global to smem B.
            // Unlike weight loading (TMA), input loading uses manual cp.async
            // because each token's data may or may not be needed depending on
            // the MoE routing decision (topk_idx > 0).
            //
            // All 32 lanes of the DMA warp cooperate:
            //   Total tile bytes = MMA_N * bK * sizeof(uint8_t) = 128*128 =
            //   16384 bytes Bytes per thread = 16384 / 32 = 512 bytes Copies
            //   per thread = 512 / 8 = 64 (each cp.async copies 8 bytes)
            //
            // Each lane computes which (row, col) of the input tile its bytes
            // correspond to, checks the routing, and either issues cp.async to
            // fetch from global memory or writes zeros for unrouted tokens.
            //
            // The destination smem addresses are swizzled with SWIZZLE_128B to
            // match the layout expected by the UMMA B descriptor:
            //   swizzled_offset = linear_offset ^ (((linear_offset >> 7) & 7)
            //   << 4)
            // This XOR formula maps byte offset L to L ^ ((row_in_128B_group %
            // 8) * 16).
            {
              uint8_t *dst_base = &shared_storage.B[smem_wr_buffer][0];
              constexpr int BYTES_PER_THREAD = (MMA_N * bK) / 32;
              // 16 bytes per cp.async instruction (uint4 = 16 bytes).
              // SWIZZLE_128B with bK=128: the swizzle XOR is (row%8)*16,
              // so 16-byte aligned addresses stay 16-byte aligned after
              // swizzle.
              constexpr int CP_BYTES = 16;
              constexpr int COPIES_PER_THREAD = BYTES_PER_THREAD / CP_BYTES;
              int base_byte = lane_idx * BYTES_PER_THREAD;
#pragma unroll
              for (int c = 0; c < COPIES_PER_THREAD; ++c) {
                int byte_off = base_byte + c * CP_BYTES;
                int row = byte_off / bK;
                int col = byte_off % bK;
                int32_t token_idx_n = n_tile * MMA_N + row;
                // Guard routing index read: MMA tile may exceed BATCH_SIZE
                int32_t topk_idx_n = (token_idx_n < BATCH_SIZE)
                                         ? tRoutingIndex(token_idx_n)
                                         : 0;

                // SWIZZLE_128B: swizzled = linear ^ ((row%8)*16)
                int linear_off = row * bK + col;
                int swizzled_off = linear_off ^ (((linear_off >> 7) & 7) << 4);
                uint32_t dst_smem =
                    cute::cast_smem_ptr_to_uint(dst_base + swizzled_off);

                if (token_idx_n < BATCH_SIZE && topk_idx_n > 0) {
                  uint8_t const *src_row;
                  if constexpr (W13_LINEAR) {
                    src_row = reinterpret_cast<uint8_t const *>(
                        &mInput(token_idx_n, k_tile * bK + col));
                  } else {
                    src_row = reinterpret_cast<uint8_t const *>(&mInput(
                        token_idx_n, topk_idx_n - 1, k_tile * bK + col));
                  }
                  // cp.async.ca with 16-byte transfer (halves instruction
                  // count)
                  asm volatile(
                      "cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(
                          dst_smem),
                      "l"(src_row));
                } else {
                  // Zero-fill 16 bytes for unrouted tokens
                  *reinterpret_cast<uint4 *>(dst_base + swizzled_off) =
                      make_uint4(0, 0, 0, 0);
                }
              }
              // Commit all outstanding cp.async instructions as a group.
              // The barrier tracks completion of this group.
              asm volatile("cp.async.commit_group;\n" ::: "memory");
            }
            // Signal that the B (input) tile for this stage is ready in smem.
            // cpasync_barrier_arrive_noinc: each of 32 lanes arrives at the
            // barrier after its cp.async group is committed.
            cutlass::arch::cpasync_barrier_arrive_noinc(
                &shared_storage.b_full_mbar_ptr[smem_wr_buffer]);

            // Lookahead peek: try to check if the NEXT smem buffer is already
            // available (MMA warp consumed it). This overlaps barrier checking
            // with the current K-tile's TMA/cp.async execution.
            if (tma_wr_k_tile_next < k_tile_count) {
              peek_ab_empty_status = ::kernel::try_wait_barrier(
                  shared_storage.ab_empty_mbar_ptr[smem_wr_buffer_next],
                  tma_wr_ab_empty_phase_next);
            }
            // Advance pipeline state to the next K-tile
            tma_wr_k_tile = tma_wr_k_tile_next;
            smem_wr_buffer = smem_wr_buffer_next;
            tma_wr_ab_empty_phase = tma_wr_ab_empty_phase_next;
          } // end k_tile loop
        }   // end n_tile loop
      }     // end m_tile loop
    }       // end expert loop
  }         // end warp 5 (DMA warp)
  // ================================================================
  // WARP 4 -- MMA WARP: Scale Loading + UTCCP + FP8 Block-Scaled UMMA
  // ================================================================
  //
  // This warp performs three tasks per K-tile:
  //
  //   1. SCALE LOADING: Read float32 scale factors from global memory,
  //      convert to UE8M0, pack into uint32 (4 bytes per row), write
  //      to smem scale buffers, and apply warp-transpose for UTCCP layout.
  //
  //   2. UTCCP (Unified Tensor Core CoPy): Copy the packed UE8M0 scales
  //      from shared memory into TMEM scale columns. The UMMA instruction
  //      reads scales directly from TMEM during block-scaled FP8 computation.
  //      Only one lane (elect_one_sync) issues each UTCCP instruction.
  //
  //   3. UMMA ISSUANCE: Issue 4 FP8 UMMA instructions per K-tile (one
  //      per 32-element K-subtile). Each UMMA reads A/B data from smem
  //      (via descriptors) and scales from TMEM, accumulating into the
  //      FP32 accumulator in TMEM.
  //
  // After all K-tiles for one (expert, m_tile, n_tile) work unit are
  // accumulated, this warp signals acc_full so the epilogue can read results.
  //
  // IMPORTANT: The UMMA fma() call internally uses elect_one_sync() because
  // only one lane should issue the tcgen05.mma instruction. Do NOT wrap the
  // fma() call in an outer elect_one_sync() — that would cause a deadlock
  // (inner elect_one_sync needs all 32 lanes, but outer already filtered to 1).
  //
  else if (warp_idx == 4) {
    // Wait for TMEM allocation (done by epilogue warp 0) to complete.
    // After this barrier, shared_storage.tmem_base_ptr is valid.
    tmem_allocation_result_barrier.arrive_and_wait();
    tCtAcc.data() = shared_storage.tmem_base_ptr;

    // Compute absolute TMEM column addresses for the three regions.
    // These are column indices within the TMEM allocation, used by UMMA
    // (accumulator destination) and UTCCP (scale factor destination).
    const uint32_t tmem_base = shared_storage.tmem_base_ptr;
    const uint32_t sfa_tmem = tmem_base + kTmemStartColOfSFA; // weight scales
    const uint32_t sfb_tmem = tmem_base + kTmemStartColOfSFB; // input scales

    // UTCCP type: copies 4 columns x 32 rows x 128 bits from smem to TMEM.
    // "4x32dp128bit_1cta" = 4 columns, 32 depth, 128-bit data path, 1 CTA.
    using UTCCP_t = cute::SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta;

    // ---- BUILD INSTRUCTION DESCRIPTOR ----
    // The instruction descriptor encodes the data types, tile shape, and
    // major-order for the UMMA instruction. It's a compile-time constant
    // that tells the hardware how to interpret the operand descriptors.
    //   A type: float_e4m3_t (FP8 E4M3), B type: float_e4m3_t
    //   C type: float (FP32 accumulator)
    //   Scale type: float_ue8m0_t (8-bit exponent-only)
    //   Both A and B are K-major (reduction dimension is contiguous)
    auto instr_desc =
        cute::UMMA::make_instr_desc_block_scaled<cutlass::float_e4m3_t,
                                                 cutlass::float_e4m3_t,
                                                 float,
                                                 cutlass::float_ue8m0_t,
                                                 MMA_M,
                                                 MMA_N,
                                                 cute::UMMA::Major::K,
                                                 cute::UMMA::Major::K>();

    // ---- BUILD SCALE FACTOR DESCRIPTOR ----
    // UTCCP needs a smem descriptor for the source of scale factor data.
    // This uses SWIZZLE_NONE layout (scales don't need swizzling).
    // The address field is updated per-stage before each UTCCP call
    // via replace_smem_desc_addr().
    auto sf_desc = kernel::sm100::make_sf_desc(nullptr);

    // ---- BUILD UMMA OPERAND DESCRIPTORS ----
    // These smem descriptors tell the UMMA instruction where to find operand
    // data in shared memory. They encode:
    //   - Base smem address (16-byte aligned)
    //   - Stride Byte Offset (SBO): distance between 32-row groups
    //   - Layout type: SWIZZLE_128B (matching the swizzled smem layout)
    //   - Major order: K (K dimension is contiguous)
    //
    // A descriptor: points to weight data in shared_storage.A[0]
    auto a_desc =
        kernel::sm100::make_umma_desc<cute::UMMA::Major::K, MMA_M, bK, 128>(
            reinterpret_cast<cutlass::float_e4m3_t *>(&shared_storage.A[0][0]),
            0,
            0);
    // B descriptor: points to input data in shared_storage.B[0]
    auto b_desc =
        kernel::sm100::make_umma_desc<cute::UMMA::Major::K, MMA_N, bK, 128>(
            reinterpret_cast<cutlass::float_e4m3_t *>(&shared_storage.B[0][0]),
            0,
            0);

    // ---- PRE-COMPUTE PER-STAGE DESCRIPTOR BASE ADDRESSES ----
    // The UMMA descriptor's lo-word contains the smem base address (in 16-byte
    // units). For different pipeline stages, the base address shifts by the
    // stage's buffer offset. Instead of recomputing the full descriptor per
    // stage, we pre-compute the lo-word for each stage and store it in
    // different lanes of the warp. At execution time, __shfl_sync broadcasts
    // the correct stage's lo-word to all lanes.
    //
    // Lane i holds the lo-word for stage i (i < NUM_AB_STAGE).
    // Stage offset = i * (MMA_M * bK * sizeof(uint8_t)) / 16  (in 16-byte
    // units)
    const uint32_t lane_idx = cutlass::canonical_lane_idx();
    uint32_t a_desc_lo =
        (lane_idx < (uint32_t)NUM_AB_STAGE)
            ? (a_desc.lo + lane_idx * (MMA_M * bK * sizeof(uint8_t)) / 16)
            : 0u;
    uint32_t b_desc_lo =
        (lane_idx < (uint32_t)NUM_AB_STAGE)
            ? (b_desc.lo + lane_idx * (MMA_N * bK * sizeof(uint8_t)) / 16)
            : 0u;

    int total_k_tile_count =
        0; // cumulative K-tile counter (for pipeline stage)
    int num_tiles_executed = 0; // cumulative work-unit counter (for ACC stage)

    // Same triple-nested loop structure as the DMA warp: expert x m_tile x
    // n_tile. Both warps must iterate in the same order so their pipeline stage
    // indices stay synchronized (DMA fills stage S, MMA reads stage S).
    for (int activated_expert_offset = expert_offset;
         activated_expert_offset < num_activated_experts;
         activated_expert_offset += EXPERT_STRIDE) {
      int32_t expert_idx = mMask[activated_expert_offset];
      cute::Tensor tRoutingIndex = mRoutingIndices(expert_idx, cute::_);

      for (int m_tile = 0; m_tile < fp8_num_m_tiles; ++m_tile) {
        for (int n_tile = 0; n_tile < fp8_num_n_tiles; ++n_tile) {
          // Which accumulator buffer to use for this work unit (rotating)
          int acc_buf_idx = num_tiles_executed % NUM_ACC_STAGE;

          // Pipeline stage tracking (must match DMA warp's numbering)
          int num_prev_k_blk = total_k_tile_count;
          total_k_tile_count += k_tile_count;

          int mma_rd_k_tile = 0;
          int smem_rd_buf = (num_prev_k_blk + mma_rd_k_tile) % NUM_AB_STAGE;
          int mma_rd_ab_full_phase =
              (num_prev_k_blk + mma_rd_k_tile) / NUM_AB_STAGE % 2;

          // Optimistic peeks: check if A/B/scales are already ready
          bool peek_a =
              ::kernel::try_wait_barrier(shared_storage.a_full_mbar_ptr[smem_rd_buf],
                               mma_rd_ab_full_phase);
          bool peek_b =
              ::kernel::try_wait_barrier(shared_storage.b_full_mbar_ptr[smem_rd_buf],
                               mma_rd_ab_full_phase);
          int sf_phase = (num_prev_k_blk) / NUM_AB_STAGE % 2;
          bool peek_sf = ::kernel::try_wait_barrier(
              shared_storage.sf_ready_mbar_ptr[smem_rd_buf], sf_phase);

          // Wait for epilogue to finish reading the previous accumulator in
          // this buffer slot before we overwrite it with new UMMA results.
          int acc_empty_phase = num_tiles_executed / NUM_ACC_STAGE % 2 ^ 1;
          cute::wait_barrier(shared_storage.acc_empty_mbar_ptr[acc_buf_idx],
                             acc_empty_phase);

          // first_tile: controls whether the UMMA accumulates (1) or overwrites
          // (0). The first K-tile's first sub-tile clears the accumulator; all
          // subsequent sub-tiles and K-tiles accumulate into it.
          bool first_tile = true;

          // ---- K-TILE LOOP: Accumulate all K-tiles for this work unit ----
          for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
            // Pre-compute next K-tile's pipeline state for lookahead peeks
            int mma_rd_k_tile_next = mma_rd_k_tile + 1;
            int smem_rd_buf_next =
                (num_prev_k_blk + mma_rd_k_tile_next) % NUM_AB_STAGE;
            int mma_rd_ab_full_phase_next = smem_rd_buf_next == 0
                                                ? mma_rd_ab_full_phase ^ 1
                                                : mma_rd_ab_full_phase;

            // Wait for DMA warp to finish loading this K-tile's A/B data into
            // smem
            if (!peek_a) {
              cute::wait_barrier(shared_storage.a_full_mbar_ptr[smem_rd_buf],
                                 mma_rd_ab_full_phase);
            }
            if (!peek_b) {
              cute::wait_barrier(shared_storage.b_full_mbar_ptr[smem_rd_buf],
                                 mma_rd_ab_full_phase);
            }

            // Wait for scale warp (warp 6) to finish transpose (if peek failed)
            if (!peek_sf) {
              cute::wait_barrier(shared_storage.sf_ready_mbar_ptr[smem_rd_buf],
                                 sf_phase);
            }

            // ============================================================
            // UTCCP: Copy transposed scales from smem -> TMEM
            // ============================================================
            // Issue UTCCP instructions to copy scale factors from smem into
            // TMEM. Only one lane issues each UTCCP (hardware requirement for
            // tcgen05 instructions — elect_one_sync selects lane 0).
            //
            // Each UTCCP copies 128 uint32 values (= 4 columns x 32 rows x 128
            // bits) from the smem scale buffer into the TMEM scale columns. SFA
            // goes to columns [kTmemStartColOfSFA .. +4) SFB goes to columns
            // [kTmemStartColOfSFB .. +4)
            if (cute::elect_one_sync()) {
              // Update sf_desc's smem address to point at this stage's SFA
              // buffer
              kernel::sm100::replace_smem_desc_addr(
                  sf_desc, shared_storage.sfa_smem[smem_rd_buf]);
              UTCCP_t::copy(static_cast<uint64_t>(sf_desc), sfa_tmem);
              // Same for SFB
              kernel::sm100::replace_smem_desc_addr(
                  sf_desc, shared_storage.sfb_smem[smem_rd_buf]);
              UTCCP_t::copy(static_cast<uint64_t>(sf_desc), sfb_tmem);
            }
            __syncwarp(); // ensure UTCCP is issued before proceeding

            // ============================================================
            // STEP 3: ISSUE FP8 UMMA INSTRUCTIONS
            // ============================================================
            // Each bK=128 K-tile is processed as 4 UMMA sub-tiles of 32
            // K-elements. The UMMA instruction:
            // tcgen05.mma.kind::mxf8f6f4.block_scale
            //
            // First, broadcast the correct per-stage smem descriptor base
            // address from the lane that pre-computed it (lane index =
            // smem_rd_buf).
            // __shfl_sync(0xffffffff, value, src_lane) broadcasts src_lane's
            // value to all 32 lanes. This avoids recomputing the descriptor for
            // each stage.
            auto const a_desc_base_lo =
                __shfl_sync(0xffffffff, a_desc_lo, smem_rd_buf);
            auto const b_desc_base_lo =
                __shfl_sync(0xffffffff, b_desc_lo, smem_rd_buf);

            // Issue UMMA sub-tile instructions (bK/UMMA_K = 128/32 = 4).
            // WARNING: Do NOT add an outer elect_one_sync around this block —
            // the fma() call contains its own elect_one_sync internally, and
            // nesting them causes a deadlock.
            if (cute::elect_one_sync()) {
#pragma unroll
              for (int k_sub = 0; k_sub < bK / UMMA_K; ++k_sub) {
                // sfa_id/sfb_id select which byte of the packed uint32 scale to
                // use. k_sub=0 -> byte 0 (bits 0-7), k_sub=1 -> byte 1 (bits
                // 8-15), etc. Since all 4 bytes are identical (same scale
                // replicated), the choice doesn't matter for per-128-K
                // quantization — but it would matter if finer-grained
                // (per-32-K) quantization were used in the future.
                const uint32_t sfa_id = k_sub;
                const uint32_t sfb_id = k_sub;
                // Build the runtime instruction descriptor with this sub-tile's
                // scale factor byte selector. This modifies the base instr_desc
                // to include sfa_id and sfb_id fields.
                auto runtime_instr_desc =
                    kernel::sm100::make_runtime_instr_desc_with_sf_id(
                        instr_desc, sfa_id, sfb_id);

                // Advance the UMMA smem descriptors to point at this K-subtile.
                // For K-major layout, advancing by k_sub*UMMA_K elements moves
                // the start address by (k_sub * 32 * sizeof(fp8)) / 16 in the
                // descriptor's 16-byte-granularity address field.
                b_desc.lo =
                    kernel::sm100::advance_umma_desc_lo<cute::UMMA::Major::K,
                                                        MMA_N,
                                                        128,
                                                        cutlass::float_e4m3_t>(
                        b_desc_base_lo, 0, k_sub * UMMA_K);
                a_desc.lo =
                    kernel::sm100::advance_umma_desc_lo<cute::UMMA::Major::K,
                                                        MMA_M,
                                                        128,
                                                        cutlass::float_e4m3_t>(
                        a_desc_base_lo, 0, k_sub * UMMA_K);

                // Issue the FP8 block-scaled UMMA instruction:
                //   C[tmem_col] += A[smem] * B[smem] * scale_a[tmem] *
                //   scale_b[tmem]
                //
                // Arguments:
                //   a_desc/b_desc: smem descriptors pointing to FP8 operand
                //   sub-tiles tmem_base + acc_buf_idx * MMA_N: accumulator
                //   column in TMEM accumulate flag: 0 = overwrite (first
                //   sub-tile of first K-tile),
                //                    1 = accumulate (all subsequent sub-tiles)
                //   runtime_instr_desc: instruction descriptor with
                //   sfa_id/sfb_id sfa_tmem/sfb_tmem: TMEM column addresses for
                //   scale factors
                kernel::sm100::SM100_MMA_MXF8F6F4_SS::fma(
                    a_desc,
                    b_desc,
                    tmem_base + acc_buf_idx * MMA_N,
                    (first_tile && k_sub == 0) ? 0u : 1u,
                    runtime_instr_desc,
                    sfa_tmem,
                    sfb_tmem);
              }
            }
            first_tile = false;

            // Signal that we're done reading A/B from this smem stage.
            // umma_arrive is a special barrier arrival that waits for all
            // outstanding UMMA instructions to complete before arriving.
            // This ensures the DMA warp won't overwrite smem data that UMMA
            // is still reading asynchronously.
            cutlass::arch::umma_arrive(
                &shared_storage.ab_empty_mbar_ptr[smem_rd_buf]);

            // Lookahead peek for the next K-tile's A/B/scale data
            if (mma_rd_k_tile_next < k_tile_count) {
              peek_a = ::kernel::try_wait_barrier(
                  shared_storage.a_full_mbar_ptr[smem_rd_buf_next],
                  mma_rd_ab_full_phase_next);
              peek_b = ::kernel::try_wait_barrier(
                  shared_storage.b_full_mbar_ptr[smem_rd_buf_next],
                  mma_rd_ab_full_phase_next);
              int sf_phase_next =
                  (num_prev_k_blk + mma_rd_k_tile_next) / NUM_AB_STAGE % 2;
              peek_sf = ::kernel::try_wait_barrier(
                  shared_storage.sf_ready_mbar_ptr[smem_rd_buf_next],
                  sf_phase_next);
              sf_phase = sf_phase_next;
            }

            // Advance pipeline state to the next K-tile
            mma_rd_k_tile = mma_rd_k_tile_next;
            smem_rd_buf = smem_rd_buf_next;
            mma_rd_ab_full_phase = mma_rd_ab_full_phase_next;
          } // end k_tile loop

          // All K-tiles for this (expert, m_tile, n_tile) have been
          // accumulated. Signal the epilogue that the accumulator is ready to
          // read. umma_arrive ensures all UMMA writes to the accumulator TMEM
          // columns are complete before the epilogue reads them.
          cutlass::arch::umma_arrive(
              &shared_storage.acc_full_mbar_ptr[acc_buf_idx]);
          num_tiles_executed++;
        } // end n_tile loop
      }   // end m_tile loop
    }     // end expert loop
  }       // end warp 4 (MMA warp)
  // ================================================================
  // WARP 6 -- SCALE WARP: Load float32 scales, convert to UE8M0, transpose
  // ================================================================
  //
  // This warp runs in parallel with the DMA warp (warp 5) and MMA warp (warp
  // 4). For each K-tile, it:
  //   1. Loads float32 scales from global memory (weight and input)
  //   2. Converts to UE8M0 (exponent extraction)
  //   3. Packs 4 copies into uint32 (one per UMMA sub-tile)
  //   4. Warp-transposes for UTCCP column-major layout
  //   5. Signals sf_ready barrier so MMA warp can issue UTCCP
  //
  // This removes scale work from the MMA warp's critical path, overlapping
  // it with DMA loads and prior UMMA execution.
  //
  else if (warp_idx == 6) {
    const uint32_t lane_idx = cutlass::canonical_lane_idx();

    int total_k_tile_count = 0;
    for (int activated_expert_offset = expert_offset;
         activated_expert_offset < num_activated_experts;
         activated_expert_offset += EXPERT_STRIDE) {
      int32_t expert_idx = mMask[activated_expert_offset];
      cute::Tensor tRoutingIndex = mRoutingIndices(expert_idx, cute::_);

      for (int m_tile = 0; m_tile < fp8_num_m_tiles; ++m_tile) {
        for (int n_tile = 0; n_tile < fp8_num_n_tiles; ++n_tile) {
          int num_prev_k_blk = total_k_tile_count;
          total_k_tile_count += k_tile_count;

          for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
            int smem_buf = (num_prev_k_blk + k_tile) % NUM_AB_STAGE;

            // ---- Load + convert + pack SFA (weight scales) ----
            uint32_t *sfa_buf = shared_storage.sfa_smem[smem_buf];
#pragma unroll
            for (int i = lane_idx; i < 128; i += 32) {
              int row = m_tile * MMA_M + i;
              int global_row = expert_idx * ORIG_OUTPUT_SIZE + row;
              float sf_val =
                  (row < OUTPUT_SIZE) ? mWeightScale(global_row, k_tile) : 1.0f;
              uint32_t ue8m0 = (__float_as_uint(sf_val) >> 23) & 0xFF;
              sfa_buf[i] = ue8m0 | (ue8m0 << 8) | (ue8m0 << 16) | (ue8m0 << 24);
            }

            // ---- Load + convert + pack SFB (input scales) ----
            uint32_t *sfb_buf = shared_storage.sfb_smem[smem_buf];
#pragma unroll
            for (int i = lane_idx; i < 128; i += 32) {
              uint32_t ue8m0 = 0x7F; // default: UE8M0(1.0) for padding
              if (i < MMA_N) {
                int32_t token_idx_n = n_tile * MMA_N + i;
                // Guard: MMA_N tile may exceed BATCH_SIZE
                int32_t topk_idx = (token_idx_n < BATCH_SIZE)
                                       ? tRoutingIndex(token_idx_n)
                                       : 0;
                if (token_idx_n < BATCH_SIZE && topk_idx > 0) {
                  float sf_val;
                  if constexpr (W13_LINEAR) {
                    sf_val = mInputScale(token_idx_n, k_tile);
                  } else {
                    sf_val = mInputScale(token_idx_n, topk_idx - 1, k_tile);
                  }
                  ue8m0 = (__float_as_uint(sf_val) >> 23) & 0xFF;
                }
              }
              sfb_buf[i] = ue8m0 | (ue8m0 << 8) | (ue8m0 << 16) | (ue8m0 << 24);
            }
            __syncwarp();

            // ---- Warp-transpose for UTCCP layout ----
            fp8_utccp_warp_transpose(sfa_buf);
            fp8_utccp_warp_transpose(sfb_buf);
            cutlass::arch::fence_view_async_shared();

            // ---- Signal MMA warp that scales are ready ----
            if (cute::elect_one_sync()) {
              cute::arrive_barrier(shared_storage.sf_ready_mbar_ptr[smem_buf]);
            }
          } // end k_tile loop
        }   // end n_tile loop
      }     // end m_tile loop
    }       // end expert loop
  }         // end warp 6 (scale warp)
  // ================================================================
  // WARPS 0-3 -- EPILOGUE: TMEM -> BF16 -> Global Memory
  // ================================================================
  //
  // The epilogue reads FP32 accumulators from TMEM, converts them to BF16,
  // and scatter-writes results to the correct positions in the output tensor
  // based on MoE routing indices.
  //
  // This section is architecturally identical to the BF16 MoE task's epilogue
  // because the FP32 accumulator layout in TMEM is the same regardless of
  // whether the UMMA input was FP8 or BF16 — the accumulator shape depends
  // only on (MMA_M, MMA_N), not on the input data type.
  //
  // Data flow: TMEM (FP32) -> registers (FP32) -> convert -> registers (BF16)
  // -> global mem
  //
  // Thread mapping: each of 128 threads (warps 0-3) handles one row of the
  // MMA_M=128-row accumulator tile. Thread i reads row i from TMEM and writes
  // it to output[token, topk_slot, m_tile*MMA_M + i].
  //
  else if (warp_idx < 4) {
    // ---- TMEM ALLOCATION (warp 0 only) ----
    // Allocate TMEM columns for accumulators + scale factors.
    // The base column address is stored in shared_storage.tmem_base_ptr
    // and broadcast to all warps via the tmem_allocation_result_barrier.
    if (warp_idx == 0) {
      tmem_allocator.allocate(kNumTmemColsTotal, &shared_storage.tmem_base_ptr);
    }
    // Wait for allocation to complete, then set the TMEM fragment's base
    // address.
    tmem_allocation_result_barrier.arrive_and_wait();
    tCtAcc.data() = shared_storage.tmem_base_ptr;

    // ---- TMEM LOAD AND CONVERSION SETUP ----
    // CuTe provides SM100_TMEM_LOAD_32dp32b1x: a hardware instruction that
    // loads 32 data points of 32 bits each from TMEM into thread registers.
    // We use the BF16 TiledMMA's fragment structure to determine the correct
    // TMEM addressing for each thread.
    using TypeC = bf16_t;
    cutlass::NumericConverter<AccType, AccType>
        converterBias;                                   // identity (unused)
    cutlass::NumericConverter<TypeC, AccType> converter; // FP32 -> BF16

    // Build a tiled TMEM->register copy operation.
    // tiled_copy_t2r maps each thread to its TMEM rows/columns.
    cute::TiledCopy tiled_copy_t2r =
        cute::make_tmem_copy(cute::SM100_TMEM_LOAD_32dp32b1x{},
                             tCtAcc(cute::_, cute::_, cute::_, 0));
    cute::ThrCopy thr_copy_t2r = tiled_copy_t2r.get_slice(threadIdx.x);
    // tTR_tAcc: source partition (TMEM side)
    cute::Tensor tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc);

    // tTR_rAcc: destination partition (register side)
    // We need a register tensor shaped to match the TMEM load output.
    // Use a fake BF16 tensor to derive the correct shape, then create an
    // AccType (FP32) tensor with the same shape for the actual TMEM load.
    cute::Tensor tCgC_fake = cute::make_tensor<TypeC>(
        cute::shape(tCtAcc(cute::_, cute::_, cute::_, 0)));
    cute::Tensor tTR_rAcc_fake = thr_copy_t2r.partition_D(tCgC_fake);
    cute::Tensor tTR_rAcc =
        cute::make_tensor<AccType>(cute::shape(tTR_rAcc_fake));

    // ---- EPILOGUE LOOP (same expert x m_tile x n_tile iteration order) ----
    int num_tiles_executed = 0;
    for (int activated_expert_offset = expert_offset;
         activated_expert_offset < num_activated_experts;
         activated_expert_offset += EXPERT_STRIDE) {
      int32_t expert_idx = mMask[activated_expert_offset];
      cute::Tensor tRoutingIndex = mRoutingIndices(expert_idx, cute::_);
      for (int m_tile = 0; m_tile < fp8_num_m_tiles; ++m_tile) {
        for (int n_tile = 0; n_tile < fp8_num_n_tiles; ++n_tile) {
          int acc_buf_idx = num_tiles_executed % NUM_ACC_STAGE;
          int acc_full_phase = num_tiles_executed / NUM_ACC_STAGE % 2;

          // Register buffer for BF16 output values (one per N-dim token)
          cute::Tensor tCrC =
              cute::make_tensor<TypeC>(cute::shape(tTR_rAcc(0, cute::_, 0, 0)));

          // Wait for MMA warp to finish accumulating all K-tiles
          cute::wait_barrier(shared_storage.acc_full_mbar_ptr[acc_buf_idx],
                             acc_full_phase);

          // TMEM -> registers: bulk load FP32 accumulators from TMEM
          cute::copy(tiled_copy_t2r,
                     tTR_tAcc(cute::_, cute::_, cute::_, cute::_, acc_buf_idx),
                     tTR_rAcc);

          // Synchronize all 4 epilogue warps, then signal MMA warp that the
          // accumulator buffer is free to reuse. Only one elected lane per
          // warp arrives at the barrier (4 total arrivals match
          // initialization).
          epilogue_wg_barrier.arrive_and_wait();
          if (cute::elect_one_sync()) {
            cute::arrive_barrier(
                shared_storage.acc_empty_mbar_ptr[acc_buf_idx]);
          }

          // Convert FP32 accumulator values to BF16
          CUTE_UNROLL
          for (int i = 0; i < tCrC.size(); i++) {
            tCrC[i] = converter(tTR_rAcc[i]);
          }

          // Scatter-write BF16 results to global memory.
          // Each thread writes one output row (m_idx = m_tile*MMA_M +
          // threadIdx.x) for each token in the N-tile. Only routed tokens are
          // written (topk_idx > 0). Unrouted tokens' output positions are left
          // unchanged.
          //
          // Output tensor layout: mOutput(n_idx, topk_idx, m_idx)
          //   n_idx:    token index (batch dimension)
          //   topk_idx: top-k slot (0-indexed, after subtracting 1 from
          //   routing) m_idx:    output row (weight dimension)
          CUTE_UNROLL
          for (int i = 0; i < MMA_N; ++i) {
            int32_t m_idx =
                m_tile * MMA_M + threadIdx.x;   // output row for this thread
            int32_t n_idx = n_tile * MMA_N + i; // token index
            // Guard: MMA_N tile may exceed BATCH_SIZE
            int32_t topk_idx = (n_idx < BATCH_SIZE)
                                   ? tRoutingIndex(n_idx)
                                   : 0;
            if (n_idx < BATCH_SIZE && topk_idx > 0 && m_idx < OUTPUT_SIZE) {
              // topk_idx is 1-indexed in routing table, convert to 0-indexed
              mOutput(n_idx, topk_idx - 1, m_idx) = tCrC[i];
            }
          }
          // Wait for all epilogue threads to finish writing before proceeding
          // to the next work unit (prevents race on acc_empty signal).
          epilogue_wg_barrier.arrive_and_wait();
          num_tiles_executed++;
        }
      }
    }
  }
  // Warp 7: idle — falls through the if-else chain.

  __syncthreads();

  // Free the TMEM columns allocated at the start of the epilogue.
  // Only warp 0 performs the deallocation (matching the allocation).
  if (warp_idx == 0) {
    tmem_allocator.free(shared_storage.tmem_base_ptr, kNumTmemColsTotal);
  }
} // end fp8_moe_group_gemm_sm100_task_impl

} // namespace kernel
