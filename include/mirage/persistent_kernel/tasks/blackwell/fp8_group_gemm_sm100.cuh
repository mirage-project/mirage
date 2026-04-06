#pragma once
#include <cstdio>
#include <iostream>
// Use PR-647's sm100_utils for correct K-major UMMA descriptor construction
#include "sm100_utils.cuh"

// Cutlass includes
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// CuTe includes
#include <cute/algorithm/cooperative_copy.hpp>
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm100.hpp>    // SM100_UTCCP_4x32dp128bit_1cta
#include <cute/arch/mma_sm100_desc.hpp>   // UMMA::SmemDescriptor, make_instr_desc_block_scaled
#include <cute/arch/mma_sm100_umma.hpp>   // SM100_MMA_MXF8F6F4_SS
#include <cute/arch/tmem_allocator_sm100.hpp>
#include <cute/atom/mma_traits_sm100.hpp>  // Layout_K_SW128_Atom, UMMA namespace
#include <cute/numeric/integral_constant.hpp>
#include <cute/tensor.hpp>

#include "../common/dmem_layout.cuh"
#include "../common/worker_config.h"
#include "../hopper/barrier.cuh"
#include "../hopper/smem_layout_tma.cuh"
#include "../hopper/tma.cuh"
#include "storage.cuh"

namespace kernel {

// -------------------------------------------------------------------
// Inline PTX helpers (CuTe versions aren't in the project's CUTLASS)
// -------------------------------------------------------------------
__device__ __forceinline__ uint32_t fp8_ld_shared(const uint32_t *ptr) {
  uint32_t val;
  asm volatile("ld.shared.u32 %0, [%1];"
               : "=r"(val)
               : "r"(cute::cast_smem_ptr_to_uint(ptr)));
  return val;
}
__device__ __forceinline__ void fp8_st_shared(uint32_t *ptr, uint32_t val) {
  asm volatile("st.shared.u32 [%0], %1;"
               :
               : "r"(cute::cast_smem_ptr_to_uint(ptr)), "r"(val));
}
__device__ __forceinline__ void fp8_tcgen05_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;");
}

// -------------------------------------------------------------------
// Shared-memory storage for FP8 MoE group GEMM.
// Uses raw uint8_t arrays for FP8 tiles (no CuTe layout wrappers on A/B)
// to avoid smem_ptr_flag_bits element-size assertions.
// -------------------------------------------------------------------
template <class BF16_ASmemLayout,  // BF16 layout used only for accumulator shape
          class BF16_BSmemLayout,  // BF16 layout used only for accumulator shape
          class BF16_BCpLayout,    // BF16 layout used for CP_ASYNC structure
          int Num_Experts,
          int Num_AB_Stage,
          int Num_ACC_Stage,
          int MMA_M_val,  // MMA_M
          int MMA_N_val,  // MMA_N
          int bK_val>     // bK
struct MoEFP8SharedStorage {
  // Raw FP8 (uint8_t) weight tile: [Num_AB_Stage, MMA_M * bK]
  alignas(128) uint8_t A[Num_AB_Stage][MMA_M_val * bK_val];
  // Raw FP8 (uint8_t) input tile:  [Num_AB_Stage, MMA_N * bK]
  alignas(128) uint8_t B[Num_AB_Stage][MMA_N_val * bK_val];

  // UE8M0 scale factor buffers — 128 uint32_t per stage for UTCCP.
  // Each uint32 packs 4 UE8M0 bytes (32 uint32 = 128 scales).
  // Buffer sized to 128 for warp-transpose compatibility.
  alignas(128) uint32_t sfa_smem[Num_AB_Stage][128];
  alignas(128) uint32_t sfb_smem[Num_AB_Stage][128];

  // Barriers (sf_ready removed: scale work merged into MMA warp)
  alignas(16) cute::uint64_t a_full_mbar_ptr[Num_AB_Stage];
  alignas(16) cute::uint64_t b_full_mbar_ptr[Num_AB_Stage];
  alignas(16) cute::uint64_t ab_empty_mbar_ptr[Num_AB_Stage];
  alignas(16) cute::uint64_t acc_full_mbar_ptr[Num_ACC_Stage];
  alignas(16) cute::uint64_t acc_empty_mbar_ptr[Num_ACC_Stage];

  alignas(16) cute::uint32_t expert_mask[Num_Experts];
  alignas(16) cute::uint32_t tmem_base_ptr;

  // BF16-typed smem views are only used for the accumulator TMEM epilogue
  // (the A/B CuTe tensors come from the BF16 TiledMMA infrastructure but
  //  we never actually dereference them as smem tensors — they're only used
  //  for fragment shape computation)
};

// -------------------------------------------------------------------
// Adapted from DeepGEMM's sm100_utils.cuh: make_umma_desc for K-major tiles.
template <uint32_t BLOCK_MN, uint32_t BLOCK_K, uint32_t kSwizzleMode>
__device__ __forceinline__
cute::UMMA::SmemDescriptor fp8_make_umma_desc_k_major(uint8_t* base_smem_ptr) {
  // For K-major: swizzle must equal BLOCK_K * sizeof(dtype)
  static_assert(kSwizzleMode == BLOCK_K * sizeof(uint8_t), "Swizzle mismatch");
  // atom_base: try 32 (SWIZZLE_128B_BASE32B) to match warp group size of 32
  constexpr uint32_t atom_base = 32;
  constexpr uint32_t num_non_contiguous = BLOCK_MN / atom_base;
  // SBO = num_non_contiguous * BLOCK_K * sizeof(uint8_t) = 8*128*1 = 1024 bytes
  const uint32_t stride_byte_offset = num_non_contiguous * BLOCK_K * sizeof(uint8_t);
  const uint32_t leading_byte_offset = 0;
  // LayoutType for K-major SWIZZLE_128B
  constexpr auto layout_type = cute::UMMA::LayoutType::SWIZZLE_128B;

  cute::UMMA::SmemDescriptor desc;
  desc.desc_ = 0;
  desc.version_ = 1;
  desc.lbo_mode_ = 0;
  desc.layout_type_ = static_cast<uint8_t>(layout_type);
  const auto uint_ptr = cute::cast_smem_ptr_to_uint(base_smem_ptr);
  desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
  desc.base_offset_ = 0;
  desc.stride_byte_offset_ = stride_byte_offset >> 4;
  desc.leading_byte_offset_ = leading_byte_offset >> 4;
  return desc;
}

// Adapted from DeepGEMM: advance the lo-word of a K-major smem descriptor.
// For K-major: stride_k = 1, so advance = (offset + k_idx) * sizeof(uint8_t) / 16.
__device__ __forceinline__
uint32_t fp8_advance_umma_desc_lo(uint32_t base_lo, uint32_t offset, uint32_t k_idx) {
  // For K-major: stride_k = 1
  return base_lo + (((offset + k_idx * 1) * sizeof(uint8_t)) >> 4u);
}

// Build a smem descriptor for UTCCP scale factor source.
// UTCCP uses SWIZZLE_NONE layout, SBO = 8 * 16 = 128 bytes.
// This matches DeepGEMM's make_sf_desc.
__device__ __forceinline__
cute::UMMA::SmemDescriptor fp8_make_sf_smem_desc(const uint32_t *smem_ptr) {
  cute::UMMA::SmemDescriptor desc;
  desc.desc_ = 0;
  desc.version_ = 1;
  desc.lbo_mode_ = 0;
  desc.layout_type_ = 0; // SWIZZLE_NONE
  const uint32_t uint_ptr = cute::cast_smem_ptr_to_uint(smem_ptr);
  desc.start_address_ = static_cast<uint16_t>(uint_ptr >> 4);
  desc.base_offset_ = 0;
  desc.stride_byte_offset_ = 8; // SBO = 8*16 = 128 bytes, stored as 128/16 = 8
  desc.leading_byte_offset_ = 0;
  return desc;
}

// In-place warp-transpose of 128 uint32_t elements in smem.
// UTCCP requires scale factors in column-major layout.
// Identical to DeepGEMM's utccp_required_smem_warp_transpose.
__device__ __forceinline__
void fp8_utccp_warp_transpose(uint32_t *smem_ptr) {
  const uint32_t lane_idx = cutlass::canonical_lane_idx();
  uint32_t values[4];
  #pragma unroll
  for (uint32_t i = 0; i < 4; ++i)
    values[i] = fp8_ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
  __syncwarp();
  #pragma unroll
  for (uint32_t i = 0; i < 4; ++i)
    fp8_st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
}

// -------------------------------------------------------------------
// FP8 block-scaled MoE group GEMM — per-expert MPK task.
//
// One MPK task = one expert slot (expert_offset indexes into activated
// experts list). Computes:
//   output[routed_tokens, :] = input_fp8[routed_tokens, :] @ weight_fp8[e, :, :]
//
// Block-scaled FP8 UMMA: SM100 tcgen05.mma.kind::mxf8f6f4.block_scale
//   A = FP8 E4M3 weights [MMA_M, bK=128], scale UE8M0 (one per 128 K-elements)
//   B = FP8 E4M3 inputs  [bK=128, MMA_N], scale UE8M0 (one per 128 K-elements)
//   C = FP32 accumulators (stored in TMEM)
//   Output = BF16 (cast from FP32 in epilogue)
//
// Warp roles (8 warps × 32 threads = 256 threads):
//   Warps 0-3 (0..127):   Epilogue (TMEM→BF16→global mem)
//   Warp 4    (128..159):  MMA warp (UTCCP + UMMA issuance)
//   Warp 5    (160..191):  DMA warp (TMA weight load + CP_ASYNC input load)
//   Warp 6    (192..223):  Scale warp (float32→UE8M0, warp-transpose, signal)
//   Warp 7    (224..255):  Idle
// -------------------------------------------------------------------
template <typename TMA_Weight,       // tma_2d<uint8_t, ...> for FP8 weights
          class InputTensor,         // bfloat16_t or uint8_t input tensor
          class InputScaleTensor,    // float32 input scale [batch, K/128]
          class WeightScaleTensor,   // float32 weight scale [experts*N, K/128]
          class IndicesTensor,       // int32 routing indices [num_experts, batch]
          class MaskTensor,          // int32 expert mask [num_experts+1]
          class OutputTensor,        // BF16 output
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
    fp8_moe_group_gemm_sm100_task_impl(const TMA_Weight &tma_weight,
                                       InputTensor mInput,
                                       InputScaleTensor mInputScale,
                                       WeightScaleTensor mWeightScale,
                                       IndicesTensor mRoutingIndices,
                                       MaskTensor mMask,
                                       OutputTensor mOutput,
                                       int const expert_offset) {
  using namespace cute;

  using bf16_t = cute::bfloat16_t;
  using AccType = float;
  using ue8m0_t = cutlass::float_ue8m0_t;

  // bK = 128: one k-tile = one scale-block (128 K-elements per scale)
  constexpr int bK = 128;
  // UMMA processes 32 K-elements per call → 4 UMMAs per K-tile
  constexpr int UMMA_K = 32;
  // TMEM layout: accumulator cols + SFA cols + SFB cols
  // With MMA_N=128 (aligned to DeepGEMM): acc = 128*2=256, SFA=4, SFB=4 → 512
  constexpr int num_tmem_acc_cols  = MMA_N * NUM_ACC_STAGE;
  constexpr int kNumSFATmemCols    = 4;
  constexpr int kNumSFBTmemCols    = 4;
  constexpr int kTmemStartColOfSFA = num_tmem_acc_cols;
  constexpr int kTmemStartColOfSFB = num_tmem_acc_cols + kNumSFATmemCols;
  // Round up to next valid TMEM size
  constexpr int kNumTmemColsRaw    = num_tmem_acc_cols + kNumSFATmemCols + kNumSFBTmemCols;
  constexpr int kNumTmemColsTotal  = kNumTmemColsRaw <= 64  ? 64
                                   : kNumTmemColsRaw <= 128 ? 128
                                   : kNumTmemColsRaw <= 256 ? 256 : 512;

  // --- Use BF16 TiledMMA only for accumulator tensor structure ---
  // The FP32 accumulator layout in TMEM is the same for BF16 and FP8 UMMAs
  // with the same (MMA_M, MMA_N). We use the BF16 infrastructure for the
  // epilogue TMEM load/store, but issue the actual MMA manually (FP8).
  cute::TiledMMA tiled_mma_bf16 = cute::make_tiled_mma(
      cute::SM100_MMA_F16BF16_SS<bf16_t, bf16_t, AccType,
                                  MMA_M, MMA_N,
                                  UMMA::Major::K, UMMA::Major::K>{});

  auto mma_coord_vmnk = cute::make_coord(0, cute::_, cute::_, cute::_);
  auto mma_v = cute::get<0>(mma_coord_vmnk);

  // Coordinate view of weight global tensor (no data pointer needed here)
  cute::Tensor mA = cute::make_coord_tensor(cute::make_layout(
      cute::make_shape(ORIG_OUTPUT_SIZE * NUM_EXPERTS, REDUCTION_SIZE),
      cute::make_stride(cute::E<1>{} * cute::Int<REDUCTION_SIZE>{}, cute::E<0>{})));

  auto mma_tiler =
      cute::make_shape(cute::Int<MMA_M>{}, cute::Int<MMA_N>{}, cute::Int<bK>{});
  auto mma_coord = cute::select<1, 2, 3>(mma_coord_vmnk);
  auto cd_tiler  =
      cute::make_shape(cute::Int<MMA_N>{}, cute::Int<MMA_M>{}, cute::Int<bK>{});

  cute::Tensor gA = cute::local_tile(
      mA, mma_tiler, mma_coord, cute::Step<cute::_1, cute::X, cute::_1>{});
  // (gB removed: input copy is done manually with cp.async PTX)

  // Accumulator shape — use BF16 TiledMMA only for C/accumulator TMEM structure
  // (FP8 UMMA and BF16 UMMA with same (MMA_M, MMA_N) produce identical accumulators)
  auto mma_shape_A_acc = cute::partition_shape_A(
      tiled_mma_bf16,
      cute::make_shape(cute::Int<MMA_M>{},
                       cute::size<2>(mma_tiler) / 2, // bK/2 for BF16 tile counting
                       cute::Int<NUM_AB_STAGE>{}));

  // Shared memory storage — raw arrays, no CuTe layout wrappers on A/B
  using SharedStorage = MoEFP8SharedStorage<decltype(mma_shape_A_acc), // placeholder
                                             decltype(mma_shape_A_acc), // placeholder
                                             decltype(mma_shape_A_acc), // placeholder
                                             NUM_EXPERTS,
                                             NUM_AB_STAGE,
                                             NUM_ACC_STAGE,
                                             MMA_M, MMA_N, bK>;
  extern __shared__ char shared_memory[];
  uintptr_t aligned_smem =
      (reinterpret_cast<uintptr_t>(shared_memory) + 127) / 128 * 128;
  SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(aligned_smem);

  int warp_idx = cutlass::canonical_warp_idx_sync();

  // Barrier initialization (warp 0 of epilogue group)
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
    // sf_ready removed: scale work merged into MMA warp
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_ACC_STAGE>(shared_storage.acc_full_mbar_ptr, 1);
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterBarrier,
        NUM_ACC_STAGE>(shared_storage.acc_empty_mbar_ptr, 4);
  }

  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
      32 + 128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);
  cutlass::arch::NamedBarrier epilogue_wg_barrier(
      128, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);

  // BF16 MMA used only for accumulator TMEM fragment layout
  cute::ThrMMA cta_mma = tiled_mma_bf16.get_slice(mma_v);
  cute::Tensor tCgA = cta_mma.partition_A(gA);

  // TMA transaction bytes for one FP8 weight tile [MMA_M × bK]
  // = MMA_M * bK * sizeof(uint8_t) = 128 * 128 * 1 = 16384
  int tma_transaction_bytes_A = MMA_M * bK * sizeof(uint8_t);

  // Tile counts computed directly from tensor dimensions (avoid CuTe BF16 proxy issues)
  // k_tile_count: number of bK-wide K-tiles = REDUCTION_SIZE / bK
  // num_m_tiles:  number of MMA_M-wide output-dim tiles = ORIG_OUTPUT_SIZE / MMA_M
  // num_n_tiles:  number of MMA_N-wide token-dim tiles = BATCH_SIZE / MMA_N
  constexpr int fp8_k_tile_count  = REDUCTION_SIZE / bK;
  constexpr int fp8_num_m_tiles   = ORIG_OUTPUT_SIZE / MMA_M;
  constexpr int fp8_num_n_tiles   = (BATCH_SIZE + MMA_N - 1) / MMA_N; // ceiling div

  // CP_ASYNC for input loading (uint8_t FP8 elements)
  constexpr int TILE_SIZE           = bK;
  constexpr int WEIGHT_TMA_TILE_SIZE = bK;
  constexpr int OUTPUT_ATOM_SIZE     = MMA_M;
  constexpr int B_PARAM              = 3;
  constexpr int M_PARAM              = 3;
  constexpr int S_PARAM              = 3;
  uint8_t *shared_weight = &shared_storage.A[0][0];

  using Barrier = cutlass::arch::ClusterTransactionBarrier;
  Barrier *a_full_mbar_ptr =
      reinterpret_cast<Barrier *>(shared_storage.a_full_mbar_ptr);

  using WeightSmem = smem_tma<uint8_t,
                               B_PARAM,
                               M_PARAM,
                               S_PARAM,
                               OUTPUT_ATOM_SIZE,
                               WEIGHT_TMA_TILE_SIZE,
                               1>;
  WeightSmem weight_smem(shared_weight);

  // (Input B copy is done manually with cp.async PTX in the DMA warp)

  // Accumulator TMEM fragment (BF16 TiledMMA used only for structure)
  auto acc_shape = cute::partition_shape_C(
      tiled_mma_bf16,
      cute::make_shape(cute::size<0>(mma_tiler),
                       cute::size<1>(mma_tiler),
                       cute::Int<NUM_ACC_STAGE>{}));
  auto tCtAcc = tiled_mma_bf16.make_fragment_C(acc_shape);

  cutlass::arch::fence_barrier_init();
  __syncthreads();

  int k_tile_count          = fp8_k_tile_count;
  int num_activated_experts = mMask(NUM_EXPERTS);

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};

  __syncthreads();

  // ================================================================
  // WARP 5 — DMA warp: TMA weight + CP_ASYNC input
  // ================================================================
  if (warp_idx == 5) {
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

          int tma_wr_k_tile          = 0;
          int smem_wr_buffer         = (num_prev_k_blk + tma_wr_k_tile) % NUM_AB_STAGE;
          int tma_wr_ab_empty_phase  =
              (num_prev_k_blk + tma_wr_k_tile) / NUM_AB_STAGE % 2 ^ 1;

          bool peek_ab_empty_status = kernel::try_wait_barrier(
              shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
              tma_wr_ab_empty_phase);

          for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
            int tma_wr_k_tile_next        = tma_wr_k_tile + 1;
            int smem_wr_buffer_next       =
                (num_prev_k_blk + tma_wr_k_tile_next) % NUM_AB_STAGE;
            int tma_wr_ab_empty_phase_next =
                smem_wr_buffer_next == 0 ? tma_wr_ab_empty_phase ^ 1
                                         : tma_wr_ab_empty_phase;

            if (!peek_ab_empty_status) {
              cute::wait_barrier(
                  shared_storage.ab_empty_mbar_ptr[smem_wr_buffer],
                  tma_wr_ab_empty_phase);
            }

            // TMA load: FP8 weight tile [MMA_M × bK]
            if (cute::elect_one_sync()) {
              int tma_coords_A[2] = {
                  k_tile * TILE_SIZE,
                  m_tile * OUTPUT_ATOM_SIZE + expert_idx * ORIG_OUTPUT_SIZE};
              weight_smem.set_ptr(shared_weight +
                                  smem_wr_buffer * OUTPUT_ATOM_SIZE * TILE_SIZE);
              cute::set_barrier_transaction_bytes(
                  shared_storage.a_full_mbar_ptr[smem_wr_buffer],
                  tma_transaction_bytes_A);
              tma_weight.tma_cp_async(a_full_mbar_ptr[smem_wr_buffer],
                                      weight_smem.base_ptr,
                                      tma_coords_A);
            }

            // Manual FP8 input copy: each of 32 lanes copies part of
            // [MMA_N, bK] tile = 16×128 = 2048 bytes total.
            // 32 threads × 64 bytes each = 2048 bytes.
            {
              uint8_t *dst_base =
                  &shared_storage.B[smem_wr_buffer][0];
              // Each lane copies 64 bytes (4 × 16B via cp.async)
              // Manual FP8 input copy with SWIZZLE_128B to match A and UMMA descriptor.
              // Swizzle<B=3, M=4, S=3>: swizzled = L ^ (((L >> (B+S)) & ((1<<B)-1)) << S)
              //                        = L ^ (((L >> 6) & 7) << 3)
              // Produces 8-byte aligned addresses, so use 8-byte cp.async.
              constexpr int BYTES_PER_THREAD = (MMA_N * bK) / 32;
              constexpr int COPIES_PER_THREAD = BYTES_PER_THREAD / 8; // 8 bytes per cp.async
              int base_byte = lane_idx * BYTES_PER_THREAD;
              #pragma unroll
              for (int c = 0; c < COPIES_PER_THREAD; ++c) {
                int byte_off = base_byte + c * 8;
                int row = byte_off / bK;
                int col = byte_off % bK;
                int32_t token_idx_n = n_tile * MMA_N + row;
                int32_t topk_idx_n  = tRoutingIndex(token_idx_n);

                // Apply SWIZZLE_128B for uint8_t:
                // row = L/128, swizzled = L ^ ((row%8) * 16) = L ^ (((L>>7)&7)<<4)
                int linear_off = row * bK + col;
                int swizzled_off = linear_off ^ (((linear_off >> 7) & 7) << 4);
                uint32_t dst_smem = cute::cast_smem_ptr_to_uint(
                    dst_base + swizzled_off);

                if (token_idx_n < BATCH_SIZE && topk_idx_n > 0) {
                  const uint8_t *src_row;
                  if constexpr (W13_LINEAR) {
                    src_row = reinterpret_cast<const uint8_t*>(
                        &mInput(token_idx_n, k_tile * bK + col));
                  } else {
                    src_row = reinterpret_cast<const uint8_t*>(
                        &mInput(token_idx_n, topk_idx_n - 1, k_tile * bK + col));
                  }
                  asm volatile(
                      "cp.async.ca.shared.global [%0], [%1], 8;\n"
                      :: "r"(dst_smem), "l"(src_row));
                } else {
                  *reinterpret_cast<uint2*>(dst_base + swizzled_off) =
                      make_uint2(0, 0);
                }
              }
              // Commit cp.async group and arrive on barrier
              asm volatile("cp.async.commit_group;\n" ::: "memory");
            }
            cutlass::arch::cpasync_barrier_arrive_noinc(
                &shared_storage.b_full_mbar_ptr[smem_wr_buffer]);

            if (tma_wr_k_tile_next < k_tile_count) {
              peek_ab_empty_status = kernel::try_wait_barrier(
                  shared_storage.ab_empty_mbar_ptr[smem_wr_buffer_next],
                  tma_wr_ab_empty_phase_next);
            }
            tma_wr_k_tile        = tma_wr_k_tile_next;
            smem_wr_buffer       = smem_wr_buffer_next;
            tma_wr_ab_empty_phase = tma_wr_ab_empty_phase_next;
          }
        }
      }
    }
  }
  // ================================================================
  // WARP 4 — MMA warp: scales + UTCCP + FP8 block-scaled UMMA
  //          (Scale work previously in warp 6 is merged here)
  //
  // Warp roles: warps 0-3=epilogue, warp 4=MMA+scale, warp 5=DMA, warps 6-7=idle
  // ================================================================
  else if (warp_idx == 4) {
    tmem_allocation_result_barrier.arrive_and_wait();
    tCtAcc.data() = shared_storage.tmem_base_ptr;

    // TMEM column addresses for accumulators and scale factors
    const uint32_t tmem_base = shared_storage.tmem_base_ptr;
    const uint32_t sfa_tmem  = tmem_base + kTmemStartColOfSFA;
    const uint32_t sfb_tmem  = tmem_base + kTmemStartColOfSFB;

    using UTCCP_t = cute::SM100::TMEM::UTCCP::SM100_UTCCP_4x32dp128bit_1cta;

    // Build instruction descriptor (PR-647 approach)
    auto instr_desc = cute::UMMA::make_instr_desc_block_scaled<
        cutlass::float_e4m3_t, cutlass::float_e4m3_t, float,
        cutlass::float_ue8m0_t, MMA_M, MMA_N,
        cute::UMMA::Major::K, cute::UMMA::Major::K>();

    // SF descriptor (SWIZZLE_NONE for UTCCP, as in PR-647)
    auto sf_desc = kernel::sm100::make_sf_desc(nullptr);

    // Build base smem descriptors using PR-647's sm100::make_umma_desc
    // A (weight): K-major, BLOCK_MN=MMA_M, BLOCK_K=128, swizzle=128
    auto a_desc = kernel::sm100::make_umma_desc<
        cute::UMMA::Major::K, MMA_M, bK, 128>(
        reinterpret_cast<cutlass::float_e4m3_t*>(&shared_storage.A[0][0]),
        0, 0);
    // B (input): K-major, BLOCK_MN=MMA_N, BLOCK_K=128, swizzle=128
    auto b_desc = kernel::sm100::make_umma_desc<
        cute::UMMA::Major::K, MMA_N, bK, 128>(
        reinterpret_cast<cutlass::float_e4m3_t*>(&shared_storage.B[0][0]),
        0, 0);

    // Each lane i holds stage i's descriptor lo-word (PR-647 __shfl_sync pattern)
    const uint32_t lane_idx = cutlass::canonical_lane_idx();
    uint32_t a_desc_lo = (lane_idx < (uint32_t)NUM_AB_STAGE)
        ? (a_desc.lo + lane_idx * (MMA_M * bK * sizeof(uint8_t)) / 16) : 0u;
    uint32_t b_desc_lo = (lane_idx < (uint32_t)NUM_AB_STAGE)
        ? (b_desc.lo + lane_idx * (MMA_N * bK * sizeof(uint8_t)) / 16) : 0u;

    int total_k_tile_count  = 0;
    int num_tiles_executed  = 0;

    for (int activated_expert_offset = expert_offset;
         activated_expert_offset < num_activated_experts;
         activated_expert_offset += EXPERT_STRIDE) {
      int32_t expert_idx = mMask[activated_expert_offset];
      cute::Tensor tRoutingIndex = mRoutingIndices(expert_idx, cute::_);

      for (int m_tile = 0; m_tile < fp8_num_m_tiles; ++m_tile) {
        for (int n_tile = 0; n_tile < fp8_num_n_tiles; ++n_tile) {
          int acc_buf_idx    = num_tiles_executed % NUM_ACC_STAGE;

          int num_prev_k_blk = total_k_tile_count;
          total_k_tile_count += k_tile_count;

          int mma_rd_k_tile        = 0;
          int smem_rd_buf          = (num_prev_k_blk + mma_rd_k_tile) % NUM_AB_STAGE;
          int mma_rd_ab_full_phase =
              (num_prev_k_blk + mma_rd_k_tile) / NUM_AB_STAGE % 2;


          bool peek_a = kernel::try_wait_barrier(
              shared_storage.a_full_mbar_ptr[smem_rd_buf], mma_rd_ab_full_phase);
          bool peek_b = kernel::try_wait_barrier(
              shared_storage.b_full_mbar_ptr[smem_rd_buf], mma_rd_ab_full_phase);

          int acc_empty_phase = num_tiles_executed / NUM_ACC_STAGE % 2 ^ 1;
          cute::wait_barrier(shared_storage.acc_empty_mbar_ptr[acc_buf_idx],
                             acc_empty_phase);


          bool first_tile = true;

          for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
            int mma_rd_k_tile_next      = mma_rd_k_tile + 1;
            int smem_rd_buf_next        =
                (num_prev_k_blk + mma_rd_k_tile_next) % NUM_AB_STAGE;
            int mma_rd_ab_full_phase_next =
                smem_rd_buf_next == 0 ? mma_rd_ab_full_phase ^ 1
                                      : mma_rd_ab_full_phase;

            if (!peek_a)
              cute::wait_barrier(shared_storage.a_full_mbar_ptr[smem_rd_buf],
                                 mma_rd_ab_full_phase);
            if (!peek_b)
              cute::wait_barrier(shared_storage.b_full_mbar_ptr[smem_rd_buf],
                                 mma_rd_ab_full_phase);

// --- Inline scale loading (PR-647 format) ---
            // PR-647 stores ONE packed uint32 PER ROW where each uint32 holds 4 UE8M0
            // bytes — one per 32-element K-subtile within the 128-K block:
            //   bits 0-7:   UE8M0 for k_sub=0 (K[0..31])
            //   bits 8-15:  UE8M0 for k_sub=1 (K[32..63])
            //   bits 16-23: UE8M0 for k_sub=2 (K[64..95])
            //   bits 24-31: UE8M0 for k_sub=3 (K[96..127])
            // For our per-128-K granularity: all 4 sub-tiles share one scale → same
            // UE8M0 replicated in all 4 bytes.
            // SFA: weight scales — 128 uint32 entries (one per weight row)
            uint32_t *sfa_buf = shared_storage.sfa_smem[smem_rd_buf];
            #pragma unroll
            for (int i = lane_idx; i < 128; i += 32) {
              int row = m_tile * MMA_M + i;
              int global_row = expert_idx * ORIG_OUTPUT_SIZE + row;
              float sf_val = (row < OUTPUT_SIZE)
                  ? mWeightScale(global_row, k_tile)
                  : 1.0f;
              uint32_t ue8m0 = (__float_as_uint(sf_val) >> 23) & 0xFF;
              // Replicate into all 4 bytes (all sub-tiles share same scale)
              sfa_buf[i] = ue8m0 | (ue8m0 << 8) | (ue8m0 << 16) | (ue8m0 << 24);
            }
            // SFB: input scales — 128 uint32 entries (one per token slot)
            uint32_t *sfb_buf = shared_storage.sfb_smem[smem_rd_buf];
            #pragma unroll
            for (int i = lane_idx; i < 128; i += 32) {
              uint32_t ue8m0 = 0x7F; // padding = UE8M0(1.0)
              if (i < MMA_N) {
                int32_t token_idx_n = n_tile * MMA_N + i;
                int32_t topk_idx    = tRoutingIndex(token_idx_n);
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

            // Apply warp-transpose for UTCCP layout (same as PR-647)
            fp8_utccp_warp_transpose(sfa_buf);
            fp8_utccp_warp_transpose(sfb_buf);
            cutlass::arch::fence_view_async_shared();

            // UTCCP: copy UE8M0 scales from smem → TMEM (PR-647 exact approach)
            if (cute::elect_one_sync()) {
              // SFA: 128 uint32 → 4 TMEM columns at kTmemStartColOfSFA
              kernel::sm100::replace_smem_desc_addr(
                  sf_desc, shared_storage.sfa_smem[smem_rd_buf]);
              UTCCP_t::copy(static_cast<uint64_t>(sf_desc), sfa_tmem);
              // SFB: 128 uint32 → 4 TMEM columns at kTmemStartColOfSFB
              kernel::sm100::replace_smem_desc_addr(
                  sf_desc, shared_storage.sfb_smem[smem_rd_buf]);
              UTCCP_t::copy(static_cast<uint64_t>(sf_desc), sfb_tmem);
            }
            __syncwarp();

            // Broadcast per-stage base lo-word via __shfl_sync (PR-647 exact approach)
            const auto a_desc_base_lo =
                __shfl_sync(0xffffffff, a_desc_lo, smem_rd_buf);
            const auto b_desc_base_lo =
                __shfl_sync(0xffffffff, b_desc_lo, smem_rd_buf);

            // Issue 4 FP8 UMMAs per K-tile (PR-647 exact approach)
            if (cute::elect_one_sync()) {
              #pragma unroll
              for (int k_sub = 0; k_sub < bK / UMMA_K; ++k_sub) {
                // sfa_id selects which byte of the packed uint32 to use
                // k_sub 0→byte0, 1→byte1, 2→byte2, 3→byte3
                const uint32_t sfa_id = k_sub;
                const uint32_t sfb_id = k_sub;
                auto runtime_instr_desc =
                    kernel::sm100::make_runtime_instr_desc_with_sf_id(
                        instr_desc, sfa_id, sfb_id);

                b_desc.lo = kernel::sm100::advance_umma_desc_lo<
                    cute::UMMA::Major::K, MMA_N, 128,
                    cutlass::float_e4m3_t>(b_desc_base_lo, 0, k_sub * UMMA_K);
                a_desc.lo = kernel::sm100::advance_umma_desc_lo<
                    cute::UMMA::Major::K, MMA_M, 128,
                    cutlass::float_e4m3_t>(a_desc_base_lo, 0, k_sub * UMMA_K);

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


            cutlass::arch::umma_arrive(
                &shared_storage.ab_empty_mbar_ptr[smem_rd_buf]);


            if (mma_rd_k_tile_next < k_tile_count) {
              peek_a = kernel::try_wait_barrier(
                  shared_storage.a_full_mbar_ptr[smem_rd_buf_next],
                  mma_rd_ab_full_phase_next);
              peek_b = kernel::try_wait_barrier(
                  shared_storage.b_full_mbar_ptr[smem_rd_buf_next],
                  mma_rd_ab_full_phase_next);
            }

            mma_rd_k_tile        = mma_rd_k_tile_next;
            smem_rd_buf          = smem_rd_buf_next;
            mma_rd_ab_full_phase = mma_rd_ab_full_phase_next;
          }

          cutlass::arch::umma_arrive(
              &shared_storage.acc_full_mbar_ptr[acc_buf_idx]);
          num_tiles_executed++;
        }
      }
    }
  }
  // ================================================================
  // WARPS 0-3 — Epilogue: TMEM → BF16 → global memory
  // (Identical to BF16 MoE task since accumulators are FP32)
  // ================================================================
  else if (warp_idx < 4) {
    if (warp_idx == 0) {
      tmem_allocator.allocate(kNumTmemColsTotal, &shared_storage.tmem_base_ptr);
    }
    tmem_allocation_result_barrier.arrive_and_wait();
    tCtAcc.data() = shared_storage.tmem_base_ptr;

    // CuTe-based epilogue: TMEM → registers → BF16 → global memory
    // Uses the BF16 TiledMMA's fragment structure for correct TMEM data path mapping.
    using TypeC = bf16_t;
    cutlass::NumericConverter<AccType, AccType> converterBias;
    cutlass::NumericConverter<TypeC, AccType>   converter;

    cute::TiledCopy tiled_copy_t2r =
        cute::make_tmem_copy(cute::SM100_TMEM_LOAD_32dp32b1x{},
                             tCtAcc(cute::_, cute::_, cute::_, 0));
    cute::ThrCopy thr_copy_t2r = tiled_copy_t2r.get_slice(threadIdx.x);
    cute::Tensor tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc);

    cute::Tensor tCgC_fake =
        cute::make_tensor<TypeC>(cute::shape(tCtAcc(cute::_, cute::_, cute::_, 0)));
    cute::Tensor tTR_rAcc_fake = thr_copy_t2r.partition_D(tCgC_fake);
    cute::Tensor tTR_rAcc =
        cute::make_tensor<AccType>(cute::shape(tTR_rAcc_fake));

    int num_tiles_executed = 0;
    for (int activated_expert_offset = expert_offset;
         activated_expert_offset < num_activated_experts;
         activated_expert_offset += EXPERT_STRIDE) {
      int32_t expert_idx = mMask[activated_expert_offset];
      cute::Tensor tRoutingIndex = mRoutingIndices(expert_idx, cute::_);
      for (int m_tile = 0; m_tile < fp8_num_m_tiles; ++m_tile) {
        for (int n_tile = 0; n_tile < fp8_num_n_tiles; ++n_tile) {
          int acc_buf_idx    = num_tiles_executed % NUM_ACC_STAGE;
          int acc_full_phase = num_tiles_executed / NUM_ACC_STAGE % 2;

          cute::Tensor tCrC =
              cute::make_tensor<TypeC>(cute::shape(tTR_rAcc(0, cute::_, 0, 0)));

          cute::wait_barrier(shared_storage.acc_full_mbar_ptr[acc_buf_idx],
                             acc_full_phase);
          cute::copy(tiled_copy_t2r,
                     tTR_tAcc(cute::_, cute::_, cute::_, cute::_, acc_buf_idx),
                     tTR_rAcc);

          epilogue_wg_barrier.arrive_and_wait();
          if (cute::elect_one_sync()) {
            cute::arrive_barrier(
                shared_storage.acc_empty_mbar_ptr[acc_buf_idx]);
          }

          CUTE_UNROLL
          for (int i = 0; i < tCrC.size(); i++) {
            tCrC[i] = converter(tTR_rAcc[i]);
          }

          CUTE_UNROLL
          for (int i = 0; i < MMA_N; ++i) {
            int32_t m_idx    = m_tile * MMA_M + threadIdx.x;
            int32_t n_idx    = n_tile * MMA_N + i;
            int32_t topk_idx = tRoutingIndex(n_idx);
            if (n_idx < BATCH_SIZE && topk_idx > 0) {
              mOutput(n_idx, topk_idx - 1, m_idx) = tCrC[i];
            }
          }
          epilogue_wg_barrier.arrive_and_wait();
          num_tiles_executed++;
        }
      }
    }
  }
  // Warp 7: idle (falls through)

  __syncthreads();

  if (warp_idx == 0) {
    tmem_allocator.free(shared_storage.tmem_base_ptr, kNumTmemColsTotal);
  }
} // end fp8_moe_group_gemm_sm100_task_impl

} // namespace kernel
