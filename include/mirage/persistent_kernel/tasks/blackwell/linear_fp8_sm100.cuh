#pragma once
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>

#include "linear_fp8_sm100_common.cuh"

namespace mirage::blackwell::linear_fp8_sm100_detail {

using namespace mirage::blackwell::linear_fp8_sm100;
using namespace mirage::blackwell::linear_fp8_sm100::sm100;

} // namespace mirage::blackwell::linear_fp8_sm100_detail

namespace kernel {

template <cute::UMMA::Major kMajorA,
          cute::UMMA::Major kMajorB,
          uint32_t kGranKA,
          uint32_t kGranKB,
          uint32_t SHAPE_M,
          uint32_t SHAPE_N,
          uint32_t SHAPE_K,
          uint32_t BLOCK_M,
          uint32_t BLOCK_N,
          uint32_t BLOCK_K,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode,
          uint32_t kSwizzleBMode,
          uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast,
          bool kIsMulticastOnA,
          uint32_t kNumSMs,
          bool kWithResidual,
          mirage::blackwell::linear_fp8_sm100::GemmType kGemmType,
          bool kWithAccumulation,
          typename a_dtype_t,
          typename b_dtype_t,
          typename cd_dtype_t,
          typename epilogue_type_t>
__device__ __noinline__ void
    linear_fp8_sm100_task_impl(int *grouped_layout,
                               uint32_t shape_m,
                               uint32_t shape_n,
                               uint32_t shape_k,
                               cute::TmaDescriptor const &tensor_map_a,
                               cute::TmaDescriptor const &tensor_map_b,
                               cute::TmaDescriptor const &tensor_map_sfa,
                               cute::TmaDescriptor const &tensor_map_sfb,
                               cute::TmaDescriptor const &tensor_map_residual,
                               cute::TmaDescriptor const &tensor_map_cd) {
  using namespace mirage::blackwell::linear_fp8_sm100_detail;

#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) ||                    \
    defined(__CLION_IDE__)
  using Barrier = cutlass::arch::ClusterTransactionBarrier;
  using Allocator = cute::conditional_t<kNumMulticast == 1,
                                        cute::TMEM::Allocator1Sm,
                                        cute::TMEM::Allocator2Sm>;

  if constexpr (kWithAccumulation) {
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, float>,
                     "Invalid C/D data dtype");
  }
  if constexpr (kWithResidual) {
    DG_STATIC_ASSERT(cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>,
                     "Residual fusion requires BF16 output");
  }

  constexpr uint32_t LAYOUT_AD_M = 128;
  constexpr uint32_t WAVE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
  constexpr uint32_t kNumMWaves = BLOCK_M / WAVE_BLOCK_M;
  constexpr uint32_t kNumTMAStoreStages = kWithResidual ? 1 : 2;
  constexpr uint32_t kNumUTCCPAlignedElems = 128;
  DG_STATIC_ASSERT(BLOCK_K == 128, "Invalid block K");
  DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0 && 2 % kNumMWaves == 0,
                   "Invalid block M");

  constexpr uint32_t kNumSFAStagesPerLoad = kGranKA == 32 ? 1 : 4;
  constexpr uint32_t kNumSFBStagesPerLoad = kGranKB == 32 ? 1 : 4;
  DG_STATIC_ASSERT(kGranKA == 32 || kGranKA == 128,
                   "Invalid granularity K for A");
  DG_STATIC_ASSERT(kGranKB == 32 || kGranKB == 128,
                   "Invalid granularity K for B");

  shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
  shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
  shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;
  const uint32_t shape_sfa_k = ceil_div(shape_k, kGranKA * 4);
  const uint32_t shape_sfb_k = ceil_div(shape_k, kGranKB * 4);

  bool is_leader_cta = cute::block_rank_in_cluster() == 0;
  auto const warp_idx = cutlass::canonical_warp_idx_sync();
  auto const lane_idx = get_lane_idx();

  extern __shared__ __align__(1024) uint8_t smem_buffer[];

  constexpr uint32_t LOAD_BLOCK_M =
      BLOCK_M / (kIsMulticastOnA ? kNumMulticast : 1);
  constexpr uint32_t LOAD_BLOCK_N =
      BLOCK_N / (kIsMulticastOnA ? 1 : kNumMulticast);
  constexpr uint32_t STORE_BLOCK_M = cute::min<uint32_t>(BLOCK_M, LAYOUT_AD_M);
  constexpr uint32_t STORE_BLOCK_N = kSwizzleCDMode / sizeof(cd_dtype_t);
  constexpr uint32_t kNumUMMAStoreThreads = STORE_BLOCK_M;
  DG_STATIC_ASSERT(!kIsMulticastOnA || kNumMulticast == 1, "Invalid multicast");
  DG_STATIC_ASSERT(LOAD_BLOCK_M == BLOCK_M,
                   "Only support tensor memory layout A/D");
  DG_STATIC_ASSERT(kNumMulticast == 1 || kNumMulticast == 2,
                   "Only support 1/2 multicast");
  DG_STATIC_ASSERT(kNumUMMAStoreThreads % 32 == 0, "Invalid store block M");

  // TMA box dims are clamped to min(BLOCK, globalDim) in the TMA descriptor.
  // arrive_and_expect_tx must match the actual TMA transfer size, not SMEM
  // size.
  constexpr uint32_t kTmaBoxM_A =
      (SHAPE_M != 0) ? cute::min<uint32_t>(LOAD_BLOCK_M, SHAPE_M)
                     : LOAD_BLOCK_M;
  constexpr uint32_t kAlignedShapeM =
      (SHAPE_M != 0) ? ((SHAPE_M + 3u) / 4u) * 4u : 0u;
  constexpr uint32_t kTmaBoxM_SFA =
      (SHAPE_M != 0 && kAlignedShapeM < BLOCK_M) ? kAlignedShapeM : BLOCK_M;
  // CD/residual TMA box: clamped batch dim
  constexpr uint32_t kTmaBoxM_CD =
      (SHAPE_M != 0) ? cute::min<uint32_t>(STORE_BLOCK_M, SHAPE_M)
                     : STORE_BLOCK_M;
  constexpr uint32_t kTmaResidualBytes = kSwizzleCDMode * kTmaBoxM_CD;

  constexpr uint32_t SMEM_CD_SIZE_PER_STAGE = STORE_BLOCK_M * kSwizzleCDMode;
  constexpr uint32_t SMEM_CD_SIZE = SMEM_CD_SIZE_PER_STAGE * kNumTMAStoreStages;
  constexpr uint32_t SMEM_RESIDUAL_SIZE =
      kWithResidual ? STORE_BLOCK_M * kSwizzleCDMode : 0;
  constexpr uint32_t SMEM_A_SIZE_PER_STAGE =
      LOAD_BLOCK_M * BLOCK_K * sizeof(a_dtype_t);
  constexpr uint32_t SMEM_B_SIZE_PER_STAGE =
      LOAD_BLOCK_N * BLOCK_K * sizeof(b_dtype_t);
  constexpr uint32_t SF_BLOCK_M =
      constexpr_align(BLOCK_M, kNumUTCCPAlignedElems);
  constexpr uint32_t SF_BLOCK_N =
      constexpr_align(BLOCK_N, kNumUTCCPAlignedElems);
  constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t);
  constexpr uint32_t SMEM_SFB_SIZE_PER_STAGE = SF_BLOCK_N * sizeof(uint32_t);
  DG_STATIC_ASSERT(SMEM_CD_SIZE % 1024 == 0 &&
                       SMEM_A_SIZE_PER_STAGE % 1024 == 0 &&
                       SMEM_B_SIZE_PER_STAGE % 1024 == 0,
                   "Shared memory of A/B must be aligned to 1024 bytes");
  DG_STATIC_ASSERT(kNumTMAStoreStages >= 1, "Invalid number of TMA stages");

  static constexpr uint32_t UMMA_A_SIZE_PER_STAGE =
      constexpr_align(LOAD_BLOCK_M, LAYOUT_AD_M) * BLOCK_K * sizeof(a_dtype_t);
  DG_STATIC_ASSERT(UMMA_A_SIZE_PER_STAGE <=
                       SMEM_A_SIZE_PER_STAGE +
                           SMEM_B_SIZE_PER_STAGE * kNumStages,
                   "Memory Out of bound for UMMA");

  constexpr uint32_t kNumSFATmemCols = SF_BLOCK_M / 32;
  constexpr uint32_t kNumSFBTmemCols = SF_BLOCK_N / 32;
  constexpr uint32_t kNumEpilogueStages =
      (2 * kNumMWaves * BLOCK_N + kNumSFATmemCols + kNumSFBTmemCols) > 512 ? 1
                                                                           : 2;

  constexpr uint32_t kNumAccumTmemCols =
      kNumEpilogueStages * kNumMWaves * BLOCK_N;
  constexpr uint32_t kNumTmemCols =
      get_num_aligned_tmem_cols<kNumAccumTmemCols + kNumSFATmemCols +
                                kNumSFBTmemCols>();
  constexpr uint32_t kTmemStartColOfSFA = kNumAccumTmemCols;
  constexpr uint32_t kTmemStartColOfSFB = kNumAccumTmemCols + kNumSFATmemCols;

  if (warp_idx == 0 && cute::elect_one_sync()) {
    cute::prefetch_tma_descriptor(&tensor_map_a);
    cute::prefetch_tma_descriptor(&tensor_map_b);
    cute::prefetch_tma_descriptor(&tensor_map_sfa);
    cute::prefetch_tma_descriptor(&tensor_map_sfb);
    if constexpr (kWithResidual) {
      cute::prefetch_tma_descriptor(&tensor_map_residual);
    }
    cute::prefetch_tma_descriptor(&tensor_map_cd);
  }

  auto smem_cd = PatternVisitor([&](uint32_t const &i) {
    return reinterpret_cast<cd_dtype_t *>(smem_buffer +
                                          i * SMEM_CD_SIZE_PER_STAGE);
  });
  if constexpr (kWithResidual) {
    DG_STATIC_ASSERT(
        kNumMWaves == 1 && BLOCK_N == STORE_BLOCK_N,
        "Fused residual currently supports a single CD store tile");
  }
  auto smem_residual = reinterpret_cast<cd_dtype_t *>(
      smem_buffer + kNumTMAStoreStages * SMEM_CD_SIZE_PER_STAGE);
  auto smem_a = PatternVisitor([&](uint32_t const &i) {
    return reinterpret_cast<a_dtype_t *>(smem_buffer + SMEM_CD_SIZE +
                                         SMEM_RESIDUAL_SIZE +
                                         i * SMEM_A_SIZE_PER_STAGE);
  });
  auto smem_b = PatternVisitor([&](uint32_t const &i) {
    return reinterpret_cast<b_dtype_t *>(
        smem_buffer + SMEM_CD_SIZE + SMEM_RESIDUAL_SIZE +
        kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
  });

  auto sf_start_ptr =
      smem_buffer + SMEM_CD_SIZE + SMEM_RESIDUAL_SIZE +
      kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
  auto smem_sfa = PatternVisitor([=](uint32_t const &i) {
    return reinterpret_cast<uint32_t *>(sf_start_ptr +
                                        i * SMEM_SFA_SIZE_PER_STAGE);
  });
  auto smem_sfb = PatternVisitor([=](uint32_t const &i) {
    return reinterpret_cast<uint32_t *>(sf_start_ptr +
                                        kNumStages * SMEM_SFA_SIZE_PER_STAGE +
                                        i * SMEM_SFB_SIZE_PER_STAGE);
  });

  auto barrier_start_ptr = reinterpret_cast<Barrier *>(
      smem_buffer + SMEM_CD_SIZE + SMEM_RESIDUAL_SIZE +
      kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE) +
      kNumStages * (SMEM_SFA_SIZE_PER_STAGE + SMEM_SFB_SIZE_PER_STAGE));
  auto full_barriers = PatternVisitor(
      [=](uint32_t const &i) { return barrier_start_ptr + (i); });
  auto empty_barriers = PatternVisitor(
      [=](uint32_t const &i) { return barrier_start_ptr + (kNumStages + i); });
  auto with_sf_full_barriers = PatternVisitor([=](uint32_t const &i) {
    return barrier_start_ptr + (kNumStages * 2 + i);
  });
  auto tmem_full_barriers = PatternVisitor([=](uint32_t const &i) {
    return barrier_start_ptr + (kNumStages * 3 + i);
  });
  auto tmem_empty_barriers = PatternVisitor([=](uint32_t const &i) {
    return barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages + i);
  });
  auto residual_full_barrier =
      barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages * 2);
  auto residual_empty_barrier =
      barrier_start_ptr + (kNumStages * 3 + kNumEpilogueStages * 2 + 1);

  auto tmem_ptr_in_smem = reinterpret_cast<uint32_t *>(
      barrier_start_ptr + kNumStages * 3 + kNumEpilogueStages * 2 +
      (kWithResidual ? 2 : 0));
  DG_STATIC_ASSERT(32 <= kNumTmemCols && kNumTmemCols <= 512,
                   "Invalid tensor memory columns");

  if (kNumMulticast > 1) {
    cute::cluster_sync();
  }

  if (warp_idx == 1 && cute::elect_one_sync()) {
#pragma unroll
    for (uint32_t i = 0; i < kNumStages; ++i) {
      full_barriers[i]->init(1);
      empty_barriers[i]->init(1);
      with_sf_full_barriers[i]->init(kNumMulticast * 32);
    }
#pragma unroll
    for (uint32_t i = 0; i < kNumEpilogueStages; ++i) {
      tmem_full_barriers[i]->init(1);
      tmem_empty_barriers[i]->init(kNumMulticast * kNumUMMAStoreThreads);
    }
    if constexpr (kWithResidual) {
      residual_full_barrier->init(1);
      residual_empty_barrier->init(1);
    }
    cutlass::arch::fence_barrier_init();
  } else if (warp_idx == 2) {
    Allocator().allocate(kNumTmemCols, tmem_ptr_in_smem);
  }
  kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();

  uint32_t m_block_idx, n_block_idx;
  auto scheduler =
      Scheduler<kGemmType,
                BLOCK_M,
                BLOCK_N,
                kNumGroups,
                kNumMulticast,
                kIsMulticastOnA,
                kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

  uint32_t stage_idx = 0, phase = 0;
  auto advance_pipeline = [&](uint32_t &k_block_idx) {
    ++k_block_idx;
    stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
    phase ^= stage_idx == 0;
  };

  if (warp_idx == 0 && cute::elect_one_sync()) {
    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
      auto const accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
      auto const accum_phase_idx =
          (scheduler.current_iter / kNumEpilogueStages) & 1;
      auto const &num_total_k_blocks =
          ceil_div(scheduler.current_shape_k, BLOCK_K);
      if constexpr (kWithResidual) {
        auto const residual_phase = scheduler.current_iter & 1;
        residual_empty_barrier->wait(residual_phase ^ 1);
        uint32_t const m_idx = scheduler.template get_global_idx<
            (kGemmType == GemmType::MGroupedMasked),
            IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
        uint32_t const n_idx = epilogue_type_t::template apply_index_n<BLOCK_N>(
            n_block_idx * BLOCK_N);
        tma_copy<BLOCK_N, BLOCK_M, kSwizzleCDMode, cd_dtype_t>(
            &tensor_map_residual,
            residual_full_barrier,
            smem_residual,
            n_idx,
            m_idx);
        residual_full_barrier->arrive_and_expect_tx(kTmaResidualBytes);
      }
      for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks;
           advance_pipeline(k_block_idx)) {
        empty_barriers[stage_idx]->wait(phase ^ 1);

        uint32_t m_idx = scheduler.template get_global_idx<
            (kGemmType == GemmType::MGroupedMasked),
            IndexType::MN>(shape_m, BLOCK_M, m_block_idx);
        uint32_t n_idx =
            scheduler.template get_global_idx<(kMajorB == cute::UMMA::Major::K),
                                              IndexType::MN>(
                shape_n, BLOCK_N, n_block_idx, m_block_idx);

        uint32_t k_idx = k_block_idx * BLOCK_K;
        uint32_t k_a_idx =
            scheduler
                .template get_global_idx<(kMajorA == cute::UMMA::Major::MN),
                                         IndexType::K>(
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);
        uint32_t k_b_idx =
            scheduler
                .template get_global_idx<(kMajorB == cute::UMMA::Major::MN),
                                         IndexType::K>(
                    shape_k, BLOCK_K, k_block_idx, m_block_idx);

        constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
        const uint32_t batch_idx =
            (kIsBatchedMM ? scheduler.current_group_idx : 0);
        if constexpr (kMajorA == cute::UMMA::Major::K) {
          tma_copy<BLOCK_K,
                   LOAD_BLOCK_M,
                   kSwizzleAMode,
                   a_dtype_t,
                   kIsBatchedMM>(&tensor_map_a,
                                 full_barriers[stage_idx],
                                 smem_a[stage_idx],
                                 k_a_idx,
                                 m_idx,
                                 1,
                                 batch_idx);
        }
        if constexpr (kMajorB == cute::UMMA::Major::K) {
          tma_copy<BLOCK_K,
                   LOAD_BLOCK_N,
                   kSwizzleBMode,
                   b_dtype_t,
                   kIsBatchedMM>(&tensor_map_b,
                                 full_barriers[stage_idx],
                                 smem_b[stage_idx],
                                 k_b_idx,
                                 n_idx,
                                 1,
                                 batch_idx);
        }
        // Actual TMA bytes: use clamped box dims (min(BLOCK, globalDim))
        auto num_arrival_bytes =
            BLOCK_K * kTmaBoxM_A * sizeof(a_dtype_t) +
            SMEM_B_SIZE_PER_STAGE /
                (std::is_same_v<b_dtype_t, cutlass::float_e4m3_t> ? 1 : 2);

        if (k_block_idx % kNumSFAStagesPerLoad == 0) {
          tma_copy<BLOCK_M, 1, 0>(
              &tensor_map_sfa,
              full_barriers[stage_idx],
              smem_sfa[stage_idx],
              m_block_idx * BLOCK_M,
              scheduler.template get_global_idx<(!is_m_grouped_contiguous(
                                                    kGemmType)),
                                                IndexType::SF_K>(
                  shape_sfa_k,
                  1,
                  ceil_div(k_idx, BLOCK_K * kNumSFAStagesPerLoad)));
          num_arrival_bytes += kTmaBoxM_SFA * sizeof(uint32_t);
        }
        if (k_block_idx % kNumSFBStagesPerLoad == 0) {
          tma_copy<BLOCK_N, 1, 0>(
              &tensor_map_sfb,
              full_barriers[stage_idx],
              smem_sfb[stage_idx],
              n_block_idx * BLOCK_N,
              scheduler.template get_global_idx<true, IndexType::SF_K>(
                  shape_sfb_k,
                  1,
                  ceil_div(k_idx, BLOCK_K * kNumSFBStagesPerLoad),
                  m_block_idx));
          num_arrival_bytes += BLOCK_N * sizeof(uint32_t);
        }

        full_barriers[stage_idx]->arrive_and_expect_tx(num_arrival_bytes);
      }
    }
  } else if (warp_idx == 1 && is_leader_cta) {
    constexpr uint32_t UMMA_M =
        LAYOUT_AD_M * (kIsMulticastOnA ? 1 : kNumMulticast);
    constexpr uint32_t UMMA_N = BLOCK_N * (kIsMulticastOnA ? kNumMulticast : 1);
    constexpr uint32_t UMMA_K = 32;
    auto instr_desc =
        cute::UMMA::make_instr_desc_block_scaled<a_dtype_t,
                                                 b_dtype_t,
                                                 float,
                                                 cutlass::float_ue8m0_t,
                                                 UMMA_M,
                                                 UMMA_N,
                                                 kMajorA,
                                                 kMajorB>();
    auto sf_desc = make_sf_desc(nullptr);

    DG_STATIC_ASSERT(kNumStages <= 32, "Too many stages");
    auto a_desc = make_umma_desc<kMajorA, LOAD_BLOCK_M, BLOCK_K, kSwizzleAMode>(
        smem_a[0], 0, 0);
    auto b_desc = make_umma_desc<kMajorB, LOAD_BLOCK_N, BLOCK_K, kSwizzleBMode>(
        smem_b[0], 0, 0);
    uint32_t a_desc_lo = lane_idx < kNumStages
                             ? a_desc.lo + lane_idx * SMEM_A_SIZE_PER_STAGE / 16
                             : 0u;
    uint32_t b_desc_lo = lane_idx < kNumStages
                             ? b_desc.lo + lane_idx * SMEM_B_SIZE_PER_STAGE / 16
                             : 0u;

    DG_STATIC_ASSERT(
        (UMMA_M == 64 && UMMA_N % 8 == 0 && 8 <= UMMA_N && UMMA_N <= 256) ||
            (UMMA_M == 128 && UMMA_N % 16 == 0 && 16 <= UMMA_N &&
             UMMA_N <= 256) ||
            (UMMA_M == 256 && UMMA_N % 16 == 0 && 16 <= UMMA_N &&
             UMMA_N <= 256),
        "Invalid MMA instruction shape");

    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
      auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
      auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;
      tmem_empty_barriers[accum_stage_idx]->wait(accum_phase_idx ^ 1);
      tcgen05_after_thread_sync();

      auto empty_barrier_arrive = [&](bool const &do_tmem_full_arrive) {
        auto umma_arrive = [](const uint64_t *barrier) {
          cutlass::arch::umma_arrive(barrier);
        };
        umma_arrive(reinterpret_cast<uint64_t *>(empty_barriers[stage_idx]));
        if (do_tmem_full_arrive) {
          umma_arrive(reinterpret_cast<uint64_t *>(
              tmem_full_barriers[accum_stage_idx]));
        }
      };

      auto const &num_total_k_blocks =
          ceil_div(scheduler.current_shape_k, BLOCK_K);
      for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks;
           advance_pipeline(k_block_idx)) {
        with_sf_full_barriers[stage_idx]->wait(phase);
        tcgen05_after_thread_sync();

        using cute_utccp_t = cute::SM100_UTCCP_4x32dp128bit_1cta;
        const uint32_t sfa_stage_in_group_idx =
            k_block_idx % kNumSFAStagesPerLoad;
        if (sfa_stage_in_group_idx == 0 && cute::elect_one_sync()) {
#pragma unroll
          for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++i) {
            auto smem_ptr = smem_sfa[stage_idx] + i * kNumUTCCPAlignedElems;
            replace_smem_desc_addr(sf_desc, smem_ptr);
            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFA + i * 4);
          }
        }
        const uint32_t sfb_stage_in_group_idx =
            k_block_idx % kNumSFBStagesPerLoad;
        if (sfb_stage_in_group_idx == 0 && cute::elect_one_sync()) {
#pragma unroll
          for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++i) {
            auto smem_ptr = smem_sfb[stage_idx] + i * kNumUTCCPAlignedElems;
            replace_smem_desc_addr(sf_desc, smem_ptr);
            cute_utccp_t::copy(sf_desc, kTmemStartColOfSFB + i * 4);
          }
        }
        __syncwarp();

        using mma_t =
            mirage::blackwell::linear_fp8_sm100::sm100::SM100_MMA_MXF8F6F4_SS;
        auto const &a_desc_base_lo =
            __shfl_sync(0xffffffff, a_desc_lo, static_cast<int>(stage_idx));
        auto const &b_desc_base_lo =
            __shfl_sync(0xffffffff, b_desc_lo, static_cast<int>(stage_idx));
        if (cute::elect_one_sync()) {
#pragma unroll
          for (uint32_t k = 0; k < BLOCK_K / UMMA_K; ++k) {
            const uint32_t sfa_id =
                (kGranKA == 32 ? k : sfa_stage_in_group_idx);
            const uint32_t sfb_id =
                (kGranKB == 32 ? k : sfb_stage_in_group_idx);
            auto const &runtime_instr_desc =
                make_runtime_instr_desc_with_sf_id(instr_desc, sfa_id, sfb_id);

            b_desc.lo =
                advance_umma_desc_lo<kMajorB,
                                     LOAD_BLOCK_N,
                                     kSwizzleBMode,
                                     b_dtype_t>(b_desc_base_lo, 0, k * UMMA_K);
#pragma unroll
            for (uint32_t w = 0; w < kNumMWaves; ++w) {
              DG_STATIC_ASSERT((WAVE_BLOCK_M * BLOCK_K) % 128 == 0,
                               "Invalid swizzling offset");
              a_desc.lo = advance_umma_desc_lo<kMajorA,
                                               LOAD_BLOCK_M,
                                               kSwizzleAMode,
                                               a_dtype_t>(
                  a_desc_base_lo, w * WAVE_BLOCK_M * BLOCK_K, k * UMMA_K);
              mma_t::fma(a_desc,
                         b_desc,
                         accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N,
                         k_block_idx > 0 || k > 0,
                         runtime_instr_desc,
                         kTmemStartColOfSFA + w * (kNumUTCCPAlignedElems / 32),
                         kTmemStartColOfSFB);
            }
          }
        }
        empty_barrier_arrive(k_block_idx == num_total_k_blocks - 1);
      }
    }

    auto const &iter_idx = scheduler.current_iter - 1;
    if (kNumMulticast > 1 && iter_idx >= 0) {
      auto const &accum_phase_idx = (iter_idx / kNumEpilogueStages) & 1;
      tmem_empty_barriers[iter_idx % kNumEpilogueStages]->wait(accum_phase_idx);
    }
  } else if (warp_idx == 2) {
    auto utccp_required_smem_warp_transpose = [&](uint32_t const *smem_ptr) {
      DG_STATIC_ASSERT(kNumUTCCPAlignedElems == 128,
                       "Invalid aligned elements");
      uint32_t values[4];
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
        values[i] = ld_shared(smem_ptr + (i ^ (lane_idx >> 3)) * 32 + lane_idx);
      __syncwarp();
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i)
        st_shared(smem_ptr + lane_idx * 4 + (i ^ (lane_idx >> 3)), values[i]);
    };

    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
      auto const &num_total_k_blocks =
          ceil_div(scheduler.current_shape_k, BLOCK_K);
      for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks;
           advance_pipeline(k_block_idx)) {
        full_barriers[stage_idx]->wait(phase);

        if (k_block_idx % kNumSFAStagesPerLoad == 0) {
#pragma unroll
          for (uint32_t i = 0; i < SF_BLOCK_M / kNumUTCCPAlignedElems; ++i) {
            utccp_required_smem_warp_transpose(smem_sfa[stage_idx] +
                                               i * kNumUTCCPAlignedElems);
          }
          cutlass::arch::fence_view_async_shared();
        }
        if (k_block_idx % kNumSFBStagesPerLoad == 0) {
#pragma unroll
          for (uint32_t i = 0; i < SF_BLOCK_N / kNumUTCCPAlignedElems; ++i) {
            utccp_required_smem_warp_transpose(smem_sfb[stage_idx] +
                                               i * kNumUTCCPAlignedElems);
          }
          cutlass::arch::fence_view_async_shared();
        }

        with_sf_full_barriers[stage_idx]->arrive(0u);
      }
    }
  } else if (warp_idx >= kNumNonEpilogueThreads / 32 &&
             warp_idx < (kNumNonEpilogueThreads + kNumUMMAStoreThreads) / 32) {
    auto const epilogue_warp_idx = warp_idx - (kNumNonEpilogueThreads / 32);
    DG_TRAP_ONLY_DEVICE_ASSERT(ld_shared(tmem_ptr_in_smem) == 0);

    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup =
        kNumBankGroupBytes / sizeof(cd_dtype_t);
    DG_STATIC_ASSERT(kSwizzleCDMode > 0, "TMA D must be swizzled");
    DG_STATIC_ASSERT(STORE_BLOCK_N % kNumElemsPerBankGroup == 0,
                     "Invalid swizzling");

    uint32_t tma_stage_idx = 0;
    auto advance_store_pipeline = [&]() {
      tma_stage_idx = (tma_stage_idx + 1) % kNumTMAStoreStages;
    };

    while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
      auto accum_stage_idx = scheduler.current_iter % kNumEpilogueStages;
      auto accum_phase_idx = (scheduler.current_iter / kNumEpilogueStages) & 1;

      tmem_full_barriers[accum_stage_idx]->wait(accum_phase_idx);
      if constexpr (kWithResidual) {
        auto const residual_phase = scheduler.current_iter & 1;
        residual_full_barrier->wait(residual_phase);
      }
      tcgen05_after_thread_sync();

      DG_STATIC_ASSERT(kNumEpilogueThreads == 128,
                       "Epilogue threads not enough");
      DG_STATIC_ASSERT(BLOCK_N % STORE_BLOCK_N == 0, "Invalid block sizes");

#pragma unroll
      for (uint32_t w = 0; w < kNumMWaves; ++w) {
        constexpr uint32_t kNumStores = BLOCK_N / STORE_BLOCK_N;
#pragma unroll
        for (uint32_t s = 0; s < kNumStores; ++s, advance_store_pipeline()) {
          if (epilogue_warp_idx == 0) {
            cute::tma_store_wait<kNumTMAStoreStages - 1>();
          }
          cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);

          auto const m_idx = scheduler.template get_global_idx<
                                 (!is_m_grouped_contiguous(kGemmType)),
                                 IndexType::MN>(shape_m, BLOCK_M, m_block_idx) +
                             w * WAVE_BLOCK_M;
          auto const n_idx =
              epilogue_type_t::template apply_index_n<STORE_BLOCK_N>(
                  n_block_idx * BLOCK_N + s * STORE_BLOCK_N);

#pragma unroll
          for (uint32_t i = 0; i < STORE_BLOCK_N / kNumElemsPerBankGroup; ++i) {
            auto bank_group_index =
                i + lane_idx * (kSwizzleCDMode / kNumBankGroupBytes);

            constexpr bool kHasShortcut =
                (kSwizzleCDMode / kNumBankGroupBytes) == 8;
            auto row =
                kHasShortcut ? (i / 8 + lane_idx) : (bank_group_index / 8);
            auto col = kHasShortcut ? (i) : (bank_group_index % 8);
            col ^= row % (kSwizzleCDMode / 16);

            uint32_t tmem_addr = accum_stage_idx * kNumMWaves * BLOCK_N +
                                 w * BLOCK_N + s * STORE_BLOCK_N +
                                 i * kNumElemsPerBankGroup;
            auto smem_ptr =
                reinterpret_cast<uint8_t *>(smem_cd[tma_stage_idx]) +
                epilogue_warp_idx * 32 * kSwizzleCDMode +
                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
            auto residual_smem_ptr =
                reinterpret_cast<uint8_t *>(smem_residual) +
                epilogue_warp_idx * 32 * kSwizzleCDMode +
                row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;

            uint32_t values[kNumElemsPerBankGroup];
            if constexpr (cute::is_same_v<cd_dtype_t, float>) {
              DG_STATIC_ASSERT(kNumElemsPerBankGroup == 4, "Invalid type");
              cute::SM100_TMEM_LOAD_32dp32b4x::copy(
                  tmem_addr, values[0], values[1], values[2], values[3]);
              cutlass::arch::fence_view_async_tmem_load();
              st_shared(smem_ptr, values[0], values[1], values[2], values[3]);
            } else {
              DG_STATIC_ASSERT(
                  kNumElemsPerBankGroup == 8 &&
                      cute::is_same_v<cd_dtype_t, cutlass::bfloat16_t>,
                  "Invalid type");
              cute::SM100_TMEM_LOAD_32dp32b8x::copy(tmem_addr,
                                                    values[0],
                                                    values[1],
                                                    values[2],
                                                    values[3],
                                                    values[4],
                                                    values[5],
                                                    values[6],
                                                    values[7]);
              cutlass::arch::fence_view_async_tmem_load();
              if constexpr (kWithResidual) {
                auto residual_words_ptr =
                    reinterpret_cast<uint32_t const *>(residual_smem_ptr);
                uint32_t residual_words[4];
                residual_words[0] = ld_shared(residual_words_ptr + 0);
                residual_words[1] = ld_shared(residual_words_ptr + 1);
                residual_words[2] = ld_shared(residual_words_ptr + 2);
                residual_words[3] = ld_shared(residual_words_ptr + 3);
                add_packed_bf16x2_into_fp32_bits(
                    residual_words[0], values[0], values[1]);
                add_packed_bf16x2_into_fp32_bits(
                    residual_words[1], values[2], values[3]);
                add_packed_bf16x2_into_fp32_bits(
                    residual_words[2], values[4], values[5]);
                add_packed_bf16x2_into_fp32_bits(
                    residual_words[3], values[6], values[7]);
              }
              st_shared(smem_ptr,
                        cast_into_bf16_and_pack(values[0], values[1]),
                        cast_into_bf16_and_pack(values[2], values[3]),
                        cast_into_bf16_and_pack(values[4], values[5]),
                        cast_into_bf16_and_pack(values[6], values[7]));
            }
          }

          if (w == kNumMWaves - 1 && s == BLOCK_N / STORE_BLOCK_N - 1) {
            tcgen05_before_thread_sync();
            tmem_empty_barriers[accum_stage_idx]->arrive(0u);
          }

          cute::tma_store_fence();
          cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
          if (epilogue_warp_idx == 0 && cute::elect_one_sync()) {
            if constexpr (kGemmType == GemmType::Batched) {
              using cute_tma_t =
                  cute::conditional_t<kWithAccumulation,
                                      cute::SM90_TMA_REDUCE_ADD_3D,
                                      cute::SM90_TMA_STORE_3D>;
              cute_tma_t::copy(&tensor_map_cd,
                               smem_cd[tma_stage_idx],
                               n_idx,
                               m_idx,
                               scheduler.current_group_idx);
            } else {
              using cute_tma_t =
                  cute::conditional_t<kWithAccumulation,
                                      cute::SM90_TMA_REDUCE_ADD_2D,
                                      cute::SM90_TMA_STORE_2D>;
              cute_tma_t::copy(
                  &tensor_map_cd, smem_cd[tma_stage_idx], n_idx, m_idx);
            }
            cute::tma_store_arrive();
          }
        }
      }
      if constexpr (kWithResidual) {
        cutlass::arch::NamedBarrier::sync(kNumUMMAStoreThreads, 0);
        if (epilogue_warp_idx == 0 && cute::elect_one_sync()) {
          residual_empty_barrier->arrive(0u);
        }
      }
    }
  }

  kNumMulticast > 1 ? cute::cluster_sync() : __syncthreads();
  if (warp_idx == 0) {
    Allocator().free(0, kNumTmemCols);
  }

#else
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    DG_DEVICE_ASSERT(false && "This kernel only support sm_100f");
  }
#endif
}

template <cute::UMMA::Major kMajorA,
          cute::UMMA::Major kMajorB,
          uint32_t kGranKA,
          uint32_t kGranKB,
          uint32_t SHAPE_M,
          uint32_t SHAPE_N,
          uint32_t SHAPE_K,
          uint32_t BLOCK_M,
          uint32_t BLOCK_N,
          uint32_t BLOCK_K,
          uint32_t kNumGroups,
          uint32_t kSwizzleAMode,
          uint32_t kSwizzleBMode,
          uint32_t kSwizzleCDMode,
          uint32_t kNumStages,
          uint32_t kNumNonEpilogueThreads,
          uint32_t kNumEpilogueThreads,
          uint32_t kNumMulticast,
          bool kIsMulticastOnA,
          uint32_t kNumSMs,
          bool kWithResidual,
          mirage::blackwell::linear_fp8_sm100::GemmType kGemmType,
          bool kWithAccumulation,
          typename a_dtype_t,
          typename b_dtype_t,
          typename cd_dtype_t,
          typename epilogue_type_t>
__global__ void __launch_bounds__(kNumNonEpilogueThreads + kNumEpilogueThreads,
                                  1)
    linear_fp8_sm100_wrapper(
        int *grouped_layout,
        uint32_t shape_m,
        uint32_t shape_n,
        uint32_t shape_k,
        const __grid_constant__ cute::TmaDescriptor tensor_map_a,
        const __grid_constant__ cute::TmaDescriptor tensor_map_b,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sfa,
        const __grid_constant__ cute::TmaDescriptor tensor_map_sfb,
        const __grid_constant__ cute::TmaDescriptor tensor_map_residual,
        const __grid_constant__ cute::TmaDescriptor tensor_map_cd) {
  linear_fp8_sm100_task_impl<kMajorA,
                             kMajorB,
                             kGranKA,
                             kGranKB,
                             SHAPE_M,
                             SHAPE_N,
                             SHAPE_K,
                             BLOCK_M,
                             BLOCK_N,
                             BLOCK_K,
                             kNumGroups,
                             kSwizzleAMode,
                             kSwizzleBMode,
                             kSwizzleCDMode,
                             kNumStages,
                             kNumNonEpilogueThreads,
                             kNumEpilogueThreads,
                             kNumMulticast,
                             kIsMulticastOnA,
                             kNumSMs,
                             kWithResidual,
                             kGemmType,
                             kWithAccumulation,
                             a_dtype_t,
                             b_dtype_t,
                             cd_dtype_t,
                             epilogue_type_t>(grouped_layout,
                                              shape_m,
                                              shape_n,
                                              shape_k,
                                              tensor_map_a,
                                              tensor_map_b,
                                              tensor_map_sfa,
                                              tensor_map_sfb,
                                              tensor_map_residual,
                                              tensor_map_cd);
}

} // namespace kernel

#pragma clang diagnostic pop
