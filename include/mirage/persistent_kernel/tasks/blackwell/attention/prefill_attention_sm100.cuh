#pragma once
#include <cstdio>
#include <iostream>

// Cutlass includes
#include <cutlass/half.h> // F16 data type
// #include <cutlass/util/print_error.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

// CuTe includes
#include <cute/algorithm/cooperative_copy.hpp> // Auto vectorized copy operation
#include <cute/arch/cluster_sm90.hpp> // CuTe functions for querying the details of cluster launched
#include <cute/arch/tmem_allocator_sm100.hpp> // TMEM allocator for SM100
#include <cute/numeric/integral_constant.hpp> // Compile time in constants such as _1, _256 etc.
#include <cute/tensor.hpp>                    // CuTe tensor implementation
// using namespace cute;

#include "../../hopper/barrier.cuh"
#include "../../hopper/smem_layout_tma.cuh"
#include "../../hopper/tma.cuh"
#include "../storage.cuh"

namespace kernel {

struct PrefillAttnSM100WarpConfig {
  enum class WarpRole { SoftmaxCorrEpi, MMA, Load, Empty };

  static constexpr WarpRole warp_idx_to_WarpRole(int warp_idx) {
    if (warp_idx < 4) return WarpRole::SoftmaxCorrEpi;          //   0 -  3
    if (warp_idx == 4) return WarpRole::MMA;                    //   4
    if (warp_idx == 5) return WarpRole::Load;                   //   5
    return WarpRole::Empty;                                     //   7
  }

  static const int NumWarpsSoftmaxCorrEpi = 4;
  static const int NumWarpsLoad = 1;
  static const int NumWarpsMMA = 1;

  static const int NumWarps = 8;
};

template <typename T_,
          class QTensor,
          typename TMA_K,
          typename TMA_V,
          class OTensor,
          int NUM_QO_HEADS,
          int NUM_KV_HEADS,
          int HEAD_DIM_QK,
          int HEAD_DIM_V,
          int PAGE_SIZE,
          int Q_TILE_SIZE,
          int KV_TILE_SIZE,
          int NUM_Q_STAGE=2,
          int NUM_KV_STAGE=3,
          int NUM_ACC_STAGE=1,
          int NUM_EPI_STAGE=2,
          class WarpConfig = PrefillAttnSM100WarpConfig>
__device__ __noinline__ void
    prefill_attn_sm100(
      QTensor mQ,
      const TMA_K &tma_k,
      const TMA_V &tma_v,
      OTensor mO,
      int const *prefill_work_indptr,
      int const *worker_batch_indices,
      int const *worker_kv_head_indices,
      int const *worker_packed_qo_indices,
      int const *worker_kv_start,
      int const *worker_kv_end,
      int const *paged_kv_indptr_buffer_ptr,
      int const *paged_kv_indices_buffer_ptr,
      int const *paged_kv_last_page_len_buffer_ptr,
      int worker_idx) {
    int work_num = prefill_work_indptr[worker_idx + 1] - prefill_work_indptr[worker_idx];
    if (work_num == 0) {
        return;
    }

    int warp_idx = cutlass::canonical_warp_idx_sync();

    // We use the (128, 128) MMA size for prefill attention computation
    int const MMA_M = 128;
    int const MMA_N = 128;

    static_assert(HEAD_DIM_QK == 128, "Only HEAD_DIM_QK of 128 is supported for SM100 prefill attention kernel.");
    static_assert(HEAD_DIM_V == 128, "Only HEAD_DIM_V of 128 is supported for SM100 prefill attention kernel.");

    if (cute::thread0()){
        cute::print("mQ shape:\n"); cute::print(mQ); cute::print("\n");
        cute::print("mO shape:\n"); cute::print(mO); cute::print("\n");
    } __syncthreads();

    auto mma_tiler_qk = cute::make_shape(MMA_M, MMA_N, HEAD_DIM_QK); // (MMA_M, MMA_N, MMA_K)
    auto mma_tiler_pv = cute::make_shape(MMA_M, HEAD_DIM_V, MMA_N); // (MMA_M, MMA_N, MMA_K)

    // Q and K both in SMEM with K major layout
    cute::TiledMMA tiled_mma_qk = cute::make_tiled_mma(
      cute::SM100_MMA_F16BF16_SS<T_,
                                 T_,
                                 float, // Mma's Q, K, and Accumulator types
                                 MMA_M,
                                 MMA_N, // Mma M and N dimensions
                                 cute::UMMA::Major::K,
                                 cute::UMMA::Major::K>{}); // Q and K tiles layouts
    
    // P in TMEM, V in SMEM with K major layout
    cute::TiledMMA tiled_mma_pv = cute::make_tiled_mma(
      cute::SM100_MMA_F16BF16_TS<T_,
                                 T_,
                                 float, // Mma's P, V, and Accumulator types
                                 MMA_M,
                                 HEAD_DIM_V, // Mma M and N dimensions
                                 cute::UMMA::Major::K,
                                 cute::UMMA::Major::MN>{}); // P and V tiles layouts

    auto mma_shape_Q =
      cute::partition_shape_A(tiled_mma_qk,
                              cute::make_shape(cute::Int<MMA_M>{},
                                               cute::Int<HEAD_DIM_QK>{},
                                               cute::Int<NUM_Q_STAGE>{}));

    auto mma_shape_K =
      cute::partition_shape_B(tiled_mma_qk,
                              cute::make_shape(cute::Int<MMA_N>{},
                                               cute::Int<HEAD_DIM_QK>{},
                                               cute::Int<NUM_KV_STAGE>{}));

    auto mma_shape_P =
      cute::partition_shape_A(tiled_mma_pv,
                              cute::make_shape(cute::Int<MMA_M>{},
                                               cute::Int<MMA_N>{},
                                               cute::Int<NUM_ACC_STAGE>{}));
    
    auto mma_shape_V =
      cute::partition_shape_B(tiled_mma_pv,
                              cute::make_shape(cute::Int<HEAD_DIM_V>{},
                                               cute::Int<MMA_N>{},
                                               cute::Int<NUM_KV_STAGE>{}));
    
    // TODO: figure out the correct O shape
    auto mma_shape_O =
      cute::partition_shape_C(tiled_mma_pv,
                              cute::make_shape(cute::Int<MMA_M>{},
                                               cute::Int<HEAD_DIM_V>{},
                                               cute::Int<NUM_EPI_STAGE>{}));

    auto sQ_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_Q);
    auto sK_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_K);
    auto tP_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_P);
    auto sV_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_V);
    auto sO_layout = cute::UMMA::tile_to_mma_shape(
      cute::UMMA::Layout_K_SW128_Atom<T_>{}, mma_shape_O);


    using SharedStorage = PrefillAttnStorage<T_,
                                           T_,
                                           decltype(sO_layout),
                                           decltype(sQ_layout),
                                           decltype(sK_layout),
                                           Q_TILE_SIZE,
                                           NUM_Q_STAGE,
                                           NUM_KV_STAGE,
                                           NUM_EPI_STAGE>;

    extern __shared__ char shared_memory[];
    uintptr_t aligned_smem =
      (reinterpret_cast<uintptr_t>(shared_memory) + 127) / 128 * 128;
    SharedStorage &shared_storage =
      *reinterpret_cast<SharedStorage *>(aligned_smem);
    
    
    if (cute::thread0()){
        printf("size of SharedStorage: %lu bytes\n", sizeof(SharedStorage));
        cute::print("tiled_mma_qk:\n"); cute::print(tiled_mma_qk); cute::print("\n");
        cute::print("tiled_mma_pv:\n"); cute::print(tiled_mma_pv); cute::print("\n");
        cute::print("mma_shape_Q:\n"); cute::print(mma_shape_Q); cute::print("\n");
        cute::print("mma_shape_K:\n"); cute::print(mma_shape_K); cute::print("\n");
        cute::print("mma_shape_P:\n"); cute::print(mma_shape_P); cute::print("\n");
        cute::print("mma_shape_V:\n"); cute::print(mma_shape_V); cute::print("\n");
        cute::print("mma_shape_O:\n"); cute::print(mma_shape_O); cute::print("\n");
        cute::print("sQ_layout:\t"); cute::print(sQ_layout); cute::print("\n");
        cute::print("sK_layout:\t"); cute::print(sK_layout); cute::print("\n");
        cute::print("sV_layout:\t"); cute::print(sV_layout); cute::print("\n");
        cute::print("tP_layout:\t"); cute::print(tP_layout); cute::print("\n");
        cute::print("sO_layout:\t"); cute::print(sO_layout); cute::print("\n");
    } __syncthreads();

    // mbar init with first N warps
    if (warp_idx == 0){
      cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        NUM_Q_STAGE>(shared_storage.load_q_full_mbar_ptr, /* arrival count */ WarpConfig::NumWarpsLoad);
      cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        NUM_Q_STAGE>(shared_storage.load_q_empty_mbar_ptr, /* arrival count */ WarpConfig::NumWarpsMMA);
    } else if (warp_idx == 1){
      cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        NUM_EPI_STAGE>(shared_storage.softmax_corr_empty_mbar_ptr, /* arrival count */ WarpConfig::NumWarpsSoftmaxCorrEpi * 32);
      cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        2>(shared_storage.softmax_corr_full_mbar_ptr, /* arrival count */ WarpConfig::NumWarpsSoftmaxCorrEpi * 32);
    } else if (warp_idx == 2){
      cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        8>(shared_storage.s0_s1_sequence_mbar_ptr, /* arrival count */ 32);
    } else if (warp_idx == 3){ // probably not needed as we are using the same warp group for Softmax+Corr+Epi
      cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        NUM_EPI_STAGE>(shared_storage.corr_epi_full_mbar_ptr, /* arrival count */ WarpConfig::NumWarpsSoftmaxCorrEpi * 32);
      cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        2>(shared_storage.corr_epi_empty_mbar_ptr, /* arrival count */ WarpConfig::NumWarpsSoftmaxCorrEpi * 32);
    } else if (warp_idx == 4){
      cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        2>(shared_storage.P_full_O_rescaled_mbar_ptr, /* arrival count */ WarpConfig::NumWarpsSoftmaxCorrEpi * 32);
      cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        2>(shared_storage.S_full_mbar_ptr, /* arrival count */ WarpConfig::NumWarpsMMA);
      cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        2>(shared_storage.O_full_mbar_ptr, /* arrival count */ WarpConfig::NumWarpsMMA);
    } else if (warp_idx == 5){
      cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        2>(shared_storage.P_full_2_mbar_ptr, /* arrival count */ WarpConfig::NumWarpsSoftmaxCorrEpi * 32);
    } else if (warp_idx == 6){
      cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        NUM_KV_STAGE>(shared_storage.load_kv_full_mbar_ptr, /* arrival count */ WarpConfig::NumWarpsLoad);
      cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        NUM_KV_STAGE>(shared_storage.load_kv_empty_mbar_ptr, /* arrival count */ WarpConfig::NumWarpsMMA);
    } else if (warp_idx == 7){
      cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier,
        1>(shared_storage.tmem_dealloc_mbar_ptr, /* arrival count */ WarpConfig::NumWarpsSoftmaxCorrEpi * 32);
    }
    cutlass::arch::fence_barrier_init();

    cute::Tensor sQ = shared_storage.tensor_sQ();
    cute::Tensor sK = shared_storage.tensor_sK();
    cute::Tensor sV = shared_storage.tensor_sV();
    cute::Tensor sO = shared_storage.tensor_sO();

    cute::Tensor sScale = shared_storage.tensor_sScale();

    cute::ThrMMA thr_mma_qk = tiled_mma_qk.get_slice(0); // 1SM
    cute::ThrMMA thr_mma_pv = tiled_mma_pv.get_slice(0); // 1SM

    auto qk_acc_shape =
      cute::partition_shape_C(tiled_mma_qk,
                              cute::make_shape(cute::Int<MMA_M>{},
                                               cute::Int<MMA_N>{},
                                              cute::Int<2>{}));
    cute::Tensor tStS = tiled_mma_qk.make_fragment_C(qk_acc_shape);

    auto pv_acc_shape =
      cute::partition_shape_C(tiled_mma_pv,
                              cute::make_shape(cute::Int<MMA_M>{},
                                               cute::Int<HEAD_DIM_V>{},
                                               cute::Int<NUM_Q_STAGE>{}));
    cute::Tensor tOtO_fake = tiled_mma_pv.make_fragment_C(pv_acc_shape);
    cute::Tensor tOtO = cute::make_tensor(tOtO_fake.data() + 256, tOtO_fake.layout()); // half tmem (256 columns) for S and half for O 
    
    cute::Tensor tP = cute::make_tensor(tStS.data(), tP_layout.layout_b());
    cute::Tensor tOrP = thr_mma_pv.make_fragment_A(tP)(cute::_, cute::_, cute::_, 0);

    if (cute::thread0()){
        cute::print("tStS shape:\n"); cute::print(tStS); cute::print("\n");
        cute::print("tOtO shape:\n"); cute::print(tOtO); cute::print("\n");
        cute::print("tP shape:\n"); cute::print(tP); cute::print("\n");
        cute::print("tOrP shape:\n"); cute::print(tOrP); cute::print("\n");
    } __syncthreads();

    using TmemAllocator = cute::TMEM::Allocator1Sm;
    TmemAllocator tmem_allocator{};
    
    __syncthreads();

    /*
     *********************
     * 
     *       LOAD
     * 
     *********************
     */

    if (WarpConfig::warp_idx_to_WarpRole(warp_idx) == WarpConfig::WarpRole::Load) {
      
    }

    
    
  } // end prefill_attn_sm100

} // namespace kernel