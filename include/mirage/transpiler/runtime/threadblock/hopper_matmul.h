// matmul.h - Implementation of threadblock level matmul operators
//
// WARNING To understand code in this file, you have to be familiar with the
// CuTe library. The CuTe library is a part of the Cutlass library by NVIDIA
// which provides (not so) easy-to-use primitives for writing high-performance
// kernels. You may read the CuTe documentation at
// https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md
// or you can find some amazing blogs in Chinese here:
// https://zhuanlan.zhihu.com/p/661182311
//

#pragma once

#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <threadblock/shared_reg_copy.h>
using namespace cute;

#include "element_unary.h"

namespace tb {

template <class T, class BLK_MN0>
CUTE_HOST_DEVICE constexpr auto smem_layout_selector() {
  if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_SW128_Atom<T>{}) == 0) {
    return GMMA::Layout_MN_SW128_Atom<T>{};
  } else if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_SW64_Atom<T>{}) == 0) {
    return GMMA::Layout_MN_SW64_Atom<T>{};
  } else if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_SW32_Atom<T>{}) == 0) {
    return GMMA::Layout_MN_SW32_Atom<T>{};
  } else if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_INTER_Atom<T>{}) ==
                       0) {
    return GMMA::Layout_MN_INTER_Atom<T>{};
  } else {
    static_assert(BLK_MN0 % size<0>(GMMA::Layout_MN_INTER_Atom<T>{}) == 0,
                  "BLK_MN0 must be a multiple of "
                  "size<0>(GMMA::Layout_MN_INTER_Atom<T>{})");
  }
}

template <typename T,
          class MMAAtom,
          class TiledMMAThrLayout,
          bool IS_LDMATRIX_AVAIL,
          bool IS_STMATRIX_AVAIL,
          class SmemLayoutA_, // [K, M]
          class SmemLayoutB_, // [N, K]
          class SmemLayoutC_, // [N, M]
          int NUM_THREADS,
          int NUM_EXPS_BEFORE_STORE, // Since matmul may use some advanced
                                     // instructions (like stmatrix) to store
                                     // data, it does not use the standard
                                     // "epilogue" semantic
          bool IS_STORE_ACCUM>
class Hopper_Matmul {
public:
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutA_{}) == _2{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutB_{}) == _2{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutC_{}) == _2{});

  using SmemLayoutA = typename Dim01Swapper<SmemLayoutA_>::Result; // [M, K]
  using SmemLayoutB = SmemLayoutB_;                                // [N, K]
  using SmemLayoutC = typename Dim01Swapper<SmemLayoutC_>::Result; // [M, N]

  using TileALayout = smem_layout_selector<SmemLayoutA>;

  using TileBLayout = smem_layout_selector<SmemLayoutB>;

  // Shape checking
  // Expect A have a shape of [M, K], B have a shape of [N, K], and
  // C have a shape of [M, N]
  using M = decltype(get<0>(shape(SmemLayoutA{})));
  using K = decltype(get<1>(shape(SmemLayoutA{})));
  using N = decltype(get<0>(shape(SmemLayoutB{})));
  CUTE_STATIC_ASSERT_V(K{} == get<1>(shape(SmemLayoutB{})));
  CUTE_STATIC_ASSERT_V(M{} == get<0>(shape(SmemLayoutC{})));
  CUTE_STATIC_ASSERT_V(N{} == get<1>(shape(SmemLayoutC{})));

  using TiledMMA = decltype(make_tiled_mma(
      SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{}));

  static constexpr int TILED_MMA_NUM_THREADS = thr_size(TiledMMA{});
  static_assert(TILED_MMA_NUM_THREADS <= NUM_THREADS);

  using R2STiledCopyCSelector =
      R2STiledCopySelector<T, IS_STMATRIX_AVAIL, SmemLayoutC>;
  using R2STiledCopyCAtom = typename R2STiledCopyCSelector::Result;
  static constexpr R2STiledCopyType R2S_TILED_COPY_C_TYPE =
      R2STiledCopyCSelector::TYPE;
  using R2STiledCopyC =
      decltype(make_tiled_copy_C(R2STiledCopyCAtom{}, TiledMMA{}));

  static __device__ __forceinline__ auto get_mma_rC(int thread_idx) {
    // Make a fake tensor
    Tensor sC_fake =
        make_tensor(make_smem_ptr((half_t *)nullptr), SmemLayoutC{});
    TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(thread_idx);
    Tensor mma_rC =
        thr_mma.partition_fragment_C(sC_fake); // (MMA, MMA_M, MMA_N)
    clear(mma_rC);
    return mma_rC;
  }

  template <class AccumRegFrag>
  static __device__ __forceinline__ void write_back_mma_rC(
      T *__restrict__ c_ptr, AccumRegFrag const &mma_rC, int thread_idx) {
    if (thread_idx >= TILED_MMA_NUM_THREADS) {
      return;
    }
    Tensor sC = make_tensor(make_smem_ptr(c_ptr), SmemLayoutC{}); // [M, N]
    R2STiledCopyC r2s_tiled_copy_C;
    ThrCopy r2s_tiled_copy_C_thr = r2s_tiled_copy_C.get_slice(thread_idx);
    Tensor r2s_rC =
        r2s_tiled_copy_C_thr.retile_S(mma_rC);            // (R2S, R2S_M, R2S_N)
    Tensor r2s_sC = r2s_tiled_copy_C_thr.partition_D(sC); // (R2S, R2S_M, R2S_N)
    r2s_copy_with_oob_protection<T,
                                 M,
                                 N,
                                 NUM_EXPS_BEFORE_STORE,
                                 IS_STORE_ACCUM>(
        r2s_tiled_copy_C, r2s_rC, r2s_sC, thread_idx);
  }

  template <typename MMARc>
  static __device__ __forceinline__ void
      run(MMARc &mma_rC,
          T *__restrict__ a_ptr, // Do not define a_ptr and b_ptr as const here,
                                 // since we may pad remaining part on the
                                 // k-axis with 0
          T *__restrict__ b_ptr,
          char const *__restrict__ smem_allzero_ptr,
          int thread_idx) {
    if (thread_idx >= TILED_MMA_NUM_THREADS) {
      return;
    }
    TiledMMA tiled_mma;

    auto sA_l = tile_to_shape(TileALayout{}, shape(SmemLayoutA{}));
    auto sB_l = tile_to_shape(TileBLayout{}, shape(SmemLayoutB{}));

    Tensor sA = make_tensor(make_smem_ptr(a_ptr), sA_l); // [M, K]
    Tensor sB = make_tensor(make_smem_ptr(b_ptr), sB_l); // [N, K]

    ThrMMA thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

    Tensor tCrA = thr_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB); // (MMA,MMA_N,MMA_K,PIPE)

    // auto k_tile_count = size<3>(tCrA);

    auto k_tile_count = 1;

    CUTE_UNROLL
    for (int i_k = 0; i_k < k_tile_count; ++i_k) {
      cute::warpgroup_arrive();
      gemm(tiled_mma, mma_rC, tCrA, tCrB, mma_rC);
      cute::warpgroup_commit_batch();
      cute::warpgroup_wait<0>();
    }

    if (thread0()) {
      // print("see A: ");
      // print_tensor(sA);

      // print("see B: ");
      // print_tensor(sB);
      print("mma_rC: ");
      print_tensor(mma_rC);
      // printf("\n");

      // print("tCrB: ");
      // cute::print(tCrB);
      // printf("\n");
      // print("tCsA: ");
      // cute::print(tCsA);
      // printf("\n");
      // print("tCsB: ");
      // cute::print(tCsB);
      // printf("\n");
    }
  }
};

} // namespace tb
