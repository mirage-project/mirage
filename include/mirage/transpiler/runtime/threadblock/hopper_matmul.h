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
using namespace cute;

#include "element_unary.h"

namespace tb {

template <int x, int y>
C<std::max(x, y)> max(C<x> const &, C<y> const &) {
  return {};
}

// Dim01Swapper - Swap the first two dims of the input layout
//
// Assume the shape of the input layout $i$ is (A0, A1, A2), then the output
// layout $o$ has a shape of (A1, A0, A2), and $i(a0, a1, a2) = o(a1, a0, a2)$
// holds
template <class InputLayout>
class Dim01Swapper {
  CUTE_STATIC_ASSERT_V(rank(InputLayout{}) == _2{});

  using A0 = decltype(get<0>(shape(InputLayout{})));
  using A1 = decltype(get<1>(shape(InputLayout{})));
  using TransposeCoordLayout = Layout<Shape<A1, A0>, Stride<A0, _1>>;
  using Result_ = decltype(composition(InputLayout{}, TransposeCoordLayout{}));

public:
  using Result =
      decltype(coalesce(Result_{}, Step<_1, _1>{})); // By-mode coalescing
};

enum class R2STiledCopyType { UNIVERSAL, STMATRIX_N, STMATRIX_T };
// Select the R2S (register -> shared) copy atom
template <typename T, bool IS_STMATRIX_AVAIL, class Layout>
class R2STiledCopySelector {
  // TODO(intlsy) Add support for STMATRIX
  // Reminder: Out-of-bound handling after adding support for STMATRIX
  static_assert(!IS_STMATRIX_AVAIL);

public:
  // TODO(intlsy) Coalesc to uint32_t whenever possible
  using Result = Copy_Atom<UniversalCopy<uint16_t>, T>;
  static constexpr R2STiledCopyType TYPE = R2STiledCopyType::UNIVERSAL;
};

template <class T,
          class M,
          class K,
          class TiledCopy,
          class SrcEngine,
          class SrcLayout,
          class DstEngine,
          class DstLayout>
CUTE_HOST_DEVICE void s2r_copy_with_oob_protection(
    TiledCopy const &tiled_copy,
    Tensor<SrcEngine, SrcLayout> const &src, // [S2R, S2R_M] or [S2R, S2R_N]
    Tensor<DstEngine, DstLayout> &&dst,      // The same as src
    char const *smem_allzero_ptr, // Points to somewhere on the shared memory
                                  // with at least 16 bytes of zeros
    int s2r_k,
    int thread_idx) {
  static_assert(SrcLayout::rank == 2);
  static_assert(DstLayout::rank == 2);

  using TiledMN = typename TiledCopy::Tiler_MN;
  using TileM = decltype(get<0>(TiledMN{}));
  using TileK = decltype(get<1>(TiledMN{}));
  if constexpr ((M::value % TileM::value == 0) &&
                (K::value % TileK::value == 0)) {
    copy(tiled_copy, src, dst);
  } else {
    using MIndicatorLayout = Layout<Shape<M, K>, Stride<_1, _0>>;
    using KIndicatorLayout = Layout<Shape<M, K>, Stride<_0, _1>>;
    auto m_indicator_thrIdx_s2r_s2rM_s2rK =
        tiled_copy.tidfrg_S(MIndicatorLayout{});
    auto k_indicator_thrIdx_s2r_s2rM_s2rK =
        tiled_copy.tidfrg_S(KIndicatorLayout{});
    static_assert(is_static_v<decltype(m_indicator_thrIdx_s2r_s2rM_s2rK)>);
    static_assert(is_static_v<decltype(k_indicator_thrIdx_s2r_s2rM_s2rK)>);
    int offset_m = m_indicator_thrIdx_s2r_s2rM_s2rK(thread_idx, _0{}, _0{});
    int offset_k = k_indicator_thrIdx_s2r_s2rM_s2rK(thread_idx, _0{}, _0{});
    auto m_indicator_frag = m_indicator_thrIdx_s2r_s2rM_s2rK(
        thread_idx, _, make_tuple(_, _)); // [S2R, S2R_M or S2R_N, S2R_K]
    auto k_indicator_frag = k_indicator_thrIdx_s2r_s2rM_s2rK(
        thread_idx, _, make_tuple(_, _)); // Same as above

    static_assert(cosize(SrcLayout{}(_, _0{})) <= 16);
    Tensor all_zero_tensor =
        make_tensor(make_smem_ptr((T *)smem_allzero_ptr),
                    Layout<decltype(shape(SrcLayout{}(_, _0{})))>{});

    CUTE_UNROLL
    for (int i = 0; i < size<1>(src); ++i) {
      auto coord_m = offset_m + m_indicator_frag(_0{}, i, s2r_k);
      auto coord_k = offset_k + k_indicator_frag(_0{}, i, s2r_k);
      bool valid = coord_m < M{} && coord_k < K{};
      // printf("Thread %d, (%d, %d) -> (%d, %d), %d\n", thread_idx, i, s2r_k,
      // (int)coord_m, (int)coord_k, valid);
      // TODO(intlsy) Support naive UniversalCopy
      tiled_copy.call(valid ? src(_, i) : all_zero_tensor, dst(_, i));
    }
  }
}

template <class T,
          class M,
          class N,
          int NUM_EXPS_BEFORE_STORE,
          bool IS_STORE_ACCUM,
          class TiledCopy,
          class SrcEngine,
          class SrcLayout,
          class DstEngine,
          class DstLayout>
CUTE_HOST_DEVICE void r2s_copy_with_oob_protection(
    TiledCopy const &tiled_copy,
    Tensor<SrcEngine, SrcLayout> const &src, // [R2S, R2S_M, R2S_N]
    Tensor<DstEngine, DstLayout> &dst,       // The same as src
    int thread_idx) {
  static_assert(SrcLayout::rank == 3);
  static_assert(DstLayout::rank == 3);

  using TiledMN = typename TiledCopy::Tiler_MN;
  using TileM = decltype(get<0>(TiledMN{}));
  using TileN = decltype(get<1>(TiledMN{}));
  if constexpr ((M::value % TileM::value == 0) &&
                (N::value % TileN::value == 0)) {
    CUTE_UNROLL
    for (int i = 0; i < size(src); ++i) {
      // TODO(intlsy) Modify this after supporting `stmatrix` on H100
      T x = src(i);
      if constexpr (NUM_EXPS_BEFORE_STORE > 0) {
        CUTE_UNROLL
        for (int i = 0; i < NUM_EXPS_BEFORE_STORE; ++i) {
          x = perform_element_unary_op<T, ElementUnaryOpType::EXP>(x);
        }
      }
      if constexpr (IS_STORE_ACCUM) {
        dst(i) += x;
      } else {
        dst(i) = x;
      }
    }
  } else {
    using MIndicatorLayout = Layout<Shape<M, N>, Stride<_1, _0>>;
    using NIndicatorLayout = Layout<Shape<M, N>, Stride<_0, _1>>;
    auto m_indicator_thrIdx_r2s_r2sM_r2sN =
        tiled_copy.tidfrg_D(MIndicatorLayout{});
    auto n_indicator_thrIdx_r2s_r2sM_r2sN =
        tiled_copy.tidfrg_D(NIndicatorLayout{});
    static_assert(is_static_v<decltype(m_indicator_thrIdx_r2s_r2sM_r2sN)>);
    static_assert(is_static_v<decltype(n_indicator_thrIdx_r2s_r2sM_r2sN)>);
    int offset_m = m_indicator_thrIdx_r2s_r2sM_r2sN(thread_idx, _0{}, _0{});
    int offset_n = n_indicator_thrIdx_r2s_r2sM_r2sN(thread_idx, _0{}, _0{});
    auto m_indicator_frag = m_indicator_thrIdx_r2s_r2sM_r2sN(
        thread_idx, _, make_tuple(_, _)); // [R2S, R2S_M, R2S_N]
    auto n_indicator_frag = n_indicator_thrIdx_r2s_r2sM_r2sN(
        thread_idx, _, make_tuple(_, _)); // Same as above

    CUTE_UNROLL
    for (int i = 0; i < size(src); ++i) {
      auto coord_m = offset_m + m_indicator_frag(i);
      auto coord_n = offset_n + n_indicator_frag(i);
      bool valid = coord_m < M{} && coord_n < N{};
      // TODO(intlsy) Modify this after supporting `stmatrix` on H100
      // Cannot use a `if (valid)` since `stmatrix` needs all threads in a warp
      // to have the same control flow, or the program will stuck
      // printf("Thread %d, (%d) -> (%d, %d), %d\n", thread_idx, i,
      // (int)coord_m, (int)coord_n, valid);
      if (valid) {
        T x = src(i);
        if constexpr (NUM_EXPS_BEFORE_STORE > 0) {
          CUTE_UNROLL
          for (int i = 0; i < NUM_EXPS_BEFORE_STORE; ++i) {
            x = perform_element_unary_op<T, ElementUnaryOpType::EXP>(x);
          }
        }
        if constexpr (IS_STORE_ACCUM) {
          dst(i) += x;
        } else {
          dst(i) = x;
        }
      }
    }
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
class Matmul {
public:
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutA_{}) == _2{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutB_{}) == _2{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutC_{}) == _2{});

  using SmemLayoutA = typename Dim01Swapper<SmemLayoutA_>::Result; // [M, K]
  using SmemLayoutB = SmemLayoutB_;                                // [N, K]
  using SmemLayoutC = typename Dim01Swapper<SmemLayoutC_>::Result; // [M, N]

  // Shape checking
  // Expect A have a shape of [M, K], B have a shape of [N, K], and
  // C have a shape of [M, N]
  using M = decltype(get<0>(shape(SmemLayoutA{})));
  using K = decltype(get<1>(shape(SmemLayoutA{})));
  using N = decltype(get<0>(shape(SmemLayoutB{})));
  CUTE_STATIC_ASSERT_V(K{} == get<1>(shape(SmemLayoutB{})));
  CUTE_STATIC_ASSERT_V(M{} == get<0>(shape(SmemLayoutC{})));
  CUTE_STATIC_ASSERT_V(N{} == get<1>(shape(SmemLayoutC{})));

  TiledMMA tiled_mma = make_tiled_mma(
      SM90_64x64x16_F16F16F16_SS<GMMA::Major::MN, GMMA::Major::MN>{});

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

    Tensor sA = make_tensor(make_smem_ptr(a_ptr), SmemLayoutA{}); // [M, K]
    Tensor sB = make_tensor(make_smem_ptr(b_ptr), SmemLayoutB{}); // [N, K]

    ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments"
    Tensor tCrA = thr_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thr_mma.make_fragment_B(tCsB); // (MMA,MMA_N,MMA_K,PIPE)

    static constexpr int NUM_MMA_K_STAGES = size(shape<2>(sA));

    CUTE_UNROLL
    for (int i_k = 0; i_k < NUM_MMA_K_STAGES; ++i_k) {
      cute::warpgroup_arrive();
      gemm(tiled_mma, mma_rC, mma_rA(_, _, i_k), mma_rB(_, _, i_k), mma_rC);
      cute::warpgroup_commit_batch();
      cute::warpgroup_wait<0>();
    }
  }
};

} // namespace tb
