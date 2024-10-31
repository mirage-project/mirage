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

enum class S2RTiledCopyType { UNIVERSAL, LDMATRIX_N, LDMATRIX_T };
// Select the S2R (shared -> register) copy atom
template <typename T,
          bool IS_LDMATRIX_AVAIL,
          class Layout,
          int K_DIM,
          class MMAAtomK>
class S2RTiledCopySelector {
  // TODO(intlsy) Add support for architectures that do not support LDMATRIX
  // (don't forget to consider the case where the shape of the matrix is not
  // divisible by the shape of TiledMMA)
  static_assert(IS_LDMATRIX_AVAIL);

  static constexpr bool IS_DIM0_INNERMOST = (Layout{})(_1{}, _0{}) == _1{};
  static constexpr bool IS_DIM1_INNERMOST = (Layout{})(_0{}, _1{}) == _1{};
  static constexpr int CONSECUTIVE_DIM = IS_DIM0_INNERMOST ? 0 : 1;

  // TODO(intlsy) Fallback to normal copy when this is not true
  static_assert(IS_DIM0_INNERMOST || IS_DIM1_INNERMOST);

public:
  // Since we want to pipeline S->R copy and MMA, we would like the MMAAtom and
  // the CopyAtom to have the same shape on the K dim
  using CandidateLdMatrixN = std::conditional_t<
      MMAAtomK{} == _16{},
      SM75_U32x4_LDSM_N,
      std::conditional_t<MMAAtomK{} == _8{}, SM75_U32x2_LDSM_N, void>>;
  using CandidateLdMatrixT = std::conditional_t<
      MMAAtomK{} == _16{},
      SM75_U16x8_LDSM_T,
      std::conditional_t<MMAAtomK{} == _8{}, SM75_U16x4_LDSM_T, void>>;
  static_assert(!std::is_same_v<CandidateLdMatrixN, void>);
  static_assert(!std::is_same_v<CandidateLdMatrixT, void>);

  // If `ldmatrix` is available and the innermost dim is among the first two
  // dims, use `ldmatrix` or `ldmatrix.trans`. Otherwise, use the universal copy
  // atom
  using Result = std::conditional_t<CONSECUTIVE_DIM == K_DIM,
                                    Copy_Atom<CandidateLdMatrixN, T>,
                                    Copy_Atom<CandidateLdMatrixT, T>>;
  static constexpr S2RTiledCopyType TYPE = CONSECUTIVE_DIM == K_DIM
                                               ? S2RTiledCopyType::LDMATRIX_N
                                               : S2RTiledCopyType::LDMATRIX_T;
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

} // namespace tb
