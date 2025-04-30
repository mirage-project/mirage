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
  using Result = std::conditional_t<
      CONSECUTIVE_DIM == K_DIM && !std::is_same_v<T, float>,
      Copy_Atom<CandidateLdMatrixN, T>,
      std::conditional_t<CONSECUTIVE_DIM != K_DIM && !std::is_same_v<T, float>,
                         Copy_Atom<CandidateLdMatrixT, T>,
                         Copy_Atom<UniversalCopy<uint32_t>, T>>>;
  static constexpr S2RTiledCopyType TYPE =
      CONSECUTIVE_DIM == K_DIM && !std::is_same_v<T, float>
          ? S2RTiledCopyType::LDMATRIX_N
      : CONSECUTIVE_DIM != K_DIM && !std::is_same_v<T, float>
          ? S2RTiledCopyType::LDMATRIX_T
          : S2RTiledCopyType::UNIVERSAL;
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
      float x = src(i);
      if constexpr (NUM_EXPS_BEFORE_STORE > 0) {
        CUTE_UNROLL
        for (int i = 0; i < NUM_EXPS_BEFORE_STORE; ++i) {
          x = perform_element_unary_op<float, ElementUnaryOpType::EXP>(x);
        }
      }
      if constexpr (IS_STORE_ACCUM) {
        dst(i) += T(x);
      } else {
        dst(i) = T(x);
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
        float x = src(i);
        if constexpr (NUM_EXPS_BEFORE_STORE > 0) {
          CUTE_UNROLL
          for (int i = 0; i < NUM_EXPS_BEFORE_STORE; ++i) {
            x = perform_element_unary_op<float, ElementUnaryOpType::EXP>(x);
          }
        }
        if constexpr (IS_STORE_ACCUM) {
          dst(i) += T(x);
        } else {
          dst(i) = T(x);
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
          class SmemLayoutAAligned_,
          class SmemLayoutBAligned_,
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

  using SmemLayoutAAligned = typename Dim01Swapper<SmemLayoutAAligned_>::Result;
  using SmemLayoutBAligned = SmemLayoutBAligned_;
  // Shape checking
  // Expect A have a shape of [M, K], B have a shape of [N, K], and
  // C have a shape of [M, N]
  using M = decltype(get<0>(shape(SmemLayoutA{})));
  using K = decltype(get<1>(shape(SmemLayoutA{})));
  using N = decltype(get<0>(shape(SmemLayoutB{})));
  CUTE_STATIC_ASSERT_V(K{} == get<1>(shape(SmemLayoutB{})));
  CUTE_STATIC_ASSERT_V(M{} == get<0>(shape(SmemLayoutC{})));
  CUTE_STATIC_ASSERT_V(N{} == get<1>(shape(SmemLayoutC{})));

  // TiledMMA generation
  // TODO(intlsy) When the S->R copy atom is not ldmatrix, remove this 16
  using MMAAtomMNK = typename MMA_Traits<MMAAtom>::Shape_MNK;
  using MMAAtomK = decltype(get<2>(MMAAtomMNK{}));
  using MMATileM = decltype(max(get<0>(MMAAtomMNK{}), _16{}) *
                            get<0>(shape(TiledMMAThrLayout{})));
  using MMATileN = decltype(max(get<1>(MMAAtomMNK{}), _16{}) *
                            get<1>(shape(TiledMMAThrLayout{})));
  // We use `Tile<MMATileM, MMATileN, _16>` since that, ldmatrix loads a matrix
  // of shape 16x16 (assume using half_t), so as long as the MMA has a tile
  // shape of ?x?x16, there will be the same number of MMA_K and S2R_K
  using TiledMMA = decltype(make_tiled_mma(
      MMAAtom{}, TiledMMAThrLayout{}, Tile<MMATileM, MMATileN, MMAAtomK>{}));
  static constexpr int TILED_MMA_NUM_THREADS = thr_size(TiledMMA{});
  static_assert(TILED_MMA_NUM_THREADS <= NUM_THREADS);

  // CopyAtom selection
  using S2RTiledCopyASelector =
      S2RTiledCopySelector<T, IS_LDMATRIX_AVAIL, SmemLayoutA, 1, MMAAtomK>;
  using S2RTiledCopyAAtom = typename S2RTiledCopyASelector::Result;
  static constexpr S2RTiledCopyType S2R_TILED_COPY_A_TYPE =
      S2RTiledCopyASelector::TYPE;

  using S2RTiledCopyBSelector =
      S2RTiledCopySelector<T, IS_LDMATRIX_AVAIL, SmemLayoutB, 1, MMAAtomK>;
  using S2RTiledCopyBAtom = typename S2RTiledCopyBSelector::Result;
  static constexpr S2RTiledCopyType S2R_TILED_COPY_B_TYPE =
      S2RTiledCopyBSelector::TYPE;

  using R2STiledCopyCSelector =
      R2STiledCopySelector<T, IS_STMATRIX_AVAIL, SmemLayoutC>;
  using R2STiledCopyCAtom = typename R2STiledCopyCSelector::Result;
  static constexpr R2STiledCopyType R2S_TILED_COPY_C_TYPE =
      R2STiledCopyCSelector::TYPE;

  using S2RTiledCopyA =
      decltype(make_tiled_copy_A(S2RTiledCopyAAtom{}, TiledMMA{}));
  using S2RTiledCopyB =
      decltype(make_tiled_copy_B(S2RTiledCopyBAtom{}, TiledMMA{}));
  using R2STiledCopyC =
      decltype(make_tiled_copy_C(R2STiledCopyCAtom{}, TiledMMA{}));

  static __device__ __forceinline__ auto get_mma_rC(int thread_idx) {
    // Make a fake tensor
    Tensor sC_fake = make_tensor(make_smem_ptr((T *)nullptr), SmemLayoutC{});
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

    if (thread_idx < TILED_MMA_NUM_THREADS) {

      Tensor sA = make_tensor(make_smem_ptr(a_ptr), SmemLayoutA{}); // [M, K]
      Tensor sB = make_tensor(make_smem_ptr(b_ptr), SmemLayoutB{}); // [N, K]

      Tensor sA_r = make_tensor(make_smem_ptr(a_ptr), SmemLayoutAAligned{});
      Tensor sB_r = make_tensor(make_smem_ptr(b_ptr), SmemLayoutBAligned{});

      if constexpr (K{} % _8{} != _0{}) {
        // Need to pad with zero
        static constexpr int K_LEFTOVER = int(_8{} - K{} % _8{});
        if constexpr (S2R_TILED_COPY_A_TYPE == S2RTiledCopyType::LDMATRIX_N) {
          CUTE_UNROLL
          for (int i = thread_idx; i < int(M{}) * K_LEFTOVER;
               i += NUM_THREADS) {
            sA(i / K_LEFTOVER, K{} + i % K_LEFTOVER) = (T)0;
          }
        }
        if constexpr (S2R_TILED_COPY_B_TYPE == S2RTiledCopyType::LDMATRIX_N) {
          CUTE_UNROLL
          for (int i = thread_idx; i < int(N{}) * K_LEFTOVER;
               i += NUM_THREADS) {
            sB(i / K_LEFTOVER, K{} + i % K_LEFTOVER) = (T)0;
          }
        }
      }
      TiledMMA tiled_mma;
      ThrMMA thr_mma = tiled_mma.get_slice(thread_idx);

      Tensor mma_rA = thr_mma.partition_fragment_A(sA_r); // (MMA, MMA_M, MMA_K)
      Tensor mma_rB = thr_mma.partition_fragment_B(sB_r); // (MMA, MMA_N, MMA_K)

      // NOTE. If you encountered the issue
      //
      // static_assert(decltype(size(rB) == Int<RegNumB>{})::value);
      //
      // Please upgrade your Cutlass version to at least 3.5.1 (commit
      // e1976daacc7b030ba672217eb5d96f5a663df4ab) Refer to this link for more
      // information: https://github.com/NVIDIA/cutlass/issues/1766

      S2RTiledCopyA s2r_tiled_copy_A;
      ThrCopy s2r_tiled_copy_A_thr = s2r_tiled_copy_A.get_slice(thread_idx);
      Tensor s2r_sA =
          s2r_tiled_copy_A_thr.partition_S(sA); // (S2R, S2R_M, S2R_K)
      Tensor s2r_rA =
          s2r_tiled_copy_A_thr.retile_D(mma_rA); // (S2R, S2R_M, S2R_K)

      S2RTiledCopyB s2r_tiled_copy_B;
      ThrCopy s2r_tiled_copy_B_thr = s2r_tiled_copy_B.get_slice(thread_idx);
      Tensor s2r_sB =
          s2r_tiled_copy_B_thr.partition_S(sB); // (S2R, S2R_N, S2R_K)
      Tensor s2r_rB =
          s2r_tiled_copy_B_thr.retile_D(mma_rB); // (S2R, S2R_N, S2R_K)

      CUTE_STATIC_ASSERT_V(size(shape<2>(s2r_rA)) == size(shape<2>(mma_rA)));
      CUTE_STATIC_ASSERT_V(size(shape<2>(s2r_rA)) == size(shape<2>(s2r_rB)));
      static constexpr int NUM_MMA_K_STAGES = size(shape<2>(s2r_sA));

#define S2RCOPY(k_idx)                                                         \
  s2r_copy_with_oob_protection<T, M, K>(s2r_tiled_copy_A,                      \
                                        s2r_sA(_, _, k_idx),                   \
                                        s2r_rA(_, _, k_idx),                   \
                                        smem_allzero_ptr,                      \
                                        k_idx,                                 \
                                        thread_idx);                           \
  s2r_copy_with_oob_protection<T, N, K>(s2r_tiled_copy_B,                      \
                                        s2r_sB(_, _, k_idx),                   \
                                        s2r_rB(_, _, k_idx),                   \
                                        smem_allzero_ptr,                      \
                                        k_idx,                                 \
                                        thread_idx);

      // Pipeline S->R copy and MMA
      S2RCOPY(_0{});

      CUTE_UNROLL
      for (int i_k = 0; i_k < NUM_MMA_K_STAGES; ++i_k) {
        if (i_k + 1 != NUM_MMA_K_STAGES) {
          S2RCOPY(i_k + 1);
        }
        gemm(tiled_mma, mma_rC, mma_rA(_, _, i_k), mma_rB(_, _, i_k), mma_rC);
      }
    }
    __syncthreads();
  }
};

} // namespace tb
