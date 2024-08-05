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

namespace tb {

template<int x, int y>
C<std::max(x, y)> max(const C<x>&, const C<y>&) {
    return {};
}

// LayoutGrouper - Return a new layout with a rank of 3
// If the input layout has a rank of 2, append a dim of 1 to the end
// If the input layout's rank is greater than 2, group the last (rank-2) dims
// into one This convert the input layout to a shape of "[MATRIX-SIZE-0,
// MATRIX-SIZE-1, BATCH-SIZE]"
template <class RealLayout>
class LayoutGrouper {
  static_assert(is_static_v<RealLayout>);
  CUTE_STATIC_ASSERT_V(rank(RealLayout{}) >= _2{});

public:
  using Result = std::conditional_t<
      (rank(RealLayout{}) == _2{}),
      Layout<decltype(make_shape(get<0>(shape(RealLayout{})),
                                 get<1>(shape(RealLayout{})),
                                 _1{})),
             decltype(make_stride(get<0>(stride(RealLayout{})),
                                  get<1>(stride(RealLayout{})),
                                  _1{}))>,
      decltype(group<2, rank(RealLayout{})>(RealLayout{}))>;
  CUTE_STATIC_ASSERT_V(rank(Result{}) == _3{});
};

// Dim01Swapper - Swap the first two dims of the input layout
template <class InputLayout>
class Dim01Swapper {
    CUTE_STATIC_ASSERT_V(rank(InputLayout{}) == _3{});
public:
    using Result = Layout<decltype(make_shape(get<1>(shape(InputLayout{})),
                                             get<0>(shape(InputLayout{})),
                                             get<2>(shape(InputLayout{})))),
                          decltype(make_stride(get<1>(stride(InputLayout{})),
                                               get<0>(stride(InputLayout{})),
                                               get<2>(stride(InputLayout{}))))>;
};

// Select the S2R (shared -> register) copy atom
template <typename T, bool IS_LDMATRIX_AVAIL, class Layout, int K_DIM, class MMAAtomK>
class S2RTiledCopySelector {
  // TODO(intlsy) Add support for architectures that do not support LDMATRIX
  // (don't forget to consider the case where the shape of the matrix is not
  // divisible by the shape of TiledMMA)
  static_assert(IS_LDMATRIX_AVAIL);

  static constexpr bool IS_DIM0_INNERMOST = get<0>(stride(Layout{})) == _1{};
  static constexpr bool IS_DIM1_INNERMOST = get<1>(stride(Layout{})) == _1{};
  static constexpr int CONSECUTIVE_DIM = IS_DIM0_INNERMOST ? 0 : 1;

  // TODO(intlsy) Fallback to normal copy when this is not true
  static_assert(IS_DIM0_INNERMOST || IS_DIM1_INNERMOST);

public:
  // Since we want to pipeline S->R copy and MMA, we would like the MMAAtom and
  // the CopyAtom to have the same shape on the K dim
  using CandidateLdMatrixN = std::conditional_t<MMAAtomK{} == _16{}, SM75_U32x4_LDSM_N, std::conditional_t<MMAAtomK{} == _8{}, SM75_U32x2_LDSM_N, void>>;
  using CandidateLdMatrixT = std::conditional_t<MMAAtomK{} == _16{}, SM75_U16x8_LDSM_T, std::conditional_t<MMAAtomK{} == _8{}, SM75_U16x4_LDSM_T, void>>;
  static_assert(!std::is_same_v<CandidateLdMatrixN, void>);
  static_assert(!std::is_same_v<CandidateLdMatrixT, void>);

  // If `ldmatrix` is available and the innermost dim is among the first two dims,
  // use `ldmatrix` or `ldmatrix.trans`. Otherwise, use the universal copy atom
  using Result = std::conditional_t<CONSECUTIVE_DIM == K_DIM,
                                  Copy_Atom<CandidateLdMatrixN, T>,
                                  Copy_Atom<CandidateLdMatrixT, T>>;
};

// Select the R2S (register -> shared) copy atom
template <typename T, bool IS_STMATRIX_AVAIL, class Layout>
class R2STiledCopySelector {
  // TODO(intlsy) Add support for STMATRIX
  // Reminder: Out-of-bound handling after adding support for STMATRIX
  static_assert(!IS_STMATRIX_AVAIL);

public:
  // TODO(intlsy) Coalesc to uint32_t whenever possible
  using Result = Copy_Atom<UniversalCopy<uint16_t>, T>;
};


template <class T,
          class M, class K,
          class TiledCopy,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void s2r_copy_with_oob_protection(
  TiledCopy const &tiled_copy,
  Tensor<SrcEngine, SrcLayout> const& src,    // [S2R, S2R_M] or [S2R, S2R_N]
  Tensor<DstEngine, DstLayout> &&dst,   // The same as src
  const char* smem_allzero_ptr, // Points to somewhere on the shared memory with at least 16 bytes of zeros
  int s2r_k,
  int thread_idx
) {
  static_assert(SrcLayout::rank == 2);
  static_assert(DstLayout::rank == 2);

  using MIndicatorLayout = Layout<Shape<M, K>, Stride<_1, _0>>;
  using KIndicatorLayout = Layout<Shape<M, K>, Stride<_0, _1>>;
  auto m_indicator_thrIdx_s2r_s2rM_s2rK = tiled_copy.tidfrg_S(MIndicatorLayout{});
  auto k_indicator_thrIdx_s2r_s2rM_s2rK = tiled_copy.tidfrg_S(KIndicatorLayout{});
  static_assert(is_static_v<decltype(m_indicator_thrIdx_s2r_s2rM_s2rK)>);
  static_assert(is_static_v<decltype(k_indicator_thrIdx_s2r_s2rM_s2rK)>);
  int offset_m = m_indicator_thrIdx_s2r_s2rM_s2rK(thread_idx, _0{}, _0{});
  int offset_k = k_indicator_thrIdx_s2r_s2rM_s2rK(thread_idx, _0{}, _0{});
  auto m_indicator_frag = m_indicator_thrIdx_s2r_s2rM_s2rK(thread_idx, _, make_tuple(_, _)); // [S2R, S2R_M or S2R_N, S2R_K]
  auto k_indicator_frag = k_indicator_thrIdx_s2r_s2rM_s2rK(thread_idx, _, make_tuple(_, _)); // Same as above

  static_assert(cosize(SrcLayout{}(_, _0{})) <= 16);
  Tensor all_zero_tensor = make_tensor(make_smem_ptr((T*)smem_allzero_ptr), Layout<decltype(shape(SrcLayout{}(_, _0{})))>{});

  CUTE_UNROLL
  for (int i = 0; i < size<1>(src); ++i) {
    auto coord_m = offset_m + m_indicator_frag(_0{}, i, s2r_k);
    auto coord_k = offset_k + k_indicator_frag(_0{}, i, s2r_k);
    bool valid = coord_m < M{} && coord_k < K{};
    // printf("Thread %d, (%d, %d) -> (%d, %d), %d\n", thread_idx, i, s2r_k, (int)coord_m, (int)coord_k, valid);
    // TODO(intlsy) Support naive UniversalCopy
    tiled_copy.call(valid ? src(_, i) : all_zero_tensor, dst(_, i));
  }
}

template <class T,
          class M, class N,
          class TiledCopy,
          class SrcEngine, class SrcLayout,
          class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void r2s_copy_with_oob_protection(
  TiledCopy const &tiled_copy,
  Tensor<SrcEngine, SrcLayout> const& src,    // [R2S, R2S_M, R2S_N]
  Tensor<DstEngine, DstLayout> &&dst,   // The same as src
  int thread_idx
) {
  static_assert(SrcLayout::rank == 3);
  static_assert(DstLayout::rank == 3);

  using MIndicatorLayout = Layout<Shape<M, N>, Stride<_1, _0>>;
  using NIndicatorLayout = Layout<Shape<M, N>, Stride<_0, _1>>;
  auto m_indicator_thrIdx_r2s_r2sM_r2sN = tiled_copy.tidfrg_D(MIndicatorLayout{});
  auto n_indicator_thrIdx_r2s_r2sM_r2sN = tiled_copy.tidfrg_D(NIndicatorLayout{});
  static_assert(is_static_v<decltype(m_indicator_thrIdx_r2s_r2sM_r2sN)>);
  static_assert(is_static_v<decltype(n_indicator_thrIdx_r2s_r2sM_r2sN)>);
  int offset_m = m_indicator_thrIdx_r2s_r2sM_r2sN(thread_idx, _0{}, _0{});
  int offset_n = n_indicator_thrIdx_r2s_r2sM_r2sN(thread_idx, _0{}, _0{});
  auto m_indicator_frag = m_indicator_thrIdx_r2s_r2sM_r2sN(thread_idx, _, make_tuple(_, _)); // [R2S, R2S_M, R2S_N]
  auto n_indicator_frag = n_indicator_thrIdx_r2s_r2sM_r2sN(thread_idx, _, make_tuple(_, _)); // Same as above

  CUTE_UNROLL
  for (int i = 0; i < size(src); ++i) {
    auto coord_m = offset_m + m_indicator_frag(i);
    auto coord_n = offset_n + n_indicator_frag(i);
    bool valid = coord_m < M{} && coord_n < N{};
    // TODO(intlsy) Modify this after supporting `stmatrix` on H100
    // Cannot use a `if (valid)` since `stmatrix` needs all threads in a warp
    // to have the same control flow, or the program will stuck
    // printf("Thread %d, (%d) -> (%d, %d), %d\n", thread_idx, i, j, (int)coord_m, (int)coord_n, valid);
    if (valid) {
      dst(i) = src(i);
    }
  }
}

template <typename T,
          class MMAAtom,
          class TiledMMAThrLayout,
          bool IS_LDMATRIX_AVAIL,
          bool IS_STMATRIX_AVAIL,
          class RealSmemLayoutA,    // [K, M, ...]
          class RealSmemLayoutB,    // [N, K, ...]
          class RealSmemLayoutC,    // [N, M, ...]
          int NUM_THREADS>
class Matmul {
public:
  // Group the last few dims into one dim
  CUTE_STATIC_ASSERT_V(rank(RealSmemLayoutA{}) ==
                           rank(RealSmemLayoutB{}) &&
                       rank(RealSmemLayoutB{}) ==
                           rank(RealSmemLayoutC{}));
  using SmemLayoutA_ = typename LayoutGrouper<RealSmemLayoutA>::Result; // [K, M, B]
  using SmemLayoutB_ = typename LayoutGrouper<RealSmemLayoutB>::Result; // [N, K, B]
  using SmemLayoutC_ = typename LayoutGrouper<RealSmemLayoutC>::Result; // [N, M, B]

  using SmemLayoutA = typename Dim01Swapper<SmemLayoutA_>::Result; // [M, K, B]
  using SmemLayoutB = SmemLayoutB_; // [N, K, B]
  using SmemLayoutC = typename Dim01Swapper<SmemLayoutC_>::Result; // [M, N, B]

  // Currently Mirage only supports STensor with the last 2 dims (in our case,
  // the first 2 dims) > 1. So the last dim of SmemLayoutA/B/C should always
  // be 1
  // TODO(intlsy) Relax this assumption
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutA{}) == _3{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutB{}) == _3{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutC{}) == _3{});
  CUTE_STATIC_ASSERT_V(get<2>(shape(SmemLayoutA{})) == _1{});
  CUTE_STATIC_ASSERT_V(get<2>(shape(SmemLayoutB{})) == _1{});
  CUTE_STATIC_ASSERT_V(get<2>(shape(SmemLayoutC{})) == _1{});

  // Shape checking
  // Expect A have a shape of [M, K, B], B have a shape of [N, K, B], and
  // C have a shape of [M, N, B]
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
  using MMATileM = decltype(max(get<0>(MMAAtomMNK{}), _16{}) * get<0>(shape(TiledMMAThrLayout{})));
  using MMATileN = decltype(max(get<1>(MMAAtomMNK{}), _16{}) * get<1>(shape(TiledMMAThrLayout{})));
  // We use `Tile<MMATileM, MMATileN, _16>` since that, ldmatrix loads a matrix
  // of shape 16x16 (assume using half_t), so as long as the MMA has a tile shape
  // of ?x?x16, there will be the same number of MMA_K and S2R_K
  using TiledMMA = decltype(make_tiled_mma(
      MMAAtom{}, TiledMMAThrLayout{}, Tile<MMATileM, MMATileN, MMAAtomK>{}));
  static constexpr int TILED_MMA_NUM_THREADS = thr_size(TiledMMA{});
  static_assert(TILED_MMA_NUM_THREADS <= NUM_THREADS);

  // CopyAtom selection
  using S2RTiledCopyAAtom =
      typename S2RTiledCopySelector<T, IS_LDMATRIX_AVAIL, SmemLayoutA, 1, MMAAtomK>::Result;
  using S2RTiledCopyBAtom =
      typename S2RTiledCopySelector<T, IS_LDMATRIX_AVAIL, SmemLayoutB, 1, MMAAtomK>::Result;
  using R2STiledCopyCAtom =
      typename R2STiledCopySelector<T, IS_STMATRIX_AVAIL, SmemLayoutC>::Result;

  using S2RTiledCopyA =
      decltype(make_tiled_copy_A(S2RTiledCopyAAtom{}, TiledMMA{}));
  using S2RTiledCopyB =
      decltype(make_tiled_copy_B(S2RTiledCopyBAtom{}, TiledMMA{}));
  using R2STiledCopyC =
      decltype(make_tiled_copy_C(R2STiledCopyCAtom{}, TiledMMA{}));

  static __device__ __forceinline__ void run(T *__restrict__ c_ptr,
                                             T *__restrict__ a_ptr, // Do not define a_ptr and b_ptr as const here, since we may pad remaining part on the k-axis with 0
                                             T *__restrict__ b_ptr,
                                             char const *__restrict__ smem_allzero_ptr,
                                             int thread_idx) {
    if (thread_idx >= TILED_MMA_NUM_THREADS) {
      return;
    }

    Tensor sA = make_tensor(make_smem_ptr(a_ptr), SmemLayoutA{}); // [M, K, B]
    Tensor sB = make_tensor(make_smem_ptr(b_ptr), SmemLayoutB{}); // [N, K, B]
    Tensor sC = make_tensor(make_smem_ptr(c_ptr), SmemLayoutC{}); // [M, N, B]
    CUTE_STATIC_ASSERT_V(rank(sA) == _3{});

    if constexpr(K{} % _8{} != _0{}) {
      // Need to pad with zero
      static constexpr int K_LEFTOVER = int(_8{} - K{} % _8{});
      CUTE_UNROLL
      for (int i = thread_idx; i < int(M{}) * K_LEFTOVER; i += NUM_THREADS)
        sA(i / K_LEFTOVER, K{} + i % K_LEFTOVER, _0{}) = (T)0;
      CUTE_UNROLL
      for (int i = thread_idx; i < int(N{}) * K_LEFTOVER; i += NUM_THREADS)
        sB(i / K_LEFTOVER, K{} + i % K_LEFTOVER, _0{}) = (T)0;
    }
    TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(thread_idx);
    Tensor mma_rA =
        thr_mma.partition_fragment_A(sA(_, _, _0{})); // (MMA, MMA_M, MMA_K)
    Tensor mma_rB =
        thr_mma.partition_fragment_B(sB(_, _, _0{})); // (MMA, MMA_N, MMA_K)
    Tensor mma_rC =
        thr_mma.partition_fragment_C(sC(_, _, _0{})); // (MMA, MMA_M, MMA_N)
    clear(mma_rC);

    S2RTiledCopyA s2r_tiled_copy_A;
    ThrCopy s2r_tiled_copy_A_thr = s2r_tiled_copy_A.get_slice(thread_idx);
    Tensor s2r_sA =
        s2r_tiled_copy_A_thr.partition_S(sA); // (S2R, S2R_M, S2R_K, B)
    Tensor s2r_rA =
        s2r_tiled_copy_A_thr.retile_D(mma_rA); // (S2R, S2R_M, S2R_K)

    S2RTiledCopyB s2r_tiled_copy_B;
    ThrCopy s2r_tiled_copy_B_thr = s2r_tiled_copy_B.get_slice(thread_idx);
    Tensor s2r_sB =
        s2r_tiled_copy_B_thr.partition_S(sB); // (S2R, S2R_N, S2R_K, B)
    Tensor s2r_rB =
        s2r_tiled_copy_B_thr.retile_D(mma_rB); // (S2R, S2R_N, S2R_K)

    R2STiledCopyC r2s_tiled_copy_C;
    ThrCopy r2s_tiled_copy_C_thr = r2s_tiled_copy_C.get_slice(thread_idx);
    Tensor r2s_rC =
        r2s_tiled_copy_C_thr.retile_S(mma_rC); // (R2S, R2S_M, R2S_N)
    Tensor r2s_sC =
        r2s_tiled_copy_C_thr.partition_D(sC); // (R2S, R2S_M, R2S_N, B)

    CUTE_STATIC_ASSERT_V(shape<2>(s2r_rA) == shape<2>(mma_rA));
    CUTE_STATIC_ASSERT_V(shape<2>(s2r_rA) == shape<2>(s2r_rB));
    static constexpr int NUM_MMA_K_STAGES = shape<2>(s2r_sA);

    // TODO(intlsy) Eliminate unnecessary boundary checking when the shape is divisible
    #define S2RCOPY(k_idx) \
      s2r_copy_with_oob_protection<T, M, K>(\
        s2r_tiled_copy_A, \
        s2r_sA(_, _, k_idx, _0{}), \
        s2r_rA(_, _, k_idx), \
        smem_allzero_ptr, \
        k_idx, \
        thread_idx \
      ); \
      s2r_copy_with_oob_protection<T, N, K>(\
        s2r_tiled_copy_B, \
        s2r_sB(_, _, k_idx, _0{}), \
        s2r_rB(_, _, k_idx), \
        smem_allzero_ptr, \
        k_idx, \
        thread_idx \
      );

    // Pipeline S->R copy and MMA
    S2RCOPY(_0{});

    CUTE_UNROLL
    for (int i_k = 0; i_k < NUM_MMA_K_STAGES; ++i_k) {
      if (i_k+1 != NUM_MMA_K_STAGES) {
        S2RCOPY(i_k + 1);
      }
      gemm(tiled_mma, mma_rC, mma_rA(_, _, i_k), mma_rB(_, _, i_k), mma_rC);
    }

    // TODO(intlsy) Eliminate unnecessary boundary checking when the shape is divisible
    r2s_copy_with_oob_protection<T, M, N>(
      r2s_tiled_copy_C,
      r2s_rC,
      r2s_sC(_, _, _, _0{}),
      thread_idx
    );
  }
};

} // namespace tb
