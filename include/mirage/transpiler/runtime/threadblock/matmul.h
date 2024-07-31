// // matmul.h - Implementation of threadblock level matmul operators
// //
// // WARNING To understand code in this file, you have to be familiar with the
// // CuTe library. The CuTe library is a part of the Cutlass library by NVIDIA
// // which provides (not so) easy-to-use primitives for writing high-performance
// // kernels. You may read the CuTe documentation at
// // https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md
// // or you can find some amazing blogs in Chinese here:
// // https://zhuanlan.zhihu.com/p/661182311
// //

// #pragma once

// #include <cute/atom/copy_atom.hpp>
// #include <cute/atom/mma_atom.hpp>
// #include <cute/atom/mma_traits.hpp>
// #include <cute/layout.hpp>
// #include <cute/tensor.hpp>

// using namespace cute;

// namespace tb {

// template <typename T, bool LDMATRIX_AVAIL, class Layout, int K_DIM>
// class S2RTiledCopySelector {
//   static constexpr bool IS_DIM0_INNERMOST = get<0>(stride(Layout{})) == _1{};
//   static constexpr bool IS_DIM1_INNERMOST = get<1>(stride(Layout{})) == _1{};
//   static constexpr int CONSECUTIVE_DIM = IS_DIM0_INNERMOST ? 0 : 1;

// public:
//   using Result =
//       std::conditional_t<LDMATRIX_AVAIL &&
//                              (IS_DIM0_INNERMOST || IS_DIM1_INNERMOST),
//                          std::conditional_t<CONSECUTIVE_DIM == K_DIM,
//                                             Copy_Atom<SM75_U32x4_LDSM_N, T>,
//                                             Copy_Atom<SM75_U16x8_LDSM_T, T>>,
//                          Copy_Atom<UniversalCopy<uint32_t>, T>>;
// };

// template <typename T, bool STMATRIX_AVAIL, class Layout>
// class R2STiledCopySelector {
//   // TODO(intlsy) Add support for STMATRIX
//   static_assert(!STMATRIX_AVAIL);

// public:
//   using Result = Copy_Atom<UniversalCopy<uint32_t>, T>;
// };

// template <class RealLayout>
// class LayoutGrouper {
//   static_assert(is_static_v<RealLayout>);
//   CUTE_STATIC_ASSERT_V(rank(RealLayout{}) >= _2{});

// public:
//   using Result = std::conditional_t<
//       (rank(RealLayout{}) == _2{}),
//       Layout<decltype(make_shape(get<0>(shape(RealLayout{})),
//                                  get<1>(shape(RealLayout{})),
//                                  _1{})),
//              decltype(make_stride(get<0>(stride(RealLayout{})),
//                                   get<1>(stride(RealLayout{})),
//                                   _1{}))>,
//       decltype(group<2, rank(RealLayout{})>(RealLayout{}))>;
//   CUTE_STATIC_ASSERT_V(rank(Result{}) == _3{});
// };

// template <typename T,
//           class MMAAtom,
//           class TiledMMAThrLayout,
//           bool LDMATRIX_AVAIL,
//           bool STMATRIX_AVAIL,
//           class RealSmemLayoutA,
//           class RealSmemLayoutB,
//           class RealSmemLayoutC,
//           int NUM_THREADS>
// class Matmul {
// public:
//   // Group the last few dims into one dim
//   CUTE_STATIC_ASSERT_V(rank(shape(RealSmemLayoutA{})) ==
//                            rank(shape(RealSmemLayoutB{})) &&
//                        rank(shape(RealSmemLayoutB{})) ==
//                            rank(shape(RealSmemLayoutC{})));
//   using SmemLayoutA = typename LayoutGrouper<RealSmemLayoutA>::Result;
//   using SmemLayoutB = typename LayoutGrouper<RealSmemLayoutB>::Result;
//   using SmemLayoutC = typename LayoutGrouper<RealSmemLayoutC>::Result;

//   // Currently Mirage only supports STensor with the last 2 dims (in our case,
//   // the first 2 dims) > 1. So the last dim of SmemLayoutA/B/C should always
//   // be 1
//   // TODO(intlsy) Relax this assumption
//   CUTE_STATIC_ASSERT_V(get<2>(shape(SmemLayoutA{})) == _1{});
//   CUTE_STATIC_ASSERT_V(get<2>(shape(SmemLayoutB{})) == _1{});
//   CUTE_STATIC_ASSERT_V(get<2>(shape(SmemLayoutC{})) == _1{});

//   // Shape checking
//   // Expect A have a shape of [M, K, B], B have a shape of [N, K, B], and
//   // C have a shape of [M, N, B]
//   using M = decltype(get<0>(shape(SmemLayoutA{})));
//   using K = decltype(get<1>(shape(SmemLayoutA{})));
//   using N = decltype(get<0>(shape(SmemLayoutB{})));

//   static_assert(rank(shape(SmemLayoutA{})) == rank(shape(SmemLayoutB{})) &&
//                 rank(shape(SmemLayoutB{})) == rank(shape(SmemLayoutC{})));
//   CUTE_STATIC_ASSERT_V(rank(shape(SmemLayoutA{})) >= _2{});
//   CUTE_STATIC_ASSERT_V(K{} == get<1>(shape(SmemLayoutB{})));
//   CUTE_STATIC_ASSERT_V(M{} == get<0>(shape(SmemLayoutC{})));
//   CUTE_STATIC_ASSERT_V(N{} == get<1>(shape(SmemLayoutC{})));

//   // TiledMMA generation
//   static_assert(thr_size(make_tiled_mma(MMAAtom{}, TiledMMAThrLayout{})) ==
//                 NUM_THREADS);
//   using MMAAtomTrait = MMA_Traits<MMAAtom>;
//   using TiledMMA =
//       decltype(make_tiled_mma(MMAAtom{}, TiledMMAThrLayout{}, Tile<M, N, K>{}));

//   // CopyAtom selection
//   using S2RTiledCopyAAtom =
//       typename S2RTiledCopySelector<T, LDMATRIX_AVAIL, SmemLayoutA, 0>::Result;
//   using S2RTiledCopyBAtom =
//       typename S2RTiledCopySelector<T, LDMATRIX_AVAIL, SmemLayoutB, 1>::Result;
//   using R2STiledCopyCAtom =
//       typename R2STiledCopySelector<T, STMATRIX_AVAIL, SmemLayoutC>::Result;

//   using S2RTiledCopyA =
//       decltype(make_tiled_copy_A(S2RTiledCopyAAtom{}, TiledMMA{}));
//   using S2RTiledCopyB =
//       decltype(make_tiled_copy_B(S2RTiledCopyBAtom{}, TiledMMA{}));
//   using R2STiledCopyC =
//       decltype(make_tiled_copy_C(R2STiledCopyCAtom{}, TiledMMA{}));

//   static __device__ __forceinline__ void run(T *__restrict__ c_ptr,
//                                              T const *__restrict__ a_ptr,
//                                              T const *__restrict__ b_ptr,
//                                              int thread_idx) {
//     Tensor sA = make_tensor(make_smem_ptr(a_ptr), SmemLayoutA{}); // [M, K, B]
//     Tensor sB = make_tensor(make_smem_ptr(b_ptr), SmemLayoutB{}); // [N, K, B]
//     Tensor sC = make_tensor(make_smem_ptr(c_ptr), SmemLayoutC{}); // [M, N, B]
//     CUTE_STATIC_ASSERT_V(rank(sA) == _3{});

//     TiledMMA tiled_mma;
//     ThrMMA thr_mma = tiled_mma.get_slice(thread_idx);
//     Tensor mma_rA =
//         thr_mma.partition_fragment_A(sA(_, _, _0{})); // (MMA, MMA_M, MMA_K)
//     Tensor mma_rB =
//         thr_mma.partition_fragment_B(sB(_, _, _0{})); // (MMA, MMA_N, MMA_K)
//     Tensor mma_rC =
//         thr_mma.partition_fragment_C(sC(_, _, _0{})); // (MMA, MMA_M, MMA_N)
//     clear(mma_rC);

//     S2RTiledCopyA s2r_tiled_copy_A;
//     ThrCopy s2r_tiled_copy_A_thr = s2r_tiled_copy_A.get_slice(thread_idx);
//     Tensor s2r_sA =
//         s2r_tiled_copy_A_thr.partition_S(sA); // (S2R, S2R_M, S2R_K, B)
//     Tensor s2r_rA =
//         s2r_tiled_copy_A_thr.retile_D(mma_rA); // (S2R, S2R_M, S2R_K)

//     S2RTiledCopyB s2r_tiled_copy_B;
//     ThrCopy s2r_tiled_copy_B_thr = s2r_tiled_copy_B.get_slice(thread_idx);
//     Tensor s2r_sB =
//         s2r_tiled_copy_B_thr.partition_S(sB); // (S2R, S2R_N, S2R_K, B)
//     Tensor s2r_rB =
//         s2r_tiled_copy_B_thr.retile_D(mma_rB); // (S2R, S2R_N, S2R_K)

//     R2STiledCopyC r2s_tiled_copy_C;
//     ThrCopy r2s_tiled_copy_C_thr = r2s_tiled_copy_C.get_slice(thread_idx);
//     Tensor r2s_rC =
//         r2s_tiled_copy_C_thr.retile_S(mma_rC); // (R2S, R2S_M, R2S_N)
//     Tensor r2s_sC =
//         r2s_tiled_copy_C_thr.partition_D(sC); // (R2S, R2S_M, R2S_N, B)

//     // CUTE_STATIC_ASSERT_V(shape<1>(s2r_rA) == shape<1>(mma_rA));
//     // CUTE_STATIC_ASSERT_V(shape<1>(s2r_rA) == shape<2>(s2r_rB));
//     static constexpr int NUM_MMA_STAGES = shape<1>(s2r_sA);

//     if (thread0()) {
//       println(thr_mma);
//       println(sA);
//       println(sB);
//       println(sC);
//       println(mma_rA);
//       println(mma_rB);
//       println(s2r_rA);
//       println(s2r_rB);
//     }
//     // copy(s2r_tiled_copy_A, s2r_sA(_, _, _0{}, _0{}), s2r_rA(_, _, _0{}));
//     // copy(s2r_tiled_copy_B, s2r_sB(_, _0{}, _, _0{}), s2r_rB(_, _0{}, _));

//     // #pragma unroll
//     // for (int i_k = 0; i_k < NUM_MMA_STAGES; ++i_k) {
//     // 	gemm(tiled_mma, mma_rC, mma_rA(_, _, i_k), mma_rB(_, _, i_k));
//     // 	if (i_k+1 != NUM_MMA_STAGES) {
//     // 		copy(s2r_tiled_copy_A, s2r_sA(_, _, i_k+1, _0{}), s2r_rA(_, _,
//     // i_k+1)); 		copy(s2r_tiled_copy_B, s2r_sB(_, _, i_k+1, _0{}), s2r_rB(_, _,
//     // i_k+1));
//     // 	}
//     // }

//     // copy(r2s_tiled_copy_C, r2s_rC, r2s_sC(_, _, _, _0{}));
//   }
// };

// } // namespace tb
