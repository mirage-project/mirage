// blackwell_matmul_pipeline.h
#pragma once

#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cutlass/arch/reg_reconfig.h>
#include <threadblock/input.h>
#include <threadblock/matmul.h>
using namespace cute;

#include "element_unary.h"

namespace tb {

template<typename T,
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
          bool IS_STORE_ACCUM,
          bool IS_COORPERATIVE,
          bool IS_PIPELINE_A,
          bool IS_PIPELINE_B,
          int PIPELINE_STAGES,
          class ClusterShape_MNK,
          class TiledMMA>
struct Blackwell_Matmul {
public:
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutA_{}) == _2{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutB_{}) == _2{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutC_{}) == _2{});

  using SmemLayoutA = typename Dim01Swapper<SmemLayoutA_>::Result; // [M, K]
  using SmemLayoutB = SmemLayoutB_;                                // [N, K]
  using SmemLayoutC = typename Dim01Swapper<SmemLayoutC_>::Result; // [M, N]

  // NT	M-major	(M,K):(1,ldA)	N-major	(N,K):(1,ldB)
  // TN	K-major	(M,K):(ldA,1)	K-major	(N,K):(ldB,1)
  // NN	M-major	(M,K):(1,ldA)	K-major	(N,K):(ldB,1)
  // TT	K-major	(M,K):(ldA,1)	N-major	(N,K):(1,ldB)
  static constexpr UMMA::Major UmmaMajorA = UMMA::Major::K;
  static constexpr UMMA::Major UmmaMajorB = UMMA::Major::MN;
  using TileALayout =
      decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
               UmmaMajorA,
               T,
               decltype(get<0>(SmemLayoutA{})),
               decltype(get<1>(SmemLayoutA{}))>());
  using TileBLayout =
      decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
               UmmaMajorB,
               T,
               decltype(get<0>(SmemLayoutB{})),
               decltype(get<1>(SmemLayoutB{}))>());

  using M = decltype(get<0>(shape(SmemLayoutA{})));
  using K = decltype(get<1>(shape(SmemLayoutA{})));
  using N = decltype(get<0>(shape(SmemLayoutB{})));

  /*------------- UMMA atom for Blackwell -----------------*/
  // using Atom = SM100_MMA_F16BF16_SS<T, T, T, 128, 256,
  //                                UMMA::Major::K, UMMA::Major::K>;
  
  // using AtomLayoutMNK = cute::conditional_t<IS_COORPERATIVE || ENABLE_PAIR_UMMA,
  //                                           Layout<Shape<_2, _1, _1>>,
  //                                           Layout<Shape<_1, _1, _1>>>;

  // CUTE_STATIC_ASSERT_V(rank(AtomLayoutMNK{}) == _3{});
  // only support half_t for now
  static_assert(std::is_same_v<T, half_t>, "Only half_t is supported");
  
  // using UmmaAtom = cute::conditional_t<
  //     ENABLE_PAIR_UMMA,
  //     SM100_MMA_F16BF16_2x1SM_SS<T, T, T, M{}, N{}, UmmaMajorA, UmmaMajorB>,
  //     SM100_MMA_F16BF16_SS<T, T, T, M{}, N{}, UmmaMajorA, UmmaMajorB>
  // >;

  // using TiledMMA = decltype(cute::make_tiled_mma(UmmaAtom{}, AtomLayoutMNK{}));

  /*------------- SMEM layout: extra stage dim ----------------------*/
  using SAstage = decltype(tile_to_shape(SmemLayoutA{},
                        cute::make_shape(shape<0>(SmemLayoutA{}),
                                         shape<1>(SmemLayoutA{}),
                                         cute::Int<PIPELINE_STAGES>{}),
                        cute::Step<_1,_2,_3>{}));
  using SBstage = decltype(tile_to_shape(SmemLayoutB{},
                        cute::make_shape(shape<0>(SmemLayoutB{}),
                                         shape<1>(SmemLayoutB{}),
                                         cute::Int<PIPELINE_STAGES>{}),
                        cute::Step<_1,_2,_3>{}));

  using R2STiledCopyCSelector =
      R2STiledCopySelector<T, IS_STMATRIX_AVAIL, SmemLayoutC>;
  using R2STiledCopyCAtom = typename R2STiledCopyCSelector::Result;
  static constexpr R2STiledCopyType R2S_TILED_COPY_C_TYPE =
      R2STiledCopyCSelector::TYPE;
  using R2STiledCopyC =
      decltype(make_tiled_copy_C(R2STiledCopyCAtom{}, TiledMMA{}));

  // static UMMA::ScaleOut accumulate_ = UMMA::ScaleOut::Zero;

  static __device__ __forceinline__
  auto get_mma_tC(int blockIdx_x, int blockIdx_y)
  {
    Layout cluster_layout_vmnk = tiled_divide(make_layout(ClusterShape_MNK{}),
                                         make_tile(typename TiledMMA::AtomThrID{}));
    auto mma_coord_vmnk = make_coord(
                   blockIdx_x % size<0>(cluster_layout_vmnk), // Peer CTA coordinate
                   blockIdx_x / size<0>(cluster_layout_vmnk), //    MMA-M coordinate
                   blockIdx_y,                                //    MMA-N coordinate
                   _);                                        //    MMA-K coordinate
 
    auto mma_v = get<0>(mma_coord_vmnk);
    TiledMMA tiled_mma;
    ThrMMA cta_mma = tiled_mma.get_slice(mma_v);
    
    Tensor dummy_gC = make_tensor(make_gmem_ptr((T*)nullptr), SmemLayoutC{});
    auto tCgC = cta_mma.partition_C(dummy_gC);

    Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);

    return tCtAcc;
  }


  template<class TmemAccTensor>
  static __device__ __forceinline__
  void write_back_mma_tC(T * __restrict__ c_ptr,      
                         TmemAccTensor const& tCtAcc, 
                         int thread_idx)
  {
    TiledMMA tiled_mma;
    auto mma_coord_vmnk = make_coord(
                   blockIdx.x % size<0>(tiled_divide(make_layout(ClusterShape_MNK{}),
                                         make_tile(typename TiledMMA::AtomThrID{}))),
                   blockIdx.x / size<0>(tiled_divide(make_layout(ClusterShape_MNK{}),
                                         make_tile(typename TiledMMA::AtomThrID{}))),
                   blockIdx.y,
                   _);
    auto mma_v = get<0>(mma_coord_vmnk);
    auto cta_mma = tiled_mma.get_slice(mma_v);

    // tc -> rC
    Tensor rC = cta_mma.partition_fragment_C(
                  make_tensor(make_smem_ptr((T*)nullptr), SmemLayoutC{}));

    using LdAtom = SM100_TMEM_LOAD_32dp32b1x;               
    auto tiled_ld = make_tmem_copy(LdAtom{}, tCtAcc);
    auto thr_ld = tiled_ld.get_slice(thread_idx);

    copy(tiled_ld,                        
         thr_ld.partition_S(tCtAcc),         
         thr_ld.partition_D(rC));          

    R2STiledCopyC r2s;                     
    auto r2s_thr = r2s.get_slice(thread_idx);

    Tensor r2s_rC = r2s_thr.retile_S(rC); 
    Tensor r2s_sC = r2s_thr.partition_D(
                    make_tensor(make_smem_ptr(c_ptr), SmemLayoutC{}));

    r2s_copy_with_oob_protection<T,
                               M, N,
                               /*NumExp*/ 0,
                               /*StoreAccum?*/ false>(
        r2s, r2s_rC, r2s_sC, thread_idx);
  }

  // a_ptr, b_ptr are from smem, mma_tC is from tmem
  template<class TmemAccTensor>
  static __device__ __forceinline__
  void run(TmemAccTensor &mma_tC,      
           T *__restrict__ a_ptr,
           T *__restrict__ b_ptr,
           int thread_idx,
           int read_stage = 0)  
  {
    TiledMMA tiled_mma;
    // tiled_mma.accumulate_ = accumulate_;

    auto sA_l = tile_to_shape(TileALayout{},
                            make_shape(shape<0>(SmemLayoutA{}),
                                       shape<1>(SmemLayoutA{}),
                                       Int<PIPELINE_STAGES>{}),
                            Step<_1,_2,_3>{});

    auto sB_l = tile_to_shape(TileBLayout{},
                            make_shape(shape<0>(SmemLayoutB{}),
                                       shape<1>(SmemLayoutB{}),
                                       Int<PIPELINE_STAGES>{}),
                            Step<_1,_2,_3>{});

    Tensor sA = make_tensor(make_smem_ptr(a_ptr), sA_l);
    Tensor sB = make_tensor(make_smem_ptr(b_ptr), sB_l);

    // 获取CTA在集群中的排名和坐标
    auto cluster_layout_vmnk = tiled_divide(make_layout(ClusterShape_MNK{}),
                                         make_tile(typename TiledMMA::AtomThrID{}));
    auto mma_coord_vmnk = make_coord(
                   blockIdx.x % size<0>(cluster_layout_vmnk), // Peer CTA coordinate
                   blockIdx.x / size<0>(cluster_layout_vmnk), //    MMA-M coordinate
                   blockIdx.y,                                //    MMA-N coordinate
                   _);                                        //    MMA-K coordinate

    auto mma_v = get<0>(mma_coord_vmnk);
    auto cta_mma = tiled_mma.get_slice(mma_v);

    Tensor tCsA = cta_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = cta_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

    Tensor tCrA = cta_mma.make_fragment_A(tCsA);
    Tensor tCrB = cta_mma.make_fragment_B(tCsB);

    gemm(tiled_mma,
        tCrA(_,_,_, IS_PIPELINE_A ? read_stage : 0),      
        tCrB(_,_,_, IS_PIPELINE_B ? read_stage : 0),
        mma_tC); 

    cute::warpgroup_commit_batch();
    cute::warpgroup_wait<0>();

    // accumulate_ = UMMA::ScaleOut::One;
  }
};



} // namespace tb
