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
#include <threadblock/utils.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>


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
          class ClusterShape_MNK_,
          class TiledMMA_,
          class MmaTiler_MNK_>
struct Blackwell_Matmul {
public:
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutA_{}) == _2{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutB_{}) == _2{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutC_{}) == _2{});

  using ClusterShape_MNK = ClusterShape_MNK_;
  using TiledMMA = TiledMMA_;
  using MmaTiler_MNK = MmaTiler_MNK_;

  static constexpr UMMA::Major UmmaMajorA = UMMA::Major::K;
  static constexpr UMMA::Major UmmaMajorB = UMMA::Major::K;

  static constexpr int PIPELINE_STAGE_A = IS_PIPELINE_A ? PIPELINE_STAGES : 1;
  static constexpr int PIPELINE_STAGE_B = IS_PIPELINE_B ? PIPELINE_STAGES : 1;


  using SmemLayoutAtom_A =
      decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
               UmmaMajorA,
               T,                                                   
               decltype(get<0>(MmaTiler_MNK{})),
               decltype(get<2>(MmaTiler_MNK{}))>());     
  using DstMNKLayout_A = decltype(
    partition_shape_A(TiledMMA{},
    make_shape(shape<0>(MmaTiler_MNK{}), shape<2>(MmaTiler_MNK{}))));

  using DstPipeLayout_A = decltype(UMMA::tile_to_mma_shape(
          SmemLayoutAtom_A{},
          append(DstMNKLayout_A{},
                 Int<PIPELINE_STAGE_A>{}),
          Step<_1,_2,_3>{}));

  using SmemLayoutAtom_B =
      decltype(cutlass::gemm::collective::detail::sm100_smem_selector<
               UmmaMajorB,
               T,                                                   
               decltype(get<1>(MmaTiler_MNK{})),
               decltype(get<2>(MmaTiler_MNK{}))>());     
  using DstMNKLayout_B = decltype(
    partition_shape_B(TiledMMA{},
    make_shape(shape<1>(MmaTiler_MNK{}), shape<2>(MmaTiler_MNK{}))));

  using DstPipeLayout_B = decltype(UMMA::tile_to_mma_shape(
          SmemLayoutAtom_B{},
          append(DstMNKLayout_B{},
                 Int<PIPELINE_STAGE_B>{}),
          Step<_1,_2,_3>{}));

  using SmemLayoutA = typename Dim01Swapper<SmemLayoutA_>::Result; // [M, K]
  using SmemLayoutB = SmemLayoutB_;                                // [N, K]
  using SmemLayoutC = typename Dim01Swapper<SmemLayoutC_>::Result; // [M, N]

  using M = decltype(get<0>(shape(SmemLayoutA{})));
  using K = decltype(get<1>(shape(SmemLayoutA{})));
  using N = decltype(get<0>(shape(SmemLayoutB{})));

  
  static constexpr int global_N = size<1>(ClusterShape_MNK{}) * size<1>(SmemLayoutC{});
  using GmemStrideTypeC = Stride<Int<global_N>, Int<1>>; 

  using FusionOp = cutlass::epilogue::fusion::FusionOperation;

  // Use the sm100_get_tmem_load_op function to automatically select the optimal tmem load operation
  using TMemLoadOp = 
      decltype(cutlass::epilogue::collective::detail::sm100_get_tmem_load_op<
          GmemStrideTypeC,
          T, 
          T, 
          Shape<Int<32>, Int<128>>,
          FusionOp>()
      );



  static __device__ __forceinline__
  auto get_mma_tC(int blockIdx_x, int blockIdx_y, uint32_t &tmem_base_ptr)
  {
    auto cta_mma = get_cta_mma<TiledMMA, ClusterShape_MNK>(blockIdx.x, blockIdx.y);

    Tensor dummy_gC = make_tensor(make_gmem_ptr((T*)nullptr), SmemLayoutC{});
    auto tCgC = cta_mma.partition_C(dummy_gC);

    Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);
    tCtAcc.data() = tmem_base_ptr;

    return tCtAcc;
  }


  template<class TmemAccTensor>
  static __device__ __forceinline__
  void write_tC_to_gC(T *__restrict__ c_ptr,
                      TmemAccTensor const& tCtAcc,
                         int thread_idx)
  {
    // only one warp group is used for Tmem load
    if (thread_idx >= mirage::config::NUM_THREADS_PER_GROUP) {
      return;
    }
    
    TiledCopy tiled_t2r_copy = make_tmem_copy(TMemLoadOp{}, tCtAcc);
    ThrCopy   thr_t2r_copy   = tiled_t2r_copy.get_slice(threadIdx.x);

    auto mma_coord_vmnk = get_mma_coord_vmnk<TiledMMA, ClusterShape_MNK>(blockIdx.x, blockIdx.y);  
    auto mma_v = get<0>(mma_coord_vmnk);

    TiledMMA tiled_mma;
    MmaTiler_MNK mma_tiler;
    auto cta_mma = tiled_mma.get_slice(mma_v);
    auto gC = make_tensor(make_gmem_ptr(c_ptr), SmemLayoutC{});

    auto tCgC = cta_mma.partition_C(gC);         // (MmaC, NumMma_M, NumMma_N)
    auto tCgD = cta_mma.partition_C(gC);         // (MmaC, NumMma_M, NumMma_N)

    auto tDgC = thr_t2r_copy.partition_D(tCgC);                   // (CpyD, NumCpy_M, NumCpy_N)
    auto tDrC = make_fragment_like(tDgC);                         // (CpyD, NumCpy_M, NumCpy_N)
    // Load C tensor GMEM -> RMEM
    copy(tDgC, tDrC);

    auto tDtAcc = thr_t2r_copy.partition_S(tCtAcc);               // (CpyS, NumCpy_M, NumCpy_N)
    auto tDgD   = thr_t2r_copy.partition_D(tCgD);                 // (CpyD, NumCpy_M, NumCpy_N)
    auto tDrAcc = make_tensor<T>(shape(tDgD));              // (CpyD, NumCpy_M, NumCpy_N)
    // Load TMEM -> RMEM
    copy(tiled_t2r_copy, tDtAcc, tDrAcc);

    // AXPBY RMEM -> RMEM: tDrC = alpha * tDrAcc + beta * tDrC
    axpby(1.0f, tDrAcc, 0.0f, tDrC);

    copy(tDrC, tDgD);
  }


  // a_ptr, b_ptr are from smem, mma_tC is from tmem
  template<class TmemAccTensor, class BlackwellAsyncPipeline_A, class BlackwellAsyncPipeline_B>
  static __device__ __forceinline__
  void run(TmemAccTensor &mma_tC,      
           T *__restrict__ a_ptr,
           T *__restrict__ b_ptr,
           int k_iter,
           TiledMMA tiled_mma,
           int read_stage,
           BlackwellAsyncPipeline_A &blackwell_async_pipeline_20000012,
           BlackwellAsyncPipeline_B &blackwell_async_pipeline_20000013)  
  {
    if (k_iter == 0) {
      tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    }
    
    auto mma_coord_vmnk = get_mma_coord_vmnk<TiledMMA, ClusterShape_MNK>(blockIdx.x, blockIdx.y);  

    auto mma_v = get<0>(mma_coord_vmnk);
    auto cta_mma = tiled_mma.get_slice(mma_v);

    Tensor tCsA = make_tensor(make_smem_ptr(a_ptr), DstPipeLayout_A{});
    Tensor tCsB = make_tensor(make_smem_ptr(b_ptr), DstPipeLayout_B{});

    Tensor tCrA = cta_mma.make_fragment_A(tCsA);
    Tensor tCrB = cta_mma.make_fragment_B(tCsB);

    if (warp_id() == 0) {
      // Execute a MmaTile_M x MmaTile_N x MmaTile_K GEMM
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
          gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), mma_tC);
          tiled_mma.accumulate_ = UMMA::ScaleOut::One;
      }
    }
  }
};



} // namespace tb
