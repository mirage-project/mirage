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
          class ClusterShape_MNK_,
          class TiledMMA_,
          class MmaTiler_MNK_,
          class DstPipeLayout_A_,
          class DstPipeLayout_B_>
struct Blackwell_Matmul {
public:
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutA_{}) == _2{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutB_{}) == _2{});
  CUTE_STATIC_ASSERT_V(rank(SmemLayoutC_{}) == _2{});

  using ClusterShape_MNK = ClusterShape_MNK_;
  using TiledMMA = TiledMMA_;
  using MmaTiler_MNK = MmaTiler_MNK_;
  using DstPipeLayout_A = DstPipeLayout_A_;
  using DstPipeLayout_B = DstPipeLayout_B_;

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


  static constexpr int PIPELINE_STAGE_A = IS_PIPELINE_A ? PIPELINE_STAGES : 1;
  static constexpr int PIPELINE_STAGE_B = IS_PIPELINE_B ? PIPELINE_STAGES : 1;

  // // Pre-partitioned Tile Shape (MmaTile_M, MmaTile_K) to post-partitioned (MmaA, NumMma_M, NumMma_K)
  using MmaShape_A = decltype(partition_shape_A(TiledMMA{}, make_shape(size<0>(MmaTiler_MNK{}), size<2>(MmaTiler_MNK{}))));
  using MmaShape_B = decltype(partition_shape_B(TiledMMA{}, make_shape(size<1>(MmaTiler_MNK{}), size<2>(MmaTiler_MNK{}))));
  using MmaShape_C = decltype(partition_shape_C(TiledMMA{}, make_shape(size<0>(MmaTiler_MNK{}), size<1>(MmaTiler_MNK{}))));

  using R2STiledCopyCSelector =
      R2STiledCopySelector<T, IS_STMATRIX_AVAIL, SmemLayoutC>;
  using R2STiledCopyCAtom = typename R2STiledCopyCSelector::Result;
  static constexpr R2STiledCopyType R2S_TILED_COPY_C_TYPE =
      R2STiledCopyCSelector::TYPE;
  using R2STiledCopyC =
      decltype(make_tiled_copy_C(R2STiledCopyCAtom{}, TiledMMA{}));

  static __device__ __forceinline__
  auto get_cluster_layout() {
    return tiled_divide(make_layout(ClusterShape_MNK{}),
                       make_tile(typename TiledMMA::AtomThrID{}));
  }

  static __device__ __forceinline__
  auto get_mma_coord(int blockIdx_x, int blockIdx_y) {
    auto cluster_layout = get_cluster_layout();
    return make_coord(
        blockIdx_x % size<0>(cluster_layout),
        blockIdx_x / size<0>(cluster_layout),
        blockIdx_y,
        _);
  }

  static __device__ __forceinline__ auto get_cta_mma(int blockIdx_x, int blockIdx_y) {
    TiledMMA tiled_mma;
    auto mma_coord_vmnk = get_mma_coord(blockIdx_x, blockIdx_y);
    auto mma_v = get<0>(mma_coord_vmnk);
    return tiled_mma.get_slice(mma_v);
  }

  static __device__ __forceinline__
  auto get_mma_tC(int blockIdx_x, int blockIdx_y, uint32_t &tmem_base_ptr)
  {
    auto mma_coord_vmnk = get_mma_coord(blockIdx_x, blockIdx_y);
    auto mma_v = get<0>(mma_coord_vmnk);
    TiledMMA tiled_mma;
    ThrMMA cta_mma = tiled_mma.get_slice(mma_v);

    Tensor dummy_gC = make_tensor(make_gmem_ptr((float*)nullptr), SmemLayoutC{});
    auto tCgC = cta_mma.partition_C(dummy_gC);

    Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);
    tCtAcc.data() = tmem_base_ptr;

    return tCtAcc;
  }

  template<class TmemAccTensor>
  static __device__ __forceinline__
  void write_tC_to_sC(float *__restrict__ c_ptr,
                      TmemAccTensor const& tCtAcc,
                         int thread_idx)
  {
    auto cta_mma = get_cta_mma(blockIdx.x, blockIdx.y);

    using LdAtom = SM100_TMEM_LOAD_32dp32b1x;               
    // if (block(0) && thread_idx == 0) {
    //   printf("\n tCtAcc:\n");
    //   print(tCtAcc);
    //   printf("\n c_ptr: %p\n", c_ptr);
      // printf("\n tCtAcc tensor: \n");
      // print_tensor(tCtAcc);
    // }
    
    
    auto tiled_t2r_copy = make_tmem_copy(LdAtom{}, tCtAcc);
    auto thr_t2r_copy = tiled_t2r_copy.get_slice(thread_idx);

    auto tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
    Tensor sC = make_tensor(make_smem_ptr(c_ptr), SmemLayoutC{});
    Tensor tCsC = cta_mma.partition_C(sC);
    Tensor tDsC = thr_t2r_copy.partition_D(tCsC); 
    auto tDrAcc = make_tensor<float>(shape(tDsC));

    copy(tiled_t2r_copy, tDtAcc, tDrAcc);
    
    // copy(tDrAcc, tDsC);
    
    if (block(0) && thread_idx == 0) {
      printf("Completed TMEM->SMEM copy\n");
    }
  }

  template <class AccumRegFrag>
  static __device__ __forceinline__ void write_back_mma_rC(
      T *__restrict__ c_ptr, AccumRegFrag const &mma_rC, int thread_idx) {
    // if (thread_idx >= TILED_MMA_NUM_THREADS) {
    //   return;
    // }

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

  // a_ptr, b_ptr are from smem, mma_tC is from tmem
  template<class TmemAccTensor>
  static __device__ __forceinline__
  void run(TmemAccTensor &mma_tC,      
           T *__restrict__ a_ptr,
           T *__restrict__ b_ptr,
           int k_iter,
           TiledMMA &tiled_mma,
           int read_stage = 0)  
  {

    if (k_iter == 0) {
      tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;
    }
    
    auto mma_coord_vmnk = get_mma_coord(blockIdx.x, blockIdx.y);  

    auto mma_v = get<0>(mma_coord_vmnk);
    auto cta_mma = tiled_mma.get_slice(mma_v);

    Tensor tCsA = make_tensor(make_smem_ptr(a_ptr), DstPipeLayout_A{});
    Tensor tCsB = make_tensor(make_smem_ptr(b_ptr), DstPipeLayout_B{});

    Tensor tCrA = cta_mma.make_fragment_A(tCsA);
    Tensor tCrB = cta_mma.make_fragment_B(tCsB);

    // if (block(0) && threadIdx.x == 0) {
    //   printf("In Matmul run: a_ptr=%p, b_ptr=%p, tCsA shape=", a_ptr, b_ptr);
    //   print(shape(tCsA));
    //   printf(", tCsB shape=");
    //   print(shape(tCsB));
    //   printf(", tCrA shape=");
    //   print(shape(tCrA));
    //   printf(", tCrB shape=");
    //   print(shape(tCrB));
    //   printf("\n");
    //   printf("\n DstPipeLayout_A: \n");
    //   print(DstPipeLayout_A{});
    //   printf("\n DstPipeLayout_B: \n");
    //   print(DstPipeLayout_B{});
    //   printf("\n");
      // printf("\n tCrA(_,_,0): \n");
      // print(tCrA(_,_,0));
      // printf("\n tCrB(_,_,0): \n");
      // print(tCrB(_,_,0));
      // printf("\n");
    // }

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
