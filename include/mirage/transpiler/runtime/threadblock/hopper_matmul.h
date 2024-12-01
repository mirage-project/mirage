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

#include <cutlass/arch/reg_reconfig.h>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <threadblock/matmul.h>
#include <threadblock/input.h>
using namespace cute;

#include "element_unary.h"

namespace tb {


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
          bool IS_STORE_ACCUM,
          class MainloopPipeline, 
          class PipelineState>
class Hopper_Matmul {
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


//   static constexpr GMMA::Major GmmaMajorA = (stride<0>(SmemLayoutA{}) == _1{} ? GMMA::Major::MN : GMMA::Major::K);
//   static constexpr GMMA::Major GmmaMajorB = (stride<0>(SmemLayoutB{}) == _1{} ? GMMA::Major::MN : GMMA::Major::K);
//   using TileALayout = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
//       GmmaMajorA, half_t, decltype(get<0>(SmemLayoutA{})), decltype(get<1>(SmemLayoutA{}))>());

//   using TileBLayout = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
//       GmmaMajorB, half_t, decltype(get<0>(SmemLayoutB{})), decltype(get<1>(SmemLayoutB{}))>());

// Sw<3,4,3> o smem_ptr[16b](unset) o (_8,_64):(_64,_1)
//64 64, 64, 1
// hape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>
//(composition(Swizzle<3, 4, 3>{}, Layout<Shape<Int<64>, Int<64>>, Stride<Int<1>, Int<64>>>{}));

  using TileALayout = GMMA::Layout_K_SW128_Atom<T>;

  using TileBLayout = GMMA::Layout_MN_SW128_Atom<T>;

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
      SM90_64x64x16_F16F16F16_SS<GMMA::Major::K, GMMA::Major::MN>{}));

//   using TiledMMA = decltype(make_tiled_mma(
//       SM90_64x64x16_F16F16F16_SS<GmmaMajorA, GmmaMajorB>{}));

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
   
//    auto sC_l = tile_to_shape(
//         SmemLayoutC{},
//         make_shape(shape<0>(SmemLayoutC{}), shape<1>(SmemLayoutC{}), Int<2>{}), Step<_1, _2, _3>{});
    Tensor sC_fake =
        make_tensor(make_smem_ptr((half_t *)nullptr), SmemLayoutC{});

    TiledMMA tiled_mma;
    ThrMMA thr_mma = tiled_mma.get_slice(0);
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
    // if(threadIdx.x == 0 && blockIdx.x == 0){
    //   print("sC: ");
    //   print_tensor(sC);
    //   print("\n");
    // }
  }

  template <typename MMARc>
  static __device__ __forceinline__ void
      run(MMARc &mma_rC,
          T *__restrict__ a_ptr, // Do not define a_ptr and b_ptr as const here,
                                 // since we may pad remaining part on the
                                 // k-axis with 0
          T *__restrict__ b_ptr,
          char const *__restrict__ smem_allzero_ptr,
          int thread_idx,
          MainloopPipeline pipeline,
          PipelineState& smem_pipe_read) {
    // TiledMMA tiled_mma;
    // // cutlass::arch::warpgroup_reg_dealloc<232>();

    // auto sA_l = tile_to_shape(
    //     TileALayout{},
    //     make_shape(shape<0>(SmemLayoutA{}), shape<1>(SmemLayoutA{}), Int<2>{}), Step<_1, _2, _3>{});

    // auto sB_l = tile_to_shape(
    //     TileBLayout{},
    //     make_shape(shape<0>(SmemLayoutB{}), shape<1>(SmemLayoutB{}), Int<2>{}), Step<_1, _2, _3>{});

    // //  auto sA_l = tile_to_shape(
    // //     SmemLayoutA{},
    // //     make_shape(shape<0>(SmemLayoutA{}), shape<1>(SmemLayoutA{}), Int<2>{}), Step<_1, _2, _3>{});

    // // auto sB_l = tile_to_shape(
    // //     SmemLayoutB{},
    // //     make_shape(shape<0>(SmemLayoutB{}), shape<1>(SmemLayoutB{}), Int<2>{}), Step<_1, _2, _3>{});


    // Tensor sA = make_tensor(make_smem_ptr(a_ptr), sA_l); // [M, K]
    // Tensor sB = make_tensor(make_smem_ptr(b_ptr), sB_l); // [N, K]

    // ThrMMA thr_mma = tiled_mma.get_thread_slice(0);
    // Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
    // Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

    // Tensor tCrA = thr_mma.make_fragment_A(tCsA); // (MMA,MMA_M,MMA_K,PIPE)
    // Tensor tCrB = thr_mma.make_fragment_B(tCsB); // (MMA,MMA_N,MMA_K,PIPE)

    //  auto consumer_wait = [](auto& pipeline, auto& smem_pipe_read) {
    //         auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
    //         pipeline.consumer_wait(smem_pipe_read, barrier_token);
    // };
    
    // // auto k_tile_count = 64; 
    // // CUTLASS_PRAGMA_NO_UNROLL
    // //   for (; k_tile_count > 0; --k_tile_count){
    
    //  if(threadIdx.x == 0 && blockIdx.x == 0){
    //   printf("pre read\n");
    //   print(SmemLayoutA{});
    //   print("\n");
    //   print(SmemLayoutA_{});
    //   print("\n");
      
    // }
    // // PipelineState smem_pipe_release = smem_pipe_read;
    // consumer_wait(pipeline, smem_pipe_read);
    // int read_stage = smem_pipe_read.index();
    // // if(threadIdx.x == 0 && blockIdx.x == 0){
    // //   printf("tCsA stage: %d\n", read_stage);
    // //   print(tCsA);
    // //   print("\n");
    // //   // print(sB_l);
    // //   // print("\n");
      
    // // }
    
    // // warpgroup_fence_operand(mma_rC);
    // cute::warpgroup_arrive();
    // gemm(tiled_mma, tCrA(_, _, _, read_stage), tCrB(_, _, _, read_stage), mma_rC);
    // cute::warpgroup_commit_batch();
    // cute::warpgroup_wait<0>();
    // // warpgroup_fence_operand(mma_rC);

    // // if(blockIdx.x == 0 && threadIdx.x == 0){
    // //   print("mma_rC: %d\n", threadIdx.x);
    // //   print(TileALayout{});
    // //   print("\n");
    // //   print(TileBLayout{});
    // //   print("\n");
    // //   // print(stride<1>(SmemLayoutA{}));
    // //   // print("\n");
    // //   // print(SmemLayoutA{});
    // //   // print("\n");
    // // //   print(GmmaMajorA);
    // // //   print("\n");
    // // //   print(GmmaMajorB);
    // // //   print("\n");
    // // }

    
    // pipeline.consumer_release(smem_pipe_read);
    // ++smem_pipe_read;
  
}
};

} // namespace tb
