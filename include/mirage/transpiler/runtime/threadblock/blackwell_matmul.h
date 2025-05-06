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
          int PIPELINE_STAGES>               // #K‑tiles per CTA
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

  // using ClusterShape = ClusterShape_MNK;


  using M = decltype(get<0>(shape(SmemLayoutA{})));
  using K = decltype(get<1>(shape(SmemLayoutA{})));
  using N = decltype(get<0>(shape(SmemLayoutB{})));
  // fix the type for now
  // using   TA      = half_t;
  // using   TB      = half_t;
  // using   TC      = float;

  /*------------- UMMA atom 128×256×16 (fp16·acc32) -----------------*/
  using Atom   = SM100_MMA_F16BF16_SS<T,T,T,128,256,
                                      UMMA::Major::K,UMMA::Major::K>;
  using TiledMMA = decltype(make_tiled_mma(Atom{}));

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

  /*------------- TMEM load atom (32‑lane×32‑bit×1) -----------------*/
  using LdAtom  = SM100_TMEM_LOAD_32dp32b1x;

  static constexpr int TILED_MMA_NUM_THREADS = thr_size(TiledMMA{});
  // static_assert(TILED_MMA_NUM_THREADS ==
  //               NUM_THREADS * (IS_COORPERATIVE ? 2 : 1));


  using R2STiledCopyCSelector =
      R2STiledCopySelector<T, IS_STMATRIX_AVAIL, SmemLayoutC>;
  using R2STiledCopyCAtom = typename R2STiledCopyCSelector::Result;
  static constexpr R2STiledCopyType R2S_TILED_COPY_C_TYPE =
      R2STiledCopyCSelector::TYPE;
  using R2STiledCopyC =
      decltype(make_tiled_copy_C(R2STiledCopyCAtom{}, TiledMMA{}));


static __device__ __forceinline__
auto get_mma_tC(int thread_idx)
{

constexpr auto accum_shape = cute::make_shape(
    cute::make_shape(Int<128>{}, Int<256>{}),  // 👈 [M, N]
    Int<1>{}                                   //    [Stage]
);
  
  using FrgTypeC = UMMA::tmem_frg_1sm<float>;

  auto tCtC = FrgTypeC::make(accum_shape);
  
  // ③ 可选清零（如果你的 TMEM 是 output 累加器，通常要 zero init）
  // clear(tCtC);

  return tCtC;
  
}


/////////////////////////////////////////////////////////////////////////////////
// ❷ Blackwell: 先用 tcgen05.ld 把累加器从 TMEM 拉到寄存器，再写回全局
/////////////////////////////////////////////////////////////////////////////////
template<class TmemAccTensor>
static __device__ __forceinline__
void write_back_mma_tC(T * __restrict__ c_ptr,       // 目标 GMEM
                       TmemAccTensor const& tCtC,        // TMEM 里的累加器
                       int thread_idx)
{
  TiledMMA tiled_mma;
  ThrMMA   thr_mma = tiled_mma.get_thread_slice(thread_idx % 128);

  Tensor dummy_sC = make_tensor(make_smem_ptr((T*)nullptr), SmemLayoutC{});
  Tensor rC       = thr_mma.partition_fragment_C(dummy_sC);   // Reg tensor
  clear(rC);
  
  using LdAtom   = SM100_TMEM_LOAD_32dp32b1x;               
  auto  tiled_ld = make_tmem_copy(LdAtom{}, tCtC);
  auto  thr_ld   = tiled_ld.get_slice(thread_idx);

  copy(tiled_ld,                        
       thr_ld.partition_S(tCtC),         
       thr_ld.partition_D(rC));          

  R2STiledCopyC r2s;                     
  ThrCopy       r2s_thr  = r2s.get_slice(thread_idx);

  Tensor r2s_rC = r2s_thr.retile_S(rC); 
  Tensor r2s_sC = r2s_thr.partition_D(
                    make_tensor(make_smem_ptr(c_ptr), SmemLayoutC{}));

  r2s_copy_with_oob_protection<T,
                               M, N,
                               /*NumExp*/ 0,
                               /*StoreAccum?*/ false>(
      r2s, r2s_rC, r2s_sC, thread_idx);
}

// a_ptr, b_ptr are from smem, tCtC is from tmem
template<class TmemAccTensor>
static __device__ __forceinline__
void run(TmemAccTensor                 &tCtC,        // ← 现在是 TMEM 句柄
         T *__restrict__              a_ptr,
         T *__restrict__              b_ptr,
         int                            thread_idx,
         int                            read_stage)  // 由 pipeline.consumer_wait() 得到
{

  TiledMMA tiled_mma;

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
  // here it is tCsA instead of sA
  Tensor tCsA = make_tensor(make_smem_ptr(a_ptr), sA_l);
  Tensor tCsB = make_tensor(make_smem_ptr(b_ptr), sB_l);

  // slice based on cta peer id
  ThrMMA cta_mma      = tiled_mma.get_slice(_0{});
//   Tensor tCsA         = cta_mma.partition_A(sA);               // (MMA,M,K,stage)
//   Tensor tCsB         = cta_mma.partition_B(sB);

  Tensor tCrA         = cta_mma.make_fragment_A(tCsA);
  Tensor tCrB         = cta_mma.make_fragment_B(tCsB);
  
  //------------------------------------------------------------------
  // 2. 仅 slice‑leader 线程发射一次 UMMA 指令
  //------------------------------------------------------------------
  /* 128‑thread slice 的第 0 号线程觉得自己是 leader */

    // suppose this is selected warp
    gemm(tiled_mma,
        tCrA(_,_,_, IS_PIPELINE_A ? read_stage : 0),        // 选中当前 SMEM stage
        tCrB(_,_,_, IS_PIPELINE_B ? read_stage : 0),
        tCtC);                          // 累加进 TMEM
  

  //------------------------------------------------------------------
  // 3. 所有线程等待本 slice 完结（WG 级 fence）
  //------------------------------------------------------------------
  cute::warpgroup_commit_batch();
  cute::warpgroup_wait<0>();

}
};



} // namespace tb
