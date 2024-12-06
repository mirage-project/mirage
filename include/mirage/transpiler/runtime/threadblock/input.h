// input.h - Implementation of threadblock level input operators
//
// We provide these implementations:
// - Non-chunked, synchronous copy
// - Chunked, synchronous copy
// - Chunked, asynchronous copy
// - Copy using the Tensor Memory Accelerator (TMA)
//
// "Chunked" means that the copy is performed in uint128_t-sized chunks, while
// "asynchronous" means that the copy is performed using asynchronous
// instructions like cp.async. The Mirage Transpiler will choose the best
// implementation based on the layout of the input tensors and the available
// hardware.

#pragma once

#include <cutlass/arch/reg_reconfig.h>
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/pipeline/pipeline.hpp"
#include "hopper_mainloop_params.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include <cstdint>
#include <cute/layout.hpp>
using namespace cute;

namespace tb {

template <class InputLayout>
class InputDim01Swapper {
  CUTE_STATIC_ASSERT_V(rank(InputLayout{}) == _2{});

  using A0 = decltype(get<0>(shape(InputLayout{})));
  using A1 = decltype(get<1>(shape(InputLayout{})));
  using TransposeCoordLayout = Layout<Shape<A1, A0>, Stride<A0, _1>>;
  using Result_ = decltype(composition(InputLayout{}, TransposeCoordLayout{}));

public:
  using Result =
      decltype(coalesce(Result_{}, Step<_1, _1>{})); // By-mode coalescing
};

// Type 1: Non-chunked, synchronous copy
template <typename T, class DstLayout, class SrcLayout, int NUM_THREADS>
class InputNonChunkedSyncCopy {
public:
  CUTE_STATIC_ASSERT_V(cute::size(SrcLayout{}) == cute::size(DstLayout{}));
  using Numel = decltype(cute::size(DstLayout{}));

  static __device__ __forceinline__ void
      run(T *__restrict__ dst, T const *__restrict__ src, int thread_idx) {
    constexpr auto numel = Numel{};
    auto dst_layout = DstLayout{};
    auto src_layout = SrcLayout{};
#pragma unroll
    for (int elem_idx = thread_idx; elem_idx < numel; elem_idx += NUM_THREADS) {
      T res = src[src_layout(elem_idx)];
      dst[dst_layout(elem_idx)] = res;
    }
  }
};

// Get a mapping from chunk coordinate to original coordinate
//
// Assume the shape of the tensor is (A0, A1, ...) (where A0 is the innermost
// dim), then if we merge elements in the same chunk into one element, the shape
// would be (A0', A1, ...), where A0' is the number of chunks in the innermost
// dim (ceil(A0 / CHUNK_SIZE)).
//
// This class returns such a mapping. It takes a coordinate in the "chunk space"
// and converts it to a coordinate in the "original space".
template <class InputLayout, int CHUNK_SIZE>
class GetChunkedCoord2Coord {
  using InputShape = decltype(shape(InputLayout{}));
  static constexpr int INNERMOST_DIM_SIZE = get<0>(InputShape{}).value;
  static constexpr int INNERMOST_DIM_NUM_CHUNKS =
      ceil_div(INNERMOST_DIM_SIZE, CHUNK_SIZE);

  using Result_ = decltype(make_layout(
      replace<0>(InputShape{}, Int<INNERMOST_DIM_NUM_CHUNKS>{}),
      replace<0>(stride(make_layout(InputShape{}, LayoutLeft{})),
                 Int<CHUNK_SIZE>{})));

public:
  using Result = decltype(coalesce(Result_{}));
};

// Type 2: Chunked, synchronous copy
// The real innermost dim should be the first dimension
template <typename T, class DstLayout, class SrcLayout, int NUM_THREADS>
class InputChunkedSyncCopy {
public:
  CUTE_STATIC_ASSERT_V(size(SrcLayout{}) == size(DstLayout{}));

  static constexpr int CHUNK_SIZE = 16 / sizeof(T);
  using SrcChunkedCoord2Coord =
      typename GetChunkedCoord2Coord<SrcLayout, CHUNK_SIZE>::Result;
  using DstChunkedCoord2Coord =
      typename GetChunkedCoord2Coord<DstLayout, CHUNK_SIZE>::Result;
  static constexpr int NUM_CHUNKS = size(SrcChunkedCoord2Coord{}).value;

  static __device__ __forceinline__ void
      run(T *__restrict__ dst, T const *__restrict__ src, int thread_idx) {
    auto src_layout = SrcLayout{};
    auto dst_layout = DstLayout{};
    auto src_chunked_coord2coord = SrcChunkedCoord2Coord{};
    auto dst_chunked_coord2coord = DstChunkedCoord2Coord{};
#pragma unroll
    for (int chunk_idx = thread_idx; chunk_idx < NUM_CHUNKS;
         chunk_idx += 128) {
      uint128_t res =
          *((uint128_t const *)(src + src_layout(
                                          src_chunked_coord2coord(chunk_idx))));
      *((uint128_t *)(dst + dst_layout(dst_chunked_coord2coord(chunk_idx)))) =
          res;
    }
  }
};

// Type 3: Chunked, asynchronous copy
template <typename T, class DstLayout, class SrcLayout, int NUM_THREADS>
class InputChunkedAsyncCopy {
public:
  CUTE_STATIC_ASSERT_V(size(SrcLayout{}) == size(DstLayout{}));

  static constexpr int CHUNK_SIZE = 16 / sizeof(T);
  using SrcChunkedCoord2Coord =
      typename GetChunkedCoord2Coord<SrcLayout, CHUNK_SIZE>::Result;
  using DstChunkedCoord2Coord =
      typename GetChunkedCoord2Coord<DstLayout, CHUNK_SIZE>::Result;
  static constexpr int NUM_CHUNKS = size(SrcChunkedCoord2Coord{}).value;

  static __device__ __forceinline__ void
      run(T *__restrict__ dst, T const *__restrict__ src, int thread_idx) {
    auto src_layout = SrcLayout{};
    auto dst_layout = DstLayout{};
    auto src_chunked_coord2coord = SrcChunkedCoord2Coord{};
    auto dst_chunked_coord2coord = DstChunkedCoord2Coord{};
    uint32_t dst_base_addr = cute::cast_smem_ptr_to_uint(dst);

#pragma unroll
    for (int chunk_idx = thread_idx; chunk_idx < NUM_CHUNKS;
         chunk_idx += NUM_THREADS) {
      size_t src_addr =
          (size_t)(src + src_layout(src_chunked_coord2coord(chunk_idx)));
      uint32_t dst_addr =
          dst_base_addr +
          dst_layout(dst_chunked_coord2coord(chunk_idx)) * sizeof(T);
      asm volatile(
          "cp.async.cg.shared.global.L2::128B [%0], [%1], 16;" ::"r"(dst_addr),
          "l"(src_addr));
    }
  }
};

template <typename T, class DstLayout, class SrcLayout, class TMA, class MainloopPipeline, class PipelineState, bool MInput>
class InputTMAAsyncCopy {
public:
  CUTE_STATIC_ASSERT_V(rank(SrcLayout{}) == rank(DstLayout{}));
  
  using CTA_TILER = decltype(shape(DstLayout{}));
  //N major/K major

  // static constexpr GMMA::Major GmmaMajor = MInput ? (stride<0>(DstLayout{}) == _1{} ? GMMA::Major::K : GMMA::Major::MN) : (stride<0>(DstLayout{}) == _1{} ? GMMA::Major::MN : GMMA::Major::K);
  // static constexpr GMMA::Major GmmaMajor = MInput ? (DstLayout{}.stride()[0] == _1{} ? GMMA::Major::K : GMMA::Major::MN) : (DstLayout{}.stride()[0] ? GMMA::Major::MN : GMMA::Major::K);

  //if Minput is true, it's a KM input, swap it to MK major

  // static constexpr cute::GMMA::Major GmmaMajor = MInput ? GMMA::Major::K : GMMA::Major::MN;

  // using DstMNKLayout = std::conditional_t<MInput, typename InputDim01Swapper<DstLayout>::Result, DstLayout>;

  // using SmemLayoutAtom = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GmmaMajor, half_t, decltype(get<0>(DstMNKLayout{})), decltype(get<1>(DstMNKLayout{}))>());

  // using DstPipeLayout = decltype(tile_to_shape(
  //       SmemLayoutAtom{},
  //       make_shape(shape<0>(DstMNKLayout{}), shape<1>(DstMNKLayout{}), Int<kStages>{}), Step<_1, _2, _3>{}));



  static constexpr cute::GMMA::Major GmmaMajor = GMMA::Major::MN;
  using DstMNKLayout = DstLayout;
  using SmemLayoutAtom = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GmmaMajor, half_t, decltype(get<0>(DstMNKLayout{})), decltype(get<1>(DstMNKLayout{}))>());
  using DstPipeLayout = decltype(tile_to_shape(
        SmemLayoutAtom{},
        make_shape(shape<0>(DstMNKLayout{}), shape<1>(DstMNKLayout{}), Int<kStages>{}), Step<_1, _2, _3>{}));




  // using MN = std::conditional_t<MInput, decltype(get<1>(DstLayout{})), decltype(get<0>(DstLayout{}))>;
  // using K = std::conditional_t<MInput, decltype(get<0>(DstLayout{})), decltype(get<1>(DstLayout{}))>;
  //64, 16, 1, 64 -> 16, 64, 64, 1
  // using DstLayouts = MInput ?  : DstLayout;

  // static constexpr GMMA::Major GmmaMajor = MInput ? (DstLayout{}.stride()[0] == _1{} ? GMMA::Major::K : GMMA::Major::MN) : (DstLayout{}.stride()[0] ? GMMA::Major::MN : GMMA::Major::K);
  // using SmemLayoutAtom = decltype(cutlass::gemm::collective::detail::ss_smem_selector<GmmaMajor, half_t, MN, K>);
  // using SmemLayoutAtom = std::conditional_t<MInput, decltype(cutlass::gemm::collective::detail::ss_smem_selector<
  //     GmmaMajor, half_t, decltype(get<1>(DstLayout{})), decltype(get<0>(DstLayout{}))>()), decltype(cutlass::gemm::collective::detail::ss_smem_selector<
  //     GmmaMajor, half_t, decltype(get<0>(DstLayout{})), decltype(get<1>(DstLayout{}))>())>;


  // using DstPipeLayout = std::conditional_t<MInput, decltype(tile_to_shape(
  //       SmemLayoutAtom{},
  //       make_shape(shape<1>(DstLayout{}), shape<0>(DstLayout{}), Int<kStages>{}), Step<_1, _2, _3>{})), 
  //       decltype(tile_to_shape(
  //       SmemLayoutAtom{},
  //       make_shape(shape<0>(DstLayout{}), shape<1>(DstLayout{}), Int<kStages>{}), Step<_1, _2, _3>{}))>;

  // using DstPipeLayout = DstLayout;
  // 64, 16, M, K -> 16, 64, M -> K -> 8, 64, M -> K

  // using DstPipeLayout = decltype(tile_to_shape(
  //       SmemLayoutAtom{},
  //       make_shape(shape<0>(DstLayout{}), shape<1>(DstLayout{}), Int<kStages>{}), Step<_1, _2, _3>{}));
  // // using DstPipeLayout = MInput ? decltype(tile_to_shape(
  //       SmemLayoutAtom{},
  //       make_shape(shape<1>(DstLayout{}), shape<0>(DstLayout{}), Int<kStages>{}), Step<_1, _2, _3>{})) : decltype(tile_to_shape(
  //       SmemLayoutAtom{},
  //       make_shape(shape<0>(DstLayout{}), shape<1>(DstLayout{}), Int<kStages>{}), Step<_1, _2, _3>{}));
  using MainloopPipelines = MainloopPipeline;
  using PipelineStates = PipelineState;
  using SrcLayouts = SrcLayout;
  using DstLayouts = DstLayout;
  using TMAs = TMA; 
  static constexpr int tmaTransactionBytes = sizeof(T) * size(DstPipeLayout{}) / kStages;
  
  static __device__ __forceinline__ void prefetch(TMA const &tma){
    cute::prefetch_tma_descriptor(tma.get_tma_descriptor());
  }
};

template<class TMACopy1, class TMACopy2>
class TMACopyPipeline2{

public:
    using T = half_t;
    using TMA_A =  TMACopy1;
    using TMA_B =  TMACopy2;
    using CopyA = typename TMA_A::TMAs;
    using CopyB = typename TMA_B::TMAs;
    using MainloopPipeline = typename TMA_A::MainloopPipelines;
    using PipelineState = typename TMA_A::PipelineStates;
    using SrcLayout_A = typename TMA_A::SrcLayouts;
    using SrcLayout_B = typename TMA_B::SrcLayouts;
    using CTA_TILER_A = typename TMA_A::CTA_TILER;
    using CTA_TILER_B = typename TMA_B::CTA_TILER;
    using DstPipeLayout_A = typename TMACopy1::DstPipeLayout;
    using DstPipeLayout_B = typename TMACopy2::DstPipeLayout;
    static __device__ __forceinline__ void run(CopyA const &tma_a,
                                                CopyB const &tma_b,
                                             T *dst_a,
                                             T *dst_b,
                                             MainloopPipeline pipeline,
                                             PipelineState &smem_pipe_write,
                                             unsigned k_tile_count,
                                             int imapx_a,
                                             int imapy_a,
                                             int imapz_a,
                                             int imapx_b,
                                             int imapy_b,
                                             int imapz_b,
                                             int lane_predicate) {
    if(lane_predicate){

      Tensor mA = tma_a.get_tma_tensor(shape(SrcLayout_A{}));
      Tensor mB = tma_b.get_tma_tensor(shape(SrcLayout_B{}));
      
      Tensor gA_mkl = local_tile(mA, CTA_TILER_A{}, make_coord(_, _)); 
      Tensor gA = gA_mkl(_, _, _, (imapx_a > 0 ? blockIdx.x : 0));
      Tensor sA = make_tensor(make_smem_ptr(dst_a), DstPipeLayout_A{});


      auto cta_tma_a = tma_a.get_slice(Int<0>{});  // CTA slice
      Tensor tAgA = cta_tma_a.partition_S(gA); // (TMA,TMA_M,TMA_N,REST_M,REST_N)
      Tensor tAsA = cta_tma_a.partition_D(sA); // (TMA,TMA_M,TMA_N)

      Tensor gB_mkl = local_tile(mB, CTA_TILER_B{}, make_coord(_, _)); 
      Tensor gB = gB_mkl(_, _, blockIdx.x, _);
      Tensor sB = make_tensor(make_smem_ptr(dst_b), DstPipeLayout_B{});


      auto cta_tma_b = tma_b.get_slice(Int<0>{});  // CTA slice
      Tensor tBgB = cta_tma_b.partition_S(gB); // (TMA,TMA_M,TMA_N,REST_M,REST_N)
      Tensor tBsB = cta_tma_b.partition_D(sB); // (TMA,TMA_M,TMA_N)

      // auto k_tile_iter  = cute::make_coord_iterator(k_tile_count);
      // // CUTLASS_PRAGMA_NO_UNROLL
      // for (; k_tile_count > 0; --k_tile_count){
        pipeline.producer_acquire(smem_pipe_write);
        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType *tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
       
        int write_stage = smem_pipe_write.index();
        //  if(blockIdx.x == 0 && blockIdx.y == 0){
        //   printf("write2 stage %d\n", TMA_A::tmaTransactionBytes + TMA_B::tmaTransactionBytes);
        //   print("\n");
        //  }
        copy(tma_a.with(*tma_barrier), tAgA(_, _, _, k_tile_count), tAsA(_, _, _, write_stage));
        copy(tma_b.with(*tma_barrier), tBgB(_, _, _, k_tile_count), tBsB(_, _, _, write_stage));

        // pipeline.producer_commit(smem_pipe_write, (TMA_A::tmaTransactionBytes + TMA_B::tmaTransactionBytes));
        // Advance smem_pipe_write

        // ++k_tile_iter;
        ++smem_pipe_write;

      // }
      
      
    }
    
  }
};

// template<class TMACopy>
// class TMACopyPipeline1{

// public:
//     using T = half_t;
//     using TMA_A =  TMACopy;
//     using CopyA = typename TMA_A::TMAs;
//     using MainloopPipeline = typename TMA_A::MainloopPipelines;
//     using PipelineState = typename TMA_A::PipelineStates;
//     using SrcLayout_A = typename TMA_A::SrcLayouts;
//     using CTA_TILER_A = typename TMA_A::CTA_TILER;

//     using DstPipeLayout_A = typename TMACopy::DstPipeLayout;
//     static __device__ __forceinline__ void run(CopyA const &tma_a,
//                                              T *dst_a,
//                                              MainloopPipeline pipeline,
//                                              PipelineState &smem_pipe_write,
//                                              unsigned k_tile_count,
//                                              int imapx_a,
//                                              int imapy_a,
//                                              int imapz_a,
//                                              int lane_predicate) {
//     if(lane_predicate){
      
//       Tensor mA = tma_a.get_tma_tensor(shape(SrcLayout_A{}));
      
//       Tensor gA_mkl = local_tile(mA, CTA_TILER_A{}, make_coord(_, _)); 
//       Tensor gA = gA_mkl(_, _, _, (imapx_a > 0 ? blockIdx.x : 0));
//       Tensor sA = make_tensor(make_smem_ptr(dst_a), DstPipeLayout_A{});


//       auto cta_tma_a = tma_a.get_slice(Int<0>{});  // CTA slice
//       Tensor tAgA = cta_tma_a.partition_S(gA); // (TMA,TMA_M,TMA_N,REST_M,REST_N)
//       Tensor tAsA = cta_tma_a.partition_D(sA); // (TMA,TMA_M,TMA_N)

//         pipeline.producer_acquire(smem_pipe_write);
//         using BarrierType = typename MainloopPipeline::ProducerBarrierType;
//         BarrierType *tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
       
//         int write_stage = smem_pipe_write.index();
//         copy(tma_a.with(*tma_barrier), tAgA(_, _, _, k_tile_count), tAsA(_, _, _, write_stage));
//         ++smem_pipe_write;

//       // }
//     }
    
//   }
// };

template<class TMACopy>
class TMACopyPipeline1{

public:
    using T = half_t;
    using TMA_A =  TMACopy;
    using CopyA = typename TMA_A::TMAs;
    using MainloopPipeline = typename TMA_A::MainloopPipelines;
    using PipelineState = typename TMA_A::PipelineStates;
    using SrcLayout_A = typename TMA_A::SrcLayouts;
    using CTA_TILER_A = typename TMA_A::CTA_TILER;

    using DstPipeLayout_A = typename TMACopy::DstPipeLayout;
    static __device__ __forceinline__ void run(CopyA const &tma_a,
                                             T *dst_a,
                                             MainloopPipeline pipeline,
                                             PipelineState &smem_pipe_write,
                                             unsigned k_tile_count,
                                             int imapx_a,
                                             int imapy_a,
                                             int imapz_a,
                                             int lane_predicate) {
    if(lane_predicate){
      
      Tensor mA = tma_a.get_tma_tensor(shape(SrcLayout_A{}));
      Tensor gA_mkl = local_tile(mA, CTA_TILER_A{}, make_coord(_, _)); 

      if(shape<0>(SrcLayout_A{}) == _4096{}){
           Tensor gA = gA_mkl(_, _, _, 0, _, _);
           Tensor sA = make_tensor(make_smem_ptr(dst_a), DstPipeLayout_A{});

           auto cta_tma_a = tma_a.get_slice(Int<0>{});  // CTA slice
           Tensor tAgA = cta_tma_a.partition_S(gA); // (TMA,TMA_M,TMA_N,REST_M,REST_N)

           Tensor tAgAs = group_modes<0, 3>(tAgA);
           Tensor tAsA = cta_tma_a.partition_D(sA); // (TMA,TMA_M,TMA_N)

            pipeline.producer_acquire(smem_pipe_write);
            using BarrierType = typename MainloopPipeline::ProducerBarrierType;
            BarrierType *tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
            int write_stage = smem_pipe_write.index();

            // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0){
            //   print("tAgA: \n");
            //   print(tAgA);
            //   print("\n");
            //   print(tAsA);
            //   print("----------\n");
            // }
              
            // }
            // pipeline.producer_commit(smem_pipe_write, (TMA_A::tmaTransactionBytes));
            // (((_64,_64),_1),_1,_1,_1,_1,_2)
            // pipeline.producer_commit(smem_pipe_write, (TMA_A::tmaTransactionBytes));
          copy(tma_a.with(*tma_barrier), tAgAs(_, _, _, write_stage), tAsA(_, _, _, write_stage));

      }else{
        Tensor gA = gA_mkl(_, _, _, _, 0,_);
        Tensor sA = make_tensor(make_smem_ptr(dst_a), DstPipeLayout_A{});
        auto cta_tma_a = tma_a.get_slice(Int<0>{});  // CTA slice
            Tensor tAgA = cta_tma_a.partition_S(gA); // (TMA,TMA_M,TMA_N,REST_M,REST_N)
            Tensor tAsA = cta_tma_a.partition_D(sA); // (TMA,TMA_M,TMA_N)
            Tensor tAgAs = group_modes<0, 3>(tAgA);

              pipeline.producer_acquire(smem_pipe_write);
              using BarrierType = typename MainloopPipeline::ProducerBarrierType;
              BarrierType *tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
              int write_stage = smem_pipe_write.index();
              // pipeline.producer_commit(smem_pipe_write, (TMA_A::tmaTransactionBytes));

              copy(tma_a.with(*tma_barrier), tAgAs(_, _, _, write_stage), tAsA(_, _, _, write_stage));
      }
      
    
        ++smem_pipe_write;

      // }
    }
    
  }
};

} // namespace tb
