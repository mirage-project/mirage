/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <type_traits>

#include "cute/algorithm/copy.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_params.h"
#include "cutlass/epilogue/thread/scale_type.h"
#include "cutlass/functional.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_detail.hpp"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

namespace kernel {

template <typename MainloopPipeline,
          class DataType,
          class SmemLayoutA,
          class SmemLayoutB,
          class SmemLayoutC>
struct SharedStorageMMA {

  struct TensorStorage : cute::aligned_struct<128, _0> {
    cute::array_aligned<DataType, cute::cosize_v<SmemLayoutA>> smem_a;
    cute::array_aligned<DataType, cute::cosize_v<SmemLayoutB>> smem_b;
  } tensors;

  struct {
    typename MainloopPipeline::SharedStorage pipeline;
  };
};

template <typename DataType_,
          int BATCH_SIZE_,
          int OUTPUT_SIZE_,
          int REDUCTION_SIZE_,
          class GmemLayoutATag_,
          class GmemLayoutBTag_,
          class GmemLayputCTag_,
          class GmemLayoutDTag_,
          int NUM_WARPS,
          int M_,
          int N_,
          int K_,
          typename ProblemShape_,
          int O_STRIDE = OUTPUT_SIZE_,
          int NUM_STAGES_ = 3,
          typename IS_SWAPAB_ = cute::true_type>
struct MMAKernelTraits {
  using DataType = DataType_;
  using DTypeAccum = float;

  // tile size
  using TileShape_MNK = Shape<Int<M_>, Int<N_>, Int<K_>>;
  using GmemLayoutATag = GmemLayoutATag_;
  using GmemLayoutBTag = GmemLayoutBTag_;
  using GmemLayoutCTag = GmemLayputCTag_;
  using GmemLayoutDTag = GmemLayoutDTag_;
  using ProblemShape = ProblemShape_;

  using StrideA = cutlass::detail::TagToStrideA_t<GmemLayoutATag>;
  using StrideB = cutlass::detail::TagToStrideB_t<GmemLayoutBTag>;
  using StrideC = cutlass::detail::TagToStrideC_t<GmemLayoutCTag>;
  using StrideD = cutlass::detail::TagToStrideC_t<GmemLayoutDTag>;

  using IS_SWAPAB = IS_SWAPAB_;

  static constexpr int M = M_;
  static constexpr int N = N_;
  static constexpr int K = K_;

  static constexpr int NUM_THREADS = NUM_WARPS * cutlass::NumThreadsPerWarp;
  static constexpr int NUM_PRODUCER_THREADS = cutlass::NumThreadsPerWarp;
  static constexpr int BATCH_SIZE = BATCH_SIZE_;
  static constexpr int OUTPUT_SIZE = OUTPUT_SIZE_;
  static constexpr int REDUCTION_SIZE = REDUCTION_SIZE_;

  static constexpr int NUM_STAGES = NUM_STAGES_;
  static constexpr int K_PIPE_MMAS = 1;

  static constexpr int FragmentSize = 1;
  // epilogue
  using ThreadOp = cutlass::epilogue::thread::LinearCombination<
      DataType,
      FragmentSize,
      DTypeAccum,
      DTypeAccum,
      cutlass::epilogue::thread::ScaleType::Default,
      cutlass::FloatRoundStyle::round_to_nearest,
      DataType>;

  // using 2,1,1 for cooperative scheduling
  //   using AtomLayoutMNK = Layout<Shape<_1, _1, _1>>;
  using ClusterShape_MNK = Shape<Int<1>, Int<1>, Int<1>>;

  static constexpr bool IsCooperative = size<0>(TileShape_MNK{}) != Int<64>{};

  using AtomLayoutMNK = cute::conditional_t<IsCooperative,
                                            Layout<Shape<_2, _1, _1>>,
                                            Layout<Shape<_1, _1, _1>>>;

  using TileScheduler =
      cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90;
  //   enable_if_t<IsCooperative,
  //   cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90>;

  using KernelHardwareInfo = cutlass::KernelHardwareInfo;

  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;

  static constexpr cute::GMMA::Major GmmaMajorA = cutlass::gemm::collective::
      detail::gmma_ss_tag_to_major_A<DataType, GmemLayoutATag>();
  static constexpr cute::GMMA::Major GmmaMajorB = cutlass::gemm::collective::
      detail::gmma_ss_tag_to_major_B<DataType, GmemLayoutBTag>();

  using TiledMma =
      decltype(cute::make_tiled_mma(cute::GMMA::ss_op_selector<DataType,
                                                               DataType,
                                                               DTypeAccum,
                                                               TileShape_MNK,
                                                               GmmaMajorA,
                                                               GmmaMajorB>(),
                                    AtomLayoutMNK{}));

  using GmemTiledCopyA = decltype(cutlass::gemm::collective::detail::
                                      sm90_cluster_shape_to_tma_atom(
                                          shape<1>(ClusterShape_MNK{})));
  using GmemTiledCopyB = decltype(cutlass::gemm::collective::detail::
                                      sm90_cluster_shape_to_tma_atom(
                                          shape<0>(ClusterShape_MNK{})));

  using SmemLayoutAtomA =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GmmaMajorA,
               DataType,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomB =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GmmaMajorB,
               DataType,
               decltype(cute::get<1>(TileShape_MNK{})),
               decltype(cute::get<2>(TileShape_MNK{}))>());
  using SmemLayoutAtomC =
      decltype(cutlass::gemm::collective::detail::ss_smem_selector<
               GMMA::Major::K,
               DTypeAccum,
               decltype(cute::get<0>(TileShape_MNK{})),
               decltype(cute::get<1>(TileShape_MNK{}))>());

  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape_MNK{}),
                 shape<2>(TileShape_MNK{}),
                 Int<NUM_STAGES>{}),
      cute::conditional_t<::cutlass::gemm::detail::is_major<0, StrideA>(),
                          Step<_2, _1, _3>,
                          Step<_1, _2, _3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape_MNK{}),
                 shape<2>(TileShape_MNK{}),
                 Int<NUM_STAGES>{}),
      cute::conditional_t<::cutlass::gemm::detail::is_major<0, StrideB>(),
                          Step<_2, _1, _3>,
                          Step<_1, _2, _3>>{}));

  using MainloopPipeline = cutlass::PipelineTmaAsync<NUM_STAGES>;
  using PipelineState = typename cutlass::PipelineState<NUM_STAGES>;

  using SharedStorage = SharedStorageMMA<MainloopPipeline,
                                         DataType,
                                         SmemLayoutAtomA,
                                         SmemLayoutAtomB,
                                         SmemLayoutAtomC>;

  static constexpr bool SwapAB = IS_SWAPAB::value;
};

} // namespace kernel
