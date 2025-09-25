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
#include "cutlass/layout/layout.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"

namespace kernel {

using namespace cute;
using namespace cutlass;
using namespace cutlass::gemm::collective;

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
          class GmemLayoutATag,
          class GmemLayoutBTag,
          class GmemLayputCTag,
          class GmemLayoutDTag,
          int NUM_WARPS,
          int M,
          int N,
          int K,
          int O_STRIDE = OUTPUT_SIZE_,
          int NUM_STAGES_ = 3>
struct MMAKernelTraits {
  using DataType = DataType_;
  using DTypeAccum = float;

  // tile size
  using TileShape_MNK = Shape<Int<M>, Int<N>, Int<K>>;

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
      DTypeAccum,
      FragmentSize,
      DTypeAccum,
      DataType,
      cutlass::epilogue::thread::ScaleType::Default,
      cutlass::FloatRoundStyle::round_to_nearest,
      DTypeAccum>;

  // using 2,1,1 for cooperative scheduling
  using AtomLayoutMNK = Layout<Shape<_1, _1, _1>>;
  using ClusterShape_MNK = Shape<Int<1>, Int<1>, Int<1>>;

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

  using MainloopPipeline = cutlass::PipelineTmaAsync<NUM_STAGES>;
  using PipelineState = typename cutlass::PipelineState<NUM_STAGES>;

  using SharedStorage = SharedStorageMMA<MainloopPipeline,
                                         DataType,
                                         SmemLayoutAtomA,
                                         SmemLayoutAtomB,
                                         SmemLayoutAtomC>;
};

} // namespace kernel
