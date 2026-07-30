/* Copyright 2023-2024 CMU
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

#include "cutlass/gemm/warp/default_mma_tensor_op.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_smem_accumulator.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"

namespace mirage {
namespace warp {

using namespace cutlass;

template <typename WarpShape,
          typename InstructionShape,
          typename ElementType,
          typename SmemLayoutA,
          typename SmemLayoutB>
class GemmExecutor {
public:
  using MmaTensorOp = typename gemm::warp::DefaultMmaTensorOp<
      WarpShape,
      InstructionShape,
      ElementType,              // Data type of A elements
      SmemLayoutA,              // Layout of A matrix
      ElementType,              // Data type of B elements
      SmemLayoutB,              // Layout of B matrix
      ElementType,              // Data type of C elements
      cutlass::layout::RowMajor // Layout of C matrix
      >::Type;

  using WarpFragmentA = typename MmaTensorOp::FragmentA;
  using WarpFragmentB = typename MmaTensorOp::FragmentB;
  using WarpFragmentC = typename MmaTensorOp::FragmentC;

  // Define a 'FragmentIterator' to iterate over slices of accumulators
  using FragmentIteratorAccumulator =
      epilogue::warp::FragmentIteratorTensorOp<WarpShape,
                                               InstructionShape,
                                               ElementType,
                                               WarpFragmentC,
                                               cutlass::layout::RowMajor>;
  // Create an epilogue
  // Iterator to store to shared-memory
  using SmemAccumulatorLayout = cutlass::layout::RowMajor;
  using SmemIteratorD =
      typename epilogue::warp::TileIteratorTensorOp<WarpShape,
                                                    InstructionShape,
                                                    ElementType,
                                                    SmemAccumulatorLayout>;
  // We need to provide an operation for the epilogue. Let's create an
  // operation that does nothing (ScaleType::Nothing)
  using OutputOpNoOp = epilogue::thread::LinearCombination<
      typename SmemIteratorD::Element, // ElementOutput
      FragmentIteratorAccumulator::Fragment::kElements,
      ElementType,                     // ElementAccumulator
      typename SmemIteratorD::Element, // ElementCompute
      cutlass::epilogue::thread::ScaleType::Nothing>;
  using Epilogue = epilogue::threadblock::EpilogueSmemAccumulator<
      SmemIteratorD,
      FragmentIteratorAccumulator,
      SmemIteratorD,
      OutputOpNoOp>;

  // Define an epilogue 'Tile Iteterator' to iterate over slices of elements in
  // Shared Memory
  using AccumulatorTileIterator =
      epilogue::warp::TileIteratorTensorOpCanonical<WarpShape,
                                                    InstructionShape,
                                                    ElementType,
                                                    SmemAccumulatorLayout>;

  using TensorRefA = typename MmaTensorOp::IteratorA::TensorRef;
  using TensorRefB = typename MmaTensorOp::IteratorB::TensorRef;
  // using TensorRefC = typename AccumulatorTileIterator::TensorRef;
  using TensorRefC = typename SmemIteratorD::TensorRef;

  // Number of gemm iterations along the k dimension
  int const kWarpGemmIterations =
      (WarpShape::kK + InstructionShape::kK - 1) / InstructionShape::kK;

  /// cutlass fields
  typename MmaTensorOp::IteratorA warp_tile_iterator_A;
  typename MmaTensorOp::IteratorB warp_tile_iterator_B;
  SmemIteratorD smem_iterator_D;
  OutputOpNoOp output_op;

  CUTLASS_DEVICE
  GemmExecutor(TensorRefA const &ref_A,
               TensorRefB const &ref_B,
               TensorRefC const &ref_C,
               int m,
               int n,
               int k,
               int thread_idx,
               int warp_idx,
               int lane_idx)
      : output_op({}),
        warp_tile_iterator_A(ref_A, {WarpShape::kM, k}, lane_idx),
        warp_tile_iterator_B(ref_B, {k, WarpShape::kN}, lane_idx) {
    int warp_count_m = m / WarpShape::kM;
    int warp_count_n = n / WarpShape::kN;
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   warp_idx_m: the warp's position within the threadblock along the M
    //   dimension warp_idx_n: the warp's position within the threadblock along
    //   the N dimension warp_idx_k: the warp's position within the threadblock
    //   along the K dimension

    int warp_idx_mn = warp_idx % (warp_count_m * warp_count_n);
    int warp_idx_k = warp_idx / (warp_count_m * warp_count_n);
    printf("warp_idx(%d) warp_count_m(%d) warp_count_n(%d)\n",
           warp_idx,
           warp_count_m,
           warp_count_n);
    // Currently assume that we don't parittion over k within a threadblock
    // we do it across threadblocks
    assert(warp_idx_k == 0);

    int warp_idx_m = warp_idx_mn % warp_count_m;
    int warp_idx_n = warp_idx_mn / warp_count_m;

    int tile_offset_k = kWarpGemmIterations * warp_idx_k;

    // Add per-warp offsets in units of warp-level tiles
    warp_tile_iterator_A.add_tile_offset({warp_idx_m, tile_offset_k});
    warp_tile_iterator_B.add_tile_offset({tile_offset_k, warp_idx_n});
  }

  void CUTLASS_DEVICE execute_kernel(void) {
    // extern __shared__ char smem_buffer[];

    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];
    WarpFragmentC accum;
    accum.clear();
    Epilogue epilogue;

    MmaTensorOp warp_mma;

    warp_tile_iterator_A.set_kgroup_index(0);
    warp_tile_iterator_B.set_kgroup_index(0);
    warp_tile_iterator_A.load(warp_frag_A[0]);
    warp_tile_iterator_B.load(warp_frag_B[0]);
    ++warp_tile_iterator_A;
    ++warp_tile_iterator_B;

    CUTLASS_GEMM_LOOP
    for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; warp_mma_k++) {
      // Load warp-level tiles from shared memory, wrapping to k offset if
      // this is the last group as the case may be.
      if (warp_mma_k == kWarpGemmIterations - 1) {
        // TODO: Write fragments to shared memory
        // reference:
        // cutlass/examples/13/threadblock/b2b_mma_pipelined_smem_accumulators.h:376
      }
      warp_tile_iterator_A.set_kgroup_index((warp_mma_k + 1) %
                                            kWarpGemmIterations);
      warp_tile_iterator_B.set_kgroup_index((warp_mma_k + 1) %
                                            kWarpGemmIterations);
      warp_tile_iterator_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
      warp_tile_iterator_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);
      ++warp_tile_iterator_A;
      ++warp_tile_iterator_B;
      if (warp_mma_k == 0) {
        // TODO: Write fragments to shared memory
        // reference:
        // cutlass/examples/13/threadblock/b2b_mma_pipelined_smem_accumulators.h:376
      }
      warp_mma(accum,
               warp_frag_A[warp_mma_k % 2],
               warp_frag_B[warp_mma_k % 2],
               accum);
    }
    // TODO: enable epilogue
    // epilogue(OutputOpNoOp({}), smem_iterator_D, accum);
    __syncthreads();
  }
};

} // namespace warp
} // namespace mirage
