
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

#include "cutlass/arch/memory.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/numeric/numeric_types.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cuda_host_adapter.hpp"

namespace kernel {
template <typename Ktraits>
struct CollectiveEpilogue {

  using DataTypeC = typename Ktraits::DTypeAccum;
  using StrideC = typename Ktraits::StrideC;
  using StrideD = typename Ktraits::StrideD;
  using ThreadEpilogueOp = typename Ktraits::ThreadOp;

  using PipelineStorage = cutlass::PipelineTransactionAsync<0>;

  struct Params {
    DataTypeC const *ptr_C = nullptr;
    StrideC dC{};
    DataTypeC *ptr_D = nullptr;
    StrideD dD{};
  };

  CUTLASS_HOST_DEVICE
  CollectiveEpilogue(Params const &params_)
      : params(params_), epilogue_op(params_.thread) {}

  template <class ProblemShapeMNKL,
            class BlockShapeMNK,
            class BlockCoordMNKL,
            class FrgEngine,
            class FrgLayout,
            class TiledMma,
            class ResidueMNK>
  CUTLASS_DEVICE void
      store(ProblemShapeMNKL problem_shape_mnkl,
            BlockShapeMNK blk_shape_MNK,
            BlockCoordMNKL blk_coord_mnkl,
            cute::Tensor<FrgEngine, FrgLayout> const &accumulators,
            TiledMma tiled_mma,
            [[maybe_unused]] ResidueMNK,
            int thread_idx,
            [[maybe_unused]] char *) {
    using namespace cute;
    using X = Underscore;

    static_assert(cute::rank(ProblemShapeMNKL{}) == 4,
                  "ProblemShapeMNKL must be rank 4");
    static_assert(is_static<BlockShapeMNK>::value,
                  "ThreadBlock tile shape must be static");
    static_assert(cute::rank(BlockShapeMNK{}) == 3,
                  "BlockShapeMNK must be rank 3");
    static_assert(cute::rank(BlockCoordMNKL{}) == 4,
                  "BlockCoordMNKL must be rank 3");

    // Separate out problem shape for convenience
    auto M = get<0>(problem_shape_mnkl);
    auto N = get<1>(problem_shape_mnkl);
    auto L = get<3>(problem_shape_mnkl);

    // no transpose for epologue
    auto stride_c = params.dC;
    auto stride_d = params.dD;

    // Represent the full output tensor
    Tensor mC_mnl = make_tensor(make_gmem_ptr<DataTypeC>(params.ptr_C),
                                make_shape(M, N, L),
                                stride_c); // (m,n,l)
    Tensor mD_mnl = make_tensor(
        make_gmem_ptr(params.ptr_D), make_shape(M, N, L), stride_d); // (m,n,l)
    Tensor gC_mnl = local_tile(mC_mnl,
                               blk_shape_MNK,
                               make_coord(_, _, _),
                               Step<_1, _1, X>{}); // (BLK_M,BLK_N,m,n,l)
    Tensor gD_mnl = local_tile(mD_mnl,
                               blk_shape_MNK,
                               make_coord(_, _, _),
                               Step<_1, _1, X>{}); // (BLK_M,BLK_N,m,n,l)

    // Slice to get the tile this CTA is responsible for
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord_mnkl;
    Tensor gC = gC_mnl(_, _, m_coord, n_coord, l_coord); // (BLK_M,BLK_N)
    Tensor gD = gD_mnl(_, _, m_coord, n_coord, l_coord); // (BLK_M,BLK_N)

    // Partition source and destination tiles to match the accumulator
    // partitioning
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCgD = thr_mma.partition_C(gD); // (VEC,THR_M,THR_N)
    Tensor tCgC = thr_mma.partition_C(gC); // (VEC,THR_M,THR_N)

    static_assert(is_static<FrgLayout>::value,
                  "Accumulator layout must be static");
    CUTE_STATIC_ASSERT_V(
        size(tCgC) == size(tCgD),
        "Source and destination must have the same number of elements.");
    CUTE_STATIC_ASSERT_V(
        size(tCgD) == size(accumulators),
        "Accumulator count must have the same destination element count.");

    // OOB predication for tile quantization "residue"
    // Absolute coordinate tensors (dynamic)
    auto shape_MN = make_shape(M, N);
    Tensor mD_crd = make_identity_tensor(shape_MN); // (M,N)
    Tensor cD_mn = local_tile(mD_crd,
                              take<0, 2>(blk_shape_MNK),
                              make_coord(m_coord, n_coord)); // (BLK_M,BLK_N)
    Tensor tCcD_mn = thr_mma.partition_C(cD_mn); // (VEC,THR_M,THR_N)
    // Relative coordinate tensors (static)
    Tensor cD = make_coord_tensor(cD_mn.layout());     // (BLK_M,BLK_N)
    Tensor tCcD = make_coord_tensor(tCcD_mn.layout()); // (VEC,THR_M,THR_N)
    // Subtract the global "bottom right" corner from the local "top left"
    // corner to get the max relative coordinate
    auto residue_cD = shape_MN - cD_mn(_0{});     // (m,n)
    auto residue_tCcD = shape_MN - tCcD_mn(_0{}); // (m,n)

    // Fully OOB tile
    if (not elem_less(repeat_like(residue_cD, _0{}), residue_cD)) {
      return;
    }

    using FragCType = remove_cvref_t<decltype(tCgC(0))>;
    using FragDType = remove_cvref_t<decltype(tCgD(0))>;

    // source is needed
    if (epilogue_op.is_source_needed()) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        FragCType fragC;
        bool pred = elem_less(tCcD(i), residue_tCcD);
        cutlass::arch::global_load<FragCType, sizeof(FragCType)>(
            fragC, &tCgC(i), pred);
        FragDType fragD = epilogue_op(accumulators(i), fragC);
        cutlass::arch::global_store<FragDType, sizeof(FragDType)>(
            fragD, &tCgD(i), pred);
      }
    }
    // source is not needed, avoid load
    else {
      // printf("xxxxxxxxxxxxxx\n");
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(accumulators); ++i) {
        bool pred = elem_less(tCcD(i), residue_tCcD);
        FragDType fragD = epilogue_op(accumulators(i));
        cutlass::arch::global_store<FragDType, sizeof(FragDType)>(
            fragD, &tCgD(i), pred);
      }
    }
  }

private:
  Params params;
  ThreadEpilogueOp epilogue_op;
};
} // namespace kernel
