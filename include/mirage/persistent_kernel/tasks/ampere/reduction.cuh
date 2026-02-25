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
#include "element_unary.cuh"
#include "tasks/common/common_header.cuh"

namespace kernel {

using bfloat16 = type::bfloat16_t;

// reduction on dim 0
template <typename SMEM_DST, typename SMEM_SRC>
static __device__ __forceinline__ void reduction_sum_row(SMEM_DST dst,
                                                         SMEM_SRC src) {
  static constexpr int NUM_WARPS_K = SMEM_SRC::ROW / SMEM_DST::ROW;
  static_assert(SMEM_SRC::ROW % SMEM_DST::ROW == 0,
                "Incompatible reduction dimensions");

  for (int dst_elem_idx = threadIdx.x; dst_elem_idx < SMEM_DST::size();
       dst_elem_idx += TASK_BLOCK_DIM) {
    float result = 0;
    int dst_row = dst_elem_idx / SMEM_DST::COL;
    int dst_col = dst_elem_idx % SMEM_DST::COL;

#pragma unroll
    for (int k = 0; k < NUM_WARPS_K; ++k) {
      // The source rows are laid out as [warp_k_0_b_0, warp_k_0_b_1, ...,
      // warp_k_1_b_0, ...]
      int src_row = k * SMEM_DST::ROW + dst_row;
      result += float(src.at(src_row, dst_col));
    }
    dst.at(dst_elem_idx) = bfloat16(result);
  }
}

// reduction on dim 0. Add the result to the existing value.
template <typename SMEM_DST, typename SMEM_SRC>
static __device__ __forceinline__ void reduction_sum_row_add(SMEM_DST dst,
                                                             SMEM_SRC src) {
  static constexpr int NUM_WARPS_K = SMEM_SRC::ROW / SMEM_DST::ROW;
  static_assert(SMEM_SRC::ROW % SMEM_DST::ROW == 0,
                "Incompatible reduction dimensions");

  for (int dst_elem_idx = threadIdx.x; dst_elem_idx < SMEM_DST::size();
       dst_elem_idx += TASK_BLOCK_DIM) {
    float result = float(dst.at(dst_elem_idx));
    int dst_row = dst_elem_idx / SMEM_DST::COL;
    int dst_col = dst_elem_idx % SMEM_DST::COL;

#pragma unroll
    for (int k = 0; k < NUM_WARPS_K; ++k) {
      // The source rows are laid out as [warp_k_0_b_0, warp_k_0_b_1, ...,
      // warp_k_1_b_0, ...]
      int src_row = k * SMEM_DST::ROW + dst_row;
      result += float(src.at(src_row, dst_col));
    }
    dst.at(dst_elem_idx) = bfloat16(result);
  }
}

// epilogue for reduction on dim 0
template <typename SMEM_DST,
          typename SMEM_SRC,
          ElementUnaryOpType FirstOp,
          ElementUnaryOpType... RemainingOps>
static __device__ __forceinline__ void
    reduction_sum_row(SMEM_DST dst, SMEM_SRC src, float const *scalars) {
  static constexpr int REDUCTION_FACTOR = SMEM_SRC::ROW;
  for (int dst_elem_idx = threadIdx.x; dst_elem_idx < SMEM_DST::size();
       dst_elem_idx += TASK_BLOCK_DIM) {
    float result = 0;
    int dst_col = dst_elem_idx % SMEM_DST::COL;

#pragma unroll
    for (int i = 0; i < REDUCTION_FACTOR; ++i) {
      result += float(src.at(i, dst_col));
    }
    result = perform_element_unary_chain<float, FirstOp, RemainingOps...>(
        result, scalars, 0);
    dst.at(dst_elem_idx) = bfloat16(result);
  }
}

// reduction on dim 1
template <typename T, typename SMEM_DST, typename SMEM_SRC>
static __device__ __forceinline__ void reduction_sum_col(SMEM_DST dst,
                                                         SMEM_SRC src) {
  static constexpr int REDUCTION_FACTOR = SMEM_SRC::COL;
  for (int dst_elem_idx = threadIdx.x; dst_elem_idx < SMEM_DST::size();
       dst_elem_idx += TASK_BLOCK_DIM) {
    // TODO xinhaoc make this result float32
    float result = 0.0f;
    int dst_row = dst_elem_idx / SMEM_DST::COL;
#pragma unroll
    for (int i = 0; i < REDUCTION_FACTOR; ++i) {
      result += float(src.at(dst_row, i));
    }
    dst.at(dst_elem_idx) = bfloat16(result);
  }
}

// epilogue for reduction on dim 1
template <typename T,
          typename SMEM_DST,
          typename SMEM_SRC,
          ElementUnaryOpType FirstOp,
          ElementUnaryOpType... RemainingOps>
static __device__ __forceinline__ void
    reduction_sum_col(SMEM_DST dst, SMEM_SRC src, float const *scalars) {
  static constexpr int REDUCTION_FACTOR = SMEM_SRC::COL;
  for (int dst_elem_idx = threadIdx.x; dst_elem_idx < SMEM_DST::size();
       dst_elem_idx += TASK_BLOCK_DIM) {
    // TODO xinhaoc make this result float32
    float result = 0.0f;
    int dst_row = dst_elem_idx / SMEM_DST::COL;
#pragma unroll
    for (int i = 0; i < REDUCTION_FACTOR; ++i) {
      result += float(src.at(dst_row, i));
    }
    result = perform_element_unary_chain<float, FirstOp, RemainingOps...>(
        result, scalars, 0);
    dst.at(dst_elem_idx) = bfloat16(result);
  }
}

} // namespace kernel
