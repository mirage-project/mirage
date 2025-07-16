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
#include "common.h"
#include "element_unary.cuh"

namespace kernel {

using bfloat16 = type::bfloat16_t;

// reduction on dim 0
template <typename SMEM_DST, typename SMEM_SRC>
static __device__ __forceinline__ void reduction_sum_row(SMEM_DST dst,
                                                         SMEM_SRC src) {

  static constexpr int REDUCTION_FACTOR = SMEM_SRC::ROW;
  for (int dst_elem_idx = threadIdx.x; dst_elem_idx < SMEM_DST::size();
       dst_elem_idx += blockDim.x) {
    float result = 0;
    int dst_col = dst_elem_idx % SMEM_DST::COL;

#pragma unroll
    for (int i = 0; i < REDUCTION_FACTOR; ++i) {
      result += float(src.at(i, dst_col));
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
       dst_elem_idx += blockDim.x) {
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
       dst_elem_idx += blockDim.x) {
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
       dst_elem_idx += blockDim.x) {
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

// Assume that input/buffer/output have the same stride
template <typename T>
__device__ __forceinline__ void reduction_kernel(void const *input_ptr,
                                                 void const *buf_ptr,
                                                 void *output_ptr,
                                                 int num_gpus,
                                                 int my_gpu_id,
                                                 int batch_size,
                                                 int output_size,
                                                 int stride) {
  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_buffer = static_cast<T const *>(buf_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);
  for (int idx = threadIdx.x; idx < output_size * batch_size;
       idx += blockDim.x) {
    T accum = static_cast<T>(0.0f);
    int batch = idx / output_size;
    int offset = idx % output_size;
    for (int i = 0; i < num_gpus; i++) {
      if (i == my_gpu_id) {
        accum += d_input[batch * stride + offset];
      } else {
        accum += d_buffer[i * batch_size * stride + batch * stride + offset];
      }
    }
    d_output[batch * stride + offset] = accum;
  }
}

} // namespace kernel
