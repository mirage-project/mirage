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
#include "tasks/common/common_header.cuh"

namespace kernel {

template <int HIDDEN_SIZE>
__device__ __forceinline__ void
sinkhorn_normalize_rows(float *comb, float *row_sum, int valid_tokens, float eps) {
  int const row_count = valid_tokens * HIDDEN_SIZE;
  for (int row_id = threadIdx.x; row_id < row_count; row_id += blockDim.x) {
    int const token = row_id / HIDDEN_SIZE;
    int const row = row_id % HIDDEN_SIZE;
    int const base = token * HIDDEN_SIZE * HIDDEN_SIZE + row * HIDDEN_SIZE;

    float sum = 0.0f;
#pragma unroll
    for (int col = 0; col < HIDDEN_SIZE; ++col) {
      sum += comb[base + col];
    }
    row_sum[row_id] = sum;

    float const denom = sum + eps;
#pragma unroll
    for (int col = 0; col < HIDDEN_SIZE; ++col) {
      comb[base + col] /= denom;
    }
  }
}

template <int HIDDEN_SIZE>
__device__ __forceinline__ void
sinkhorn_normalize_cols(float *comb, float *col_sum, int valid_tokens, float eps) {
  int const col_count = valid_tokens * HIDDEN_SIZE;
  for (int col_id = threadIdx.x; col_id < col_count; col_id += blockDim.x) {
    int const token = col_id / HIDDEN_SIZE;
    int const col = col_id % HIDDEN_SIZE;
    int const base = token * HIDDEN_SIZE * HIDDEN_SIZE + col;

    float sum = 0.0f;
#pragma unroll
    for (int row = 0; row < HIDDEN_SIZE; ++row) {
      sum += comb[base + row * HIDDEN_SIZE];
    }
    col_sum[col_id] = sum;

    float const denom = sum + eps;
#pragma unroll
    for (int row = 0; row < HIDDEN_SIZE; ++row) {
      comb[base + row * HIDDEN_SIZE] /= denom;
    }
  }
}

template <int TOKEN_BLOCK_SIZE,
          int HIDDEN_SIZE,
          int INPUT_TOKEN_STRIDE,
          int OUTPUT_TOKEN_STRIDE>
__device__ __forceinline__ void sinkhorn_task_impl(
    void const *__restrict__ comb_res_mix_ptr,
    void *__restrict__ comb_res_mix_out_ptr,
    int valid_tokens,
    int repeat,
    float eps) {
  float const *__restrict__ comb_res_mix =
      static_cast<float const *>(comb_res_mix_ptr);
  float *__restrict__ comb_res_mix_out =
      static_cast<float *>(comb_res_mix_out_ptr);

  extern __shared__ char smem[];
  float *comb = reinterpret_cast<float *>(smem);
  float *row_sum = comb + TOKEN_BLOCK_SIZE * HIDDEN_SIZE * HIDDEN_SIZE;
  float *col_sum = row_sum + TOKEN_BLOCK_SIZE * HIDDEN_SIZE;

  int const row_count = valid_tokens * HIDDEN_SIZE;
  int const elem_count = valid_tokens * HIDDEN_SIZE * HIDDEN_SIZE;

  for (int row_id = threadIdx.x; row_id < row_count; row_id += blockDim.x) {
    int const token = row_id / HIDDEN_SIZE;
    int const row = row_id % HIDDEN_SIZE;
    int const smem_base = token * HIDDEN_SIZE * HIDDEN_SIZE + row * HIDDEN_SIZE;
    int const gmem_base = token * INPUT_TOKEN_STRIDE + row * HIDDEN_SIZE;

    float row_max = -1.0e30f;
#pragma unroll
    for (int col = 0; col < HIDDEN_SIZE; ++col) {
      row_max = fmaxf(row_max, comb_res_mix[gmem_base + col]);
    }

    float sum = 0.0f;
#pragma unroll
    for (int col = 0; col < HIDDEN_SIZE; ++col) {
      float const val = __expf(comb_res_mix[gmem_base + col] - row_max);
      comb[smem_base + col] = val;
      sum += val;
    }
    row_sum[row_id] = sum;

#pragma unroll
    for (int col = 0; col < HIDDEN_SIZE; ++col) {
      comb[smem_base + col] = comb[smem_base + col] / sum + eps;
    }
  }
  __syncthreads();

  int const steps = repeat > 0 ? repeat : 1;
  sinkhorn_normalize_cols<HIDDEN_SIZE>(comb, col_sum, valid_tokens, eps);
  __syncthreads();

  for (int iter = 1; iter < steps; ++iter) {
    sinkhorn_normalize_rows<HIDDEN_SIZE>(comb, row_sum, valid_tokens, eps);
    __syncthreads();
    sinkhorn_normalize_cols<HIDDEN_SIZE>(comb, col_sum, valid_tokens, eps);
    __syncthreads();
  }

  for (int idx = threadIdx.x; idx < elem_count; idx += blockDim.x) {
    int const token = idx / (HIDDEN_SIZE * HIDDEN_SIZE);
    int const offset = idx % (HIDDEN_SIZE * HIDDEN_SIZE);
    comb_res_mix_out[token * OUTPUT_TOKEN_STRIDE + offset] = comb[idx];
  }
}

} // namespace kernel

 
