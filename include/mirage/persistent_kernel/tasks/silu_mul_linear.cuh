
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
#include "copy_sm80.cuh"
#include "dmem_layout.cuh"
#include "element_binary.cuh"
#include "element_unary.cuh"
#include "mma.cuh"
#include "reduction.cuh"
#include "smem_layout.cuh"
#include "utils.cuh"
namespace kernel {

template <typename T>
__device__ __forceinline__ void silu_mul_linear_kernel(void const *input_ptr,
                                                       void const *mul_ptr,
                                                       void const *weight_ptr,
                                                       void *output_ptr) {

  constexpr int chunk_size = 16 / sizeof(T);
  constexpr int BATCH_SIZE = 1;
  constexpr int HIDDEN_SIZE = 64;

  constexpr int num_chunks = HIDDEN_SIZE / chunk_size;

  constexpr int num_n = HIDDEN_SIZE >> 4;
  constexpr int num_m = 1;
  constexpr int num_k = HIDDEN_SIZE >> 4;
  int warp_idx = warp_id();
  int idx_in_warp = threadIdx.x % 32;

  assert(num_m > 0 && num_n > 0 && num_k > 0);
  __restrict__ T const *d_input = static_cast<T const *>(input_ptr);
  __restrict__ T const *d_mul = static_cast<T const *>(mul_ptr);

  __restrict__ T const *d_weight = static_cast<T const *>(weight_ptr);
  T __restrict__ *d_output = static_cast<T *>(output_ptr);

  dmem_row_const<T, 1, 64, 3584> input_dmem(d_input);
  dmem_row_const<T, 1, 64, 3584> mul_dmem(d_mul);
  dmem_row_const<T, 64, 64, 64> weight_dmem(d_weight);
  dmem_row<T, 1, 64, 64> output_dmem(d_output);

  extern __shared__ T smem[];

  // copy input
  T *shared_input = (T *)(smem + 128);
  T *shared_input_buffer = (T *)(smem + 8448);

  T *shared_mul = (T *)(smem + 8576);
  T *shared_mul_buffer = (T *)(smem + 16896);

  // copy weight
  T *shared_weight = (T *)(smem + 8704);
  T *shared_weight_buffer = (T *)(smem + 256);

  // // intermidiate
  T *mul_output = (T *)(smem + 17152);
  T *silu_output = (T *)(smem + 17024);

  // out
  T *shared_output = (T *)(smem + 128);
  T *zero_buf = (T *)(smem);

  // define the swizzle mode
  // zero buffer
  smem_row<T, 1, 1, 1, 1, 8, 8> zero_buffer(zero_buf);

  smem_row<T, 1, 1, 1, 1, 64, 64> input_smem(shared_input);
  smem_row<T, 1, 1, 1, 1, 64, 64> input_smem_buffer(shared_input_buffer);

  smem_row<T, 1, 1, 1, 1, 64, 64> mul_smem(shared_mul);
  smem_row<T, 1, 1, 1, 1, 64, 64> mul_smem_buffer(shared_mul_buffer);

  smem_row<T, 3, 3, 3, 64, 64, 64> input_weight_smem(shared_weight);
  smem_row<T, 3, 3, 3, 64, 64, 64> input_weight_smem_buffer(
      shared_weight_buffer);

  smem_row<T, 1, 1, 1, 1, 64, 64> silu_smem(silu_output);

  smem_row<T, 1, 1, 1, 1, 64, 64> mul_output_smem(mul_output);

  smem_row<T, 1, 1, 1, 1, 64, 64> output_smem(shared_output);

// load input
#pragma unroll
  for (int i = threadIdx.x; i < (BATCH_SIZE * num_chunks); i += NUM_THREADS) {
    // offset
    int row = i / num_chunks;
    int col = (i % num_chunks) * chunk_size;
    load_smem(input_smem_buffer(row, col), input_dmem(row, col));
  }

  // load mul
#pragma unroll
  for (int i = threadIdx.x; i < (BATCH_SIZE * num_chunks); i += NUM_THREADS) {
    // offset
    int row = i / num_chunks;
    int col = (i % num_chunks) * chunk_size;
    load_smem(mul_smem_buffer(row, col), mul_dmem(row, col));
  }

// load weight
#pragma unroll
  for (int i = threadIdx.x; i < (HIDDEN_SIZE * HIDDEN_SIZE / chunk_size);
       i += NUM_THREADS) {
    int row = i / (HIDDEN_SIZE / chunk_size);
    int col = (i % num_chunks) * chunk_size;
    load_smem(input_weight_smem_buffer(row, col), weight_dmem(row, col));
  }
  cp_async_fence();

  //  accumulator
  float s_frag[num_m][num_n][8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    s_frag[0][0][i] = 0.0f;
    zero_buffer.at(0, i) = float2bfloat16(0.0f);
  }

  for (int for_idx = 0; for_idx < 56; for_idx++) {
    // copy
    if (for_idx + 1 != 56) {
      dmem_row_const<T, 1, 64, 3584> input_dmem_buffer(d_input +
                                                       64 * (for_idx + 1));
      dmem_row_const<T, 1, 64, 3584> mul_dmem_buffer(d_mul +
                                                     64 * (for_idx + 1));
      dmem_row_const<T, 64, 64, 64> weight_dmem_buffer(d_weight +
                                                       4096 * (for_idx + 1));

#pragma unroll
      for (int i = threadIdx.x; i < (BATCH_SIZE * num_chunks);
           i += NUM_THREADS) {
        // offset
        int row = i / num_chunks;
        int col = (i % num_chunks) * chunk_size;
        load_smem(input_smem(row, col), input_dmem_buffer(row, col));
      }

#pragma unroll
      for (int i = threadIdx.x; i < (BATCH_SIZE * num_chunks);
           i += NUM_THREADS) {
        // offset
        int row = i / num_chunks;
        int col = (i % num_chunks) * chunk_size;
        load_smem(mul_smem(row, col), mul_dmem_buffer(row, col));
      }
// load weight
#pragma unroll
      for (int i = threadIdx.x; i < (HIDDEN_SIZE * HIDDEN_SIZE / chunk_size);
           i += NUM_THREADS) {
        int row = i / (HIDDEN_SIZE / chunk_size);
        int col = (i % num_chunks) * chunk_size;
        load_smem(input_weight_smem(row, col), weight_dmem_buffer(row, col));
      }
      cp_async_fence();
      cp_async_wait<1>();
    }
    // SWAP the double buffer
    if ((for_idx & 1) == 0) {
      input_smem.set_ptr(shared_input_buffer);
      input_smem_buffer.set_ptr(shared_input);

      mul_smem.set_ptr(shared_mul_buffer);
      mul_smem_buffer.set_ptr(shared_mul);

      input_weight_smem.set_ptr(shared_weight_buffer);
      input_weight_smem_buffer.set_ptr(shared_weight);
    } else {
      input_smem.set_ptr(shared_input);
      input_smem_buffer.set_ptr(shared_input_buffer);
      mul_smem.set_ptr(shared_mul);
      mul_smem_buffer.set_ptr(shared_mul_buffer);
      input_weight_smem.set_ptr(shared_weight);
      input_weight_smem_buffer.set_ptr(shared_weight_buffer);
    }

    // do silu
    float const scalars[] = {0.0f};
    perform_element_unary_chain_kernel<false,
                                       decltype(silu_smem),
                                       decltype(input_smem),
                                       ElementUnaryOpType::SILU>(
        silu_smem, input_smem, scalars);

    // do mul
    mul<decltype(mul_output_smem), decltype(silu_smem), decltype(mul_smem)>(
        mul_output_smem, silu_smem, mul_smem);

    __syncthreads();
    uint32_t a_frag[4], b_frag[4];

    for (uint32_t n = 0; n < num_n / 4; n++) {
      for (uint32_t m = 0; m < num_m; m++) {
#pragma unroll
        for (uint32_t k = 0; k < num_k; k++) {
          int m_row = idx_in_warp % 16;
          int m_col = k * 16 + idx_in_warp / 16 * 8;
          // load B matrix,
          int n_row = idx_in_warp % 16 + k * 16;
          int n_col = warp_idx * 16 + idx_in_warp / 16 * 8;
          // load from a all zero mem
          bool is_valid = (m_row < BATCH_SIZE);
          T *src_ptr =
              is_valid ? mul_output_smem(m_row, m_col) : zero_buffer(0, 0);
          ldsm(src_ptr, &a_frag[0]);
          ldsm_t(input_weight_smem(n_row, n_col), &b_frag[0]);
          mma_m16n16k16_bf16bf16bf32(
              s_frag[m][n], a_frag, b_frag, s_frag[m][n]);
        }
      }
    }

    __syncthreads();
  }
  // reg write back to smem

  for (uint32_t n = 0; n < num_n / 4; n++) {
    for (uint32_t m = 0; m < num_m; m++) {
#pragma unroll
      for (uint32_t i = 0; i < 4; i++) {
        int row = idx_in_warp / 4 + 8 * (i % 2);
        int col = (idx_in_warp % 4) * 2 + 16 * warp_idx + 8 * (i / 2);
        if (row >= BATCH_SIZE) {
          continue;
        }
        output_smem.at(row, col) = float2bfloat16(s_frag[m][n][i * 2]);
        output_smem.at(row, col + 1) = float2bfloat16(s_frag[m][n][i * 2 + 1]);
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int i = threadIdx.x; i < (BATCH_SIZE * HIDDEN_SIZE); i += NUM_THREADS) {
    // offset
    int row = i / HIDDEN_SIZE;
    int col = (i % HIDDEN_SIZE);
    output_dmem.at(row, col) = output_smem.at(row, col);
  }
}

} // namespace kernel
