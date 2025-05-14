
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

using bfloat16 = type::bfloat16_t;

// kernel for [16, 64] and any BATCH_SIZE < 16 [x, 64]
template <typename T>
__device__ __forceinline__ void norm_linear_kernel(void const *input_ptr,
                                                   void const *weight_ptr,
                                                   void *output_ptr) {
  constexpr int chunk_size = 16 / sizeof(T);

  constexpr int BATCH_SIZE = 1;
  constexpr int OUTPUT_SIZE = 64;
  constexpr int num_chunks = 8;
  constexpr int NUM_CHUNKS_A = 8;
  constexpr int NUM_CHUNKS_B = 512;

  // using SM80_16x8x16_F16F16F16F16_TNX2 = 16X16X16, threadLayout = 1,4,1
  // ->16X64X16
  constexpr int num_n = 4;
  constexpr int num_m = 1;
  constexpr int num_k = 4;
  int warp_idx = warp_id();
  int idx_in_warp = threadIdx.x & 0x1F;

  __restrict__ T const *d_input = static_cast<T const *>(input_ptr);

  __restrict__ T const *d_weight = static_cast<T const *>(weight_ptr);
  T __restrict__ *d_output = static_cast<T *>(output_ptr);

  dmem_row_const<T, 1, 64, 3584> input_dmem(d_input);
  dmem_row_const<T, 64, 64, 64> weight_dmem(d_weight);
  dmem_row<T, 1, 64, 64> output_dmem(d_output);

  extern __shared__ char smem[];

  // copy input
  T *shared_input = (T *)(smem + 2176);
  T *shared_input_buffer = (T *)(smem + 4224);
  // copy weight
  T *shared_weight = (T *)(smem + 6272);
  T *shared_weight_buffer = (T *)(smem + 14464);
  // intermidiate
  T *mm_output = (T *)(smem + 2176);
  T *element_unary_output = (T *)(smem + 128);
  T *reduction_output = (T *)(smem + 4224);
  // out
  T *shared_output = (T *)(smem + 128);

  *((__uint128_t *)smem) = 0ul;
  T *zero_buf = (T *)(smem);

  // define the swizzle mode

  // zero buffer

  smem_row<T, 1, 1, 1, 1, 8, 8> zero_buffer(zero_buf);

  smem_row<T, 1, 1, 1, 1, 64, 64> input_smem(shared_input);
  smem_row<T, 1, 1, 1, 1, 64, 64> input_smem_buffer(shared_input_buffer);

  smem_row<T, 3, 3, 3, 64, 64, 64> input_weight_smem(shared_weight);
  smem_row<T, 3, 3, 3, 64, 64, 64> input_weight_smem_buffer(
      shared_weight_buffer);

  smem_row<T, 1, 1, 1, 1, 64, 64> element_unary_smem(element_unary_output);

  smem_row<T, 1, 1, 1, 1, 64, 64> mm_output_smem(mm_output);
  smem_row<T, 1, 1, 1, 1, 1, 1> reduction_output_smem(reduction_output);

  smem_row<T, 1, 1, 1, 1, 64, 64> output_smem(shared_output);

// load input
#pragma unroll
  for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
    // offset
    int row = i >> 3;
    int col = (i & 0x7) << 3;
    load_smem(input_smem_buffer(row, col), input_dmem(row, col));
  }

// load weight
#pragma unroll
  for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
    int row = i >> 3;
    int col = (i & 0x7) << 3;
    load_smem(input_weight_smem_buffer(row, col), weight_dmem(row, col));
  }
  cp_async_fence();

  //  accumulator
  float s_frag[num_m][num_n][8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    s_frag[0][0][i] = 0.0f;
  }

  for (int for_idx = 0; for_idx < 56; for_idx++) {
    // copy
    if (for_idx + 1 != 56) {
      dmem_row_const<T, 1, 64, 3584> input_dmem_buffer(d_input +
                                                       64 * (for_idx + 1));
      dmem_row_const<T, 64, 64, 64> weight_dmem_buffer(d_weight +
                                                       4096 * (for_idx + 1));

#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
        // offset
        int row = i >> 3;
        int col = (i & 0x7) << 3;
        load_smem(input_smem(row, col), input_dmem_buffer(row, col));
      }
// load weight
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
        int row = i >> 3;
        int col = (i & 0x7) << 3;
        load_smem(input_weight_smem(row, col), weight_dmem_buffer(row, col));
      }
      cp_async_fence();
      cp_async_wait<1>();
    }
    // SWAP the double buffer
    if ((for_idx & 1) == 0) {
      input_smem.set_ptr(shared_input_buffer);
      input_smem_buffer.set_ptr(shared_input);
      input_weight_smem.set_ptr(shared_weight_buffer);
      input_weight_smem_buffer.set_ptr(shared_weight);
    } else {
      input_smem.set_ptr(shared_input);
      input_smem_buffer.set_ptr(shared_input_buffer);
      input_weight_smem.set_ptr(shared_weight);
      input_weight_smem_buffer.set_ptr(shared_weight_buffer);
    }
    __syncthreads();
    uint32_t a_frag[4], b_frag[4];

    for (uint32_t n = 0; n < (num_n >> 2); n++) {
      for (uint32_t m = 0; m < num_m; m++) {
#pragma unroll
        int m_row = idx_in_warp & 0xF;
        int n_col = (warp_idx << 4) + ((idx_in_warp >> 4) << 3);
        for (uint32_t k = 0; k < num_k; k++) {
          // int m_row = idx_in_warp % 16;
          // int m_col = k * 16 + idx_in_warp / 16 * 8;
          int m_col = (k << 4) + ((idx_in_warp >> 4) << 3);
          // load B matrix,
          int n_row = (idx_in_warp & 0xF) + (k << 4);
          // load from a all zero mem
          bool is_valid = (m_row < BATCH_SIZE);
          T *src_ptr = is_valid ? input_smem(m_row, m_col) : zero_buffer(0, 0);
          ldsm(src_ptr, &a_frag[0]);
          ldsm_t(input_weight_smem(n_row, n_col), &b_frag[0]);
          mma_m16n16k16_bf16bf16bf32(
              s_frag[m][n], a_frag, b_frag, s_frag[m][n]);
        }
      }
    }

    float const scalars[] = {0.0f, 0.000279f};
    // perform_element_unary_chain_kernel();
    perform_element_unary_chain_kernel<true,
                                       decltype(element_unary_smem),
                                       decltype(input_smem),
                                       ElementUnaryOpType::SQUARE,
                                       ElementUnaryOpType::MULSCALAR>(
        element_unary_smem, input_smem, scalars);
    __syncthreads();
  }
  // reg write back to smem

  for (uint32_t n = 0; n < (num_n >> 2); n++) {
    for (uint32_t m = 0; m < num_m; m++) {
#pragma unroll
      for (uint32_t i = 0; i < 4; i++) {
        int row = (idx_in_warp >> 2) + ((i & 0x1) << 3);
        if (row < BATCH_SIZE) {
          // continue;
          int col =
              ((idx_in_warp & 0x3) << 1) + (warp_idx << 4) + ((i >> 1) << 3);
          mm_output_smem.at(row, col) = bfloat16(s_frag[m][n][(i << 1)]);
          mm_output_smem.at(row, col + 1) =
              bfloat16(s_frag[m][n][(i << 1) + 1]);
        }
      }
    }
  }
  __syncthreads();

  float const scalars[] = {0.0f};
  reduction_sum_col<T,
                    decltype(reduction_output_smem),
                    decltype(element_unary_smem),
                    ElementUnaryOpType::SQRT>(
      reduction_output_smem, element_unary_smem, scalars);
  __syncthreads();

  div_col(output_smem, mm_output_smem, reduction_output_smem);
  __syncthreads();
#pragma unroll
  for (int i = threadIdx.x; i < OUTPUT_SIZE; i += NUM_THREADS) {
    // offset
    int row = 0;
    output_dmem.at(row, i) = output_smem.at(row, i);
    // output_dmem.at(row, col) = mm_output_smem.at(row, col);
    //      __uint128_t v =
    //      *((__uint128_t const *)(mm_output_smem(row, col)));
    //  *((__uint128_t *)(output_dmem(row, col))) =
    //      v;
  }

  // Chunked
  //   dmem_row<__uint128_t, 1, num_chunks, num_chunks> output_dmem_chunked(
  //       reinterpret_cast<__uint128_t *>(d_output));
  //   smem_row<__uint128_t, 0, 0, 0, 1, num_chunks, num_chunks>
  //       mm_output_smem_chunked(reinterpret_cast<__uint128_t *>(mm_output));
  // #pragma unroll
  //   for (int chunk_idx = threadIdx.x; chunk_idx < num_chunks;
  //        chunk_idx += NUM_THREADS) {
  //     output_dmem_chunked.at(chunk_idx) =
  //     mm_output_smem_chunked.at(chunk_idx);
  //   }
}

} // namespace kernel