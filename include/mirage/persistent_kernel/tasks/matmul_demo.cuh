
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
#pragma once
namespace kernel {

using bfloat16 = type::bfloat16;

// a 16X64X4K kernel for reference
template <typename T, int BATCH_SIZE, int HIDDEN_SIZE, int NUM_THREADS>
__device__ __forceinline__ void norm_linear_kernel(void const *input_ptr,
                                                   void const *weight_ptr,
                                                   void *output_ptr) {
  constexpr int chunk_size = 16 / sizeof(T);

  constexpr int num_chunks = HIDDEN_SIZE / chunk_size;

  // using SM80_16x8x16_F16F16F16F16_TNX2 = 16X16X16
  constexpr int num_n = HIDDEN_SIZE / 16;
  constexpr int num_m = BATCH_SIZE / 16;
  constexpr int num_k = HIDDEN_SIZE / 16;
  int warp_idx = warp_id();
  int idx_in_warp = threadIdx.x % 32;

  assert(num_m > 0 && num_n > 0 && num_k > 0);
  // input_tile [BATCH_SIZE, HIDDEN_SIZE]
  // weight_tile [HIDDEN_SIZE, HIDDEN_SIZE]
  __restrict__ T const *d_input = static_cast<T const *>(input_ptr);

  __restrict__ T const *d_weight =
      static_cast<T const *>(weight_ptr) + blockIdx.x * 64 * 1;
  T __restrict__ *d_output = static_cast<T *>(output_ptr) + blockIdx.x * 64 * 1;

  dmem_row_const<T, 16, 64, 4096> input_dmem(d_input);
  dmem_row_const<T, 64, 64, 64> weight_dmem(d_weight);
  dmem_row<T, 16, 64, 64> output_dmem(d_output);

  extern __shared__ T smem[];

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

  // define the swizzle mode
  smem_row<T, 3, 3, 3, 16, 64, 64> input_smem(shared_input);
  smem_row<T, 3, 3, 3, 16, 64, 64> input_smem_buffer(shared_input_buffer);

  smem_row<T, 3, 3, 3, 64, 64, 64> input_weight_smem(shared_weight);
  smem_row<T, 3, 3, 3, 64, 64, 64> input_weight_smem_buffer(
      shared_weight_buffer);

  smem_row<T, 1, 1, 1, 16, 64, 64> element_unary_smem(element_unary_output);

  smem_row<T, 3, 3, 3, 16, 64, 64> mm_output_smem(mm_output);
  smem_row<T, 1, 1, 1, 16, 1, 1> reduction_output_smem(reduction_output);

  smem_row<T, 1, 1, 1, 16, 64, 64> output_smem(shared_output);

// load input
#pragma unroll
  for (int i = threadIdx.x; i < (BATCH_SIZE * num_chunks); i += NUM_THREADS) {
    // offset
    int row = i / num_chunks;
    int col = (i % num_chunks) * chunk_size;
    load_smem(input_smem_buffer(row, col), input_dmem(row, col));
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
  clear_8_floats(s_frag[0][0]);

  for (int for_idx = 0; for_idx < 64; for_idx++) {
    // copy
    if (for_idx + 1 != 64) {
      dmem_row_const<T, 16, 64, 4096> input_dmem_buffer(d_input +
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

    for (uint32_t n = 0; n < num_n / 4; n++) {
      for (uint32_t m = 0; m < num_m; m++) {
#pragma unroll
        for (uint32_t k = 0; k < num_k; k++) {
          int m_row = idx_in_warp % 16;
          int m_col = k * 16 + idx_in_warp / 16 * 8;
          // load B matrix,
          int n_row = idx_in_warp % 16 + k * 16;
          int n_col = warp_idx * 16 + idx_in_warp / 16 * 8;
          ldsm(input_smem(m_row, m_col), &a_frag[0]);
          ldsm_t(input_weight_smem(n_row, n_col), &b_frag[0]);
          //  printf("%f, %f, %f, %f, ",
          //  __bfloat162float(input_weight_smem.at(n_row, n_col)),
          //  __bfloat162float(input_weight_smem.at(n_row, n_col + 1)),
          //  __bfloat162float(input_weight_smem.at(n_row, n_col + 2)),
          //  __bfloat162float(input_weight_smem.at(n_row, n_col + 3)));
          //  if(for_idx == 0){
          // printf("threadid %d, k %d, n_row %d, n_col %d, value %f, %f, %f,
          // %f\n", threadIdx.x, k, n_row, n_col,
          //   __bfloat162float(input_weight_smem.at(n_row, n_col)),
          //   __bfloat162float(input_weight_smem.at(n_row, n_col + 1)),
          //   __bfloat162float(input_weight_smem.at(n_row, n_col + 2)),
          //   __bfloat162float(input_weight_smem.at(n_row, n_col + 3)));
          // }

          mma_m16n16k16_bf16bf16bf32(
              s_frag[m][n], a_frag, b_frag, s_frag[m][n]);

          //     if(m == 0 && n == 0 && threadIdx.x == 0 ){
          //   printf("for_idx = %d, s_frag[m][n] %f\n", for_idx,
          //   s_frag[0][0][0]);
          //  }
        }
      }
    }

    //   const float scalars[] = {0.0f, 0.000244f};
    // // perform_element_unary_chain_kernel();
    // perform_element_unary_chain_kernel<true,
    //             decltype(element_unary_smem),
    //             decltype(input_smem),
    //             ElementUnaryOpType::SQUARE,
    //             ElementUnaryOpType::MULSCALAR>(element_unary_smem,
    //             input_smem,  scalars);
    __syncthreads();
  }
  // reg write back to smem

  for (uint32_t n = 0; n < num_n / 4; n++) {
    for (uint32_t m = 0; m < num_m; m++) {
#pragma unroll
      for (uint32_t i = 0; i < 4; i++) {
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float:~:text=The%20layout%20of%20the%20fragments%20held%20by%20different%20threads%20is%20shown%20in%20Figure%2083.
        int row = idx_in_warp / 4 + 8 * (i % 2);
        int col = (idx_in_warp % 4) * 2 + 16 * warp_idx + 8 * (i / 2);
        mm_output_smem.at(row, col) = bfloat16(s_frag[m][n][i * 2]);
        mm_output_smem.at(row, col + 1) = bfloat16(s_frag[m][n][i * 2 + 1]);
        // mm_output_smem.at(row, col) = bfloat16(40.7500f);
        // mm_output_smem.at(row, col+1) = bfloat16(40.7500f);
        // printf("mm output A%f, B%f, rol%d, col %d, value %f, %f\n",
        //   s_frag[m][n][i*2], s_frag[m][n][i*2+1], row, col,
        //   float(mm_output_smem.at(row, col)),
        //   float(mm_output_smem.at(row, col+1)));
      }
    }
  }
  __syncthreads();
#pragma unroll
  for (int i = threadIdx.x; i < (BATCH_SIZE * HIDDEN_SIZE); i += NUM_THREADS) {
    // offset
    int row = i / HIDDEN_SIZE;
    int col = (i % HIDDEN_SIZE);
    // printf("output %f, row is %d, col is %d, thread is %d\n",
    //   __bfloat162float(mm_output_smem.at(row, col)), row, col, threadIdx.x);
    output_dmem.at(row, col) = mm_output_smem.at(row, col);
  }
}

} // namespace kernel