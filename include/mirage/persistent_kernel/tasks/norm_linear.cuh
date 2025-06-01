
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
#include "dispatcher.cuh"
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
// OUTPUT_SIZE = 16, 32, 64, REDUCTION_SIZE = multiple of 64
template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
__device__ __forceinline__ void norm_linear_task_impl(void const *input_ptr,
                                                      void const *weight_ptr,
                                                      void *output_ptr) {
  // Here we assume the type of T is bfloat16, so the sizeof(T) is 2
  constexpr int CHUNK_SIZE = 16 / sizeof(T);
  constexpr int TILE_SIZE = 64;
  constexpr int FORLOOP_RANGE = REDUCTION_SIZE / TILE_SIZE;

  constexpr int NUM_CHUNKS_A = BATCH_SIZE * TILE_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_B = TILE_SIZE * OUTPUT_SIZE / CHUNK_SIZE;

  constexpr int CHUNKS_PER_ROW_A = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_ROW_B = OUTPUT_SIZE / CHUNK_SIZE;

  constexpr int log2_CHUNK_SIZE = 3;
  constexpr int log2_CHUNKS_PER_ROW_A = 3;
  constexpr int log2_CHUNKS_PER_ROW_B = CHUNKS_PER_ROW_B == 2   ? 1
                                        : CHUNKS_PER_ROW_B == 4 ? 2
                                                                : 3;

  // using SM80_16x8x16_F16F16F16F16_TNX2 = 16X16X16
  constexpr int NUM_WARP_N = OUTPUT_SIZE / 16; // 1, 2, 4
  constexpr int NUM_WARP_K = 4 / NUM_WARP_N;   // 4, 2, 1

  constexpr int NUM_ITERS_M = 1;
  constexpr int NUM_ITERS_N = 1;
  constexpr int NUM_ITERS_K = 4 / NUM_WARP_K; // 1, 2, 4

  constexpr int log2_NUM_WARP_N = NUM_WARP_N == 1   ? 0
                                  : NUM_WARP_N == 2 ? 1
                                                    : 2; // 0, 1, 2

  int warp_idx = warp_id();
  int warp_row = warp_idx >> log2_NUM_WARP_N;
  int warp_col = warp_idx & (NUM_WARP_N - 1);
  int idx_in_warp = threadIdx.x & 0x1F;

  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  using InputDmem = dmem_row_const<T, BATCH_SIZE, TILE_SIZE, REDUCTION_SIZE>;
  using WeightDmem = dmem_row_const<T, TILE_SIZE, OUTPUT_SIZE, OUTPUT_SIZE>;
  using OutputDmem = dmem_row<T, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE>;

  InputDmem input_dmem(d_input);
  WeightDmem weight_dmem(d_weight);
  OutputDmem output_dmem(d_output);

  extern __shared__ char smem[];

  // zero buffer
  T *zero_buf = (T *)(smem + 0); // 128 bytes
  *((__uint128_t *)smem) = 0ul;

  // copy input
  T *shared_input = (T *)(smem + 128); // sizeof(T) * BATCH_SIZE * TILE_SIZE
  T *shared_input_buffer =
      (T *)(smem + 128 +
            sizeof(T) * BATCH_SIZE *
                TILE_SIZE); // sizeof(T) * BATCH_SIZE * TILE_SIZE

  // copy weight
  T *shared_weight =
      (T *)(smem + 128 + sizeof(T) * BATCH_SIZE * TILE_SIZE +
            sizeof(T) * BATCH_SIZE *
                TILE_SIZE); // sizeof(T) * TILE_SIZE * OUTPUT_SIZE
  T *shared_weight_buffer =
      (T *)(smem + 128 + sizeof(T) * BATCH_SIZE * TILE_SIZE +
            sizeof(T) * BATCH_SIZE * TILE_SIZE +
            sizeof(T) * TILE_SIZE *
                OUTPUT_SIZE); // sizeof(T) * TILE_SIZE * OUTPUT_SIZE

  // intermediate
  T *element_unary_output =
      (T *)(smem + 128 + sizeof(T) * BATCH_SIZE * TILE_SIZE +
            sizeof(T) * BATCH_SIZE * TILE_SIZE +
            sizeof(T) * TILE_SIZE * OUTPUT_SIZE +
            sizeof(T) * TILE_SIZE *
                OUTPUT_SIZE); // sizeof(T) * BATCH_SIZE * TILE_SIZE
  for (int i = threadIdx.x; i < BATCH_SIZE * TILE_SIZE; i += NUM_THREADS) {
    element_unary_output[i] = T(0.0f);
  }

  T *mm_intermediate =
      (T *)(smem + 128 + sizeof(T) * BATCH_SIZE * TILE_SIZE +
            sizeof(T) * BATCH_SIZE * TILE_SIZE +
            sizeof(T) * TILE_SIZE * OUTPUT_SIZE +
            sizeof(T) * TILE_SIZE * OUTPUT_SIZE +
            sizeof(T) * BATCH_SIZE *
                TILE_SIZE); // sizeof(T) * BATCH_SIZE * OUTPUT_SIZE * NUM_WARP_K

  T *mm_output = (T *)(smem + 128 + sizeof(T) * BATCH_SIZE * TILE_SIZE +
                       sizeof(T) * BATCH_SIZE * TILE_SIZE +
                       sizeof(T) * TILE_SIZE * OUTPUT_SIZE +
                       sizeof(T) * TILE_SIZE * OUTPUT_SIZE +
                       sizeof(T) * BATCH_SIZE * TILE_SIZE +
                       sizeof(T) * BATCH_SIZE * OUTPUT_SIZE *
                           NUM_WARP_K); // sizeof(T) * BATCH_SIZE * OUTPUT_SIZE

  T *reduction_output =
      (T *)(smem + 128 + sizeof(T) * BATCH_SIZE * TILE_SIZE +
            sizeof(T) * BATCH_SIZE * TILE_SIZE +
            sizeof(T) * TILE_SIZE * OUTPUT_SIZE +
            sizeof(T) * TILE_SIZE * OUTPUT_SIZE +
            sizeof(T) * BATCH_SIZE * TILE_SIZE +
            sizeof(T) * BATCH_SIZE * OUTPUT_SIZE * NUM_WARP_K +
            sizeof(T) * BATCH_SIZE * OUTPUT_SIZE); // sizeof(T) * BATCH_SIZE * 1

  // out
  T *shared_output = mm_intermediate; // reuse mm_intermediate

  // define the swizzle mode

  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  using InputSmem = smem_row<T, 0, 0, 0, BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using WeightSmem = smem_row<T, 3, 3, 3, TILE_SIZE, OUTPUT_SIZE, OUTPUT_SIZE>;
  using OutputSmem = smem_row<T, 0, 0, 0, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE>;
  using MatMulIntermediateSmem =
      smem_row<T, 0, 0, 0, BATCH_SIZE * NUM_WARP_K, OUTPUT_SIZE, OUTPUT_SIZE>;
  using ReductionOutputSmem = smem_row<T, 0, 0, 0, BATCH_SIZE, 1, 1>;

  // zero buffer
  ZeroBufferSmem zero_buffer(zero_buf);

  InputSmem input_smem(shared_input);
  InputSmem input_smem_buffer(shared_input_buffer);

  WeightSmem input_weight_smem(shared_weight);
  WeightSmem input_weight_smem_buffer(shared_weight_buffer);

  InputSmem element_unary_smem(element_unary_output);

  MatMulIntermediateSmem mm_intermediate_smem(mm_intermediate);

  OutputSmem mm_output_smem(mm_output);
  OutputSmem output_smem(shared_output);

  ReductionOutputSmem reduction_output_smem(reduction_output);

// load input
#pragma unroll
  for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
    // offset
    int row = i >> log2_CHUNKS_PER_ROW_A;
    int col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
    load_smem(input_smem_buffer(row, col), input_dmem(row, col));
  }

// load weight
#pragma unroll
  for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
    int row = i >> log2_CHUNKS_PER_ROW_B;
    int col = (i & (CHUNKS_PER_ROW_B - 1)) << log2_CHUNK_SIZE;
    load_smem(input_weight_smem_buffer(row, col), weight_dmem(row, col));
  }
  cp_async_fence();

  //  accumulator
  float s_frag[NUM_ITERS_M][NUM_ITERS_N][8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    s_frag[0][0][i] = 0.0f;
  }

  for (int for_idx = 0; for_idx < FORLOOP_RANGE; for_idx++) {
    // copy
    if (for_idx + 1 != FORLOOP_RANGE) {
      InputDmem input_dmem_buffer(d_input + TILE_SIZE * (for_idx + 1));
      WeightDmem weight_dmem_buffer(d_weight +
                                    TILE_SIZE * OUTPUT_SIZE * (for_idx + 1));

#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
        int row = i >> log2_CHUNKS_PER_ROW_A;
        int col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
        load_smem(input_smem(row, col), input_dmem_buffer(row, col));
      }
// load weight
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
        int row = i >> log2_CHUNKS_PER_ROW_B;
        int col = (i & (CHUNKS_PER_ROW_B - 1)) << log2_CHUNK_SIZE;
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
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
      int m_row = (idx_in_warp & 0xF);
      bool is_valid = (m_row < BATCH_SIZE);
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
        int n_col = (warp_col << 4) + ((idx_in_warp >> 4) << 3);
#pragma unroll
        for (uint32_t k = 0; k < NUM_ITERS_K; k++) {
          int n_row = (warp_row << (4 + log2_NUM_WARP_N)) + (k << 4) +
                      (idx_in_warp & 0xF);
          int m_col = (warp_row << (4 + log2_NUM_WARP_N)) + (k << 4) +
                      ((idx_in_warp >> 4) << 3);
          T *src_ptr = is_valid ? input_smem(m_row, m_col) : zero_buffer(0, 0);
          ldsm(src_ptr, a_frag);
          ldsm_t(input_weight_smem(n_row, n_col), b_frag);
          mma_m16n16k16_bf16bf16bf32(
              s_frag[m][n], a_frag, b_frag, s_frag[m][n]);
        }
      }
    }

    float const scalars[] = {0.0f, 1.0f / float(REDUCTION_SIZE)};
    perform_element_unary_chain_kernel<true,
                                       decltype(element_unary_smem),
                                       decltype(input_smem),
                                       ElementUnaryOpType::SQUARE,
                                       ElementUnaryOpType::MULSCALAR>(
        element_unary_smem, input_smem, scalars);
    __syncthreads();
  }

  // reg write back to smem
  for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
    for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
#pragma unroll
      for (uint32_t i = 0; i < 4; i++) {
        int row_in_warp = (idx_in_warp >> 2) + ((i & 0x1) << 3);
        if (row_in_warp < BATCH_SIZE) {
          // continue;
          int col =
              (warp_col << 4) + ((idx_in_warp & 0x3) << 1) + ((i >> 1) << 3);
          mm_intermediate_smem.at(warp_row + row_in_warp, col) =
              bfloat16(s_frag[m][n][(i << 1)]);
          mm_intermediate_smem.at(warp_row + row_in_warp, col + 1) =
              bfloat16(s_frag[m][n][(i << 1) | 0x1]);
        }
      }
    }
  }
  __syncthreads();

  reduction_sum_row<decltype(mm_output_smem), decltype(mm_intermediate_smem)>(
      mm_output_smem, mm_intermediate_smem);
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
  }
}

template <typename T>
__device__ __forceinline__ void norm_linear_task(int output_size,
                                                 void const *input_ptr,
                                                 void const *weight_ptr,
                                                 void *output_ptr) {
  DISPATCH_OUTPUT_SIZE(
      output_size, norm_linear_task_impl, T, input_ptr, weight_ptr, output_ptr);
}

} // namespace kernel
