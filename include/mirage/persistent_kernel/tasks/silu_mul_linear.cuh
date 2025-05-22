
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

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int SEQUENCE_SIZE>
__device__ __forceinline__ void silu_mul_linear_kernel(void const *input_ptr,
                                                       void const *mul_ptr,
                                                       void const *weight_ptr,
                                                       void *output_ptr) {

  constexpr int CHUNK_SIZE = 16 / sizeof(T);
  constexpr int TILE_SIZE = 64;
  constexpr int FORLOOP_RANGE = SEQUENCE_SIZE / TILE_SIZE;

  constexpr int NUM_CHUNKS_A = BATCH_SIZE * TILE_SIZE / CHUNK_SIZE;
  constexpr int NUM_CHUNKS_B = TILE_SIZE * OUTPUT_SIZE / CHUNK_SIZE;

  constexpr int CHUNKS_PER_ROW_A = TILE_SIZE / CHUNK_SIZE;
  constexpr int CHUNKS_PER_ROW_B = OUTPUT_SIZE / CHUNK_SIZE;

  constexpr int CHUNKS_PER_ROW_A_MASK = CHUNKS_PER_ROW_A - 1;
  constexpr int CHUNKS_PER_ROW_B_MASK = CHUNKS_PER_ROW_B - 1;

  constexpr int log2_CHUNK_SIZE = 3;
  constexpr int log2_CHUNKS_PER_ROW_A = 3;
  constexpr int log2_CHUNKS_PER_ROW_B = CHUNKS_PER_ROW_B == 2   ? 1
                                        : CHUNKS_PER_ROW_B == 4 ? 2
                                                                : 3;

  constexpr int num_m = 1;
  constexpr int num_n = OUTPUT_SIZE / 16;
  constexpr int num_k = TILE_SIZE / 16;
  constexpr int num_iters_n = (num_n + 3) >> 2;
  int warp_idx = warp_id();
  int idx_in_warp = threadIdx.x & 0x1F;

  assert(num_m > 0 && num_n > 0 && num_k > 0);
  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_mul = static_cast<T const *>(mul_ptr);
  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

  using InputDmem = dmem_row_const<T, BATCH_SIZE, TILE_SIZE, SEQUENCE_SIZE>;
  using WeightDmem = dmem_row_const<T, TILE_SIZE, OUTPUT_SIZE, OUTPUT_SIZE>;
  using OutputDmem = dmem_row<T, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE>;

  InputDmem input_dmem(d_input);
  InputDmem mul_dmem(d_mul);
  WeightDmem weight_dmem(d_weight);
  OutputDmem output_dmem(d_output);

  extern __shared__ char smem[];

  // zero buffer
  *((__uint128_t *)smem) = 0ul;
  T *zero_buf = (T *)(smem + 0); // 128 bytes

  // copy input, sizeof(T) * BATCH_SIZE * TILE_SIZE
  T *shared_input = (T *)(smem + 128);
  T *shared_input_buffer =
      (T *)(smem + 128 + sizeof(T) * BATCH_SIZE * TILE_SIZE);

  T *shared_mul = (T *)(smem + 128 + sizeof(T) * BATCH_SIZE * TILE_SIZE +
                        sizeof(T) * BATCH_SIZE * TILE_SIZE);
  T *shared_mul_buffer = (T *)(smem + 128 + sizeof(T) * BATCH_SIZE * TILE_SIZE +
                               sizeof(T) * BATCH_SIZE * TILE_SIZE +
                               sizeof(T) * BATCH_SIZE * TILE_SIZE);

  // copy weight, sizeof(T) * TILE_SIZE * OUTPUT_SIZE
  T *shared_weight = (T *)(smem + 128 + sizeof(T) * BATCH_SIZE * TILE_SIZE +
                           sizeof(T) * BATCH_SIZE * TILE_SIZE +
                           sizeof(T) * BATCH_SIZE * TILE_SIZE +
                           sizeof(T) * BATCH_SIZE * TILE_SIZE);
  T *shared_weight_buffer =
      (T *)(smem + 128 + sizeof(T) * BATCH_SIZE * TILE_SIZE +
            sizeof(T) * BATCH_SIZE * TILE_SIZE +
            sizeof(T) * BATCH_SIZE * TILE_SIZE +
            sizeof(T) * BATCH_SIZE * TILE_SIZE +
            sizeof(T) * TILE_SIZE * OUTPUT_SIZE);

  // intermidiate
  T *silu_output = (T *)(smem + 128 + sizeof(T) * BATCH_SIZE * TILE_SIZE +
                         sizeof(T) * BATCH_SIZE * TILE_SIZE +
                         sizeof(T) * BATCH_SIZE * TILE_SIZE +
                         sizeof(T) * BATCH_SIZE * TILE_SIZE +
                         sizeof(T) * TILE_SIZE * OUTPUT_SIZE +
                         sizeof(T) * TILE_SIZE * OUTPUT_SIZE);
  T *mul_output = (T *)(smem + 128 + sizeof(T) * BATCH_SIZE * TILE_SIZE +
                        sizeof(T) * BATCH_SIZE * TILE_SIZE +
                        sizeof(T) * BATCH_SIZE * TILE_SIZE +
                        sizeof(T) * BATCH_SIZE * TILE_SIZE +
                        sizeof(T) * TILE_SIZE * OUTPUT_SIZE +
                        sizeof(T) * TILE_SIZE * OUTPUT_SIZE +
                        sizeof(T) * BATCH_SIZE * TILE_SIZE);

  // out
  T *shared_output = shared_input; // reuse shared_input

  // define the swizzle mode

  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  using InputSmem = smem_row<T, 0, 0, 0, BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using WeightSmem = smem_row<T, 3, 3, 3, TILE_SIZE, OUTPUT_SIZE, OUTPUT_SIZE>;
  using OutputSmem = smem_row<T, 0, 0, 0, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_SIZE>;

  // zero buffer
  ZeroBufferSmem zero_buffer(zero_buf);

  InputSmem input_smem(shared_input);
  InputSmem input_smem_buffer(shared_input_buffer);

  InputSmem mul_smem(shared_mul);
  InputSmem mul_smem_buffer(shared_mul_buffer);

  WeightSmem input_weight_smem(shared_weight);
  WeightSmem input_weight_smem_buffer(shared_weight_buffer);

  InputSmem silu_smem(silu_output);

  InputSmem mul_output_smem(mul_output);

  OutputSmem output_smem(shared_output);

// load input
#pragma unroll
  for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
    // offset
    int row = i >> log2_CHUNKS_PER_ROW_A;
    int col = (i & CHUNKS_PER_ROW_A_MASK) << log2_CHUNK_SIZE;
    load_smem(input_smem_buffer(row, col), input_dmem(row, col));
  }

  // load mul
#pragma unroll
  for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
    // offset
    int row = i >> log2_CHUNKS_PER_ROW_A;
    int col = (i & CHUNKS_PER_ROW_A_MASK) << log2_CHUNK_SIZE;
    load_smem(mul_smem_buffer(row, col), mul_dmem(row, col));
  }

// load weight
#pragma unroll
  for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
    int row = i >> log2_CHUNKS_PER_ROW_B;
    int col = (i & CHUNKS_PER_ROW_B_MASK) << log2_CHUNK_SIZE;
    load_smem(input_weight_smem_buffer(row, col), weight_dmem(row, col));
  }
  cp_async_fence();

  //  accumulator
  float s_frag[num_m][num_n][8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    s_frag[0][0][i] = 0.0f;
  }

  for (int for_idx = 0; for_idx < FORLOOP_RANGE; for_idx++) {
    // copy
    if (for_idx + 1 != FORLOOP_RANGE) {
      InputDmem input_dmem_buffer(d_input + TILE_SIZE * (for_idx + 1));
      InputDmem mul_dmem_buffer(d_mul + TILE_SIZE * (for_idx + 1));
      WeightDmem weight_dmem_buffer(d_weight +
                                    TILE_SIZE * OUTPUT_SIZE * (for_idx + 1));

#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
        // offset
        int row = i >> log2_CHUNKS_PER_ROW_A;
        int col = (i & CHUNKS_PER_ROW_A_MASK) << log2_CHUNK_SIZE;
        load_smem(input_smem(row, col), input_dmem_buffer(row, col));
      }

#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_A; i += NUM_THREADS) {
        // offset
        int row = i >> log2_CHUNKS_PER_ROW_A;
        int col = (i & CHUNKS_PER_ROW_A_MASK) << log2_CHUNK_SIZE;
        load_smem(mul_smem(row, col), mul_dmem_buffer(row, col));
      }
// load weight
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_B; i += NUM_THREADS) {
        int row = i >> log2_CHUNKS_PER_ROW_B;
        int col = (i & CHUNKS_PER_ROW_B_MASK) << log2_CHUNK_SIZE;
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

    if (warp_idx < num_n) {
      uint32_t a_frag[4], b_frag[4];
      for (uint32_t n = 0; n < num_iters_n; n++) {
        int n_col = (n << 6) + (warp_idx << 4) + ((idx_in_warp >> 4) << 3);
        for (uint32_t m = 0; m < num_m; m++) {
          int m_row = (m << 4) + (idx_in_warp & 0xF);
#pragma unroll
          for (uint32_t k = 0; k < num_k; k++) {
            int m_col = (k << 4) + ((idx_in_warp >> 4) << 3);
            int n_row = (k << 4) + (idx_in_warp & 0xF);
            bool is_valid = (m_row < BATCH_SIZE);
            T *src_ptr =
                is_valid ? mul_output_smem(m_row, m_col) : zero_buffer(0, 0);
            ldsm(src_ptr, a_frag);
            ldsm_t(input_weight_smem(n_row, n_col), b_frag);
            mma_m16n16k16_bf16bf16bf32(
                s_frag[m][n], a_frag, b_frag, s_frag[m][n]);
          }
        }
      }
    }

    __syncthreads();
  }
  // reg write back to smem
  if (warp_idx < num_n) {
    for (uint32_t n = 0; n < num_iters_n; n++) {
      for (uint32_t m = 0; m < num_m; m++) {
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
          int row = (m << 4) + (idx_in_warp >> 2) + ((i & 0x1) << 3);
          if (row < BATCH_SIZE) {
            // continue;
            int col = (n << 6) + (warp_idx << 4) + ((idx_in_warp & 0x3) << 1) +
                      ((i >> 1) << 3);
            output_smem.at(row, col) = bfloat16(s_frag[m][n][(i << 1)]);
            output_smem.at(row, col + 1) =
                bfloat16(s_frag[m][n][(i << 1) | 0x1]);
          }
        }
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int i = threadIdx.x; i < OUTPUT_SIZE; i += NUM_THREADS) {
    // offset
    int row = 0;
    output_dmem.at(row, i) = output_smem.at(row, i);
  }
}

} // namespace kernel
