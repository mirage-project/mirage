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

template <typename _T,
          int _BATCH_SIZE,
          int _OUTPUT_SIZE,
          int _REDUCTION_SIZE,
          int _O_STRIDE,
          int _K_PIPE_MAX>
struct NormLinearKernelSpec {

  // --- Expose Template Parameters as Static Members ---
  using T = _T;
  static constexpr int BATCH_SIZE = _BATCH_SIZE;
  static constexpr int OUTPUT_SIZE = _OUTPUT_SIZE;
  static constexpr int REDUCTION_SIZE = _REDUCTION_SIZE;
  static constexpr int O_STRIDE = _O_STRIDE;
  static constexpr int K_PIPE_MAX = _K_PIPE_MAX;

  // --- Primary Kernel Shape Constants ---
  static constexpr int CHUNK_SIZE = 16 / sizeof(T);
  static constexpr int TILE_SIZE = 128;
  static_assert(REDUCTION_SIZE % TILE_SIZE == 0,
                "REDUCTION_SIZE must be a multiple of TILE_SIZE.");
  static constexpr int FORLOOP_RANGE = REDUCTION_SIZE / TILE_SIZE;

  // --- Derived Shape and Loop Constants ---
  static constexpr int NUM_CHUNKS_A = BATCH_SIZE * TILE_SIZE / CHUNK_SIZE;
  static constexpr int CHUNKS_PER_ROW_A = TILE_SIZE / CHUNK_SIZE;
  static constexpr int CHUNKS_PER_COL_B = TILE_SIZE / CHUNK_SIZE;
  static constexpr int log2_CHUNK_SIZE = log2_constexpr(CHUNK_SIZE);
  static constexpr int log2_CHUNKS_PER_ROW_A = log2_constexpr(CHUNKS_PER_ROW_A);
  static constexpr int log2_CHUNKS_PER_COL_B = log2_constexpr(CHUNKS_PER_COL_B);

  // --- Atom and Warp Configuration ---
  static constexpr int MAX_OUTPUT_ATOM_SIZE = max_power_of_two_le(
      (mirage::runtime::MAX_SHARE_MEMORY_SIZE / sizeof(T) -
       ((REDUCTION_SIZE + 2 * TILE_SIZE + 1) * BATCH_SIZE + REDUCTION_SIZE +
        8)) /
      (K_PIPE_MAX * TILE_SIZE +
       5 * BATCH_SIZE)); // This is the max output atom size that can be used
                         // (when num warp k is max, i.e. 4)
  static constexpr int OUTPUT_LIMIT = OUTPUT_SIZE <= 128 ? OUTPUT_SIZE : 128;
  static constexpr int OUTPUT_ATOM_SIZE = (OUTPUT_LIMIT <= MAX_OUTPUT_ATOM_SIZE)
                                              ? OUTPUT_LIMIT
                                              : MAX_OUTPUT_ATOM_SIZE;

  static constexpr int NUM_OUTPUT_ATOMS =
      OUTPUT_SIZE / OUTPUT_ATOM_SIZE; // Full output atoms
  static constexpr int LAST_OUTPUT_ATOM_SIZE = OUTPUT_SIZE % OUTPUT_ATOM_SIZE;

  static constexpr int NUM_ITERS_M =
      1; // TODO: Batch size more than 16 will cause error
  static_assert(BATCH_SIZE <= 16,
                "Batch size must be less than or equal to 16 for now.");
  // using SM80_16x8x16_F16F16F16F16_TNX2 = 16X16X16
  static constexpr int NUM_WARPS_N =
      ((OUTPUT_ATOM_SIZE + 15) / 16 == 3) ? 4 :
      ((OUTPUT_ATOM_SIZE + 15) / 16 <= 4 ? (OUTPUT_ATOM_SIZE + 15) / 16 : 4);
  static constexpr int LAST_NUM_WARPS_N =
      ((LAST_OUTPUT_ATOM_SIZE + 15) / 16 == 3) ? 4 :
      ((LAST_OUTPUT_ATOM_SIZE + 15) / 16 <= 4 ? (LAST_OUTPUT_ATOM_SIZE + 15) / 16 : 4);

  static constexpr int NUM_WARPS_K = 4 / NUM_WARPS_N;
  static constexpr int LAST_NUM_WARPS_K =
      (LAST_NUM_WARPS_N == 0) ? 0 : (4 / LAST_NUM_WARPS_N);

  // --- Shared Memory Layout Specification ---
  struct SMEM_OFFSETS {
    static constexpr size_t ZERO_BUFFER_OFFSET = 0;
    static constexpr size_t SHARED_INPUT_BUFFER_OFFSET =
        ZERO_BUFFER_OFFSET + sizeof(T) * 8;
    static constexpr size_t SHARED_NORM_WEIGHT_BUFFER_OFFSET =
        SHARED_INPUT_BUFFER_OFFSET +
        sizeof(T) * FORLOOP_RANGE * BATCH_SIZE * TILE_SIZE;
    static constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET =
        SHARED_NORM_WEIGHT_BUFFER_OFFSET +
        sizeof(T) * FORLOOP_RANGE * TILE_SIZE;
    static constexpr size_t MUL_OUTPUT_OFFSET =
        SHARED_WEIGHT_BUFFER_OFFSET +
        sizeof(T) * K_PIPE_MAX * TILE_SIZE * OUTPUT_ATOM_SIZE;
    static constexpr size_t ELEMENT_UNARY_OUTPUT_OFFSET =
        MUL_OUTPUT_OFFSET + sizeof(T) * BATCH_SIZE * TILE_SIZE;
    static constexpr size_t MM_INTERMEDIATE_OFFSET =
        ELEMENT_UNARY_OUTPUT_OFFSET + sizeof(T) * BATCH_SIZE * TILE_SIZE;
    static constexpr size_t MM_OUTPUT_OFFSET =
        MM_INTERMEDIATE_OFFSET +
        sizeof(T) * NUM_WARPS_K * BATCH_SIZE * OUTPUT_ATOM_SIZE;
    static constexpr size_t REDUCTION_OUTPUT_OFFSET =
        MM_OUTPUT_OFFSET + sizeof(T) * BATCH_SIZE * OUTPUT_ATOM_SIZE;
    static constexpr size_t SHARED_OUTPUT_OFFSET = MM_INTERMEDIATE_OFFSET;
  };
};

template <typename KernelSpec>
struct ProcessAtomFunctor {
  // --- Compile-time constants specific to the functor's task ---
  using T = typename KernelSpec::T;
  static constexpr int BATCH_SIZE = KernelSpec::BATCH_SIZE;
  static constexpr int REDUCTION_SIZE = KernelSpec::REDUCTION_SIZE;
  static constexpr int O_STRIDE = KernelSpec::O_STRIDE;
  static constexpr int K_PIPE_MAX = KernelSpec::K_PIPE_MAX;

  static constexpr int CHUNK_SIZE = KernelSpec::CHUNK_SIZE;
  static constexpr int TILE_SIZE = KernelSpec::TILE_SIZE;
  static constexpr int FORLOOP_RANGE = KernelSpec::FORLOOP_RANGE;
  static constexpr int NUM_CHUNKS_A = KernelSpec::NUM_CHUNKS_A;
  static constexpr int CHUNKS_PER_ROW_A = KernelSpec::CHUNKS_PER_ROW_A;
  static constexpr int CHUNKS_PER_COL_B = KernelSpec::CHUNKS_PER_COL_B;
  static constexpr int log2_CHUNK_SIZE = KernelSpec::log2_CHUNK_SIZE;
  static constexpr int log2_CHUNKS_PER_ROW_A =
      KernelSpec::log2_CHUNKS_PER_ROW_A;
  static constexpr int log2_CHUNKS_PER_COL_B =
      KernelSpec::log2_CHUNKS_PER_COL_B;

  static constexpr int NUM_ITERS_M = KernelSpec::NUM_ITERS_M;

  static constexpr int NUM_OUTPUT_ATOMS = KernelSpec::NUM_OUTPUT_ATOMS;
  static constexpr int LAST_OUTPUT_ATOM_SIZE =
      KernelSpec::LAST_OUTPUT_ATOM_SIZE;

  // --- Type aliases for memory accessors ---
  using InputDmem =
      dmem_row_const<T, BATCH_SIZE, REDUCTION_SIZE, REDUCTION_SIZE>;
  using NormWeightDmem = dmem_row_const<T, 1, REDUCTION_SIZE, REDUCTION_SIZE>;

  using ZeroBufferSmem = smem_row<T, 0, 0, 0, 1, 8, 8>;
  using InputSmem = smem_row<T, 0, 0, 0, BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using NormWeightSmem = smem_row<T, 0, 0, 0, 1, TILE_SIZE, TILE_SIZE>;
  using InputBufferSmem =
      smem_row<T, 0, 0, 0, FORLOOP_RANGE * BATCH_SIZE, TILE_SIZE, TILE_SIZE>;
  using NormWeightBufferSmem = smem_row<T,
                                        0,
                                        0,
                                        0,
                                        1,
                                        FORLOOP_RANGE * TILE_SIZE,
                                        FORLOOP_RANGE * TILE_SIZE>;

  using ReductionOutputSmem = smem_row<T, 0, 0, 0, BATCH_SIZE, 1, 1>;

  // --- State ---
  // Pointers to global memory
  InputDmem input_dmem;
  NormWeightDmem norm_weight_dmem;

  // Pointers to shared memory regions
  T *const zero_buf;
  T *const shared_input_buffer;
  T *const shared_norm_weight_buffer;
  T *const shared_weight_buffer;
  T *const mul_output;
  T *const element_unary_output;
  T *const mm_intermediate;
  T *const mm_output;
  T *const reduction_output;
  T *const shared_output;

  // Runtime parameters
  float const eps;

  // Thread identifiers
  int const warp_idx;
  int const lane_idx;
  int const thread_idx;

  __device__ ProcessAtomFunctor(void const *input_ptr,
                                void const *norm_weight_ptr,
                                T *zero_buf_ptr,
                                T *shared_input_buffer_ptr,
                                T *shared_norm_weight_buffer_ptr,
                                T *shared_weight_buffer_ptr,
                                T *mul_output_ptr,
                                T *element_unary_output_ptr,
                                T *mm_intermediate_ptr,
                                T *mm_output_ptr,
                                T *reduction_output_ptr,
                                T *shared_output_ptr,
                                float eps_val)
      : input_dmem(static_cast<T const *>(input_ptr)),
        norm_weight_dmem(static_cast<T const *>(norm_weight_ptr)),
        zero_buf(zero_buf_ptr), shared_input_buffer(shared_input_buffer_ptr),
        shared_norm_weight_buffer(shared_norm_weight_buffer_ptr),
        shared_weight_buffer(shared_weight_buffer_ptr),
        mul_output(mul_output_ptr),
        element_unary_output(element_unary_output_ptr),
        mm_intermediate(mm_intermediate_ptr), mm_output(mm_output_ptr),
        reduction_output(reduction_output_ptr),
        shared_output(shared_output_ptr), eps(eps_val), warp_idx(warp_id()),
        lane_idx(lane_id()), thread_idx(threadIdx.x) {}

  template <int OUTPUT_ATOM_SIZE, int NUM_WARPS_N, int NUM_WARPS_K>
  __device__ __forceinline__ void
      operator()(int output_atom_idx, T const *d_weight, T *d_output) {
    // Create local memory accessor objects that depend on template parameters
    using WeightDmem =
        dmem_col_const<T,
                       TILE_SIZE,
                       OUTPUT_ATOM_SIZE,
                       REDUCTION_SIZE>; // OUTPUT_ATOM_SIZE is dynamic
    using OutputDmem =
        dmem_row<T, BATCH_SIZE, OUTPUT_ATOM_SIZE, O_STRIDE>; // OUTPUT_ATOM_SIZE
                                                             // is dynamic

    using WeightSmem =
        smem_col<T, 3, 3, 3, TILE_SIZE, OUTPUT_ATOM_SIZE, TILE_SIZE>;
    using OutputSmem =
        smem_row<T, 0, 0, 0, BATCH_SIZE, OUTPUT_ATOM_SIZE, OUTPUT_ATOM_SIZE>;
    using WeightBufferSmem = smem_col<T,
                                      3,
                                      3,
                                      3,
                                      TILE_SIZE,
                                      K_PIPE_MAX * OUTPUT_ATOM_SIZE,
                                      TILE_SIZE>;

    WeightDmem weight_dmem(d_weight);
    OutputDmem output_dmem(d_output);

    ZeroBufferSmem zero_buffer(zero_buf);

    InputSmem input_smem(shared_input_buffer);
    NormWeightSmem norm_weight_smem(shared_norm_weight_buffer);
    WeightSmem weight_smem(shared_weight_buffer);

    InputSmem mul_output_smem(mul_output);
    InputSmem element_unary_smem(element_unary_output);
    OutputSmem mm_output_smem(mm_output);
    ReductionOutputSmem reduction_output_smem(reduction_output);

    OutputSmem output_smem(shared_output); // Reuse memory

    constexpr int NUM_CHUNKS_B = TILE_SIZE * OUTPUT_ATOM_SIZE / CHUNK_SIZE;
    constexpr int NUM_ITERS_N =
        (OUTPUT_ATOM_SIZE + (NUM_WARPS_N * 16) - 1) / (NUM_WARPS_N * 16);
    constexpr int NUM_ITERS_K = TILE_SIZE / NUM_WARPS_K / 16;
    constexpr int log2_NUM_WARPS_N = log2_constexpr(NUM_WARPS_N);
    constexpr int log2_NUM_ITERS_K = log2_constexpr(NUM_ITERS_K);

    using MatMulIntermediateSmem = smem_row<T,
                                            0,
                                            0,
                                            0,
                                            NUM_WARPS_K * BATCH_SIZE,
                                            OUTPUT_ATOM_SIZE,
                                            OUTPUT_ATOM_SIZE>;
    MatMulIntermediateSmem mm_intermediate_smem(mm_intermediate);

    int warp_row = warp_idx >> log2_NUM_WARPS_N;
    int warp_col = warp_idx & (NUM_WARPS_N - 1);

    InputBufferSmem input_buffer_smem(shared_input_buffer);
    NormWeightBufferSmem norm_weight_buffer_smem(shared_norm_weight_buffer);
    WeightBufferSmem weight_buffer_smem(shared_weight_buffer);

#pragma unroll
    for (int k_pipe = 0; k_pipe < K_PIPE_MAX - 1; k_pipe++) {
      if (output_atom_idx == 0) {
#pragma unroll
        for (int i = thread_idx; i < NUM_CHUNKS_A; i += NUM_THREADS) {
          int input_src_row = i >> log2_CHUNKS_PER_ROW_A;
          int input_dst_row =
              input_src_row +
              (k_pipe * BATCH_SIZE); // Batch size may not be power of 2
          int input_dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
          int input_src_col =
              input_dst_col + (k_pipe << log2_constexpr(TILE_SIZE));

          int norm_weight_col =
              input_dst_col + (k_pipe << log2_constexpr(TILE_SIZE));
          load_smem(input_buffer_smem(input_dst_row, input_dst_col),
                    input_dmem(input_src_row, input_src_col));
          if (input_src_row == 0) {
            load_smem(norm_weight_buffer_smem(0, input_dst_col),
                      norm_weight_dmem(0, norm_weight_col));
          }
        }
      }
#pragma unroll
      for (int i = thread_idx; i < NUM_CHUNKS_B; i += NUM_THREADS) {
        int dst_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
        int src_row = dst_row + (k_pipe << log2_constexpr(TILE_SIZE));
        int src_col = i >> log2_CHUNKS_PER_COL_B;
        int dst_col = src_col + ((k_pipe + 1) * OUTPUT_ATOM_SIZE);
        load_smem(weight_buffer_smem(dst_row, dst_col),
                  weight_dmem(src_row, src_col));
      }
      cp_async_fence();
    }

    // accumulator
    float s_frag[NUM_ITERS_M][NUM_ITERS_N][8];
#pragma unroll
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
        clear_8_floats(s_frag[m][n]);
      }
    }

    for (int for_idx = 0; for_idx < FORLOOP_RANGE; for_idx++) {
      // copy
      if (for_idx + K_PIPE_MAX - 1 < FORLOOP_RANGE) {
        if (output_atom_idx == 0) {
#pragma unroll
          for (int i = thread_idx; i < NUM_CHUNKS_A; i += NUM_THREADS) {
            int input_src_row = i >> log2_CHUNKS_PER_ROW_A;
            int input_dst_row =
                input_src_row + ((for_idx + K_PIPE_MAX - 1) * BATCH_SIZE);
            int input_dst_col = (i & (CHUNKS_PER_ROW_A - 1)) << log2_CHUNK_SIZE;
            int input_src_col = input_dst_col + ((for_idx + K_PIPE_MAX - 1)
                                                 << log2_constexpr(TILE_SIZE));
            load_smem(input_buffer_smem(input_dst_row, input_dst_col),
                      input_dmem(input_src_row, input_src_col));

            int norm_weight_col =
                input_dst_col +
                ((for_idx + K_PIPE_MAX - 1) << log2_constexpr(TILE_SIZE));
            if (input_src_row == 0) {
              load_smem(norm_weight_buffer_smem(0, norm_weight_col),
                        norm_weight_dmem(0, norm_weight_col));
            }
          }
        }
#pragma unroll
        for (int i = thread_idx; i < NUM_CHUNKS_B; i += NUM_THREADS) {
          int dst_row = (i & (CHUNKS_PER_COL_B - 1)) << log2_CHUNK_SIZE;
          int src_row = dst_row + ((for_idx + K_PIPE_MAX - 1)
                                   << log2_constexpr(TILE_SIZE));
          int col = i >> log2_CHUNKS_PER_COL_B;
          load_smem(weight_buffer_smem(dst_row, col),
                    weight_dmem(src_row, col));
        }
        cp_async_fence();
        cp_async_wait<K_PIPE_MAX - 1>();
      } else if (for_idx + K_PIPE_MAX - 1 == FORLOOP_RANGE) {
        cp_async_wait<0>();
      }

      // rotate the buffers
      input_smem.set_ptr(shared_input_buffer +
                         BATCH_SIZE * TILE_SIZE * for_idx);
      norm_weight_smem.set_ptr(shared_norm_weight_buffer + TILE_SIZE * for_idx);
      weight_buffer_smem.set_ptr(shared_weight_buffer +
                                 TILE_SIZE * OUTPUT_ATOM_SIZE *
                                     ((for_idx + 1) % K_PIPE_MAX));
      weight_smem.set_ptr(shared_weight_buffer +
                          TILE_SIZE * OUTPUT_ATOM_SIZE *
                              ((for_idx + 1) % K_PIPE_MAX));
      __syncthreads();

      mul_broadcast_row(mul_output_smem, input_smem, norm_weight_smem);
      __syncthreads();

      uint32_t a_frag[4], b_frag[4];
      for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
        int m_row = (lane_idx & 0xF);
        bool is_smem_valid = (m_row < BATCH_SIZE);
#pragma unroll
        for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
          int n_col = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4) +
                      ((lane_idx >> 4) << 3) + (lane_idx & 0x7);
          bool is_weight_valid = (n_col < OUTPUT_ATOM_SIZE);
#pragma unroll
          for (uint32_t k = 0; k < NUM_ITERS_K; k++) {
            int m_col = (warp_row << (4 + log2_NUM_ITERS_K)) + (k << 4) +
                        ((lane_idx >> 4) << 3);
            int n_row = (warp_row << (4 + log2_NUM_ITERS_K)) + (k << 4) +
                        (((lane_idx & 0xF) >> 3) << 3);
            T *a_src_ptr = is_smem_valid ? mul_output_smem(m_row, m_col)
                                         : zero_buffer(0, 0);
            T *b_src_ptr =
                is_weight_valid ? weight_smem(n_row, n_col) : zero_buffer(0, 0);

            ldsm(a_src_ptr, a_frag);
            ldsm(b_src_ptr, b_frag);
            mma_m16n16k16_bf16bf16bf32(
                s_frag[m][n], a_frag, b_frag, s_frag[m][n]);
          }
        }
      }

      if (output_atom_idx == 0) {
        float const scalars[] = {0.0f, 1.0f / float(REDUCTION_SIZE)};
        perform_element_unary_chain_kernel<true,
                                           decltype(element_unary_smem),
                                           decltype(input_smem),
                                           ElementUnaryOpType::SQUARE,
                                           ElementUnaryOpType::MULSCALAR>(
            element_unary_smem, input_smem, scalars);
      }
      __syncthreads();
    }

    // write back to shared memory
    for (uint32_t m = 0; m < NUM_ITERS_M; m++) {
#pragma unroll
      for (uint32_t n = 0; n < NUM_ITERS_N; n++) {
#pragma unroll
        for (uint32_t i = 0; i < 4; i++) {
          int row_in_warp = (lane_idx >> 2) + ((i & 0x1) << 3);
          if (row_in_warp < BATCH_SIZE) {
            int col = (n << (4 + log2_NUM_WARPS_N)) + (warp_col << 4) +
                      ((lane_idx & 0x3) << 1) + ((i >> 1) << 3);
            if (col < OUTPUT_ATOM_SIZE) {
              mm_intermediate_smem.at(warp_row + row_in_warp, col) =
                  bfloat16(s_frag[m][n][(i << 1)]);
              mm_intermediate_smem.at(warp_row + row_in_warp, col + 1) =
                  bfloat16(s_frag[m][n][(i << 1) | 0x1]);
            }
          }
        }
      }
    }
    __syncthreads();

    if (NUM_WARPS_K > 1) {
      reduction_sum_row<decltype(mm_output_smem),
                        decltype(mm_intermediate_smem)>(mm_output_smem,
                                                        mm_intermediate_smem);
      __syncthreads();
    }

    if (output_atom_idx == 0) {
      float const scalars[] = {eps, 0.0f};
      reduction_sum_col<T,
                        decltype(reduction_output_smem),
                        decltype(element_unary_smem),
                        ElementUnaryOpType::ADDSCALAR,
                        ElementUnaryOpType::SQRT>(
          reduction_output_smem, element_unary_smem, scalars);
      __syncthreads();
    }

    if (NUM_WARPS_K > 1) {
      div_col(output_smem, mm_output_smem, reduction_output_smem);
    } else {
      div_col(output_smem, mm_intermediate_smem, reduction_output_smem);
    }
    __syncthreads();

#pragma unroll
    for (int row = 0; row < BATCH_SIZE; row++) {
#pragma unroll
      for (int i = thread_idx; i < OUTPUT_ATOM_SIZE; i += NUM_THREADS) {
        output_dmem.at(row, i) = output_smem.at(row, i);
      }
    }
    if (output_atom_idx + 1 <
        (NUM_OUTPUT_ATOMS + (LAST_OUTPUT_ATOM_SIZE > 0))) {
      __syncthreads();
    }
  }
};

template <typename KernelSpec>
struct NormLinearHandler {

  using T = typename KernelSpec::T;
  static constexpr int BATCH_SIZE = KernelSpec::BATCH_SIZE;
  static constexpr int TILE_SIZE = KernelSpec::TILE_SIZE;
  static constexpr int REDUCTION_SIZE = KernelSpec::REDUCTION_SIZE;

  static constexpr int NUM_OUTPUT_ATOMS = KernelSpec::NUM_OUTPUT_ATOMS;
  static constexpr int OUTPUT_ATOM_SIZE = KernelSpec::OUTPUT_ATOM_SIZE;
  static constexpr int LAST_OUTPUT_ATOM_SIZE =
      KernelSpec::LAST_OUTPUT_ATOM_SIZE;
  static constexpr int NUM_WARPS_N = KernelSpec::NUM_WARPS_N;
  static constexpr int NUM_WARPS_K = KernelSpec::NUM_WARPS_K;
  static constexpr int LAST_NUM_WARPS_N = KernelSpec::LAST_NUM_WARPS_N;
  static constexpr int LAST_NUM_WARPS_K = KernelSpec::LAST_NUM_WARPS_K;
  using SMEM_OFFSETS = typename KernelSpec::SMEM_OFFSETS;

  // --- Member Variables ---
  // Pointers to global memory that change during execution
  T const *__restrict__ d_weight;
  T *__restrict__ d_output;

  // The functor for processing atoms
  ProcessAtomFunctor<KernelSpec> process_atom_functor;

  // --- Constructor ---
  __device__ NormLinearHandler(void const *input_ptr,
                               void const *norm_weight_ptr,
                               void const *weight_ptr,
                               float eps,
                               void *output_ptr)
      // Initialize the functor directly in the member initializer list.
      : d_weight(static_cast<T const *>(weight_ptr)),
        d_output(static_cast<T *>(output_ptr)),
        process_atom_functor(
            input_ptr,
            norm_weight_ptr,
            (T *)(get_smem_ptr(SMEM_OFFSETS::ZERO_BUFFER_OFFSET)),
            (T *)(get_smem_ptr(SMEM_OFFSETS::SHARED_INPUT_BUFFER_OFFSET)),
            (T *)(get_smem_ptr(SMEM_OFFSETS::SHARED_NORM_WEIGHT_BUFFER_OFFSET)),
            (T *)(get_smem_ptr(SMEM_OFFSETS::SHARED_WEIGHT_BUFFER_OFFSET)),
            (T *)(get_smem_ptr(SMEM_OFFSETS::MUL_OUTPUT_OFFSET)),
            (T *)(get_smem_ptr(SMEM_OFFSETS::ELEMENT_UNARY_OUTPUT_OFFSET)),
            (T *)(get_smem_ptr(SMEM_OFFSETS::MM_INTERMEDIATE_OFFSET)),
            (T *)(get_smem_ptr(SMEM_OFFSETS::MM_OUTPUT_OFFSET)),
            (T *)(get_smem_ptr(SMEM_OFFSETS::REDUCTION_OUTPUT_OFFSET)),
            (T *)(get_smem_ptr(SMEM_OFFSETS::SHARED_OUTPUT_OFFSET)),
            eps) {
    T *zero_buf = (T *)(get_smem_ptr(SMEM_OFFSETS::ZERO_BUFFER_OFFSET));
    *((__uint128_t *)zero_buf) = 0ul;

    T *element_unary_output =
        (T *)(get_smem_ptr(SMEM_OFFSETS::ELEMENT_UNARY_OUTPUT_OFFSET));
    clear_smem_buffer<T, BATCH_SIZE * TILE_SIZE>(element_unary_output);
  }

  // --- Execution Methods ---
  __device__ __forceinline__ void run() {
#pragma unroll
    for (int output_atom_idx = 0; output_atom_idx < NUM_OUTPUT_ATOMS;
         output_atom_idx++,
             d_weight += OUTPUT_ATOM_SIZE * REDUCTION_SIZE,
             d_output += OUTPUT_ATOM_SIZE) {
      process_atom_functor
          .template operator()<OUTPUT_ATOM_SIZE, NUM_WARPS_N, NUM_WARPS_K>(
              output_atom_idx, d_weight, d_output);
    }

    if constexpr (LAST_OUTPUT_ATOM_SIZE > 0) {
      process_atom_functor.template
          operator()<LAST_OUTPUT_ATOM_SIZE, LAST_NUM_WARPS_N, LAST_NUM_WARPS_K>(
              NUM_OUTPUT_ATOMS, d_weight, d_output);
    }
  }

  // Load first data that has no dependency
  // Potential usage; not implemented yet
  __device__ __forceinline__ void load_independent_data() {
    return;
  }

private:
  __device__ __forceinline__ char *get_smem_ptr(size_t offset = 0) {
    extern __shared__ char smem[];
    return smem + offset;
  }
};

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE = OUTPUT_SIZE,
          int K_PIPE_MAX = 2>
__device__ __forceinline__ void
    norm_linear_task_impl(void const *input_ptr,
                          void const *norm_weight_ptr,
                          void const *weight_ptr,
                          float eps,
                          void *output_ptr) {
  using KernelSpec = NormLinearKernelSpec<T,
                                          BATCH_SIZE,
                                          OUTPUT_SIZE,
                                          REDUCTION_SIZE,
                                          O_STRIDE,
                                          K_PIPE_MAX>;
  NormLinearHandler<KernelSpec> handler(
      input_ptr, norm_weight_ptr, weight_ptr, eps, output_ptr);
  handler.run();
}

} // namespace kernel
