/* Copyright 2026 CMU
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
//
// B37 (2026-05-15): fused RMSNorm + per-token-group FP8 quantize kernel.
//
// Replaces the two-task chain RMSnorm (5 us wave) -> Quantize (5 us wave)
// that currently feeds the qkv_a FP8 dense GEMM (~32 us). The standalone
// kernels both touch HBM for the BF16 rmsnorm output / quantize input; this
// kernel keeps the bf16 normalized row in shared memory and walks it once
// to compute per-group FP8 + scale, saving one HBM round-trip and one
// dispatch wave (~10 us / layer target).
//
// Design notes
// ------------
// * The kernel produces THREE outputs: the BF16 normalized output (so
//   downstream BF16 consumers still work, even though qkv_a feeds only the
//   FP8 path right now), the FP8 quantized output, and the per-group scale.
//   The BF16 output is computed in smem and written once at the end.
// * UE8M0 packed scale layout only (this is what the new FP8 dense GEMMs
//   consume). Float32 scale path also supported.
// * One CTA processes ROWS_PER_TASK consecutive rows; the row-loop mirrors
//   the rmsnorm_hopper / per_token_group_quantize_fp8 patterns so the
//   builder can shrink grid.x below `batch_size`.
// * row_count_cap: optional per-CTA cap so decode iterations (active_rows <
//   ROWS_PER_TASK rows for the trailing CTA) don't normalize/quantize stale
//   bf16. Matches the B34 active-rows convention used by rmsnorm_hopper.
// * IN_ROW_STRIDE / OUT_ROW_STRIDE / FP8_ROW_STRIDE: parent row width when
//   the caller passes mpk.narrow views. Per-task base pointers are already
//   offset by the runtime, so no in-kernel column shift is needed.
#pragma once
#include "../common/common_header.cuh"
#include "../common/utils.cuh"
#include "per_token_group_quantize_fp8.cuh"
#include <cstdint>
#include <type_traits>

#include <cuda_fp8.h>

namespace kernel {

template <
    typename T,
    typename DST_T,
    int BATCH_SIZE,
    int HIDDEN_DIM,
    int GROUP_SIZE,
    int NUM_THREADS = 256,
    int IN_ROW_STRIDE = HIDDEN_DIM,
    int OUT_ROW_STRIDE = HIDDEN_DIM,
    int FP8_ROW_STRIDE = HIDDEN_DIM,
    bool SCALE_UE8M0 = true,
    bool EMIT_BF16 = true,
    typename SCALE_PACKED_T = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__device__ __forceinline__ void
    fused_rmsnorm_quantize_fp8_impl(void const *__restrict__ input_ptr,
                                    void const *__restrict__ weight_ptr,
                                    void *__restrict__ output_bf16_ptr,
                                    void *__restrict__ output_fp8_ptr,
                                    void *__restrict__ output_scale_ptr,
                                    float eps,
                                    float scale_eps,
                                    float min_8bit,
                                    float max_8bit,
                                    int scale_outer_stride,
                                    int task_idx,
                                    int row_count_cap = -1) {
  static_assert(HIDDEN_DIM % NUM_THREADS == 0,
                "HIDDEN_DIM must be a multiple of NUM_THREADS");
  static_assert(HIDDEN_DIM % GROUP_SIZE == 0,
                "HIDDEN_DIM must be a multiple of GROUP_SIZE");
  if constexpr (SCALE_UE8M0) {
    static_assert(GROUP_SIZE == 128,
                  "Packed UE8M0 scale requires GROUP_SIZE == 128");
    static_assert(std::is_same_v<SCALE_PACKED_T, uint32_t>,
                  "Packed UE8M0 scale must be stored as uint32");
  }

  constexpr int WARP_SIZE = 32;
  constexpr int ELTS_PER_THREAD = HIDDEN_DIM / NUM_THREADS;
  constexpr int BYTES_PER_THREAD = ELTS_PER_THREAD * sizeof(T);
  constexpr int BYTES_PER_CP = []() {
    if constexpr (BYTES_PER_THREAD % 16 == 0) {
      return 16;
    } else if constexpr (BYTES_PER_THREAD % 8 == 0) {
      return 8;
    } else {
      static_assert(BYTES_PER_THREAD % 4 == 0);
      return 4;
    }
  }();
  constexpr int CHUNK_SIZE = BYTES_PER_CP / sizeof(T);
  constexpr int TILE_SIZE = NUM_THREADS * CHUNK_SIZE;
  static_assert(HIDDEN_DIM % TILE_SIZE == 0);
  constexpr int NUM_TILES = HIDDEN_DIM / TILE_SIZE;
  constexpr int NUM_CHUNKS_OUTPUT = HIDDEN_DIM / CHUNK_SIZE;
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  constexpr int NUM_GROUPS_PER_ROW = HIDDEN_DIM / GROUP_SIZE;
  constexpr int SCALE_ALIGNMENT = SCALE_UE8M0 ? 4 : 1;

  extern __shared__ char smem[];
  // Layout:
  //   [0                          .. HIDDEN_DIM)               input row (bf16)
  //   [HIDDEN_DIM                 .. 2*HIDDEN_DIM)             weight (bf16)
  //   [2*HIDDEN_DIM               .. 3*HIDDEN_DIM)             normalized
  //   (bf16) plus a small region for reduce + per-group packed scale bytes.
  constexpr size_t SHARED_WEIGHT_BUFFER_OFFSET = sizeof(T) * HIDDEN_DIM;
  constexpr size_t SHARED_OUTPUT_BUFFER_OFFSET =
      SHARED_WEIGHT_BUFFER_OFFSET + sizeof(T) * HIDDEN_DIM;
  constexpr size_t REDUCE_BUFFER_OFFSET =
      SHARED_OUTPUT_BUFFER_OFFSET + sizeof(T) * HIDDEN_DIM;
  // 32 warp partial sums max; in practice NUM_WARPS = 8 for NUM_THREADS=256.
  constexpr size_t REDUCE_BUFFER_BYTES = sizeof(float) * 32;
  constexpr size_t PACKED_SCALE_BYTES_OFFSET =
      REDUCE_BUFFER_OFFSET + REDUCE_BUFFER_BYTES;

  T *shared_input_buffer = reinterpret_cast<T *>(smem);
  T *shared_weight_buffer =
      reinterpret_cast<T *>(smem + SHARED_WEIGHT_BUFFER_OFFSET);
  T *shared_output_buffer =
      reinterpret_cast<T *>(smem + SHARED_OUTPUT_BUFFER_OFFSET);
  float *reduce_smem = reinterpret_cast<float *>(smem + REDUCE_BUFFER_OFFSET);
  uint8_t *packed_scale_bytes =
      reinterpret_cast<uint8_t *>(smem + PACKED_SCALE_BYTES_OFFSET);

  T const *__restrict__ d_weight = static_cast<T const *>(weight_ptr);

#pragma unroll 1
  for (int batch_idx = 0; batch_idx < BATCH_SIZE; batch_idx++) {
    if (row_count_cap >= 0 && batch_idx >= row_count_cap) {
      return;
    }
    // task_idx = global row-block index passed from the codegen
    // (= task_metadata.request_id). global_batch_idx selects the row
    // within the GLOBAL [M, K] layout for the SCALE buffer, which is
    // not pre-offset by dim_maps (UE8M0 column-major or f32 row-major
    // both need the global row index).
    int const global_batch_idx = task_idx * BATCH_SIZE + batch_idx;

    T const *__restrict__ curr_d_input =
        static_cast<T const *>(input_ptr) + batch_idx * IN_ROW_STRIDE;
    T *__restrict__ curr_d_output =
        EMIT_BF16 ? static_cast<T *>(output_bf16_ptr) +
                        batch_idx * OUT_ROW_STRIDE
                  : nullptr;
    DST_T *__restrict__ curr_d_fp8 =
        static_cast<DST_T *>(output_fp8_ptr) + batch_idx * FP8_ROW_STRIDE;
    SCALE_PACKED_T *__restrict__ d_scale =
        static_cast<SCALE_PACKED_T *>(output_scale_ptr);

    // Warm up first tile.
    {
      load_smem<T, BYTES_PER_CP>(shared_input_buffer + threadIdx.x * CHUNK_SIZE,
                                 curr_d_input + threadIdx.x * CHUNK_SIZE);
      // Weight only needs to be loaded once per task (it's the same across
      // all rows of this row-block). We still re-issue the cp_async on each
      // row to keep the pipeline simple; the L2/SMEM hits are cheap.
      load_smem<T, BYTES_PER_CP>(shared_weight_buffer +
                                     threadIdx.x * CHUNK_SIZE,
                                 d_weight + threadIdx.x * CHUNK_SIZE);
      cp_async_fence();
    }

    float sum = 0.0f;
#pragma unroll
    for (int for_idx = 0; for_idx < NUM_TILES; for_idx++) {
      if (for_idx + 1 < NUM_TILES) {
        load_smem<T, BYTES_PER_CP>(shared_input_buffer +
                                       threadIdx.x * CHUNK_SIZE +
                                       (for_idx + 1) * TILE_SIZE,
                                   curr_d_input + threadIdx.x * CHUNK_SIZE +
                                       (for_idx + 1) * TILE_SIZE);
        load_smem<T, BYTES_PER_CP>(
            shared_weight_buffer + threadIdx.x * CHUNK_SIZE +
                (for_idx + 1) * TILE_SIZE,
            d_weight + threadIdx.x * CHUNK_SIZE + (for_idx + 1) * TILE_SIZE);
        cp_async_fence();
        cp_async_wait<1>();
      } else if (for_idx + 1 == NUM_TILES) {
        cp_async_wait<0>();
      }
      __syncthreads();
#pragma unroll
      for (int i = threadIdx.x; i < TILE_SIZE; i += NUM_THREADS) {
        float val = (float)shared_input_buffer[for_idx * TILE_SIZE + i];
        sum += val * val;
      }
    }

#pragma unroll
    for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
      sum += shfl_xor_sync(sum, offset);
    }
    if (threadIdx.x % 32 == 0) {
      reduce_smem[threadIdx.x / 32] = sum;
    }
    __syncthreads();
    sum = threadIdx.x < NUM_WARPS ? reduce_smem[threadIdx.x] : 0.0f;
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
      sum += shfl_xor_sync(sum, offset);
    }
    if (threadIdx.x == 0) {
      reduce_smem[0] = sum;
    }
    __syncthreads();

    float rms_rcp = rsqrt(reduce_smem[0] / float(HIDDEN_DIM) + eps);

#pragma unroll
    for (int i = threadIdx.x; i < HIDDEN_DIM; i += NUM_THREADS) {
      float val = (float)shared_input_buffer[i];
      float w = (float)shared_weight_buffer[i];
      val *= rms_rcp * w;
      shared_output_buffer[i] = (T)val;
    }
    __syncthreads();

    // Optional BF16 store.
    if constexpr (EMIT_BF16) {
#pragma unroll
      for (int i = threadIdx.x; i < NUM_CHUNKS_OUTPUT; i += NUM_THREADS) {
        if constexpr (BYTES_PER_CP == 16) {
          *((__uint128_t *)((void *)&curr_d_output[i * CHUNK_SIZE])) =
              *((__uint128_t *)((void *)&shared_output_buffer[i * CHUNK_SIZE]));
        } else if constexpr (BYTES_PER_CP == 8) {
          *((uint64_t *)((void *)&curr_d_output[i * CHUNK_SIZE])) =
              *((uint64_t *)((void *)&shared_output_buffer[i * CHUNK_SIZE]));
        } else {
          *((uint32_t *)((void *)&curr_d_output[i * CHUNK_SIZE])) =
              *((uint32_t *)((void *)&shared_output_buffer[i * CHUNK_SIZE]));
        }
      }
    }

    // FP8 quantize phase. Reuse the warp-per-group layout from
    // per_token_group_quantize_fp8: each warp owns a contiguous group of
    // GROUP_SIZE (=128) elements.
    constexpr int ELEMENTS_PER_THREAD_FP8 = GROUP_SIZE / WARP_SIZE;
    int const thread_idx = threadIdx.x;
    int const lane_idx = thread_idx % WARP_SIZE;
    int const warp_idx = thread_idx / WARP_SIZE;
    int const num_groups_per_block = NUM_THREADS / WARP_SIZE;

#pragma unroll
    for (int group_idx = warp_idx; group_idx < NUM_GROUPS_PER_ROW;
         group_idx += num_groups_per_block) {
      int const smem_group_base = GROUP_SIZE * group_idx;
      int const out_group_base = GROUP_SIZE * group_idx;

      float local_max = scale_eps;
#pragma unroll
      for (int ele_idx = 0; ele_idx < ELEMENTS_PER_THREAD_FP8; ++ele_idx) {
        int const idx = smem_group_base + lane_idx + ele_idx * WARP_SIZE;
        float const abs_val =
            fabsf(static_cast<float>(shared_output_buffer[idx]));
        local_max = fmaxf(abs_val, local_max);
      }

      float y_scale = 0.0f;
      if constexpr (SCALE_UE8M0) {
        float group_max = group_reduce_max<WARP_SIZE>(local_max);
        group_max = fmaxf(group_max, 1e-10f);
        y_scale = group_max / max_8bit;
        const uint8_t scale_quant =
            __shfl_sync(0xffffffff, encode_ue8m0(y_scale), 0, WARP_SIZE);
        y_scale = exp2f(static_cast<float>(scale_quant) - 127.0f);
        if (lane_idx == 0) {
          packed_scale_bytes[group_idx] = scale_quant;
        }
      } else {
        float group_max = group_reduce_max<WARP_SIZE>(local_max);
        group_max = fmaxf(group_max, 1e-10f);
        y_scale = group_max / max_8bit;
        if (lane_idx == 0) {
          // [batch, num_groups] row-major. global_batch_idx = task_idx *
          // BATCH_SIZE + batch_idx maps from per-CTA row to global.
          d_scale[global_batch_idx * NUM_GROUPS_PER_ROW + group_idx] =
              static_cast<SCALE_PACKED_T>(y_scale);
        }
      }

#pragma unroll
      for (int ele_idx = 0; ele_idx < ELEMENTS_PER_THREAD_FP8; ++ele_idx) {
        int const idx = smem_group_base + lane_idx + ele_idx * WARP_SIZE;
        float const orig_val = static_cast<float>(shared_output_buffer[idx]);
        float const quant_val =
            fminf(fmaxf(orig_val / y_scale, min_8bit), max_8bit);
        curr_d_fp8[out_group_base + lane_idx + ele_idx * WARP_SIZE] =
            __nv_fp8_e4m3(quant_val);
      }
    }

    if constexpr (SCALE_UE8M0) {
      __syncthreads();
      // UE8M0 scale layout: column-major [packed_k, aligned_batch].
      // packed_k stride = scale_outer_stride (= aligned_batch in standard
      // callers). Row's packed scales go in column `global_batch_idx`.
#pragma unroll
      for (int packed_idx = thread_idx;
           packed_idx <
           (NUM_GROUPS_PER_ROW + SCALE_ALIGNMENT - 1) / SCALE_ALIGNMENT;
           packed_idx += NUM_THREADS) {
        uint32_t packed_scale = 0;
#pragma unroll
        for (int pack_idx = 0; pack_idx < SCALE_ALIGNMENT; ++pack_idx) {
          int const g = packed_idx * SCALE_ALIGNMENT + pack_idx;
          const uint8_t encoded =
              g < NUM_GROUPS_PER_ROW ? packed_scale_bytes[g] : 0;
          packed_scale |= static_cast<uint32_t>(encoded) << (pack_idx * 8);
        }
        d_scale[packed_idx * scale_outer_stride + global_batch_idx] =
            static_cast<SCALE_PACKED_T>(packed_scale);
      }
    }
    // Sync before next row reuses input/weight/output smem buffers via
    // cp_async. Required regardless of scale layout — the FP8 quantize
    // loop reads `shared_output_buffer`, so the next row's cp_async
    // overwrites of `shared_input_buffer` must wait for all warps to
    // finish their reads on this row.
    if constexpr (BATCH_SIZE > 1) {
      __syncthreads();
    }
  } // end batch_idx
}

} // namespace kernel
