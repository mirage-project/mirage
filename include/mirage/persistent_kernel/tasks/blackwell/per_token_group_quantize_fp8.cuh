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
#pragma once
#include "../common//common_header.cuh"
#include <cstdint>
#include <type_traits>

#include <cuda_fp8.h>
namespace kernel {

template <int SUBWARP_SIZE>
__device__ __forceinline__ float group_reduce_max(float val) {
#pragma unroll
  for (int offset = SUBWARP_SIZE >> 1; offset > 0; offset >>= 1) {
    float other = __shfl_xor_sync(0xffffffff, val, offset, SUBWARP_SIZE);
    val = fmaxf(val, other);
  }
  return val;
}

__device__ __forceinline__ uint8_t encode_ue8m0(float scale) {
  // Compute ceil(log2(scale)) via direct IEEE 754 exponent extraction so an
  // exact power of two doesn't get bumped one bin too high by --use_fast_math
  // replacing log2f with __log2f (which returns -6.9999995... for 2^-7).
  // For a normal positive float, scale = 2^(exp-127) * (1 + mantissa/2^23):
  //   * mantissa == 0  -> scale == 2^(exp-127)         -> ceil(log2(scale)) =
  //   exp-127
  //   * mantissa  > 0  -> scale  > 2^(exp-127)         -> ceil(log2(scale)) =
  //   exp-127+1
  // fmaxf with 1e-30f keeps scale in the normal range (smallest normal
  // ~1.18e-38) so the exponent decode below is well-defined.
  scale = fmaxf(scale, 1e-30f);
  uint32_t bits = __float_as_uint(scale);
  int exp_unbiased = static_cast<int>((bits >> 23) & 0xff) - 127;
  uint32_t mantissa = bits & 0x7fffff;
  int ue8m0 = (mantissa == 0 ? exp_unbiased : exp_unbiased + 1) + 127;
  ue8m0 = max(0, min(255, ue8m0));
  return static_cast<uint8_t>(ue8m0);
}

// 2026-05-12 (QKV-a fusion, H8): added OUTPUT_STRIDE to fix overflow when the
// INPUT is a column slice of a wider buffer (GLOBAL_STRIDE > HIDDEN_SIZE) but
// the OUTPUT is sized for HIDDEN_SIZE per row. Default keeps legacy callers
// (input and output share the same stride) unchanged.
template <
    int BATCH_SIZE,
    int HIDDEN_SIZE,
    int GROUP_SIZE,
    int GLOBAL_STRIDE,
    int GROUP_TILES,
    typename T,
    typename DST_T,
    bool SCALE_UE8M0,
    int OUTPUT_STRIDE = GLOBAL_STRIDE,
    int ROWS_PER_TASK = 1,
    typename SCALE_PACKED_T = std::conditional_t<SCALE_UE8M0, uint32_t, float>>
__device__ __forceinline__ void
    per_token_group_quantize_fp8_task_impl(void const *__restrict__ input_ptr,
                                           void *__restrict__ output_q_ptr,
                                           void *__restrict__ output_s_ptr,
                                           float const eps,
                                           float const min_8bit,
                                           float const max_8bit,
                                           int const scale_outer_stride,
                                           int const row_idx,
                                           int const group_tile_idx,
                                           int const row_count_cap = -1) {
  // B15 (2026-05-15): row_count_cap is an optional per-CTA row-count
  // upper bound (used by NEW MoE silu_out quantize where the row axis
  // is permuted-expert layout and we want to cap by per-expert
  // actual_count). -1 sentinel = no cap (use BATCH_SIZE template).
  // Pointers
  T const *input = static_cast<T const *>(input_ptr);
  DST_T *output_q = static_cast<DST_T *>(output_q_ptr);
  SCALE_PACKED_T *output_s = static_cast<SCALE_PACKED_T *>(output_s_ptr);

  // Assume each thread handles 32B of data
  // each subwarp has n threads, n = group_size * 2 / 32,
  // when group_size = 128, n = 8, i.e. each subwarp has 8 threads handling each
  // group
  constexpr int WARP_SIZE = 32;
  constexpr int ELEMENTS_PER_THREAD = GROUP_SIZE / WARP_SIZE;
  constexpr int NUM_GROUPS_PER_ROW = HIDDEN_SIZE / GROUP_SIZE;
  constexpr int SCALE_ALIGNMENT = SCALE_UE8M0 ? 4 : 1;
  __shared__ uint8_t packed_scale_bytes[NUM_GROUPS_PER_ROW];

  // Assertions
  if constexpr (SCALE_UE8M0) {
    static_assert(GROUP_SIZE == 128,
                  "Packed UE8M0 scale currently requires GROUP_SIZE == 128");
    static_assert(std::is_same_v<SCALE_PACKED_T, uint32_t>,
                  "Packed UE8M0 scale must be stored as uint32");
  }
  // GROUP_TILES tiles each cover NUM_GROUPS_PER_ROW / GROUP_TILES consecutive
  // groups of one row. To prevent neighbour tiles from racing on the same
  // packed_idx slot (or reading uninitialized bytes from packed_scale_bytes),
  // tile boundaries must fall on SCALE_ALIGNMENT-group boundaries.
  static_assert(NUM_GROUPS_PER_ROW % GROUP_TILES == 0,
                "GROUP_TILES must divide NUM_GROUPS_PER_ROW");
  static_assert(((NUM_GROUPS_PER_ROW / GROUP_TILES) % SCALE_ALIGNMENT) == 0,
                "groups per tile must be a multiple of SCALE_ALIGNMENT");

  // Calculate indices
  int const thread_idx = threadIdx.x;
  int const lane_idx = thread_idx % WARP_SIZE;
  int const warp_idx = thread_idx / WARP_SIZE;
  int const num_groups_per_block = blockDim.x / WARP_SIZE;

  // Each task quantizes ROWS_PER_TASK consecutive logical rows. Default 1
  // preserves the legacy 1-row-per-CTA contract. ROWS_PER_TASK > 1 lets a
  // single CTA handle multiple rows so the runtime can launch grid.y =
  // ceil(BATCH_SIZE / ROWS_PER_TASK) ≤ num_workers and keep one task wave
  // on the persistent runtime instead of overflowing the scheduler queue.
  int const task_idx = row_idx;

  // C11 fast path (2026-05-17): when NUM_GROUPS_PER_ROW == 1 (typical K=128
  // case used by BMM Q-side q_nope quantize), the original serial-row loop
  // leaves num_groups_per_block-1 warps idle per row (1 group per row × 1
  // warp does the work, the rest sit idle). Restructure: each warp processes
  // its OWN row in parallel, num_groups_per_block rows per outer iter.
  // For 128-thread block (= 4 warps) + ROWS_PER_TASK=32, this gives 8 iters
  // instead of 32 → ~4× speedup on the BMM Q-side q_nope quantize task.
  if constexpr (NUM_GROUPS_PER_ROW == 1 && SCALE_UE8M0) {
    // Each warp owns one row per iter. Group is always group_idx=0 so
    // packed_idx=0 (one packed uint32 per row, low byte = UE8M0).
    constexpr int ROWS_PER_ITER = 4; // = blockDim.x / WARP_SIZE for 128 threads
    static_assert(ROWS_PER_ITER == 4,
                  "C11 fast path assumes 128-thread block (4 warps)");
#pragma unroll 1
    for (int r_base = 0; r_base < ROWS_PER_TASK; r_base += ROWS_PER_ITER) {
      int const r = r_base + warp_idx;
      if (r >= ROWS_PER_TASK) {
        continue;
      }
      int const batch_idx = task_idx * ROWS_PER_TASK + r;
      if (batch_idx < 0 || batch_idx >= BATCH_SIZE) {
        continue;
      }
      if (row_count_cap >= 0 && r >= row_count_cap) {
        continue;
      }
      // 1 group per row, group_idx=0.
      int const input_row_base = batch_idx * GLOBAL_STRIDE;
      int const output_row_base = batch_idx * OUTPUT_STRIDE;
      int const input_group_base = input_row_base + 0; // GROUP_SIZE * 0
      int const output_group_base = output_row_base + 0;

      // Per-warp local max across ELEMENTS_PER_THREAD elements.
      float local_max = eps;
#pragma unroll
      for (int ele_idx = 0; ele_idx < ELEMENTS_PER_THREAD; ++ele_idx) {
        int const input_idx = input_group_base + lane_idx + ele_idx * WARP_SIZE;
        float const abs_val = fabsf(static_cast<float>(input[input_idx]));
        local_max = fmaxf(abs_val, local_max);
      }
      // Warp-reduce max.
      float group_max = group_reduce_max<WARP_SIZE>(local_max);
      group_max = fmaxf(group_max, 1e-10f);
      float y_scale = group_max / max_8bit;
      uint8_t const scale_quant =
          __shfl_sync(0xffffffff, encode_ue8m0(y_scale), 0, WARP_SIZE);
      y_scale = exp2f(static_cast<float>(scale_quant) - 127.0f);

      // Quantize this row.
#pragma unroll
      for (int ele_idx = 0; ele_idx < ELEMENTS_PER_THREAD; ++ele_idx) {
        int const input_idx = input_group_base + lane_idx + ele_idx * WARP_SIZE;
        int const output_idx =
            output_group_base + lane_idx + ele_idx * WARP_SIZE;
        float const orig_val = static_cast<float>(input[input_idx]);
        float const quant_val =
            fminf(fmaxf(orig_val / y_scale, min_8bit), max_8bit);
        output_q[output_idx] = __nv_fp8_e4m3(quant_val);
      }

      // Pack scale: only 1 byte (NUM_GROUPS_PER_ROW=1) so packed_idx=0.
      // UE8M0 K-outer layout: output_s[packed_idx * scale_outer_stride +
      // batch_idx]. Lane 0 of this warp writes its row's scale uint32 directly.
      if (lane_idx == 0) {
        uint32_t const packed_scale = static_cast<uint32_t>(scale_quant);
        output_s[0 * scale_outer_stride + batch_idx] =
            static_cast<SCALE_PACKED_T>(packed_scale);
      }
    }
    return; // Fast path complete; skip the generic loop below.
  }

#pragma unroll 1
  for (int r = 0; r < ROWS_PER_TASK; ++r) {
    int const batch_idx = task_idx * ROWS_PER_TASK + r;
    if (batch_idx < 0 || batch_idx >= BATCH_SIZE) {
      return;
    }
    // B15: optional per-CTA row-count cap (e.g., NEW MoE silu_out
    // quantize bounded by per-expert actual_count). For inactive
    // expert (cap=0) we don't even enter this body since the codegen
    // returns earlier; for active expert with actual_count K, we
    // quantize the first K rows of the CTA's tile and skip the rest.
    if (row_count_cap >= 0 &&
        batch_idx - task_idx * ROWS_PER_TASK >= row_count_cap) {
      return;
    }
    // Input row may live in a wider parent buffer (GLOBAL_STRIDE) than the
    // output (OUTPUT_STRIDE). When OUTPUT_STRIDE == GLOBAL_STRIDE (default,
    // legacy callers) the two row bases are identical.
    int const input_row_base = batch_idx * GLOBAL_STRIDE;
    int const output_row_base = batch_idx * OUTPUT_STRIDE;
    int const group_tile = min(max(group_tile_idx, 0), GROUP_TILES - 1);
    int const groups_per_tile =
        (NUM_GROUPS_PER_ROW + GROUP_TILES - 1) / GROUP_TILES;
    int const group_begin = group_tile * groups_per_tile;
    int const group_end =
        min(group_begin + groups_per_tile, NUM_GROUPS_PER_ROW);

#pragma unroll
    for (int group_idx = group_begin + warp_idx; group_idx < group_end;
         group_idx += num_groups_per_block) {
      int const input_group_base = input_row_base + GROUP_SIZE * group_idx;
      int const output_group_base = output_row_base + GROUP_SIZE * group_idx;

      float local_max = eps;
#pragma unroll
      for (int ele_idx = 0; ele_idx < ELEMENTS_PER_THREAD; ++ele_idx) {
        int const input_idx = input_group_base + lane_idx + ele_idx * WARP_SIZE;
        float const abs_val = fabsf(static_cast<float>(input[input_idx]));
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
          // float32 scale is stored as [batch, num_groups] row-major.
          output_s[batch_idx * NUM_GROUPS_PER_ROW + group_idx] =
              static_cast<SCALE_PACKED_T>(y_scale);
        }
      }

#pragma unroll
      for (int ele_idx = 0; ele_idx < ELEMENTS_PER_THREAD; ++ele_idx) {
        int const input_idx = input_group_base + lane_idx + ele_idx * WARP_SIZE;
        int const output_idx =
            output_group_base + lane_idx + ele_idx * WARP_SIZE;
        float const orig_val = static_cast<float>(input[input_idx]);
        float const quant_val =
            fminf(fmaxf(orig_val / y_scale, min_8bit), max_8bit);
        output_q[output_idx] = __nv_fp8_e4m3(quant_val);
      }
    }

    if constexpr (SCALE_UE8M0) {
      // Ensure every warp finished writing its byte into packed_scale_bytes
      // before any thread packs four bytes into a uint32 below.
      __syncthreads();
      // UE8M0 scale layout: column-major [packed_k, aligned_batch].
      // This row's packed scales go in column `batch_idx`.
#pragma unroll
      for (int packed_idx = group_begin / SCALE_ALIGNMENT + thread_idx;
           packed_idx < (group_end + SCALE_ALIGNMENT - 1) / SCALE_ALIGNMENT;
           packed_idx += blockDim.x) {
        uint32_t packed_scale = 0;
#pragma unroll
        for (int pack_idx = 0; pack_idx < SCALE_ALIGNMENT; ++pack_idx) {
          int const group_idx = packed_idx * SCALE_ALIGNMENT + pack_idx;
          const uint8_t encoded = group_idx < NUM_GROUPS_PER_ROW
                                      ? packed_scale_bytes[group_idx]
                                      : 0;
          packed_scale |= static_cast<uint32_t>(encoded) << (pack_idx * 8);
        }
        output_s[packed_idx * scale_outer_stride + batch_idx] =
            static_cast<SCALE_PACKED_T>(packed_scale);
      }
      // Reset shared mem before next iter (next batch_idx) reuses it.
      if constexpr (ROWS_PER_TASK > 1) {
        __syncthreads();
      }
    }
  } // end ROWS_PER_TASK loop
}

} // namespace kernel
