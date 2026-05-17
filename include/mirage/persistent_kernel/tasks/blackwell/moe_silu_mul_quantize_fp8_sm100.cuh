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
// C18 (2026-05-17): fused MoE silu·mul + per-token-group FP8 quantize.
//
// Replaces the two-task chain `moe_silu_mul` (writes BF16 silu_out) +
// `quantize_fp8(silu_out → silu_fp8 + silu_scale)`. The fused kernel
// reads w13_out (gate||up halves), computes silu(gate)*up per element,
// and immediately quantizes to FP8 + UE8M0 K-outer packed scale. Saves:
//   - one task launch + one HBM round-trip of silu_out (m_total * K_inter
//     bytes BF16 ≈ 4 MB at DSv3 m_total=2048 / K_inter=2048).
//   - one dispatch wave on the persistent scheduler.
//
// Layout matches the existing standalone quantize_fp8 → moe_permute path
// byte-for-byte (UE8M0 K-outer, stride = scale_outer_stride). Downstream
// consumers don't change.
//
// Algorithm
// ---------
//   * grid_dim = (m_total, 1, 1); one CTA per permuted-row, exactly
//     like the standalone silu→quantize chain.
//   * blockDim = 128 (4 warps). Each warp owns ROW_GROUPS / 4 groups via
//     the outer `for (group_idx = warp_idx; ...)` loop, identical to the
//     `per_token_group_quantize_fp8` warp-per-group pattern.
//   * Per element: each thread reads 1 bf16 gate + 1 bf16 mul, computes
//     `silu(gate) * mul`, stores the float in a per-thread register, then
//     reduces across the warp for max, encodes UE8M0, and writes FP8.
//   * No shared memory needed; everything lives in registers. The gate
//     and up halves are read from two different offsets of the same
//     w13_out row (gate at `[row][:K_INTER]`, up at `[row][K_INTER:2K]`).
//
#pragma once
#include "../common/utils.cuh"
#include "per_token_group_quantize_fp8.cuh"
#include <cstdint>
#include <cuda_fp8.h>

namespace kernel {

template <int K_INTER,           // per-row silu output dim (= 2048 DSv3)
          int GROUP_SIZE,        // 128
          int IN_ROW_STRIDE,     // distance between rows in w13_out (= 2*K_INTER)
          int OUT_ROW_STRIDE,    // distance between rows in silu_fp8 (= K_INTER)
          typename T,            // bf16
          typename DST_T,        // fp8_e4m3
          int ROWS_PER_TASK = 1>
__device__ __forceinline__ void moe_silu_mul_quantize_fp8_task_impl(
    void const *__restrict__ input_ptr,    // w13_out: [m_total, 2*K_INTER] bf16
    void *__restrict__ output_q_ptr,       // silu_fp8: [m_total, K_INTER] FP8
    void *__restrict__ output_s_ptr,       // silu_scale: K-outer UE8M0 uint32
    float const eps,                       // scale floor (e.g., 1e-10)
    float const min_8bit,                  // -448
    float const max_8bit,                  // 448
    int const scale_outer_stride,          // == aligned_batch (M_TOTAL padded)
    int const row_idx,                     // = task_metadata.request_id
    int const row_count_cap                // -1 = no cap; else stop after N
) {
  // ---- Type pointers ----
  T const *__restrict__ input = static_cast<T const *>(input_ptr);
  DST_T *__restrict__ output_q = static_cast<DST_T *>(output_q_ptr);
  uint32_t *__restrict__ output_s = static_cast<uint32_t *>(output_s_ptr);

  // ---- Constants ----
  constexpr int WARP_SIZE = 32;
  constexpr int ELEMENTS_PER_THREAD = GROUP_SIZE / WARP_SIZE;  // 4 for 128/32
  constexpr int NUM_GROUPS_PER_ROW = K_INTER / GROUP_SIZE;     // 16 for K=2048
  constexpr int SCALE_ALIGNMENT = 4;  // UE8M0: 4 bytes per uint32

  __shared__ uint8_t packed_scale_bytes[NUM_GROUPS_PER_ROW];

  static_assert(K_INTER % GROUP_SIZE == 0,
                "K_INTER must be a multiple of GROUP_SIZE");
  static_assert(GROUP_SIZE == 128,
                "Packed UE8M0 scale requires GROUP_SIZE == 128");

  // ---- Thread layout ----
  int const thread_idx = threadIdx.x;
  int const lane_idx = thread_idx % WARP_SIZE;
  int const warp_idx = thread_idx / WARP_SIZE;
  int const num_groups_per_block = blockDim.x / WARP_SIZE;  // 4 for 128 threads

  // ---- Row loop (ROWS_PER_TASK consecutive rows from this CTA) ----
  // The runtime pre-offsets `input_ptr` and `output_q_ptr` by
  // `bid.x * row_stride` via the (0, -1, -1) dim_maps, so within this
  // CTA the row offset is the LOCAL `r * row_stride` (not global
  // batch_idx * row_stride). `batch_idx` is reconstructed for the SCALE
  // buffer (which has dim_maps (-1) so no pre-offset).
#pragma unroll 1
  for (int r = 0; r < ROWS_PER_TASK; ++r) {
    int const batch_idx = row_idx * ROWS_PER_TASK + r;
    if (row_count_cap >= 0 && r >= row_count_cap) {
      return;
    }

    int const input_row_base = r * IN_ROW_STRIDE;       // gate half base
    int const output_row_base = r * OUT_ROW_STRIDE;

    // ---- Group loop (each warp processes its own groups) ----
#pragma unroll
    for (int group_idx = warp_idx; group_idx < NUM_GROUPS_PER_ROW;
         group_idx += num_groups_per_block) {
      int const input_gate_base = input_row_base + GROUP_SIZE * group_idx;
      // up half starts at K_INTER offset within the row.
      int const input_up_base = input_gate_base + K_INTER;
      int const output_group_base = output_row_base + GROUP_SIZE * group_idx;

      // Per-thread compute silu(gate) * up for ELEMENTS_PER_THREAD elements.
      float silu_mul_vals[ELEMENTS_PER_THREAD];
      float local_max = eps;
#pragma unroll
      for (int ele_idx = 0; ele_idx < ELEMENTS_PER_THREAD; ++ele_idx) {
        int const idx_in_group = lane_idx + ele_idx * WARP_SIZE;
        float const gate = static_cast<float>(input[input_gate_base + idx_in_group]);
        float const mul  = static_cast<float>(input[input_up_base + idx_in_group]);
        float const sig  = 1.0f / (1.0f + __expf(-gate));
        float const sm   = gate * sig * mul;
        silu_mul_vals[ele_idx] = sm;
        float const abs_val = fabsf(sm);
        local_max = fmaxf(abs_val, local_max);
      }

      // Warp reduce for max.
      float group_max = group_reduce_max<WARP_SIZE>(local_max);
      group_max = fmaxf(group_max, 1e-10f);
      float const y_scale_f = group_max / max_8bit;
      uint8_t const scale_quant =
          __shfl_sync(0xffffffff, encode_ue8m0(y_scale_f), 0, WARP_SIZE);
      float const y_scale = exp2f(static_cast<float>(scale_quant) - 127.0f);
      if (lane_idx == 0) {
        packed_scale_bytes[group_idx] = scale_quant;
      }

      // Quantize and write FP8.
#pragma unroll
      for (int ele_idx = 0; ele_idx < ELEMENTS_PER_THREAD; ++ele_idx) {
        int const idx_in_group = lane_idx + ele_idx * WARP_SIZE;
        float const orig = silu_mul_vals[ele_idx];
        float const quant = fminf(fmaxf(orig / y_scale, min_8bit), max_8bit);
        output_q[output_group_base + idx_in_group] = __nv_fp8_e4m3(quant);
      }
    }

    // ---- Pack scale bytes to K-outer UE8M0 uint32 ----
    __syncthreads();
#pragma unroll
    for (int packed_idx = thread_idx;
         packed_idx < (NUM_GROUPS_PER_ROW + SCALE_ALIGNMENT - 1) / SCALE_ALIGNMENT;
         packed_idx += blockDim.x) {
      uint32_t packed_scale = 0;
#pragma unroll
      for (int pack_idx = 0; pack_idx < SCALE_ALIGNMENT; ++pack_idx) {
        int const g = packed_idx * SCALE_ALIGNMENT + pack_idx;
        uint8_t const b = g < NUM_GROUPS_PER_ROW ? packed_scale_bytes[g] : 0;
        packed_scale |= static_cast<uint32_t>(b) << (pack_idx * 8);
      }
      // K-outer layout: column-major [packed_k, aligned_batch].
      output_s[packed_idx * scale_outer_stride + batch_idx] = packed_scale;
    }
    // Sync before next row (next batch_idx) reuses packed_scale_bytes.
    if constexpr (ROWS_PER_TASK > 1) {
      __syncthreads();
    }
  }
}

} // namespace kernel
