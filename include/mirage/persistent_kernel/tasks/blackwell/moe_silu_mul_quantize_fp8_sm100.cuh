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
#include "per_token_group_quantize_fp8.cuh"

#include <cstdint>
#include <cuda_fp8.h>

namespace kernel {

// ---------------------------------------------------------------------------
// MoE activation SwiGLU + fp32-block-scale FP8 quantize, in ONE task.
//
// WHY IT EXISTS (M4-I9).  The routed-expert chain is
//   w13 -> moe_silu_mul -> quantize_fp8(f32 scale) -> w2
// and `moe_act`, the bf16 tensor between the middle two, has exactly one
// consumer.  M4-I8 measured what a chain record costs at bs1: its own duration
// (4.10 us for this quantize site) PLUS ~1.15 us of event visibility or ~1.55 us
// of queue-pop latency PLUS its barrier pair, and per-task overhead is set by
// the COUNT of barriers/scoped accesses rather than their scope.  Merging the
// two tasks removes one record from the per-layer critical chain at all 40
// layers, and removes the bf16 round trip for `moe_act` (256 KiB of traffic per
// layer at mbt=16, topk=8, inter=512) because the value never leaves registers.
//
// BIT-EXACT BY CONSTRUCTION.  This merges arithmetic that HEAD already performs
// in sequence, at the SAME cast positions, on the SAME operand groups:
//
//  * the activation is `T(silu_f32) * mul_val` in type T, i.e. the exact
//    expression `silu_mul_task_impl` stores into `moe_act`
//    (silu_mul.cuh:37-39) -- fp32 sigmoid, ROUNDED TO T, then multiplied by the
//    bf16 `up` half through T's own `operator*` and kept in T.  No rounding
//    position moves; the only difference is that the T value stays in a
//    register instead of visiting global memory and being read straight back.
//  * the amax is over the SAME 128-element group with the SAME lane->element
//    map (`lane + e*WARP_SIZE`) and the SAME seed (`eps`), reduced with the
//    SAME `group_reduce_max<WARP_SIZE>`.  fmaxf is exact and order-independent,
//    so even the reduction ORDER cannot move a bit.
//  * `y_scale`, the fp32 scale store, and
//    `fp8(clamp(orig / y_scale, min_8bit, max_8bit))` are copied verbatim from
//    `per_token_group_quantize_fp8_task_impl` (SCALE_UE8M0 = false branch).
//
// Rows are handled independently, and a group's bytes and scale depend only on
// that group's own 128 elements, so redistributing rows/groups over CTAs cannot
// change a byte -- the same argument `quantize_fp8_layer`'s `row_partition`
// already relies on.  Consequently this task may run at the FINER silu grid
// (one task per (token, expert-slot)) while the standalone quantize ran one
// task per token; no work is added, it is only spread wider.
//
// NO EXTRA BARRIER.  Warp `w` produces exactly the group it consumes, so the
// activation never crosses a warp boundary and no `__syncthreads()` and no
// shared-memory staging is needed.  That matters: M4-I8's arm O measured
// ~470 ns of makespan per extra scoped load + barrier pair per chain record.
//
// Template parameters mirror the two impls being merged:
//   NUM_ROWS       rows in this task's tile (compile-time bound of the loop)
//   OUTPUT_SIZE    intermediate size per row = HIDDEN_SIZE of the quantize
//   GROUP_SIZE     scale group (128)
//   I_STRIDE       row stride of the gate|up input (2 * OUTPUT_SIZE in tensor)
//   O_STRIDE       row stride of the fp8 output
//   S_STRIDE       row stride of the fp32 scale (= OUTPUT_SIZE / GROUP_SIZE)
// ---------------------------------------------------------------------------
template <int NUM_ROWS,
          int OUTPUT_SIZE,
          int GROUP_SIZE,
          int I_STRIDE,
          int O_STRIDE,
          int S_STRIDE,
          typename T,
          typename DST_T>
__device__ __forceinline__ void moe_silu_mul_quantize_fp8_task_impl(
    void const *__restrict__ input_ptr,
    void *__restrict__ output_q_ptr,
    void *__restrict__ output_s_ptr,
    float const eps,
    float const min_8bit,
    float const max_8bit,
    int const num_active_rows) {
  T const *__restrict__ input = static_cast<T const *>(input_ptr);
  DST_T *__restrict__ output_q = static_cast<DST_T *>(output_q_ptr);
  float *__restrict__ output_s = static_cast<float *>(output_s_ptr);

  constexpr int WARP_SIZE = 32;
  constexpr int ELEMENTS_PER_THREAD = GROUP_SIZE / WARP_SIZE;
  constexpr int NUM_GROUPS_PER_ROW = OUTPUT_SIZE / GROUP_SIZE;
  static_assert(OUTPUT_SIZE % GROUP_SIZE == 0,
                "the scale group must not straddle the row boundary");
  static_assert(GROUP_SIZE % WARP_SIZE == 0,
                "one warp must cover a whole scale group");
  static_assert(S_STRIDE >= NUM_GROUPS_PER_ROW,
                "scale row stride must hold this row's groups");

  int const lane_idx = threadIdx.x % WARP_SIZE;
  int const warp_idx = threadIdx.x / WARP_SIZE;
  int const num_groups_per_block = blockDim.x / WARP_SIZE;

  for (int row = 0; row < NUM_ROWS; ++row) {
    if (row >= num_active_rows) {
      break;
    }
    long const in_base = (long)row * I_STRIDE;
    long const out_base = (long)row * O_STRIDE;

    for (int group_idx = warp_idx; group_idx < NUM_GROUPS_PER_ROW;
         group_idx += num_groups_per_block) {
      int const group_base = GROUP_SIZE * group_idx;

      // ---- SwiGLU, in T, at HEAD's cast positions -----------------------
      T act[ELEMENTS_PER_THREAD];
      float local_max = eps;
#pragma unroll
      for (int ele_idx = 0; ele_idx < ELEMENTS_PER_THREAD; ++ele_idx) {
        int const off = group_base + lane_idx + ele_idx * WARP_SIZE;
        float const input_val = float(input[in_base + off]);
        T const mul_val = input[in_base + OUTPUT_SIZE + off];
        act[ele_idx] = T(input_val / (1.0f + expf(-input_val))) * mul_val;
        float const abs_val = fabsf(float(act[ele_idx]));
        local_max = fmaxf(abs_val, local_max);
      }

      // ---- per-group fp32 block scale (quantize_fp8, f32-scale branch) ---
      float group_max = group_reduce_max<WARP_SIZE>(local_max);
      group_max = fmaxf(group_max, 1e-10f);
      float const y_scale = group_max / max_8bit;
      if (lane_idx == 0) {
        output_s[row * S_STRIDE + group_idx] = y_scale;
      }

#pragma unroll
      for (int ele_idx = 0; ele_idx < ELEMENTS_PER_THREAD; ++ele_idx) {
        int const off = group_base + lane_idx + ele_idx * WARP_SIZE;
        float const orig_val = float(act[ele_idx]);
        float const quant_val =
            fminf(fmaxf(orig_val / y_scale, min_8bit), max_8bit);
        output_q[out_base + off] = DST_T(quant_val);
      }
    }
  }
}

} // namespace kernel
