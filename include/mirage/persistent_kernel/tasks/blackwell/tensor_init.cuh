/* Copyright 2025 Mirage Team
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
#include "tasks/common/common_header.cuh"
namespace kernel {

// Vectorized zero-fill for the splitk-linear output tile.
//
// Caller wires this as a (1 input, 2 outputs) MPK task whose grid_dim and
// per-tensor input_maps mirror the downstream splitk linear:
//   * output_ptrs[0] -> the linear's output buffer (this kernel zeroes it)
//   * output_ptrs[1] -> the linear's input (untouched; pure dep edge)
//   * input_ptrs[0]  -> the linear's input (untouched; pure dep edge)
//
// Stores are 16-byte (int4) — alignment is guaranteed by the splitk linear's
// own constraint that per-task OUTPUT_SIZE and full OUTPUT_STRIDE are
// multiples of 128 bf16 (= 256 bytes).
template <int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__device__ __forceinline__ void
    tensor_init_zero_sm100_task_impl(void *target_ptr) {
  using bf16 = cute::bfloat16_t;
  static_assert(OUTPUT_SIZE % 8 == 0,
                "tensor_init: OUTPUT_SIZE must be multiple of 8 (16B vec)");
  bf16 *base = static_cast<bf16 *>(target_ptr);
  constexpr int VEC = 8; // 8 bf16 = 16 bytes
  constexpr int VEC_PER_ROW = OUTPUT_SIZE / VEC;
  int4 const zero = {0, 0, 0, 0};
#pragma unroll
  for (int row = 0; row < BATCH_SIZE; ++row) {
    int4 *row_ptr = reinterpret_cast<int4 *>(base + row * OUTPUT_STRIDE);
    for (int i = threadIdx.x; i < VEC_PER_ROW; i += blockDim.x) {
      row_ptr[i] = zero;
    }
  }
} // tensor_init_zero_sm100_task_impl

// GATE-ONLY (poison-fill correctness gate for the skip-after-step0 lever):
// byte-fill the buffer with 0xff (=> NaN for fp32/bf16/fp16, toxic for fp8/int)
// STARTING at POISON_OFFSET_BYTES (spare the barrier head [0,16)), leaving the
// self-maintaining grid-barrier counters intact. Used INSTEAD of the step0 zero
// on decode steps>=1 to prove every DATA region is overwritten-before-read: if
// the buffer is truly self-maintaining, the poison is destroyed before use and
// decode stays coherent; a read-before-write propagates NaN -> collapse.
// TOTAL_BYTES / POISON_OFFSET_BYTES are byte counts; base is bf16-typed only
// because the caller allocates the buffer as bf16 (bytes/2 elements).
template <int TOTAL_BYTES, int POISON_OFFSET_BYTES>
__device__ __forceinline__ void
    tensor_init_poison_sm100_task_impl(void *target_ptr) {
  static_assert(POISON_OFFSET_BYTES % 16 == 0,
                "poison offset must be 16B aligned");
  static_assert(TOTAL_BYTES % 16 == 0, "poison total must be 16B aligned");
  uint8_t *base = static_cast<uint8_t *>(target_ptr);
  int4 const poison = {
      (int)0xffffffff, (int)0xffffffff, (int)0xffffffff, (int)0xffffffff};
  constexpr int VEC0 = POISON_OFFSET_BYTES / 16;
  constexpr int VEC_TOTAL = TOTAL_BYTES / 16;
  int4 *vp = reinterpret_cast<int4 *>(base);
  for (int i = VEC0 + threadIdx.x; i < VEC_TOTAL; i += blockDim.x) {
    vp[i] = poison;
  }
} // tensor_init_poison_sm100_task_impl

} // namespace kernel
