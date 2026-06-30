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

} // namespace kernel
