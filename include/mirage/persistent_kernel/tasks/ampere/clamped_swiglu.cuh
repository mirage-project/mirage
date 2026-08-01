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

// GPT-OSS's gated activation: both halves are clamped, the gate is gated by a
// scaled sigmoid, and the up half is offset by one.
//
//   gate = min(gate, limit)
//   up   = clamp(up, -limit, limit)
//   out  = (up + 1) * gate * sigmoid(gate * alpha)
//
// The two halves are laid out like silu_mul's: gate first, up OUTPUT_SIZE
// elements later. GPT-OSS ships them interleaved instead, which the weight
// loader de-interleaves once so this stays a plain strided read.
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int I_STRIDE,
          int O_STRIDE>
__device__ __forceinline__ void
    clamped_swiglu_task_impl(void const *input_ptr,
                             void *output_ptr,
                             float limit,
                             float alpha,
                             int num_active_tokens) {
  T const *__restrict__ d_gate = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_up = static_cast<T const *>(input_ptr) + OUTPUT_SIZE;
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

#pragma unroll
  for (int i = threadIdx.x; i < num_active_tokens * OUTPUT_SIZE;
       i += blockDim.x) {
    int batch_idx = i / OUTPUT_SIZE;
    int offset = i % OUTPUT_SIZE;
    float gate = float(d_gate[batch_idx * I_STRIDE + offset]);
    float up = float(d_up[batch_idx * I_STRIDE + offset]);
    gate = fminf(gate, limit);
    up = fminf(fmaxf(up, -limit), limit);
    float glu = gate / (1.0f + expf(-gate * alpha));
    d_output[batch_idx * O_STRIDE + offset] = T((up + 1.0f) * glu);
  }
}

} // namespace kernel
