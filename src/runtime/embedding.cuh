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
namespace kernel {

template <typename T>
__device__ __forceinline__ void
    embedding_kernel(void const *__restrict__ input_ptr,
                     void const *__restrict__ embedding_ptr,
                     void *__restrict__ output_ptr) {
  const __restrict__ uint16_t *input_ids =
      static_cast<uint16_t const *>(input_ptr);
  const __restrict__ T *embedding = static_cast<T const *>(embedding_ptr);
  T __restrict__ *output = static_cast<T *>(output_ptr);
  constexpr int BATCH_SIZE = 1;
  constexpr int OUT_DIM = 3584;

  for (int i = threadIdx.x; i < BATCH_SIZE * OUT_DIM;
       i += blockDim.x * gridDim.x) {
    int idx = i / OUT_DIM;
    int off = i % OUT_DIM;
    uint16_t wordIdx = input_ids[idx];
    output[i] = embedding[wordIdx * OUT_DIM + off];
  }
}

} // namespace kernel
