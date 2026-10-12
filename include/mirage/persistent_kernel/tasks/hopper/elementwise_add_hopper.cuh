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
#include "tasks/common/common_header.cuh"

// Element-wise addition for Hopper: output = input_a + input_b.
//
// Same contract as elementwise_add_task_impl (blackwell/elementwise_add_sm100):
// each task covers a [BATCH_SIZE, OUTPUT_SIZE] tile of row-major tensors whose
// row stride is STRIDE. Rows are moved as 16-byte vectors by all
// WORKER_NUM_THREADS worker threads, with several vectors in flight per
// thread. The sum is formed in fp32 and rounded once.
//
// The vector path needs OUTPUT_SIZE and STRIDE to be multiples of the vector
// width and the three pointers to be 16-byte aligned; anything else takes the
// scalar path.

namespace kernel {

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int STRIDE>
__device__ __forceinline__ void elementwise_add_hopper_task_impl(
    void const *input_a_ptr, void const *input_b_ptr, void *output_ptr) {
  T const *__restrict__ a = static_cast<T const *>(input_a_ptr);
  T const *__restrict__ b = static_cast<T const *>(input_b_ptr);
  T *__restrict__ out = static_cast<T *>(output_ptr);

  constexpr int VEC = 16 / sizeof(T);
  constexpr int TOTAL = BATCH_SIZE * OUTPUT_SIZE;
  constexpr bool SHAPE_ALLOWS_VEC = OUTPUT_SIZE % VEC == 0 && STRIDE % VEC == 0;
  int const tid = threadIdx.x;

  bool const aligned =
      ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) |
        reinterpret_cast<uintptr_t>(out)) &
       15) == 0;

  if constexpr (SHAPE_ALLOWS_VEC) {
    if (aligned) {
      constexpr int VECS_PER_ROW = OUTPUT_SIZE / VEC;
      constexpr int TOTAL_VECS = BATCH_SIZE * VECS_PER_ROW;
      // Vectors issued per thread before the first add, so the loads of a
      // decode-sized tile (a few KB) are all outstanding at once.
      constexpr int UNROLL = 4;
      union Pack {
        uint4 u;
        T e[VEC];
      };
      for (int base = 0; base < TOTAL_VECS;
           base += UNROLL * WORKER_NUM_THREADS) {
        Pack va[UNROLL], vb[UNROLL];
        int offs[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
          int const v = base + u * WORKER_NUM_THREADS + tid;
          int const row = v / VECS_PER_ROW;
          int const col = (v % VECS_PER_ROW) * VEC;
          offs[u] = row * STRIDE + col;
          if (v < TOTAL_VECS) {
            va[u].u = *reinterpret_cast<uint4 const *>(a + offs[u]);
            vb[u].u = *reinterpret_cast<uint4 const *>(b + offs[u]);
          }
        }
#pragma unroll
        for (int u = 0; u < UNROLL; u++) {
          int const v = base + u * WORKER_NUM_THREADS + tid;
          if (v < TOTAL_VECS) {
            Pack vo;
#pragma unroll
            for (int e = 0; e < VEC; e++) {
              vo.e[e] = T(float(va[u].e[e]) + float(vb[u].e[e]));
            }
            *reinterpret_cast<uint4 *>(out + offs[u]) = vo.u;
          }
        }
      }
      return;
    }
  }

  for (int i = tid; i < TOTAL; i += WORKER_NUM_THREADS) {
    int const row = i / OUTPUT_SIZE;
    int const col = i % OUTPUT_SIZE;
    int const idx = row * STRIDE + col;
    out[idx] = T(float(a[idx]) + float(b[idx]));
  }
}

} // namespace kernel
