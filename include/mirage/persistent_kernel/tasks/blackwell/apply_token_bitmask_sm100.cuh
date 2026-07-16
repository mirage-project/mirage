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

#include <cstdint>
#include <math.h>

namespace kernel {

// System-scope acquire load, pairing with the CPU's release store on mask_seq.
static __device__ __forceinline__ int32_t
    apply_mask_ld_acquire_sys_i32(int32_t const volatile *addr) {
  int32_t v;
  asm volatile("ld.acquire.sys.b32 %0, [%1];" : "=r"(v) : "l"(addr));
  return v;
}

// Constrained decoding: copy [BATCH_SIZE, vocab] logits to out, then (when flag
// is set) mask each active request's next-token row to xgrammar's bitmask.
// Looping over requests keeps request_ids/qo_indptr (sized by request count) in
// bounds; request b's next-token logits are row qo_indptr[b+1]-1. The CPU
// publishes mask_seq[row]=step and the bitmask before that step is decoded.
template <typename T, int BATCH_SIZE>
__device__ __forceinline__ void apply_token_bitmask_sm100_kernel(
    void const *__restrict__ in_ptr,
    void *__restrict__ out_ptr,
    int32_t const *__restrict__ bitmask, // [total_inflight, bitmask_words]
    int32_t const volatile *__restrict__ mask_seq, // [total_inflight], pinned
    int32_t const volatile *__restrict__ flag,     // [1], pinned: 0=off
    int const *__restrict__ request_ids, // [num_requests], slot -> row
    int const *__restrict__ step,        // [total_inflight]
    int const *__restrict__ qo_indptr,   // [num_requests+1]
    int vocab_size,
    int bitmask_words,
    int num_requests) {
  T const *__restrict__ in = static_cast<T const *>(in_ptr);
  T *__restrict__ out = static_cast<T *>(out_ptr);
  T const neg_inf = static_cast<T>(-INFINITY);

  for (size_t i = threadIdx.x; i < static_cast<size_t>(BATCH_SIZE) * vocab_size;
       i += blockDim.x) {
    out[i] = in[i];
  }
  __syncthreads();

  if (apply_mask_ld_acquire_sys_i32(flag) == 0) {
    return; // unconstrained
  }

  for (int b = 0; b < num_requests; b++) {
    int row = request_ids[b];
    int pos = (row >= 0) ? qo_indptr[b + 1] - 1 : -1;
    if (pos < 0) {
      continue;
    }
    if (threadIdx.x == 0) {
      while (apply_mask_ld_acquire_sys_i32(&mask_seq[row]) < step[row]) {
        __nanosleep(20);
      }
    }
    __syncthreads();
    int32_t const *__restrict__ row_mask =
        bitmask + static_cast<size_t>(row) * bitmask_words;
    size_t const base = static_cast<size_t>(pos) * vocab_size;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
      if (((row_mask[v >> 5] >> (v & 31)) & 1) == 0) {
        out[base + v] = neg_inf;
      }
    }
    __syncthreads();
  }
}

} // namespace kernel
