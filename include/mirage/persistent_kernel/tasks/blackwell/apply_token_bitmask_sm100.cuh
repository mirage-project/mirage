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

// System-scope acquire load (pairs with the CPU's release store on mask_seq).
// Implemented inline so this header has no dependency on mpk_atoms.cuh include
// order.
static __device__ __forceinline__ int32_t
    apply_mask_ld_acquire_sys_i32(int32_t volatile const *addr) {
  int32_t v;
  asm volatile("ld.acquire.sys.b32 %0, [%1];" : "=r"(v) : "l"(addr));
  return v;
}

// Constrained-decoding logit masking (xgrammar).
//
// in_ptr     : [batch_size, vocab_size], bf16, the raw logits (read-only).
// out_ptr    : [batch_size, vocab_size], bf16, the masked logits (written).
//              Distinct from in_ptr — downstream argmax/sampling reads out_ptr.
//              (A distinct output, rather than in-place, keeps the MPK
//              producer/consumer event wiring unambiguous: this task is the
//              sole producer of out_ptr.)
// bitmask    : [total_inflight, bitmask_words] packed int32, bit j set ⇒ token
//              j allowed.  Indexed by buffer row (not batch position).
// mask_seq   : [total_inflight] int32, pinned. The CPU publishes
//              mask_seq[row] = decode step the row's bitmask is valid for.
// flag       : [1] int32, pinned. 0 ⇒ unconstrained → plain copy (no wait, no
//              bit tests); 1 ⇒ wait for the CPU mask and apply it.
// request_ids: [batch_size] int, maps batch position → buffer row (-1 = idle).
// step       : [total_inflight] int, the row's current decode step (the step
//              whose logits we are about to sample). The task waits until the
//              CPU has published a mask for at least this step.
//
// Contract: when flag==1 the CPU MUST publish a valid bitmask + mask_seq for
// every active row each step (an all-ones mask for rows with no grammar).
template <typename T, int BATCH_SIZE>
__device__ __forceinline__ void apply_token_bitmask_sm100_kernel(
    void const *__restrict__ in_ptr,
    void *__restrict__ out_ptr,
    int32_t const *__restrict__ bitmask,
    int32_t volatile const *__restrict__ mask_seq,
    int32_t volatile const *__restrict__ flag,
    int const *__restrict__ request_ids,
    int const *__restrict__ step,
    int vocab_size,
    int bitmask_words,
    int num_active_tokens) {
  T const *__restrict__ in = static_cast<T const *>(in_ptr);
  T *__restrict__ out = static_cast<T *>(out_ptr);
  T const neg_inf = static_cast<T>(-INFINITY);

  // The masking task is present in every compiled graph; the runtime flag is
  // read once and toggles between a plain copy (unconstrained) and a masked
  // copy (constrained), so flipping it switches decoding modes with no
  // recompile.
  bool const constrained = (apply_mask_ld_acquire_sys_i32(flag) != 0);

  for (int b = 0; b < num_active_tokens; b++) {
    int row = request_ids[b];
    bool const do_mask = constrained && (row >= 0);
    int32_t const *__restrict__ row_mask = nullptr;

    if (do_mask) {
      // Wait until the CPU has produced this row's mask for the current step.
      // mask_seq increases monotonically (one mask per decode step, in lockstep
      // with token production), so ">= target" is the catch-up condition.
      if (threadIdx.x == 0) {
        int target = step[row];
        while (apply_mask_ld_acquire_sys_i32(&mask_seq[row]) < target) {
          __nanosleep(20);
        }
      }
      __syncthreads();
      row_mask = bitmask + static_cast<size_t>(row) * bitmask_words;
    }

    size_t const base = static_cast<size_t>(b) * vocab_size;
    for (int v = threadIdx.x; v < vocab_size; v += blockDim.x) {
      T val = in[base + v];
      if (do_mask && (((row_mask[v >> 5] >> (v & 31)) & 1) == 0)) {
        val = neg_inf; // disallowed token ⇒ zero probability downstream
      }
      out[base + v] = val;
    }
    __syncthreads();
  }
}

} // namespace kernel
