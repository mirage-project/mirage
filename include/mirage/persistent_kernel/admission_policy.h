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

// MODE_OFFLINE admission arithmetic (M3-I9).
//
// `prepare_next_batch` fills `MPK_MAX_NUM_BATCHED_TOKENS` greedily in slot
// order, giving each prefilling request `min(remaining, budget_left)` -- i.e.
// the whole remaining budget.  A request that finishes prefill then takes one
// token per iteration forever, so the j-th slot to finish prefill only ever
// gets `mbt - j` tokens: 569 prompt tokens that fit in 36 iterations take 108
// at bs16 (M3-I1 backlog #4; measured replay in
// demo/qwen3_5/accept/opt/m3i9/).
//
// MPK_MAX_TOKENS_PER_REQUEST caps what ONE request may take from ONE
// iteration.  Setting it to `mbt / max_num_batched_requests` gives every
// request an equal share, which is simultaneously
//   - the packing optimum (203 -> 131 iterations at bs16), and
//   - the only setting under which no live slot is ever migrated by the step-3
//     compaction, which matters because the GDN conv/recurrent state pools are
//     slot-indexed and are NOT migrated with the request (HAZARD-COMPACTION).
//
// DEFAULT IS A NO-OP, BY CONSTRUCTION.  The default is
// MPK_MAX_NUM_BATCHED_TOKENS, and both call sites pass
// `budget_left = MPK_MAX_NUM_BATCHED_TOKENS - num_tokens` with `num_tokens >=
// 0` (it starts at 0 and only ever increases by non-negative `num_new_tokens`).
// So `budget_left <= MPK_MAX_NUM_BATCHED_TOKENS == cap`, hence
// `min(remaining, budget_left) <= cap` and the extra clamp is the identity for
// every reachable state.  Codegen is expected to be byte-identical with the
// macro unset.
//
// The DECODE branch is deliberately not capped: `min(1, budget_left)` is
// causality (token n+1 depends on token n), not a scheduling choice, and it is
// already <= any cap >= 1.

#pragma once

#ifndef MPK_MAX_TOKENS_PER_REQUEST
#define MPK_MAX_TOKENS_PER_REQUEST MPK_MAX_NUM_BATCHED_TOKENS
#endif

#if defined(__CUDACC__)
#define MPK_ADMISSION_FN __host__ __device__ __forceinline__
#else
#define MPK_ADMISSION_FN inline
#endif

namespace mirage {
namespace mpk {

// Tokens a PREFILLING request contributes to this iteration.
//   remaining   = prompt_length - step   (> 0 for a prefilling request)
//   budget_left = MPK_MAX_NUM_BATCHED_TOKENS - num_tokens   (>= 0)
//   cap         = MPK_MAX_TOKENS_PER_REQUEST                (>= 1)
// Never negative: `budget_left >= 0` and `cap >= 1`, so the result is in
// [0, min(remaining, budget_left)].
MPK_ADMISSION_FN int
    admission_prefill_tokens(int remaining, int budget_left, int cap) {
  int k = (remaining < budget_left) ? remaining : budget_left;
  return (k < cap) ? k : cap;
}

} // namespace mpk
} // namespace mirage

#ifdef MPK_MAX_NUM_BATCHED_TOKENS
static_assert(MPK_MAX_TOKENS_PER_REQUEST >= 1,
              "MPK_MAX_TOKENS_PER_REQUEST must be >= 1: a cap of 0 would stall "
              "every prefill and the scheduler would return false on the first "
              "iteration");
static_assert(MPK_MAX_TOKENS_PER_REQUEST <= MPK_MAX_NUM_BATCHED_TOKENS,
              "MPK_MAX_TOKENS_PER_REQUEST above MPK_MAX_NUM_BATCHED_TOKENS is "
              "meaningless: the per-iteration budget already bounds it");
#endif
