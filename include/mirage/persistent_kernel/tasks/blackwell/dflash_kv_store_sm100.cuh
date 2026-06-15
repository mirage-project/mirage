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

namespace kernel {

// ============================================================================
// DFlash standalone paged KV-cache store (correctness-first).
//
// Writes already-projected/normed/roped K (or V) for a set of tokens into the
// draft's OWN paged KV cache at arbitrary slots. Used by L4 materialization to
// write committed context K/V (target-derived), OVERWRITING any temporary
// block-K/V the draft attention wrote at those slots in a prior decode step.
//
//   kv_in        : [num_tokens, NKV, D]            (flattened [num_tokens,
//   NKV*D]) slot_mapping : [num_tokens] int32             (absolute slot per
//   token) cache (out)  : [num_pages, page_size, NKV, D]    bf16, NHD
//
// Paged layout: slot s -> page s/page_size, offset s%page_size. Since pages are
// physically contiguous (draft cache, sequential allocation), the flat index of
// slot s is simply s*NKV*D + (kv_head*D + dim). Plain store == overwrite.
//
// One task handles all num_tokens (scatter by slot); thread p owns element
// (tok, kv_head, dim). blockIdx-agnostic.
// ============================================================================
template <typename T, int NKV, int D>
__device__ __forceinline__ void
    dflash_kv_store_sm100(void const *kv_in_ptr,
                          void const *slot_mapping_ptr,
                          void *cache_ptr,
                          int num_tokens,
                          int page_size) {
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }
  constexpr int KVD = NKV * D;
  T const *kv_in = static_cast<T const *>(kv_in_ptr);
  int const *slot = static_cast<int const *>(slot_mapping_ptr);
  T *cache = static_cast<T *>(cache_ptr);

  int const total = num_tokens * KVD;
  for (int p = threadIdx.x; p < total; p += NUM_THREADS) {
    int const tok = p / KVD;
    int const rem = p % KVD; // kv_head * D + dim
    int const s = slot[tok];
    int const page = s / page_size;
    int const off = s % page_size;
    size_t const cidx = (size_t)(page * page_size + off) * KVD + rem;
    cache[cidx] = kv_in[(size_t)tok * KVD + rem];
  }
}

} // namespace kernel
