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

#include <cuda_bf16.h>

namespace kernel {

// bs=1 contiguous KV append: writes the new token rows' [c_latent(D_V) |
// k_pe(D_K-D_V)] into the per-layer contiguous KV buffer at rows
// [row_start, row_start + num_new_tokens). Replaces the paged-cache append +
// page gather: with a single sequence the logical position IS the physical
// row, so no page table is needed. The decode kernels read this buffer via
// their contiguous branch (page_indices == nullptr).
//
// c_latent_new / k_pe_new may be narrow views of a wider row (qkv_a_out);
// the row strides carry the parent width.
template <int D_K, // total KV dim (576 = 512 latent + 64 rope)
          int D_V, // latent dim (512)
          int K_PE_ROW_STRIDE = D_K - D_V,
          int C_LATENT_ROW_STRIDE = D_V>
__device__ __forceinline__ void
    mla_kv_append_sm100_task_impl(void const *c_latent_new_ptr,
                                  void const *k_pe_new_ptr,
                                  void *kv_buf_ptr,
                                  int row_start,
                                  int num_new_tokens) {
  using T = __nv_bfloat16;
  constexpr int NUM_THREADS = 128;
  constexpr int ROPE_DIM = D_K - D_V;
  int const tid = threadIdx.x;

  if (num_new_tokens <= 0 || row_start < 0) {
    return;
  }

  T const *c_latent_new = reinterpret_cast<T const *>(c_latent_new_ptr);
  T const *k_pe_new = reinterpret_cast<T const *>(k_pe_new_ptr);
  T *kv_buf = reinterpret_cast<T *>(kv_buf_ptr);

  for (int tok = 0; tok < num_new_tokens; tok++) {
    T *dst = kv_buf + (size_t)(row_start + tok) * D_K;
    T const *src_lat = c_latent_new + (size_t)tok * C_LATENT_ROW_STRIDE;
    T const *src_pe = k_pe_new + (size_t)tok * K_PE_ROW_STRIDE;
    // c_latent: D_V bf16, vectorized 8 bf16 per uint4
    for (int d = tid * 8; d + 8 <= D_V; d += NUM_THREADS * 8) {
      *reinterpret_cast<uint4 *>(dst + d) =
          *reinterpret_cast<uint4 const *>(src_lat + d);
    }
    // k_pe: ROPE_DIM bf16
    for (int d = tid * 8; d + 8 <= ROPE_DIM; d += NUM_THREADS * 8) {
      *reinterpret_cast<uint4 *>(dst + D_V + d) =
          *reinterpret_cast<uint4 const *>(src_pe + d);
    }
  }
}

} // namespace kernel
