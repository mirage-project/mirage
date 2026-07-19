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
#include "../common/utils.cuh"
#include "../common/worker_config.h"
namespace kernel {

// ============================================================================
// transpose_scale_sm100
// ============================================================================
//
// Tiny helper task that transposes a 2D packed UE8M0 uint32 scale buffer
// from `(M, K_PACKED)` row-major (the format that
// per_token_group_quantize_fp8 produces) to `(K_PACKED, M)` row-major (the
// format the PR-674 fp8_group_gemm_smallm/largem_sm100 SFA/SFB TMA
// descriptors expect).
//
// Multi-CTA fan-out: the single-CTA version was 53 μs
// for the DSv3 NEW MoE silu_scale shape (M=16384, K_PACKED=2 → 32K
// uint32 = 128 KB transfer). Split the M dimension across grid.x CTAs
// — each CTA owns a contiguous chunk of M rows and does its own
// transpose. No cross-CTA sync required (disjoint writes).
//
// Wrapper now picks grid_dim=(min(num_workers, M/16), 1, 1) so a
// single wave handles the full transpose.
//

template <int M, int K_PACKED>
__device__ __forceinline__ void transpose_scale_sm100_task_impl(
    void const *in_ptr, void *out_ptr, int cta_idx, int num_ctas) {
  uint32_t const *__restrict__ in = static_cast<uint32_t const *>(in_ptr);
  uint32_t *__restrict__ out = static_cast<uint32_t *>(out_ptr);
  int const tid = threadIdx.x;
  int const nthreads = blockDim.x;
  // Stripe M across grid.x CTAs. Each CTA processes M_per_cta rows.
  int const m_lo = (M * cta_idx) / num_ctas;
  int const m_hi = (M * (cta_idx + 1)) / num_ctas;
  for (int m = m_lo; m < m_hi; ++m) {
    for (int k = tid; k < K_PACKED; k += nthreads) {
      out[(size_t)k * M + m] = in[(size_t)m * K_PACKED + k];
    }
  }
}

} // namespace kernel
