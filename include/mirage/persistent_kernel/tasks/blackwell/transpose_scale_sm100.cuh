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
// One CTA only (grid_dim = (1, 1, 1)). Cooperative thread copy. Buffer is
// small (~32-100 KB for DSv3 MoE shapes) so single-CTA bandwidth is fine
// here; if perf becomes an issue, fan out to grid_dim=(K_PACKED, 1, 1) and
// have each CTA handle one column.
//

template <int M, int K_PACKED>
__device__ __forceinline__ void
    transpose_scale_sm100_task_impl(void const *in_ptr, void *out_ptr) {
  uint32_t const *__restrict__ in = static_cast<uint32_t const *>(in_ptr);
  uint32_t *__restrict__ out = static_cast<uint32_t *>(out_ptr);
  int const tid = threadIdx.x;
  int const nthreads = blockDim.x;
  int const total = M * K_PACKED;
  for (int i = tid; i < total; i += nthreads) {
    int m = i / K_PACKED;
    int k = i % K_PACKED;
    out[(size_t)k * M + m] = in[(size_t)m * K_PACKED + k];
  }
}

} // namespace kernel
