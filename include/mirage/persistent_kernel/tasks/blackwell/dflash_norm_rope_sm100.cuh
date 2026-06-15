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
// DFlash per-head RMSNorm + RoPE (correctness-first).
//
// Applies q_norm/k_norm (RMSNorm over HEAD_DIM, fp32 internal) then NeoX RoPE
// to each (token, head). Used to prepare q and k (both context and block) for
// the non-causal attention. v skips this (raw).
//
//   x   : [N, NUM_HEADS, HEAD_DIM]  (flattened [N, NUM_HEADS*HEAD_DIM])
//   weight : [HEAD_DIM]             (shared across heads)
//   cos/sin: [N, HEAD_DIM]          (per token; YaRN-baked, NeoX duplicated
//   form) out : [N, NUM_HEADS, HEAD_DIM]
//
// One thread handles one (token, head): full HEAD_DIM norm + rope, no
// cross-lane.
// ============================================================================
template <typename T, int NUM_HEADS, int HEAD_DIM>
__device__ __forceinline__ void dflash_norm_rope_sm100(void const *x_ptr,
                                                       void const *weight_ptr,
                                                       void const *cos_ptr,
                                                       void const *sin_ptr,
                                                       void *out_ptr,
                                                       int num_tokens,
                                                       float eps) {
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }
  constexpr int HALF = HEAD_DIM / 2;
  T const *x = static_cast<T const *>(x_ptr);
  T const *weight = static_cast<T const *>(weight_ptr);
  T const *cos = static_cast<T const *>(cos_ptr);
  T const *sin = static_cast<T const *>(sin_ptr);
  T *out = static_cast<T *>(out_ptr);

  int const total = num_tokens * NUM_HEADS;
  for (int p = threadIdx.x; p < total; p += NUM_THREADS) {
    int const tok = p / NUM_HEADS;
    int const head = p % NUM_HEADS;
    T const *xr = x + (size_t)(tok * NUM_HEADS + head) * HEAD_DIM;
    T const *cr = cos + (size_t)tok * HEAD_DIM;
    T const *sr = sin + (size_t)tok * HEAD_DIM;
    T *orow = out + (size_t)(tok * NUM_HEADS + head) * HEAD_DIM;

    // RMSNorm over HEAD_DIM (fp32)
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < HEAD_DIM; ++i) {
      float v = static_cast<float>(xr[i]);
      ss += v * v;
    }
    float rms = rsqrtf(ss / static_cast<float>(HEAD_DIM) + eps);

    // normalized value buffer
    float nv[HEAD_DIM];
#pragma unroll
    for (int i = 0; i < HEAD_DIM; ++i) {
      nv[i] = static_cast<float>(xr[i]) * rms * static_cast<float>(weight[i]);
    }

    // NeoX RoPE: out[i] = nv[i]*cos[i] - nv[i+HALF]*sin[i]  (i < HALF)
    //            out[i] = nv[i]*cos[i] + nv[i-HALF]*sin[i]  (i >= HALF)
#pragma unroll
    for (int i = 0; i < HEAD_DIM; ++i) {
      float c = static_cast<float>(cr[i]);
      float s = static_cast<float>(sr[i]);
      float rot = (i < HALF) ? -nv[i + HALF] : nv[i - HALF];
      orow[i] = static_cast<T>(nv[i] * c + rot * s);
    }
  }
}

} // namespace kernel
