/* Copyright 2026 CMU
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

// mHC K2: per-token affine + split + activation.
//
// Input mixes layout per token: [pre(n) | post(n) | res_logits(n*n)],
// total mix_hc = n*n + 2*n elements.
//
// scale[0]=alpha_pre, scale[1]=alpha_post, scale[2]=alpha_res.
//
// Per-element: y = mix * alpha[region] + bias.
//   pre:  H_pre  = sigmoid(y)
//   post: H_post = 2 * sigmoid(y)
//   res:  H_res_logits = y          (sinkhorn applied downstream)

namespace kernel {

template <typename T_in, int BATCH_SIZE, int N>
__device__ __forceinline__ void
    affine_split_activation_sm100_task_impl(void const *mixes_ptr,
                                            void const *scale_ptr,
                                            void const *base_ptr,
                                            void *h_pre_ptr,
                                            void *h_post_ptr,
                                            void *h_res_logits_ptr) {
  constexpr int MIX_HC = N * N + 2 * N;

  T_in const *__restrict__ d_mixes = static_cast<T_in const *>(mixes_ptr);
  float const *__restrict__ d_scale = static_cast<float const *>(scale_ptr);
  float const *__restrict__ d_base = static_cast<float const *>(base_ptr);
  float *__restrict__ d_h_pre = static_cast<float *>(h_pre_ptr);
  float *__restrict__ d_h_post = static_cast<float *>(h_post_ptr);
  float *__restrict__ d_h_res_logits = static_cast<float *>(h_res_logits_ptr);

  // Cache scale[3] in registers across all threads.
  float const alpha_pre = d_scale[0];
  float const alpha_post = d_scale[1];
  float const alpha_res = d_scale[2];

  for (int row = 0; row < BATCH_SIZE; ++row) {
    for (int j = threadIdx.x; j < MIX_HC; j += blockDim.x) {
      float mix = static_cast<float>(d_mixes[row * MIX_HC + j]);
      float bias = d_base[j];

      float alpha;
      int region;  // 0 = pre, 1 = post, 2 = res
      int local;   // index within region
      if (j < N) {
        alpha = alpha_pre;
        region = 0;
        local = j;
      } else if (j < 2 * N) {
        alpha = alpha_post;
        region = 1;
        local = j - N;
      } else {
        alpha = alpha_res;
        region = 2;
        local = j - 2 * N;
      }

      float y = mix * alpha + bias;

      if (region == 0) {
        // sigmoid(y)
        d_h_pre[row * N + local] = 1.0f / (1.0f + __expf(-y));
      } else if (region == 1) {
        // 2 * sigmoid(y)
        d_h_post[row * N + local] = 2.0f / (1.0f + __expf(-y));
      } else {
        // identity (sinkhorn applied downstream)
        d_h_res_logits[row * (N * N) + local] = y;
      }
    }
  }
}

} // namespace kernel
