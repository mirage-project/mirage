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

// tasks/sm90_rmsnorm.cuh -- the sm90_rmsnorm task type: RMSNorm of one token,
// y = bf16(x / rms(x) * weight) in fp32 (as GPT-OSS's RMSNorm), x bf16. Task x
// of grid (T, 1, 1): row x.
#pragma once
#include "sm90_common.cuh"

namespace static_mk {

// PARAMS {H, eps (fp32 bits)[, 1: add]}; BUF_SLOTS {input [T, H] bf16, weight
// [H] bf16, output [T, H] bf16}; with add: {input, add [T, H] bf16, weight,
// output, sum [T, H] bf16}: the row is bf16(input + add) (fp32, rounded once),
// written to sum (a residual stream: the previous layer's x1 and its experts)
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_rmsnorm(Maps const &,
                                                 G const &g,
                                                 KernelLocals &,
                                                 StaticTask const &tk) {
  constexpr int H = PARAMS::v[0];
  constexpr bool ADD = param_or<PARAMS, 2, 0>() != 0;
  constexpr int A = ADD ? 1 : 0; // the slots after input shift by one
  using Row = Sm90Row<H>;
  __shared__ float part[SM90_WARPS];
  sm90_task_begin<IN...>(g);
  size_t const row = (size_t)tk.x * Row::VEC;
  uint4 const *x = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 0>()) + row;
  uint4 const *w = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 1 + A>());
  uint4 *out = buf_at<uint4>(g, slot_at<BUF_SLOTS, 2 + A>()) + row;
  float xf[Row::PER][8];
  uint4 wv[Row::PER];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < Row::PER; j++) {
    int const v = Row::vec(j);
    if (v < Row::VEC) {
      bf16x8_to_f32(x[v], xf[j]);
      if constexpr (ADD) {
        float a[8];
        bf16x8_to_f32(
            (buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 1>()) + row)[v], a);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          xf[j][i] += a[i];
        }
        uint4 const q = f32_to_bf16x8(xf[j]);
        (buf_at<uint4>(g, slot_at<BUF_SLOTS, 4>()) + row)[v] = q;
        bf16x8_to_f32(q, xf[j]);
      }
      wv[j] = w[v];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        ss += xf[j][i] * xf[j][i];
      }
    }
  }
  float const rs =
      rsqrtf(sm90_block_sum(ss, part) / H + __int_as_float(PARAMS::v[1]));
#pragma unroll
  for (int j = 0; j < Row::PER; j++) {
    int const v = Row::vec(j);
    if (v < Row::VEC) {
      float gw[8];
      bf16x8_to_f32(wv[j], gw);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        xf[j][i] = xf[j][i] * rs * gw[i];
      }
      out[v] = f32_to_bf16x8(xf[j]);
    }
  }
  sm90_task_end<SELF>(g);
}

SM90_TASK_SMEM(sm90_rmsnorm, 0)

} // namespace static_mk
