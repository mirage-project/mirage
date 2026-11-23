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

// tasks/gpt_oss_moe_norm.cuh -- the gpt_oss_moe_norm task type (Blackwell
// GPT-OSS with FlashInfer's trtllm-gen MoE): the previous layer's MoE finalize,
// the residual add and the next RMSNorm of one token, in the layer kernel
// instead of FlashInfer's finalizeKernel (one kernel and its boundary less a
// layer). Task x of grid (T, 1, 1): row x.
//
// FlashInfer ran with do_finalize=False and left gemm2_output [rows, HP] bf16
// (each routed (token, expert) pair's expert output, permuted), the experts'
// weights [R, TOPK] bf16 and expanded_idx_to_permuted_idx [R TOPK] i32 (-1: a
// slot not computed). The host writes their addresses into a table, int64 [4]
// {gemm2 output, permuted idx, weights, R}, after the MoE call (stream order:
// before this kernel's griddepcontrol.wait in kernel_begin; read through L2).
//
//   y   = row t < R: bf16(sum over k = 0..TOPK-1 with p = idx[t TOPK + k] != -1
//         of w[t TOPK + k] gemm2[p][h]), fp32 fma.rn.ftz in that order from +0;
//         row t >= R: 0
//         (FlashInfer 0.7.2's finalizeKernel, JIT-built with -use_fast_math,
//         bit for bit: the moe output the unfused path leaves in moe_y)
//   sum = bf16(input + y)  (fp32, rounded once; the residual stream x_res)
//   out = bf16(sum / rms(sum) weight)  (fp32, as sm90_rmsnorm)
//
// PARAMS {H, eps (fp32 bits), TOPK, HP}; BUF_SLOTS {input [T, H] bf16, table
// int64 [4], weight [H] bf16, output [T, H] bf16, sum [T, H] bf16}.
#pragma once
#include "sm90_common.cuh"

namespace static_mk {

// finalizeKernel's accumulate (`data += float{w} * float{x}`, contracted and
// flushed to zero under -use_fast_math)
__device__ __forceinline__ float gpt_oss_fma_rn_ftz(float a, float b, float c) {
  float d;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_gpt_oss_moe_norm(Maps const &,
                                                     G const &g,
                                                     KernelLocals &,
                                                     StaticTask const &tk) {
  constexpr int H = PARAMS::v[0];
  constexpr int TOPK = PARAMS::v[2];
  constexpr int HP = PARAMS::v[3];
  static_assert(H % 8 == 0 && TOPK >= 1 && TOPK <= 8 && HP >= H && HP % 8 == 0,
                "gpt_oss_moe_norm: 1..8 experts a token, gemm2 rows of HP >= H "
                "bf16 (16-B vectors)");
  using Row = Sm90Row<H>;
  __shared__ float part[SM90_WARPS];
  sm90_task_begin<IN...>(g);
  size_t const row = (size_t)tk.x * Row::VEC;
  uint4 const *x = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 0>()) + row;
  uint4 const *w = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 2>());
  uint4 *out = buf_at<uint4>(g, slot_at<BUF_SLOTS, 3>()) + row;
  uint4 *sum = buf_at<uint4>(g, slot_at<BUF_SLOTS, 4>()) + row;
  // this token's experts (the same in every thread)
  longlong2 const *tab = buf_at<longlong2 const>(g, slot_at<BUF_SLOTS, 1>());
  longlong2 const t0 = __ldcg(tab), t1 = __ldcg(tab + 1);
  bool const have = (long long)tk.x < t1.y;
  uint4 const *g2 = reinterpret_cast<uint4 const *>(t0.x);
  int perm[TOPK];
  float wk[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; k++) {
    perm[k] = -1;
    wk[k] = 0.f;
  }
  if (have) {
    int const *pi = reinterpret_cast<int const *>(t0.y) + tk.x * TOPK;
    __nv_bfloat16 const *wp =
        reinterpret_cast<__nv_bfloat16 const *>(t1.x) + tk.x * TOPK;
#pragma unroll
    for (int k = 0; k < TOPK; k++) {
      perm[k] = __ldcg(pi + k);
      wk[k] = __bfloat162float(__ldcg(wp + k));
    }
  }
  float xf[Row::PER][8];
  uint4 wv[Row::PER];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < Row::PER; j++) {
    int const v = Row::vec(j);
    if (v < Row::VEC) {
      uint4 r[TOPK];
#pragma unroll
      for (int k = 0; k < TOPK; k++) { // every expert row's loads in flight
        if (perm[k] != -1) {
          r[k] = __ldcg(g2 + (size_t)perm[k] * (HP / 8) + v);
        }
      }
      bf16x8_to_f32(x[v], xf[j]);
      wv[j] = w[v];
      float acc[8];
#pragma unroll
      for (int i = 0; i < 8; i++) {
        acc[i] = 0.f;
      }
#pragma unroll
      for (int k = 0; k < TOPK; k++) {
        if (perm[k] != -1) {
          float e[8];
          bf16x8_to_f32(r[k], e);
#pragma unroll
          for (int i = 0; i < 8; i++) {
            acc[i] = gpt_oss_fma_rn_ftz(wk[k], e[i], acc[i]);
          }
        }
      }
      float a[8];
      bf16x8_to_f32(f32_to_bf16x8(acc), a); // finalize's bf16 output
#pragma unroll
      for (int i = 0; i < 8; i++) {
        xf[j][i] += a[i];
      }
      uint4 const q = f32_to_bf16x8(xf[j]);
      sum[v] = q;
      bf16x8_to_f32(q, xf[j]);
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

SM90_TASK_SMEM(gpt_oss_moe_norm, 0)

} // namespace static_mk
