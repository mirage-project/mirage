// tasks/sm90_rmsnorm.cuh -- the sm90_rmsnorm task type: RMSNorm of one token,
// y = bf16 or f16(x / rms(x) * weight) in fp32 (as GPT-OSS's RMSNorm), x bf16;
// an f16 y feeds the f16 MoE (sm90_moe_ring). Task x of grid (T, 1, 1): row x.
#pragma once
#include "sm90_common.cuh"

namespace static_mk {

// PARAMS {H, eps (fp32 bits), F16 (1: the output f16)}; BUF_SLOTS {input
// [T, H] bf16, weight [H] bf16, output [T, H]}
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
  constexpr bool F16 = PARAMS::v[2] == 1;
  using Row = Sm90Row<H>;
  __shared__ float part[SM90_WARPS];
  sm90_task_begin<IN...>(g);
  uint4 const *x =
      buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 0>()) + (size_t)tk.x * Row::VEC;
  uint4 const *w = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 1>());
  uint4 *out =
      buf_at<uint4>(g, slot_at<BUF_SLOTS, 2>()) + (size_t)tk.x * Row::VEC;
  float xf[Row::PER][8];
  uint4 wv[Row::PER];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < Row::PER; j++) {
    int const v = Row::vec(j);
    if (v < Row::VEC) {
      bf16x8_to_f32(x[v], xf[j]);
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
      out[v] = F16 ? f32_to_f16x8(xf[j]) : f32_to_bf16x8(xf[j]);
    }
  }
  sm90_task_end<SELF>(g);
}

SM90_TASK_SMEM(sm90_rmsnorm, 0)

} // namespace static_mk
