// tasks/sm90_mul_sum_add.cuh -- the sm90_mul_sum_add task type: the MoE
// combine (as MPK's mul_sum_add_sm100.cuh): one token's top-k expert outputs
// weighted, summed, plus the residual. Task x of grid (T, 1, 1).
#pragma once
#include "sm90_common.cuh"

namespace static_mk {

// PARAMS {H, TOPK}; BUF_SLOTS {expert outputs [T, TOPK, H] bf16, weights [T,
// TOPK] fp32, residual [T, H] bf16, output [T, H] bf16}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_mul_sum_add(Maps const &,
                                                     G const &g,
                                                     KernelLocals &,
                                                     StaticTask const &tk) {
  constexpr int H = PARAMS::v[0], TOPK = PARAMS::v[1];
  using Row = Sm90Row<H>;
  sm90_task_begin<IN...>(g);
  uint4 const *eo = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 0>()) +
                    (size_t)tk.x * TOPK * Row::VEC;
  float const *w =
      buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()) + (size_t)tk.x * TOPK;
  uint4 const *res =
      buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 2>()) + (size_t)tk.x * Row::VEC;
  uint4 *out =
      buf_at<uint4>(g, slot_at<BUF_SLOTS, 3>()) + (size_t)tk.x * Row::VEC;
  float wk[TOPK];
#pragma unroll
  for (int k = 0; k < TOPK; k++) {
    wk[k] = w[k];
  }
  uint4 ev[Row::PER][TOPK], rv[Row::PER];
#pragma unroll
  for (int j = 0; j < Row::PER; j++) {
    int const v = Row::vec(j);
    if (v < Row::VEC) {
#pragma unroll
      for (int k = 0; k < TOPK; k++) {
        ev[j][k] = eo[k * Row::VEC + v];
      }
      rv[j] = res[v];
    }
  }
#pragma unroll
  for (int j = 0; j < Row::PER; j++) {
    int const v = Row::vec(j);
    if (v < Row::VEC) {
      float acc[8], e[8];
      bf16x8_to_f32(rv[j], acc);
#pragma unroll
      for (int k = 0; k < TOPK; k++) {
        bf16x8_to_f32(ev[j][k], e);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          acc[i] += wk[k] * e[i];
        }
      }
      out[v] = f32_to_bf16x8(acc);
    }
  }
  sm90_task_end<SELF>(g);
}

SM90_TASK_SMEM(sm90_mul_sum_add, 0)

} // namespace static_mk
