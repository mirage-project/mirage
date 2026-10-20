// tasks/sm90_bias.cuh -- the sm90_bias task type: a GEMM tile node's fp32
// output to bf16 with a bias row (and a residual), keeping its first N_OUT
// columns (a GEMM padded to whole 64-row tiles, e.g. GPT-OSS's 32-expert
// router). Task x of grid (T, 1, 1): token x.
#pragma once
#include "sm90_common.cuh"

namespace static_mk {

// PARAMS {N_IN (the input's row length), N_OUT, has_residual}; BUF_SLOTS
// {input [T, N_IN] fp32, bias [N_OUT] bf16, residual [T, N_OUT] bf16 (-1:
// none), output [T, N_OUT] bf16}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_bias(Maps const &,
                                              G const &g,
                                              KernelLocals &,
                                              StaticTask const &tk) {
  constexpr int N_IN = PARAMS::v[0], N_OUT = PARAMS::v[1];
  constexpr bool RESIDUAL = PARAMS::v[2] == 1;
  using Row = Sm90Row<N_OUT>;
  static_assert(N_OUT <= N_IN && N_IN % 4 == 0,
                "sm90_bias: a prefix of the columns, 16-B vectors");
  sm90_task_begin<IN...>(g);
  float const *in =
      buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()) + (size_t)tk.x * N_IN;
  uint4 const *bias = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 1>());
  uint4 const *res = RESIDUAL
                         ? buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 2>()) +
                               (size_t)tk.x * Row::VEC
                         : nullptr;
  uint4 *out =
      buf_at<uint4>(g, slot_at<BUF_SLOTS, 3>()) + (size_t)tk.x * Row::VEC;
  float a[Row::PER][8];
  uint4 bv[Row::PER], rv[Row::PER];
#pragma unroll
  for (int j = 0; j < Row::PER; j++) {
    int const v = Row::vec(j);
    if (v < Row::VEC) {
      f32x8_load(in, v, a[j]);
      bv[j] = bias[v];
      if constexpr (RESIDUAL) {
        rv[j] = res[v];
      }
    }
  }
#pragma unroll
  for (int j = 0; j < Row::PER; j++) {
    int const v = Row::vec(j);
    if (v < Row::VEC) {
      float b[8], r[8] = {};
      bf16x8_to_f32(bv[j], b);
      if constexpr (RESIDUAL) {
        bf16x8_to_f32(rv[j], r);
      }
#pragma unroll
      for (int i = 0; i < 8; i++) {
        a[j][i] = a[j][i] + b[i] + r[i];
      }
      out[v] = f32_to_bf16x8(a[j]);
    }
  }
  sm90_task_end<SELF>(g);
}

SM90_TASK_SMEM(sm90_bias, 0)

} // namespace static_mk
