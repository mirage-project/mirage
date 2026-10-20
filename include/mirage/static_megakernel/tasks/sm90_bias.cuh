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
  static_assert(N_OUT <= N_IN, "sm90_bias: keeps a prefix of the columns");
  sm90_task_begin<IN...>(g);
  float const *in =
      buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()) + (size_t)tk.x * N_IN;
  __nv_bfloat16 const *bias =
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>());
  __nv_bfloat16 *out =
      buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 3>()) + (size_t)tk.x * N_OUT;
  for (int i = threadIdx.x; i < N_OUT; i += blockDim.x) {
    float v = in[i] + __bfloat162float(bias[i]);
    if constexpr (RESIDUAL) {
      v += __bfloat162float(buf_at<__nv_bfloat16 const>(
          g, slot_at<BUF_SLOTS, 2>())[(size_t)tk.x * N_OUT + i]);
    }
    out[i] = __float2bfloat16(v);
  }
  sm90_task_end<SELF>(g);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sm90_bias() {
  return 0;
}

} // namespace static_mk
