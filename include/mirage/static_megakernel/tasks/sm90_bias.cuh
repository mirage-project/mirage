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
  // 8 columns a thread at a time, every load of the row issued before the
  // first use (a strided loop waits out the memory latency per element)
  static_assert(N_OUT % 8 == 0 && N_IN % 4 == 0, "sm90_bias: 16-B vectors");
  constexpr int VEC = N_OUT / 8, PER = (VEC + 255) / 256;
  float const *in =
      buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()) + (size_t)tk.x * N_IN;
  uint4 const *bias = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 1>());
  uint4 const *res =
      RESIDUAL
          ? buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 2>()) + (size_t)tk.x * VEC
          : nullptr;
  uint4 *out = buf_at<uint4>(g, slot_at<BUF_SLOTS, 3>()) + (size_t)tk.x * VEC;
  float4 a[PER][2];
  uint4 bv[PER], rv[PER];
#pragma unroll
  for (int j = 0; j < PER; j++) {
    int const v = threadIdx.x + 256 * j;
    if (v < VEC) {
      a[j][0] = reinterpret_cast<float4 const *>(in)[2 * v];
      a[j][1] = reinterpret_cast<float4 const *>(in)[2 * v + 1];
      bv[j] = bias[v];
      if constexpr (RESIDUAL) {
        rv[j] = res[v];
      }
    }
  }
#pragma unroll
  for (int j = 0; j < PER; j++) {
    int const v = threadIdx.x + 256 * j;
    if (v < VEC) {
      float f[8], b[8], r[8] = {};
      bf16x8_to_f32(bv[j], b);
      if constexpr (RESIDUAL) {
        bf16x8_to_f32(rv[j], r);
      }
      float const x[8] = {a[j][0].x,
                          a[j][0].y,
                          a[j][0].z,
                          a[j][0].w,
                          a[j][1].x,
                          a[j][1].y,
                          a[j][1].z,
                          a[j][1].w};
#pragma unroll
      for (int i = 0; i < 8; i++) {
        f[i] = x[i] + b[i] + r[i];
      }
      out[v] = f32_to_bf16x8(f);
    }
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
