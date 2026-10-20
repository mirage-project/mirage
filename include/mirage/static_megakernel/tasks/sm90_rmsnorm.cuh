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
  // 8 columns a thread at a time, the row's loads issued before the sums
  static_assert(H % 8 == 0, "sm90_rmsnorm: 16-B vectors");
  constexpr int VEC = H / 8, PER = (VEC + 255) / 256;
  __shared__ float part[8];
  sm90_task_begin<IN...>(g);
  uint4 const *x =
      buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 0>()) + (size_t)tk.x * VEC;
  uint4 const *w = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 1>());
  uint4 *out = buf_at<uint4>(g, slot_at<BUF_SLOTS, 2>()) + (size_t)tk.x * VEC;
  uint4 xv[PER], wv[PER];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < PER; j++) {
    int const v = threadIdx.x + 256 * j;
    if (v < VEC) {
      xv[j] = x[v];
      wv[j] = w[v];
    }
  }
#pragma unroll
  for (int j = 0; j < PER; j++) {
    if (threadIdx.x + 256 * j < VEC) {
      float f[8];
      bf16x8_to_f32(xv[j], f);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        ss += f[i] * f[i];
      }
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, o);
  }
  if ((threadIdx.x & 31) == 0) {
    part[threadIdx.x >> 5] = ss;
  }
  __syncthreads();
  float tot = 0.f;
#pragma unroll
  for (int k = 0; k < 8; k++) {
    tot += part[k];
  }
  float const rs = rsqrtf(tot / H + __int_as_float(PARAMS::v[1]));
#pragma unroll
  for (int j = 0; j < PER; j++) {
    int const v = threadIdx.x + 256 * j;
    if (v < VEC) {
      float f[8], gw[8];
      bf16x8_to_f32(xv[j], f);
      bf16x8_to_f32(wv[j], gw);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        f[i] = f[i] * rs * gw[i];
      }
      out[v] = F16 ? f32_to_f16x8(f) : f32_to_bf16x8(f);
    }
  }
  sm90_task_end<SELF>(g);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sm90_rmsnorm() {
  return 0;
}

} // namespace static_mk
