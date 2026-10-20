// tasks/sm90_router.cuh -- the sm90_router task type: GPT-OSS's router
// logits, logits[t][e] = bf16(h[t] . W[e] + b[e]) for the T token rows. Task x
// of grid (E / EPT, 1, 1): experts [x EPT, x EPT + EPT). The 256 threads take
// K in 8-value vectors (each vector's T activations and EPT weights in
// registers), then the sums are reduced over each warp (shuffles) and the 8
// warps (shared memory). Replaces a one-task GEMM of the padded 64-row weight
// plus a bias node: E / EPT tasks, each EPT rows.
#pragma once
#include "sm90_common.cuh"

namespace static_mk {

// PARAMS {K, E, EPT (experts a task)}; BUF_SLOTS {h [T, K] bf16, weight [E, K]
// bf16, bias [E] bf16, logits [T, E] bf16}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_router(Maps const &,
                                                G const &g,
                                                KernelLocals &,
                                                StaticTask const &tk) {
  constexpr int K = PARAMS::v[0], E = PARAMS::v[1], EPT = PARAMS::v[2];
  constexpr int VEC = K / 8, PER = (VEC + 255) / 256, NS = EPT * T;
  static_assert(K % 8 == 0 && E % EPT == 0, "sm90_router: 16-B vectors");
  __shared__ float part[8][NS]; // per warp: its (expert, token) sums
  sm90_task_begin<IN...>(g);
  uint4 const *h = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 0>());
  uint4 const *w = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 1>()) +
                   (size_t)tk.x * EPT * VEC;
  float acc[EPT][T] = {};
#pragma unroll
  for (int j = 0; j < PER; j++) {
    int const v = threadIdx.x + 256 * j;
    if (v < VEC) {
      uint4 hv[T], wv[EPT];
#pragma unroll
      for (int t = 0; t < T; t++) {
        hv[t] = h[t * VEC + v];
      }
#pragma unroll
      for (int e = 0; e < EPT; e++) {
        wv[e] = w[e * VEC + v];
      }
#pragma unroll
      for (int e = 0; e < EPT; e++) {
        float we[8];
        bf16x8_to_f32(wv[e], we);
#pragma unroll
        for (int t = 0; t < T; t++) {
          float ht[8];
          bf16x8_to_f32(hv[t], ht);
#pragma unroll
          for (int i = 0; i < 8; i++) {
            acc[e][t] += we[i] * ht[i];
          }
        }
      }
    }
  }
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
#pragma unroll
  for (int e = 0; e < EPT; e++) {
#pragma unroll
    for (int t = 0; t < T; t++) {
      float s = acc[e][t];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, o);
      }
      if (lane == 0) {
        part[warp][e * T + t] = s;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < NS) {
    int const e = threadIdx.x / T, t = threadIdx.x % T;
    float s = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      s += part[k][threadIdx.x];
    }
    int const ex = tk.x * EPT + e;
    s += __bfloat162float(
        buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>())[ex]);
    buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 3>())[t * E + ex] =
        __float2bfloat16(s);
  }
  sm90_task_end<SELF>(g);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sm90_router() {
  return 0;
}

} // namespace static_mk
