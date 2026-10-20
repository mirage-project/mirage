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

// PARAMS {K, E, EPT (experts a task), F16 (1: h is f16)}; BUF_SLOTS {h [T, K]
// bf16 or f16, weight [E, K] bf16, bias [E] bf16, logits [T, E] bf16}
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
  constexpr bool F16 = PARAMS::v[3] == 1;
  using Row = Sm90Row<K>;
  constexpr int VEC = Row::VEC, NS = EPT * T;
  static_assert(E % EPT == 0, "sm90_router: whole tasks of experts");
  __shared__ float part[SM90_WARPS][NS]; // per warp: its (expert, token) sums
  sm90_task_begin<IN...>(g);
  uint4 const *h = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 0>());
  uint4 const *w = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 1>()) +
                   (size_t)tk.x * EPT * VEC;
  float acc[EPT][T] = {};
#pragma unroll
  for (int j = 0; j < Row::PER; j++) {
    int const v = Row::vec(j);
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
          if constexpr (F16) {
            f16x8_to_f32(hv[t], ht);
          } else {
            bf16x8_to_f32(hv[t], ht);
          }
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
      float const s = sm90_warp_sum(acc[e][t]);
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
    for (int k = 0; k < SM90_WARPS; k++) {
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

SM90_TASK_SMEM(sm90_router, 0)

} // namespace static_mk
