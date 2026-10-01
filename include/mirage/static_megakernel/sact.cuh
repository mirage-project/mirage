// sact.cuh -- the shared-expert activation of h_s features [64 tile, 64 tile + 64): wait until the NSPLIT K parts of shared gate_up
// row tile `tile` (rows [128 tile, +64) gate, [+64, +128) up) are added into spart (counter C_SGU[tile]), then
// h_s[t][64 tile + f] = bf16(SiTU(gate, up)), then counter C_HS += 1. Warps 0..1 (64 features) work.
#pragma once
#include "runtime.cuh"
#include "moe_types.cuh"

namespace static_mk {

template <int NSPLIT>
__device__ __forceinline__ void sact_task(G const &g, int tile) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) cnt_wait(g.cnt + C_SGU + tile, NSPLIT);
  __syncthreads();
  if (warp < 2) {
    int const f = warp * 32 + lane;
    float gv[T], uv[T];
#pragma unroll
    for (int t = 0; t < T; t++) {
      gv[t] = g.spart[t * (2 * SHR) + tile * 128 + f];
      uv[t] = g.spart[t * (2 * SHR) + tile * 128 + 64 + f];
    }
#pragma unroll
    for (int t = 0; t < T; t++) g.hs[t * SHR + tile * 64 + f] = __float2bfloat16(situ(gv[t], uv[t]));
    __syncwarp(); asm volatile("bar.sync 2, 64;" ::: "memory");
    if (threadIdx.x == 0) cnt_add(g.cnt + C_HS, 1);
  }
  __syncthreads();
}

}  // namespace static_mk
