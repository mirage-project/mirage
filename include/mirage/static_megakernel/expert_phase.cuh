// expert_phase.cuh -- the two steps between an SM's placed tasks and its expert queue: wait until z_q of every GPU has landed in
// this GPU's copy of the exchange region (wait_z_landed), then the phase switch (fresh ring barriers and counters, the z_q scale
// chunks to TMEM, the routing pairs to shared memory).
#pragma once
#include "config.cuh"
#include "runtime.cuh"
#include "moe_types.cuh"
#include "expert_queue.cuh"

namespace static_mk {

// fresh ring barriers and stage counters; the 28 z_q scale chunks -> TMEM columns 72.. (warp 1); the 128 routing pairs -> shared
// memory (warps 2..5, table_poll)
__device__ __forceinline__ void expert_phase_switch(G const &g, Maps const &maps, uint32_t base, uint32_t tb, int &gl, int &gi0, int &gi1) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) ring_reinit(false);
  gl = gi0 = gi1 = 0;
  __syncthreads();
  if (threadIdx.x == 0) {
    mbar_expect(rt.misc0, KT_LAT * SF_CHUNK);
    tma3(&maps.xsf28, rt.misc0, base + OFF_XSF, 0, 0, 0, EVICT_LAST);
  }
  mbar_wait(rt.misc0, 0); fence_after();
  if (warp == 1 && lane == 0) {
    for (int kt = 0; kt < KT_LAT; kt++) cp_sf(tb + 72 + 4 * kt, base + OFF_XSF + kt * SF_CHUNK);
    tc_commit(rt.misc0 + 16);
  }
  if (warp >= 2 && warp <= 5) table_poll(g, threadIdx.x - 64);
  mbar_wait(rt.misc0 + 16, 0); fence_after();
  __syncthreads();
}

// wait until z_q (all K tiles) and its scale chunks from every GPU have landed: each thread polls its 16-B pieces (0xFF-prefilled)
__device__ __forceinline__ void wait_z_landed(G const &g) {
  constexpr int NZV = (int)((RG_ZSF + KT_LAT * SF_CHUNK) / 16), PER = (NZV + 255) / 256;
  bool ok[PER];
  bool all;
#pragma unroll
  for (int q = 0; q < PER; q++) ok[q] = (threadIdx.x + 256 * q) >= NZV;
  do {
    uint4 u[PER];
#pragma unroll
    for (int q = 0; q < PER; q++) if (!ok[q]) u[q] = ld16_relaxed(g.rv + (size_t)(threadIdx.x + 256 * q) * 16);
    all = true;
#pragma unroll
    for (int q = 0; q < PER; q++) { if (!ok[q]) ok[q] = valid16(u[q]); all = all && ok[q]; }
  } while (!all);
}

}  // namespace static_mk
