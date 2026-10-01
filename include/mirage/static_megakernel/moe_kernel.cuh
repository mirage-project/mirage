// moe_kernel.cuh -- the MoE layer's kernel pieces and task functions for the generated layer (static_schedule.py). The generated
// kernel is:
//
//   __global__ void __launch_bounds__(256, 1) layer_kernel(const __grid_constant__ static_mk::Maps maps, static_mk::G g, StaticTask const *static_tasks) {
//     static_mk::KernelLocals L;
//     static_mk::kernel_begin(maps, g, L);
//     StaticTask const *my = static_tasks + blockIdx.x * static_mk::MAX_TASKS_PER_SM;     // this SM's list
//     for (int ti = 0; ti < static_mk::MAX_TASKS_PER_SM; ti++) {
//       StaticTask const tk = my[ti];                                              // {node, x, y, z}
//       if (tk.node < 0) break;
//       switch (tk.node) {                                                          // one case per graph node
//         case 30: static_mk::run_gemm_tile<StaticGrid<7, 14, 1>, StaticParams<0>, ...>(maps, g, L, tk); break;
//         case 31: static_mk::run_route<StaticGrid<8, 1, 1>, StaticParams<>, StaticGrid<7, 14, 1>, ...>(maps, g, L, tk); break;
//         ...
//       }
//     }
//     static_mk::kernel_end(g, L);
//   }
//
// A task function's template arguments: its node's grid (SELF), its node's params (PARAMS), then for each of its inputs the grid
// of the node that writes it (StaticGrid<0, 0, 0> for a graph input). The task's grid position is (tk.x, tk.y, tk.z).
#pragma once
#include "config.cuh"
#include "runtime.cuh"
#include "moe_types.cuh"
#include "gemm_tile.cuh"
#include "route.cuh"
#include "quant.cuh"
#include "sact.cuh"
#include "expert_queue.cuh"
#include "tail.cuh"
#include "expert_phase.cuh"

namespace static_mk {

// the kernel's local variables (every function is inlined, so they stay in registers)
struct KernelLocals {
  uint32_t base;        // 1024-aligned dynamic shared memory start (shared-window address)
  char *sm;             // the same, as a pointer
  int sm_id;
  uint32_t tb;          // TMEM base
  int gl, gi0, gi1;     // ring stage counters: loader, issuer 0, issuer 1 (they advance by the same stages)
  int wt;               // accumulator tiles so far (all roles)
  bool zq_resident;     // the phase switch to the expert queue is done (the ring was reset)
};

// before the task loop: shared memory, barriers, the start barrier over all GPUs, TMEM
__device__ __forceinline__ void kernel_begin(Maps const &maps, G const &g, KernelLocals &L) {
  (void)maps;
  extern __shared__ __align__(1024) char smem_raw[];
  L.base = (su32(smem_raw) + 1023u) & ~1023u;
  L.sm = smem_raw + (L.base - su32(smem_raw));
  L.sm_id = blockIdx.x;
  int const sm_id = L.sm_id;
  cta_prologue(L.sm);
  if (threadIdx.x == 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");   // PDL: from here on global memory is read (no-op without a programmatic launch)
    g.stamps[sm_id * NSTAMP + STAMP_START] = gtime();
    // start barrier: SM 0 of each GPU writes the launch number into its hello slot on every GPU; every SM waits for all slots
    if (sm_id == 0) push16(g, RG_HELLO + (size_t)g.rank * 16, make_uint4(g.gen, g.gen, g.gen, g.gen));
    for (int r = 0; r < g.tp; r++) {
      volatile unsigned const *hp = (volatile unsigned const *)(g.rv + RG_HELLO + (size_t)r * 16);
      while (*hp < g.gen) { }
    }
    *g.start_barrier = gtime();
  }
  L.tb = tmem_alloc_512();     // its __syncthreads also publishes rt
  L.gl = 0; L.gi0 = 0; L.gi1 = 0;
  L.wt = 0;
  L.zq_resident = false;
}

// GEMM tile (x, y): weight rows [128 x, 128 x + 128) times K part y of SELF::y; PARAMS: {kind (EpiKind: which weight / output), K}
template <class SELF, class PARAMS, class... IN>
__device__ __forceinline__ void run_gemm_tile(Maps const &maps, G const &g, KernelLocals &L, StaticTask const &tk) {
  constexpr int kind = PARAMS::v[0], K = PARAMS::v[1];
  static_assert(K > 0 && K % (128 * SELF::y) == 0, "params = {kind, K}; each K part is whole 128-column tiles");
  constexpr int pieces = K / 128 / SELF::y;   // K tiles of 128 per task
  CUtensorMap const *wmap = (kind == EPI_ROUTER) ? &maps.wg : (kind == EPI_LATENT) ? &maps.wdown : (kind == EPI_SGU) ? &maps.wsgu : &maps.wsd;
  CUtensorMap const *amap = (kind == EPI_SDOWN) ? &maps.hs : &maps.x;
  TileJob const j = make_bf16_tile(wmap, amap, tk.x * 128, tk.y * pieces, pieces);
  if (kind == EPI_SDOWN) {   // after the expert queue (if it ran on this SM): wait for h_s; fresh ring and accumulator barriers
    __syncthreads();
    if (threadIdx.x == 0) { cnt_wait(g.cnt + C_HS, N_SACT); if (L.zq_resident) ring_reinit(true); }
    if (L.zq_resident) { L.gl = L.gi0 = L.gi1 = 0; L.wt = 0; }
    __syncthreads();
  }
  gemm_tile_task(g, j, L.base, L.tb, L.gl, L.gi0, L.gi1, L.wt, kind, tk.x * 128, tk.y);
  L.wt++;
}

// route task x = token x; adds the router's LOGITS::y partial sums
template <class SELF, class PARAMS, class LOGITS, class... REST>
__device__ __forceinline__ void run_route(Maps const &, G const &g, KernelLocals &L, StaticTask const &tk) {
  route_task<LOGITS::y>(g, tk.x, L.sm);
}

// quant task x = z columns [128 x, 128 x + 128); adds latent_down's Z::y partial sums
template <class SELF, class PARAMS, class Z, class... REST>
__device__ __forceinline__ void run_quant(Maps const &, G const &g, KernelLocals &, StaticTask const &tk) {
  quant_task<Z::y>(g, tk.x);
}

// sact task x = h_s features [64 x, 64 x + 64); adds shared gate_up's SGU::y partial sums
template <class SELF, class PARAMS, class SGU, class... REST>
__device__ __forceinline__ void run_sact(Maps const &, G const &g, KernelLocals &, StaticTask const &tk) {
  sact_task<SGU::y>(g, tk.x);
}

// expert queue task (one per SM): wait for z_q, phase switch, routing table, then the queue (expert_queue.cuh)
template <class SELF, class PARAMS, class... IN>
__device__ __forceinline__ void run_expert_queue(Maps const &maps, G const &g, KernelLocals &L, StaticTask const &) {
  int const sm_id = L.sm_id;
  __syncthreads();   // every warp is done with the placed tiles (the loader runs ahead of the landings)
  for (int i = threadIdx.x; i < NE; i += 256) cntE_s[i] = 0;
  wait_z_landed(g);
  if (threadIdx.x == 0) g.stamps[sm_id * NSTAMP + STAMP_QUEUE_START] = gtime();
  if (!L.zq_resident) { expert_phase_switch(g, maps, L.base, L.tb, L.gl, L.gi0, L.gi1); L.zq_resident = true; }
  build_table();
  __syncthreads();
  run_expert_dynamic(g, maps, L.base, L.tb, L.gl, L.gi0, L.gi1, L.wt);
  if (threadIdx.x == 0) g.stamps[sm_id * NSTAMP + STAMP_QUEUE_END] = gtime();
  __syncthreads();
}

// tail task (one per SM, after the expert queue task): [R|S] over the GPUs, RMSNorm, latent_up, y
template <class SELF, class PARAMS, class... IN>
__device__ __forceinline__ void run_tail(Maps const &maps, G const &g, KernelLocals &L, StaticTask const &) {
  tail_task(g, maps, L.base, L.tb, L.sm, L.gl, L.gi0, L.gi1, L.wt, L.sm_id);
}

// after the task loop
__device__ __forceinline__ void kernel_end(G const &g, KernelLocals &L) {
  __syncthreads();
  if (threadIdx.x == 0) g.stamps[L.sm_id * NSTAMP + STAMP_END] = gtime();
  tmem_dealloc_512(L.tb);
}

}  // namespace static_mk
