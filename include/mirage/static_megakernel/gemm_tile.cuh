// gemm_tile.cuh -- one bf16 GEMM tile: out[t][row0 + r] = sum over K tiles [k0, k0 + nk) of weight[row0 + r][k] * act[t][k], for
// r < 128 and the T = 8 tokens, streamed through the ring (runtime.cuh). Used for the router, latent_down, shared gate_up and
// shared down GEMMs (run_gemm_tile) and for the tail's latent_up. `kind` says where the 128 x 8 fp32 result goes.
#pragma once
#include "runtime.cuh"
#include "moe_types.cuh"

namespace static_mk {

// where a tile's result goes (task K part `split`, thread row `row`, v[t] = token t)
enum EpiKind {
  EPI_ROUTER = 0,   // lpart[split][t][row0 + row] = v[t]   one 0xFF-prefilled slot per K part; route adds the parts in a fixed order
  EPI_LATENT = 1,   // zpart[split][t][row0 + row] = v[t]   the same for latent_down; quant adds the parts
  EPI_SGU = 2,      // spart[t][row0 + row] += v[t]         red.add, then counter C_SGU[row0 / 128] += 1; sact waits for all K parts
  EPI_SDOWN = 3,    // Sout[t][row0 + row] = v[t]           plain store (one K part); the tail reads Sout after all SMs are done
};

// a bf16 job: weight rows [row0, row0 + 128), K tiles [k0, k0 + nk); amap = the activation (nullptr: already in the stages)
__device__ __forceinline__ TileJob make_bf16_tile(CUtensorMap const *wmap, CUtensorMap const *amap, int row0, int k0, int nk) {
  TileJob j = {};
  j.kind = 0; j.nst = nk; j.row0 = row0; j.k0 = k0; j.wmap = wmap; j.amap = amap;
  return j;
}

template <int KIND>
__device__ __forceinline__ void tile_epilogue(G const &g, int row0, int split, int row, float const *v) {
  if (KIND == EPI_SDOWN) {
    for (int t = 0; t < T; t++) g.Sout[t * H + row0 + row] = v[t];
  } else if (KIND == EPI_ROUTER) {
    for (int t = 0; t < T; t++) g.lpart[((size_t)split * T + t) * NE + row0 + row] = v[t];
  } else if (KIND == EPI_LATENT) {
    for (int t = 0; t < T; t++) g.zpart[((size_t)split * T + t) * LAT + row0 + row] = v[t];
  } else {
    for (int t = 0; t < T; t++) red_add(g.spart + t * (2 * SHR) + row0 + row, v[t]);
  }
}

// The whole CTA runs one tile (all 256 threads enter; roles by warp). `kind` is a run-time argument so that the streaming code
// exists once in the binary; a template parameter here instantiated the body four times (measured 2026-09-24: +3.5k SASS
// instructions, 60 -> 132 MMA sites), and fetching that code cold costs time in this kernel.
// wt = tiles run so far on this SM since the last reset: accumulator stage wt % 4, parity (wt / 4) % 2.
// Returns after the epilogue published the result (warps 2..5) or after the last stage was armed / issued (the others).
__device__ __forceinline__ void gemm_tile_task(G const &g, TileJob const &j, uint32_t base, uint32_t tb, int &gl, int &gi0, int &gi1,
                                               int wt, int kind, int row0, int split) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int const a = wt & 3, par = (wt >> 2) & 1;
  if (warp == 0) {
    if (lane == 0) load_job(j, base, gl);
  } else if (warp == 1 || warp == 6) {
    if (lane == 0) {
      int &gi = (warp == 1) ? gi0 : gi1;
      if (wt >= 4) mbar_wait(rt.acc_empty0 + 8 * a, ((wt >> 2) - 1) & 1);   // the stage's previous tile is drained
      issue_job(j, base, tb, warp == 1 ? 0 : 1, gi, a, 0);
    }
  } else if (warp >= 2 && warp <= 5) {
    float v[8];
    int const row = drain_acc(tb, a, par, v);
    switch (kind) {
      case EPI_ROUTER: tile_epilogue<EPI_ROUTER>(g, row0, split, row, v); break;
      case EPI_LATENT: tile_epilogue<EPI_LATENT>(g, row0, split, row, v); break;
      case EPI_SGU:    tile_epilogue<EPI_SGU>(g, row0, split, row, v); break;
      default:         tile_epilogue<EPI_SDOWN>(g, row0, split, row, v); break;
    }
    asm volatile("bar.sync 1, 128;" ::: "memory");   // the 128 epilogue threads
    if (threadIdx.x == 64 && kind == EPI_SGU) cnt_add(g.cnt + C_SGU + row0 / 128, 1);
  }
}

}  // namespace static_mk
