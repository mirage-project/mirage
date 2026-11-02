// tasks/scaled_gemm_tile.cuh -- Compute one FP8 GEMM tile with fp32 scales.
// scaled_gemm_tile_layer declares this task. run_scaled_gemm_tile runs it.
// The task follows gemm_tile.cuh with e4m3 operands and fp32 scales.
// scale_promotion.cuh defines the terms used below.
//   weight      e4m3 pieces, [N / 128][K / 128] (the piece layout of
//               mirage.mpk.fp8_fp32scale_weights)
//   activation  z_q, the output of a group_quant node (or of a node with the
//               same output, dsa_q_norm_quant): e4m3 [T][K], then the fp32
//               scales [K / 128][T] (one for each token and K tile)
// out[t][row0 + r] = the sum over the pieces kt of the task of
// sw(r, kt) sx(t, kt) partial(r, t, kt), for r < 128. The MMA uses scale 1 for
// both operands. Each piece has its own TMEM accumulator. The epilogue
// multiplies it by the fp32 scales (scale_promotion.cuh issue_promoted,
// drain_promoted). `mode` (Combine, core.cuh) tells how the node's output
// buffer combines the K parts, as for gemm_tile.
// The packer preserves the checkpoint scale values. PARAMS[3] selects their
// layout (scale_promotion.cuh):
//   0  BlockScales: one scale for each piece, [N / 128][K / 128] (a plain
//      weight: shared down; pack_w2)
//   1  GateUpScales: each piece is 64 gate rows, then their 64 up rows,
//      [N / 256][K / 128][2] (shared gate_up; pack_w13). The output is then in
//      the layout that situ_and_mul and silu_and_mul read.
#pragma once
#include "../core.cuh"
#include "../scale_promotion.cuh" // the promotion of each K tile
#include "gemm_tile.cuh"          // tile_epilogue
#include "group_quant.cuh"        // the layout of the activation (zq32_bytes)
#include <type_traits>

namespace static_mk {

// After the ring, reserve 1 KB for the W2 mbarriers of expert tasks.
// Those mbarriers can still be live when this task runs on the same SM.
// Place the constant scale chunk after that area, then the task's scales:
// bs [NATOMS][WS::PER_PIECE], followed by xs [NATOMS][T].
constexpr int SGT_OFF_CSF = RING_BYTES + 1024,
              SGT_OFF_SC = SGT_OFF_CSF + SF_CHUNK;
// TMEM columns of the constant scale chunk: issuer 0 at 232, issuer 1 at 236
// (scaled_moe_experts puts the same chunk at column 232, Smem::CONST_SF)
constexpr int SGT_CONST_SF = 232;

// a job of NATOMS pieces: piece row m, K tiles [k0, k0 + NATOMS)
template <int NATOMS>
__device__ __forceinline__ TileJob make_scaled_tile(
    CUtensorMap const *wmap, CUtensorMap const *amap, int m, int k0, int KT) {
  TileJob j = {};
  j.kind = 1;
  j.nst = PromoRounds<NATOMS>::NST;
  j.natoms = NATOMS;
  j.wmap = wmap;      // one piece for each box
  j.wmap2 = wmap + 1; // two pieces for each box
  j.amap = amap;      // one K tile of the activation
  j.amap2 = amap + 2; // two K tiles (the act8kt map, two slots after)
  j.tile0 = m * KT + k0;
  j.k0 = k0;
  return j;
}

// The whole CTA runs one tile. All 256 threads enter; the warps take roles as
// in gemm_tile_task.
//   wt   the accumulator ticks so far on this SM. The task uses
//        PromoRounds<NATOMS>::ROUNDS ticks.
//   wsf  the weight scales of the task, [NATOMS][WS::PER_PIECE] (WS: their
//        layout)
//   xsf  the activation scales of the task, [NATOMS][T]
//   out, N, cnt, row0, split: as gemm_tile_task
template <int NATOMS, class WS>
__device__ __forceinline__ void scaled_gemm_tile_task(TileJob const &j,
                                                      uint32_t base,
                                                      uint32_t tb,
                                                      int &gl,
                                                      int &gi0,
                                                      int &gi1,
                                                      int wt,
                                                      int mode,
                                                      void *out,
                                                      int N,
                                                      uint32_t *cnt,
                                                      int row0,
                                                      int split,
                                                      float const *wsf,
                                                      float const *xsf) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x < SF_CHUNK / 4) { // the constant scale chunk (UE8M0 1)
    const_sf_fill(base + SGT_OFF_CSF, threadIdx.x);
  }
  __syncthreads();
  if (warp == 0) {
    if (lane == 0) {
      load_job<128, 16384, false>(j, base, gl);
    }
  } else if (warp == 1 || warp == 6) {
    int const iss = warp == 1 ? 0 : 1;
    int &gi = iss == 0 ? gi0 : gi1;
    gi = __shfl_sync(0xffffffffu, gi, 0); // gemm_tile advances lane 0 only
    uint32_t const sfc = tb + SGT_CONST_SF + 4 * iss;
    if (issuer_lane<true>()) { // the lane that issues the MMAs: the copy is
                               // before them in its tcgen05 order
      cp_sf(sfc, base + SGT_OFF_CSF);
    }
    fence_before();
    __syncwarp();
    fence_after();
    issue_promoted<IDESC_MX_E4M3, NATOMS>(base, tb, iss, gi, wt, sfc);
  } else if (warp >= 2 && warp <= 5) {
    float *const bs = reinterpret_cast<float *>(rt_sm() + SGT_OFF_SC),
                 *const xs = bs + WS::PER_PIECE * NATOMS;
    for (int i = threadIdx.x - 64; i < NATOMS * WS::PER_PIECE; i += 128) {
      bs[i] = wsf[i];
    }
    for (int i = threadIdx.x - 64; i < NATOMS * T; i += 128) {
      xs[i] = ld_relaxed_f32(xsf + i);
    }
    asm volatile("bar.sync 1, 128;" ::: "memory");
    float v[8];
    int const row = drain_promoted<NATOMS, WS>(tb, wt, bs, xs, v);
    switch (mode) {
      case COMBINE_SLOTS:
        tile_epilogue<COMBINE_SLOTS>(out, N, row0, split, row, v);
        break;
      case COMBINE_ADD:
        tile_epilogue<COMBINE_ADD>(out, N, row0, split, row, v);
        break;
      default:
        tile_epilogue<COMBINE_STORE>(out, N, row0, split, row, v);
        break;
    }
    asm volatile(
        "bar.sync 1, 128;" ::
            : "memory"); // the 128 epilogue threads: their stores are done
    if (threadIdx.x == 64 && mode != COMBINE_SLOTS) {
      cnt_add(cnt + (mode == COMBINE_ADD ? row0 / 128 : 0), 1);
    }
  }
}

// GEMM tile (x, y): weight rows [128 x, 128 x + 128) times K part y of
// SELF::y -> the node's output buffer (SELF::buf).
//   PARAMS  {mode (Combine), K, N (the weight's rows), the layout of the weight
//           scales (0 BlockScales, 1 GateUpScales)}
//   BUF_SLOTS  {the weight scales (a graph tensor, fp32)}
//   MAP_SLOTS  {the weight's map (one piece for each box; two pieces at the
//              next slot), the activation's map (one K tile, set 0; set 1 at
//              the next slot; then two K tiles, set 0 and set 1)}
//   ACT     the group_quant node that writes the activation. Its counter
//           counts its tasks.
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class ACT,
          class... IN>
__device__ __forceinline__ void run_scaled_gemm_tile(Maps const &maps,
                                                     G const &g,
                                                     KernelLocals &L,
                                                     StaticTask const &tk) {
  constexpr int mode = PARAMS::v[0], K = PARAMS::v[1], N = PARAMS::v[2],
                SL = PARAMS::v[3], KT = K / 128, WMAP = slot_at<MAP_SLOTS, 0>(),
                AMAP = slot_at<MAP_SLOTS, 1>(), WSF = slot_at<BUF_SLOTS, 0>();
  static_assert(K > 0 && K % (128 * SELF::y) == 0,
                "scaled_gemm_tile: each K part is whole 128-column tiles");
  constexpr int NATOMS = KT / SELF::y; // K tiles of the task
  static_assert(N == 128 * SELF::x,
                "scaled_gemm_tile: 128 weight rows per task");
  static_assert(mode == COMBINE_SLOTS || mode == COMBINE_ADD ||
                    mode == COMBINE_STORE,
                "mode: Combine");
  static_assert(mode != COMBINE_STORE || SELF::y == 1,
                "plain stores: one K part");
  static_assert(SL == 0 || (SL == 1 && N % 256 == 0),
                "the weight scales: BlockScales, or GateUpScales (pieces in "
                "gate / up pairs)");
  using WS = std::conditional_t<SL == 1, GateUpScales, BlockScales>;
  static_assert(SELF::buf >= 0 && (mode == COMBINE_SLOTS || SELF::counter >= 0),
                "the node's output buffer and counters");
  static_assert(WMAP + 1 < MAX_MAPS && AMAP + 3 < MAX_MAPS && WSF >= 0,
                "map slots and the weight scales");
  static_assert(
      ACT::x > 0 && ACT::counter >= 0 && ACT::buf >= 0 &&
          exchange_bytes[ACT::buf] == zq32_bytes(K),
      "scaled_gemm_tile: the activation is the output of a group_quant node "
      "(or of a node with the same output)");
  // the activation is in set rg_set() of the exchange region
  TileJob const j = make_scaled_tile<NATOMS>(
      &maps.m[WMAP], &maps.m[AMAP + dbuf_par], tk.x, tk.y * NATOMS, KT);
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_wait(g.cnt + ACT::counter, ACT::x * ACT::y * ACT::z);
    asm volatile("fence.proxy.async.global;" ::: "memory"); // then TMA reads
    if (L.ring_relaid) {
      ring_reinit(true);
    }
  }
  if (L.ring_relaid) {
    L.gl = L.gi0 = L.gi1 = 0;
    L.wt = 0;
  }
  __syncthreads();
  int const k0 = tk.y * NATOMS;
  float const *const wsf =
      buf_at<float const>(g, WSF) + WS::first(tk.x, KT, k0);
  float const *const xsf =
      reinterpret_cast<float const *>(g.rv + exchange_offset[ACT::buf] +
                                      rg_set() + (size_t)T * K) +
      (size_t)k0 * T;
  scaled_gemm_tile_task<NATOMS, WS>(j,
                                    L.base,
                                    L.tb,
                                    L.gl,
                                    L.gi0,
                                    L.gi1,
                                    L.wt,
                                    mode,
                                    g.buf[SELF::buf],
                                    N,
                                    g.cnt +
                                        (SELF::counter < 0 ? 0 : SELF::counter),
                                    tk.x * 128,
                                    tk.y,
                                    wsf,
                                    xsf);
  L.wt += PromoRounds<NATOMS>::ROUNDS;
}

// dynamic shared memory: the ring, the constant scale chunk, the scales
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_scaled_gemm_tile() {
  return SGT_OFF_SC + PARAMS::v[1] / 128 / SELF::y * (2 + T) * 4;
}

} // namespace static_mk
