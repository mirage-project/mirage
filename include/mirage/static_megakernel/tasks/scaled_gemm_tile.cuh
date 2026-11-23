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
// UE8M0 (PARAMS[3] = 2): the weight and activation scales are powers of two
// (weights requantized as SGLang does for DeepGEMM on Blackwell; group_quant
// with UE8M0). The MMA applies them (scale_vec), so each issuer keeps one
// accumulator for all its pieces and the epilogue reads it once
// (scaled_gemm_tile_ue8m0_task).
// The packer preserves the checkpoint scale values. PARAMS[3] selects their
// layout (scale_promotion.cuh):
//   0  BlockScales: one scale for each piece, [N / 128][K / 128] (a plain
//      weight: shared down; pack_w2)
//   1  GateUpScales: each piece is 64 gate rows, then their 64 up rows,
//      [N / 256][K / 128][2] (shared gate_up; pack_w13). The output is then in
//      the layout that situ_and_mul and silu_and_mul read.
//   2  BlockScales, powers of two (UE8M0 above)
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
                                                      float const *xsf,
                                                      int s0 = 0) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x < SF_CHUNK / 4) { // the constant scale chunk (UE8M0 1)
    const_sf_fill(base + SGT_OFF_CSF, threadIdx.x);
  }
  __syncthreads();
  if (warp == 0) {
    if (lane == 0) { // from stage s0 (the activation-wait prefill issued the
                     // ones before: run_scaled_gemm_tile)
      load_job<128, 16384, false>(j, base, gl, -1, s0);
    }
  } else if (warp == 1 || warp == 6) {
    int const iss = warp == 1 ? 0 : 1;
    int &gi = iss == 0 ? gi0 : gi1;
    gi = __shfl_sync(0xffffffffu, gi, 0); // gemm_tile advances lane 0 only
    uint32_t const sfc = tb + SGT_CONST_SF + 4 * iss;
    if constexpr (!F8_UNSCALED) { // T = 64: no scale chunk (runtime.cuh
                                  // mma_f8)
      if (issuer_lane<true>()) {  // the lane that issues the MMAs: the copy is
                                  // before them in its tcgen05 order
        cp_sf(sfc, base + SGT_OFF_CSF);
      }
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
    float v[T];
    int const row = drain_promoted<NATOMS, WS>(tb, wt, bs, xs, v);
    switch (mode) {
      case COMBINE_SLOTS:
        tile_epilogue<COMBINE_SLOTS>(out, N, row0, split, row, v);
        break;
      case COMBINE_ADD:
        tile_epilogue<COMBINE_ADD>(out, N, row0, split, row, v);
        break;
      case COMBINE_STORE_BF16:
        tile_epilogue<COMBINE_STORE_BF16>(out, N, row0, split, row, v);
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

// UE8M0 jobs: TMEM columns of the constant scale chunk (issuer i at
// SGT_UE_CSF + 4 i) and of the activation-side chunks, one per ring scale slot
// (SGT_UE_SF + 4 (2 st + at)), past the accumulators (one set, columns below
// 2 ACC_STAGES TW)
constexpr int SGT_UE_CSF = 448, SGT_UE_SF = 456;
static_assert(2 * ACC_STAGES * TW <= SGT_UE_CSF &&
                  SGT_UE_SF + 4 * 2 * SMAX <= 512,
              "scaled_gemm_tile UE8M0: the scale chunks clear of the "
              "accumulators");

// issuer iss of a UE8M0 job of NATOMS pieces: issue_job_warp (runtime.cuh) with
// e4m3 pieces. The weight side scales by the constant chunk csf (UE8M0 1); the
// activation side of piece atom by a chunk whose row t holds the exponent of
// sx(t, atom) sw(atom) (ex[atom][t] + ew[atom] - 127; both UE8M0 bytes), which
// the warp writes into the stage's scale slot (OFF_WSF: the e4m3 loader leaves
// it free) and the elected lane copies to TMEM before the piece's MMAs. Each
// issuer adds its pieces into its accumulator of tick slot a. The whole warp
// calls the function.
template <int NATOMS>
__device__ __forceinline__ void issue_ue8m0(uint32_t base,
                                            uint32_t tb,
                                            int iss,
                                            int &g,
                                            int a,
                                            uint32_t csf,
                                            uint8_t const *ex,
                                            uint8_t const *ew) {
  constexpr int NST = (NATOMS + 1) / 2;
  int const lane = threadIdx.x & 31;
  uint32_t const acc = tb + TW * a + ACC_STAGES * TW * iss;
  bool first = true;
  for (int s = 0; s < NST; s++, g++) {
    int const st = g % SMAX;
    mbar_wait(rt.full0 + 8 * st, (g / SMAX) & 1);
    fence_after();
    bool const one = issuer_lane<true>();
    if ((s & 1) != iss) { // the other issuer's stage: seen
      if (one) {
        mbar_arrive(rt.empty0 + 8 * st);
      }
      continue;
    }
    int const na = (NATOMS - 2 * s < 2) ? NATOMS - 2 * s : 2;
    for (int at = 0; at < na; at++) {
      int const atom = 2 * s + at, slot = st * 2 + at;
      // the chunk (cp_sf's layout): lane l holds rows 32 c + l, c = 0 .. 3, at
      // bytes 16 l + 4 c; a row past the tokens scales by 1
      uint32_t const chunk = base + OFF_WSF + slot * SF_CHUNK;
      uint32_t w[4];
#pragma unroll
      for (int c = 0; c < 4; c++) {
        int const t = 32 * c + lane;
        int const e =
            t < T ? min(max((int)ex[atom * T + t] + (int)ew[atom] - 127, 1),
                        254)
                  : 127;
        w[c] = (uint32_t)e * 0x01010101u;
      }
      asm volatile(
          "st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(chunk + 16 * lane),
          "r"(w[0]),
          "r"(w[1]),
          "r"(w[2]),
          "r"(w[3])
          : "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
      __syncwarp();
      uint32_t const sfb = tb + SGT_UE_SF + 4 * slot;
      uint64_t const dw = mkdesc(base + st * FSTAGE + at * 16384);
      uint64_t const db =
          mkdesc(base + st * FSTAGE + W_STAGE + at * (TW * 128));
      if (one) {
        cp_sf(sfb, chunk);
        mma_mx(acc,
               dw,
               db,
               idesc_mx(IDESC_MX_E4M3, 0),
               first ? 0u : 1u,
               csf,
               sfb);
        mma_mx(acc, dw + 2, db + 2, idesc_mx(IDESC_MX_E4M3, 1), 1u, csf, sfb);
        mma_mx(acc, dw + 4, db + 4, idesc_mx(IDESC_MX_E4M3, 2), 1u, csf, sfb);
        mma_mx(acc, dw + 6, db + 6, idesc_mx(IDESC_MX_E4M3, 3), 1u, csf, sfb);
      }
      first = false;
    }
    if (one) {
      tc_commit(rt.empty0 + 8 * st);
    }
  }
  if (issuer_lane<true>()) { // an issuer without a stage (NST 1) arrives too
    tc_commit(rt.acc_full0 + 8 * a);
  }
}

// The whole CTA runs one UE8M0 tile (PARAMS[3] = 2), with roles as
// scaled_gemm_tile_task: one accumulator tick (wt) for the job.
//   wsf  the weight scales of the task, [NATOMS] fp32 powers of two
//   xsf  the activation scales of the task, [NATOMS][T] fp32 powers of two
template <int NATOMS>
__device__ __forceinline__ void scaled_gemm_tile_ue8m0_task(TileJob const &j,
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
                                                            float const *xsf,
                                                            int s0 = 0) {
  constexpr int NST = (NATOMS + 1) / 2;
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int const a = acc_slot(wt), par = acc_round(wt) & 1;
  // the scales' exponents: ex [NATOMS][T], then ew [NATOMS] (UE8M0 bytes)
  uint8_t *const ex = reinterpret_cast<uint8_t *>(rt_sm() + SGT_OFF_SC),
                 *const ew = ex + NATOMS * T;
  for (int i = threadIdx.x; i < NATOMS * T; i += 256) {
    ex[i] = (uint8_t)(__float_as_uint(ld_relaxed_f32(xsf + i)) >> 23);
  }
  if (threadIdx.x < NATOMS) {
    ew[threadIdx.x] = (uint8_t)(__float_as_uint(wsf[threadIdx.x]) >> 23);
  }
  if (threadIdx.x < SF_CHUNK / 4) { // the constant scale chunk (UE8M0 1)
    const_sf_fill(base + SGT_OFF_CSF, threadIdx.x);
  }
  __syncthreads();
  if (warp == 0) {
    if (lane == 0) { // from stage s0 (run_scaled_gemm_tile's prefill)
      load_job<128, 16384, false>(j, base, gl, -1, s0);
    }
  } else if (warp == 1 || warp == 6) {
    int const iss = warp == 1 ? 0 : 1;
    int &gi = iss == 0 ? gi0 : gi1;
    gi = __shfl_sync(0xffffffffu, gi, 0); // gemm_tile advances lane 0 only
    uint32_t const csf = tb + SGT_UE_CSF + 4 * iss;
    if (wt >= ACC_STAGES) { // the tick's previous job is drained
      mbar_wait(rt.acc_empty0 + 8 * a, (acc_round(wt) - 1) & 1);
    }
    if (issuer_lane<true>()) { // before the MMAs in this lane's tcgen05 order
      cp_sf(csf, base + SGT_OFF_CSF);
    }
    fence_before();
    __syncwarp();
    fence_after();
    issue_ue8m0<NATOMS>(base, tb, iss, gi, a, csf, ex, ew);
  } else if (warp >= 2 && warp <= 5) {
    int const row = (warp & 3) * 32 + lane;
    uint32_t const lanes = (uint32_t)((warp & 3) * 32) << 16;
    float v[TW];
    mbar_wait(rt.acc_full0 + 8 * a, par);
    fence_after();
    tmem_ld_cols_nw<TW>(tb + TW * a + lanes, v);
    if constexpr (NST >= 2) { // issuer 1 has pieces
      float v2[TW];
      tmem_ld_cols_nw<TW>(tb + TW * a + ACC_STAGES * TW + lanes, v2);
      tmem_keep_cols<TW>(v);
      tmem_keep_cols<TW>(v2);
#pragma unroll
      for (int t = 0; t < T; t++) {
        v[t] += v2[t];
      }
    } else {
      tmem_keep_cols<TW>(v);
    }
    fence_before();
    mbar_arrive(rt.acc_empty0 + 8 * a); // 128 arrivals
    switch (mode) {
      case COMBINE_SLOTS:
        tile_epilogue<COMBINE_SLOTS>(out, N, row0, split, row, v);
        break;
      case COMBINE_ADD:
        tile_epilogue<COMBINE_ADD>(out, N, row0, split, row, v);
        break;
      case COMBINE_STORE_BF16:
        tile_epilogue<COMBINE_STORE_BF16>(out, N, row0, split, row, v);
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
//           scales (0 BlockScales, 1 GateUpScales, 2 UE8M0 BlockScales)[, L
//           (skip_seq: a launch with every seq_lens <= L only counts the
//           tasks)]}
//   BUF_SLOTS  {the weight scales (a graph tensor, fp32)[, seq_lens
//              (skip_seq)]}
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
  constexpr int SKIP_LEN = param_or<PARAMS, 4, 0>();
  if constexpr (SKIP_LEN > 0) {
    static_assert(mode == COMBINE_ADD && SELF::counter >= 0,
                  "skip_seq: an add GEMM");
    // every token's seq_lens <= SKIP_LEN: the readers read nothing this launch,
    // so the task only counts itself (the buffer stays 0), as gemm_tile's
    int const *seq = buf_at<int const>(g, slot_at<BUF_SLOTS, 1>());
    if (__syncthreads_and(threadIdx.x >= T || seq[threadIdx.x] <= SKIP_LEN)) {
      if (threadIdx.x == 0) {
        cnt_add(cnt_now(g) + SELF::counter + tk.x, 1);
      }
      return;
    }
  }
  static_assert(K > 0 && K % (128 * SELF::y) == 0,
                "scaled_gemm_tile: each K part is whole 128-column tiles");
  constexpr int NATOMS = KT / SELF::y; // K tiles of the task
  static_assert(N == 128 * SELF::x,
                "scaled_gemm_tile: 128 weight rows per task");
  static_assert(mode == COMBINE_SLOTS || mode == COMBINE_ADD ||
                    mode == COMBINE_STORE || mode == COMBINE_STORE_BF16,
                "mode: Combine");
  static_assert((mode != COMBINE_STORE && mode != COMBINE_STORE_BF16) ||
                    SELF::y == 1,
                "plain stores: one K part");
  static_assert(SL == 0 || (SL == 1 && N % 256 == 0) || SL == 2,
                "the weight scales: BlockScales, GateUpScales (pieces in "
                "gate / up pairs), or BlockScales of powers of two (UE8M0)");
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
  // the activation-wait prefill (runtime.cuh): the weights' first stages before
  // the wait (not after moe_experts re-laid the ring)
  int const npre = L.ring_relaid ? 0 : prefill_stages(j.nst);
  if (threadIdx.x == 0) {
    if (npre) {
      prefill_weights<128, 16384, false>(j, L.base, L.gl, npre);
    }
    cnt_wait(cnt_now(g) + ACT::counter, ACT::x * ACT::y * ACT::z);
    asm volatile("fence.proxy.async.global;" ::: "memory"); // then TMA reads
#ifdef STATIC_TIMING_BUILD
    if (npre && L.stage_stamps) {
      L.stage_stamps[0] = gtime(); // the activation ready
    }
#endif
    if (L.ring_relaid) {
      ring_reinit(true);
    }
    if (npre) {
      prefill_acts(j, L.base, L.gl, npre);
    }
  }
  if (L.ring_relaid) {
    L.gl = L.gi0 = L.gi1 = 0;
    L.wt = 0;
  }
  L.gl += npre; // the loader continues at stage npre
  __syncthreads();
  int const k0 = tk.y * NATOMS;
  float const *const wsf =
      buf_at<float const>(g, WSF) + WS::first(tk.x, KT, k0);
  float const *const xsf =
      reinterpret_cast<float const *>(g.rv + exchange_offset[ACT::buf] +
                                      rg_set() + (size_t)T * K) +
      (size_t)k0 * T;
  uint32_t *const cnt = cnt_now(g) + (SELF::counter < 0 ? 0 : SELF::counter);
  if constexpr (SL == 2) {
    scaled_gemm_tile_ue8m0_task<NATOMS>(j,
                                        L.base,
                                        L.tb,
                                        L.gl,
                                        L.gi0,
                                        L.gi1,
                                        L.wt,
                                        mode,
                                        buf_at<void>(g, SELF::buf),
                                        N,
                                        cnt,
                                        tk.x * 128,
                                        tk.y,
                                        wsf,
                                        xsf,
                                        npre);
    L.wt += 1;
  } else {
    scaled_gemm_tile_task<NATOMS, WS>(j,
                                      L.base,
                                      L.tb,
                                      L.gl,
                                      L.gi0,
                                      L.gi1,
                                      L.wt,
                                      mode,
                                      buf_at<void>(g, SELF::buf),
                                      N,
                                      cnt,
                                      tk.x * 128,
                                      tk.y,
                                      wsf,
                                      xsf,
                                      npre);
    L.wt += PromoRounds<NATOMS>::ROUNDS;
  }
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
