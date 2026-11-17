// tasks/gemm_fp8_tile.cuh -- one block-FP8 GEMM tile, DeepSeek-V4.1's
// fp8_gemm (inference/kernel.py: e4m3 weights with an e8m0 scale per 32 x 32
// block, e4m3 activations with one per token and 32 columns, fp32
// accumulation): out[t][row0 + r] = sum over K tiles [kt0, kt0 + 2 nst) of the
// tile's w[row0 + r][k] sw x_q[t][k] sx, for r < 128 and the T tokens. The MMA
// applies the scales (kind::mxf8f6f4 block_scale, one scale per row and 32 K:
// per 128 x 128 tile one 512-B scale chunk, row r's scale for K 32 j .. + 31
// at byte 16 (r % 32) + 4 (r / 32) + j; static_megakernel.py
// w8_scale_chunks, act_quant.cuh). The
// gemm_fp8_tile task type (gemm_fp8_tile_layer; run_gemm_fp8_tile below).
// It streams through the ring of runtime.cuh (its barriers and stage counters,
// gemm_tile's warp roles and epilogue), a stage = 2 K tiles of 128 (Fp8Job).
// mode (Combine, core.cuh): how the K parts are combined in the node's output
// buffer, as gemm_tile's. x_q is an act_quant, q_norm or sparse_attn node's
// output (act_quant.cuh: x_q, then its scale chunks), read through two maps
// (gemm_fp8_tile_layer). Grouped (groups G > 1, wo_a: model.py L788-789,
// einsum "bsgd,grd->bsgr"): the weight's N rows in G groups of N / G, group
// g's rows times x_q's columns [g K, (g + 1) K) (x_q [T][G K]).
#pragma once
#include "../core.cuh"
#include "act_quant.cuh" // xq_exponent, xq_pack4 (fused act_quant)
#include "gemm_tile.cuh" // tile_epilogue

namespace static_mk {

// A stage (ring slot st, its 2 K tiles at = 0, 1):
//   base + st FSTAGE + 16 KB at            the weight tile: 128 rows x 128 B
//   base + st FSTAGE + W_STAGE + 1 KB at   the activation tile: T rows x 128 B
//   base + OFF_WSF + (2 st + at) SF_CHUNK  the weight tile's scale chunk
//   base + OFF_XSF + (2 st + at) SF_CHUNK  the activation tile's (x_q's)
// (128-B swizzled boxes, as moe_experts' W13 stages). TMEM: the accumulators
// as gemm_tile's (columns 0..63); per (st, at) the weight scales at column 232
// + 4 (2 st + at) (as moe_experts' W13 weight scales), the activation scales
// at 272 + 4 (2 st + at): a scale copy never overwrites the slot of a tile
// whose MMAs may still run. 272.. are also moe_experts' W2 weight scales: on
// one SM the two never overlap in time (each starts behind a __syncthreads,
// which the other's epilogue reaches only after its last MMAs are done).
constexpr int OFF_XSF = RING_BYTES; // SMAX x 2 activation scale chunks
constexpr int FP8_SMEM = OFF_XSF + SMAX * 2 * SF_CHUNK;
constexpr uint32_t FP8_STAGE_TX =
    2 * (16384 + SF_CHUNK + T * 128 + SF_CHUNK); // the bytes of one stage
constexpr uint32_t FP8_W_TX =
    2 * (16384 + SF_CHUNK); // a stage's weight bytes alone (x_q made in the
                            // task: fused act_quant, fp8_fused_act)
constexpr int TM_WSF = 232, TM_XSF = TM_WSF + 4 * 2 * SMAX;
static_assert(TM_XSF == TM_WSF + 4 * 2 * SMAX && TM_XSF + 4 * 2 * SMAX <= 512,
              "gemm_fp8_tile: the TMEM columns of the scales");

__device__ __forceinline__ uint32_t idesc_mx8(uint32_t sf) {
  return idesc_mx(IDESC_MX_E4M3, sf);
} // scale id sf of 0..3 (the K 32 of the MMA's K 128 tile)

// a job: weight rows [128 rb, + 128), K tiles [kt0, kt0 + 2 nst); each map's
// box holds 2 K tiles (2 scale chunks)
struct Fp8Job {
  CUtensorMap const *wmap; // the weight: box {128 B, 128 rows, 2 K tiles}
  CUtensorMap const *wsf;  // its scale chunks [N / 128][K / 128], box 2
  CUtensorMap const *amap; // x_q: box {128 B, T rows, 2 K tiles}
  CUtensorMap const *asf;  // x_q's scale chunks [K / 128], box 2
  // weight row block, first K tile, stages, K / 128 (the weight's scale chunks
  // per row block), x_q's K tile of the weight's K tile 0 (the row block's
  // group g: g K / 128; 0 ungrouped)
  int rb, kt0, nst, ktall, akt;
};

// the tensor map's box at (c0, c1, c2) into L2 only: no shared memory, no
// barrier (TMA prefetch), cache hint `hint`
__device__ __forceinline__ void
    tma3_l2(void const *d, int c0, int c1, int c2, uint64_t hint) {
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint"
               " [%0, {%1, %2, %3}], %4;" ::"l"((uint64_t)d),
               "r"(c0),
               "r"(c1),
               "r"(c2),
               "l"(hint)
               : "memory");
}

// timing build: what the 2 stage stamps record. FP8_STAMP 1: [0] the job's
// first stage issued, [1] its first stage landed; 2: [0] the loader saw x_q
// ready, [1] the last stage landed; 3: [0] the last stage landed, [1] the
// epilogue has the accumulator
#ifndef FP8_STAMP
#define FP8_STAMP 1
#endif
#ifndef FP8_POLL_NS // the loader's sleep between reads of x_q's counter
#define FP8_POLL_NS STATIC_CNT_POLL_NS
#endif
// loader (warp 0, lane 0): arm the job's stages. The weight tiles do not
// depend on x_q (loads that need no producer go before the wait): the
// first min(nst, SMAX) stages' weight tiles and their scale chunks are issued
// first, then the loader waits for x_q (act_cnt reaching act_target; the fence:
// the producer's generic stores -> these TMA reads), issues those stages' x_q
// tiles, and arms the other stages as before (one wait, one expect, 4 TMA
// issues each). A stage's `full` expects all its bytes up front. The stages
// the ring cannot take before x_q is ready (s >= min(nst, SMAX)): their weight
// tiles and scale chunks first prefetched into L2 (no shared memory, no
// barrier), so that their TMA loads come from L2. act_cnt == nullptr (fused
// act_quant): the weights only, no wait (the task's warps 2..5 make x_q in
// shared memory, fp8_fused_act)
__device__ __forceinline__ void
    load_fp8_job(Fp8Job const &j,
                 uint32_t base,
                 int &g,
                 uint32_t const *act_cnt,
                 uint32_t act_target,
                 uint32_t const *wait0 STAGE_STAMP_PARAM) {
  int const npre = j.nst < SMAX ? j.nst : SMAX;
  for (int s = npre; s < j.nst; s++) {
    int const kt = j.kt0 + 2 * s;
    tma3_l2(j.wmap, 0, j.rb * 128, kt, EVICT_LAST);
    tma3_l2(j.wsf, 0, 0, j.rb * j.ktall + kt, EVICT_LAST);
  }
  if (wait0) { // weights_after: its node has finished a task
    while (cnt_ld(wait0) == 0) {
      poll_pause(FP8_POLL_NS);
    }
  }
  int const g0 = g;
  int st = g % SMAX;
  uint32_t ph = ((g / SMAX) - 1) & 1; // `empty` parity, advanced with g
  uint32_t const tx = act_cnt ? FP8_STAGE_TX : FP8_W_TX;
  auto weights = [&](int s, int st_) {
    uint32_t const full = rt.full0 + 8 * st_;
    int const kt = j.kt0 + 2 * s;
    tma3(j.wmap, full, base + st_ * FSTAGE, 0, j.rb * 128, kt, EVICT_FIRST);
    tma3(j.wsf,
         full,
         base + OFF_WSF + st_ * 2 * SF_CHUNK,
         0,
         0,
         j.rb * j.ktall + kt,
         EVICT_FIRST);
  };
  auto activations = [&](int s, int st_) {
    uint32_t const full = rt.full0 + 8 * st_;
    int const kt = j.kt0 + 2 * s;
    tma3(j.amap,
         full,
         base + st_ * FSTAGE + W_STAGE,
         0,
         0,
         j.akt + kt,
         EVICT_LAST);
    tma3(j.asf,
         full,
         base + OFF_XSF + st_ * 2 * SF_CHUNK,
         0,
         0,
         j.akt + kt,
         EVICT_LAST);
  };
  for (int s = 0; s < npre; s++, g++) { // the weights ahead of x_q
    if (g >= SMAX) {
      mbar_wait(rt.empty0 + 8 * st, ph);
    }
    STAGE_STAMP(FP8_STAMP == 1 && s == 0)
    mbar_expect(rt.full0 + 8 * st, tx);
    weights(s, st);
    if (++st == SMAX) {
      st = 0;
      ph ^= 1u;
    }
  }
  if (act_cnt) {
    while (cnt_ld(act_cnt) < act_target) { // the poll interval: FP8_POLL_NS
      poll_pause(FP8_POLL_NS);
    }
    STAGE_STAMP(FP8_STAMP == 2)
    asm volatile("fence.proxy.async.global;" ::: "memory");
    for (int s = 0; s < npre; s++) {
      activations(s, (g0 + s) % SMAX);
    }
  }
  for (int s = npre; s < j.nst; s++, g++) {
    if (g >= SMAX) {
      mbar_wait(rt.empty0 + 8 * st, ph);
    }
    STAGE_STAMP(FP8_STAMP == 1 && s == 0)
    mbar_expect(rt.full0 + 8 * st, tx);
    weights(s, st);
    if (act_cnt) {
      activations(s, st);
    }
    if (++st == SMAX) {
      st = 0;
      ph ^= 1u;
    }
  }
}

// issuer iss (0 or 1, lane 0): the stages with s % 2 == iss into accumulator
// stage a; both issuers wait on EVERY stage in order (the parity rule,
// runtime.cuh). Per K tile: both scale chunks -> TMEM, then 4 MMAs of K 32
// (scale id k). timing build: *stamp = when the job's first stage has landed
__device__ __forceinline__ int issue_fp8_job(Fp8Job const &j,
                                             uint32_t base,
                                             uint32_t tb,
                                             int iss,
                                             int g,
                                             int a STAGE_STAMP_PARAM) {
  uint32_t const acc = tb + 8 * a + 32 * iss;
  bool first = true;
  for (int s = 0; s < j.nst; s++, g++) {
    int const st = g % SMAX;
    mbar_wait(rt.full0 + 8 * st, (g / SMAX) & 1);
    fence_after();
    STAGE_STAMP(FP8_STAMP == 1 ? s == 0 : s == j.nst - 1)
    if ((s & 1) != iss) { // seen: the loader re-arms the slot only after both
                          // issuers passed it
      mbar_arrive(rt.empty0 + 8 * st);
      continue;
    }
    for (int at = 0; at < 2; at++) {
      int const c = 2 * st + at;
      uint32_t const sfa = tb + TM_WSF + 4 * c, sfb = tb + TM_XSF + 4 * c;
      cp_sf(sfa, base + OFF_WSF + c * SF_CHUNK);
      cp_sf(sfb, base + OFF_XSF + c * SF_CHUNK);
      uint64_t const dw = mkdesc(base + st * FSTAGE + at * 16384),
                     dx = mkdesc(base + st * FSTAGE + W_STAGE + at * 1024);
      for (int k = 0; k < 4; k++) {
        mma_mx(acc,
               dw + 2 * k,
               dx + 2 * k,
               idesc_mx8(k),
               (first && k == 0) ? 0u : 1u,
               sfa,
               sfb);
      }
      first = false;
    }
    tc_commit(rt.empty0 + 8 * st); // `empty` count 2: these MMAs done + the
                                   // other issuer's arrive
  }
  tc_commit(rt.acc_full0 + 8 * a); // `acc_full` count 2: both issuers own a
                                   // stage (nst >= 2)
  return g;
}

// The whole CTA runs one tile, as gemm_tile_task (all 256 threads enter; roles
// by warp; mode, out, N and cnt run-time arguments, so that the code exists
// once for all nodes): warp 0 loads, warps 1 and 6 issue, warps 2..5 drain
// accumulator stage wt % 4 (parity (wt / 4) % 2) and store row `row` of the
// tile, then count the task (mode != COMBINE_SLOTS); act_cnt / act_target:
// x_q's readiness (load_fp8_job). Returns after the epilogue published the
// result (warps 2..5) or after the last stage was armed / issued (the others).
// rg: the calling warp's ring stage counter (warp 0: the loader's, 1 and 6:
// their issuer's), returned advanced. One copy of the code for every
// gemm_fp8_tile node (noinline): a node's first task on an SM then runs code
// another GEMM node already fetched into the instruction cache
__device__ __noinline__ int
    gemm_fp8_task(Fp8Job const j,
                  uint32_t base,
                  uint32_t tb,
                  int rg,
                  int wt,
                  int mode,
                  void *out,
                  int N,
                  uint32_t *cnt,
                  int split,
                  uint32_t const *act_cnt,
                  uint32_t act_target,
                  uint32_t const *wait0 STAGE_STAMP_PARAM) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int const a = wt & 3, par = (wt >> 2) & 1;
  if (warp == 0) {
    if (lane == 0) {
      load_fp8_job(
          j, base, rg, act_cnt, act_target, wait0 STAGE_STAMP_ARG(stamp));
    }
  } else if (warp == 1 || warp == 6) {
    if (lane == 0) {
      if (wt >= 4) {
        mbar_wait(rt.acc_empty0 + 8 * a,
                  ((wt >> 2) - 1) & 1); // the stage's previous tile is drained
      }
      rg =
          issue_fp8_job(j,
                        base,
                        tb,
                        warp == 1 ? 0 : 1,
                        rg,
                        a STAGE_STAMP_ARG((warp == 1 && stamp)
                                              ? stamp + (FP8_STAMP <= 2 ? 1 : 0)
                                              : nullptr));
    }
  } else if (warp >= 2 && warp <= 5) {
    float v[8];
    int const row = drain_acc<128>(tb, a, par, v);
#ifdef STATIC_TIMING_BUILD
    if (FP8_STAMP == 3 && threadIdx.x == 64 && stamp) {
      stamp[1] = gtime();
    }
#endif
    int const row0 = j.rb * 128;
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
      cnt_add(cnt + (mode == COMBINE_ADD ? j.rb : 0), 1);
    }
  }
  return rg;
}

// fused act_quant (wo_b after wo_a: model.py L790, wo_b's linear quantizes its
// input; the act_quant node between them removed): the CTA's 256 threads make
// the job's x_q columns [c0, c0 + 256 nst) from ACT's P K parts (fp32
// [P][T][KA], 0xFF before the launch: polled) as act_quant_parts_task does:
// the parts added in part order -> bf16 -> per 32 columns ex =
// xq_exponent(amax), e4m3 (xq_pack4) -> straight into the job's ring stages,
// where the TMA would land them (stage s in slot (g0 + s) % SMAX: K tile at
// of the stage T rows of 128 B at its activation tile + 1024 at, 128-B
// swizzle: 16-B chunk q of row t at chunk q ^ t; its scale chunk at OFF_XSF
// as act_quant's, line t row t's 4 scale bytes, lines t + 8, + 16, + 24
// zeros). The job's loader loads only the weights, so these bytes are the
// task's; their last readers were the SM's earlier MMAs, all done once warps
// 2..5 drained the previous tile (before barrier 4, which every writer passes
// first: run_gemm_fp8_tile); the next job's loader refills a slot only
// after this job's MMAs on it (`empty`). Round r (nst rounds): unit u = 256 r +
// threadIdx.x = token u / (32 nst), columns 8 (u % (32 nst)) .. + 7 of the job
// (4 threads a 32-column group, 16 a 128-column K tile, as
// act_quant_parts_task): 2 P float4 in flight per thread (not all 8 tokens'
// loads at once: the layer kernel's registers are all used, and a spill slows
// every task type). Every thread polls its own loads (STATIC_POLL_NS apart).
// Ends with the async-proxy fence (the MMAs read the area). No dry pass: the
// tasks that start after wo_a find its parts there already
template <int P, int KA>
__device__ __noinline__ void
    fp8_fused_act(float const *part, int c0, int nst, uint32_t base, int g0) {
  static_assert(T == 8 && KA % 256 == 0,
                "fused act_quant: 8 tokens, 256 threads, whole stages");
  int const i = threadIdx.x, g8 = 32 * nst; // 8-column units of a token
#pragma unroll 1
  for (int r = 0; r < nst; r++) {
    int const u = 256 * r + i, t = u / g8, cg = u - t * g8, kk = cg >> 4,
              jl = cg & 15, st = (g0 + (kk >> 1)) % SMAX;
    float const *const src = part + (size_t)t * KA + c0 + 8 * cg;
    float4 v[P][2];
    bool ok[P] = {};
    bool all;
    do {
      all = true;
#pragma unroll
      for (int p = 0; p < P; p++) {
        if (!ok[p]) {
          v[p][0] = ldf4_poll(src + (size_t)p * T * KA);
          v[p][1] = ldf4_poll(src + (size_t)p * T * KA + 4);
        }
      }
#pragma unroll
      for (int p = 0; p < P; p++) {
        if (!ok[p]) {
          ok[p] = validf4(v[p][0]) && validf4(v[p][1]);
        }
        all = all && ok[p];
      }
      if (!all) {
        poll_pause(STATIC_POLL_NS);
      }
    } while (!all);
    float y[8];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      y[4 * h] = v[0][h].x;
      y[4 * h + 1] = v[0][h].y;
      y[4 * h + 2] = v[0][h].z;
      y[4 * h + 3] = v[0][h].w;
#pragma unroll
      for (int p = 1; p < P; p++) { // in part order (add_parts8)
        y[4 * h] += v[p][h].x;
        y[4 * h + 1] += v[p][h].y;
        y[4 * h + 2] += v[p][h].z;
        y[4 * h + 3] += v[p][h].w;
      }
    }
    float amax = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      y[k] = __bfloat162float(__float2bfloat16_rn(y[k])); // the GEMM's bf16
      amax = fmaxf(amax, fabsf(y[k]));
    }
    for (int d = 1; d < 4; d <<= 1) { // over the 4 threads of the group
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
    }
    int const ex = xq_exponent(amax);
    uint32_t const q0 = xq_pack4(y, ex), q1 = xq_pack4(y + 4, ex);
    uint32_t const s4 = xq_scale_word<16>(ex); // 16 threads: a K tile
    // K tile kk of the job: stage kk / 2 (ring slot st), tile kk % 2; the
    // 16-B chunk (cg % 16) / 2 of row t, its 8-B half cg % 2
    uint32_t const tile = base + st * FSTAGE + W_STAGE + (kk & 1) * 1024;
    asm volatile(
        "st.shared.v2.b32 [%0], {%1, %2};" ::"r"(
            tile + t * 128 + ((((cg >> 1) & 7) ^ t) << 4) + 8 * (cg & 1)),
        "r"(q0),
        "r"(q1)
        : "memory");
    if (jl < 4) { // line t + 8 jl of the scale chunk (xq_scale_line)
      asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(
                       base + OFF_XSF + (2 * st + (kk & 1)) * SF_CHUNK +
                       (t + 8 * jl) * 16),
                   "r"(jl == 0 ? s4 : 0u),
                   "r"(0u),
                   "r"(0u),
                   "r"(0u)
                   : "memory");
    }
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// the same x_q as the SM's previous GEMM job's (fused act_quant, same node and
// columns): its nst stages' activation tiles and scale chunks copied from
// ring slots (gp + s) % SMAX to (g0 + s) % SMAX (the CTA's 256 threads, 16 B
// each, generic proxy), then the async-proxy fence. Those slots are intact:
// this job's loader fills slots g0 .. g0 + nst - 1 only, and 2 nst <= SMAX
__device__ __forceinline__ void
    fp8_fused_copy(uint32_t base, int gp, int g0, int nst) {
  for (int k = threadIdx.x; k < nst * 192; k += 256) {
    int const s = k / 192, r = k - s * 192, a = (gp + s) % SMAX,
              b = (g0 + s) % SMAX; // per stage: 2 tiles (128 x 16 B), then
                                   // 2 scale chunks (64 x 16 B)
    uint32_t const src = r < 128 ? base + a * FSTAGE + W_STAGE + 16 * r
                                 : base + OFF_XSF + 2 * a * SF_CHUNK +
                                       16 * (r - 128),
                   dst = r < 128 ? base + b * FSTAGE + W_STAGE + 16 * r
                                 : base + OFF_XSF + 2 * b * SF_CHUNK +
                                       16 * (r - 128);
    uint32_t w0, w1, w2, w3;
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(w0), "=r"(w1), "=r"(w2), "=r"(w3)
                 : "r"(src)
                 : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(dst),
                 "r"(w0),
                 "r"(w1),
                 "r"(w2),
                 "r"(w3)
                 : "memory");
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}
// the SM's last fused job: {its accumulator tile number L.wt, its node's
// ACT::buf, its first K tile, its first ring slot}; written by thread 0 after
// the job's barrier 5, read after the next fused job's barrier 4. Trusted only
// for a node's task that is not its first on the SM (warm): the SM's earlier
// fused task of this launch wrote it
__shared__ int fq_last[4];

// task tk's job (run_gemm_fp8_tile below)
template <class SELF, class PARAMS, class MAP_SLOTS>
__device__ __forceinline__ Fp8Job fp8_job_of(Maps const &maps,
                                             StaticTask const &tk) {
  constexpr int K = PARAMS::v[1], N = PARAMS::v[2], GR = PARAMS::v[3],
                FUSED = param_or0<PARAMS, 5>();
  int const q0 = tk.y * (K / 256 / SELF::y), q1 = q0 + K / 256 / SELF::y;
  CUtensorMap const *amap = nullptr, *asf = nullptr; // fused: x_q made in the
                                                     // task, no maps
  if constexpr (!FUSED) {
    amap = &maps.m[slot_at<MAP_SLOTS, 2>()];
    asf = &maps.m[slot_at<MAP_SLOTS, 3>()];
  }
  return {&maps.m[slot_at<MAP_SLOTS, 0>()],
          &maps.m[slot_at<MAP_SLOTS, 1>()],
          amap,
          asf,
          tk.x,
          2 * q0,
          q1 - q0,
          K / 128,
          tk.x / (N / GR / 128) * (K / 128)};
}

// GEMM tile (x, y): weight rows [128 x, 128 x + 128) times K part y of SELF::y
// -> the node's output buffer (SELF::buf). PARAMS: {mode (Combine), K (the
// weight's columns), N (the weight's rows), G (groups), W (weights_after: its
// node's counter + 1, optional: 0), F (fused act_quant, optional: 0)}. W: the
// loader issues the weights after that node finished a task (a timing hint,
// not a dependency: the launch's first microseconds' HBM for the nodes on the
// way to x_q). MAP_SLOTS: {the weight, its scale chunks, x_q, x_q's
// scale chunks} (Fp8Job). ACT: the node that writes x_q [T][G K], its tasks
// counted in its counter: act_quant or q_norm (params[0] = the width) or
// sparse_attn (params {H, D, ..}: the width H D). F = 1 (fused act_quant, wo_b
// after wo_a): ACT is a gemm_fp8_tile node in slots (ACT::y K parts of [T][K]
// fp32, 0xFF from the launch's resets, no reader re-arms them), the task's x_q
// columns made by the CTA in the job's ring stages (fp8_fused_act) instead of
// an act_quant node and the TMA, after the loader issued the weights (it does
// not wait); MAP_SLOTS {the weight, its scale chunks}; the issuers and the
// epilogue start after named barrier 5 (all warps). warm (a fused node is
// `dry`): not the node's first task on the SM, so fq_last is this launch's
// (fp8_fused_copy)
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class ACT,
          class... IN>
__device__ __forceinline__ void run_gemm_fp8_tile(Maps const &maps,
                                                  G const &g,
                                                  KernelLocals &L,
                                                  StaticTask const &tk,
                                                  bool warm = true) {
  constexpr int mode = PARAMS::v[0], K = PARAMS::v[1], N = PARAMS::v[2],
                GR = PARAMS::v[3],
                WAFTER = param_or0<PARAMS, 4>(), // weights_after's counter + 1
      FUSED = param_or0<PARAMS, 5>();            // act_quant in the task
  constexpr int WMAP = slot_at<MAP_SLOTS, 0>(), WSF = slot_at<MAP_SLOTS, 1>();
  static_assert(K % (256 * SELF::y) == 0 && K / 256 / SELF::y >= 2,
                "params = {mode, K, N}; each K part is whole stages of 2 K "
                "tiles, at least 2 (both issuers need one)");
  static_assert(N == 128 * SELF::x, "128 weight rows per task");
  static_assert(GR >= 1 && N % (128 * GR) == 0,
                "groups of whole 128-row blocks");
  static_assert(mode == COMBINE_SLOTS || mode == COMBINE_ADD ||
                    mode == COMBINE_STORE,
                "mode: Combine");
  static_assert(mode != COMBINE_STORE || SELF::y == 1,
                "plain stores: one K part");
  static_assert(SELF::buf >= 0 && (mode == COMBINE_SLOTS || SELF::counter >= 0),
                "the node's output buffer and counters");
  static_assert(WMAP >= 0 && WSF >= 0 && WMAP < MAX_MAPS && WSF < MAX_MAPS,
                "map slots");
  if constexpr (FUSED) {
    static_assert(GR == 1 && ACT::buf >= 0 && ACT::v[0] == COMBINE_SLOTS &&
                      ACT::v[2] == K && ACT::y >= 1 && ACT::y <= 4,
                  "fused act_quant: x from a gemm_fp8_tile node's K parts in "
                  "slots (at most 4: in registers), one group");
  } else {
    constexpr int AMAP = slot_at<MAP_SLOTS, 2>(), ASF = slot_at<MAP_SLOTS, 3>();
    static_assert(AMAP >= 0 && ASF >= 0 && AMAP < MAX_MAPS && ASF < MAX_MAPS,
                  "map slots");
    static_assert(ACT::x == T && ACT::counter >= 0 && ACT::buf >= 0 &&
                      (ACT::v[0] == GR * K || ACT::v[0] * ACT::v[1] == GR * K),
                  "x_q: an act_quant, q_norm or sparse_attn node's output of "
                  "width G K, its tasks counted");
  }
  Fp8Job const j = fp8_job_of<SELF, PARAMS, MAP_SLOTS>(maps, tk);
  void *const out = g.buf[SELF::buf];
  // the loader waits for x_q (ACT counted all its tasks) after it issued the
  // weights (load_fp8_job). No CTA barrier at the start: the ring's barriers
  // order this task's loads and MMAs after the previous GEMM's on this SM (its
  // epilogue may still drain), and a task that uses the ring area as scratch
  // (sparse_attn, topk_route) ends with one after its last shared-memory
  // access. After moe_experts on this SM: fresh ring and accumulator barriers
  if (L.ring_relaid) {
    __syncthreads();
    if (threadIdx.x == 0) {
      ring_reinit(true);
    }
    L.gl = L.gi0 = L.gi1 = 0;
    L.wt = 0;
    __syncthreads();
  }
  int const warp = threadIdx.x >> 5;
  auto task = [&]() {
    int const rg =
        gemm_fp8_task(j,
                      L.base,
                      L.tb,
                      warp == 0   ? L.gl
                      : warp == 1 ? L.gi0
                                  : L.gi1,
                      L.wt,
                      mode,
                      out,
                      N,
                      g.cnt + (SELF::counter < 0 ? 0 : SELF::counter),
                      tk.y,
                      FUSED ? nullptr : g.cnt + ACT::counter,
                      ACT::x * ACT::y * ACT::z,
                      WAFTER > 0 ? g.cnt + (WAFTER - 1)
                                 : nullptr STAGE_STAMP_ARG(L.stage_stamps));
    if (warp == 0) { // the warps' ring counters (lane 0 runs the loops)
      L.gl = rg;
    } else if (warp == 1) {
      L.gi0 = rg;
    } else if (warp == 6) {
      L.gi1 = rg;
    }
  };
  if constexpr (FUSED) {
    // the loader (warp 0) issues the job's weights first (it does not wait:
    // act_cnt nullptr), then the CTA's 8 warps make x_q in the job's stages
    // (fp8_fused_act; the issuers and the epilogue would only wait for it),
    // then the issuers (warps 1, 6) and the epilogue (warps 2..5) run the job.
    // The job's first ring slot g0 % SMAX: issuer 0's counter (lane 0 of warp
    // 1 runs its loops) as the count of warp 1's lanes below it (bar.red.popc
    // on named barrier 4: no shared memory); named barrier 5: x_q is in the
    // stages (after each thread's async-proxy fence)
    if (warp == 0) {
      task();
    }
    int const s0 = warp == 1 ? __shfl_sync(0xffffffffu, L.gi0, 0) % SMAX : 0;
    uint32_t const vote = warp == 1 && (threadIdx.x & 31) < s0;
    uint32_t g0;
    asm volatile("{\n.reg .pred p;\nsetp.ne.u32 p, %1, 0;\n"
                 "bar.red.popc.u32 %0, 4, 256, p;\n}"
                 : "=r"(g0)
                 : "r"(vote)
                 : "memory");
    // the SM's previous GEMM job was this node's with the same columns (the
    // hand plan's second wo_b task on an SM: 148 SMs, a multiple of the 4 K
    // parts): copy its x_q instead of polling and quantizing again
    static_assert(2 * (K / SELF::y / 256) <= SMAX,
                  "fused act_quant: two jobs' stages in the ring at once");
    if (warm && fq_last[0] == L.wt - 1 && fq_last[1] == ACT::buf &&
        fq_last[2] == j.kt0) {
      fp8_fused_copy(L.base, fq_last[3], g0, j.nst);
    } else {
      fp8_fused_act<ACT::y, K>(
          buf_at<float const>(g, ACT::buf), j.kt0 * 128, j.nst, L.base, g0);
    }
    asm volatile("bar.sync 5, 256;" ::: "memory");
    if (threadIdx.x == 0) {
      fq_last[0] = L.wt;
      fq_last[1] = ACT::buf;
      fq_last[2] = j.kt0;
      fq_last[3] = g0;
    }
    if (warp != 0) {
      task();
    }
    if constexpr (sizeof(BUF_SLOTS::v) / sizeof(int) > 1) {
      // the x_q tap (BUF_SLOTS {a graph tensor}: act_quant's layout, [T][K]
      // e4m3, then per 128 columns a 512-B scale chunk; what the act_quant node
      // wrote, for the checks): row block 0's tasks (one per K slice) copy
      // their stages' tiles (un-swizzled) and scale chunks from shared memory,
      // warps 0 and 7 (idle while the MMAs and the epilogue run)
      if (tk.x == 0 && (warp == 0 || warp == 7)) {
        unsigned char *const tap =
            buf_at<unsigned char>(g, slot_at<BUF_SLOTS, 0>());
        for (int k = (threadIdx.x & 31) + (warp == 7 ? 32 : 0);
             k < 2 * j.nst * 96;
             k += 64) {
          int const kk = k / 96, r = k - kk * 96,
                    st = ((int)g0 + (kk >> 1)) % SMAX, at = kk & 1,
                    kb = j.kt0 + kk; // 64 16-B chunks of the tile, 32 lines
          uint32_t const src =
              r < 64
                  ? L.base + st * FSTAGE + W_STAGE + at * 1024 +
                        (r >> 3) * 128 + (((r & 7) ^ (r >> 3)) << 4)
                  : L.base + OFF_XSF + (2 * st + at) * SF_CHUNK + 16 * (r - 64);
          unsigned char *const dst =
              r < 64
                  ? tap + (size_t)(r >> 3) * K + kb * 128 + 16 * (r & 7)
                  : tap + (size_t)T * K + (size_t)kb * SF_CHUNK + 16 * (r - 64);
          uint4 v;
          asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];"
                       : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                       : "r"(src)
                       : "memory");
          *reinterpret_cast<uint4 *>(dst) = v;
        }
      }
    }
  } else {
    task();
  }
  L.wt++;
}

// dynamic shared memory: the ring and the activation scale chunks after it
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_gemm_fp8_tile() {
  return FP8_SMEM;
}

} // namespace static_mk
