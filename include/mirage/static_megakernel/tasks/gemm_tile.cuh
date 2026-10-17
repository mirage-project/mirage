// tasks/gemm_tile.cuh -- one bf16 GEMM tile: out[t][row0 + r] = sum over K
// tiles [k0, k0 + nk) of weight[row0 + r][k] * act[t][k], for r < ROWS (128, or
// 64) and the T tokens, streamed through the ring (runtime.cuh): the gemm_tile
// task type (gemm_tile_layer; run_gemm_tile below). `mode` (Combine, core.cuh)
// says how the K parts of the result are combined in the node's output buffer.
// The activation comes through its tensor map (after its producer counted all
// its tasks), or, polled (`polled`), from its producer's 0xFF-prefilled bf16
// buffer straight into the ring's stages, each 16 B as soon as it is written.
#pragma once
#include "../core.cuh"

namespace static_mk {

// a bf16 job: weight rows [row0, row0 + ROWS), K tiles [k0, k0 + nk) (the rows
// come from wmap's box); amap = the activation (nullptr: already in the stages)
__device__ __forceinline__ TileJob make_bf16_tile(CUtensorMap const *wmap,
                                                  CUtensorMap const *amap,
                                                  int row0,
                                                  int k0,
                                                  int nk) {
  TileJob j = {};
  j.kind = 0;
  j.nst = nk;
  j.row0 = row0;
  j.k0 = k0;
  j.wmap = wmap;
  j.amap = amap;
  return j;
}

template <int MODE>
__device__ __forceinline__ void tile_epilogue(
    void *out, int N, int row0, int split, int row, float const *v) {
  if (MODE == COMBINE_STORE) {
    for (int t = 0; t < T; t++) {
      reinterpret_cast<float *>(out)[t * N + row0 + row] = v[t];
    }
  } else if (MODE == COMBINE_SLOTS) {
    for (int t = 0; t < T; t++) {
      reinterpret_cast<float *>(out)[((size_t)split * T + t) * N + row0 + row] =
          v[t];
    }
  } else {
    for (int t = 0; t < T; t++) {
      red_add_fx(reinterpret_cast<unsigned long long *>(out) + t * N + row0 +
                     row,
                 v[t]);
    }
  }
}

// The whole CTA runs one tile (all 256 threads enter; roles by warp). `mode`,
// out, N and cnt are run-time arguments so that the streaming code exists once
// in the binary: as template parameters they would give one copy of the body
// per node, and each copy is fetched cold into the instruction cache the first
// time it runs. out = the node's output buffer, N its row length (the weight's
// rows), cnt = the node's counters (COMBINE_ADD, COMBINE_STORE). wt = tiles run
// so far on this SM since the last reset: accumulator stage wt % 4, parity (wt
// / 4) % 2. ROWS = the tile's weight rows (128, or 64: MMA M = 64, wmap has a
// 64-row box; the epilogue threads without a row only arrive). POLLED: polled =
// the activation [T][act_k] bf16, polled into the stages by all threads (j.amap
// == nullptr; the ring was just reset: the job's stages are slots 0 .. j.nst -
// 1); sm = the shared memory (generic address of base). Not POLLED: the
// activation comes through j.amap (polled, act_k, sm unused). Returns after the
// epilogue published the result (warps 2..5) or after the last stage was armed
// / issued (the others).
template <int ROWS = 128, bool POLLED = false>
__device__ __forceinline__ void gemm_tile_task(TileJob const &j,
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
                                               __nv_bfloat16 const *polled,
                                               int act_k,
                                               char *sm STAGE_STAMP_PARAM) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int const a = wt & 3, par = (wt >> 2) & 1;
  if constexpr (POLLED) {
    if (warp == 0 && lane == 0) {
      load_job<ROWS>(j,
                     base,
                     gl,
                     -1,
                     0,
                     1 << 20 STAGE_STAMP_ARG(
                         stamp)); // timing build: stamp[0] first load issued
    }
    // the weight boxes load meanwhile. Per token, j.nst K tiles x 16 vectors of
    // 8 columns (each written whole by one 16-B store of the producer) -> the
    // stages. Vector cg of token t: stage cg / 16 (K tile j.k0 + cg / 16),
    // 64-column half (cg / 8) % 2, 16-B chunk cg % 8 of row t; 128-B swizzle:
    // chunk c ^ (t % 8)
    int const c0 = j.k0 * 128;
    for (int v = threadIdx.x; v < T * j.nst * 16; v += 256) {
      int const t = v / (j.nst * 16), cg = v % (j.nst * 16), c = c0 + cg * 8;
      uint4 x;
      do {
        x = ld16_poll(reinterpret_cast<unsigned char const *>(
            polled + (size_t)t * act_k + c));
      } while (!valid16(x));
      int const s_ = cg >> 4, a_ = (cg >> 3) & 1, k_ = cg & 7;
      *reinterpret_cast<uint4 *>(sm + s_ * FSTAGE + W_STAGE + a_ * 1024 +
                                 t * 128 + ((k_ ^ (t & 7)) << 4)) = x;
    }
    asm volatile("fence.proxy.async.shared::cta;" ::
                     : "memory"); // these stores -> visible to the tensor core
    __syncthreads();
  }
  if (warp == 0) {
    if constexpr (!POLLED) {
      if (lane == 0) {
        load_job<ROWS>(j,
                       base,
                       gl,
                       -1,
                       0,
                       1 << 20 STAGE_STAMP_ARG(
                           stamp)); // timing build: stamp[0] first load issued
      }
    }
  } else if (warp == 1 || warp == 6) {
    if (lane == 0) {
      int &gi = (warp == 1) ? gi0 : gi1;
      if (wt >= 4) {
        mbar_wait(rt.acc_empty0 + 8 * a,
                  ((wt >> 2) - 1) & 1); // the stage's previous tile is drained
      }
      issue_job<ROWS>(j,
                      base,
                      tb,
                      warp == 1 ? 0 : 1,
                      gi,
                      a STAGE_STAMP_ARG((warp == 1 && stamp)
                                            ? stamp + 1
                                            : nullptr)); // stamp[1] landed
    }
  } else if (warp >= 2 && warp <= 5) {
    float v[8];
    int const row = drain_acc<ROWS>(tb, a, par, v);
    if (ROWS == 128 || row >= 0) {
      switch (mode) { // 128 rows: every thread has one
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
    }
    asm volatile(
        "bar.sync 1, 128;" ::
            : "memory"); // the 128 epilogue threads: their stores are done
    if (threadIdx.x == 64 &&
        mode != COMBINE_SLOTS) { // per 128-row block (a 64-row tile adds 1 too)
                                 // / per task
      cnt_add(cnt + (mode == COMBINE_ADD ? row0 / 128 : 0), 1);
    }
  }
}

// the ring's barriers made fresh for a GEMM task behind moe_experts on its SM
// (which re-laid the ring: L.ring_relaid), when the task waits for nothing else
// first. Out of line: one copy for all GEMM nodes instead of one inlined copy
// per node (the branch is rarely taken: a GEMM that reads only graph inputs
// normally runs before moe_experts, whose inputs it computes).
__device__ __noinline__ void ring_reinit_behind_queue() {
  ring_reinit(true);
}

// GEMM tile (x, y): weight rows [ROWS x, ROWS x + ROWS) times K part y of
// SELF::y -> the node's output buffer (SELF::buf). PARAMS: {mode (Combine), K,
// N (the weight's rows), polled (1: the activation is ACT's 0xFF-prefilled bf16
// buffer, polled into the stages; the node's rows are split over the GPUs and
// its output holds this GPU's rows only)}. MAP_SLOTS: {the weight's map
// (Maps::m; its 64-row box at the next slot), the activation's map (-1:
// polled)}. ROWS = N / SELF::x: 128, or 64 (e.g. N = 896 rows in 14 tasks).
// ACT: the node that writes the activation (none: a graph input); without
// polling its tasks are counted in its counter.
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class ACT,
          class... IN>
__device__ __forceinline__ void run_gemm_tile(Maps const &maps,
                                              G const &g,
                                              KernelLocals &L,
                                              StaticTask const &tk) {
  constexpr int mode = PARAMS::v[0], K = PARAMS::v[1], N = PARAMS::v[2],
                WMAP = slot_at<MAP_SLOTS, 0>(), AMAP = slot_at<MAP_SLOTS, 1>();
  constexpr bool POLLED = param_at<PARAMS, 3>() == 1;
  static_assert(
      K > 0 && K % (128 * SELF::y) == 0,
      "params = {mode, K, N, polled}; each K part is whole 128-column tiles");
  constexpr int pieces = K / 128 / SELF::y; // K tiles of 128 per task
  constexpr int ROWS = N / SELF::x;
  static_assert(ROWS * SELF::x == N && (ROWS == 128 || ROWS == 64),
                "128 or 64 weight rows per task");
  static_assert(mode == COMBINE_SLOTS || mode == COMBINE_ADD ||
                    mode == COMBINE_STORE,
                "mode: Combine");
  static_assert(mode != COMBINE_STORE || SELF::y == 1,
                "plain stores: one K part");
  static_assert(SELF::buf >= 0 && (mode == COMBINE_SLOTS || SELF::counter >= 0),
                "the node's output buffer and counters");
  static_assert(WMAP + (POLLED ? 0 : 1) < MAX_MAPS && AMAP < MAX_MAPS,
                "map slots");
  if constexpr (POLLED) {
    static_assert(ROWS == 128 && AMAP < 0 && ACT::buf >= 0,
                  "polled: 128-row tasks, no activation map, ACT's buffer");
    static_assert(pieces >= 2 && pieces <= SMAX,
                  "polled: 2..SMAX K tiles per task (both issuers need a "
                  "stage; the activation is "
                  "written into the stages up front)");
    // fresh ring and accumulator barriers: the job's stages must be slots 0 ..
    // pieces - 1
    __syncthreads();
    if (threadIdx.x == 0) {
      ring_reinit(true);
    }
    L.gl = L.gi0 = L.gi1 = 0;
    L.wt = 0;
    __syncthreads();
    TileJob const j = make_bf16_tile(
        &maps.m[WMAP], nullptr, tk.x * ROWS, tk.y * pieces, pieces);
    // the output: this GPU's rows only, [K parts][T][N / GPUS] (its row blocks
    // [rank N / GPUS, + N / GPUS)), not the full width: a smaller buffer, and
    // the buffers allocated after it keep their addresses
    gemm_tile_task<ROWS, true>(j,
                               L.base,
                               L.tb,
                               L.gl,
                               L.gi0,
                               L.gi1,
                               L.wt,
                               mode,
                               g.buf[SELF::buf],
                               N / GPUS,
                               g.cnt + (SELF::counter < 0 ? 0 : SELF::counter),
                               tk.x * ROWS - g.rank * (N / GPUS),
                               tk.y,
                               buf_at<__nv_bfloat16 const>(g, ACT::buf),
                               K,
                               L.sm STAGE_STAMP_ARG(L.stage_stamps));
    L.wt++;
  } else {
    TileJob const j = make_bf16_tile(&maps.m[ROWS == 64 ? WMAP + 1 : WMAP],
                                     &maps.m[AMAP],
                                     tk.x * ROWS,
                                     tk.y * pieces,
                                     pieces);
    if constexpr (ACT::x > 0) { // the activation is a node's output: wait until
                                // all its tasks are counted
      static_assert(ACT::counter >= 0,
                    "the activation's producer counts its tasks");
      __syncthreads();
      if (threadIdx.x == 0) {
        cnt_wait(g.cnt + ACT::counter, ACT::x * ACT::y * ACT::z);
        if (L.ring_relaid) {
          ring_reinit(true);
        }
      }
      if (L.ring_relaid) {
        L.gl = L.gi0 = L.gi1 = 0;
        L.wt = 0;
      }
      __syncthreads();
    } else if (L.ring_relaid) { // after moe_experts on this SM: fresh ring and
                                // accumulator barriers
      __syncthreads();
      if (threadIdx.x == 0) {
        ring_reinit_behind_queue();
      }
      L.gl = L.gi0 = L.gi1 = 0;
      L.wt = 0;
      __syncthreads();
    }
    gemm_tile_task<ROWS>(j,
                         L.base,
                         L.tb,
                         L.gl,
                         L.gi0,
                         L.gi1,
                         L.wt,
                         mode,
                         g.buf[SELF::buf],
                         N,
                         g.cnt + (SELF::counter < 0 ? 0 : SELF::counter),
                         tk.x * ROWS,
                         tk.y,
                         nullptr,
                         0,
                         L.sm STAGE_STAMP_ARG(L.stage_stamps));
    L.wt++;
  }
}

// dynamic shared memory: the ring
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_gemm_tile() {
  return RING_BYTES;
}

} // namespace static_mk
