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
// the warp-MMA tile's push (gemm_tile_wmma) stages its [T][64] bf16 in the 2 KB
// below 225 KB from the base (the launch adds 1 KB of alignment and the
// kernel's static shared memory must fit too: 226 KB failed at
// cudaFuncSetAttribute), above every other task type's need: the SM's next task
// may start while the epilogue warps still push
constexpr int PUSH_STAGE_OFF = 225 * 1024 - T * 128 * 2;
// the PDL prefill (STATIC_PDL_PREFILL, the KDA layer): the first PREFILL_STAGES
// ring stages of the projection tiles on CTAs below PREFILL_SMS issued before
// the PDL wait (10-10, full K3 in SGLang: more stages or SMs slowed SGLang's
// attn_res before it more than they gained)
constexpr int PREFILL_STAGES = 3, PREFILL_SMS = 100;
// the prefill's TMAs issued by thread 64, not thread 0: thread 0's
// griddepcontrol.wait never waits for a TMA issue (10-09/10: thread 0 issued
// them and its wait came 1.9 us later)
constexpr int PREFILL_THREAD = 64;
template <int ROWS = 128, bool POLLED = false>
__device__ __forceinline__ void
    gemm_tile_task(TileJob const &j,
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
                   char *sm,
                   int s_first = 0 STAGE_STAMP_PARAM) {
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
                       s_first,
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

// warp-MMA resident tile (params[5] = 2; KDA's o_proj, 64 rows, 8 tokens):
// the weight tile whole in shared memory as in the resident path (loaded as the
// task starts); each of the 8 warps then polls its K slice of the activation
// (ACT's 0xFF-prefilled bf16 output [T][K], no counter) into the activation
// pieces and multiplies it on the tensor core with mma.sync m16n8k16 (M = 16
// weight rows, N = the 8 tokens); the warps' partial sums are added through
// shared memory and pushed as in the push epilogue. (The resident path's 96
// tcgen05 MMAs at M = 64, N = 8 took 2.2 us after its activation landed, k46.)
__device__ __forceinline__ bool valid16h(uint4 const &v) { // every bf16 written
  return __vcmpeq2(v.x, 0xFFFFFFFFu) == 0 && __vcmpeq2(v.y, 0xFFFFFFFFu) == 0 &&
         __vcmpeq2(v.z, 0xFFFFFFFFu) == 0 && __vcmpeq2(v.w, 0xFFFFFFFFu) == 0;
}
__device__ __forceinline__ void ldsm_x4(uint32_t a, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(a));
}
__device__ __forceinline__ void
    ldsm_x2(uint32_t a, uint32_t &r0, uint32_t &r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(r0), "=r"(r1)
               : "r"(a));
}
__device__ __forceinline__ void
    mma_16816(float (&c)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
template <int K>
__device__ __forceinline__ void
    gemm_tile_wmma(TileJob const &j,
                   uint32_t base,
                   char *sm,
                   int N,
                   uint32_t *cnt,
                   int row0,
                   G const *gp,
                   size_t push_off,
                   __nv_bfloat16 const *act STAGE_STAMP_PARAM) {
  constexpr int ROWS = 64, NST = K / 128, KW = K / 8, KS = KW / 16,
                VPT = KW / 8, PER = T * VPT / 32;
  static_assert(T == 8 && K % 128 == 0 && KW % 16 == 0 && (T * VPT) % 32 == 0,
                "8 tokens; each warp's K slice whole 16-column steps");
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint32_t const wbar = rt.full0, aoff = base + NST * ROWS * 256;
  if (threadIdx.x == 0) { // the weight tile: NST pieces of 128 columns, all at
                          // once on one barrier
    mbar_expect(wbar, NST * ROWS * 256);
    for (int s = 0; s < NST; s++) {
      tma3(j.wmap,
           wbar,
           base + s * ROWS * 256,
           0,
           j.row0,
           (j.k0 + s) * 2,
           EVICT_FIRST);
    }
  }
  // the warp's columns [KW warp, + KW) of every token, 16 B per poll (all the
  // lane's vectors in flight; the unwritten ones polled again), into the
  // activation pieces (128-B swizzle, as the TMA lays them)
  // pass 0 runs the code from the activation's staging to the push dry (no
  // poll, no weight wait, no stores into the weight slots, no push or count)
  // while y_rec is still being computed, so its instructions are fetched
  // before the activation lands (as the recurrence's and attn's dry passes;
  // the layer 0.2 us faster, 29.1 vs 29.3, k136); pass 1 for real. One code
  // copy: not unrolled
#pragma unroll 1
  for (int pass = 0; pass < 2; pass++) {
    bool const dry = pass == 0;
    uint4 x[PER];
#pragma unroll
    for (int i = 0; i < PER; i++) {
      x[i] = make_uint4(~0u, ~0u, ~0u, ~0u);
    }
    bool done = dry;
    while (!done) {
#pragma unroll
      for (int i = 0; i < PER; i++) {
        if (!valid16h(x[i])) {
          int const v = lane + 32 * i;
          x[i] = ld16_poll(reinterpret_cast<unsigned char const *>(
              act + (size_t)(v / VPT) * K + warp * KW + (v % VPT) * 8));
        }
      }
      done = true;
#pragma unroll
      for (int i = 0; i < PER; i++) {
        done = done && valid16h(x[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < PER; i++) {
      int const v = lane + 32 * i, t = v / VPT, c = warp * KW + (v % VPT) * 8;
      *reinterpret_cast<uint4 *>(sm + (aoff - base) + (c >> 7) * A_STAGE +
                                 ((c >> 6) & 1) * 1024 + t * 128 +
                                 ((((c >> 3) & 7) ^ (t & 7)) << 4)) = x[i];
    }
    __syncwarp();
#if defined(STATIC_TIMING_BUILD)
    if (threadIdx.x == 0 && stamp) { // stamp[0]: warp 0's activation slice in
      stamp[0] = gtime();
    }
#endif
    if (!dry) { // the weight tile landed
      mbar_wait(wbar, 0);
    }
    // k16 step kk: piece kk / 8, 64-column half (kk / 4) % 2, 16-B chunks
    // 2 (kk % 4), + 1; A (rows) by ldmatrix x4, B (tokens) by x2 (the next
    // step's fragments loaded ahead: no change, 29.8 both, k71)
    float acc[4][4] = {};
#pragma unroll 4
    for (int q = 0; q < KS; q++) {
      int const kk = warp * KS + q, kq = kk & 3;
      uint32_t const wh = base + (kk >> 3) * ROWS * 256 +
                          ((kk >> 2) & 1) * ROWS * 128,
                     ah = aoff + (kk >> 3) * A_STAGE + ((kk >> 2) & 1) * 1024;
      uint32_t b0, b1;
      {
        int const t = lane & 7, ch = 2 * kq + ((lane >> 3) & 1);
        ldsm_x2(ah + t * 128 + ((ch ^ t) << 4), b0, b1);
      }
#pragma unroll
      for (int mb = 0; mb < 4; mb++) {
        int const r = mb * 16 + (lane & 7) + ((lane >> 3) & 1) * 8,
                  ch = 2 * kq + (lane >> 4);
        uint32_t a[4];
        ldsm_x4(wh + r * 128 + ((ch ^ (r & 7)) << 4), a);
        mma_16816(acc[mb], a, b0, b1);
      }
    }
    __syncthreads(); // every warp done with the weight slots: they hold the
                     // sums
#if defined(STATIC_TIMING_BUILD)
    if (threadIdx.x == 0 && stamp) { // stamp[1]: every warp's MMAs done
      stamp[1] = gtime();
    }
#endif
    float *red = reinterpret_cast<float *>(sm); // [warp][token][row]
    {
      int const g = lane >> 2, n = 2 * (lane & 3);
#pragma unroll
      for (int mb = 0; mb < 4; mb++) {
        float *p = red + warp * T * ROWS + mb * 16 + g;
        if (!dry) { // (dry: the slots may still be landing)
          p[n * ROWS] = acc[mb][0];
          p[(n + 1) * ROWS] = acc[mb][1];
          p[n * ROWS + 8] = acc[mb][2];
          p[(n + 1) * ROWS + 8] = acc[mb][3];
        }
      }
    }
    __syncthreads();
    __nv_bfloat16 *st = reinterpret_cast<__nv_bfloat16 *>(sm + PUSH_STAGE_OFF);
    for (int o = threadIdx.x; o < T * ROWS; o += blockDim.x) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < 8; w++) {
        v += red[w * T * ROWS + o];
      }
      st[o] = __float2bfloat16(v); // [token][row], as the push epilogue's stage
    }
    __syncthreads();
    if (!dry && threadIdx.x < T * (ROWS / 8)) {
      int const t = threadIdx.x / (ROWS / 8), c = threadIdx.x % (ROWS / 8);
      push16(*gp,
             push_off + (size_t)gp->rank * ((size_t)T * N * 2) +
                 ((size_t)t * N + row0 + c * 8) * 2,
             *reinterpret_cast<uint4 const *>(st + t * ROWS + c * 8));
    }
    // these generic stores before the SM's next TMA into the same bytes
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (!dry && threadIdx.x == 0) {
      cnt_add(cnt, 1);
    }
  }
}

// the PDL prefill's loads: the first n stages' weights loaded before the
// activation's producer is done. prefill_weights claims stage slots g .. g + n
// - 1 (waiting on `empty` as load_job does), expects each stage's weight AND
// activation bytes and issues only the weight; prefill_acts, after the wait,
// issues the activations into the same stages. The issuers wait on `full` as
// usual: a stage completes when both have landed.
template <int ROWS>
__device__ __forceinline__ void
    prefill_weights(TileJob const &j, uint32_t base, int g, int n) {
  int st = g % SMAX;
  uint32_t ph = ((g / SMAX) - 1) & 1;
  for (int s = 0; s < n; s++, g++) {
    if (g >= SMAX) {
      mbar_wait(rt.empty0 + 8 * st, ph);
    }
    uint32_t const full = rt.full0 + 8 * st, dst = base + st * FSTAGE;
    mbar_expect(full, ROWS * 256 + A_STAGE);
    tma3(j.wmap, full, dst, 0, j.row0, (j.k0 + s) * 2, EVICT_FIRST);
    if (++st == SMAX) {
      st = 0;
      ph ^= 1u;
    }
  }
}
__device__ __forceinline__ void
    prefill_acts(TileJob const &j, uint32_t base, int g, int n) {
  int st = g % SMAX;
  for (int s = 0; s < n; s++) {
    tma3(j.amap,
         rt.full0 + 8 * st,
         base + st * FSTAGE + W_STAGE,
         0,
         0,
         (j.k0 + s) * 2,
         EVICT_LAST);
    if (++st == SMAX) {
      st = 0;
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
  constexpr bool WMMA =
      param_or<PARAMS, 5>() == 2; // warp-MMA tile (below; params[4] unused)
  // uneven K parts (params[6] = B > 0, params[7] = S): parts 0 .. B - 1 of S
  // K tiles each, the rest split evenly over parts B .. y - 1 (slots, ring)
  constexpr int UB = param_or<PARAMS, 6>(), US = param_or<PARAMS, 7>();
  constexpr int USMALL = UB > 0 ? (K / 128 - UB * US) / (SELF::y - UB) : 0;
  static_assert(
      UB == 0 || (mode == COMBINE_SLOTS && UB < SELF::y && K % 128 == 0 &&
                  USMALL >= 2 && UB * US + (SELF::y - UB) * USMALL == K / 128),
      "uneven K parts: slots; B big parts of S K tiles, the rest even");
  static_assert(
      K > 0 && (UB > 0 || K % (128 * SELF::y) == 0),
      "params = {mode, K, N, polled}; each K part is whole 128-column tiles");
  constexpr int pieces = K / 128 / SELF::y; // K tiles of 128 per task
  constexpr int ROWS = N / SELF::x;
  static_assert(ROWS * SELF::x == N && (ROWS == 128 || ROWS == 64),
                "128 or 64 weight rows per task");
  static_assert(mode == COMBINE_SLOTS || mode == COMBINE_ADD ||
                    mode == COMBINE_STORE || mode == COMBINE_PUSH,
                "mode: Combine");
  // push: the node's exchange buffer offset, as a compile-time constant
  // (exchange_offset is a host-side constexpr table)
  constexpr size_t PUSH_OFF =
      mode == COMBINE_PUSH ? exchange_offset[SELF::buf] : 0;
  static_assert(
      mode != COMBINE_PUSH || (WMMA && SELF::y == 1 && SELF::buf >= 0),
      "gemm_tile push: the warp-MMA tile, one K part, its exchange buffer");
  static_assert(param_or<PARAMS, 5>() == 0 || WMMA,
                "params[5]: 0, or 2 (the warp-MMA tile)");
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
                               L.sm,
                               0 STAGE_STAMP_ARG(L.stage_stamps));
    L.wt++;
  } else if constexpr (WMMA) { // (gemm_tile_wmma)
    static_assert(
        mode == COMBINE_PUSH && ROWS == 64 && SELF::y == 1 && ACT::buf >= 0 &&
            (K / 128) * ROWS * 256 + (K / 128) * A_STAGE <= PUSH_STAGE_OFF,
        "warp MMA: push, 64-row tiles, one K part, ACT's buffer");
    TileJob const j = make_bf16_tile(
        &maps.m[WMAP + 1], &maps.m[AMAP], tk.x * ROWS, 0, pieces);
    __syncthreads(); // the SM's previous task is done with the ring
    if (threadIdx.x == 0) {
      ring_reinit(true);
    }
    L.gl = L.gi0 = L.gi1 = 0;
    L.wt = 0;
    __syncthreads();
    gemm_tile_wmma<K>(j,
                      L.base,
                      L.sm,
                      N,
                      g.cnt + SELF::counter,
                      tk.x * ROWS,
                      &g,
                      PUSH_OFF,
                      buf_at<__nv_bfloat16 const>(g, ACT::buf)
                          STAGE_STAMP_ARG(L.stage_stamps));
    __syncthreads();
    if (threadIdx.x == 0) {
      ring_reinit(true);
    }
    __syncthreads();
  } else {
    TileJob const j = make_bf16_tile(
        &maps.m[ROWS == 64 ? WMAP + 1 : WMAP],
        &maps.m[AMAP],
        tk.x * ROWS,
        UB == 0 ? tk.y * pieces
                : (tk.y < UB ? tk.y * US : UB * US + (tk.y - UB) * USMALL),
        UB == 0 ? pieces : (tk.y < UB ? US : USMALL));
    int npre =
        0; // the PDL prefill: stages whose weights were issued before the wait
#ifdef STATIC_PDL_PREFILL
    if constexpr (ACT::x ==
                  0) { // this SM's first task, its first stages' weights issued
                       // in kernel_begin (prefill_gemm_tile)
      if (L.pre_n) {
        if (L.pre_node != tk.node || L.pre_x != tk.x || L.pre_y != tk.y ||
            L.gl != 0) {
          __trap();
        }
        npre = L.pre_n;
        L.pre_n = 0;
        if (threadIdx.x == 0) {
          prefill_acts(j,
                       L.base,
                       0,
                       npre); // the activation: after the wait (kernel_begin)
        }
        L.gl = npre;
        __syncthreads();
      }
    }
#endif
    if constexpr (ACT::x > 0) { // the activation is a node's output: wait until
                                // all its tasks are counted
      static_assert(ACT::counter >= 0,
                    "the activation's producer counts its tasks");
      __syncthreads();
#ifdef STATIC_PDL_PREFILL
      if (L.pre_n) { // the first stages' weights issued in kernel_begin
                     // (prefill_gemm_tile)
        if (L.pre_node != tk.node || L.pre_x != tk.x || L.pre_y != tk.y ||
            L.gl != 0 || L.ring_relaid) {
          __trap();
        }
        npre = L.pre_n;
        L.pre_n = 0;
      }
#endif
      if (threadIdx.x == 0) {
        cnt_wait(g.cnt + ACT::counter, ACT::x * ACT::y * ACT::z);
        if (L.ring_relaid) {
          ring_reinit(true);
        }
        if (npre) { // the activations of the stages whose weights were issued
                    // early (the PDL prefill)
          prefill_acts(j, L.base, L.gl, npre);
        }
      }
      L.gl += npre; // every thread: the loader continues at stage npre
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
    gemm_tile_task<ROWS, false>(j,
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
                                L.sm,
                                npre STAGE_STAMP_ARG(L.stage_stamps));
    L.wt++;
  }
}

#ifdef STATIC_PDL_PREFILL
// PDL prefill (STATIC_PDL_PREFILL): called by kernel_begin for the SM's first
// list entry, before the resets and before griddepcontrol.wait: a GEMM tile
// whose activation is a graph input (the previous kernel's output) issues its
// first stages' weights now (they do not depend on the previous kernel), up to
// the ring's SMAX stages; run_gemm_tile issues the activation into the same
// stages after the wait. Every thread records the prefill (L.pre_*); thread 0
// issues.
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class ACT,
          class... IN>
__device__ __forceinline__ void prefill_gemm_tile(Maps const &maps,
                                                  G const &g,
                                                  KernelLocals &L,
                                                  StaticTask const &tk) {
  constexpr int K = PARAMS::v[1], N = PARAMS::v[2],
                WMAP = slot_at<MAP_SLOTS, 0>(), AMAP = slot_at<MAP_SLOTS, 1>();
  constexpr bool POLLED = param_at<PARAMS, 3>() == 1;
  constexpr int VAR = param_or<PARAMS, 5>(); // 2: warp MMA
  constexpr int UB = param_or<PARAMS, 6>(), US = param_or<PARAMS, 7>();
  constexpr int USMALL = UB > 0 ? (K / 128 - UB * US) / (SELF::y - UB) : 0;
  constexpr int pieces = K / 128 / SELF::y;
  constexpr int ROWS = N / SELF::x;
  // a graph input (the previous kernel's output) or a node's output (its wait
  // in run_gemm_tile comes after the weights): the first stages' weights before
  // griddepcontrol.wait either way
  if constexpr (!POLLED && VAR == 0 && AMAP >= 0) {
    TileJob const j = make_bf16_tile(
        &maps.m[ROWS == 64 ? WMAP + 1 : WMAP],
        &maps.m[AMAP],
        tk.x * ROWS,
        UB == 0 ? tk.y * pieces
                : (tk.y < UB ? tk.y * US : UB * US + (tk.y - UB) * USMALL),
        UB == 0 ? pieces : (tk.y < UB ? US : USMALL));
    // PREFILL_STAGES: at most this many stages prefilled (10-10: the 5 stages'
    // 22 MB queued at HBM delayed the previous kernel's end ~1.8 us in the
    // one-layer test; SGLang's projection GEMM has ~12 MB of weights in flight
    // before its wait)
    constexpr int PMAX = PREFILL_STAGES < SMAX ? PREFILL_STAGES : SMAX;
    int const npre = j.nst < 1 ? 0 : (j.nst < PMAX ? j.nst : PMAX);
    // PREFILL_SMS: only CTAs below it prefill (SGLang's projection GEMM loads
    // early on 96 CTAs; 10-10: our 136 SMs' prefill slowed SGLang's attn_res
    // before us by 0.7-1.1 us in the server)
    if (threadIdx.x == PREFILL_THREAD && npre &&
        (int)blockIdx.x < PREFILL_SMS) {
      prefill_weights<ROWS>(j, L.base, 0, npre);
    }
    if ((int)blockIdx.x >= PREFILL_SMS) {
      L.pre_n =
          0; // (no prefill on this SM: run_gemm_tile loads every stage itself)
      return;
    }
    L.pre_n = npre;
    L.pre_node = tk.node;
    L.pre_x = tk.x;
    L.pre_y = tk.y;
  }
}
#endif

// dynamic shared memory: the ring
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_gemm_tile() {
  constexpr int K = PARAMS::v[1], ROWS = PARAMS::v[2] / SELF::x;
  constexpr int NST = K / 128;
  constexpr int RES = param_or<PARAMS, 5>() == 2
                          ? NST * ROWS * 256 + NST * A_STAGE
                          : 0; // warp MMA: the whole tile + the activation
  constexpr int BASE =
      PARAMS::v[0] == COMBINE_PUSH ? PUSH_STAGE_OFF + T * 128 * 2 : RING_BYTES;
  return RES > BASE ? RES : BASE;
}

} // namespace static_mk
