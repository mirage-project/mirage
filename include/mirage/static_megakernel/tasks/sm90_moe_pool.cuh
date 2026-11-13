// tasks/sm90_moe_pool.cuh -- the sm90_moe_pool task type: GPT-OSS's routed
// experts (W13 with the clamped SwiGLU, then W2) as a pool, the MoE node of the
// other static models' kind (scaled_moe_experts): one task per SM, the SMs
// share the step's work at run time.
//
// The weights are sm90_moe_ring_pack's blocks: [E][N / PR][K tiles][PR x 68]
// (W13: PR = 128, 64 gate rows and the up rows of the same outputs; W2: PR =
// 64). A unit is UB consecutive blocks of one expert (rows [y UB PR, (y + 1) UB
// PR)): its ring stage holds NB K tiles of each of its blocks (block j's at
// [j][b]), one bulk copy per block, and every row goes through the same
// decode, MMAs (its K tiles in order, split by parity over the warp halves) and
// epilogue as an sm90_moe_ring task's, so the results do not depend on UB. The
// step picks UB per projection at run time: the big units (UB13, UB2: the
// node's params) when there are at least as many of them as SMs, else one
// block a unit (few experts: more, shorter units); with big W2 units the last
// slots' W2 still go one block a unit, so that the queue ends fine-grained.
//
// Past 8 tokens a unit's MMAs run over the tokens routed to its expert only,
// packed (n8 groups of them: a column is a token, whichever, so the sums do not
// change); a big unit whose slot has more groups than its accumulators hold
// runs as UB one-block units.
//
// The units are those of the active slots only (the gate's routed-expert mask:
// slot a = its a-th expert), W13's slot by slot, then W2's; each SM claims them
// from a queue counter, one ahead (the first before the gate is done). A W2
// unit waits for its slot's W13 units (a counter per slot); a W13 unit waits
// for nothing (the pool starts after the gate), so the claims cannot deadlock:
// an SM waiting in a W2 unit holds no W13 unit, as every W13 unit comes before
// every W2 unit in the queue.
#pragma once
#include "sm90_moe_ring.cuh"

namespace static_mk {

// a unit's ring stage: NB K tiles of each of its UB blocks of PR rows, the
// most that fit
template <int PR, int UB>
struct MoePoolStage {
  static constexpr int BLOCK = PR * 68; // one block's K tile
  static constexpr int act_off(int nb) {
    return (nb * UB * BLOCK + 1023) / 1024 * 1024;
  }
  static constexpr bool fits(int nb) {
    return act_off(nb) + nb * A_STAGE <= FSTAGE;
  }
  static constexpr int NB = fits(5)   ? 5
                            : fits(4) ? 4
                            : fits(3) ? 3
                            : fits(2) ? 2
                                      : 1;
  static constexpr int ACT_OFF = act_off(NB);
};

template <int V>
struct IC { // a compile-time int through a generic lambda
  static constexpr int value = V;
};

#ifdef STATIC_SM90
// One unit through the SM's ring: expert e's blocks [y UB, y UB + UB) of the
// projection (N rows, K columns) times the activation (amap; GLU the tokens'
// rows, W2 slot a's T rows of its [A T, K]), into out (GLU [A][T][N / 2] f16,
// W2 [T][TOPK][N] bf16 at the tokens' top-k slots). READY: whether the
// activation is there (thread 0 polls it after the first weights). Thread 0
// also claims the SM's next unit (queue -> *next; claim) once this one's first
// stages are on their way. NG: the token groups (n8 MMAs) it holds at most.
template <bool GLU,
          int PR,
          int UB,
          int NG,
          int N,
          int K,
          int TOPK,
          bool HAS_BIAS,
          bool F32_SCALES,
          class READY>
__device__ __forceinline__ void sm90_moe_unit(G const &g,
                                              KernelLocals &L,
                                              void const *amap,
                                              uint8_t const *wblocks,
                                              __nv_bfloat16 const *bias_all,
                                              void *out,
                                              int const *routing,
                                              uint32_t routed,
                                              int a,
                                              int y,
                                              int e,
                                              float limit,
                                              float alpha,
                                              READY act_ready,
                                              unsigned *queue,
                                              int *next,
                                              bool claim = true) {
  using S = MoePoolStage<PR, UB>;
  constexpr int BLOCK = S::BLOCK, NB = S::NB, ROWS = PR * UB;
  constexpr int MT = ROWS / 64, BT = PR / 64; // m64 tiles: a unit's, a block's
  constexpr int KT = (K + 127) / 128, NS = (KT + NB - 1) / NB;
  // warps 4..7's sums for warps 0..3: behind the ring when they fit there
  // with the stage counters, else in the ring once every warp is past its last
  // stage (no copy in flight)
  constexpr int XCH = 128 * MT * NG * 16;
  constexpr bool XCH_RING = XCH + 4 * (SMAX + 1) > RING_BYTES - OFF_WSF;
  constexpr int XCH_BYTES = XCH_RING ? 0 : XCH; // behind the ring
  static_assert(
      PR % 64 == 0 && N % ROWS == 0 && !(GLU && PR != 128),
      "sm90_moe_pool: blocks of 64-row tiles (GLU: 128, gate and up)");
  static_assert(OFF_WSF + XCH_BYTES + 4 * (SMAX + 1) <= RING_BYTES &&
                    XCH <= SMAX * FSTAGE && S::fits(NB) && T % 8 == 0 &&
                    NG <= T / 8,
                "sm90_moe_pool: the stage, the exchange area, the counters");
  __syncthreads(); // the SM's previous unit is done with the ring
  int *const done = reinterpret_cast<int *>(L.sm + OFF_WSF + XCH_BYTES);
  if (threadIdx.x == 0) {
    for (int i = 0; i < SMAX; i++) {
      done[i] = 0;
    }
    done[SMAX] = L.gl;
  }
  __syncthreads();
  int const c0 = done[SMAX], row0 = y * ROWS;
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // block j of the unit: expert e's block y UB + j, its K tiles contiguous
  uint8_t const *blocks =
      wblocks + (size_t)(e * (N / PR) + y * UB) * KT * BLOCK;
  auto weights = [&](int s) {
    int const c = c0 + s, st = c % SMAX;
    if (c >= SMAX) {
      mbar_wait(rt.empty0 + 8 * st, ((c / SMAX) - 1) & 1);
    }
    int const nt = min(NB, KT - s * NB);
    mbar_expect(rt.full0 + 8 * st, nt * (UB * BLOCK + A_STAGE));
#pragma unroll
    for (int j = 0; j < UB; j++) {
      bulk_g2s(blocks + ((size_t)j * KT + s * NB) * BLOCK,
               rt.full0 + 8 * st,
               L.base + st * FSTAGE + j * NB * BLOCK,
               nt * BLOCK);
    }
  };
  auto activation = [&](int s) {
    int const st = (c0 + s) % SMAX;
    for (int b = 0; b < min(NB, KT - s * NB); b++) {
      tma3(amap,
           rt.full0 + 8 * st,
           L.base + st * FSTAGE + S::ACT_OFF + b * A_STAGE,
           0,
           GLU ? 0 : a * T,
           (s * NB + b) * 2,
           EVICT_LAST);
    }
  };
  constexpr int PRE = NS < SMAX ? NS : SMAX;
  if (threadIdx.x == 0) {
    int pre = 0;
    do {
      weights(pre++);
    } while (pre < PRE && !act_ready());
    while (!act_ready()) {
      __nanosleep(200);
    }
    asm volatile("fence.proxy.async.global;" ::: "memory"); // TMA reads it
    for (int s = 0; s < PRE; s++) {
      if (s >= pre) {
        weights(s);
      }
      activation(s);
    }
    if (claim) {
      *next = atomicAdd(queue, 1u);
    }
  }
  int const g8 = lane >> 2, q = lane & 3, kpar = warp >> 2;
  int const rbase = 16 * (warp & 3) + g8; // rows rbase (+ 8) of each m64 tile
  // past 8 tokens the MMAs' columns are compact: column j is the j-th token
  // routed to e (routed: bit t; GLU gathers their rows, W2's act rows are
  // already compact), ng groups of them. At 8 tokens (one group, nothing to
  // save) column j is token j.
  constexpr bool COMPACT = T > 8;
  int const cnt = COMPACT ? __popc(routed) : T, ng = (cnt + 7) >> 3;
  auto token = [&](int j) {
    return j < cnt ? (int)__fns(routed, 0, j + 1) : 0;
  };
  int gtok[NG]; // GLU: this lane's gathered row of each group
#pragma unroll
  for (int n = 0; n < NG; n++) {
    gtok[n] = COMPACT ? token(8 * n + (lane & 7)) : 0;
  }
  float acc[MT][NG][4] = {};
  for (int s = 0; s < NS; s++) {
    int const c = c0 + s, st = c % SMAX;
    mbar_wait(rt.full0 + 8 * st, (c / SMAX) & 1);
    char const *stage = L.sm + st * FSTAGE;
#pragma unroll
    for (int b = 0; b < NB; b++) {
      int const kt = s * NB + b;
      if ((kt & 1) != kpar || kt >= KT) {
        continue;
      }
      uint32_t bf[NG][8][2];
#pragma unroll
      for (int n = 0; n < NG; n++) {
        if (n == 0 || n < ng) { // (a slot has a token)
          char const *box = stage + S::ACT_OFF + b * A_STAGE;
          if constexpr (GLU && COMPACT) {
            sm90_b_frags_k128_sw128_rows(box, gtok[n], bf[n]);
          } else {
            sm90_b_frags_k128_sw128(box + n * 1024, bf[n]);
          }
        }
      }
#pragma unroll
      for (int mt = 0; mt < MT; mt++) {
        // tile mt: block mt / BT's rows 64 (mt % BT) ..
        char const *blk = stage + ((mt / BT) * NB + b) * BLOCK;
        uint32_t r[8][4];
        float sc[2][4];
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
          int const row = 64 * (mt % BT) + rbase + 8 * rr;
          uint4 const v =
              *reinterpret_cast<uint4 const *>(blk + row * 64 + q * 16);
          uint32_t const sw =
              *reinterpret_cast<uint32_t const *>(blk + PR * 64 + row * 4);
          uint32_t const words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
          for (int i = 0; i < 4; i++) {
            uint32_t p[4];
            uint32_t const s8 = (sw >> (8 * i)) & 0xffu;
            if constexpr (F32_SCALES) {
              moe_decode_word<false>(words[i], 0u, p);
              sc[rr][i] = moe_scale_f32(s8);
            } else {
              moe_decode_word<true>(words[i], moe_scale_f16x2(s8), p);
            }
#pragma unroll
            for (int cc = 0; cc < 4; cc++) {
              r[2 * i + (cc >> 1)][rr + 2 * (cc & 1)] = p[cc];
            }
          }
        }
#pragma unroll
        for (int n = 0; n < NG; n++) {
          if (n > 0 && n >= ng) {
            continue;
          }
          if constexpr (F32_SCALES) {
#pragma unroll
            for (int i = 0; i < 4; i++) {
              float d[4] = {};
              mma_m16n8k16(d, r[2 * i], bf[n][2 * i][0], bf[n][2 * i][1]);
              mma_m16n8k16(
                  d, r[2 * i + 1], bf[n][2 * i + 1][0], bf[n][2 * i + 1][1]);
              acc[mt][n][0] += sc[0][i] * d[0],
                  acc[mt][n][1] += sc[0][i] * d[1];
              acc[mt][n][2] += sc[1][i] * d[2],
                  acc[mt][n][3] += sc[1][i] * d[3];
            }
          } else {
#pragma unroll
            for (int kb = 0; kb < 8; kb++) {
              mma_m16n8k16(acc[mt][n], r[kb], bf[n][kb][0], bf[n][kb][1]);
            }
          }
        }
      }
    }
    __syncwarp();
    if (lane == 0) {
      __threadfence_block();
      if (atomicAdd(done + st, 1) == 7) {
        done[st] = 0;
        mbar_arrive(rt.empty0 + 8 * st);
        if (s + SMAX < NS) {
          weights(s + SMAX);
          activation(s + SMAX);
        }
      }
    }
  }
  L.gl = L.gi0 = c0 + NS;
  if constexpr (XCH_RING) {
    __syncthreads();
  }
  float4 *xacc = reinterpret_cast<float4 *>(L.sm + (XCH_RING ? 0 : OFF_WSF));
  int const xi = (warp & 3) * 32 + lane;
  if (kpar == 1) {
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
      for (int n = 0; n < NG; n++) {
        xacc[(xi * MT + mt) * NG + n] = make_float4(
            acc[mt][n][0], acc[mt][n][1], acc[mt][n][2], acc[mt][n][3]);
      }
    }
  }
  __syncthreads();
  if (kpar == 0) {
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
      for (int n = 0; n < NG; n++) {
        float4 const o = xacc[(xi * MT + mt) * NG + n];
        acc[mt][n][0] += o.x, acc[mt][n][1] += o.y, acc[mt][n][2] += o.z,
            acc[mt][n][3] += o.w;
      }
    }
    __nv_bfloat16 const *bias = HAS_BIAS ? bias_all + (size_t)e * N : nullptr;
#pragma unroll
    for (int n = 0; n < NG; n++) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int const col = 8 * n + 2 * q + (i & 1); // (compact) column
        if (COMPACT && col >= cnt) {
          continue;
        }
        if constexpr (GLU) { // block j: gate tile 2 j, up tile 2 j + 1
          constexpr int I = N / 2;
#pragma unroll
          for (int j = 0; j < UB; j++) {
            int const o = (y * UB + j) * 64 + rbase + 8 * (i >> 1);
            float gate = acc[2 * j][n][i], up = acc[2 * j + 1][n][i];
            if constexpr (HAS_BIAS) {
              gate += __bfloat162float(bias[o]);
              up += __bfloat162float(bias[I + o]);
            }
            gate = fminf(gate, limit);
            up = fminf(fmaxf(up, -limit), limit);
            static_cast<__half *>(out)[((size_t)a * T + col) * I + o] =
                __float2half((up + 1.0f) * gate /
                             (1.0f + __expf(-gate * alpha)));
          }
        } else {
          int const tok = COMPACT ? token(col) : col,
                    slot = routing[e * T + tok] - 1;
          if (slot >= 0) {
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
              int const row = row0 + 64 * mt + rbase + 8 * (i >> 1);
              float v = acc[mt][n][i];
              if constexpr (HAS_BIAS) {
                v += __bfloat162float(bias[row]);
              }
              static_cast<__nv_bfloat16 *>(
                  out)[((size_t)tok * TOPK + slot) * N + row] =
                  __float2bfloat16(v);
            }
          }
        }
      }
    }
  }
}
#endif // STATIC_SM90

// PARAMS {N13 (W13's rows: gate and up of I outputs, 2 I), H (W13's K, W2's
// rows), I (W2's K), E, TOPK, A (slots), HAS_BIAS, limit (fp32 bits), alpha
// (fp32 bits), F32_SCALES, UB13, UB2 (the big units' blocks: W13 128-row
// blocks, W2 64-row blocks; 1: always one)}; BUF_SLOTS {the active-expert list
// [E + 1] i32 (its producer, the gate: its routed-expert mask's counter lines
// give the experts), routing indices [E, T] i32, W13 blocks, W13 bias [E, N13]
// bf16 (-1: none), W2 blocks, W2 bias [E, H] (-1: none), act [A, T, I] f16
// (W13's output, W2's input), output [T, TOPK, H] bf16}; MAP_SLOTS {W13's
// activation [T, H] f16, act [A T, I]}; counters {the tasks (for the
// consumers), the queue (its own line: 32), W13's units done per slot (64 +
// a)}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_moe_pool(Maps const &maps,
                                                  G const &g,
                                                  KernelLocals &L,
                                                  StaticTask const &) {
#ifndef STATIC_SM90
  static_assert(sizeof(SELF) == 0, "sm90_moe_pool: Hopper only");
#else
  constexpr int N13 = PARAMS::v[0], H = PARAMS::v[1], I = PARAMS::v[2],
                E = PARAMS::v[3], TOPK = PARAMS::v[4], A = PARAMS::v[5];
  constexpr bool HAS_BIAS = PARAMS::v[6] == 1, F32_SCALES = PARAMS::v[9] == 1;
  // a unit holds at most 16 accumulator tiles (m64 tiles x token groups: 64
  // floats a thread; 8: 20B at 32 requests 7.42 -> 7.98 ms a step, 24 or 32:
  // 7.71): a big unit's slot with more token groups (tokens routed to the
  // expert / 8) than fit runs it a block at a time
  constexpr int NG = T / 8, UB13 = PARAMS::v[10], UB2 = PARAMS::v[11];
  constexpr int TL = 16;
  constexpr int NG13 = TL / (2 * UB13) < NG ? TL / (2 * UB13) : NG,
                NG2 = TL / UB2 < NG ? TL / UB2 : NG;
  static_assert(NG13 >= 1 && NG2 >= 1,
                "sm90_moe_pool: big units of <= 4 blocks");
  constexpr int B13 = N13 / 128, B2 = H / 64; // blocks a slot
  constexpr int QUEUE = SELF::counter + 32, SLOT = SELF::counter + 64;
  static_assert(N13 == 2 * I && N13 % 128 == 0 && H % 64 == 0 && A <= E &&
                    B13 % PARAMS::v[10] == 0 && B2 % PARAMS::v[11] == 0,
                "sm90_moe_pool: W13 128-row blocks of gate and up, W2 64-row "
                "blocks, whole big units");
  float const limit = __int_as_float(PARAMS::v[7]),
              alpha = __int_as_float(PARAMS::v[8]);
  __shared__ int expert[A], claim[2], n_active;
  __shared__ uint32_t routed[A]; // slot a's tokens (bit t)
  // every input's producer (the gate: the expert mask, the routing, W13's
  // activation) done
  __syncthreads();
  if (threadIdx.x == 0) {
    // the first claim needs nothing of the gate: off the path after it
    claim[0] = atomicAdd(g.cnt + QUEUE, 1u);
    sm90_wait_inputs<~0u, IN...>(g);
    using LIST = typename sm90_nth<3, IN...>::type;
    int n = 0;
    if constexpr (LIST::x > 0 && LIST::counter >= 0) {
      for (int j = 0; j < E / 32; j++) { // the experts ascending
        uint32_t m = __ldcg(g.cnt + LIST::counter + 2 + j);
        for (; m && n < A; m &= m - 1) {
          expert[n++] = 32 * j + __ffs(m) - 1;
        }
      }
    } else { // a list that is a graph input
      int const *active = buf_at<int const>(g, slot_at<BUF_SLOTS, 0>());
      for (; n < active[E] && n < A; n++) {
        expert[n] = active[n];
      }
    }
    n_active = n;
  }
  __syncthreads();
  int const n = n_active;
  int const *routing = buf_at<int const>(g, slot_at<BUF_SLOTS, 1>());
  int const lane = threadIdx.x & 31;
  for (int a = threadIdx.x >> 5; T > 8 && a < n; a += SM90_THREADS / 32) {
    uint32_t const m = __ballot_sync(
        ~0u, lane < T && __ldcg(routing + expert[a] * T + lane) > 0);
    if (lane == 0) {
      routed[a] = m;
    }
  }
  __syncthreads();
  // the big units when every SM gets one (the grid: one task per SM)
  bool const big13 = UB13 > 1 && n * (B13 / UB13) >= (int)SELF::x;
  bool const big2 = UB2 > 1 && n * (B2 / UB2) >= (int)SELF::x;
  int const y13 = big13 ? B13 / UB13 : B13, y2 = big2 ? B2 / UB2 : B2;
  // the queue ends in small units (the last slots' W2 one block a unit, about
  // an SM's worth: the SMs finish together). Without big W2 units every slot's
  // W2 is one block a unit.
  int const tail = big2 ? min(n, ((int)SELF::x + B2 - 1) / B2) : n;
  int const w13_units = n * y13, w2_big = (n - tail) * y2;
  int const units = w13_units + w2_big + tail * B2;

  uint8_t const *w13p = buf_at<uint8_t const>(g, slot_at<BUF_SLOTS, 2>());
  __nv_bfloat16 const *b13 =
      HAS_BIAS ? buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 3>())
               : nullptr;
  uint8_t const *w2 = buf_at<uint8_t const>(g, slot_at<BUF_SLOTS, 4>());
  __nv_bfloat16 const *b2 =
      HAS_BIAS ? buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 5>())
               : nullptr;
  void *act = buf_at<void>(g, slot_at<BUF_SLOTS, 6>());
  void *out = buf_at<void>(g, slot_at<BUF_SLOTS, 7>());
  void const *map13 = &maps.m[slot_at<MAP_SLOTS, 0>()];
  void const *map2 = &maps.m[slot_at<MAP_SLOTS, 1>()];
  for (int k = 0;; k++) {
    int const u = claim[k & 1];
    if (u >= units) {
      break;
    }
    int *const next = claim + ((k + 1) & 1);
    bool const w13 = u < w13_units;
    int const v = u - w13_units;
    bool const big = w13 ? big13 : big2 && v < w2_big; // else one block
    int const a = w13   ? u / y13
                  : big ? v / y2
                        : n - tail + (v - w2_big) / B2,
              y = w13   ? u % y13
                  : big ? v % y2
                        : (v - w2_big) % B2;
    int const e = expert[a];
    // the slot's token groups (8 tokens: one)
    int const ng = T > 8 ? (__popc(routed[a]) + 7) >> 3 : 1;
    if (w13) {
      auto ready = [] { return true; };
      auto unit = [&](auto ub, auto ngc, int yy, bool cl) {
        sm90_moe_unit<true,
                      128,
                      ub.value,
                      ngc.value,
                      N13,
                      H,
                      TOPK,
                      HAS_BIAS,
                      F32_SCALES>(g,
                                  L,
                                  map13,
                                  w13p,
                                  b13,
                                  act,
                                  routing,
                                  T > 8 ? routed[a] : 0u,
                                  a,
                                  yy,
                                  e,
                                  limit,
                                  alpha,
                                  ready,
                                  g.cnt + QUEUE,
                                  next,
                                  cl);
      };
      if (!big) {
        unit(IC<1>{}, IC<NG>{}, y, true);
      } else if (ng <= NG13) {
        unit(IC<UB13>{}, IC<NG13>{}, y, true);
      } else {
        for (int j = 0; j < UB13; j++) {
          unit(IC<1>{}, IC<NG>{}, y * UB13 + j, j == 0);
        }
      }
      __syncthreads(); // its act rows written
      if (threadIdx.x == 0) {
        cnt_add(g.cnt + SLOT + a, 1); // release: the rows before the count
      }
    } else {
      auto ready = [&] { return cnt_ld(g.cnt + SLOT + a) >= (uint32_t)y13; };
      auto unit = [&](auto ub, auto ngc, int yy, bool cl) {
        sm90_moe_unit<false,
                      64,
                      ub.value,
                      ngc.value,
                      H,
                      I,
                      TOPK,
                      HAS_BIAS,
                      F32_SCALES>(g,
                                  L,
                                  map2,
                                  w2,
                                  b2,
                                  out,
                                  routing,
                                  T > 8 ? routed[a] : 0u,
                                  a,
                                  yy,
                                  e,
                                  limit,
                                  alpha,
                                  ready,
                                  g.cnt + QUEUE,
                                  next,
                                  cl);
      };
      if (!big) {
        unit(IC<1>{}, IC<NG>{}, y, true);
      } else if (ng <= NG2) {
        unit(IC<UB2>{}, IC<NG2>{}, y, true);
      } else {
        for (int j = 0; j < UB2; j++) {
          unit(IC<1>{}, IC<NG>{}, y * UB2 + j, j == 0);
        }
      }
      __syncthreads(); // claim[] written before its next read
    }
  }
  sm90_task_end<SELF>(g);
#endif
}

SM90_TASK_SMEM(sm90_moe_pool, RING_BYTES)

} // namespace static_mk
