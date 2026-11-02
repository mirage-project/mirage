// tasks/delta_rule_decode.cuh (HIP, gfx950) -- K3 KDA's gated delta-rule recurrence in plain decode: T requests, one new
// token each (row t = request t; SGLang's fused_recurrent update, amd/notes/sglang_v0521_decode_contract.md B.3). Task
// (x, y) of grid (heads, Y, 1) = head x for the requests R y .. R y + R - 1 (R = ceil(T / Y), at most 16), one after
// another. Inputs: short_conv_decode's output (after its counter): the bf16 row [T][PROJ_N] (g at COL_G, f_a at COL_FA,
// beta at COL_BETA) and q / k / v in fp32 ([T][3 HD] after the row); W_fb [HD][128] bf16; prm = A_log [heads], dt_bias
// [HD], the output norm's weight [128] (fp32); the state pool [slots][heads][128 v][128 k] fp32 at slot[t], read and
// written in place (a padded row, slot < 0: a zero state, not written).
//   f_b[k] = bf16(f_a . W_fb[128 h + k]); gk = -5 sigmoid(exp(A_log) (f_b + dt_bias)); beta = sigmoid(b); q^ = q / |q| /
//   sqrt(128), k^ = k / |k| (eps 1e-6); S *= exp(gk) (per k); d = beta (v - S k^); S += d k^T; o = bf16(S q^); y =
//   bf16(o rsqrt(mean_v o^2 + eps) w_v sigmoid(g)) -> the node's output (0xFF-prefilled: o_proj polls it).
// Threads: thread i holds S[v = 4 (i / 8) .. + 3][k = 16 (i % 8) .. + 15]; the k sums over 8 lanes by DPP (drd_sum8). The
// next request's state is loaded while the current one is computed. f_b for the task's requests by one MFMA pass. One
// count per task, posted before the last request's state is written back.
#pragma once
#include "../core.cuh"

namespace static_mk {

// STATIC_DRD_COAL 1 (default 0): a thread's 16 state columns are 4 kq + 32 c' .. + 3 (c' < 4), not 16 kq .. + 15: one wave load /
// store instruction then covers 8 whole 128-B lines (8 rows x the kq lanes' 8 x 16 B) instead of 32 lines a quarter each (MI355X
// dec399, T 64 FLAT: the 3 pairs' 192 KB of state loads ~16 GB/s per CU, task start -> inputs landed 12 us). The state's memory
// layout and the values are unchanged; the LDS reads of k / q / the decay follow the same columns
#ifndef STATIC_DRD_COAL
#define STATIC_DRD_COAL 0
#endif
#if STATIC_DRD_COAL
#define DRD_CS 32 // (state column step between a thread's 4 float4s)
#define DRD_CL 8  // (LDS column step per 4 values)
#else
#define DRD_CS 4
#define DRD_CL 1
#endif
// STATIC_DRD_CIR_PAR (default 0): with the conv in the recurrence (CIR), q / k / v, f_a and beta polled by different threads at
// once, the conv window and weights loaded before the poll (the default: q / k / v, then the window, then f_a, then beta, each
// waiting for the one before: MI355X dec241, T 16: inputs landed +7.6 us after the projection's last task)
#ifndef STATIC_DRD_CIR_PAR
#define STATIC_DRD_CIR_PAR 0
#endif
// STATIC_DRD_CIR_EARLY (default 0; CIR with STATIC_DRD_CIR_PAR, one request per task): thread v's item - its conv window and
// weights and the first round of its K-part loads - issued before the request's state, so a task that starts after its inputs
// landed (behind a projection task on its CU: MI355X dec265, T 16, the latest recurrence tasks start ~118, inputs ~124) waits
// one round trip for all of them, instead of the state, then the slot, then the window and the polls one after the other
#ifndef STATIC_DRD_CIR_EARLY
#define STATIC_DRD_CIR_EARLY 0
#endif
// STATIC_DRD_G_EARLY (default 0; with STATIC_DRD_CIR_EARLY): g's K parts issued with the other early loads (threads 129 .. 160)
// and taken at the input stage, not polled after the state update (dec279 T 16: the last recurrence task's update 131.9 -> o_proj
// sees y ~136.0; for the late tasks g has landed long before)
#ifndef STATIC_DRD_G_EARLY
#define STATIC_DRD_G_EARLY 0
#endif
static_assert(!STATIC_DRD_G_EARLY || STATIC_DRD_CIR_EARLY, "STATIC_DRD_G_EARLY: with STATIC_DRD_CIR_EARLY");
// STATIC_DRD_L2NEXT (defined: on): a task pre-reads the state of the CU's next list entry, if a recurrence task, into the
// XCD's L2 (one 4-B load per 128-B line, the values unused; 64 KB per request): that task's state loads then hit the L2
// (amd/bench/l2_warm.hip: 64 KB per CU 0.64 vs 3.04 us cold). Its first request's state only

__device__ __forceinline__ float drd_sum8(float v) { // the sum over lanes 8 k .. 8 k + 7 (every lane)
  v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, false));  // quad_perm [1,0,3,2]
  v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, false));  // quad_perm [2,3,0,1]
  v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x141, 0xF, 0xF, false)); // row_half_mirror
  return v;
}

namespace drd {
constexpr int DK = 128, DV = 128;
// an optional param (0 when the node has fewer)
template <class PARAMS, int I>
constexpr int param_or0() {
  if constexpr (sizeof(PARAMS::v) / sizeof(int) > I + 1) {
    return PARAMS::v[I];
  } else {
    return 0;
  }
}
template <class A, class... B>
struct first_input {
  using type = A;
};
// LDS (floats from L.sm), R = requests per task: q^ [R][128], k^ [R][128], v [R][128], decay [R][128], the gate's g
// [R][128], o [R][128], beta [R], the norms [R][2], f_a in bf16 [R][128]
template <int R>
struct Lay {
  static constexpr int QN = 0, KN = QN + R * DK, VV = KN + R * DK, DEC = VV + R * DV, GT = DEC + R * DK,
                       OO = GT + R * DV, BETA = OO + R * DV, RED = BETA + R, FAB = RED + 2 * R, END = FAB + R * DK / 2;
};
} // namespace drd

template <int PPART, int PROJ_N>
__device__ __forceinline__ float4 drd_proj4(float const *proj, int t, int c) {
  float4 zp[PPART];
  bool all;
  do {
#pragma unroll
    for (int p = 0; p < PPART; p++) {
      zp[p] = as_float4(ld16_at<AUX_DEV>(proj, (uint32_t)((((size_t)p * T + t) * PROJ_N + c) * 4)));
    }
    all = true;
#pragma unroll
    for (int p = 0; p < PPART; p++) {
      all = all && validf4(zp[p]);
    }
    if (!all) {
      tail_pause();
    }
  } while (!all);
  float zv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int p = 0; p < PPART; p++) { // part order
    zv[0] += zp[p].x;
    zv[1] += zp[p].y;
    zv[2] += zp[p].z;
    zv[3] += zp[p].w;
  }
  return make_float4(round_bf16(zv[0]), round_bf16(zv[1]), round_bf16(zv[2]), round_bf16(zv[3]));
}

#if STATIC_DRD_CIR_EARLY
// drd_proj4 in two halves (STATIC_DRD_CIR_EARLY): the first round's loads issued early, then checked (re-loaded until all landed)
template <int PPART, int PROJ_N>
__device__ __forceinline__ void drd_proj4_issue(float const *proj, int t, int c, float4 (&zp)[PPART]) {
#pragma unroll
  for (int p = 0; p < PPART; p++) {
    zp[p] = as_float4(ld16_at<AUX_DEV>(proj, (uint32_t)((((size_t)p * T + t) * PROJ_N + c) * 4)));
  }
}
template <int PPART, int PROJ_N>
__device__ __forceinline__ float4 drd_proj4_finish(float const *proj, int t, int c, float4 (&zp)[PPART]) {
  for (;;) {
    bool all = true;
#pragma unroll
    for (int p = 0; p < PPART; p++) {
      all = all && validf4(zp[p]);
    }
    if (all) {
      break;
    }
    tail_pause();
    drd_proj4_issue<PPART, PROJ_N>(proj, t, c, zp);
  }
  float zv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int p = 0; p < PPART; p++) { // part order
    zv[0] += zp[p].x;
    zv[1] += zp[p].y;
    zv[2] += zp[p].z;
    zv[3] += zp[p].w;
  }
  return make_float4(round_bf16(zv[0]), round_bf16(zv[1]), round_bf16(zv[2]), round_bf16(zv[3]));
}
#endif

// PPART > 0 (MPK_KDA_SPLIT_PROJ: short_conv_decode only for q / k / v): f_a and beta read from the projection's K parts after the
// conv's count, g only before the output (the plan puts the g / f_a / beta rows' projection tasks last: the conv and the
// recurrence's first steps do not wait for them; MI355X dec62, T = 1: the projection's 94 second-wave tasks ended ~4.5 us after
// the first wave, and the conv waited for them)
// CIR (delta_rule_decode_layer conv=...: no short_conv_decode node): the task makes its own head's q / k / v of its requests -
// the projection's K parts added (polled, part order, bf16, as short_conv_decode), the 4-tap conv over the request's window
// (conv_w [3 HD][4] fp32, pool [slots][3][3 HD] bf16) and the new input, SiLU, bf16 - and moves those columns' windows on in
// place (this task alone holds them: head x of request t's slot); f_a, beta and g from the K parts as PPART > 0
template <int PROJ_N, int COL_G, int COL_FA, int COL_BETA, int HD, int EPS_BITS, int R, int PPART = 0, bool CIR = false>
__device__ __forceinline__ void delta_rule_decode_task(KernelLocals &L,
                                                       __hip_bfloat16 const *row, // [T][PROJ_N] bf16, then [T][3 HD] fp32
                                                       __hip_bfloat16 const *wfb, // [HD][128]
                                                       float const *prm,          // A_log [heads], dt_bias [HD], o_norm w
                                                       float *state_pool,         // [slots][heads][128][128]
                                                       int const *slots,          // [T]
                                                       uint32_t const *in_cnt,    // the conv's counter, complete at in_need
                                                       uint32_t in_need,
                                                       __hip_bfloat16 *y, // [T][HD]
                                                       uint32_t *cnt,
                                                       int head,
                                                       int grp_y,
                                                       float const *proj = nullptr,
                                                       float const *conv_w = nullptr,         // CIR
                                                       __hip_bfloat16 *pool = nullptr         // CIR
#ifdef STATIC_DRD_L2NEXT
                                                       , int2 nxt = make_int2(-1, 0) // the next task's (head, grp_y)
#endif
                                                       ) {
  using namespace drd;
  using Ly = Lay<R>;
  static_assert(PROJ_N % 8 == 0 && COL_G % 8 == 0 && COL_FA % 8 == 0 && HD % 128 == 0 && R >= 1 && R <= 16,
                "delta_rule_decode: 16-B aligned columns, whole heads, at most 16 requests per task");
  static_assert(!CIR || PPART > 0, "delta_rule_decode CIR: the projection's K parts");
  constexpr int heads = HD / 128;
  float const eps = __int_as_float(EPS_BITS);
  int const tid = task_tid(), lane = tid & 63, wave = tid >> 6, tok = lane % 16, grp = lane / 16;
  int const kq = tid & 7, kb = (STATIC_DRD_COAL ? 4 : 16) * kq, vb4 = 4 * (tid >> 3); // the state block: rows vb4 .. + 3, columns kb .. + 15 (COAL: kb + 32 c' .. + 3)
  int const r0 = R * grp_y, nr = T - r0 < R ? T - r0 : R;     // this task's requests r0 .. r0 + nr - 1
  if (nr <= 0) {
    task_post(cnt, 1);
    return;
  }
  auto state_at = [&](int s) { return state_pool + (((size_t)s * heads + head) * DV + vb4) * DK + kb; };
  auto load_state = [&](int s, float (&S)[4][16]) {
    if (s >= 0) {
      float const *sp = state_at(s);
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int c = 0; c < 4; c++) {
          float4 const s4 = *reinterpret_cast<float4 const *>(sp + (size_t)j * DK + DRD_CS * c);
          S[j][4 * c] = s4.x;
          S[j][4 * c + 1] = s4.y;
          S[j][4 * c + 2] = s4.z;
          S[j][4 * c + 3] = s4.w;
        }
    } else {
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int c = 0; c < 16; c++) {
          S[j][c] = 0.f;
        }
    }
  };
  // 1. what needs no input: the first request's state, the W_fb fragments, the parameters (in flight while the conv
  // finishes; the state is this step's own: no other task writes it)
  float S[4][16];
  int s_cur = slots[r0];
#if STATIC_DRD_CIR_EARLY
  static_assert(!CIR || (STATIC_DRD_CIR_PAR && R == 1 && PPART > 0), "STATIC_DRD_CIR_EARLY: CIR_PAR, one request per task");
  // (STATIC_DRD_CIR_EARLY: item v = tid's window, conv weights and first K-part loads, before the state; t = r0, its slot s_cur)
  uint2 eu0 = make_uint2(0u, 0u), eu1 = eu0, eu2 = eu0;
  float4 ew4[4];
  float4 ezp[PPART > 0 ? PPART : 1];
  int ecol = 0;
  if constexpr (CIR) {
    constexpr int NCONV = 3 * HD;
    if (tid < 96) {
      int const st = tid >> 5, c = 4 * (tid & 31);
      ecol = st * HD + head * 128 + c;
      if (s_cur >= 0) {
        __hip_bfloat16 const *const win = pool + (size_t)s_cur * 3 * NCONV + ecol;
        eu0 = *reinterpret_cast<uint2 const *>(win);
        eu1 = *reinterpret_cast<uint2 const *>(win + NCONV);
        eu2 = *reinterpret_cast<uint2 const *>(win + 2 * NCONV);
      }
#pragma unroll
      for (int j = 0; j < 4; j++) {
        ew4[j] = *reinterpret_cast<float4 const *>(conv_w + (size_t)(ecol + j) * 4);
      }
    } else if (tid < 128) {
      ecol = COL_FA + 4 * (tid - 96);
    } else if (tid == 128) {
      ecol = (COL_BETA + head) & ~3;
    } else if (STATIC_DRD_G_EARLY && tid < 161) {
      ecol = COL_G + head * 128 + 4 * (tid - 129);
    }
    if (tid < (STATIC_DRD_G_EARLY ? 161 : 129)) {
      drd_proj4_issue<PPART, PROJ_N>(proj, r0, ecol, ezp);
    }
  }
#endif
  load_state(s_cur, S);
#ifdef STATIC_DRD_L2NEXT
  float wv0 = 0.f, wv1 = 0.f;
  if (nxt.x >= 0 && R * nxt.y < T) {
    int const sn = slots[R * nxt.y];
    if (sn >= 0) {
      float const *pn = state_pool + ((size_t)sn * heads + nxt.x) * DV * DK; // [128][128] fp32: 512 lines of 128 B
      wv0 = pn[(size_t)tid * 32];
      wv1 = pn[(size_t)(tid + 256) * 32];
    }
  }
#endif
  // W_fb^T rows k = 16 (2 wave + i) + l % 16, columns c = 32 s + 8 (l / 16) .. + 7
  v4f wfa[2][4];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int st = 0; st < 4; st++) {
      wfa[i][st] = *reinterpret_cast<v4f const *>(wfb + ((size_t)head * 128 + 16 * (2 * wave + i) + tok) * 128 + 32 * st +
                                                  8 * grp);
    }
  float const a_exp = __expf(prm[head]);
  float dtb[2][4]; // dt_bias of the k rows this lane gets from the MFMA
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int r = 0; r < 4; r++) {
      dtb[i][r] = prm[heads + head * 128 + 16 * (2 * wave + i) + 4 * grp + r];
    }
  float *const sm = reinterpret_cast<float *>(L.sm);
  __hip_bfloat16 *const fab = reinterpret_cast<__hip_bfloat16 *>(sm + Ly::FAB);
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads(); // the LDS may still be read by the previous task
  // 2. the conv's outputs
  if constexpr (!CIR) {
    if (tid == 0) {
      cnt_wait_in(in_cnt, in_need);
    }
    __syncthreads();
  }
  if constexpr (CIR) { // request v / 96, stream (v % 96) / 32 (q, k, v), columns 4 ((v % 96) % 32) .. + 3 of this head
    constexpr int NCONV = 3 * HD;
#if STATIC_DRD_CIR_PAR
    // (STATIC_DRD_CIR_PAR: q / k / v (+ conv), f_a and beta polled by different threads at once, each item's conv window and
    // weights loaded before its poll: items [0, 96 nr) q / k / v, [96 nr, 128 nr) f_a, [128 nr, 129 nr) beta)
    static_assert(PPART > 0, "STATIC_DRD_CIR_PAR: the projection's K parts");
#if STATIC_DRD_CIR_EARLY
    if (tid < (STATIC_DRD_G_EARLY ? 161 : 129)) { // (the item's loads issued at the start)
      int const v = tid, t = r0, s = s_cur;
      if (v < 96) {
        int const st = v >> 5, c = 4 * (v & 31), col = ecol;
        __hip_bfloat16 *const win = pool + (size_t)(s >= 0 ? s : 0) * 3 * NCONV + col;
        uint2 const u0 = eu0, u1 = eu1, u2 = eu2;
        float4 const z = drd_proj4_finish<PPART, PROJ_N>(proj, t, col, ezp); // (bf16 values)
        float y4[4] = {0.f, 0.f, 0.f, 0.f};
        if (s >= 0) {
          float const a0[4] = {bf16_lo(u0.x), bf16_hi(u0.x), bf16_lo(u0.y), bf16_hi(u0.y)};
          float const a1[4] = {bf16_lo(u1.x), bf16_hi(u1.x), bf16_lo(u1.y), bf16_hi(u1.y)};
          float const a2[4] = {bf16_lo(u2.x), bf16_hi(u2.x), bf16_lo(u2.y), bf16_hi(u2.y)};
          float const xn[4] = {z.x, z.y, z.z, z.w};
#pragma unroll
          for (int j = 0; j < 4; j++) { // (short_conv_decode's order)
            float acc = 0.f;
            acc = fmaf(a0[j], ew4[j].x, acc);
            acc = fmaf(a1[j], ew4[j].y, acc);
            acc = fmaf(a2[j], ew4[j].z, acc);
            acc = fmaf(xn[j], ew4[j].w, acc);
            y4[j] = round_bf16(acc / (1.0f + __expf(-acc)));
          }
          *reinterpret_cast<uint2 *>(win) = u1; // the window after the token: (c1, c2, x)
          *reinterpret_cast<uint2 *>(win + NCONV) = u2;
          *reinterpret_cast<uint2 *>(win + 2 * NCONV) = make_uint2(pack_bf16x2(xn[0], xn[1]), pack_bf16x2(xn[2], xn[3]));
        }
        *reinterpret_cast<float4 *>(sm + (st == 0 ? Ly::QN : st == 1 ? Ly::KN : Ly::VV) + c) =
            make_float4(y4[0], y4[1], y4[2], y4[3]);
      } else if (v < 128) {
        int const c = 4 * (v - 96);
        float4 const f = drd_proj4_finish<PPART, PROJ_N>(proj, t, ecol, ezp);
        *reinterpret_cast<uint2 *>(fab + c) = make_uint2(pack_bf16x2(f.x, f.y), pack_bf16x2(f.z, f.w));
      } else if (v == 128) {
        int const cb = COL_BETA + head;
        float4 const b4 = drd_proj4_finish<PPART, PROJ_N>(proj, t, ecol, ezp);
        float const bv[4] = {b4.x, b4.y, b4.z, b4.w};
        sm[Ly::BETA] = 1.f / (1.f + __expf(-bv[cb & 3]));
      } else { // (STATIC_DRD_G_EARLY: g)
        float4 const g4 = drd_proj4_finish<PPART, PROJ_N>(proj, t, ecol, ezp);
        *reinterpret_cast<float4 *>(sm + Ly::GT + 4 * (v - 129)) = g4;
      }
    }
    if (false)
#endif
    for (int v = tid; v < nr * 129; v += 256) {
      if (v < nr * 96) {
        int const m = v / 96, w = v % 96, st = w >> 5, c = 4 * (w & 31), t = r0 + m, col = st * HD + head * 128 + c;
        int const s = slots[t];
        __hip_bfloat16 *const win = pool + (size_t)(s >= 0 ? s : 0) * 3 * NCONV + col;
        uint2 u0 = make_uint2(0u, 0u), u1 = u0, u2 = u0;
        if (s >= 0) {
          u0 = *reinterpret_cast<uint2 const *>(win);
          u1 = *reinterpret_cast<uint2 const *>(win + NCONV);
          u2 = *reinterpret_cast<uint2 const *>(win + 2 * NCONV);
        }
        float4 w4[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          w4[j] = *reinterpret_cast<float4 const *>(conv_w + (size_t)(col + j) * 4);
        }
        float4 const z = drd_proj4<PPART, PROJ_N>(proj, t, col); // (bf16 values)
        float y4[4] = {0.f, 0.f, 0.f, 0.f};
        if (s >= 0) {
          float const a0[4] = {bf16_lo(u0.x), bf16_hi(u0.x), bf16_lo(u0.y), bf16_hi(u0.y)};
          float const a1[4] = {bf16_lo(u1.x), bf16_hi(u1.x), bf16_lo(u1.y), bf16_hi(u1.y)};
          float const a2[4] = {bf16_lo(u2.x), bf16_hi(u2.x), bf16_lo(u2.y), bf16_hi(u2.y)};
          float const xn[4] = {z.x, z.y, z.z, z.w};
#pragma unroll
          for (int j = 0; j < 4; j++) { // (short_conv_decode's order)
            float acc = 0.f;
            acc = fmaf(a0[j], w4[j].x, acc);
            acc = fmaf(a1[j], w4[j].y, acc);
            acc = fmaf(a2[j], w4[j].z, acc);
            acc = fmaf(xn[j], w4[j].w, acc);
            y4[j] = round_bf16(acc / (1.0f + __expf(-acc)));
          }
          *reinterpret_cast<uint2 *>(win) = u1; // the window after the token: (c1, c2, x)
          *reinterpret_cast<uint2 *>(win + NCONV) = u2;
          *reinterpret_cast<uint2 *>(win + 2 * NCONV) = make_uint2(pack_bf16x2(xn[0], xn[1]), pack_bf16x2(xn[2], xn[3]));
        }
        *reinterpret_cast<float4 *>(sm + (st == 0 ? Ly::QN : st == 1 ? Ly::KN : Ly::VV) + m * DK + c) =
            make_float4(y4[0], y4[1], y4[2], y4[3]);
      } else if (v < nr * 128) {
        int const m = (v - nr * 96) >> 5, c = 4 * ((v - nr * 96) & 31);
        float4 const f = drd_proj4<PPART, PROJ_N>(proj, r0 + m, COL_FA + c);
        *reinterpret_cast<uint2 *>(fab + m * DK + c) = make_uint2(pack_bf16x2(f.x, f.y), pack_bf16x2(f.z, f.w));
      } else {
        int const m = v - nr * 128, cb = COL_BETA + head;
        float4 const b4 = drd_proj4<PPART, PROJ_N>(proj, r0 + m, cb & ~3);
        float const bv[4] = {b4.x, b4.y, b4.z, b4.w};
        sm[Ly::BETA + m] = 1.f / (1.f + __expf(-bv[cb & 3]));
      }
    }
#else
    for (int v = tid; v < nr * 96; v += 256) {
      int const m = v / 96, w = v % 96, st = w >> 5, c = 4 * (w & 31), t = r0 + m, col = st * HD + head * 128 + c;
      float4 const z = drd_proj4<PPART, PROJ_N>(proj, t, col); // (bf16 values)
      int const s = slots[t];
      float y4[4] = {0.f, 0.f, 0.f, 0.f};
      if (s >= 0) {
        __hip_bfloat16 *const win = pool + (size_t)s * 3 * NCONV + col;
        uint2 const u0 = *reinterpret_cast<uint2 const *>(win), u1 = *reinterpret_cast<uint2 const *>(win + NCONV),
                    u2 = *reinterpret_cast<uint2 const *>(win + 2 * NCONV);
        float const a0[4] = {bf16_lo(u0.x), bf16_hi(u0.x), bf16_lo(u0.y), bf16_hi(u0.y)};
        float const a1[4] = {bf16_lo(u1.x), bf16_hi(u1.x), bf16_lo(u1.y), bf16_hi(u1.y)};
        float const a2[4] = {bf16_lo(u2.x), bf16_hi(u2.x), bf16_lo(u2.y), bf16_hi(u2.y)};
        float const xn[4] = {z.x, z.y, z.z, z.w};
#pragma unroll
        for (int j = 0; j < 4; j++) { // (short_conv_decode's order)
          float4 const w4 = *reinterpret_cast<float4 const *>(conv_w + (size_t)(col + j) * 4);
          float acc = 0.f;
          acc = fmaf(a0[j], w4.x, acc);
          acc = fmaf(a1[j], w4.y, acc);
          acc = fmaf(a2[j], w4.z, acc);
          acc = fmaf(xn[j], w4.w, acc);
          y4[j] = round_bf16(acc / (1.0f + __expf(-acc)));
        }
        *reinterpret_cast<uint2 *>(win) = u1; // the window after the token: (c1, c2, x)
        *reinterpret_cast<uint2 *>(win + NCONV) = u2;
        *reinterpret_cast<uint2 *>(win + 2 * NCONV) = make_uint2(pack_bf16x2(xn[0], xn[1]), pack_bf16x2(xn[2], xn[3]));
      }
      *reinterpret_cast<float4 *>(sm + (st == 0 ? Ly::QN : st == 1 ? Ly::KN : Ly::VV) + m * DK + c) =
          make_float4(y4[0], y4[1], y4[2], y4[3]);
    }
#endif
  } else {
  // (another XCD's CUs wrote them past their L2: device-scope loads)
  float const *qkv = reinterpret_cast<float const *>(row + (size_t)T * PROJ_N);
  for (int v = tid; v < nr * 32; v += 256) { // request v / 32, columns 4 (v % 32) .. + 3 of q, k, v
    int const m = v >> 5, c = 4 * (v & 31);
    size_t const qo = ((size_t)(r0 + m) * 3 * HD + head * 128 + c) * 4;
    uint4 const q4 = ld16_at<AUX_DEV>(qkv, (uint32_t)qo), k4 = ld16_at<AUX_DEV>(qkv, (uint32_t)(qo + HD * 4)),
                v4 = ld16_at<AUX_DEV>(qkv, (uint32_t)(qo + 2 * HD * 4));
    *reinterpret_cast<uint4 *>(sm + Ly::QN + m * DK + c) = q4;
    *reinterpret_cast<uint4 *>(sm + Ly::KN + m * DK + c) = k4;
    *reinterpret_cast<uint4 *>(sm + Ly::VV + m * DV + c) = v4;
  }
  }
  if constexpr (PPART > 0) { // f_a and beta from the projection's K parts (g before the output, below)
#if STATIC_DRD_CIR_PAR
    if constexpr (!CIR) {
#endif
    for (int v = tid; v < nr * 32; v += 256) { // request v / 32: f_a columns 4 (v % 32) .. + 3
      int const m = v >> 5, c = 4 * (v & 31);
      float4 const f = drd_proj4<PPART, PROJ_N>(proj, r0 + m, COL_FA + c);
      *reinterpret_cast<uint2 *>(fab + m * DK + c) = make_uint2(pack_bf16x2(f.x, f.y), pack_bf16x2(f.z, f.w));
    }
    if (tid < nr) {
      int const cb = COL_BETA + head;
      float4 const b4 = drd_proj4<PPART, PROJ_N>(proj, r0 + tid, cb & ~3);
      float const bv[4] = {b4.x, b4.y, b4.z, b4.w};
      sm[Ly::BETA + tid] = 1.f / (1.f + __expf(-bv[cb & 3]));
    }
#if STATIC_DRD_CIR_PAR
    }
#endif
  } else {
  for (int v = tid; v < nr * 16; v += 256) { // request v / 16: f_a and g, 8 bf16 at 8 (v % 16)
    int const m = v >> 4, c = 8 * (v & 15);
    size_t const rb = (size_t)(r0 + m) * PROJ_N;
    uint4 const f = ld16_at<AUX_DEV>(row, (uint32_t)((rb + COL_FA + c) * 2));
    uint4 const g4 = ld16_at<AUX_DEV>(row, (uint32_t)((rb + COL_G + head * 128 + c) * 2));
    *reinterpret_cast<uint4 *>(fab + m * DK + c) = f;
    uint32_t const gw[4] = {g4.x, g4.y, g4.z, g4.w};
#pragma unroll
    for (int e = 0; e < 8; e++) {
      sm[Ly::GT + m * DV + c + e] = (e & 1) ? bf16_hi(gw[e / 2]) : bf16_lo(gw[e / 2]);
    }
  }
  if (tid < nr) { // beta: the 16-B vector holding it
    size_t const eb = (size_t)(r0 + tid) * PROJ_N + COL_BETA + head;
    uint4 const b4 = ld16_at<AUX_DEV>(row, (uint32_t)((eb & ~(size_t)7) * 2));
    uint32_t const bw[4] = {b4.x, b4.y, b4.z, b4.w};
    int const j = (int)(eb & 7);
    float const b = (j & 1) ? bf16_hi(bw[j / 2]) : bf16_lo(bw[j / 2]);
    sm[Ly::BETA + tid] = 1.f / (1.f + __expf(-b));
  }
  }
  __syncthreads();
#ifdef STATIC_TIMING_BUILD
  if (tid == 0 && L.stage_stamps) { // (stage stamp 0: the inputs landed: q / k / v (CIR: convolved), f_a, beta)
    L.stage_stamps[0] = gtime();
  }
#endif
  // 3. per request: the norms of q and k (wave w: requests w, w + 4, ..); f_b = bf16(f_a . W_fb^T) by MFMA (D: lane ->
  // request l % 16, register r -> k = 16 (2 wave + i) + 4 (l / 16) + r) and the decay exp(gk)
  for (int m = wave; m < nr; m += 4) {
    float const qa = sm[Ly::QN + m * DK + lane], qb = sm[Ly::QN + m * DK + lane + 64];
    float const ka = sm[Ly::KN + m * DK + lane], kb2 = sm[Ly::KN + m * DK + lane + 64];
    float sq = qa * qa + qb * qb, sk = ka * ka + kb2 * kb2;
    sq = wave_sum64(sq);
    sk = wave_sum64(sk);
    if (lane == 0) {
      sm[Ly::RED + 2 * m] = rsqrtf(sq + 1e-6f) * 0.08838834764831845f; // 128^-0.5
      sm[Ly::RED + 2 * m + 1] = rsqrtf(sk + 1e-6f);
    }
  }
  {
    v4f acc[2] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int st = 0; st < 4; st++) {
      v4f b = {0.f, 0.f, 0.f, 0.f};
      if (tok < nr) {
        b = *reinterpret_cast<v4f const *>(fab + tok * DK + 32 * st + 8 * grp);
      }
#pragma unroll
      for (int i = 0; i < 2; i++) {
        acc[i] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, wfa[i][st]), __builtin_bit_cast(v8bf, b),
                                                         acc[i], 0, 0, 0);
      }
    }
    if (tok < nr) {
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int r = 0; r < 4; r++) {
          int const k = 16 * (2 * wave + i) + 4 * grp + r;
          float const fbv = round_bf16(acc[i][r]);
          float const sg = 1.f / (1.f + __expf(-a_exp * (fbv + dtb[i][r])));
          sm[Ly::DEC + tok * DK + k] = __expf(-5.f * sg);
        }
    }
  }
  __syncthreads();
  // 4. the requests in order: thread i's state block; the next request's state loaded first
#pragma unroll 1
  for (int m = 0; m < nr; m++) {
    float Sn[4][16];
    int const s_next = m + 1 < nr ? slots[r0 + m + 1] : -1;
    load_state(s_next, Sn);
    float const rq = sm[Ly::RED + 2 * m], rk = sm[Ly::RED + 2 * m + 1], beta = sm[Ly::BETA + m];
    float dk[16], kn[16], qn[16];
#pragma unroll
    for (int c = 0; c < 16; c += 4) {
      float4 const d4 = *reinterpret_cast<float4 const *>(sm + Ly::DEC + m * DK + kb + DRD_CL * c);
      float4 const k4 = *reinterpret_cast<float4 const *>(sm + Ly::KN + m * DK + kb + DRD_CL * c);
      float4 const q4 = *reinterpret_cast<float4 const *>(sm + Ly::QN + m * DK + kb + DRD_CL * c);
      dk[c] = d4.x, dk[c + 1] = d4.y, dk[c + 2] = d4.z, dk[c + 3] = d4.w;
      kn[c] = k4.x * rk, kn[c + 1] = k4.y * rk, kn[c + 2] = k4.z * rk, kn[c + 3] = k4.w * rk;
      qn[c] = q4.x * rq, qn[c + 1] = q4.y * rq, qn[c + 2] = q4.z * rq, qn[c + 3] = q4.w * rq;
    }
    float p[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float a = 0.f;
#pragma unroll
      for (int c = 0; c < 16; c++) {
        S[j][c] *= dk[c];
        a = fmaf(S[j][c], kn[c], a);
      }
      p[j] = drd_sum8(a);
    }
    float o[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float const d = beta * (sm[Ly::VV + m * DV + vb4 + j] - p[j]);
      float a = 0.f;
#pragma unroll
      for (int c = 0; c < 16; c++) {
        S[j][c] = fmaf(d, kn[c], S[j][c]);
        a = fmaf(S[j][c], qn[c], a);
      }
      o[j] = drd_sum8(a);
    }
    if (kq == 0) { // (SGLang ROCm: o stored in bf16, then rms_norm_gated)
      *reinterpret_cast<float4 *>(sm + Ly::OO + m * DV + vb4) =
          make_float4(round_bf16(o[0]), round_bf16(o[1]), round_bf16(o[2]), round_bf16(o[3]));
    }
    // the state after the token, in place (the last request's after the count, below)
    if ((m + 1 < nr || !1) && s_cur >= 0) {
      float *sp = state_at(s_cur);
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int c = 0; c < 4; c++) {
          *reinterpret_cast<float4 *>(sp + (size_t)j * DK + DRD_CS * c) =
              make_float4(S[j][4 * c], S[j][4 * c + 1], S[j][4 * c + 2], S[j][4 * c + 3]);
        }
    }
    if (m + 1 < nr || !1) {
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int c = 0; c < 16; c++) {
          S[j][c] = Sn[j][c];
        }
      s_cur = s_next;
    }
  }
  __syncthreads();
#ifdef STATIC_TIMING_BUILD
  if (tid == 0 && L.stage_stamps) { // (stage stamp 1: the requests' states updated, o made)
    L.stage_stamps[1] = gtime();
  }
#endif
  if constexpr (PPART > 0 && !(STATIC_DRD_G_EARLY && CIR)) { // g of this head: the last of the projection's rows the plan schedules
    for (int v = tid; v < nr * 32; v += 256) {
      int const m = v >> 5, c = 4 * (v & 31);
      float4 const g4 = drd_proj4<PPART, PROJ_N>(proj, r0 + m, COL_G + head * 128 + c);
      *reinterpret_cast<float4 *>(sm + Ly::GT + m * DV + c) = g4;
    }
  }
  // 5. o_norm per request (wave w: requests w, w + 4, ..), the gate, y
  for (int m = wave; m < nr; m += 4) {
    float const oa = sm[Ly::OO + m * DV + lane], ob = sm[Ly::OO + m * DV + lane + 64];
    float s2 = oa * oa + ob * ob;
    s2 = wave_sum64(s2);
    if (lane == 0) {
      sm[Ly::RED + 2 * m] = rsqrtf(s2 / DV + eps);
    }
  }
  __syncthreads();
  for (int v = tid; v < nr * 16; v += 256) { // request v / 16, v 8 (v % 16) .. + 7
    int const m = v >> 4, vb = 8 * (v & 15);
    float const rs = sm[Ly::RED + 2 * m];
    float o8[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      float const gv = sm[Ly::GT + m * DV + vb + e], wv = prm[heads + HD + vb + e];
      o8[e] = sm[Ly::OO + m * DV + vb + e] * rs * wv * (1.f / (1.f + __expf(-gv))); // (fused_norm_gate.py order)
    }
    st16_dev(y + (size_t)(r0 + m) * HD + head * 128 + vb,
             pack_bf16x8(make_float4(o8[0], o8[1], o8[2], o8[3]), make_float4(o8[4], o8[5], o8[6], o8[7])));
  }
  task_post(cnt, 1);
#ifdef STATIC_DRD_L2NEXT
  asm volatile("" ::"v"(wv0), "v"(wv1)); // (the pre-reads' values: used here, long landed)
#endif
  // the last request's state, in place, after the count: o_proj waits for y only (the next launch reads the pool)
  if (s_cur >= 0) {
    float *sp = state_at(s_cur);
#pragma unroll
    for (int j = 0; j < 4; j++)
#pragma unroll
      for (int c = 0; c < 4; c++) {
        *reinterpret_cast<float4 *>(sp + (size_t)j * DK + DRD_CS * c) =
            make_float4(S[j][4 * c], S[j][4 * c + 1], S[j][4 * c + 2], S[j][4 * c + 3]);
      }
  }
}

// FLAT (delta_rule_decode_layer pairs_per_task P > 0; params + [2, P]): task x = the (head, request) pairs P x .. P x + P - 1 in
// head-major order (pair p: head p / T, request p % T; a task spans at most two heads), every pair's state loaded at the task's
// start (P states in registers, all in flight together), then each pair's step as delta_rule_decode_task's. At T = 64 the 768
// pairs are 3 per CU on all 256 CUs (requests per task R must divide T: R 1 = 3 tasks per CU one after another, each ~13 us for
// its 64 KB; R 4 = 192 CUs with 4 each; MI355X dec180 / dec189: the recurrence 53 / 44 us after the conv)
// STATIC_DRD_FLAT_LATE 1 (default 0; FLAT): every pair's state written back after the task's count (they are all held in
// registers anyway), not only the last one's: o_proj waits for the count (MI355X dec203, T 32 P 2: the step's end -> count
// 11 us median, the count waiting for the first pair's 64 KB of state stores)
#ifndef STATIC_DRD_FLAT_LATE
#define STATIC_DRD_FLAT_LATE 0
#endif
__device__ __forceinline__ void drd_st_state(float *p, float a, float b, float c, float d) {
  *reinterpret_cast<float4 *>(p) = make_float4(a, b, c, d);
}
// CIR (FLAT + conv, params[5] 3; MPK_KDA_CONV_IN_REC=1 with MPK_KDA_FLAT=P): no conv node - each pair's q / k / v made from the
// projection's PPART K parts (polled, as the CIR task), its window in pool updated; f_a, g, beta from the K parts too
template <int PROJ_N, int COL_G, int COL_FA, int COL_BETA, int HD, int EPS_BITS, int P, int PPART = 0, bool CIR = false>
__device__ __forceinline__ void delta_rule_flat_task(KernelLocals &L,
                                                     __hip_bfloat16 const *row, // [T][PROJ_N] bf16, then [T][3 HD] fp32
                                                     __hip_bfloat16 const *wfb, // [HD][128]
                                                     float const *prm,          // A_log [heads], dt_bias [HD], o_norm w
                                                     float *state_pool,         // [slots][heads][128][128]
                                                     int const *slots,          // [T]
                                                     uint32_t const *in_cnt,    // the conv's counter, complete at in_need
                                                     uint32_t in_need,
                                                     __hip_bfloat16 *y, // [T][HD]
                                                     uint32_t *cnt,
                                                     int x,
                                                     float const *proj = nullptr,          // CIR: [PPART][T][PROJ_N] fp32
                                                     float const *conv_w = nullptr,        // CIR: [3 HD][4] fp32
                                                     __hip_bfloat16 *pool = nullptr) {     // CIR: [slots][3][3 HD] bf16
  using namespace drd;
  using Ly = Lay<P>;
  constexpr int heads = HD / 128, NPAIR = heads * T;
  static_assert(PROJ_N % 8 == 0 && COL_G % 8 == 0 && COL_FA % 8 == 0 && HD % 128 == 0 && P >= 1 && P <= 4 && P <= T,
                "delta_rule_decode FLAT: 16-B aligned columns, whole heads, 1 .. 4 pairs per task");
  float const eps = __int_as_float(EPS_BITS);
  int const tid = task_tid_all(), lane = tid & 63, wave = tid >> 6, tok = lane % 16, grp = lane / 16;
  int const kq = tid & 7, kb = (STATIC_DRD_COAL ? 4 : 16) * kq, vb4 = 4 * (tid >> 3); // the state block: rows vb4 .. + 3, columns kb .. + 15 (COAL: kb + 32 c' .. + 3)
  int const p0 = P * x, np = NPAIR - p0 < P ? NPAIR - p0 : P;
#ifdef STATIC_DRD_HOLD
  // (core.cuh STATIC_DRD_HOLD: the next list entry's node, by a scalar load - the list is written by the host only)
  bool const hold = L.nxt != nullptr &&
                    reinterpret_cast<__attribute__((address_space(4))) int const *>(reinterpret_cast<size_t>(L.nxt))[0] == STATIC_DRD_HOLD;
#endif
  if (np <= 0) {
    task_post(cnt, 1);
    return;
  }
  int const h0 = p0 / T, h1 = (p0 + np - 1) / T; // the task's heads (h1 = h0 or h0 + 1)
  auto hd_of = [&](int m) { return (p0 + m) / T; };
  auto rq_of = [&](int m) { return (p0 + m) % T; };
  auto state_at = [&](int s, int h) { return state_pool + (((size_t)s * heads + h) * DV + vb4) * DK + kb; };
  // 1. every pair's state (in flight together), the W_fb fragments of the task's heads, the parameters
  float S[P][4][16];
#pragma unroll
  for (int m = 0; m < P; m++) {
    int const sl = m < np ? slots[rq_of(m)] : -1;
    if (sl >= 0) {
      float const *sp = state_at(sl, hd_of(m));
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int c = 0; c < 4; c++) {
          float4 const s4 = *reinterpret_cast<float4 const *>(sp + (size_t)j * DK + DRD_CS * c);
          S[m][j][4 * c] = s4.x;
          S[m][j][4 * c + 1] = s4.y;
          S[m][j][4 * c + 2] = s4.z;
          S[m][j][4 * c + 3] = s4.w;
        }
    } else {
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int c = 0; c < 16; c++) {
          S[m][j][c] = 0.f;
        }
    }
  }
  float *const sm = reinterpret_cast<float *>(L.sm);
  __hip_bfloat16 *const fab = reinterpret_cast<__hip_bfloat16 *>(sm + Ly::FAB);
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads(); // the LDS may still be read by the previous task
  // 2. the conv's outputs: q / k / v of each pair's head, f_a, g, beta
  if constexpr (CIR) { // pair v / 161, item v % 161: [0, 96) q / k / v (stream w / 32, columns 4 (w % 32) .. + 3: the conv, as
                       // the CIR task's), [96, 128) f_a, [128, 160) g, 160 beta
    static_assert(PPART > 0, "delta_rule_decode FLAT CIR: the projection's K parts");
    constexpr int NCONV = 3 * HD;
    for (int v = tid; v < np * 161; v += 256) {
      int const m = v / 161, w = v % 161, t = rq_of(m), h = hd_of(m);
      if (w < 96) {
        int const st = w >> 5, c = 4 * (w & 31), col = st * HD + h * 128 + c;
        int const s = slots[t];
        __hip_bfloat16 *const win = pool + (size_t)(s >= 0 ? s : 0) * 3 * NCONV + col;
        uint2 u0 = make_uint2(0u, 0u), u1 = u0, u2 = u0;
        if (s >= 0) {
          u0 = *reinterpret_cast<uint2 const *>(win);
          u1 = *reinterpret_cast<uint2 const *>(win + NCONV);
          u2 = *reinterpret_cast<uint2 const *>(win + 2 * NCONV);
        }
        float4 w4[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          w4[j] = *reinterpret_cast<float4 const *>(conv_w + (size_t)(col + j) * 4);
        }
        float4 const z = drd_proj4<PPART, PROJ_N>(proj, t, col); // (bf16 values)
        float y4[4] = {0.f, 0.f, 0.f, 0.f};
        if (s >= 0) {
          float const a0[4] = {bf16_lo(u0.x), bf16_hi(u0.x), bf16_lo(u0.y), bf16_hi(u0.y)};
          float const a1[4] = {bf16_lo(u1.x), bf16_hi(u1.x), bf16_lo(u1.y), bf16_hi(u1.y)};
          float const a2[4] = {bf16_lo(u2.x), bf16_hi(u2.x), bf16_lo(u2.y), bf16_hi(u2.y)};
          float const xn[4] = {z.x, z.y, z.z, z.w};
#pragma unroll
          for (int j = 0; j < 4; j++) { // (short_conv_decode's order)
            float acc = 0.f;
            acc = fmaf(a0[j], w4[j].x, acc);
            acc = fmaf(a1[j], w4[j].y, acc);
            acc = fmaf(a2[j], w4[j].z, acc);
            acc = fmaf(xn[j], w4[j].w, acc);
            y4[j] = round_bf16(acc / (1.0f + __expf(-acc)));
          }
          *reinterpret_cast<uint2 *>(win) = u1; // the window after the token: (c1, c2, x)
          *reinterpret_cast<uint2 *>(win + NCONV) = u2;
          *reinterpret_cast<uint2 *>(win + 2 * NCONV) = make_uint2(pack_bf16x2(xn[0], xn[1]), pack_bf16x2(xn[2], xn[3]));
        }
        *reinterpret_cast<float4 *>(sm + (st == 0 ? Ly::QN : st == 1 ? Ly::KN : Ly::VV) + m * DK + c) =
            make_float4(y4[0], y4[1], y4[2], y4[3]);
      } else if (w < 128) {
        int const c = 4 * (w - 96);
        float4 const f = drd_proj4<PPART, PROJ_N>(proj, t, COL_FA + c);
        *reinterpret_cast<uint2 *>(fab + m * DK + c) = make_uint2(pack_bf16x2(f.x, f.y), pack_bf16x2(f.z, f.w));
      } else if (w < 160) {
        int const c = 4 * (w - 128);
        float4 const g4 = drd_proj4<PPART, PROJ_N>(proj, t, COL_G + h * 128 + c);
        *reinterpret_cast<float4 *>(sm + Ly::GT + m * DV + c) = g4;
      } else {
        int const cb = COL_BETA + h;
        float4 const b4 = drd_proj4<PPART, PROJ_N>(proj, t, cb & ~3);
        float const bv[4] = {b4.x, b4.y, b4.z, b4.w};
        sm[Ly::BETA + m] = 1.f / (1.f + __expf(-bv[cb & 3]));
      }
    }
  } else {
  if (tid == 0) {
    cnt_wait_in(in_cnt, in_need);
  }
  __syncthreads();
  float const *qkv = reinterpret_cast<float const *>(row + (size_t)T * PROJ_N);
  for (int v = tid; v < np * 32; v += 256) { // pair v / 32, columns 4 (v % 32) .. + 3 of q, k, v
    int const m = v >> 5, c = 4 * (v & 31);
    size_t const qo = ((size_t)rq_of(m) * 3 * HD + hd_of(m) * 128 + c) * 4;
    uint4 const q4 = ld16_at<AUX_DEV>(qkv, (uint32_t)qo), k4 = ld16_at<AUX_DEV>(qkv, (uint32_t)(qo + HD * 4)),
                v4 = ld16_at<AUX_DEV>(qkv, (uint32_t)(qo + 2 * HD * 4));
    *reinterpret_cast<uint4 *>(sm + Ly::QN + m * DK + c) = q4;
    *reinterpret_cast<uint4 *>(sm + Ly::KN + m * DK + c) = k4;
    *reinterpret_cast<uint4 *>(sm + Ly::VV + m * DV + c) = v4;
  }
  for (int v = tid; v < np * 16; v += 256) { // pair v / 16: f_a and g, 8 bf16 at 8 (v % 16)
    int const m = v >> 4, c = 8 * (v & 15);
    size_t const rb = (size_t)rq_of(m) * PROJ_N;
    uint4 const f = ld16_at<AUX_DEV>(row, (uint32_t)((rb + COL_FA + c) * 2));
    uint4 const g4 = ld16_at<AUX_DEV>(row, (uint32_t)((rb + COL_G + hd_of(m) * 128 + c) * 2));
    *reinterpret_cast<uint4 *>(fab + m * DK + c) = f;
    uint32_t const gw[4] = {g4.x, g4.y, g4.z, g4.w};
#pragma unroll
    for (int e = 0; e < 8; e++) {
      sm[Ly::GT + m * DV + c + e] = (e & 1) ? bf16_hi(gw[e / 2]) : bf16_lo(gw[e / 2]);
    }
  }
  if (tid < np) { // beta: the 16-B vector holding it
    size_t const eb = (size_t)rq_of(tid) * PROJ_N + COL_BETA + hd_of(tid);
    uint4 const b4 = ld16_at<AUX_DEV>(row, (uint32_t)((eb & ~(size_t)7) * 2));
    uint32_t const bw[4] = {b4.x, b4.y, b4.z, b4.w};
    int const j = (int)(eb & 7);
    float const b = (j & 1) ? bf16_hi(bw[j / 2]) : bf16_lo(bw[j / 2]);
    sm[Ly::BETA + tid] = 1.f / (1.f + __expf(-b));
  }
  }
  __syncthreads();
#ifdef STATIC_TIMING_BUILD
  if (tid == 0 && L.stage_stamps) { // (stage stamp 0: the inputs landed)
    L.stage_stamps[0] = gtime();
  }
#endif
  // 3. per pair: the norms of q and k; f_b by MFMA with each of the task's heads (columns: the pairs), the decay
  for (int m = wave; m < np; m += 4) {
    float const qa = sm[Ly::QN + m * DK + lane], qb = sm[Ly::QN + m * DK + lane + 64];
    float const ka = sm[Ly::KN + m * DK + lane], kb2 = sm[Ly::KN + m * DK + lane + 64];
    float sq = qa * qa + qb * qb, sk = ka * ka + kb2 * kb2;
    sq = wave_sum64(sq);
    sk = wave_sum64(sk);
    if (lane == 0) {
      sm[Ly::RED + 2 * m] = rsqrtf(sq + 1e-6f) * 0.08838834764831845f; // 128^-0.5
      sm[Ly::RED + 2 * m + 1] = rsqrtf(sk + 1e-6f);
    }
  }
#pragma unroll 1
  for (int h = h0; h <= h1; h++) {
    v4f wfa[2][4];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
      for (int st = 0; st < 4; st++) {
        wfa[i][st] = *reinterpret_cast<v4f const *>(wfb + ((size_t)h * 128 + 16 * (2 * wave + i) + tok) * 128 + 32 * st +
                                                    8 * grp);
      }
    float const a_exp = __expf(prm[h]);
    v4f acc[2] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int st = 0; st < 4; st++) {
      v4f b = {0.f, 0.f, 0.f, 0.f};
      if (tok < np) {
        b = *reinterpret_cast<v4f const *>(fab + tok * DK + 32 * st + 8 * grp);
      }
#pragma unroll
      for (int i = 0; i < 2; i++) {
        acc[i] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, wfa[i][st]), __builtin_bit_cast(v8bf, b),
                                                         acc[i], 0, 0, 0);
      }
    }
    if (tok < np && hd_of(tok) == h) {
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int r = 0; r < 4; r++) {
          int const k = 16 * (2 * wave + i) + 4 * grp + r;
          float const fbv = round_bf16(acc[i][r]);
          float const sg = 1.f / (1.f + __expf(-a_exp * (fbv + prm[heads + h * 128 + k])));
          sm[Ly::DEC + tok * DK + k] = __expf(-5.f * sg);
        }
    }
  }
  __syncthreads();
#if defined(STATIC_TIMING_BUILD) && defined(STATIC_DRD_STAMP_S3)
  if (tid == 0 && L.stage_stamps) { // (STATIC_DRD_STAMP_S3, timing build: stage stamp 1 = f_b and the decay made, before the pairs)
    L.stage_stamps[1] = gtime();
  }
#endif
  // 4. the pairs in order, each state written back after its step (the last one's after the count)
#pragma unroll
  for (int m = 0; m < P; m++) {
    if (m < np) {
      float const rq = sm[Ly::RED + 2 * m], rk = sm[Ly::RED + 2 * m + 1], beta = sm[Ly::BETA + m];
      float dk[16], kn[16], qn[16];
#pragma unroll
      for (int c = 0; c < 16; c += 4) {
        float4 const d4 = *reinterpret_cast<float4 const *>(sm + Ly::DEC + m * DK + kb + DRD_CL * c);
        float4 const k4 = *reinterpret_cast<float4 const *>(sm + Ly::KN + m * DK + kb + DRD_CL * c);
        float4 const q4 = *reinterpret_cast<float4 const *>(sm + Ly::QN + m * DK + kb + DRD_CL * c);
        dk[c] = d4.x, dk[c + 1] = d4.y, dk[c + 2] = d4.z, dk[c + 3] = d4.w;
        kn[c] = k4.x * rk, kn[c + 1] = k4.y * rk, kn[c + 2] = k4.z * rk, kn[c + 3] = k4.w * rk;
        qn[c] = q4.x * rq, qn[c + 1] = q4.y * rq, qn[c + 2] = q4.z * rq, qn[c + 3] = q4.w * rq;
      }
      float pp[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float a = 0.f;
#pragma unroll
        for (int c = 0; c < 16; c++) {
          S[m][j][c] *= dk[c];
          a = fmaf(S[m][j][c], kn[c], a);
        }
        pp[j] = drd_sum8(a);
      }
      float o[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float const d = beta * (sm[Ly::VV + m * DV + vb4 + j] - pp[j]);
        float a = 0.f;
#pragma unroll
        for (int c = 0; c < 16; c++) {
          S[m][j][c] = fmaf(d, kn[c], S[m][j][c]);
          a = fmaf(S[m][j][c], qn[c], a);
        }
        o[j] = drd_sum8(a);
      }
      if (kq == 0) { // (SGLang ROCm: o stored in bf16, then rms_norm_gated)
        *reinterpret_cast<float4 *>(sm + Ly::OO + m * DV + vb4) =
            make_float4(round_bf16(o[0]), round_bf16(o[1]), round_bf16(o[2]), round_bf16(o[3]));
      }
      int const sl = slots[rq_of(m)];
      if ((m + 1 < np || !1) && !STATIC_DRD_FLAT_LATE && sl >= 0) {
        float *sp = state_at(sl, hd_of(m));
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
          for (int c = 0; c < 4; c++) {
            drd_st_state(sp + (size_t)j * DK + DRD_CS * c, S[m][j][4 * c], S[m][j][4 * c + 1], S[m][j][4 * c + 2],
                         S[m][j][4 * c + 3]);
          }
      }
    }
  }
  __syncthreads();
#ifdef STATIC_TIMING_BUILD
#ifndef STATIC_DRD_STAMP_S3
  if (tid == 0 && L.stage_stamps) { // (stage stamp 1: the pairs' states updated, o made)
    L.stage_stamps[1] = gtime();
  }
#endif
#endif
  // 5. o_norm per pair, the gate, y
  for (int m = wave; m < np; m += 4) {
    float const oa = sm[Ly::OO + m * DV + lane], ob = sm[Ly::OO + m * DV + lane + 64];
    float s2 = oa * oa + ob * ob;
    s2 = wave_sum64(s2);
    if (lane == 0) {
      sm[Ly::RED + 2 * m] = rsqrtf(s2 / DV + eps);
    }
  }
  __syncthreads();
  for (int v = tid; v < np * 16; v += 256) { // pair v / 16, v 8 (v % 16) .. + 7
    int const m = v >> 4, vb = 8 * (v & 15);
    float const rs = sm[Ly::RED + 2 * m];
    float o8[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
      float const gv = sm[Ly::GT + m * DV + vb + e], wv = prm[heads + HD + vb + e];
      o8[e] = sm[Ly::OO + m * DV + vb + e] * rs * wv * (1.f / (1.f + __expf(-gv))); // (fused_norm_gate.py order)
    }
    st16_dev(y + (size_t)rq_of(m) * HD + hd_of(m) * 128 + vb,
             pack_bf16x8(make_float4(o8[0], o8[1], o8[2], o8[3]), make_float4(o8[4], o8[5], o8[6], o8[7])));
  }
  task_post(cnt, 1);
  // the last pair's state, in place, after the count: o_proj waits for y only (the next launch reads the pool)
  {
#pragma unroll
    for (int m = 0; m < P; m++) {
      if (STATIC_DRD_FLAT_LATE ? m < np : m == np - 1) {
        int const sl = slots[rq_of(m)];
        if (sl >= 0) {
          float *sp = state_at(sl, hd_of(m));
#ifdef STATIC_DRD_HOLD
          if (hold && m == np - 1) { // (into LDS; the next task, node STATIC_DRD_HOLD, stores it after its own work)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
              for (int c = 0; c < 4; c++) {
                *reinterpret_cast<float4 *>(L.sm + STATIC_DRD_HOLD_OFF + drd_hold_slot(j, c, tid)) =
                    make_float4(S[m][j][4 * c], S[m][j][4 * c + 1], S[m][j][4 * c + 2], S[m][j][4 * c + 3]);
              }
            L.hold_dst = sp;
            continue;
          }
#endif
#pragma unroll
          for (int j = 0; j < 4; j++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
              drd_st_state(sp + (size_t)j * DK + DRD_CS * c, S[m][j][4 * c], S[m][j][4 * c + 1], S[m][j][4 * c + 2],
                           S[m][j][4 * c + 3]);
            }
        }
      }
    }
  }
}

// task (x, y) = head x, requests R y .. + R - 1 (R = ceil(T / grid y)); CONV = the short_conv_decode node (its output,
// counter, tasks); params = [PROJ_N, COL_G, COL_FA, COL_BETA, eps bits]; buffer slots = W_fb, prm, the state pool, the
// per-request slots int32 [T]
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class CONV,
          class... IN>
__device__ __forceinline__ void run_delta_rule_decode(Maps const &,
                                                      G const &g,
                                                      KernelLocals &L,
                                                      StaticTask const &tk) {
  static_assert(SELF::buf >= 0 && SELF::counter >= 0, "delta_rule_decode: its output and a counter");
  if constexpr (drd::param_or0<PARAMS, 5>() == 1) { // CIR: CONV = the projection (its K parts), no conv node
    constexpr int HD = drd::param_or0<PARAMS, 6>();
    static_assert(CONV::buf >= 0 && CONV::v[0] == COMBINE_SLOTS && CONV::v[2] == PARAMS::v[0] && HD % 128 == 0,
                  "delta_rule_decode CIR: the projection's K parts in slots");
    delta_rule_decode_task<PARAMS::v[0], PARAMS::v[1], PARAMS::v[2], PARAMS::v[3], HD, PARAMS::v[4],
                           (T + SELF::y - 1) / SELF::y, CONV::y, true>(L,
                                                        nullptr,
                                                        buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
                                                        buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()),
                                                        buf_at<float>(g, slot_at<BUF_SLOTS, 2>()),
                                                        buf_at<int const>(g, slot_at<BUF_SLOTS, 3>()),
                                                        nullptr,
                                                        0u,
                                                        buf_at<__hip_bfloat16>(g, SELF::buf),
                                                        cnt_of(g) + SELF::counter,
                                                        tk.x,
                                                        tk.y,
                                                        buf_at<float const>(g, CONV::buf),
                                                        buf_at<float const>(g, slot_at<BUF_SLOTS, 4>()),
                                                        buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 5>())
#ifdef STATIC_DRD_L2NEXT
                                                        , L.nxt != nullptr && L.nxt[0] == tk.node ? make_int2(L.nxt[1], L.nxt[2])
                                                                                                  : make_int2(-1, 0)
#endif
                                                        );
    return;
  } else if constexpr (drd::param_or0<PARAMS, 5>() == 3) { // FLAT + CIR: P = params[6], HD = params[7]; CONV = the projection
    constexpr int HD = drd::param_or0<PARAMS, 7>();
    static_assert(CONV::buf >= 0 && CONV::v[0] == COMBINE_SLOTS && CONV::v[2] == PARAMS::v[0] && HD % 128 == 0,
                  "delta_rule_decode FLAT CIR: the projection's K parts in slots");
    delta_rule_flat_task<PARAMS::v[0], PARAMS::v[1], PARAMS::v[2], PARAMS::v[3], HD, PARAMS::v[4],
                         drd::param_or0<PARAMS, 6>(), CONV::y, true>(L,
                                                                    nullptr,
                                                                    buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
                                                                    buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()),
                                                                    buf_at<float>(g, slot_at<BUF_SLOTS, 2>()),
                                                                    buf_at<int const>(g, slot_at<BUF_SLOTS, 3>()),
                                                                    nullptr,
                                                                    0u,
                                                                    buf_at<__hip_bfloat16>(g, SELF::buf),
                                                                    cnt_of(g) + SELF::counter,
                                                                    tk.x,
                                                                    buf_at<float const>(g, CONV::buf),
                                                                    buf_at<float const>(g, slot_at<BUF_SLOTS, 4>()),
                                                                    buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 5>()));
    return;
  } else if constexpr (drd::param_or0<PARAMS, 5>() == 2) { // FLAT: P = params[6] pairs per task
    static_assert(CONV::buf >= 0 && CONV::counter >= 0 && drd::first_input<IN...>::type::buf < 0,
                  "delta_rule_decode FLAT: short_conv_decode's output and counter, no projection input");
    delta_rule_flat_task<PARAMS::v[0], PARAMS::v[1], PARAMS::v[2], PARAMS::v[3], CONV::v[0] / 3, PARAMS::v[4],
                         drd::param_or0<PARAMS, 6>()>(L,
                                                     buf_at<__hip_bfloat16 const>(g, CONV::buf),
                                                     buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
                                                     buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()),
                                                     buf_at<float>(g, slot_at<BUF_SLOTS, 2>()),
                                                     buf_at<int const>(g, slot_at<BUF_SLOTS, 3>()),
                                                     cnt_of(g) + CONV::counter,
                                                     (uint32_t)(CONV::x * CONV::y * CONV::z),
                                                     buf_at<__hip_bfloat16>(g, SELF::buf),
                                                     cnt_of(g) + SELF::counter,
                                                     tk.x);
    return;
  } else {
  static_assert(CONV::buf >= 0 && CONV::counter >= 0, "delta_rule_decode: short_conv_decode's output and counter");
  constexpr int HD = CONV::v[0] / 3;
  using P1 = typename drd::first_input<IN...>::type; // the projection (its K parts) with a conv only for q / k / v, else W_fb
  constexpr int PPART = P1::buf >= 0 ? P1::y : 0;
  static_assert(PPART == 0 || (P1::v[0] == COMBINE_SLOTS && P1::v[2] == PARAMS::v[0]),
                "delta_rule_decode: proj = the projection's K parts in slots");
  delta_rule_decode_task<PARAMS::v[0], PARAMS::v[1], PARAMS::v[2], PARAMS::v[3], HD, PARAMS::v[4],
                         (T + SELF::y - 1) / SELF::y, PPART>(L,
                                                      buf_at<__hip_bfloat16 const>(g, CONV::buf),
                                                      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
                                                      buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()),
                                                      buf_at<float>(g, slot_at<BUF_SLOTS, 2>()),
                                                      buf_at<int const>(g, slot_at<BUF_SLOTS, 3>()),
                                                      cnt_of(g) + CONV::counter,
                                                      (uint32_t)(CONV::x * CONV::y * CONV::z),
                                                      buf_at<__hip_bfloat16>(g, SELF::buf),
                                                      cnt_of(g) + SELF::counter,
                                                      tk.x,
                                                      tk.y,
                                                      PPART > 0 ? buf_at<float const>(g, P1::buf) : nullptr
#ifdef STATIC_DRD_L2NEXT
                                                      , nullptr, nullptr,
                                                      L.nxt != nullptr && L.nxt[0] == tk.node ? make_int2(L.nxt[1], L.nxt[2])
                                                                                              : make_int2(-1, 0)
#endif
                                                      );
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_delta_rule_decode() {
  if constexpr (drd::param_or0<PARAMS, 5>() == 2 || drd::param_or0<PARAMS, 5>() == 3) { // FLAT (3: + CIR)
    return drd::Lay<drd::param_or0<PARAMS, 6>()>::END * 4;
  }
  return drd::Lay<(T + SELF::y - 1) / SELF::y>::END * 4;
}

} // namespace static_mk
