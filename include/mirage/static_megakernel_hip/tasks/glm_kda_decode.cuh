// tasks/glm_kda_decode.cuh (HIP, gfx950) -- GLM-5.3-Flash's KDA attention in plain decode: K3's delta_rule_decode (the
// same gate -5 sigmoid(exp(A_log) (f_b + dt_bias)), delta rule on the [V][K] fp32 state, sigmoid-gated o_norm, the same conv
// pool / state layouts) with GLM's low-rank output gate: g = bf16(g_a . W_gb^T) per head, made in the task like f_b (a
// second MFMA pass; g_a = 128 columns of the projection shared by the heads). Only K3's CIR path (the 4-tap conv in the
// task, the projection's K parts polled), one task per (head, R requests). Task (x, y) = head x, requests R y .. + R - 1.
#pragma once
// STATIC_DRD_COAL default 1 for GLM (delta_rule_decode.cuh keeps K3's default 0; GLM kernels include it only through this file):
// a thread's state columns 4 kq + 32 c', one wave load / store = 8 whole 128-B lines. MI355X 11-09 glmkda33 (checks pass, A B A
// B): KDA T 64 97.6 / 97.4 vs 106.5 / 107.2 us (decode ends 45.5 vs 58.1), T 32 75.1 / 75.2 vs 80.7 / 81.0; KDA + MoE pair T 64
// 409.3 / 400.1 vs 417.4 / 411.3, T 32 306.8 / 297.7 vs 308.8 / 300.6
#ifndef STATIC_DRD_COAL
#define STATIC_DRD_COAL 1
#endif
#include "delta_rule_decode.cuh" // drd:: helpers (Lay, drd_proj4, drd_sum8, ...)

// STATIC_DRD_STATE_WT (0 default; 2: amd/serving/glm/build_glm_libs.sh sets it for the KDA + MoE pairs at T >= 32): the new
// states stored nt sc1 (not kept in the L2) instead of plain stores - the 64 MB at
// T 64 drain during the decode, not as dirty L2 lines written back later (MI355X 11-08 glmpair7: in the one-kernel layer the
// experts phase after the KDA part was +14 us, +3 with no state stores)
#ifndef STATIC_DRD_STATE_WT
#define STATIC_DRD_STATE_WT 0
#endif

namespace static_mk {
__device__ __forceinline__ void drd_state_st(float *p, float a, float b, float c, float d) {
  static_assert(STATIC_DRD_STATE_WT == 0 || STATIC_DRD_STATE_WT == 2, "STATIC_DRD_STATE_WT: 0 or 2");
  if constexpr (STATIC_DRD_STATE_WT == 2) { // (2: nt sc1 - not kept in the L2)
    st16_dev_nt(p, make_uint4(__float_as_uint(a), __float_as_uint(b), __float_as_uint(c), __float_as_uint(d)));
  } else {
    *reinterpret_cast<float4 *>(p) = make_float4(a, b, c, d);
  }
}

// LDS: K3's (Lay), then g_a in bf16 [R][128]
template <int R>
struct GLay : drd::Lay<R> {
  static constexpr int GAB = drd::Lay<R>::END, END = GAB + R * drd::DK / 2;
};

template <int PROJ_N, int COL_G, int COL_FA, int COL_BETA, int HD, int EPS_BITS, int R, int PPART, bool CIR = true>
__device__ __forceinline__ void glm_kda_decode_task(KernelLocals &L,
                                                       __hip_bfloat16 const *row, // [T][PROJ_N] bf16, then [T][3 HD] fp32
                                                       __hip_bfloat16 const *wfb, // [HD][128]
                                                       __hip_bfloat16 const *wgb, // [HD][128]: g_b (the output gate, low rank)
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
  using Ly = GLay<R>;
  static_assert(PROJ_N % 8 == 0 && COL_G % 8 == 0 && COL_FA % 8 == 0 && HD % 128 == 0 && R >= 1 && R <= 16,
                "delta_rule_decode: 16-B aligned columns, whole heads, at most 16 requests per task");
  static_assert(!CIR || PPART > 0, "delta_rule_decode CIR: the projection's K parts");
  static_assert(CIR && PPART > 0 && !STATIC_DRD_G_EARLY, "glm_kda_decode: the conv in the task (CIR), g from g_a (no G_EARLY)");
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
    } else if (tid < 161) { // g_a (128 columns, every head's)
      ecol = COL_G + 4 * (tid - 129);
    }
    if (tid < 161) {
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
  v4f wga[2][4]; // W_gb^T, as W_fb's
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int st = 0; st < 4; st++) {
      wga[i][st] = *reinterpret_cast<v4f const *>(wgb + ((size_t)head * 128 + 16 * (2 * wave + i) + tok) * 128 + 32 * st +
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
  __hip_bfloat16 *const gab = reinterpret_cast<__hip_bfloat16 *>(sm + Ly::GAB);
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
    if (tid < 161) { // (the item's loads issued at the start)
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
      } else { // g_a
        float4 const g4 = drd_proj4_finish<PPART, PROJ_N>(proj, t, ecol, ezp);
        *reinterpret_cast<uint2 *>(gab + 4 * (v - 129)) = make_uint2(pack_bf16x2(g4.x, g4.y), pack_bf16x2(g4.z, g4.w));
      }
    }
    if (false)
#endif
    for (int v = tid; v < nr * 161; v += 256) {
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
      } else if (v < nr * 129) {
        int const m = v - nr * 128, cb = COL_BETA + head;
        float4 const b4 = drd_proj4<PPART, PROJ_N>(proj, r0 + m, cb & ~3);
        float const bv[4] = {b4.x, b4.y, b4.z, b4.w};
        sm[Ly::BETA + m] = 1.f / (1.f + __expf(-bv[cb & 3]));
      } else { // g_a
        int const m = (v - nr * 129) >> 5, c = 4 * ((v - nr * 129) & 31);
        float4 const g4 = drd_proj4<PPART, PROJ_N>(proj, r0 + m, COL_G + c);
        *reinterpret_cast<uint2 *>(gab + m * DK + c) = make_uint2(pack_bf16x2(g4.x, g4.y), pack_bf16x2(g4.z, g4.w));
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
    for (int v = tid; v < nr * 32; v += 256) { // request v / 32: g_a columns 4 (v % 32) .. + 3
      int const m = v >> 5, c = 4 * (v & 31);
      float4 const g4 = drd_proj4<PPART, PROJ_N>(proj, r0 + m, COL_G + c);
      *reinterpret_cast<uint2 *>(gab + m * DK + c) = make_uint2(pack_bf16x2(g4.x, g4.y), pack_bf16x2(g4.z, g4.w));
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
  { // the output gate: g = bf16(g_a . W_gb^T) by MFMA (as f_b: lane -> request l % 16, register r -> v = 16 (2 wave + i) +
    // 4 (l / 16) + r), to the gate's LDS rows
    v4f acc[2] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int st = 0; st < 4; st++) {
      v4f b = {0.f, 0.f, 0.f, 0.f};
      if (tok < nr) {
        b = *reinterpret_cast<v4f const *>(gab + tok * DK + 32 * st + 8 * grp);
      }
#pragma unroll
      for (int i = 0; i < 2; i++) {
        acc[i] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, wga[i][st]), __builtin_bit_cast(v8bf, b),
                                                         acc[i], 0, 0, 0);
      }
    }
    if (tok < nr) {
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int r = 0; r < 4; r++) {
          sm[Ly::GT + tok * DV + 16 * (2 * wave + i) + 4 * grp + r] = round_bf16(acc[i][r]);
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
    if (m + 1 < nr && s_cur >= 0) {
      float *sp = state_at(s_cur);
#pragma unroll
      for (int j = 0; j < 4; j++)
#pragma unroll
        for (int c = 0; c < 4; c++) {
          drd_state_st(sp + (size_t)j * DK + DRD_CS * c, S[j][4 * c], S[j][4 * c + 1], S[j][4 * c + 2], S[j][4 * c + 3]);
        }
    }
    if (m + 1 < nr) {
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
        drd_state_st(sp + (size_t)j * DK + DRD_CS * c, S[j][4 * c], S[j][4 * c + 1], S[j][4 * c + 2], S[j][4 * c + 3]);
      }
  }
}


// task (x, y) = head x, requests R y .. + R - 1 (R = ceil(T / grid y)); PROJ = the projection (gemm_tile, K parts in slots);
// params = [PROJ_N, COL_GA, COL_FA, COL_BETA, eps bits, HD]; buffer slots = W_fb, prm (A_log [heads], dt_bias [HD], o_norm
// weight [128]), the state pool, the per-request slots int32 [T], conv_w [3 HD][4] fp32, the conv pool, W_gb
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class PROJ, class... IN>
__device__ __forceinline__ void run_glm_kda_decode(Maps const &, G const &g, KernelLocals &L, StaticTask const &tk) {
  static_assert(SELF::buf >= 0 && SELF::counter >= 0, "glm_kda_decode: its output and a counter");
  constexpr int HD = PARAMS::v[5];
  static_assert(PROJ::buf >= 0 && PROJ::v[0] == COMBINE_SLOTS && PROJ::v[2] == PARAMS::v[0] && HD % 128 == 0,
                "glm_kda_decode: the projection's K parts in slots");
  glm_kda_decode_task<PARAMS::v[0], PARAMS::v[1], PARAMS::v[2], PARAMS::v[3], HD, PARAMS::v[4], (T + SELF::y - 1) / SELF::y,
                      PROJ::y, true>(L,
                                     nullptr,
                                     buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
                                     buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 6>()),
                                     buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()),
                                     buf_at<float>(g, slot_at<BUF_SLOTS, 2>()),
                                     buf_at<int const>(g, slot_at<BUF_SLOTS, 3>()),
                                     nullptr,
                                     0u,
                                     buf_at<__hip_bfloat16>(g, SELF::buf),
                                     cnt_of(g) + SELF::counter,
                                     tk.x,
                                     tk.y,
                                     buf_at<float const>(g, PROJ::buf),
                                     buf_at<float const>(g, slot_at<BUF_SLOTS, 4>()),
                                     buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 5>())
#ifdef STATIC_DRD_L2NEXT
                                     , L.nxt != nullptr && L.nxt[0] == tk.node ? make_int2(L.nxt[1], L.nxt[2]) : make_int2(-1, 0)
#endif
  );
}

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class... IN>
constexpr int smem_glm_kda_decode() {
  return GLay<(T + SELF::y - 1) / SELF::y>::END * 4;
}

} // namespace static_mk
