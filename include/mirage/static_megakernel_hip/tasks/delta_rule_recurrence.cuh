// tasks/delta_rule_recurrence.cuh (HIP, gfx950) -- K3 KDA's gated delta-rule
// recurrence, one head per task, the T tokens in order (math: mpk-kernels
// tests/kda_reference.py core(); B300 design: mirage_kda include/mirage/
// static_megakernel/tasks/delta_rule_recurrence.cuh). Task x = head x of grid
// (heads, 1, 1). Inputs: short_conv_silu's output (after its counter): the
// bf16 row [T][PROJ_N] (g at COL_G, f_a at COL_FA, beta at COL_BETA) and q / k /
// v in fp32 ([T][3 HD] after the row); W_fb [HD][128] bf16; prm = A_log
// [heads], dt_bias [HD], the output norm's weight [128] (fp32); the state from
// the framework's pool [slots][heads][128 v][128 k] fp32 at *slot (read, not
// written: the framework folds the accepted tokens from the rings).
//   f_b[t][k] = bf16(f_a[t] . W_fb[128 h + k]); gk = -5 sigmoid(exp(A_log) (f_b
//   + dt_bias)) (natural log); beta = sigmoid(b); q^ = q / |q| / sqrt(128),
//   k^ = k / |k| (eps 1e-6); per token: S *= exp(gk) (per k); d = beta (v - S
//   k^); S += d k^T; o = S q^; then y = bf16(o rsqrt(mean_v o^2 + eps) w_v
//   sigmoid(g)) -> the node's output (0xFF-prefilled: o_proj polls it);
//   per token gk -> ring_g [slots][heads][L][128] and beta -> ring_beta
//   [slots][heads][L] (fp32, L = info[3]).
// Threads: thread i holds S[v = 4 (i / 8) .. + 3][k = 16 (i % 8) .. + 15] in
// registers (as pairs: packed fp32 arithmetic); per token it reads 16 k, q and
// decay values from LDS (LDS bandwidth bound the token loop: 64 per thread
// before), the k reductions are shuffles over 8 lanes. f_b by MFMA 16x16x32
// bf16 (W_fb fragments in registers from the task's start). One count per head.
#pragma once
#include "../core.cuh"

// The token loop's sums over the 8 lanes of a row block by DPP moves (quad_perm xor 1, xor 2, row_half_mirror; MI355X
// 10-13 kda16: with the wave sums by DPP 52.8 / 53.1 vs 56.8 / 57.2 us).
// The decay folded into the token vectors (MI355X 10-15 kda67: -0.2..-0.5 us): with P_m = prod_{i <= m} exp(gk_i) (per
// k), a_m = P_m k_m, b_m = k_m / P_m, c_m = P_m q_m (P_m >= e^-40: no underflow).
// The T tokens in chunk form (the state is not written back: only o is needed). With e_i = rk_i d_i: p_m = S0 a_m +
// sum_{i<m} e_i (b_i . a_m), e_m = rk_m beta_m (v_m - rk_m p_m), o_m = S0 c_m + sum_{i<=m} e_i (b_i . c_m). S0 [a_0 ..
// a_7 | c_0 .. c_7] by fp32 MFMA 16x16x4 (the state loaded in its A layout: wave w rows 32 w .. + 31; no chain over the
// tokens: the sequential loop is VALU-bound at one wave per SIMD), the Gram terms b_i . a_m, b_i . c_m once per head,
// then per v row (thread v < 128) an 8-step substitution.

namespace static_mk {

namespace drr {
constexpr int DK = 128, DV = 128;
// LDS (floats from L.sm): f_a [T][128], k^ [T][128], q^ [T][128], decay
// [T][128], v [T][128], o [T][128], beta [T], the norms' partials
constexpr int FA = 0, KN = FA + T * DK, QN = KN + T * DK, DEC = QN + T * DK,
              VV = DEC + T * DK, OO = VV + T * DV, BETA = OO + T * DV,
              RED = BETA + T, FAB = RED + 4 * T * 2, GK = FAB + T * DK / 2, // (FAB: f_a in bf16 [T][128])
              GG = GK,                // (GG: the Gram terms [2][T][T])
              PD = GG + 2 * T * T,    // (PD: the S0 products [2 T][DV])
              END = PD + 2 * T * DV;
} // namespace drr

typedef float drr_v2f __attribute__((ext_vector_type(2)));

__device__ __forceinline__ float drr_sum8(float v) { // the sum over lanes 8 k .. 8 k + 7 (every lane)
  v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0xB1, 0xF, 0xF, false));  // quad_perm [1,0,3,2]
  v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x4E, 0xF, 0xF, false));  // quad_perm [2,3,0,1]
  v += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), 0x141, 0xF, 0xF, false)); // row_half_mirror
  return v;
}

template <int PROJ_N, int COL_G, int COL_FA, int COL_BETA, int HD, int EPS_BITS>
__device__ __forceinline__ void delta_rule_recurrence_task(
    KernelLocals &L,
    __hip_bfloat16 const *row,   // [T][PROJ_N] bf16, then [T][3 HD] fp32
    __hip_bfloat16 const *wfb,   // [HD][128]
    float const *prm,            // A_log [heads], dt_bias [HD], o_norm w [128]
    float const *state_pool,     // [slots][heads][128][128]
    float *ring_g,
    float *ring_beta,
    int const *info,
    int const *slot_p,
    uint32_t const *in_cnt,      // the conv's counter, complete at in_need
    uint32_t in_need,
    __hip_bfloat16 *y,           // [T][HD]
    uint32_t *cnt,
    int head STAGE_STAMP_PARAM) {
  using namespace drr;
  static_assert(PROJ_N % 8 == 0 && COL_G % 8 == 0 && COL_FA % 8 == 0 && HD % 128 == 0,
                "delta_rule_recurrence: 16-B aligned columns, whole heads");
  constexpr int heads = HD / 128;
  float const eps = __int_as_float(EPS_BITS);
  int const tid = task_tid_all(), lane = tid & 63, wave = tid >> 6, tok = lane % 16, grp = lane / 16;
  int const kq = tid & 7, kb = 16 * kq, vb4 = 4 * (tid >> 3); // the state block: rows vb4 .. + 3, columns kb .. + 15
  int const ring_len = info[3], slot = slot_p[0];
  // 1. what needs no input: this thread's state block, its W_fb fragments, the
  // parameters (in flight while the conv finishes)
  // the state in the fp32 MFMA's A layout: tile i = rows v = 32 wave + 16 i + l % 16,
  // lane group l / 16 holds k = 32 (l / 16) + s, s = 0 .. 31 (the MFMA's K slot)
  float Sm[2][32];
#pragma unroll
  for (int i = 0; i < 2; i++) {
    float4 const *sp = reinterpret_cast<float4 const *>(
        state_pool + (((size_t)slot * heads + head) * DV + 32 * wave + 16 * i + tok) * DK + 32 * grp);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float4 const s4 = sp[j];
      Sm[i][4 * j] = s4.x;
      Sm[i][4 * j + 1] = s4.y;
      Sm[i][4 * j + 2] = s4.z;
      Sm[i][4 * j + 3] = s4.w;
    }
  }
  // W_fb^T rows k = 16 (2 wave + i) + l % 16, columns c = 32 s + 8 (l / 16) .. + 7
  v4f wfa[2][4];
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int st = 0; st < 4; st++) {
      wfa[i][st] = *reinterpret_cast<v4f const *>(wfb + ((size_t)head * 128 + 16 * (2 * wave + i) + tok) * 128 +
                                                  32 * st + 8 * grp);
    }
  float const a_exp = __expf(prm[head]);
  float dtb[2][4]; // dt_bias of the k rows this lane gets from the MFMA
#pragma unroll
  for (int i = 0; i < 2; i++)
#pragma unroll
    for (int r = 0; r < 4; r++) {
      dtb[i][r] = prm[heads + head * 128 + 16 * (2 * wave + i) + 4 * grp + r];
    }
  float wvo[8]; // o_norm's weights of this thread's 8 v (step 5)
#pragma unroll
  for (int e = 0; e < 8; e++) {
    wvo[e] = prm[heads + HD + 8 * (tid & 15) + e];
  }
  float *const sm = reinterpret_cast<float *>(L.sm);
  __hip_bfloat16 *const fab = reinterpret_cast<__hip_bfloat16 *>(sm + FAB);
  // the state, W_fb and parameter loads complete before the wait (the compiler
  // may not sink them past this point into the steps after the conv)
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
  __syncthreads(); // the LDS may still be read by the previous task
  // 2. the conv's outputs
  if (tid == 0) {
    cnt_wait_in(in_cnt, in_need);
#if STATIC_STAMP_SET == 0
    STAGE_STAMP_AT(0) // (timing build: stage stamp 0 = the conv's outputs complete)
#endif
  }
  __syncthreads();
  // (another XCD's CUs wrote them past their L2: device-scope loads, every one
  // of them issued before the first use: one memory round trip)
  float const *qkv = reinterpret_cast<float const *>(row + (size_t)T * PROJ_N);
  int const mq = tid >> 5, cq = 4 * (tid & 31);
  size_t const qo = ((size_t)mq * 3 * HD + head * 128 + cq) * 4;
  uint4 const q4 = ld16_at<AUX_DEV>(qkv, (uint32_t)qo), k4 = ld16_at<AUX_DEV>(qkv, (uint32_t)(qo + HD * 4)),
              v4 = ld16_at<AUX_DEV>(qkv, (uint32_t)(qo + 2 * HD * 4));
  int const mf = (tid >> 4) & (T - 1), cf = 8 * (tid & 15);
  uint4 const f = ld16_at<AUX_DEV>(row, (uint32_t)(((size_t)mf * PROJ_N + COL_FA + cf) * 2));
  int const mg = tid >> 4, vg = 8 * (tid & 15); // (step 5: the gate of token mg, v vg .. + 7, threads < T x 16)
  uint32_t const g_off = (uint32_t)(((size_t)(mg & (T - 1)) * PROJ_N + COL_G + head * 128 + vg) * 2);
  uint4 const g4 = ld16_at<AUX_DEV>(row, g_off);
  size_t const eb = (size_t)(tid & (T - 1)) * PROJ_N + COL_BETA + head;
  uint4 const b4 = ld16_at<AUX_DEV>(row, (uint32_t)((eb & ~(size_t)7) * 2));
  *reinterpret_cast<uint4 *>(sm + QN + mq * DK + cq) = q4;
  *reinterpret_cast<uint4 *>(sm + KN + mq * DK + cq) = k4;
  *reinterpret_cast<uint4 *>(sm + VV + mq * DV + cq) = v4;
  if (tid < T * 16) { // f_a: 8 bf16 per thread (the MFMA's B operand)
    *reinterpret_cast<uint4 *>(fab + mf * DK + cf) = f;
  }
  if (tid < T) { // beta: the 16-B vector holding it
    uint32_t const bw[4] = {b4.x, b4.y, b4.z, b4.w};
    int const j = (int)(eb & 7);
    float const b = (j & 1) ? bf16_hi(bw[j / 2]) : bf16_lo(bw[j / 2]);
    float const sb = 1.f / (1.f + __expf(-b));
    sm[BETA + tid] = sb;
    ring_beta[((size_t)slot * heads + head) * ring_len + tid] = sb;
  }
  __syncthreads();
#if STATIC_STAMP_SET == 1
  if (tid == 0) { // (stage stamp set 1: stamp 0 = q / k / v, f_a, beta loaded)
    STAGE_STAMP_AT(0)
  }
#endif
  // 3. per token: the norms of q and k (wave w: tokens 2 w, 2 w + 1); f_b =
  // bf16(f_a . W_fb^T) by MFMA (D: lane -> token l % 16, register r -> k = 16
  // (2 wave + i) + 4 (l / 16) + r) and the decay
  {
    int const w = tid >> 6;
#pragma unroll
    for (int mm = 0; mm < 2; mm++) {
      int const m = 2 * w + mm;
      float qa = sm[QN + m * DK + lane], qb = sm[QN + m * DK + lane + 64];
      float ka = sm[KN + m * DK + lane], kb2 = sm[KN + m * DK + lane + 64];
      float sq = qa * qa + qb * qb, sk = ka * ka + kb2 * kb2;
      sq = wave_sum64(sq);
      sk = wave_sum64(sk);
      if (lane == 0) {
        sm[RED + 2 * m] = rsqrtf(sq + 1e-6f) * 0.08838834764831845f; // 128^-0.5
        sm[RED + 2 * m + 1] = rsqrtf(sk + 1e-6f);
      }
    }
  }
  {
    v4f acc[2] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int st = 0; st < 4; st++) {
      v4f b = {0.f, 0.f, 0.f, 0.f};
      if (tok < T) {
        b = *reinterpret_cast<v4f const *>(fab + tok * DK + 32 * st + 8 * grp);
      }
#pragma unroll
      for (int i = 0; i < 2; i++) {
        acc[i] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, wfa[i][st]), __builtin_bit_cast(v8bf, b),
                                                         acc[i], 0, 0, 0);
      }
    }
    if (tok < T) {
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int r = 0; r < 4; r++) {
          int const k = 16 * (2 * wave + i) + 4 * grp + r;
          float const fbv = round_bf16(acc[i][r]);
          float const sg = 1.f / (1.f + __expf(-a_exp * (fbv + dtb[i][r])));
          float const gk = -5.f * sg;
          sm[DEC + tok * DK + k] = __expf(gk);
          ring_g[(((size_t)slot * heads + head) * ring_len + tok) * DK + k] = gk;
        }
    }
  }
  __syncthreads();
#if STATIC_STAMP_SET == 5
  if (tid == 0) { // (stamp set 5: stamp 0 = the norms, f_b and the decay done, 1 = FOLD's products done)
    STAGE_STAMP_AT(0)
  }
#endif
  if (tid < DK) { // k = tid: KN <- P_m k, QN <- P_m q, DEC <- k / P_m (the norms above are of the raw k, q)
    float pm = 1.f;
#pragma unroll
    for (int m = 0; m < T; m++) {
      pm *= sm[DEC + m * DK + tid];
      float const kv = sm[KN + m * DK + tid];
      sm[KN + m * DK + tid] = kv * pm;
      sm[QN + m * DK + tid] *= pm;
      sm[DEC + m * DK + tid] = kv / pm;
    }
  }
  __syncthreads();
#if STATIC_STAMP_SET == 5
  if (tid == 0) {
    STAGE_STAMP_AT(1)
  }
#endif
#if STATIC_STAMP_SET == 1
  if (tid == 0) { // (stage stamp set 1: stamp 1 = f_b and the decay done)
    STAGE_STAMP_AT(1)
  }
#endif
#if STATIC_STAMP_SET == 2
  if (tid == 0) { // (stage stamp set 2: stamp 0 = the token loop's start)
    STAGE_STAMP_AT(0)
  }
#endif
  static_assert(T == 8, "delta_rule_recurrence: CHUNK's thread maps (2 T T Gram terms x 2 halves = 256 threads)");
  { // 4a. the Gram terms: thread -> (which, i, m), half h of the 128 k
    int const cb = tid >> 1, h = tid & 1, wh = cb >> 6, gi = (cb >> 3) & 7, gm = cb & 7;
    float const *const bv = sm + DEC + gi * DK + 64 * h, *const xv = sm + (wh ? QN : KN) + gm * DK + 64 * h;
    // (16-B pieces in a rotated order: the lanes' rows are 512 B apart, so piece
    // (j + 2 gm + h) mod 16 puts the 16 rows a wave reads in different banks;
    // 8 pieces of each vector in flight per batch)
    int const rot = 2 * gm + h;
    drr_v2f ga = {0.f, 0.f};
#pragma unroll
    for (int jb = 0; jb < 16; jb += 8) {
      float4 b8[8], x8[8];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        int const jr = (jb + j + rot) & 15;
        b8[j] = *reinterpret_cast<float4 const *>(bv + 4 * jr);
        x8[j] = *reinterpret_cast<float4 const *>(xv + 4 * jr);
      }
#pragma unroll
      for (int j = 0; j < 8; j++) {
        ga = __builtin_elementwise_fma(drr_v2f{b8[j].x, b8[j].y}, drr_v2f{x8[j].x, x8[j].y}, ga);
        ga = __builtin_elementwise_fma(drr_v2f{b8[j].z, b8[j].w}, drr_v2f{x8[j].z, x8[j].w}, ga);
      }
    }
    float gs = ga.x + ga.y;
    gs += __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(gs), 0xB1, 0xF, 0xF, false)); // + the other half
    if (h == 0) {
      sm[GG + cb] = gs;
    }
  }
  { // 4b. S0 [a | c] by MFMA (B: lane -> column n = l % 16 (a_n, then c_{n - 8}), k = 32 (l / 16) + s; D: lane ->
    // column l % 16, register r -> row 4 (l / 16) + r of the tile)
    int const n = tok;
    float const *const bsrc = sm + (n < T ? KN + n * DK : QN + (n - T) * DK) + 32 * grp;
    float4 bq[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      bq[j] = *reinterpret_cast<float4 const *>(bsrc + 4 * j);
    }
    float const *const bf = reinterpret_cast<float const *>(bq);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      v4f acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int s_ = 0; s_ < 32; s_++) {
        acc = __builtin_amdgcn_mfma_f32_16x16x4f32(Sm[i][s_], bf[s_], acc, 0, 0, 0);
      }
      *reinterpret_cast<float4 *>(sm + PD + n * DV + 32 * wave + 16 * i + 4 * grp) =
          make_float4(acc[0], acc[1], acc[2], acc[3]);
    }
  }
#if STATIC_STAMP_SET == 3
  if (tid == 0) { // (stamp set 3, CHUNK: stamp 0 = S0 a_m, S0 c_m in LDS (thread 0))
    STAGE_STAMP_AT(0)
  }
#endif
  __syncthreads(); // the Gram terms, the products
  if (tid < DV) { // 4c. row v = tid: e_m in token order, then o_m
    int const v = tid;
    float gg[2 * T * T];
#pragma unroll
    for (int i = 0; i < 2 * T * T / 4; i++) {
      *reinterpret_cast<float4 *>(gg + 4 * i) = *reinterpret_cast<float4 const *>(sm + GG + 4 * i);
    }
    float pa[T], pc[T], vr[T], e[T];
#pragma unroll
    for (int m = 0; m < T; m++) {
      pa[m] = sm[PD + m * DV + v];
      pc[m] = sm[PD + (T + m) * DV + v];
      vr[m] = sm[VV + m * DV + v];
    }
#pragma unroll
    for (int m = 0; m < T; m++) {
      float const rkm = sm[RED + 2 * m + 1];
      float p = pa[m];
#pragma unroll
      for (int i = 0; i < m; i++) {
        p = fmaf(e[i], gg[i * T + m], p);
      }
      e[m] = rkm * (sm[BETA + m] * (vr[m] - p * rkm));
    }
#pragma unroll
    for (int m = 0; m < T; m++) {
      float o = pc[m];
#pragma unroll
      for (int i = 0; i <= m; i++) {
        o = fmaf(e[i], gg[T * T + i * T + m], o);
      }
      sm[OO + m * DV + v] = round_bf16(o * sm[RED + 2 * m]); // (SGLang ROCm: o stored in bf16, then rms_norm_gated)
    }
  }
#if STATIC_STAMP_SET == 3
  if (tid == 0) { // (stamp 1 = the substitution done (thread 0))
    STAGE_STAMP_AT(1)
  }
#endif
  __syncthreads();
#if STATIC_STAMP_SET == 0
  if (tid == 0) { // (stage stamp 1: the token loop done)
    STAGE_STAMP_AT(1)
  }
#endif
  // 5. o_norm per token (wave w: tokens 2 w, 2 w + 1), the gate, y
  {
    int const w = tid >> 6, lane = tid & 63;
#pragma unroll
    for (int mm = 0; mm < 2; mm++) {
      int const m = 2 * w + mm;
      float const oa = sm[OO + m * DV + lane], ob = sm[OO + m * DV + lane + 64];
      float s2 = oa * oa + ob * ob;
      s2 = wave_sum64(s2);
      if (lane == 0) {
        sm[RED + 2 * m] = rsqrtf(s2 / DV + eps);
      }
    }
  }
  __syncthreads();
  if (tid < T * DV / 8) { // token tid / 16, v 8 (tid % 16) .. + 7
    int const m = mg, vb = vg;
    float o8[8];
    float const rs = sm[RED + 2 * m];
    uint32_t const gw[4] = {g4.x, g4.y, g4.z, g4.w};
#pragma unroll
    for (int e = 0; e < 8; e++) {
      float const gv = (e & 1) ? bf16_hi(gw[e / 2]) : bf16_lo(gw[e / 2]);
      float const wv = wvo[e];
      o8[e] = sm[OO + m * DV + vb + e] * rs * wv * (1.f / (1.f + __expf(-gv))); // (fused_norm_gate.py order)
    }
    st16_dev(y + (size_t)m * HD + head * 128 + vb,
             pack_bf16x8(make_float4(o8[0], o8[1], o8[2], o8[3]),
                         make_float4(o8[4], o8[5], o8[6], o8[7])));
  }
  task_post(cnt, 1);
}

// task x = head x; CONV = the short_conv_silu node (its output, counter,
// tasks); params = [PROJ_N, COL_G, COL_FA, COL_BETA, eps bits]; buffer slots =
// W_fb, prm, the state pool, ring_g, ring_beta, info
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class CONV,
          class... IN>
__device__ __forceinline__ void run_delta_rule_recurrence(Maps const &,
                                                          G const &g,
                                                          KernelLocals &L,
                                                          StaticTask const &tk) {
  static_assert(CONV::buf >= 0 && CONV::counter >= 0,
                "delta_rule_recurrence: short_conv_silu's output and counter");
  static_assert(SELF::buf >= 0 && SELF::counter >= 0,
                "delta_rule_recurrence: its output and a counter");
  constexpr int HD = CONV::v[0] / 3;
  delta_rule_recurrence_task<PARAMS::v[0], PARAMS::v[1], PARAMS::v[2], PARAMS::v[3], HD,
                             PARAMS::v[4]>(
      L,
      buf_at<__hip_bfloat16 const>(g, CONV::buf),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<float>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<float>(g, slot_at<BUF_SLOTS, 4>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 5>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 6>()),
      cnt_of(g) + CONV::counter,
      (uint32_t)(CONV::x * CONV::y * CONV::z),
      buf_at<__hip_bfloat16>(g, SELF::buf),
      cnt_of(g) + SELF::counter,
      tk.x STAGE_STAMP_ARG(L.stage_stamps));
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_delta_rule_recurrence() {
  return drr::END * 4;
}

} // namespace static_mk
