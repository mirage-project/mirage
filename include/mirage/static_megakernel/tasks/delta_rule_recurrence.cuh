// tasks/delta_rule_recurrence.cuh -- the gated delta-rule recurrence of a KDA
// layer (Kimi K3), one head per task, all T tokens, on the kernel's 256
// threads. Task x = head x of grid (heads, 1, 1).
//
// Math (WY form): cs[t][k] = cumulative log2 decay from f_b = f_a . W_fb;
//   W_t = k^_t 2^cs_t, B_s = k^_s 2^-cs_s, WQ_t = q_t 2^cs_t;
//   D1 = S0 . [W; WQ]^T (tensor core, tf32) gives P0[v][t], OQ0[v][t];
//   A[t][s] = W_t . B_s (s < t), AQ[t][s] = WQ_t . B_s (s <= t) (CUDA cores);
//   u_t = beta_t (v_t - P0_t - sum_{s<t} A[t,s] u_s);
//   o_t = rq_t (OQ0_t + sum_{s<=t} AQ[t,s] u_s) -> gated RMSNorm -> y_rec;
//   S_T = 2^cs_T (S0 + U . B) (D2 on the tensor core, bf16 hi/lo) -> state.
// Shared memory: the production layout from the 1024-aligned base, inside the
// GEMM ring's region [0, RING_BYTES): the ring's data is free at task entry (a
// GEMM task before it on the SM has consumed every stage it loaded) and is
// reloaded by TMA before any later GEMM reads it; the ring's barriers are not
// in this region. Entry and exit barriers make that hold. TMEM: columns
// tb + 64.. (f_b, then D1) and tb + 128.. (D2); the GEMM ring uses tb + 0..63.
// Inputs are read by pointer (graph tensors or node outputs whose producers
// finished): conv_out [3][T][heads x 128] (q, k, v), proj_out [T][PROJ_N] (g
// at COL_G, f_a at COL_FA, beta at COL_BETA), prm (A_log [heads], dt_bias
// [heads x 128], o_norm weight [128]) fp32, state [heads][128 v][128 k] fp32
// (read, then written back: the next launch's state). Output: y_rec [T][heads x
// 128] bf16 (the node's buffer), then one count on the node's counter.
// SERVE (params[8], serving builds; SGLang's KDA target-verify contract): the
// state is the framework's pool [slots][heads][128 v][128 k] at *slot, read and
// NOT written back (the framework folds the accepted tokens from the rings);
// per token the gate's natural-log decay -5 sigmoid(exp(A_log) (f_b + dt_bias))
// to ring_g [slots][heads][L][128] and sigmoid(beta) to ring_beta
// [slots][heads][L], fp32 (L: info[3])
#pragma once
#include "../core.cuh"

namespace static_mk {

// S0 is staged by cp.async (16 B each, no registers held) and waited for right
// before D1: phase 0 does not wait for the 64 KB load (10-10: phase 0 took 7.3
// us under the projection's HBM stream), so the dry pass runs while S0 lands

namespace rec {

constexpr int DK = 128, DV = 128;
// shared-memory layout (bytes from the 1024-aligned base), as production
constexpr int WFB = 0, FA = 32768, SQ = 34816, SK = SQ + T * DK * 4,
              SEG = SK + T * DK * 4, SVE = SEG + T * DK * 4,
              SGR = SVE + T * DV * 4, SOALL = SGR + T * DV * 4,
              SW = SOALL + T * DV * 4, SB = SW + T * DK * 4,
              SWQ = SB + T * DK * 4, SE8 = SWQ + T * DK * 4, SA = SE8 + DK * 4,
              SAQ = SA + T * T * 4, MS = SAQ + T * T * 4, // S0 fp32 [128][128]
    MT = MS + DV * DK * 4, // [W; WQ] fp32, 128-B swizzled (8 KB)
    MUH = MT + 2 * T * DK * 4, MUL = MUH + DV * 2 * T * 2,
              MBH = MUL + DV * 2 * T * 2, MBL = MBH + DK * 2 * T * 2,
              MBAR =
                  MBL + DK * 2 * T * 2, // 3 mbarriers: W_fb TMA, f_a, commits
    SBB = MBAR + 32, SRQ = SBB + T * 4, SRK = SRQ + T * 4,
              SQKP = SRK + T * 4, // [T][2 warps][q, k]
    END = SQKP + T * 2 * 2 * 4;
static_assert(MS % 1024 == 0 && MT % 1024 == 0 && FA % 1024 == 0,
              "the MMA operands' alignment");
static_assert(END <= RING_BYTES,
              "the recurrence fits in the GEMM ring's region");

// no-swizzle K-major element offsets (bf16 counts): BIG = 128-column operands,
// SMALL = 16-column operands; BIG32 = fp32, core matrix 8 rows x 4 elements
__device__ __forceinline__ int nsk_small(int r, int c) {
  return ((r >> 3) << 7) + ((c >> 3) << 6) + ((r & 7) << 3) + (c & 7);
}
__device__ __forceinline__ int nsk_big32(int r, int c) {
  return ((r >> 3) << 10) + ((c >> 2) << 5) + ((r & 7) << 2) + (c & 3);
}
// smem matrix descriptors: no swizzle (lbo / sbo in 16-B units), and 128-B
// swizzle with a 1024-B stride (the GEMM tiles' form)
__device__ __forceinline__ uint64_t ns_desc(uint32_t addr,
                                            int lbo16,
                                            int sbo16) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)lbo16 << 16) |
         ((uint64_t)sbo16 << 32) | (1ULL << 46);
}
constexpr uint64_t SW_DESC =
    ((uint64_t)((8 * 128 & 0x3FFFF) >> 4) << 32) | (1ULL << 46) | (2ULL << 61);
__device__ __forceinline__ uint64_t sw_desc(uint32_t addr) {
  return SW_DESC | ((uint64_t)addr >> 4);
}
// instruction descriptors: f32 accumulate; bf16 (1 << 7 | 1 << 10) or tf32 (2
// << 7 | 2 << 10); N >> 3 at bit 17, M >> 4 at bit 24
constexpr uint32_t ID_FB = (1U << 4) | (1U << 7) | (1U << 10) |
                           ((uint32_t)(T >> 3) << 17) | (8U << 24); // M128 N=T
constexpr uint32_t ID_D1 =
    (1U << 4) | (2U << 7) | (2U << 10) | (2U << 17) | (8U << 24); // tf32 N=16
constexpr uint32_t ID_D2 =
    (1U << 4) | (1U << 7) | (1U << 10) | (16U << 17) | (8U << 24); // N=128

__device__ __forceinline__ void
    mma_f16(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t en) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}" ::"r"(d),
      "l"(a),
      "l"(b),
      "r"(id),
      "r"(en));
}
__device__ __forceinline__ void
    mma_tf32(uint32_t d, uint64_t a, uint64_t b, uint32_t id, uint32_t en) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p;\n}" ::"r"(d),
      "l"(a),
      "l"(b),
      "r"(id),
      "r"(en));
}
__device__ __forceinline__ void rec_commit(uint32_t mbar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
               "cluster.b64 [%0];" ::"r"(mbar)
               : "memory");
}
// this warp's 32 TMEM lanes: lanes 32 (warp & 3) .. + 31 (a warp reaches only
// its own lane quarter); x8 / x16 / x32 columns from column `col`
__device__ __forceinline__ uint32_t lane_addr(uint32_t col) {
  return ((uint32_t)((threadIdx.x >> 5) & 3) * 32 << 16) + col;
}
__device__ __forceinline__ void rec_tmem_x8(uint32_t col, float *f) {
  asm volatile("tcgen05.fence::after_thread_sync;");
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
      : "=f"(f[0]),
        "=f"(f[1]),
        "=f"(f[2]),
        "=f"(f[3]),
        "=f"(f[4]),
        "=f"(f[5]),
        "=f"(f[6]),
        "=f"(f[7])
      : "r"(lane_addr(col)));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}
__device__ __forceinline__ void rec_tmem_x16(uint32_t col, float *f) {
  asm volatile("tcgen05.fence::after_thread_sync;");
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
               "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
               : "=f"(f[0]),
                 "=f"(f[1]),
                 "=f"(f[2]),
                 "=f"(f[3]),
                 "=f"(f[4]),
                 "=f"(f[5]),
                 "=f"(f[6]),
                 "=f"(f[7]),
                 "=f"(f[8]),
                 "=f"(f[9]),
                 "=f"(f[10]),
                 "=f"(f[11]),
                 "=f"(f[12]),
                 "=f"(f[13]),
                 "=f"(f[14]),
                 "=f"(f[15])
               : "r"(lane_addr(col)));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}
__device__ __forceinline__ void
    rec_tmem_x32_issue(uint32_t col, float *f) { // pair with rec_tmem_wait
  asm volatile("tcgen05.fence::after_thread_sync;");
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%"
      "20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, [%32];"
      : "=f"(f[0]),
        "=f"(f[1]),
        "=f"(f[2]),
        "=f"(f[3]),
        "=f"(f[4]),
        "=f"(f[5]),
        "=f"(f[6]),
        "=f"(f[7]),
        "=f"(f[8]),
        "=f"(f[9]),
        "=f"(f[10]),
        "=f"(f[11]),
        "=f"(f[12]),
        "=f"(f[13]),
        "=f"(f[14]),
        "=f"(f[15]),
        "=f"(f[16]),
        "=f"(f[17]),
        "=f"(f[18]),
        "=f"(f[19]),
        "=f"(f[20]),
        "=f"(f[21]),
        "=f"(f[22]),
        "=f"(f[23]),
        "=f"(f[24]),
        "=f"(f[25]),
        "=f"(f[26]),
        "=f"(f[27]),
        "=f"(f[28]),
        "=f"(f[29]),
        "=f"(f[30]),
        "=f"(f[31])
      : "r"(lane_addr(col)));
}
__device__ __forceinline__ void rec_tmem_wait() {
  asm volatile("tcgen05.wait::ld.sync.aligned;");
}
__device__ __forceinline__ float bf(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
__device__ __forceinline__ __nv_bfloat16 tobf(float v) {
  return __float2bfloat16(v);
}
__device__ __forceinline__ float sigmoid_fast(float x) {
  return __fdividef(1.0f, 1.0f + __expf(-x));
}
// the "not written yet" sentinel of these inputs: 0x8000 (bf16 -0.0, the
// producers store +0 for it), or 0xFFFF (a NaN) when the producer's buffer is
// filled with 0xFF before each launch (DAF: no counter wait)
template <unsigned S = 0x8000u>
__device__ __forceinline__ bool sent32(unsigned x) {
  return ((x & 0xffffu) == S) || ((x >> 16) == S);
}
template <unsigned S = 0x8000u>
__device__ __forceinline__ bool sent128(int4 v) {
  return sent32<S>(v.x) || sent32<S>(v.y) || sent32<S>(v.z) || sent32<S>(v.w);
}
__device__ __forceinline__ int4 ld_relaxed128(void const *p) {
  int4 v;
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
// the staging polls' loads: without the memory clobber the loads of one poll
// round are in flight together (one round trip, not one per load; with it the
// staging took 4.8 us from the counters to staged, 2.7 without)
__device__ __forceinline__ unsigned rec_poll32(void const *p) {
  unsigned v;
  asm volatile("ld.relaxed.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ int4 rec_poll128(void const *p) {
  int4 v;
  asm volatile("ld.relaxed.gpu.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p));
  return v;
}
__device__ __forceinline__ uint2 rec_poll64(void const *p) {
  uint2 v;
  asm volatile("ld.relaxed.gpu.global.v2.u32 {%0,%1}, [%2];"
               : "=r"(v.x), "=r"(v.y)
               : "l"(p));
  return v;
}
__device__ __forceinline__ bool sent64f(
    uint2 v) { // fp32 inputs of an armed buffer: 0xFFFFFFFF = not written yet
  return v.x == 0xFFFFFFFFu || v.y == 0xFFFFFFFFu;
}
__device__ __forceinline__ unsigned short rec_poll16(void const *p) {
  unsigned short v;
  asm volatile("ld.relaxed.gpu.global.b16 %0, [%1];" : "=h"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ float4 rec_pollf4(
    float const
        *p) { // fp32 16 B, no memory clobber (loads of a round in flight)
  float4 v;
  asm volatile("ld.relaxed.gpu.global.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p));
  return v;
}
__device__ __forceinline__ __nv_bfloat16
    rec_bf_pos0(float z) { // bf16, -0 stored as +0 (short_conv_silu's rounding)
  __nv_bfloat16 h = __float2bfloat16(z);
  return __bfloat16_as_ushort(h) == 0x8000u ? __float2bfloat16(0.f) : h;
}
// the 64 A / AQ dots: isAQ << 8 | row << 4 | ss
__constant__ unsigned short aq_map[64] = {
    16,  32,  33,  48,  49,  50,  64,  65,  66,  67,  80,  81,  82,
    83,  84,  96,  97,  98,  99,  100, 101, 112, 113, 114, 115, 116,
    117, 118, 256, 272, 273, 288, 289, 290, 304, 305, 306, 307, 320,
    321, 322, 323, 324, 336, 337, 338, 339, 340, 341, 352, 353, 354,
    355, 356, 357, 358, 368, 369, 370, 371, 372, 373, 374, 375};
constexpr int DOTS_PER_WARP = 10; // 64 dots over warps 1..7: dot (w - 1) + 7 q

// D1 = S0 (MS, no swizzle) x [W; WQ] (MT) on the tensor core, tf32, into the
// TMEM columns RC; thread 0
__device__ __forceinline__ void rec_d1_issue(uint32_t sb, uint32_t RC) {
  asm volatile("tcgen05.fence::after_thread_sync;");
  uint64_t a = ns_desc(sb + MS, 8, 256);
  uint32_t const th = sb + MT;
  uint64_t const bq[4] = {
      sw_desc(th), sw_desc(th + 2048), sw_desc(th + 4096), sw_desc(th + 6144)};
#pragma unroll
  for (int ks = 0; ks < 16; ks++) {
    mma_tf32(RC, a, bq[ks >> 2] + (uint64_t)(ks & 3) * 2, ID_D1, ks ? 1u : 0u);
    a += 16;
  }
}

} // namespace rec

template <int PROJ_N,
          int COL_G,
          int COL_FA,
          int COL_BETA,
          int QKV_SS,
          int QKV_SR,
          int DAF = 0,
          int ONORM_EPS_BITS = 0, // the gated RMSNorm's eps as fp32 bits (0:
                                  // 1e-6, the harness's)
          int SERVE = 0,
          int QKV32 = 0> // q / k / v in fp32 after conv's [T][PROJ_N] bf16 row
                         // (short_conv_silu QKV32; DAF only)
__device__ __forceinline__ void
    delta_rule_recurrence_task(Maps const &maps,
                               G const &g,
                               KernelLocals &L,
                               int wfb_map,
                               __nv_bfloat16 const *conv,
                               __nv_bfloat16 const *proj,
                               float const *prm_all,
                               float *state_all,
                               __nv_bfloat16 *yrec,
                               uint32_t *cnt,
                               uint32_t const *const wait_cnt[2],
                               uint32_t const wait_n[2],
                               int nheads,
                               int head,
                               int const *slot_p = nullptr,
                               float *ring_g = nullptr,
                               float *ring_beta = nullptr,
                               int const *info = nullptr) {
  using namespace rec;
  constexpr unsigned SNT = DAF ? 0xFFFFu : 0x8000u; // the inputs' sentinel
  static_assert(
      !QKV32 || DAF,
      "delta_rule_recurrence: fp32 q / k / v only from an armed buffer");
  int const tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int const HD = nheads * DK; // y_rec / conv row width
  char *const s = L.sm;
  uint32_t const sb = L.base;
  float *sQ = reinterpret_cast<float *>(s + SQ),
        *sK = reinterpret_cast<float *>(s + SK);
  float *sEG = reinterpret_cast<float *>(s + SEG),
        *sVE = reinterpret_cast<float *>(s + SVE);
  float *sGR = reinterpret_cast<float *>(s + SGR),
        *s_oall = reinterpret_cast<float *>(s + SOALL);
  float *sW = reinterpret_cast<float *>(s + SW),
        *sB = reinterpret_cast<float *>(s + SB);
  float *sWQ = reinterpret_cast<float *>(s + SWQ),
        *sE8 = reinterpret_cast<float *>(s + SE8);
  float *sA = reinterpret_cast<float *>(s + SA),
        *sAQ = reinterpret_cast<float *>(s + SAQ);
  float *mS32 = reinterpret_cast<float *>(s + MS);
  __nv_bfloat16 *mUh = reinterpret_cast<__nv_bfloat16 *>(s + MUH),
                *mUl = reinterpret_cast<__nv_bfloat16 *>(s + MUL);
  __nv_bfloat16 *mBh = reinterpret_cast<__nv_bfloat16 *>(s + MBH),
                *mBl = reinterpret_cast<__nv_bfloat16 *>(s + MBL);
  float *s_bb = reinterpret_cast<float *>(s + SBB),
        *s_rq = reinterpret_cast<float *>(s + SRQ);
  float *s_rk = reinterpret_cast<float *>(s + SRK),
        *s_qkp = reinterpret_cast<float *>(s + SQKP);
  uint32_t const wfb_mb = sb + MBAR, fa_mb = sb + MBAR + 8,
                 done_mb = sb + MBAR + 16;
  uint32_t const RC = L.tb + 64, RD2 = L.tb + 128; // TMEM: f_b / D1, D2
  float const *prm = prm_all;
  size_t ring_base =
      0; // SERVE: this slot and head's first ring entry (in tokens)
  int ring_len = 0;
  if constexpr (SERVE) {
    int const slot = *slot_p;
    ring_len = info[3];
    state_all += (size_t)slot * nheads * DV * DK;
    ring_base = ((size_t)slot * nheads + head) * ring_len;
  }
  float *state = state_all + (size_t)head * DV * DK;
  constexpr float scale = 0.08838834764831845f; // DK^-1/2
  constexpr float gate_scale =
      -5.0f * 1.4426950408889634f; // lower bound * log2 e

  __syncthreads(); // the ring's data region is free (every earlier task's loads
                   // consumed)
  // ---- phase 0: barriers, the W_fb TMA, S0 staging (no inputs needed)
  if (tid == 0) {
    mbar_init(wfb_mb, 1);
    mbar_init(fa_mb, 1);
    mbar_init(done_mb, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    mbar_expect(wfb_mb, 32768);
    tma3(&maps.m[wfb_map], wfb_mb, sb + WFB, 0, head * DK, 0, EVICT_FIRST);
  }
  float nw[4];
#pragma unroll
  for (int jj = 0; jj < 4; jj++) {
    nw[jj] = prm[nheads + nheads * DK + lane * 4 + jj];
  }
  unsigned aq_mm[DOTS_PER_WARP];
#pragma unroll
  for (int q = 0; q < DOTS_PER_WARP; q++) {
    int const dd = (warp - 1) + 7 * q;
    aq_mm[q] = (warp >= 1 && dd < 64) ? aq_map[dd] : 0xFFFFu;
  }
  { // S0: the same 16-B pieces, global -> shared by cp.async (one group, waited
    // before D1)
#pragma unroll
    for (int e = tid * 8; e < DV * DK; e += 256 * 8) {
      int const r = e / DK, c = e % DK;
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                       su32(mS32 + nsk_big32(r, c))),
                   "l"(state + (size_t)r * DK + c)
                   : "memory");
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                       su32(mS32 + nsk_big32(r, c + 4))),
                   "l"(state + (size_t)r * DK + c + 4)
                   : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  if (tid < DV) {
    int4 const z4 = make_int4(0, 0, 0, 0);
    *reinterpret_cast<int4 *>(mUh + nsk_small(tid, 8)) = z4;
    *reinterpret_cast<int4 *>(mUl + nsk_small(tid, 8)) = z4;
    *reinterpret_cast<int4 *>(mBh + nsk_small(tid, 8)) = z4;
    *reinterpret_cast<int4 *>(mBl + nsk_small(tid, 8)) = z4;
  }
  float const a_log_exp = expf(prm[head]);
  // the gate (thread d < DK = TMEM lane d): per token from f_b, its running sum
  // -> sEG; needs f_b only
  auto gate = [&](bool store) {
    float f[8];
    rec_tmem_x8(RC, f);
    float const gpr = prm[nheads + head * DK + tid];
    float cs = 0.f;
#pragma unroll
    for (int tok = 0; tok < T; tok++) {
      float const gv = bf(tobf(f[tok])) + gpr;
      float const x = a_log_exp * gv;
      float const sg = __fdividef(1.0f, 1.0f + __expf(-x));
      cs += gate_scale * sg;
      sEG[tok * DK + tid] = cs;
      if constexpr (SERVE) {
        if (store) { // the natural-log decay, as the framework's ring holds it
          ring_g[(ring_base + tok) * DK + tid] = -5.0f * sg;
        }
      }
    }
  };
  // pass 0: the code from the f_b result to the
  // publish, dry, during the producers' wait (no MMA, no mbarrier wait, no
  // global store or count; garbage operands, shared-memory results above
  // the W_fb TMA's region and rewritten by pass 1), so its instructions are
  // fetched before the inputs are ready. Pass 1: the producers' wait, the
  // staging, then the same code for real. One code copy: not unrolled. (The
  // layer 1.7 us faster than without pass 0: 38.0 vs 39.7, k37.)
#pragma unroll 1
  for (int pass = 0; pass < 2; pass++) {
    bool const dry = pass == 0;
    if (!dry) {
      if (tid == 0) { // q / k / v / g / f_a / beta produced by nodes: all their
                      // tasks done (their counters)
        if constexpr (DAF) { // one thread spins (no sleep) on the last
                             // token's last f_a chunk; every value is then
                             // polled below
          while (sent128<SNT>(
              rec_poll128(proj + COL_FA + (size_t)(T - 1) * PROJ_N + 120))) {
          }
        } else {
          for (int i = 0; i < 2; i++) {
            if (wait_cnt[i]) {
              cnt_wait(wait_cnt[i], wait_n[i]);
            }
          }
        }
      }
      __syncthreads();
      // ---- phase 1: f_a (all 128 columns, 8 tokens) -> smem, f_b = W_fb . f_a
      // on the tensor core
      if (warp == 0) {
        __nv_bfloat16 const *fa_g = proj + COL_FA;
        if (!DAF && lane == 0 && (!wait_cnt[0] || !wait_cnt[1])) {
          // without the producers' counters: one lane watches the last 16 B
          // with backoff, then the warp reads all 128 chunks
          while (
              sent128(ld_relaxed128(fa_g + (size_t)(T - 1) * PROJ_N + 120))) {
            __nanosleep(200);
          }
        }
        __syncwarp();
        int4 fv[4];
        { // the lane's 4 chunks polled together
          int spins = 0;
          for (;;) {
            bool any = false;
#pragma unroll
            for (int q = 0; q < 4; q++) {
              int const id = lane + 32 * q, row = id >> 4, c = id & 15;
              fv[q] = rec_poll128(fa_g + (size_t)row * PROJ_N + c * 8);
            }
#pragma unroll
            for (int q = 0; q < 4; q++) {
              any = any || sent128<SNT>(fv[q]);
            }
            if (!any) {
              break;
            }
            if (!DAF && ++spins > 8) {
              __nanosleep(64);
            }
          }
        }
#pragma unroll
        for (int q = 0; q < 4; q++) {
          int const id = lane + 32 * q, row = id >> 4, c = id & 15;
          int4 const v = fv[q];
          uint32_t const dst = sb + FA + (c >> 3) * 1024 + row * 128 +
                               (((c & 7) ^ (row & 7)) << 4);
          asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::"r"(dst),
                       "r"(v.x),
                       "r"(v.y),
                       "r"(v.z),
                       "r"(v.w)
                       : "memory");
        }
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
        if (lane == 0) {
          mbar_arrive(fa_mb);
        }
      }
      // no barrier here: warps 1..7 go on to phase 2 (tid 0 waits for f_a on
      // fa_mb)
      if (tid == 0) {
        mbar_wait(wfb_mb, 0);
        mbar_wait(fa_mb, 0);
        asm volatile("tcgen05.fence::after_thread_sync;");
        uint64_t a = sw_desc(sb + WFB), b = sw_desc(sb + FA);
        mma_f16(RC, a, b, ID_FB, 0);
#pragma unroll
        for (int k2 = 1; k2 < 4; k2++) {
          a += 2;
          b += 2;
          mma_f16(RC, a, b, ID_FB, 1);
        }
        uint64_t a2 = sw_desc(sb + WFB + 128 * 128),
                 b2 = sw_desc(sb + FA + T * 128);
#pragma unroll
        for (int k2 = 0; k2 < 4; k2++) {
          mma_f16(RC, a2, b2, ID_FB, 1);
          a2 += 2;
          b2 += 2;
        }
        rec_commit(done_mb); // waited after the staging (parity 0)
      }
      // the gate now, before the staging: f_b was issued above, ~1 us before
      // the conv's output arrives, so warps 0..3 wait for it themselves and
      // compute the gate before they poll q / k / v (the same arithmetic as
      // after the staging, which only the dry pass runs now)
      if (tid < DK) { // (the layer 0.5 us faster than after the staging: 29.7
                      // vs 30.2, k119)
        mbar_wait(done_mb, 0); // f_b
        __syncwarp();
        gate(true);
      }
      // ---- phase 2: stage q / k / v / g / beta of this head (data-as-flag), q
      // / k norms; two passes of 4 tokens
      {
        constexpr int SP = 2, SSTEP = T / SP; // tokens per thread, their step
        int const l = tid & 63, d = l * 2, tk0 = tid >> 6;
        unsigned uq[SP], uk[SP], uv[SP], ug[SP];
        uint2 fq[SP], fk[SP], fv[SP]; // QKV32
        unsigned short ub[SP];
        float const *const conv32 =
            reinterpret_cast<float const *>(conv + (size_t)T * PROJ_N);
#pragma unroll
        for (int p = 0; p < SP; p++) {
          ub[p] = (unsigned short)SNT;
        }
        {
          int spins = 0;
          for (;;) {
            bool any = false;
#pragma unroll
            for (int p = 0; p < SP; p++) {
              int const tk = tk0 + SSTEP * p;
              if constexpr (QKV32) {
                fq[p] = rec_poll64(conv32 + (size_t)0 * QKV_SS +
                                   (size_t)tk * QKV_SR + head * DK + d);
                fk[p] = rec_poll64(conv32 + (size_t)1 * QKV_SS +
                                   (size_t)tk * QKV_SR + head * DK + d);
                fv[p] = rec_poll64(conv32 + (size_t)2 * QKV_SS +
                                   (size_t)tk * QKV_SR + head * DV + d);
              } else {
                uq[p] = rec_poll32(conv + (size_t)0 * QKV_SS +
                                   (size_t)tk * QKV_SR + head * DK + d);
                uk[p] = rec_poll32(conv + (size_t)1 * QKV_SS +
                                   (size_t)tk * QKV_SR + head * DK + d);
                uv[p] = rec_poll32(conv + (size_t)2 * QKV_SS +
                                   (size_t)tk * QKV_SR + head * DV + d);
              }
              ug[p] = rec_poll32(proj + (size_t)tk * PROJ_N + COL_G +
                                 head * DK + d);
              if (l == 0) { // beta in the same round
                ub[p] =
                    rec_poll16(proj + (size_t)tk * PROJ_N + COL_BETA + head);
              }
            }
#pragma unroll
            for (int p = 0; p < SP; p++) {
              if constexpr (QKV32) {
                any = any || sent64f(fq[p]) || sent64f(fk[p]) ||
                      sent64f(fv[p]) || sent32<SNT>(ug[p]) ||
                      (l == 0 && ub[p] == SNT);
              } else {
                any = any || sent32<SNT>(uq[p]) || sent32<SNT>(uk[p]) ||
                      sent32<SNT>(uv[p]) || sent32<SNT>(ug[p]) ||
                      (l == 0 && ub[p] == SNT);
              }
            }
            if (!any) {
              break;
            }
            if (!DAF && ++spins > 8) {
              __nanosleep(64);
            }
          }
        }
#pragma unroll
        for (int p = 0; p < SP; p++) {
          int const tk = tk0 + SSTEP * p;
          float2 q2, k2, v2;
          if constexpr (QKV32) {
            q2 =
                make_float2(__uint_as_float(fq[p].x), __uint_as_float(fq[p].y));
            k2 =
                make_float2(__uint_as_float(fk[p].x), __uint_as_float(fk[p].y));
            v2 =
                make_float2(__uint_as_float(fv[p].x), __uint_as_float(fv[p].y));
          } else {
            q2 = __bfloat1622float2(
                *reinterpret_cast<__nv_bfloat162 const *>(&uq[p]));
            k2 = __bfloat1622float2(
                *reinterpret_cast<__nv_bfloat162 const *>(&uk[p]));
            v2 = __bfloat1622float2(
                *reinterpret_cast<__nv_bfloat162 const *>(&uv[p]));
          }
          float2 const g2 = __bfloat1622float2(
              *reinterpret_cast<__nv_bfloat162 const *>(&ug[p]));
          sQ[tk * DK + d] = q2.x;
          sQ[tk * DK + d + 1] = q2.y;
          sK[tk * DK + d] = k2.x;
          sK[tk * DK + d + 1] = k2.y;
          sVE[tk * DV + d] = v2.x;
          sVE[tk * DV + d + 1] = v2.y;
          sGR[tk * DV + d] = g2.x;
          sGR[tk * DV + d + 1] = g2.y;
          if (l == 0) {
            // beta as SGLang's kda_decode_mtp: rcp.approx(1 + fast exp(-b)),
            // fp32 for the recurrence and the ring
            float const sgb = sigmoid_fast(
                bf(*reinterpret_cast<__nv_bfloat16 const *>(&ub[p])));
            s_bb[tk] = sgb;
            if constexpr (SERVE) { // beta as the framework's ring holds it
                                   // (fp32 sigmoid)
              ring_beta[ring_base + tk] = sgb;
            }
          }
          float qs = q2.x * q2.x + q2.y * q2.y, ks = k2.x * k2.x + k2.y * k2.y;
          for (int d2 = 16; d2 >= 1; d2 >>= 1) {
            qs += __shfl_xor_sync(0xffffffffu, qs, d2);
            ks += __shfl_xor_sync(0xffffffffu, ks, d2);
          }
          if (lane == 0) { // [token][the token's warp 0 / 1][q, k]
            s_qkp[(tk * 2 + (warp & 1)) * 2 + 0] = qs;
            s_qkp[(tk * 2 + (warp & 1)) * 2 + 1] = ks;
          }
        }
        __syncthreads();
        if (l == 0) {
#pragma unroll
          for (int p = 0; p < SP; p++) {
            int const tk = tk0 + SSTEP * p;
            float const qsum =
                s_qkp[(tk * 2 + 0) * 2 + 0] + s_qkp[(tk * 2 + 1) * 2 + 0];
            float const ksum =
                s_qkp[(tk * 2 + 0) * 2 + 1] + s_qkp[(tk * 2 + 1) * 2 + 1];
            s_rq[tk] = rsqrtf(qsum + 1e-6f) * scale;
            s_rk[tk] = rsqrtf(ksum + 1e-6f);
          }
        }
      }
      __syncthreads();
#if defined(STATIC_TIMING_BUILD)
      if (tid == 0 && L.stage_stamps) { // stage 0: this head's inputs staged
                                        // (q, k, v, g, beta) and normed
        L.stage_stamps[0] = gtime();
      }
#endif
    }
    if (dry) { // the dry pass's gate (pass 1 computed it before its staging)
      __syncthreads();
      if (tid < DK) {
        gate(false);
      }
      __syncthreads();
    }
    // ---- phase 3: operand tables: 4 adjacent k per thread
    {
      int const e4 = tid * 4, tt = e4 / DK, k = e4 % DK;
      float wv[4], bv[4], qv[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float const cs = sEG[e4 + j], kk = sK[e4 + j] * s_rk[tt];
        float const ep = exp2f(cs), em = exp2f(-cs);
        wv[j] = kk * ep;
        bv[j] = kk * em;
        qv[j] = sQ[e4 + j] * ep;
      }
      *reinterpret_cast<float4 *>(sW + e4) =
          make_float4(wv[0], wv[1], wv[2], wv[3]);
      *reinterpret_cast<float4 *>(sB + e4) =
          make_float4(bv[0], bv[1], bv[2], bv[3]);
      *reinterpret_cast<float4 *>(sWQ + e4) =
          make_float4(qv[0], qv[1], qv[2], qv[3]);
      // [W (rows 0..7); WQ (rows 8..15)] fp32, 128-B swizzled K-major:
      // K-quarter (32 k) at +2048 B, 16-B unit = 4 k
      auto swz = [&](int r) {
        uint32_t const addr =
            sb + MT + (k >> 5) * 2048 + r * 128 + (((k & 31) >> 2) << 4);
        return (addr ^ (((addr >> 7) & 7) << 4)) - sb;
      };
      *reinterpret_cast<float4 *>(s + swz(tt)) =
          make_float4(wv[0], wv[1], wv[2], wv[3]);
      *reinterpret_cast<float4 *>(s + swz(T + tt)) =
          make_float4(qv[0], qv[1], qv[2], qv[3]);
    }
    if (tid < DK) {
      sE8[tid] = exp2f(sEG[(T - 1) * DK + tid]);
    }
    if (!dry) { // S0 landed (this thread's pieces; the barrier below:
                // everyone's) before D1 reads it (the async proxy: the fence
                // below)
      asm volatile("cp.async.wait_all;" ::: "memory");
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    // ---- phase 4: D1 on the tensor core (thread 0), the 64 A / AQ dots on
    // warps 1..7
    if (tid == 0 && !dry) {
      rec_d1_issue(sb, RC);
      rec_commit(done_mb); // parity 1
    }
    if (warp >= 1) {
      float acc[DOTS_PER_WARP];
#pragma unroll
      for (int q = 0; q < DOTS_PER_WARP; q++) {
        acc[q] = 0.f;
        if (aq_mm[q] != 0xFFFFu) {
          int const row = (aq_mm[q] >> 4) & 15, ss = aq_mm[q] & 15;
          float4 const xv = *reinterpret_cast<float4 const *>(
              ((aq_mm[q] & 256) ? sWQ : sW) + row * DK + lane * 4);
          float4 const yv =
              *reinterpret_cast<float4 const *>(sB + ss * DK + lane * 4);
          acc[q] = fmaf(xv.x, yv.x, acc[q]);
          acc[q] = fmaf(xv.y, yv.y, acc[q]);
          acc[q] = fmaf(xv.z, yv.z, acc[q]);
          acc[q] = fmaf(xv.w, yv.w, acc[q]);
        }
      }
#pragma unroll
      for (int d2 = 16; d2 >= 1; d2 >>= 1) {
#pragma unroll
        for (int q = 0; q < DOTS_PER_WARP; q++) {
          acc[q] += __shfl_xor_sync(0xffffffffu, acc[q], d2);
        }
      }
      if (lane == 0) {
#pragma unroll
        for (int q = 0; q < DOTS_PER_WARP; q++) {
          if (aq_mm[q] != 0xFFFFu) {
            int const row = (aq_mm[q] >> 4) & 15, ss = aq_mm[q] & 15;
            ((aq_mm[q] & 256) ? sAQ : sA)[row * T + ss] = acc[q];
          }
        }
      }
    }
    __syncthreads();
    if (tid == 0 && !dry) {
      mbar_wait(done_mb, 1); // D1
    }
    __syncthreads();
    // ---- phase 5: thread v = tid: P0 / OQ0, solve u, U hi / lo, output
    // tri-sum
    if (tid < DV) {
      float pq[16];
      rec_tmem_x16(RC, pq);
      float u[T];
#pragma unroll
      for (int tt = 0; tt < T; tt++) {
        float acc = sVE[tt * DV + tid] - pq[tt];
#pragma unroll
        for (int ss = 0; ss < T; ss++) {
          if (ss < tt) {
            acc = fmaf(-sA[tt * T + ss], u[ss], acc);
          }
        }
        u[tt] = acc * s_bb[tt];
      }
      __nv_bfloat16 uh[8], ul[8];
#pragma unroll
      for (int ss = 0; ss < T; ss++) {
        uh[ss] = tobf(u[ss]);
        ul[ss] = tobf(u[ss] - bf(uh[ss]));
      }
      *reinterpret_cast<int4 *>(mUh + nsk_small(tid, 0)) =
          *reinterpret_cast<int4 const *>(uh);
      *reinterpret_cast<int4 *>(mUl + nsk_small(tid, 0)) =
          *reinterpret_cast<int4 const *>(ul);
#pragma unroll
      for (int tt = 0; tt < T; tt++) {
        float o = pq[T + tt];
#pragma unroll
        for (int ss = 0; ss < T; ss++) {
          if (ss <= tt) {
            o = fmaf(sAQ[tt * T + ss], u[ss], o);
          }
        }
        s_oall[tt * DV + tid] = s_rq[tt] * o;
      }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    // ---- phase 6: gated RMSNorm over v, warp = token -> y_rec (bf16; -0
    // stored as +0)
    {
      int const tok = warp;
      float4 const o4 =
          *reinterpret_cast<float4 const *>(s_oall + tok * DV + lane * 4);
      float ss = o4.x * o4.x + o4.y * o4.y + o4.z * o4.z + o4.w * o4.w;
      for (int d2 = 16; d2 >= 1; d2 >>= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, d2);
      }
      float const rstd =
          rsqrtf(ss / (float)DV +
                 (ONORM_EPS_BITS ? __int_as_float(ONORM_EPS_BITS) : 1e-6f));
      float const ov[4] = {o4.x, o4.y, o4.z, o4.w};
      alignas(8) __nv_bfloat16 hv4[4];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        int const d = lane * 4 + j;
        __nv_bfloat16 hv =
            tobf(((ov[j] * rstd) * nw[j]) * sigmoid_fast(sGR[tok * DV + d]));
        if (__bfloat16_as_ushort(hv) == 0x8000u) {
          hv = tobf(0.f);
        }
        hv4[j] = hv;
      }
      if (!dry) { // the lane's 4 values as one 8-B store (4 2-B stores: the
                  // layer 0.2 us slower, 29.6 vs 29.4, k127)
        *reinterpret_cast<uint2 *>(yrec + (size_t)tok * HD + head * DV +
                                   lane * 4) =
            *reinterpret_cast<uint2 const *>(hv4);
      }
    }
    __syncthreads();
    // ---- phase 7: y_rec of this head done
    if (tid == 0 && !dry) {
      cnt_add(cnt,
              1); // a reader of y_rec (o_proj) waits for one count per head
#if defined(STATIC_TIMING_BUILD)
      if (L.stage_stamps) { // stage 1: the publish (y_rec of this head written)
        L.stage_stamps[1] = gtime();
      }
#endif
    }
  }
  // ---- phase 8: D2 = U . B^T on the tensor core, S_T = 2^cs_T (S0 + D2) back
  // to global (the next launch's state); SERVE: none (the framework folds the
  // accepted tokens)
  if constexpr (!SERVE) {
    if (tid < DK) { // k = tid: B^T row k, 8 tokens -> 16 contiguous bytes in
                    // the SMALL layout
      __nv_bfloat16 bh[8], bl[8];
#pragma unroll
      for (int tt = 0; tt < T; tt++) {
        float const v = sB[tt * DK + tid];
        bh[tt] = tobf(v);
        bl[tt] = tobf(v - bf(bh[tt]));
      }
      *reinterpret_cast<int4 *>(mBh + nsk_small(tid, 0)) =
          *reinterpret_cast<int4 const *>(bh);
      *reinterpret_cast<int4 *>(mBl + nsk_small(tid, 0)) =
          *reinterpret_cast<int4 const *>(bl);
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    if (tid == 0) {
      asm volatile("tcgen05.fence::after_thread_sync;");
      uint32_t const uh = sb + MUH, ul = sb + MUL, bh = sb + MBH, bl = sb + MBL;
      mma_f16(RD2, ns_desc(uh, 8, 16), ns_desc(bh, 8, 16), ID_D2, 0);
      mma_f16(RD2, ns_desc(uh, 8, 16), ns_desc(bl, 8, 16), ID_D2, 1);
      mma_f16(RD2, ns_desc(ul, 8, 16), ns_desc(bh, 8, 16), ID_D2, 1);
      rec_commit(done_mb);
      mbar_wait(done_mb, 0); // D2 (the third commit: parity 0 again)
    }
    __syncthreads();
    float *sDS = mS32; // S0's region is dead: D2 as fp32 [128 v][128 k]
    { // warp w: TMEM lanes 32 (w & 3).., columns 64 (w >> 2) + {0, 32}
      int const v = (warp & 3) * 32 + lane, c0 = (warp >> 2) * 64;
      float ca[32], cb[32];
      rec_tmem_x32_issue(RD2 + c0, ca);
      rec_tmem_x32_issue(RD2 + c0 + 32, cb);
      rec_tmem_wait();
#pragma unroll
      for (int j4 = 0; j4 < 8; j4++) {
        *reinterpret_cast<float4 *>(sDS + (size_t)v * DK + c0 + j4 * 4) =
            make_float4(
                ca[j4 * 4], ca[j4 * 4 + 1], ca[j4 * 4 + 2], ca[j4 * 4 + 3]);
        *reinterpret_cast<float4 *>(sDS + (size_t)v * DK + c0 + 32 + j4 * 4) =
            make_float4(
                cb[j4 * 4], cb[j4 * 4 + 1], cb[j4 * 4 + 2], cb[j4 * 4 + 3]);
      }
    }
    __syncthreads();
    for (int e = tid * 8; e < DV * DK; e += 256 * 8) {
      int const r = e / DK, c = e % DK;
      float4 const s0a =
          *reinterpret_cast<float4 const *>(state + (size_t)r * DK + c);
      float4 const s0b =
          *reinterpret_cast<float4 const *>(state + (size_t)r * DK + c + 4);
      float4 const da =
          *reinterpret_cast<float4 const *>(sDS + (size_t)r * DK + c);
      float4 const db =
          *reinterpret_cast<float4 const *>(sDS + (size_t)r * DK + c + 4);
      float4 const ea = *reinterpret_cast<float4 const *>(sE8 + c),
                   eb = *reinterpret_cast<float4 const *>(sE8 + c + 4);
      float const sv[8] = {
          s0a.x, s0a.y, s0a.z, s0a.w, s0b.x, s0b.y, s0b.z, s0b.w};
      float const dv[8] = {da.x, da.y, da.z, da.w, db.x, db.y, db.z, db.w};
      float const ev[8] = {ea.x, ea.y, ea.z, ea.w, eb.x, eb.y, eb.z, eb.w};
      float ov[8];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        ov[j] = (sv[j] + dv[j]) * ev[j];
      }
      __stcg(reinterpret_cast<float4 *>(state + (size_t)r * DK + c),
             make_float4(ov[0], ov[1], ov[2], ov[3]));
      __stcg(reinterpret_cast<float4 *>(state + (size_t)r * DK + c + 4),
             make_float4(ov[4], ov[5], ov[6], ov[7]));
    }
  } // !SERVE
  asm volatile("tcgen05.fence::before_thread_sync;");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads(); // the ring's region is free for the next task's TMA
}

// task x = head x: params = [PROJ_N, COL_G, COL_FA, COL_BETA, q/k/v stream
// stride, q/k/v row stride]; CONV, PROJ = the producers of conv / proj; buffer
// slots = conv_out, proj_out, prm, state (by pointer); map slots = W_fb
// (128-row box)
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class CONV,
          class PROJ,
          class... IN>
__device__ __forceinline__ void run_delta_rule_recurrence(
    Maps const &maps, G const &g, KernelLocals &L, StaticTask const &tk) {
  static_assert(SELF::buf >= 0 && SELF::counter >= 0,
                "delta_rule_recurrence: its output y_rec and a counter");
  // the producers of conv and proj (StaticNode<0, 0, 0> for a graph input: no
  // wait)
  uint32_t const *const wait_cnt[2] = {
      CONV::counter >= 0 ? g.cnt + CONV::counter : nullptr,
      PROJ::counter >= 0 ? g.cnt + PROJ::counter : nullptr};
  uint32_t const wait_n[2] = {(uint32_t)(CONV::x * CONV::y * CONV::z),
                              (uint32_t)(PROJ::x * PROJ::y * PROJ::z)};
  // SERVE (params[8]): buffer slots 4.. = slot, ring_g, ring_beta, info
  int const *sv_slot = nullptr, *sv_info = nullptr;
  float *sv_ring_g = nullptr, *sv_ring_beta = nullptr;
  if constexpr (param_or<PARAMS, 8>() != 0) {
    sv_slot = buf_at<int const>(g, slot_at<BUF_SLOTS, 4>());
    sv_ring_g = buf_at<float>(g, slot_at<BUF_SLOTS, 5>());
    sv_ring_beta = buf_at<float>(g, slot_at<BUF_SLOTS, 6>());
    sv_info = buf_at<int const>(g, slot_at<BUF_SLOTS, 7>());
  }
  delta_rule_recurrence_task<PARAMS::v[0],
                             PARAMS::v[1],
                             PARAMS::v[2],
                             PARAMS::v[3],
                             PARAMS::v[4],
                             PARAMS::v[5],
                             param_or<PARAMS, 6>(),
                             param_or<PARAMS, 7>(),
                             param_or<PARAMS, 8>(),
                             param_or<PARAMS, 9>()>(
      maps,
      g,
      L,
      MAP_SLOTS::v[0],
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<float>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<__nv_bfloat16>(g, SELF::buf),
      g.cnt + SELF::counter,
      wait_cnt,
      wait_n,
      SELF::x,
      tk.x,
      sv_slot,
      sv_ring_g,
      sv_ring_beta,
      sv_info);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_delta_rule_recurrence() {
  return RING_BYTES; // its layout ends at rec::END <= RING_BYTES (static_assert
                     // above)
}

} // namespace static_mk
