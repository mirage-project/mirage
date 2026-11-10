// tasks/sum_rmsnorm.cuh (HIP) -- as include/mirage/static_megakernel/tasks/
// sum_rmsnorm.cuh: the adding half of an all-reduce (allreduce_send.cuh)
// followed by RMSNorm: task (x, y) of grid (Q, T, 1) = token y, part x of Q (1,
// 2 or 4) of the row's columns: add the row over the GPUs (fp32, GPU order),
// sum of squares; the Q tasks of a token swap their sums through ss_part (mode
// 0) or each adds the whole row's squares itself (mode 1), then RMSNorm -> Rn
// (bf16; polled by its reader: device-scope stores). Mode 2 (the swap in a
// cluster's shared memory) does not exist on gfx950: no CTA clusters
// (compiler.compile_plan keeps the plain launch on HIP). The sum of squares is
// reduced per wavefront (64 lanes), then over the 4 wavefronts in order: the
// same order on every GPU.
#pragma once
#include "../core.cuh"

#ifndef STATIC_SRN_POLL2
#define STATIC_SRN_POLL2 0
#endif
#ifndef STATIC_MOE_R_RS
#define STATIC_MOE_R_RS 0
#endif
namespace static_mk {

template <int LAT>
__host__ __device__ constexpr size_t r_rank() {
  return (size_t)T * LAT * 2;
}
template <int LAT>
__host__ __device__ constexpr int rn_threads() {
  static_assert(
      LAT % 16 == 0 && LAT / 16 <= 256,
      "sum_rmsnorm: one thread per 16 columns of the row, at most 256");
  return LAT / 16;
}

__device__ __forceinline__ float sq8(float const *s) {
  float ss = 0.f;
  ss += s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + s[3] * s[3];
  ss += s[4] * s[4] + s[5] * s[5] + s[6] * s[6] + s[7] * s[7];
  return ss;
}
// the block's sum of ss (every thread passes its own; thread 0 gets the total)
__device__ __forceinline__ float block_sum(float ss, float *red) {
  for (int d = 32; d > 0; d >>= 1) {
    ss += __shfl_xor(ss, d);
  }
  if ((STATIC_TIDX & 63) == 0) {
    red[STATIC_TIDX >> 6] = ss;
  }
  __syncthreads();
  return red[0] + red[1] + red[2] + red[3];
}
// gamma of columns c .. c + 7: loaded at the task's start, before the wait for
// the row (MI355X 10-11: loaded after the row's sum, a weight read from HBM
// after the L2 flush, one round trip on the chain)
__device__ __forceinline__ uint4 gamma8(__hip_bfloat16 const *gamma, int c) {
  return *reinterpret_cast<uint4 const *>(gamma + c);
}
// Rn columns c .. c + 7 of token t from their 8 sums: one 16-B store
template <int LAT>
__device__ __forceinline__ void rn_store8(uint4 const &gw,
                                          __hip_bfloat16 *rn,
                                          int t,
                                          int c,
                                          float const *s,
                                          float rstd) {
  uint32_t const g4[4] = {gw.x, gw.y, gw.z, gw.w};
  uint32_t o[4];
#pragma unroll
  for (int k = 0; k < 4; k++) {
    o[k] = pack_bf16x2(s[2 * k] * rstd * bf16_lo(g4[k]),
                       s[2 * k + 1] * rstd * bf16_hi(g4[k]));
  }
  st16_dev(rn + (size_t)t * LAT + c, make_uint4(o[0], o[1], o[2], o[3]));
}

// mode 0: columns [q LAT / Q, (q + 1) LAT / Q) of R for token t; the Q tasks of
// the token swap their sums of squares through ss_part[t][q] (device scope)
template <int Q, size_t ROFF, int LAT, int EPS_BITS>
__device__ __forceinline__ void sum_rmsnorm_task(G const &g,
                                                 __hip_bfloat16 const *gamma,
                                                 float *ss_part,
                                                 __hip_bfloat16 *rn,
                                                 int q,
                                                 int t STAGE_STAMP_PARAM) {
  static_assert(Q == 1 || Q == 2 || Q == 4, "1, 2 or 4 tasks per token");
  constexpr size_t R_RANK = r_rank<LAT>();
  constexpr int NTH = rn_threads<LAT>(), COLS = LAT / Q, NV = COLS / 8,
                VPT = (NV + NTH - 1) / NTH;
  int const v = STATIC_TIDX;
  unsigned char const *row =
      g.rv + ROFF + rg_set() + (size_t)t * LAT * 2 + (size_t)q * COLS * 2;
  float s[VPT][8];
  float ss = 0.f;
  uint4 gw[VPT];
#pragma unroll
  for (int j = 0; j < VPT; j++) {
    int const vec = j * NTH + v;
    gw[j] = v < NTH && vec < NV ? gamma8(gamma, q * COLS + 8 * vec)
                                : make_uint4(0, 0, 0, 0);
  }
#if STATIC_SRN_POLL2
  {
    int vecs[VPT];
    bool on[VPT];
#pragma unroll
    for (int j = 0; j < VPT; j++) {
      vecs[j] = j * NTH + v;
      on[j] = v < NTH && vecs[j] < NV;
    }
    allreduce_land_bf16_n<R_RANK, VPT>(g, row, vecs, on, s);
#pragma unroll
    for (int j = 0; j < VPT; j++) {
      if (on[j]) {
        ss += sq8(s[j]);
      }
    }
  }
#else
#pragma unroll
  for (int j = 0; j < VPT; j++) {
    int const vec = j * NTH + v;
    if (v < NTH && vec < NV) {
      allreduce_land_bf16<R_RANK>(g, row, vec, s[j]);
      ss += sq8(s[j]);
    }
  }
#endif
  __shared__ float red_q[4];
  __shared__ float rstd_s;
  float tot = block_sum(ss, red_q);
  if (STATIC_TIDX == 0) {
    STAGE_STAMP_AT(0)
    if constexpr (Q > 1) {
      __hip_atomic_store(
          ss_part + t * 4 + q, tot, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      float4 q4;
      auto landed = [](float4 const &x) {
        return Q == 4 ? validf4(x) : validf(x.x) && validf(x.y);
      };
      do {
        q4 = ldf4_dev(ss_part + t * 4);
        if (!landed(q4)) {
          __builtin_amdgcn_s_sleep(1);
        }
      } while (!landed(q4));
      tot = Q == 4 ? q4.x + q4.y + q4.z + q4.w : q4.x + q4.y;
    }
    rstd_s = rsqrtf(tot / LAT + __int_as_float(EPS_BITS));
    STAGE_STAMP_AT(1)
  }
  __syncthreads();
  float const rstd = rstd_s;
#pragma unroll
  for (int j = 0; j < VPT; j++) {
    int const vec = j * NTH + v;
    if (v < NTH && vec < NV) {
      rn_store8<LAT>(gw[j], rn, t, q * COLS + 8 * vec, s[j], rstd);
    }
  }
  __syncthreads();
}

#if STATIC_MOE_R_RS
// STATIC_MOE_R_RS (allreduce_send.cuh: token t's R rows only on GPU t / (T / GPUS)), one task per token: on the owner GPU the task
// adds the GPUS rows and RMSNorms them as mode 0, writes Rn and pushes the Rn row into slot `owner` row t of every other GPU's
// exchange buffer (a row the reduce-scatter leaves empty there; re-armed with the buffer); on another GPU the task polls that row
// and copies it into its Rn - the same bytes on every GPU
template <size_t ROFF, int LAT, int EPS_BITS>
__device__ __forceinline__ void sum_rmsnorm_rs_task(G const &g,
                                                    __hip_bfloat16 const *gamma,
                                                    __hip_bfloat16 *rn,
                                                    int t STAGE_STAMP_PARAM) {
  static_assert(T % GPUS == 0, "STATIC_MOE_R_RS: whole tokens per GPU");
  constexpr size_t R_RANK = r_rank<LAT>();
  constexpr int NTH = rn_threads<LAT>(), NV = LAT / 8, VPT = (NV + NTH - 1) / NTH;
  int const v = STATIC_TIDX, own = t / (T / GPUS);
  size_t const rrow = ROFF + (size_t)t * LAT * 2; // token t's row in slot 0
  if (SOLO || own == g.rank) {
    unsigned char const *row = g.rv + rrow + rg_set();
    float s[VPT][8];
    float ss = 0.f;
    uint4 gw[VPT];
#pragma unroll
    for (int j = 0; j < VPT; j++) {
      int const vec = j * NTH + v;
      gw[j] = v < NTH && vec < NV ? gamma8(gamma, 8 * vec) : make_uint4(0, 0, 0, 0);
    }
#if STATIC_SRN_POLL2
    {
      int vecs[VPT];
      bool on[VPT];
#pragma unroll
      for (int j = 0; j < VPT; j++) {
        vecs[j] = j * NTH + v;
        on[j] = v < NTH && vecs[j] < NV;
      }
      allreduce_land_bf16_n<R_RANK, VPT>(g, row, vecs, on, s);
#pragma unroll
      for (int j = 0; j < VPT; j++) {
        if (on[j]) {
          ss += sq8(s[j]);
        }
      }
    }
#else
#pragma unroll
    for (int j = 0; j < VPT; j++) {
      int const vec = j * NTH + v;
      if (v < NTH && vec < NV) {
        allreduce_land_bf16<R_RANK>(g, row, vec, s[j]);
        ss += sq8(s[j]);
      }
    }
#endif
    __shared__ float red_rs[4];
    __shared__ float rstd_rs;
    float const tot = block_sum(ss, red_rs);
    if (STATIC_TIDX == 0) {
      STAGE_STAMP_AT(0)
      rstd_rs = rsqrtf(tot / LAT + __int_as_float(EPS_BITS));
    }
    __syncthreads();
    float const rstd = rstd_rs;
#pragma unroll
    for (int j = 0; j < VPT; j++) {
      int const vec = j * NTH + v;
      if (v < NTH && vec < NV) {
        uint32_t const g4[4] = {gw[j].x, gw[j].y, gw[j].z, gw[j].w};
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
          o[k] = pack_bf16x2(s[j][2 * k] * rstd * bf16_lo(g4[k]), s[j][2 * k + 1] * rstd * bf16_hi(g4[k]));
        }
        uint4 const ov = make_uint4(o[0], o[1], o[2], o[3]);
#if STATIC_MOE_R_RS == 2
        // (2: the Rn row into this GPU's own exchange buffer, slot 0 row t ^ (T / GPUS) - a token of GPU own ^ 1, so empty here; the
        // other GPUs' tasks load it from here: the transfer spread over their 56 tasks instead of 8 tasks' remote stores, which go out
        // ~0.6 us apart per wave - dec468: the pushed rows landed 10-16 us after the owners' sums)
        st16_sys(g.rv + ROFF + (size_t)(t ^ (T / GPUS)) * LAT * 2 + (size_t)vec * 16 + rg_set(), ov);
#else
        if constexpr (!SOLO) {
#pragma unroll
          for (int r = 0; r < GPUS; r++) {
            if (r != g.rank) {
              st16_sys(g.peer[r] + rrow + (size_t)own * R_RANK + (size_t)vec * 16 + rg_set(), ov);
            }
          }
        }
#endif
        st16_dev(rn + (size_t)t * LAT + 8 * vec, ov);
      }
    }
  } else {
#if STATIC_MOE_R_RS == 2
    static_assert(((T / GPUS) & (T / GPUS - 1)) == 0, "STATIC_MOE_R_RS 2: T / GPUS a power of 2");
    unsigned char *po = g.peer[0];
#pragma unroll
    for (int q = 1; q < GPUS; q++) {
      po = own == q ? g.peer[q] : po;
    }
    unsigned char const *row = po + ROFF + (size_t)(t ^ (T / GPUS)) * LAT * 2 + rg_set();
#else
    unsigned char const *row = g.rv + rrow + (size_t)own * R_RANK + rg_set();
#endif
    uint4 u[VPT];
    for (;;) {
      bool all = true;
#pragma unroll
      for (int j = 0; j < VPT; j++) {
        int const vec = j * NTH + v;
        bool const on = v < NTH && vec < NV;
        u[j] = on ? ld16_at<AUX_SYS>(row, (uint32_t)((size_t)vec * 16)) : make_uint4(0u, 0u, 0u, 0u);
        all = all && (!on || valid16w(u[j]));
      }
      if (all) {
        break;
      }
      tail_pause();
    }
    if (STATIC_TIDX == 0) {
      STAGE_STAMP_AT(0)
    }
#pragma unroll
    for (int j = 0; j < VPT; j++) {
      int const vec = j * NTH + v;
      if (v < NTH && vec < NV) {
        st16_dev(rn + (size_t)t * LAT + 8 * vec, u[j]);
      }
    }
  }
  if (STATIC_TIDX == 0) {
    STAGE_STAMP_AT(1)
  }
  __syncthreads();
}
template <class N>
constexpr bool srn_send_rblk() { // (the allreduce_send node's params[2] 1: R from moe_experts; v ends with a 0)
  if constexpr (sizeof(N::v) / sizeof(int) > 3) {
    return N::v[2] == 1;
  } else {
    return false;
  }
}
#endif

// mode 1: the task adds the squares of the WHOLE row itself, writes Rn only for
// its columns
template <int Q, size_t ROFF, int LAT, int EPS_BITS>
__device__ __forceinline__ void
    sum_rmsnorm_recompute_task(G const &g,
                               __hip_bfloat16 const *gamma,
                               __hip_bfloat16 *rn,
                               int q,
                               int t STAGE_STAMP_PARAM) {
  static_assert(Q == 1 || Q == 2 || Q == 4, "1, 2 or 4 tasks per token");
  constexpr size_t R_RANK = r_rank<LAT>();
  constexpr int NTH = rn_threads<LAT>(), NV = LAT / 8, VPT = NV / NTH,
                COLS = LAT / Q;
  int const v = STATIC_TIDX;
  unsigned char const *row = g.rv + ROFF + rg_set() + (size_t)t * LAT * 2;
  float s[VPT][8];
  float ss = 0.f;
  uint4 gw[VPT];
#pragma unroll
  for (int j = 0; j < VPT; j++) {
    int const c = 8 * (j * NTH + v);
    gw[j] = v < NTH && c >= q * COLS && c < (q + 1) * COLS
                ? gamma8(gamma, c)
                : make_uint4(0, 0, 0, 0);
  }
#if STATIC_SRN_POLL2
  {
    int vecs[VPT];
    bool on[VPT];
#pragma unroll
    for (int j = 0; j < VPT; j++) {
      vecs[j] = j * NTH + v;
      on[j] = v < NTH;
    }
    allreduce_land_bf16_n<R_RANK, VPT>(g, row, vecs, on, s);
#pragma unroll
    for (int j = 0; j < VPT; j++) {
      if (on[j]) {
        ss += sq8(s[j]);
      }
    }
  }
#else
#pragma unroll
  for (int j = 0; j < VPT; j++) {
    if (v < NTH) {
      allreduce_land_bf16<R_RANK>(g, row, j * NTH + v, s[j]);
      ss += sq8(s[j]);
    }
  }
#endif
  __shared__ float red_r[4];
  __shared__ float rstd_s;
  float const tot = block_sum(ss, red_r);
  if (STATIC_TIDX == 0) {
    STAGE_STAMP_AT(0)
    rstd_s = rsqrtf(tot / LAT + __int_as_float(EPS_BITS));
    STAGE_STAMP_AT(1)
  }
  __syncthreads();
  float const rstd = rstd_s;
#pragma unroll
  for (int j = 0; j < VPT; j++) {
    int const c = 8 * (j * NTH + v);
    if (v < NTH && c >= q * COLS && c < (q + 1) * COLS) {
      rn_store8<LAT>(gw[j], rn, t, c, s[j], rstd);
    }
  }
  __syncthreads();
}

// sum_rmsnorm task (part x of SELF::x, token y): SEND = the allreduce_send node
// whose output it adds (rows of LAT = its params[0]). PARAMS {mode (0 or 1 on
// HIP), eps (fp32 bits)}; BUF_SLOTS {gamma ([LAT] bf16), ss_part (scratch,
// [T][4] fp32, 0xFF before each launch)}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SEND,
          class... IN>
__device__ __forceinline__ void run_sum_rmsnorm(Maps const &,
                                                G const &g,
                                                KernelLocals &L,
                                                StaticTask const &tk) {
  constexpr int LAT = SEND::v[0], EPS = PARAMS::v[1];
  static_assert(SEND::buf >= 0,
                "sum_rmsnorm: rows sent by an allreduce_send node");
  static_assert(SELF::y == T, "sum_rmsnorm: grid (tasks per token, T tokens)");
  static_assert(param_at<PARAMS, 0>() == 0 || param_at<PARAMS, 0>() == 1,
                "sum_rmsnorm on gfx950: mode 0 (swap through ss_part) or 1 "
                "(recompute); no clusters for mode 2");
  static_assert(SELF::buf >= 0, "sum_rmsnorm: its output Rn");
  constexpr size_t ROFF = exchange_offset[SEND::buf];
  __hip_bfloat16 *const rn = buf_at<__hip_bfloat16>(g, SELF::buf);
  __hip_bfloat16 const *const gamma =
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>());
#if STATIC_MOE_R_RS
  if constexpr (srn_send_rblk<SEND>()) {
    static_assert(SELF::x == 1, "STATIC_MOE_R_RS: one sum_rmsnorm task per token");
    sum_rmsnorm_rs_task<ROFF, LAT, EPS>(g, gamma, rn, tk.y STAGE_STAMP_ARG(L.stage_stamps));
    return;
  }
#endif
  if constexpr (param_at<PARAMS, 0>() == 1) {
    sum_rmsnorm_recompute_task<SELF::x, ROFF, LAT, EPS>(
        g, gamma, rn, tk.x, tk.y STAGE_STAMP_ARG(L.stage_stamps));
  } else {
    sum_rmsnorm_task<SELF::x, ROFF, LAT, EPS>(
        g,
        gamma,
        buf_at<float>(g, slot_at<BUF_SLOTS, 1>()),
        rn,
        tk.x,
        tk.y STAGE_STAMP_ARG(L.stage_stamps));
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sum_rmsnorm() {
  return 0;
}

} // namespace static_mk
