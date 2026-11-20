// tasks/sum_rmsnorm.cuh -- the adding half of an all-reduce
// (allreduce_send.cuh) followed by RMSNorm: task (x, y) of grid (Q, T, 1) =
// token y, part x of Q (1, 2 or 4) of the row's columns: add the row over the
// GPUs (fp32, GPU order), sum of squares; the Q tasks of a token swap their
// sums (Q = 2 in a cluster of 2 CTAs: in shared memory; else through ss_part;
// or each task adds the whole row's squares itself, recompute), then RMSNorm ->
// Rn (bf16). LAT = the row's width (the allreduce_send node's params[0]); a
// task's threads: one per 16 columns of the row (LAT / 16 <= 256), each 8
// columns of the task's part at a time.
#pragma once
#include "../core.cuh"

namespace static_mk {

template <int LAT>
__host__ __device__ constexpr size_t r_rank() {
  return (size_t)T * LAT * 2;
} // one GPU's partial sums in the sent buffer
template <int LAT>
__host__ __device__ constexpr int rn_threads() {
  static_assert(
      LAT % 16 == 0 && LAT / 16 <= 256,
      "sum_rmsnorm: one thread per 16 columns of the row, at most 256");
  return LAT / 16;
}

// the sum of squares of 8 columns, in two groups of 4 (the same order in every
// variant below)
__device__ __forceinline__ float sq8(float const *s) {
  float ss = 0.f;
  ss += s[0] * s[0] + s[1] * s[1] + s[2] * s[2] + s[3] * s[3];
  ss += s[4] * s[4] + s[5] * s[5] + s[6] * s[6] + s[7] * s[7];
  return ss;
}
// Rn columns c .. c + 7 of token t from their 8 sums: one 16-B store (rn = the
// sum_rmsnorm node's buffer [T][LAT])
template <int LAT>
__device__ __forceinline__ void rn_store8(__nv_bfloat16 const *gamma,
                                          __nv_bfloat16 *rn,
                                          int t,
                                          int c,
                                          float const *s,
                                          float rstd) {
  float4 n0, n1;
  n0.x = s[0] * rstd * __bfloat162float(gamma[c]);
  n0.y = s[1] * rstd * __bfloat162float(gamma[c + 1]);
  n0.z = s[2] * rstd * __bfloat162float(gamma[c + 2]);
  n0.w = s[3] * rstd * __bfloat162float(gamma[c + 3]);
  n1.x = s[4] * rstd * __bfloat162float(gamma[c + 4]);
  n1.y = s[5] * rstd * __bfloat162float(gamma[c + 5]);
  n1.z = s[6] * rstd * __bfloat162float(gamma[c + 6]);
  n1.w = s[7] * rstd * __bfloat162float(gamma[c + 7]);
  *reinterpret_cast<uint4 *>(rn + (size_t)t * LAT + c) = pack_bf16x8(n0, n1);
}

// sum_rmsnorm task (q, t) of Q tasks per token (Q = 1, 2 or 4; no cluster
// launch): columns [q LAT / Q, (q + 1) LAT / Q) of R for token t: add R over
// the GPUs, RMSNorm. The Q tasks of a token swap their sums of squares through
// ss_part[t][q] (they run at the same time, on different SMs); Q = 1: one task
// has the whole row. Thread v < NTH takes the task's vectors v, v + NTH (8
// columns each; LAT 3584, NTH 224: Q = 4: 112 vectors, Q = 2: 224, Q = 1: 448,
// two per thread, landed one after the other).
template <int Q, size_t ROFF, int LAT, int EPS_BITS>
__device__ __forceinline__ void sum_rmsnorm_task(G const &g,
                                                 __nv_bfloat16 const *gamma,
                                                 float *ss_part,
                                                 __nv_bfloat16 *rn,
                                                 int q,
                                                 int t STAGE_STAMP_PARAM) {
  static_assert(Q == 1 || Q == 2 || Q == 4, "1, 2 or 4 tasks per token");
  constexpr size_t R_RANK = r_rank<LAT>();
  constexpr int NTH = rn_threads<LAT>(), COLS = LAT / Q, NV = COLS / 8,
                VPT = (NV + NTH - 1) / NTH;
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31, v = threadIdx.x;
  unsigned char const *row =
      g.rv + ROFF + rg_set() + (size_t)t * LAT * 2 + (size_t)q * COLS * 2;
  float s[VPT][8];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < VPT; j++) {
    int const vec = j * NTH + v;
    if (v < NTH && vec < NV) {
      allreduce_land_bf16<R_RANK>(g, row, vec, s[j]);
      ss += sq8(s[j]);
    }
  }
  for (int d = 16; d > 0; d >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, d);
  }
  __shared__ float red_q[8];
  if (lane == 0) {
    red_q[warp] = ss;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    STAGE_STAMP_AT(0)
  } // timing build: R of all GPUs landed here (and this task's sum of squares)
  if (threadIdx.x == 0) {
    float tot = 0.f;
    for (int w_ = 0; w_ < 8; w_++) {
      tot += red_q[w_];
    }
    if constexpr (Q > 1) { // swap with the token's other tasks: their sums in
                           // ss_part[t][0..Q)
      ss_part[t * 4 + q] = tot;
      float4 q4;
      auto landed = [](float4 const &x) {
        return Q == 4 ? validf4(x) : validf(x.x) && validf(x.y);
      };
      do {
        q4 = ldf4_relaxed(ss_part + t * 4);
        if (!landed(q4)) {
          __nanosleep(64);
        }
      } while (!landed(q4));
      tot = Q == 4 ? q4.x + q4.y + q4.z + q4.w : q4.x + q4.y;
    }
    red_q[0] = rsqrtf(tot / LAT + __int_as_float(EPS_BITS));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    STAGE_STAMP_AT(1)
  } // timing build: the sums of squares are complete
  float const rstd = red_q[0];
#pragma unroll
  for (int j = 0; j < VPT; j++) {
    int const vec = j * NTH + v;
    if (v < NTH && vec < NV) {
      rn_store8<LAT>(gamma, rn, t, q * COLS + 8 * vec, s[j], rstd);
      if constexpr (REARM) {
        allreduce_rearm<R_RANK>(g, row, vec); // after Rn is written
      }
    }
  }
  __syncthreads();
}

// sum_rmsnorm task (q, t) of Q that adds the squares of the WHOLE row itself
// (no swap with the token's other tasks): it lands all LAT columns of R for
// token t (2 vectors per thread), and writes Rn only for its columns [q LAT /
// Q, (q + 1) LAT / Q). Every task of the row reads the whole row, so none can
// re-arm it: not usable with STATIC_RESET_IN_KERNEL.
template <int Q, size_t ROFF, int LAT, int EPS_BITS>
__device__ __forceinline__ void
    sum_rmsnorm_recompute_task(G const &g,
                               __nv_bfloat16 const *gamma,
                               __nv_bfloat16 *rn,
                               int q,
                               int t STAGE_STAMP_PARAM) {
  static_assert(Q == 1 || Q == 2 || Q == 4, "1, 2 or 4 tasks per token");
  static_assert(!REARM || Q < 0,
                "STATIC_RESET_IN_KERNEL: every task of the row reads the whole "
                "row, none can re-arm it");
  constexpr size_t R_RANK = r_rank<LAT>();
  constexpr int NTH = rn_threads<LAT>(), NV = LAT / 8, VPT = NV / NTH,
                COLS = LAT / Q;
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31, v = threadIdx.x;
  unsigned char const *row = g.rv + ROFF + rg_set() + (size_t)t * LAT * 2;
  float s[VPT][8];
  float ss = 0.f;
#pragma unroll
  for (int j = 0; j < VPT; j++) {
    if (v < NTH) {
      allreduce_land_bf16<R_RANK>(g, row, j * NTH + v, s[j]);
      ss += sq8(s[j]);
    }
  }
  for (int d = 16; d > 0; d >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, d);
  }
  __shared__ float red_r[8];
  if (lane == 0) {
    red_r[warp] = ss;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    STAGE_STAMP_AT(0)
  } // timing build: the whole row of R landed here (and its sum of squares)
  if (threadIdx.x == 0) {
    float tot = 0.f;
    for (int w_ = 0; w_ < 8; w_++) {
      tot += red_r[w_];
    }
    red_r[0] = rsqrtf(tot / LAT + __int_as_float(EPS_BITS));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    STAGE_STAMP_AT(1)
  } // timing build: the sum of squares is complete
  float const rstd = red_r[0];
#pragma unroll
  for (int j = 0; j < VPT; j++) {
    int const c = 8 * (j * NTH + v);
    if (v < NTH && c >= q * COLS && c < (q + 1) * COLS) {
      rn_store8<LAT>(gamma, rn, t, c, s[j], rstd);
    }
  }
  __syncthreads();
}

// sum_rmsnorm task (q, t) of 2 per token, the 2 tasks on the 2 CTAs of one
// cluster (the kernel launched with clusters of 2; the plan puts the pair on
// CTAs 2k, 2k + 1): columns [q LAT / 2, (q + 1) LAT / 2) of R for token t,
// thread v < NTH columns 8 v .. 8 v + 7: add R over the GPUs, the sum of
// squares, swapped with the other CTA in shared memory (thread 0 writes its sum
// into the other CTA's peer_ss: mapa + st.shared::cluster, then both wait at
// the cluster barrier), RMSNorm -> Rn.
template <size_t ROFF, int LAT, int EPS_BITS>
__device__ __forceinline__ void
    sum_rmsnorm_cluster_task(G const &g,
                             __nv_bfloat16 const *gamma,
                             __nv_bfloat16 *rn,
                             int q,
                             int t STAGE_STAMP_PARAM) {
  constexpr size_t R_RANK = r_rank<LAT>();
  constexpr int NTH = rn_threads<LAT>(),
                COLS = LAT / 2; // NTH threads x 8 columns per task
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31, v = threadIdx.x;
  unsigned char const *row =
      g.rv + ROFF + rg_set() + (size_t)t * LAT * 2 + (size_t)q * COLS * 2;
  float s[8];
  if (v < NTH) {
    allreduce_land_bf16<R_RANK>(g, row, v, s);
  } else {
    for (int k = 0; k < 8; k++) {
      s[k] = 0.f;
    }
  }
  float ss = sq8(s);
  for (int d = 16; d > 0; d >>= 1) {
    ss += __shfl_xor_sync(0xffffffffu, ss, d);
  }
  __shared__ float red_c[8];
  __shared__ float peer_ss; // the other task's sum of squares, written by the
                            // other CTA of the cluster
  if (lane == 0) {
    red_c[warp] = ss;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    STAGE_STAMP_AT(0)
  } // timing build: R of all GPUs landed here (and this task's sum of squares)
  float tot = 0.f;
  if (threadIdx.x == 0) {
    for (int w_ = 0; w_ < 8; w_++) {
      tot += red_c[w_];
    }
    uint32_t rank, remote;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote)
                 : "r"(su32(&peer_ss)), "r"(rank ^ 1u));
    asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(remote), "f"(tot)
                 : "memory");
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait."
               "acquire.aligned;" ::
                   : "memory");
  if (threadIdx.x == 0) {
    red_c[0] = rsqrtf((tot + peer_ss) / LAT + __int_as_float(EPS_BITS));
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    STAGE_STAMP_AT(1)
  } // timing build: the sums of squares are complete
  float const rstd = red_c[0];
  if (v < NTH) { // Rn: this thread's 8 columns, one 16-B store
    rn_store8<LAT>(gamma, rn, t, q * COLS + 8 * v, s, rstd);
    if constexpr (REARM) {
      allreduce_rearm<R_RANK>(g, row, v); // after Rn is written
    }
  }
  __syncthreads();
}
// sum_rmsnorm task (part x of SELF::x, token y): SEND = the allreduce_send node
// whose output it adds (rows of LAT = its params[0]). PARAMS {mode, eps (fp32
// bits)}: mode 2: the 2 tasks of a token are the 2 CTAs of a cluster (the
// kernel launched with clusters of 2, the pair on CTAs 2k, 2k + 1:
// compiler.compile_plan decides); 1: each task adds the whole row's squares
// itself; 0: the token's tasks swap their sums through ss_part. BUF_SLOTS
// {gamma (a graph tensor, [LAT] bf16), ss_part (scratch, [T][4] fp32, 0xFF
// before each launch)}
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
  static_assert(param_at<PARAMS, 0>() != 2 || SELF::x == 2,
                "the cluster swap: 2 tasks per token");
  static_assert(SELF::buf >= 0, "sum_rmsnorm: its output Rn");
  constexpr size_t ROFF = exchange_offset[SEND::buf];
  __nv_bfloat16 *const rn = buf_at<__nv_bfloat16>(g, SELF::buf);
  __nv_bfloat16 const *const gamma =
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>());
  if constexpr (param_at<PARAMS, 0>() == 2) {
    sum_rmsnorm_cluster_task<ROFF, LAT, EPS>(
        g, gamma, rn, tk.x, tk.y STAGE_STAMP_ARG(L.stage_stamps));
  } else if constexpr (param_at<PARAMS, 0>() == 1) {
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
