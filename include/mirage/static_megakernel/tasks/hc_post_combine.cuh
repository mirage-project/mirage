// tasks/hc_post_combine.cuh -- the end of a DeepSeek-V4.1 sublayer, in one task
// type: the GPUs' partial sums land, the mHC post, and (NORM) the next
// sublayer's input. Task (q, t) of grid (Q, T, 1) = token t, columns [q C, q C
// + C), C = H / Q; thread v = 8 columns c .. c + 7:
//   s      = the GPUs' bf16 sums of out added in fp32, GPU order -> bf16
//            (allreduce_land_bf16, the buffer of the node that sent them)
//   R'_j   = bf16(post_j s + sum_i comb[i][j] R_i), rounded as SGLang's
//            _mhc_post_torch
//   NORM:  y = sum_j pre_j R'_j (fp32; pre = this sublayer's: model.py Block
//            collapses the next sublayer's input with it)
//          the sum of y^2 over the task's columns -> ssq[t Q + q] (0xFF before
//          the launch; every task of the token polls the Q of them: they run
//          at the same time, on different SMs)
//          x = bf16(bf16(y) rsqrt(mean y^2 + eps) gamma), as SGLang's fused
//          norm (hc_pre.cuh)
// The coefficients are hc_coef's (its counter: a thread waits, then they stay in
// shared memory); R and gamma are loaded before the waits. Then the node's
// counter += 1.
#pragma once
#include "../core.cuh"
#include "hc_pre.cuh"

namespace static_mk {

// measurement only (-DHC_NOINLINE): the task's body as a function of its own, so
// that its registers and code are not part of the kernel's one function
#ifndef HC_TASK_INLINE
#ifdef HC_NOINLINE
#define HC_TASK_INLINE __noinline__
#else
#define HC_TASK_INLINE __forceinline__
#endif
#endif

// timing build: the task's 2 stage stamps at the marks HCP_STAMP0 and HCP_STAMP1
// (thread 0, the real pass): 0 its first 8 columns of every GPU landed, 1 every
// thread's columns landed and combined (after the barrier), 2 this part's sum
// of squares stored, 3 every part's squares read, 4 x stored (before the
// re-arm)
#ifndef HCP_STAMP0
#define HCP_STAMP0 0
#endif
#ifndef HCP_STAMP1
#define HCP_STAMP1 3
#endif
// HCP_V: the thread whose marks 0, 5 (its columns combined and stored) and 6 (its
// warp's squares reduced, before the barrier) are stamped
#ifndef HCP_V
#define HCP_V 0
#endif
// measurement only: HCP_EXTRA_DRY more dry passes before the usual one; marks 20 + 2 d
// / 21 + 2 d: thread HCP_V before / after the combine of dry pass d (0 = the first)
#ifndef HCP_EXTRA_DRY
#define HCP_EXTRA_DRY 0
#endif
#define HCP_MARK(p)                                                            \
  if ((p) == HCP_STAMP0) {                                                     \
    STAGE_STAMP_AT(0)                                                          \
  }                                                                            \
  if ((p) == HCP_STAMP1) {                                                     \
    STAGE_STAMP_AT(1)                                                          \
  }

// -DHCP_FAST_END (11-19, default off): the task's end without its trailing
// waits. A thread re-arms its GPUs' slots right after it has read them (each
// 16-B slot has one reader, this thread), not after x behind two barriers; and
// the node's counter += 1 is relaxed: no node waits on it (the next kernel
// reads x after this one has ended), so its release fence, which waits for
// x's stores, leaves the kernel's last task. The same values: outputs
// identical. Not with H1 (STATIC_X_FLAG releases the x flag at the end).
// -DHCP_FAST_REARM (11-20, default off): only the first half of HCP_FAST_END
// (the re-arm right after the read), the counter's release kept: for the
// kernels whose later nodes wait on this node's counter (the Engram gate, the
// LM head's GEMM)
#if defined(HCP_FAST_REARM) && defined(STATIC_X_FLAG)
#error "HCP_FAST_REARM: not with STATIC_X_FLAG (H1)"
#endif
#ifdef HCP_FAST_END
#ifdef STATIC_X_FLAG
#error "HCP_FAST_END: not with STATIC_X_FLAG (H1)"
#endif
__device__ __forceinline__ void hcp_cnt_add_relaxed(uint32_t *c, uint32_t v) {
  asm volatile("red.relaxed.gpu.global.add.u32 [%0], %1;" ::"l"(c), "r"(v)
               : "memory");
}
#endif

template <size_t SOFF, int H, int HC, int Q, bool NORM>
__device__ HC_TASK_INLINE void
    hc_post_combine_task(G const &g,
                         float const *coef,
                         uint32_t const *coef_cnt,
                         uint32_t coef_need,
                         __nv_bfloat16 const *res,
                         __nv_bfloat16 *rout,
                         __nv_bfloat16 const *gamma,
                         __nv_bfloat16 *x,
                         float *ssq,
                         uint32_t *done_cnt,
                         int q,
                         int t,
                         float norm_eps,
                         bool warm STAGE_STAMP_PARAM) {
  constexpr size_t S_RANK = (size_t)T * H * 2; // one GPU's slot
  constexpr int COLS = H / Q, CB = HC * HC;
  constexpr int NV = COLS / 8;
  static_assert(COLS * Q == H && COLS % 8 == 0 && NV <= 256,
                "hc_post_combine: whole 8-column vectors, one per thread");
  int const v = threadIdx.x, lane = v & 31;
  int const c = q * COLS + 8 * v;
  unsigned char const *row =
      g.rv + SOFF + rg_set() + (size_t)t * H * 2 + (size_t)q * COLS * 2;
#ifdef I1C_HCP
  // I1c (11-19; -DI1C_HCP, the index layers' attention kernels only: their
  // allocation spilled these, the others' did not): R and gamma are only
  // prefetched into L1 here (a prefetch holds no registers: these 5 uint4
  // spilled across the waits for the sums), loaded where they are used; the
  // same values
  if (v < NV) {
#pragma unroll
    for (int i = 0; i < HC; i++) {
      asm volatile("prefetch.global.L1 [%0];" ::"l"(
          res + ((size_t)t * HC + i) * H + c));
    }
    if constexpr (NORM) {
      asm volatile("prefetch.global.L1 [%0];" ::"l"(gamma + c));
    }
  }
#else
  uint4 ru[HC], gm = {};
  if (v < NV) {
#pragma unroll
    for (int i = 0; i < HC; i++) {
      ru[i] =
          *reinterpret_cast<uint4 const *>(res + ((size_t)t * HC + i) * H + c);
    }
    if constexpr (NORM) {
      gm = __ldg(reinterpret_cast<uint4 const *>(gamma + c));
    }
  }
#endif
  __shared__ float coef_s[HC_COEF];
  __shared__ float red_s[8];
  __shared__ float rn_s;
  // warm = false (this node's first task on the SM): a dry pass of the same
  // code on stand-ins, with no waits or stores, while the sums are on their way
#pragma unroll 1
  for (int pass = warm ? 1 : -HCP_EXTRA_DRY; pass < 2; pass++) {
  bool const real = pass == 1;
  if (v == 0 && real) {
    cnt_wait(coef_cnt, coef_need); // acquire: the coefficients of every token
  }
  __syncthreads();
  if (v < 2 * HC + CB) {
    coef_s[v] = real ? coef[(size_t)t * HC_COEF + v] : 0.f;
  }
  __syncthreads();
  float ss = 0.f;
#ifdef I1_BF16
  // I1 (11-19): y's bf16 roundings packed (4 registers, not 8 floats): they
  // stay live across the squares' exchange below, where the 8 floats spilled
  // to local memory (an L1 miss each when the ring fills shared memory); the
  // same values (a bf16 converted to float is exact)
  uint4 ybp = {};
#else
  float yb[8];
#endif
  if (v < NV) {
    float pre[HC], post[HC], comb[HC][HC];
#pragma unroll
    for (int j = 0; j < HC; j++) {
      pre[j] = coef_s[j];
      post[j] = coef_s[HC + j];
#pragma unroll
      for (int i = 0; i < HC; i++) {
        comb[i][j] = coef_s[2 * HC + i * HC + j];
      }
    }
    float s[8];
    if (real) {
      allreduce_land_bf16<S_RANK>(g, row, v, s);
#if defined(HCP_FAST_END) || defined(HCP_FAST_REARM)
      if constexpr (REARM) { // this thread's slots, read: 0xFF for launch n + 2
        asm volatile("" ::: "memory"); // (after the polled loads)
        allreduce_rearm<S_RANK>(g, row, v);
      }
#endif
#ifdef HCP_SYNCWARP // the warp's threads leave the polling loop in groups: one group again
      static_assert(NV % 32 == 0, "HCP_SYNCWARP: whole warps");
      __syncwarp();
#endif
    } else {
#pragma unroll
      for (int k = 0; k < 8; k++) {
        s[k] = 0.f;
      }
    }
    if (v == HCP_V && real) {
      HCP_MARK(0)
    } // timing build: the thread's 8 columns of every GPU landed
#if defined(HCP_SYNCWARP) && defined(STATIC_TIMING_BUILD)
    __syncwarp(); // (the stamping thread is back with its warp)
#endif
#pragma unroll
    for (int k = 0; k < 8; k++) { // the 8 columns
      s[k] = __bfloat162float(__float2bfloat16(s[k])); // the all-reduced bf16
    }
    if (v == HCP_V && !real) {
      HCP_MARK(20 + 2 * (pass + HCP_EXTRA_DRY))
    }
    float y[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#ifdef I1C_HCP
    uint4 ru[HC]; // (I1c: prefetched at the task's start)
#pragma unroll
    for (int i = 0; i < HC; i++) {
      ru[i] =
          *reinterpret_cast<uint4 const *>(res + ((size_t)t * HC + i) * H + c);
    }
#endif
#ifdef HCP_LOOP
    // the streams as a loop that is not unrolled: an SM runs this code once a
    // launch, and code it has not run before costs about 5 ns an instruction
    // to fetch (3.7 us for the unrolled 4 streams, 0.4 us once it is cached);
    // as a loop only the first stream's pass is fetched. The same operations
    // in the same order: the same bits.
    float rf[HC][8];
#pragma unroll
    for (int i = 0; i < HC; i++) {
#pragma unroll
      for (int k = 0; k < 8; k++) {
        rf[i][k] = __bfloat162float(
            reinterpret_cast<__nv_bfloat16 const *>(&ru[i])[k]);
      }
    }
#pragma unroll 1
    for (int j = 0; j < HC; j++) {
      float const pj = coef_s[j], qj = coef_s[HC + j];
      float cj[HC];
#pragma unroll
      for (int i = 0; i < HC; i++) {
        cj[i] = coef_s[2 * HC + i * HC + j];
      }
      uint4 ou;
      __nv_bfloat16 *ob = reinterpret_cast<__nv_bfloat16 *>(&ou);
#pragma unroll
      for (int k = 0; k < 8; k++) {
        float a = __fmul_rn(cj[0], rf[0][k]);
#pragma unroll
        for (int i = 1; i < HC; i++) {
          a = __fadd_rn(a, __fmul_rn(cj[i], rf[i][k]));
        }
        ob[k] = __float2bfloat16(__fadd_rn(__fmul_rn(qj, s[k]), a));
        if constexpr (NORM) {
          y[k] += pj * __bfloat162float(ob[k]);
        }
      }
      if (real) {
        *reinterpret_cast<uint4 *>(rout + ((size_t)t * HC + j) * H + c) = ou;
      }
      if (v == HCP_V && real) {
        HCP_MARK(10 + j)
      }
    }
#else
#pragma unroll
    for (int j = 0; j < HC; j++) {
      uint4 ou;
      __nv_bfloat16 *ob = reinterpret_cast<__nv_bfloat16 *>(&ou);
#pragma unroll
      for (int k = 0; k < 8; k++) {
        auto rik = [&](int i) {
          return __bfloat162float(
              reinterpret_cast<__nv_bfloat16 const *>(&ru[i])[k]);
        };
        float a = __fmul_rn(comb[0][j], rik(0));
#pragma unroll
        for (int i = 1; i < HC; i++) {
          a = __fadd_rn(a, __fmul_rn(comb[i][j], rik(i)));
        }
        ob[k] = __float2bfloat16(__fadd_rn(__fmul_rn(post[j], s[k]), a));
        if constexpr (NORM) {
          y[k] += pre[j] * __bfloat162float(ob[k]);
        }
      }
      if (real) {
        *reinterpret_cast<uint4 *>(rout + ((size_t)t * HC + j) * H + c) = ou;
      }
      if (v == HCP_V && real) {
        HCP_MARK(10 + j) // timing build: the thread's stream j stored
      }
    }
#endif
    if (v == HCP_V && real) {
      HCP_MARK(5)
    }
    if (v == HCP_V && !real) {
      HCP_MARK(21 + 2 * (pass + HCP_EXTRA_DRY))
    }
    if constexpr (NORM) {
#pragma unroll
      for (int k = 0; k < 8; k++) { // the squares of the fp32 sum
        ss += y[k] * y[k];
#ifdef I1_BF16
        reinterpret_cast<__nv_bfloat16 *>(&ybp)[k] = __float2bfloat16(y[k]);
#else
        yb[k] = __bfloat162float(__float2bfloat16(y[k]));
#endif
      }
    }
  }
  if constexpr (NORM) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    if (lane == 0) {
      red_s[v >> 5] = ss;
    }
    if (v == HCP_V && real) {
      HCP_MARK(6)
    }
    __syncthreads();
    if (v == 0 && real) {
      HCP_MARK(1)
    }
    if (v == 0) {
      float tot = 0.f;
#pragma unroll
      for (int w = 0; w < 8; w++) {
        tot += red_s[w];
      }
      float all = 1.f;
      if (real) {
#if defined(HCP_SSQ) && HCP_SSQ == 2 // measurement: the store as an atomic exchange
        atomicExch(reinterpret_cast<unsigned *>(ssq + (size_t)t * Q + q),
                   __float_as_uint(tot));
#else
        ssq[(size_t)t * Q + q] = tot;
#endif
#if defined(HCP_SSQ) && HCP_SSQ == 1 // measurement: a fence after the store
        __threadfence();
#endif
        HCP_MARK(2)
        all = 0.f;
#ifdef HCP_VPOLL
        // the token's Q sums as 16-B vectors, all loads of a round in flight
        // together (cnt_ld: one round trip a part)
        static_assert(Q % 4 == 0, "HCP_VPOLL: the parts in fours");
        uint4 u4[Q / 4];
        bool ok;
        do {
          ok = true;
#pragma unroll
          for (int p = 0; p < Q / 4; p++) {
            u4[p] = ld16_poll(reinterpret_cast<unsigned char const *>(
                ssq + (size_t)t * Q + 4 * p));
          }
#pragma unroll
          for (int p = 0; p < Q / 4; p++) {
            ok = ok && u4[p].x != 0xFFFFFFFFu && u4[p].y != 0xFFFFFFFFu &&
                 u4[p].z != 0xFFFFFFFFu && u4[p].w != 0xFFFFFFFFu;
          }
        } while (!ok);
#pragma unroll
        for (int p = 0; p < Q / 4; p++) { // in part order: the same on every GPU
          all += __uint_as_float(u4[p].x);
          all += __uint_as_float(u4[p].y);
          all += __uint_as_float(u4[p].z);
          all += __uint_as_float(u4[p].w);
        }
#else
        for (int p = 0; p < Q; p++) { // in part order: the same on every GPU
          uint32_t const *b =
              reinterpret_cast<uint32_t const *>(ssq + (size_t)t * Q + p);
          uint32_t u = cnt_ld(b);
          while (u == 0xFFFFFFFFu) {
            u = cnt_ld(b);
          }
          all += __uint_as_float(u);
        }
#endif
      }
      rn_s = rsqrtf(all / (float)H + norm_eps);
      if (real) {
        HCP_MARK(3) // timing build: every part's squares in
      }
    }
    __syncthreads();
    if (v < NV) {
      float const rn = rn_s;
#ifdef I1C_HCP
      uint4 const gm = __ldg(reinterpret_cast<uint4 const *>(
          gamma + c)); // (I1c: prefetched at the task's start)
#endif
      __nv_bfloat16 const *gb = reinterpret_cast<__nv_bfloat16 const *>(&gm);
      uint4 ou;
      __nv_bfloat16 *ob = reinterpret_cast<__nv_bfloat16 *>(&ou);
#pragma unroll
      for (int k = 0; k < 8; k++) { // SGLang's fused norm: one rounding
#ifdef I1_BF16
        float const ybk = __bfloat162float(
            reinterpret_cast<__nv_bfloat16 const *>(&ybp)[k]);
        ob[k] = __float2bfloat16(ybk * rn * __bfloat162float(gb[k]));
#else
        ob[k] = __float2bfloat16(yb[k] * rn * __bfloat162float(gb[k]));
#endif
      }
      if (real) {
        *reinterpret_cast<uint4 *>(x + (size_t)t * H + c) = ou;
      }
    }
    if (v == 0 && real) {
      HCP_MARK(4)
    }
  }
  if constexpr (REARM) {
#if !defined(HCP_FAST_END) && !defined(HCP_FAST_REARM) // (else re-armed after
                                                         // the landing)
    __syncthreads(); // a pair of lanes has read its vector
#ifdef STATIC_X_FLAG
    // H1: x and R' of this task are stored (the barrier): the next kernel's
    // flag now, before the re-arm (the re-arm writes only this kernel's own
    // exchange region, which the next kernel never reads)
    if constexpr (NORM) {
      if (v == 0 && real) {
        x_flag_store(g, t * Q + q);
      }
    }
#endif
    if (v < NV && real) {
      allreduce_rearm<S_RANK>(g, row, v);
    }
#endif
  }
  __syncthreads();
  }
  if (v == 0) {
#ifdef HCP_FAST_END
    hcp_cnt_add_relaxed(done_cnt, 1); // (no node waits on it)
#else
    cnt_add(done_cnt, 1); // release: this task's columns
#endif
  }
}

// hc_post_combine task (part x, token y). SEND: the node that sent the sums (its
// buffer [GPUS][T][H] bf16 in the exchange region); RES: the streams (a graph
// input); COEF: the hc_coef node; PARAMS {H, HC, NORM, norm eps (fp32 bits)};
// BUF_SLOTS {R, R' ([T][HC H] bf16 graph tensors); NORM: gamma [H] bf16, x
// [T][H] bf16, ssq (scratch)}; its counter: its tasks
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SEND,
          class RES,
          class COEF,
          class... IN>
#ifdef HC_RUN_NOINLINE // measurement only: the task's dispatch as a function of its own
__device__ __noinline__ void run_hc_post_combine(
#else
__device__ __forceinline__ void run_hc_post_combine(
#endif
        Maps const &,
                                                    G const &g,
                                                    KernelLocals &L,
                                                    StaticTask const &tk,
                                                    bool warm = true) {
  constexpr int H = PARAMS::v[0], HC = PARAMS::v[1];
  constexpr bool NORM = PARAMS::v[2] != 0;
  static_assert(
      SEND::buf >= 0 && exchange_bytes[SEND::buf] == (size_t)GPUS * T * H * 2,
      "hc_post_combine: the sent sums, [GPUS][T][H] bf16 in the exchange "
      "region");
  static_assert(COEF::buf >= 0 && COEF::counter >= 0 && COEF::x == T,
                "hc_post_combine: an hc_coef node's coefficients");
  static_assert(SELF::y == T && SELF::counter >= 0,
                "hc_post_combine: grid (parts, T tokens), its counter");
  __nv_bfloat16 const *gamma = nullptr;
  __nv_bfloat16 *x = nullptr;
  float *ssq = nullptr;
  if constexpr (NORM) {
    gamma = buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>());
    x = buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 3>());
    ssq = buf_at<float>(g, slot_at<BUF_SLOTS, 4>());
  }
  hc_post_combine_task<exchange_offset[SEND::buf], H, HC, SELF::x, NORM>(
      g,
      buf_at<float const>(g, COEF::buf),
      cnt_now(g) + COEF::counter,
      COEF::x,
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 1>()),
      gamma,
      x,
      ssq,
      cnt_now(g) + SELF::counter,
      tk.x,
      tk.y,
      __int_as_float(PARAMS::v[3]),
      warm STAGE_STAMP_ARG(L.stage_stamps));
#ifdef STATIC_X_FLAG
  // H1: XFLAG_N words, one per task; with the re-arm the task stored its word
  // before it (hc_post_combine_task), else here (x and the streams of this
  // task's columns stored: the task's last barrier is before its counter)
  if constexpr (NORM) {
    static_assert(SELF::x * SELF::y == XFLAG_N,
                  "hc_post_combine (H1): XFLAG_N words, one per task");
    if constexpr (!REARM) {
      if (threadIdx.x == 0) {
        x_flag_store(g, tk.y * SELF::x + tk.x);
      }
    }
  }
#endif
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_hc_post_combine() {
  return 0;
}

} // namespace static_mk
