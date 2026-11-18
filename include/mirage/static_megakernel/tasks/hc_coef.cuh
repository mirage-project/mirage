// tasks/hc_coef.cuh -- DeepSeek-V4.1's mHC coefficients of a sublayer
// (kernel.py hc_split_sinkhorn), from the partial mixes of hc_mix. Task t of
// grid (T, 1, 1) = token t: adds the KS K parts of a gemm_tile node (the mixes
// as rows 0 .. NMIX - 1 from fn's bf16 part and rows NMIX .. 2 NMIX - 1 from the
// rest of fn: 0xFF prefill, polled), plus the sum of squares of R[t],
// / rms(R[t]), z = m s + b, then on warp 0's lanes pre = sigmoid(z[:HC]) + eps,
// post = 2 sigmoid(z[HC : 2 HC]), comb by softmax and Sinkhorn (hc_pre.cuh
// hc_comb_post). Output: row t of the node's buffer [T][HC_COEF] fp32: pre
// [HC], post [HC], comb [HC][HC] (row i = from stream i), then 0s; pre also to
// a graph tensor [T][HC] (the framework's state after a layer whose boundary it
// runs itself). Then the node's counter += 1.
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

template <int KS, int HC, int H, int NMIX, int ITERS>
__device__ HC_TASK_INLINE void hc_coef_task(float const *mix,
                                             __nv_bfloat16 const *res,
                                             float const *hc_small,
                                             float *coef,
                                             float *h_pre,
                                             float *h_post,
                                             float *h_comb,
                                             uint32_t *done_cnt,
                                             int t,
                                             float rms_eps,
                                             float hc_eps,
                                             float post_mult,
                                             char *scratch) {
  constexpr int CB = HC * HC; // comb lanes; pre: CB .. CB + HC - 1, post next
  static_assert(NMIX == CB + 2 * HC && NMIX + 1 <= HC_MIX_STRIDE &&
                    HC_COEF == 32,
                "hc_coef: the coefficients are one warp's lanes");
  constexpr int Q = HC_MIX_STRIDE / 4; // float4s per (part, token)
  static_assert(KS * Q <= 256 && NMIX % 4 == 0 && (HC * H) % 2048 == 0,
                "hc_coef: a thread per (K part, float4 of the mixes)");
  static_assert(32 % Q == 0, "hc_coef: a warp holds whole rows of the mixes");
  float *mix_s = reinterpret_cast<float *>(scratch); // [8][HC_MIX_STRIDE]
  float *coef_s = mix_s + 8 * HC_MIX_STRIDE;         // [HC_COEF]
  int const lane = threadIdx.x & 31;
  int const ci = hc_coef_index<HC>(lane);
  float base = 0.f, scale = 0.f;
  if (threadIdx.x < NMIX) {
    base = hc_small[ci];
    scale = hc_small[NMIX + (lane < CB ? 2 : (lane < CB + HC ? 0 : 1))];
  }
  // R[t]'s sum of squares, 80 values per thread (the official mean over HC H)
  float ssq = 0.f;
#pragma unroll 1
  for (int i = 0; i < (HC * H) / 256 / 8; i++) {
    uint4 const v = ld16_relaxed(reinterpret_cast<unsigned char const *>(
        res + (size_t)t * (HC * H) + threadIdx.x * ((HC * H) / 256) + 8 * i));
    float2 const f0 = __bfloat1622float2(
                     *reinterpret_cast<__nv_bfloat162 const *>(&v.x)),
                 f1 = __bfloat1622float2(
                     *reinterpret_cast<__nv_bfloat162 const *>(&v.y)),
                 f2 = __bfloat1622float2(
                     *reinterpret_cast<__nv_bfloat162 const *>(&v.z)),
                 f3 = __bfloat1622float2(
                     *reinterpret_cast<__nv_bfloat162 const *>(&v.w));
    ssq += f0.x * f0.x + f0.y * f0.y + f1.x * f1.x + f1.y * f1.y +
           f2.x * f2.x + f2.y * f2.y + f3.x * f3.x + f3.y * f3.y;
  }
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    ssq += __shfl_xor_sync(0xffffffffu, ssq, o);
  }
  // thread p = K part p / Q, float4 p % Q of the mixes: the row's bf16 part
  // of fn (rows 4 q ..) + the rest (rows NMIX + 4 q ..); the rows above NMIX
  // and the K parts past KS: nothing. A slot is re-armed by this one reader.
  float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
  {
    int const pp = threadIdx.x / Q, q = threadIdx.x % Q;
    if (pp < KS && 4 * q < NMIX) {
      float const *const row = mix + ((size_t)pp * T + t) * 128;
      float const *const m0 = row + 4 * q, *const m1 = row + NMIX + 4 * q;
      float4 v0 = ldf4_poll(m0), v1 = ldf4_poll(m1);
      while (!validf4(v0)) {
        v0 = ldf4_poll(m0);
      }
      while (!validf4(v1)) {
        v1 = ldf4_poll(m1);
      }
      a = f4_add(v0, v1);
      if constexpr (REARM) {
        rearm16(m0);
        rearm16(m1);
      }
    }
  }
  if (lane == NMIX / 4) { // entry NMIX of the row: the warp's sum of squares
    a.x += ssq;
  }
#pragma unroll
  for (int o = Q; o < 32; o <<= 1) {
    a = f4_add(a, f4_shfl_xor(a, o));
  }
  __syncthreads(); // the ring area is free once every warp has left the
                   // previous task
  if (lane < Q) {
    reinterpret_cast<float4 *>(mix_s)[(threadIdx.x >> 5) * Q + lane] = a;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    float m = 0.f;
#pragma unroll
    for (int w = 0; w < 8; w++) {
      m += mix_s[w * HC_MIX_STRIDE + lane];
    }
    float const rr =
        rsqrtf(__shfl_sync(0xffffffffu, m, NMIX) / (float)(HC * H) + rms_eps);
    float const z = __shfl_sync(0xffffffffu, m, ci) * rr * scale + base;
    float const val = lane >= CB && lane < CB + HC ? hc_sigmoid(z) + hc_eps
                                                   : (lane < NMIX ? z : 0.f);
    coef_s[ci] = val;
  }
  __syncthreads();
  if (threadIdx.x < 32) {
    float const zv = coef_s[ci];
    float const c = hc_comb_post<HC, ITERS>(zv, lane, hc_eps, post_mult);
    if (lane < NMIX) { // this lane's coefficient: its place in the row is ci
      float const val = lane >= CB && lane < CB + HC ? zv : c;
      coef[(size_t)t * HC_COEF + ci] = val;
      if (lane >= CB && lane < CB + HC) {
        h_pre[(size_t)t * HC + lane - CB] = val;
      }
      if (h_post != nullptr && ci >= 2 * HC) { // the framework's triplet
        h_comb[(size_t)t * CB + ci - 2 * HC] = val;
      } else if (h_post != nullptr && ci >= HC) {
        h_post[(size_t)t * HC + ci - HC] = val;
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt, 1); // release: the token's coefficients
  }
}

// hc_coef task (token x). MIX: the hc_mix node (its K parts MIX::x, params {HC,
// H, NMIX}); PARAMS {rms eps, hc eps, post multiplier (fp32 bits), Sinkhorn
// iterations}; BUF_SLOTS {hc_small (base [NMIX], scale [3] fp32), pre [T][HC]
// fp32}; its buffer: the coefficients; its counter: T
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class MIX,
          class... IN>
#ifdef HC_RUN_NOINLINE // measurement only: the task's dispatch as a function of its own
__device__ __noinline__ void run_hc_coef(
#else
__device__ __forceinline__ void run_hc_coef(
#endif
        Maps const &,
                                            G const &g,
                                            KernelLocals &L,
                                            StaticTask const &tk) {
  static_assert(MIX::buf >= 0 && MIX::x == 1,
                "hc_coef: its input is a gemm_tile node's K parts in slots");
  static_assert(SELF::x == T && SELF::buf >= 0 && SELF::counter >= 0,
                "hc_coef: a task per token, its buffer and counter");
  constexpr int HC = 4, K = MIX::v[1];
  float *h_post = nullptr, *h_comb = nullptr; // (slots -1: not exported)
  if constexpr (slot_at<BUF_SLOTS, 3>() >= 0) {
    h_post = buf_at<float>(g, slot_at<BUF_SLOTS, 3>());
    h_comb = buf_at<float>(g, slot_at<BUF_SLOTS, 4>());
  }
  hc_coef_task<MIX::y, HC, K / HC, HC * HC + 2 * HC, PARAMS::v[3]>(
      buf_at<float const>(g, MIX::buf),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<float>(g, SELF::buf),
      buf_at<float>(g, slot_at<BUF_SLOTS, 1>()),
      h_post,
      h_comb,
      cnt_now(g) + SELF::counter,
      tk.x,
      __int_as_float(PARAMS::v[0]),
      __int_as_float(PARAMS::v[1]),
      __int_as_float(PARAMS::v[2]),
      L.sm);
}

// dynamic shared memory: the scratch, inside the ring area
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_hc_coef() {
  static_assert((8 * HC_MIX_STRIDE + HC_COEF) * 4 <= RING_BYTES,
                "hc_coef: its scratch in the ring area");
  return RING_BYTES;
}

} // namespace static_mk
