// tasks/gather_f32.cuh -- the gathering half of an all-gather (11-20, the
// DeepSeek-V4.1 LM head): task (x, y) of grid (Q, T, 1) = token y, part x of
// the gathered row: the GPUs' rows as sent by an allreduce_send node ([GPUS][T]
// [W] bf16 in the exchange region; GPU r's row is columns [r N, r N + N) of the
// output, its first N of W sent columns: W - N padding) polled per 16 B, as
// fp32 into the node's output [T][GPUS N] (a graph tensor: the framework's
// logits). Re-armed after the read (REARM: this task is each vector's one
// reader).
// AM (11-20, DSV41_HEAD_ARGMAX): also token t's greedy next token, as
// torch.argmax(logits[t]): the index of the first maximal value, a NaN the
// maximum (the first NaN), -0.0 == 0.0. Each task its part's (value, index)
// into the node's buffer [T][Q] (8 B), then the token's counter; the task that
// counts last reduces the Q parts and stores the index into ids[t] (int32).
#pragma once
#include "../core.cuh"

namespace static_mk {

// (AM) (v1, i1) comes before (v2, i2) in torch.argmax's order
__device__ __forceinline__ bool am_before(float v1, int i1, float v2, int i2) {
  bool const n1 = v1 != v1, n2 = v2 != v2;
  if (n1 || n2) {
    return n1 && (!n2 || i1 < i2);
  }
  if (v1 != v2) {
    return v1 > v2;
  }
  return i1 < i2;
}

template <size_t SOFF, int W, int N, int Q, bool AM = false>
__device__ __forceinline__ void gather_f32_task(G const &g,
                                                float *__restrict__ out,
                                                int q,
                                                int t,
                                                unsigned long long *part = nullptr,
                                                uint32_t *cnt = nullptr,
                                                int *ids = nullptr) {
  constexpr size_t S_RANK = (size_t)T * W * 2; // one GPU's rows as sent
  constexpr int NV = GPUS * (N / 8), PER = (NV + Q - 1) / Q;
  static_assert(N % 8 == 0 && N <= W && W % 8 == 0,
                "gather_f32: whole 8-column vectors of each GPU's row");
  constexpr int KV = (PER + 255) / 256; // vectors a thread
  static_assert(KV <= 4, "gather_f32: at most 4 vectors a thread (more parts)");
  unsigned char const *row = g.rv + SOFF + rg_set() + (size_t)t * W * 2;
  int idx[KV];
  uint4 u[KV];
  uint32_t todo = 0; // bit k: vector k still to land
#pragma unroll
  for (int k = 0; k < KV; k++) {
    int const i = q * PER + threadIdx.x + 256 * k;
    idx[k] = i < (q + 1) * PER && i < NV ? i : -1;
    if (idx[k] >= 0) {
      todo |= 1u << k;
    }
  }
  auto at = [&](int i) {
    int const r = i / (N / 8);
    return row + (size_t)r * S_RANK + (size_t)(i - r * (N / 8)) * 16;
  };
  while (todo) { // all of this thread's loads in flight at once
#pragma unroll
    for (int k = 0; k < KV; k++) {
      if (todo >> k & 1u) {
        u[k] = ld16_poll(at(idx[k]));
      }
    }
#pragma unroll
    for (int k = 0; k < KV; k++) {
      if ((todo >> k & 1u) && valid16(u[k])) {
        todo &= ~(1u << k);
      }
    }
  }
  // (AM) first this part's (value, index) and its count (the count's release
  // then waits for no logits store), then the logits; the token's last part
  // reduces the parts after its stores
  __shared__ int am_last;
  if constexpr (AM) {
    static_assert(Q <= 32, "gather_f32 AM: the parts reduced by one warp");
    __shared__ float am_v[8];
    __shared__ int am_i[8];
    float bv = -__int_as_float(0x7f800000); // this thread's best: -inf, no
    int bi = 0x7fffffff;                     // index (any real one before it)
#pragma unroll
    for (int k = 0; k < KV; k++) {
      if (idx[k] >= 0) { // (a thread's columns increase with k and e: the
                         // first maximum = a strictly larger value, or the
                         // first NaN, or the first value at all)
        int const i = idx[k], r = i / (N / 8), c8 = i - r * (N / 8);
        __nv_bfloat16 const *b = reinterpret_cast<__nv_bfloat16 const *>(&u[k]);
#pragma unroll
        for (int e = 0; e < 8; e++) {
          float const v = __bfloat162float(b[e]);
          if (v > bv || (v != v && bv == bv) || bi == 0x7fffffff) {
            bv = v;
            bi = r * N + 8 * c8 + e;
          }
        }
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      float const ov = __shfl_xor_sync(0xffffffffu, bv, o);
      int const oi = __shfl_xor_sync(0xffffffffu, bi, o);
      if (am_before(ov, oi, bv, bi)) {
        bv = ov;
        bi = oi;
      }
    }
    if ((threadIdx.x & 31) == 0) {
      am_v[threadIdx.x >> 5] = bv;
      am_i[threadIdx.x >> 5] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      for (int w = 1; w < 8; w++) {
        if (am_before(am_v[w], am_i[w], bv, bi)) {
          bv = am_v[w];
          bi = am_i[w];
        }
      }
      part[(size_t)t * Q + q] =
          (unsigned long long)__float_as_uint(bv) << 32 | (unsigned)bi;
      unsigned old;
      asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], 1;"
                   : "=r"(old)
                   : "l"(cnt)
                   : "memory"); // release: this part; acquire: the others
      am_last = old == Q - 1;   // the token's last part: all Q stored
    }
  }
#pragma unroll
  for (int k = 0; k < KV; k++) {
    if (idx[k] >= 0) {
      int const i = idx[k], r = i / (N / 8), c8 = i - r * (N / 8);
      __nv_bfloat16 const *b = reinterpret_cast<__nv_bfloat16 const *>(&u[k]);
      float4 *o = reinterpret_cast<float4 *>(out + (size_t)t * GPUS * N +
                                             (size_t)r * N + 8 * c8);
      o[0] = make_float4(__bfloat162float(b[0]), __bfloat162float(b[1]),
                         __bfloat162float(b[2]), __bfloat162float(b[3]));
      o[1] = make_float4(__bfloat162float(b[4]), __bfloat162float(b[5]),
                         __bfloat162float(b[6]), __bfloat162float(b[7]));
      if constexpr (REARM) {
        rearm16(at(i));
      }
    }
  }
  if constexpr (AM) {
    __syncthreads(); // (thread 0's acquire, then this barrier: the parts)
    if (am_last && threadIdx.x < 32) { // lane p: part p, all loads at once
      int const p = threadIdx.x;
      float bv = -__int_as_float(0x7f800000);
      int bi = 0x7fffffff;
      if (p < Q) {
        unsigned long long pv;
        asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];"
                     : "=l"(pv)
                     : "l"(part + (size_t)t * Q + p)
                     : "memory");
        bv = __uint_as_float((unsigned)(pv >> 32));
        bi = (int)(unsigned)pv;
      }
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        float const ov = __shfl_xor_sync(0xffffffffu, bv, o);
        int const oi = __shfl_xor_sync(0xffffffffu, bi, o);
        if (am_before(ov, oi, bv, bi)) {
          bv = ov;
          bi = oi;
        }
      }
      if (p == 0) {
        ids[t] = bi;
      }
    }
  }
  __syncthreads();
}

// gather_f32 task (part x, token y): SEND = the allreduce_send node (rows of
// W = its params[0]); PARAMS {N: each GPU's columns of the output[, 1: AM]};
// BUF_SLOTS {the output [T][GPUS N] fp32[, AM: ids [T] int32]}; AM: the node's
// buffer [T][Q] u64 (the parts) and T counters (one a token)
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SEND,
          class... IN>
__device__ __forceinline__ void run_gather_f32(Maps const &,
                                               G const &g,
                                               KernelLocals &,
                                               StaticTask const &tk) {
  constexpr int W = SEND::v[0], N = PARAMS::v[0];
  static_assert(SEND::buf >= 0 &&
                    exchange_bytes[SEND::buf] == (size_t)GPUS * T * W * 2,
                "gather_f32: rows as sent, [GPUS][T][W] bf16 in the exchange "
                "region");
  static_assert(SELF::y == T, "gather_f32: grid (parts, T tokens)");
  constexpr bool AM = param_or<PARAMS, 1, 0>() == 1;
  static_assert(!AM || (SELF::buf >= 0 && SELF::counter >= 0),
                "gather_f32 AM: the node's buffer (the parts) and counters");
  gather_f32_task<exchange_offset[SEND::buf], W, N, SELF::x, AM>(
      g,
      buf_at<float>(g, slot_at<BUF_SLOTS, 0>()),
      tk.x,
      tk.y,
      AM ? buf_at<unsigned long long>(g, AM ? SELF::buf : 0) : nullptr,
      AM ? cnt_now(g) + (AM ? SELF::counter : 0) + tk.y : nullptr,
      AM ? buf_at<int>(g, slot_at<BUF_SLOTS, AM ? 1 : 0>()) : nullptr);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_gather_f32() {
  return 0;
}

} // namespace static_mk
