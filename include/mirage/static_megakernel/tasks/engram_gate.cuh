// tasks/engram_gate.cuh -- DeepSeek-V4.1's Engram gate (inference/model.py
// Engram.forward after the wkv projection), task (i, t) of grid (HC, T, 1):
// copy i of token t's HC streams R' (a node's output, [T][HC H] bf16) and its
// row of kv, the wkv projection of its n-gram rows ((HC + 1) H bf16: one key of
// H per copy, then the shared value), gathered from the GPUs: an allreduce_send
// node's buffer [GPUS][T][W] bf16 in the exchange region, GPU r's columns
// [r W, r W + W) -> in fp32:
//   rstd = rsqrt(mean(h^2) + eps) rsqrt(mean(key_i^2) + eps)
//   dot  = sum(h w_i key_i) rstd H^-1/2      (w = q_weight k_weight, fp32)
//   gate = sigmoid(copysign(sqrt(max(|dot|, clamp)), dot))
//   R''  = bf16(h + gate value)
// The same formulas as the official code; its sums are torch's reductions,
// these the kernel's (a fixed order: per thread, then the warps, then the 8
// warps in order), so R'' matches to fp32 rounding, not bit for bit.
// Few instructions (11-20: the task runs once a launch on its SM, after the
// post, so its code comes cold, about 68 ns a 128-B line from HBM; a 28-KB
// version of one task a token took 15 us of instruction fetch), and a dry pass
// on its SM's first task while kv is on its way. Re-arm (REARM): a key vector
// has one reader, this task, after its use; the value has HC readers, so task
// (i, t) re-arms quarter i of token t's value in the other set of the exchange
// region (read by this node's previous launch, which has ended; written next
// by its next launch).
// -DHCP_FAST_END is an error with this task: the post's counter it waits on
// must be a release.
#pragma once
#include "../core.cuh"

#ifdef HCP_FAST_END
#error "engram_gate: waits on the post's counter (HCP_FAST_END makes it relaxed)"
#endif

namespace static_mk {

// SOFF: kv's offset in the exchange region; W: one GPU's columns of kv. Thread
// v takes 8-column vectors v, v + 256, .. of copy i (H / 8 of them). TEN
// (11-20, DSV41_ENGRAM=2: kv gathered by an earlier kernel into a graph tensor,
// one set, kvt its local copy [GPUS][T][W] bf16): the value re-armed by task
// (0, t) once the HC tasks of token t have read it (vcnt: the node's counter
// of token t, each task +1; the HC tasks run at the same time)
// NX (11-20, DSV41_EG_X=1, with TEN): task (0, t), once all HC tasks of token t
// have stored R'', also computes the next sublayer's input as the mHC post
// SGLang does after an Engram (deepseek_v4_mhc _combine -> hc_combine_norm):
// y = bf16(sum_k pre_k R''_k) (fp32 sum), x = bf16(y rsqrt(mean(y^2) +
// norm_eps) gamma) (the squares of the bf16 y) into xo [T][H]
template <size_t SOFF, int H, int HC, int W, bool TEN = false, bool NX = false>
__device__ __forceinline__ void
    engram_gate_task(G const &g,
                     __nv_bfloat16 const *__restrict__ R,
                     float const *__restrict__ wqk,
                     __nv_bfloat16 *__restrict__ out,
                     uint32_t const *post_cnt,
                     uint32_t post_need,
                     int i,
                     int t,
                     float eps,
                     float clamp,
                     bool warm,
                     unsigned char const *kvt = nullptr,
                     uint32_t *vcnt = nullptr,
                     float const *__restrict__ pre = nullptr,
                     __nv_bfloat16 const *__restrict__ gam = nullptr,
                     __nv_bfloat16 *__restrict__ xo = nullptr,
                     float norm_eps = 0.f STAGE_STAMP_PARAM) {
  constexpr int V = H / 8, NJ = (V + 255) / 256, VQ = V / HC;
  constexpr size_t S_RANK = (size_t)T * W * 2; // one GPU's part of kv
  static_assert(H % 8 == 0 && W % 8 == 0 && (HC + 1) * H == GPUS * W &&
                    V % HC == 0 && VQ <= 256,
                "engram_gate: kv's (HC + 1) H columns in GPUS parts of W, "
                "whole 8-column vectors, the value's quarters a thread each");
  __shared__ float red_eg[8][3]; // the warps' sums
  int const v = threadIdx.x, lane = v & 31, warp = v >> 5;
  size_t const set = TEN ? 0 : rg_set(),
               other = TEN ? 0 : (size_t)EXCHANGE_SET - set;
  unsigned char const *base = TEN ? kvt : g.rv + SOFF;
  auto at = [&](size_t s, int c) { // kv column c (a multiple of 8) of token t
    int const r = c / W;
    return base + s + (size_t)t * W * 2 + (size_t)r * S_RANK +
           (size_t)(c - r * W) * 2;
  };
  __nv_bfloat16 const *Ri = R + ((size_t)t * HC + i) * H;
  float const *wi = wqk + (size_t)i * H;
#pragma unroll 1
  for (int pass = warm ? 1 : 0; pass < 2; pass++) { // pass 0: dry
    bool const real = pass == 1;
    uint4 gmx[NX ? NJ : 1]; // (NX, task (0, t)) gamma, loaded before the waits
    if constexpr (NX) {
      if (i == 0) {
#pragma unroll
        for (int j = 0; j < NJ; j++) {
          int const c8 = v + 256 * j;
          if (c8 < V) {
            gmx[j] = __ldg(reinterpret_cast<uint4 const *>(gam + 8 * c8));
          }
        }
      }
    }
    if (v == 0 && real) {
      cnt_wait(post_cnt, post_need); // acquire: R' of every token is stored
      STAGE_STAMP_AT(0) // timing build: the post seen complete
    }
    __syncthreads();
    uint4 k[NJ], h[NJ], val[NJ];
    float4 w0[NJ], w1[NJ];
    uint32_t todo = 0; // bit j: key j (bit NJ + j: value j) still to land
#pragma unroll
    for (int j = 0; j < NJ; j++) {
      int const c8 = v + 256 * j;
      if (c8 < V) {
        k[j] = ld16_poll(at(set, i * H + 8 * c8));
        val[j] = ld16_poll(at(set, HC * H + 8 * c8));
        h[j] = *reinterpret_cast<uint4 const *>(Ri + 8 * c8);
        w0[j] = *reinterpret_cast<float4 const *>(wi + 8 * c8);
        w1[j] = *reinterpret_cast<float4 const *>(wi + 8 * c8 + 4);
        todo |= (1u << j) | (1u << (NJ + j));
      }
    }
    while (real && todo) { // (dry: the values as they are)
#pragma unroll
      for (int j = 0; j < NJ; j++) {
        if ((todo >> j & 1u) && valid16(k[j])) {
          todo &= ~(1u << j);
        }
        if ((todo >> (NJ + j) & 1u) && valid16(val[j])) {
          todo &= ~(1u << (NJ + j));
        }
      }
#pragma unroll
      for (int j = 0; j < NJ; j++) {
        int const c8 = v + 256 * j;
        if (todo >> j & 1u) {
          k[j] = ld16_poll(at(set, i * H + 8 * c8));
        }
        if (todo >> (NJ + j) & 1u) {
          val[j] = ld16_poll(at(set, HC * H + 8 * c8));
        }
      }
    }
    if (v == 0 && real) {
      STAGE_STAMP_AT(1) // timing build: thread 0's kv landed
    }
    float ssh = 0.f, ssk = 0.f, d = 0.f;
#pragma unroll
    for (int j = 0; j < NJ; j++) {
      if (v + 256 * j < V) {
        __nv_bfloat16 const *hb = reinterpret_cast<__nv_bfloat16 const *>(&h[j]);
        __nv_bfloat16 const *kb = reinterpret_cast<__nv_bfloat16 const *>(&k[j]);
        float const w[8] = {w0[j].x, w0[j].y, w0[j].z, w0[j].w,
                            w1[j].x, w1[j].y, w1[j].z, w1[j].w};
#pragma unroll
        for (int e = 0; e < 8; e++) {
          float const hf = __bfloat162float(hb[e]), kf = __bfloat162float(kb[e]);
          ssh += hf * hf;
          ssk += kf * kf;
          d += hf * w[e] * kf; // (h w) key, as h * weight * key
        }
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      ssh += __shfl_xor_sync(0xffffffffu, ssh, o);
      ssk += __shfl_xor_sync(0xffffffffu, ssk, o);
      d += __shfl_xor_sync(0xffffffffu, d, o);
    }
    if (lane == 0) {
      red_eg[warp][0] = ssh;
      red_eg[warp][1] = ssk;
      red_eg[warp][2] = d;
    }
    __syncthreads();
    ssh = ssk = d = 0.f;
#pragma unroll
    for (int q = 0; q < 8; q++) { // the warps in order: every thread the same
      ssh += red_eg[q][0];
      ssk += red_eg[q][1];
      d += red_eg[q][2];
    }
    float const rstd =
        rsqrtf(ssh / (float)H + eps) * rsqrtf(ssk / (float)H + eps);
    float const dot = d * rstd * (float)(1.0 / sqrt((double)H));
    float const z = copysignf(sqrtf(fmaxf(fabsf(dot), clamp)), dot);
    float const gate = 1.f / (1.f + expf(-z));
#pragma unroll
    for (int j = 0; j < NJ; j++) {
      int const c8 = v + 256 * j;
      if (c8 < V) {
        __nv_bfloat16 const *hb = reinterpret_cast<__nv_bfloat16 const *>(&h[j]);
        __nv_bfloat16 const *vb =
            reinterpret_cast<__nv_bfloat16 const *>(&val[j]);
        float a[8];
#pragma unroll
        for (int e = 0; e < 8; e++) {
          a[e] = __bfloat162float(hb[e]) + gate * __bfloat162float(vb[e]);
        }
        uint4 const o = pack_bf16x8(make_float4(a[0], a[1], a[2], a[3]),
                                    make_float4(a[4], a[5], a[6], a[7]));
        if (real) {
          *reinterpret_cast<uint4 *>(out + ((size_t)t * HC + i) * H +
                                     8 * c8) = o;
          if constexpr (REARM) {
            rearm16(at(set, i * H + 8 * c8)); // key j: read (its one reader)
          }
        }
      }
    }
    if constexpr (TEN) { // the value: re-armed by task (0, t) after all HC
      __syncthreads();   // (every thread's value loads have landed)
      if (REARM && real) {
        if (v == 0) {
          cnt_add(vcnt, 1); // release: this task's value reads
          if (i == 0) {
            cnt_wait(vcnt, HC); // acquire: all HC tasks of token t read it
          }
        }
        __syncthreads();
        if (i == 0) {
#pragma unroll 1
          for (int c8 = v; c8 < V; c8 += 256) {
            rearm16(at(0, HC * H + 8 * c8));
          }
        }
      }
      if constexpr (NX) { // the next input (every HC task of token t stored
                          // R'': the counter above, then the barrier)
        static_assert(TEN && REARM, "engram_gate NX: kv a tensor, a serving build");
        // (the dry pass too, without the store: its code then is in the
        // instruction cache; R'' through L2 (ld.cg), not a line the dry pass
        // left in L1)
        if (i == 0) {
          __shared__ float red_x[8];
          float p[HC];
#pragma unroll
          for (int k = 0; k < HC; k++) {
            p[k] = pre[(size_t)t * HC + k];
          }
          uint4 yb[NJ];
          float ss = 0.f;
#pragma unroll
          for (int j = 0; j < NJ; j++) {
            int const c8 = v + 256 * j;
            if (c8 < V) {
              float y[8];
#pragma unroll
              for (int e = 0; e < 8; e++) {
                y[e] = 0.f;
              }
#pragma unroll
              for (int k = 0; k < HC; k++) {
                uint4 const r = __ldcg(reinterpret_cast<uint4 const *>(
                    out + ((size_t)t * HC + k) * H + 8 * c8));
                __nv_bfloat16 const *rb = reinterpret_cast<__nv_bfloat16 const *>(&r);
#pragma unroll
                for (int e = 0; e < 8; e++) {
                  y[e] += p[k] * __bfloat162float(rb[e]);
                }
              }
              __nv_bfloat16 *b = reinterpret_cast<__nv_bfloat16 *>(&yb[j]);
#pragma unroll
              for (int e = 0; e < 8; e++) {
                b[e] = __float2bfloat16(y[e]);
                float const r = __bfloat162float(b[e]);
                ss += r * r;
              }
            }
          }
#pragma unroll
          for (int o = 16; o > 0; o >>= 1) {
            ss += __shfl_xor_sync(0xffffffffu, ss, o);
          }
          if (lane == 0) {
            red_x[warp] = ss;
          }
          __syncthreads();
          float tot = 0.f;
#pragma unroll
          for (int q = 0; q < 8; q++) {
            tot += red_x[q];
          }
          float const rn = rsqrtf(tot / (float)H + norm_eps);
#pragma unroll
          for (int j = 0; j < NJ; j++) {
            int const c8 = v + 256 * j;
            if (c8 < V) {
              uint4 const gm = gmx[j];
              __nv_bfloat16 const *gb = reinterpret_cast<__nv_bfloat16 const *>(&gm);
              __nv_bfloat16 const *b = reinterpret_cast<__nv_bfloat16 const *>(&yb[j]);
              uint4 ou;
              __nv_bfloat16 *ob = reinterpret_cast<__nv_bfloat16 *>(&ou);
#pragma unroll
              for (int e = 0; e < 8; e++) {
                ob[e] = __float2bfloat16(__bfloat162float(b[e]) * rn * __bfloat162float(gb[e]));
              }
              if (real) {
                *reinterpret_cast<uint4 *>(xo + (size_t)t * H + 8 * c8) = ou;
              }
            }
          }
        }
      }
    } else if (REARM && real && v < VQ) { // quarter i of the value, the other
                                          // set
      rearm16(at(other, HC * H + 8 * (VQ * i + v)));
    }
    __syncthreads();
  }
}

// engram_gate task (copy x, token y): SEND = the allreduce_send node that
// gathered kv (rows of W = its params[0]); POST = the node whose output R' is
// (its counter: all its tasks); PARAMS {H, HC, eps (fp32 bits), clamp (fp32
// bits)}; BUF_SLOTS {R' [T][HC H] bf16, w [HC][H] fp32, R'' [T][HC H] bf16}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SEND,
          class POST,
          class... IN>
__device__ __forceinline__ void run_engram_gate(Maps const &,
                                                G const &g,
                                                KernelLocals &L,
                                                StaticTask const &tk,
                                                bool warm = true) {
  constexpr bool TEN = param_or<PARAMS, 4, 0>() == 1; // kv: a graph tensor
  constexpr bool NX = param_or<PARAMS, 6, 0>() == 1;   // and the next input
  constexpr int H = PARAMS::v[0], HC = PARAMS::v[1],
                W = TEN ? param_or<PARAMS, 5, 8>() : SEND::v[0];
  static_assert(TEN || (SEND::buf >= 0 && exchange_bytes[SEND::buf] ==
                                              (size_t)GPUS * T * W * 2),
                "engram_gate: kv as sent, [GPUS][T][W] bf16 in the exchange "
                "region");
  static_assert(!TEN || SELF::counter >= 0,
                "engram_gate (kv a tensor): its counters, one a token");
  static_assert(POST::counter >= 0, "engram_gate: R' from a node that counts "
                                    "its tasks");
  static_assert(SELF::x == HC && SELF::y == T,
                "engram_gate: grid (HC copies, T tokens)");
  engram_gate_task<TEN ? 0 : exchange_offset[TEN ? 0 : SEND::buf], H, HC, W, TEN, NX>(
      g,
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 2>()),
      cnt_now(g) + POST::counter,
      (uint32_t)(POST::x * POST::y * POST::z),
      tk.x,
      tk.y,
      __int_as_float(PARAMS::v[2]),
      __int_as_float(PARAMS::v[3]),
      warm,
      TEN ? buf_at<unsigned char const>(g, slot_at<BUF_SLOTS, 3>()) : nullptr,
      TEN ? cnt_now(g) + SELF::counter + tk.y : nullptr,
      NX ? buf_at<float const>(g, slot_at<BUF_SLOTS, NX ? 4 : 0>()) : nullptr,
      NX ? buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, NX ? 5 : 0>()) : nullptr,
      NX ? buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, NX ? 6 : 0>()) : nullptr,
      NX ? __int_as_float(param_or<PARAMS, 7, 0>()) : 0.f STAGE_STAMP_ARG(L.stage_stamps));
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_engram_gate() {
  return 0;
}

} // namespace static_mk
