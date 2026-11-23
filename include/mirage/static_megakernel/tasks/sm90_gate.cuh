/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// tasks/sm90_gate.cuh -- the sm90_gate task type: what GPT-OSS does between
// o_proj and the routed experts, for one token, in one task: x1 = bf16(o +
// bias + x), h = f16(RMSNorm(x1)), router logits = bf16(h . W^T + b), and
// the token's top-k of their softmax, renormalized (ties to the lower expert):
// its weights and its column of the
// routing indices. The top-k runs in one warp, lane l the experts l, l + 32,
// ... (E a multiple of 32: GPT-OSS-20B 32, 120B 128). Task (x, y) of grid (T,
// S, 1): token x, the router's experts [y E / S, (y + 1) E / S) (S > 1: the
// logits through the token's column of the routing indices, as fp32 bits; the
// token's last task to count (its counter) takes its top-k). Every expert's
// logit is one warp's, its lanes over K, whatever S: the same sums.
#pragma once
#include "sm90_common.cuh"

namespace static_mk {

// PARAMS {H, E, TOPK, eps (fp32 bits)[, HS (h as MXFP8), logits only]} (S:
// the grid's y, the compiler's choice); BUF_SLOTS {o [T, H] fp32, o's bias
// [H] bf16, x (the residual) [T, H] bf16, the norm's weight [H] bf16, the
// router [E, H] bf16, its bias [E] bf16, the step's token count [1] i32, x1 [T,
// H] bf16, h [T, H] f16, weights [T, TOPK] fp32, routing indices [E, T] i32};
// counters {the tasks (for the consumers), the tokens' router tasks done (T;
// S > 1)}; dynamic
// shared memory: the task's E / S router rows
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_gate(Maps const &,
                                              G const &g,
                                              KernelLocals &L,
                                              StaticTask const &tk) {
  constexpr int H = PARAMS::v[0], E = PARAMS::v[1], TOPK = PARAMS::v[2],
                S = SELF::y;
  static_assert(E % 32 == 0 && E <= 32 * SM90_WARPS && TOPK <= E,
                "sm90_gate: experts a multiple of 32 (E / 32 a lane), at "
                "most 32 a router warp");
  static_assert(S >= 1 && E % (S * SM90_WARPS) == 0,
                "sm90_gate: a task's experts a multiple of its warps");
  constexpr int EPS_ = E / S; // a task's experts
  constexpr int EPL = E / 32; // experts a lane in the top-k
  using Row = Sm90Row<H>;
  constexpr int VEC = Row::VEC;
  constexpr int TOK = SELF::counter + 1;
  // o: [P][T][H] fp32, the K parts of o_proj ("store", its grid y), added here
  constexpr int P = Sm90First<IN...>::type::y;
  // MX (HS > 0): h as MXFP8, FlashInfer's padded MoE input (bf16(h) in e4m3,
  // a UE8M0 scale per 32: values [T][HS] bytes, the columns past H left as
  // they are, then the scales [T][HS / 32], FlashInfer's linear layout)
  constexpr int HS = param_or<PARAMS, 4, 0>();
  constexpr bool MX = HS > 0;
  // LG: the router's logits only, bf16(logit + bias) [T][E] at routing's
  // start (FlashInfer's MoE takes the top-k: its routing kernel), each split
  // its own experts' -- no top-k, no count between the splits
  constexpr bool LG = param_or<PARAMS, 5, 0>() != 0;
  static_assert(HS == 0 || (HS >= H && HS % 32 == 0 && H % 32 == 0),
                "h's row length");
  // TPT tokens a task (grid x = T / TPT, the compiler's choice with the
  // logits only): one load of the router rows for them all
  constexpr int TPT = T / SELF::x;
  static_assert(TPT * SELF::x == T && (TPT == 1 || LG),
                "sm90_gate: several tokens a task with the logits only");
  __shared__ float part[SM90_WARPS], logit[S == 1 ? E : 1];
  __shared__ uint4 hs[VEC]; // h as rounded (f16), for the router's warps
  __shared__ int last_split;
  int const ntok = *buf_at<int const>(g, slot_at<BUF_SLOTS, 6>());
  int const split = tk.y;
  // the weights (no producer: o_proj's bias, the norm's, the router's bias)
  // in registers before the wait, out of the path after it
  int const t0 = tk.x * TPT, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint4 obv[Row::PER], nwv[Row::PER];
  {
    uint4 const *ob = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 1>());
    uint4 const *nw = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 3>());
#pragma unroll
    for (int j = 0; j < Row::PER; j++) {
      int const v = Row::vec(j);
      if (v < VEC) {
        obv[j] = ob[v];
        nwv[j] = nw[v];
      }
    }
  }
  float rbv[EPL];
  {
    __nv_bfloat16 const *rb =
        buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 5>());
#pragma unroll
    for (int j = 0; j < EPL; j++) {
      rbv[j] = __bfloat162float(rb[32 * j + lane]);
    }
  }
  // the task's router rows (no producer) into shared memory before the wait,
  // after the barrier that ends the SM's earlier tasks (cp.async: issued here
  // whatever the compiler schedules; landed by the sync after hs)
  uint4 *wsm = reinterpret_cast<uint4 *>(L.sm);
  __syncthreads(); // this SM's earlier tasks are done with shared memory
  if (t0 < ntok) {
    uint4 const *src = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 4>()) +
                       (size_t)split * EPS_ * VEC;
    for (int i = threadIdx.x; i < EPS_ * VEC; i += SM90_THREADS) {
      uint32_t const d = (uint32_t)__cvta_generic_to_shared(wsm + i);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d),
                   "l"(src + i)
                   : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  sm90_task_begin<IN...>(g);
  int *routing = buf_at<int>(g, slot_at<BUF_SLOTS, 10>());
#pragma unroll 1
  for (int it = 0; it < TPT; it++) {
    int const t = t0 + it;
    float const *o =
        buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()) + (size_t)t * H;
    uint4 const *res =
        buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 2>()) + (size_t)t * VEC;
    uint4 *x1 = buf_at<uint4>(g, slot_at<BUF_SLOTS, 7>()) + (size_t)t * VEC;
    uint4 *hout = buf_at<uint4>(g, slot_at<BUF_SLOTS, 8>()) + (size_t)t * VEC;
    // 1. x1 = bf16(o + bias + x), and the row as rounded
    float xr[Row::PER][8];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < Row::PER; j++) {
      int const v = Row::vec(j);
      if (v < VEC) {
        float f[8], b[8], r[8];
        f32x8_load(o, v, f);
#pragma unroll
        for (int p = 1; p < P; p++) {
          float q[8];
          f32x8_load(o + (size_t)p * T * H, v, q);
#pragma unroll
          for (int i = 0; i < 8; i++) {
            f[i] += q[i];
          }
        }
        bf16x8_to_f32(obv[j], b);
        bf16x8_to_f32(res[v], r);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          f[i] = f[i] + b[i] + r[i];
        }
        uint4 const q = f32_to_bf16x8(f);
        if (split == 0) {
          x1[v] = q;
        }
        bf16x8_to_f32(q, xr[j]);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          ss += xr[j][i] * xr[j][i];
        }
      }
    }
    // 2. h = f16(x1 / rms(x1) * w), kept as rounded for the router (in hs)
    float const rs =
        rsqrtf(sm90_block_sum(ss, part) / H + __int_as_float(PARAMS::v[3]));
#pragma unroll
    for (int j = 0; j < Row::PER; j++) {
      int const v = Row::vec(j);
      if (v < VEC) {
        float w[8];
        bf16x8_to_f32(nwv[j], w);
#pragma unroll
        for (int i = 0; i < 8; i++) {
          xr[j][i] = xr[j][i] * rs * w[i];
        }
        uint4 const q = f32_to_f16x8(xr[j]);
        if (split == 0 && !MX) {
          hout[v] = q;
        }
        hs[v] = q;
      }
    }
    if constexpr (MX) { // 4 threads' vectors a scale group (vec(j): threadIdx.x
                        // + 256 j); every lane in the shuffles
      uint8_t *hq = buf_at<uint8_t>(g, slot_at<BUF_SLOTS, 8>());
#pragma unroll
      for (int j = 0; j < Row::PER; j++) {
        int const v = Row::vec(j);
        float f[8], am = 0.f;
        if (v < VEC) {
          bf16x8_to_f32(f32_to_bf16x8(xr[j]), f);
#pragma unroll
          for (int i = 0; i < 8; i++) {
            am = fmaxf(am, fabsf(f[i]));
          }
        }
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
        am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
        if (v < VEC && split == 0) {
          int ex = ceil_log2_bits(fmaxf(am, 1.0e-30f) * (1.0f / 448.0f));
          ex = ex < -127 ? -127 : (ex > 127 ? 127 : ex);
          float const sc = __uint_as_float((uint32_t)(127 - ex) << 23);
          uint2 p;
          p.x = (uint32_t)cvt_e4m3(f[0] * sc) |
                ((uint32_t)cvt_e4m3(f[1] * sc) << 8) |
                ((uint32_t)cvt_e4m3(f[2] * sc) << 16) |
                ((uint32_t)cvt_e4m3(f[3] * sc) << 24);
          p.y = (uint32_t)cvt_e4m3(f[4] * sc) |
                ((uint32_t)cvt_e4m3(f[5] * sc) << 8) |
                ((uint32_t)cvt_e4m3(f[6] * sc) << 16) |
                ((uint32_t)cvt_e4m3(f[7] * sc) << 24);
          *reinterpret_cast<uint2 *>(hq + (size_t)t * HS + 8 * v) = p;
          if ((v & 3) == 0) {
            hq[(size_t)T * HS + (size_t)t * (HS / 32) + v / 4] =
                (uint8_t)(ex + 127);
          }
        }
      }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads(); // hs written
    // 3. the logits: warp w experts [w EPW, w EPW + EPW) (rows in shared
    // memory), its lanes over K, reduced over the warp
    constexpr int EPW = EPS_ / SM90_WARPS;
    constexpr int UNROLL = EPW <= 4 ? 4 : 1; // the EPW rows' loads in registers
    int const e0 = split * EPS_ + warp * EPW; // the warp's first expert
    if (t < ntok) {
      uint4 const *rwr = wsm + (size_t)warp * EPW * VEC;
      float acc[EPW] = {};
#pragma unroll UNROLL
      for (int v = lane; v < VEC; v += 32) {
        float hf[8];
        f16x8_to_f32(hs[v], hf);
        uint4 wv[EPW];
#pragma unroll
        for (int e = 0; e < EPW; e++) {
          wv[e] = rwr[(size_t)e * VEC + v];
        }
#pragma unroll
        for (int e = 0; e < EPW; e++) {
          float w[8];
          bf16x8_to_f32(wv[e], w);
#pragma unroll
          for (int i = 0; i < 8; i++) {
            acc[e] += w[i] * hf[i];
          }
        }
      }
#pragma unroll
      for (int e = 0; e < EPW; e++) {
        acc[e] = sm90_warp_sum(acc[e]);
      }
      if (lane < EPW) {
        float a = 0.f;
#pragma unroll
        for (int e = 0; e < EPW; e++) {
          a = e == lane ? acc[e] : a;
        }
        if constexpr (LG) {
          __nv_bfloat16 const *rb =
              buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 5>());
          reinterpret_cast<__nv_bfloat16 *>(routing)[t * E + e0 + lane] =
              __float2bfloat16(a + __bfloat162float(rb[e0 + lane]));
        } else if constexpr (S == 1) {
          logit[e0 + lane] = a;
        } else {
          routing[(e0 + lane) * T + t] = __float_as_int(a);
        }
      }
    } else if (lane < EPW) {
      if constexpr (LG) {
        reinterpret_cast<__nv_bfloat16 *>(routing)[t * E + e0 + lane] =
            __float2bfloat16(0.f);
      } else if constexpr (S == 1) {
        logit[e0 + lane] = 0.f;
      } else {
        routing[(e0 + lane) * T + t] = 0;
      }
    }
    if constexpr (TPT > 1) {
      __syncthreads(); // hs, part and logit read: the next token's
    }
  } // the task's tokens
  int const t = t0;
  if constexpr (LG) { // the top-k is FlashInfer's
    sm90_task_end<SELF>(g);
    return;
  }
  if constexpr (S > 1) {
    __threadfence(); // the logits before the count below
  }
  __syncthreads();
  if constexpr (S > 1) {
    // the token's last task to count takes its top-k (the others' logits in
    // L2 before their counts)
    if (threadIdx.x == 0) {
      __threadfence();
      last_split = atomicAdd(g.cnt + TOK + t, 1u) == S - 1;
      __threadfence();
    }
    __syncthreads();
    if (!last_split) {
      sm90_task_end<SELF>(g);
      return;
    }
  }
  // 4. warp 0, lane l the experts j 32 + l: the token's top-k (rows past
  // the step's tokens: none)
  if (warp == 0) {
    float l[EPL], m = -INFINITY;
#pragma unroll
    for (int j = 0; j < EPL; j++) {
      int const e = 32 * j + lane;
      float lg;
      if constexpr (S == 1) {
        lg = logit[e];
      } else {
        lg = __int_as_float(__ldcg(routing + e * T + t));
      }
      l[j] = __bfloat162float(__float2bfloat16(lg + rbv[j]));
      m = fmaxf(m, l[j]);
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
      m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, s));
    }
    float p[EPL], sum = 0.f;
#pragma unroll
    for (int j = 0; j < EPL; j++) {
      p[j] = expf(l[j] - m);
      sum += p[j];
    }
#pragma unroll
    for (int s = 16; s > 0; s >>= 1) {
      sum += __shfl_xor_sync(0xffffffffu, sum, s);
    }
    int rank[EPL]; // 1 + the expert's top-k slot, 0: not chosen
#pragma unroll
    for (int j = 0; j < EPL; j++) {
      p[j] /= sum;
      rank[j] = 0;
    }
    float wsum = 0.f, wk[TOPK];
#pragma unroll
    for (int k = 0; k < TOPK; k++) {
      // this lane's best candidate (its lower expert on a tie), then the
      // warp's
      float best = -1.f;
      int who = 0;
#pragma unroll
      for (int j = EPL - 1; j >= 0; j--) {
        if (rank[j] == 0 && p[j] >= best) {
          best = p[j], who = 32 * j + lane;
        }
      }
#pragma unroll
      for (int s = 16; s > 0; s >>= 1) {
        float const ob_ = __shfl_xor_sync(0xffffffffu, best, s);
        int const ow = __shfl_xor_sync(0xffffffffu, who, s);
        if (ob_ > best || (ob_ == best && ow < who)) {
          best = ob_, who = ow;
        }
      }
      wk[k] = best;
      wsum += best;
#pragma unroll
      for (int j = 0; j < EPL; j++) {
        rank[j] = who == 32 * j + lane ? k + 1 : rank[j];
      }
    }
    bool const active_tok = t < ntok;
#pragma unroll
    for (int j = 0; j < EPL; j++) {
      routing[(32 * j + lane) * T + t] = active_tok ? rank[j] : 0;
    }
    if (lane < TOPK) {
      float w = 0.f;
#pragma unroll
      for (int k = 0; k < TOPK; k++) {
        w = k == lane ? wk[k] : w;
      }
      buf_at<float>(g, slot_at<BUF_SLOTS, 9>())[t * TOPK + lane] =
          active_tok ? w / wsum : 0.f;
    }
  }
  sm90_task_end<SELF>(g);
}

SM90_TASK_SMEM(sm90_gate, PARAMS::v[1] / SELF::y * PARAMS::v[0] * 2)

} // namespace static_mk
