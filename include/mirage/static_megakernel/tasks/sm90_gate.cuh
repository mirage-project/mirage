// tasks/sm90_gate.cuh -- the sm90_gate task type: what GPT-OSS does between
// o_proj and the routed experts, for one token, in one task (a node each was
// a dependency hop each): x1 = bf16(o + bias + x) (sm90_bias), h = f16(
// RMSNorm(x1)) (sm90_rmsnorm), the router logits bf16(h . W^T + b)
// (sm90_router), and the token's top-k of their softmax, renormalized
// (sm90_topk: ties to the lower expert): its weights and its column of the
// routing indices. The top-k runs in one warp, lane l the experts l, l + 32,
// ... (E a multiple of 32: GPT-OSS-20B 32, 120B 128). The tokens' experts meet
// in a bit mask (E / 32 words of the node's counter line), and the last task
// done writes the active-expert list. Task x of grid (T, 1, 1): token x.
#pragma once
#include "sm90_common.cuh"

namespace static_mk {

// PARAMS {H, E, TOPK, eps (fp32 bits)}; BUF_SLOTS {o [T, H] fp32, o's bias [H]
// bf16, x (the residual) [T, H] bf16, the norm's weight [H] bf16, the router
// [E, H] bf16, its bias [E] bf16, the step's token count [1] i32, x1 [T, H]
// bf16, h [T, H] f16, weights [T, TOPK] fp32, routing indices [E, T] i32, the
// active-expert list [E + 1] i32}; counters {the tasks (for the consumers),
// the tasks done (for the last), the expert mask (E / 32 words)}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_gate(Maps const &,
                                              G const &g,
                                              KernelLocals &,
                                              StaticTask const &tk) {
  constexpr int H = PARAMS::v[0], E = PARAMS::v[1], TOPK = PARAMS::v[2];
  static_assert(E % 32 == 0 && E <= 32 * SM90_WARPS && TOPK <= E,
                "sm90_gate: experts a multiple of 32 (E / 32 a lane), at "
                "most 32 a router warp");
  constexpr int EPL = E / 32; // experts a lane in the top-k
  using Row = Sm90Row<H>;
  constexpr int VEC = Row::VEC;
  constexpr int DONE = SELF::counter + 1, MASK = SELF::counter + 2;
  __shared__ float part[SM90_WARPS], logit[E];
  __shared__ uint4 hs[VEC]; // h as rounded (f16), for the router's warps
  __shared__ int last;
  {
    // the router (no producer) into L2 while o_proj finishes
    char const *rp = reinterpret_cast<char const *>(
        buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 4>()));
    for (int i = threadIdx.x; i < E * H * 2 / 128; i += SM90_THREADS) {
      asm volatile("prefetch.global.L2 [%0];" ::"l"(rp + 128 * i));
    }
  }
  sm90_task_begin<IN...>(g);
  int const t = tk.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  float const *o =
      buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()) + (size_t)t * H;
  uint4 const *ob = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 1>());
  uint4 const *res =
      buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 2>()) + (size_t)t * VEC;
  uint4 const *nw = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 3>());
  uint4 const *rw = buf_at<uint4 const>(g, slot_at<BUF_SLOTS, 4>());
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
      bf16x8_to_f32(ob[v], b);
      bf16x8_to_f32(res[v], r);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        f[i] = f[i] + b[i] + r[i];
      }
      uint4 const q = f32_to_bf16x8(f);
      x1[v] = q;
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
      bf16x8_to_f32(nw[v], w);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        xr[j][i] = xr[j][i] * rs * w[i];
      }
      uint4 const q = f32_to_f16x8(xr[j]);
      hout[v] = q;
      hs[v] = q;
    }
  }
  __syncthreads(); // hs written
  // 3. the logits: warp w experts [w EPW, w EPW + EPW), its lanes over K
  // (each step its EPW rows' vectors in flight), reduced over the warp
  constexpr int EPW = E / SM90_WARPS;
  constexpr int UNROLL = EPW <= 4 ? 4 : 1; // the EPW rows' loads in registers
  {
    uint4 const *rwr = rw + (size_t)warp * EPW * VEC;
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
      logit[warp * EPW + lane] = a;
    }
  }
  __syncthreads();
  // 4. warp 0, lane l the experts j 32 + l: the token's top-k (rows past
  // the step's tokens: none)
  if (warp == 0) {
    int const ntok = *buf_at<int const>(g, slot_at<BUF_SLOTS, 6>());
    __nv_bfloat16 const *rb =
        buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 5>());
    float l[EPL], m = -INFINITY;
#pragma unroll
    for (int j = 0; j < EPL; j++) {
      int const e = 32 * j + lane;
      l[j] = __bfloat162float(
          __float2bfloat16(logit[e] + __bfloat162float(rb[e])));
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
    int *routing = buf_at<int>(g, slot_at<BUF_SLOTS, 10>());
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
#pragma unroll
    for (int j = 0; j < EPL; j++) {
      uint32_t const bits = __ballot_sync(0xffffffffu, rank[j] > 0);
      if (lane == 0 && active_tok) {
        atomicOr(g.cnt + MASK + j, bits);
      }
    }
    if (lane == 0) {
      __threadfence(); // the mask (and the routing column) before the count
      last = atomicAdd(g.cnt + DONE, 1u) == T - 1;
    }
    __syncwarp();
    // 5. the last task: the active-expert list, ascending
    if (__shfl_sync(0xffffffffu, last, 0)) {
      __threadfence();
      int *list = buf_at<int>(g, slot_at<BUF_SLOTS, 11>());
      int before = 0; // the active experts below word j
#pragma unroll
      for (int j = 0; j < EPL; j++) {
        uint32_t const mask = *(uint32_t volatile *)(g.cnt + MASK + j);
        if ((mask >> lane) & 1u) {
          list[before + __popc(mask & ((1u << lane) - 1u))] = 32 * j + lane;
        }
        before += __popc(mask);
      }
      if (lane == 0) {
        list[E] = before;
      }
    }
  }
  sm90_task_end<SELF>(g);
}

SM90_TASK_SMEM(sm90_gate, 0)

} // namespace static_mk
