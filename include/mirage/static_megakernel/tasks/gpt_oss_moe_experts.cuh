// Static GPT-OSS-120B routed experts.
//
// This is a single StaticMegakernel task node.  It reuses the runtime's
// TCGEN05 MXFP8 x MXFP4 issue path for W13 and uses the same 128x128 MXFP4
// MMA primitive for a serialized, one-K-tile W2 ring.  The W2 streaming is
// deliberate: for the GPT-OSS padded intermediate (2944, 23 K tiles), the
// legacy moe_experts ExpertSmem layout cannot hold its 3-stage W2 ring or its
// full set of TMEM scales.
//
// The arithmetic contract follows GPT-OSS' gated activation and FlashInfer's
// Blackwell block-scaled expert path: W13 gate/up biases, gate clamp [*, 7],
// up clamp [-7, 7], SwiGLU alpha=1.702 and beta=1, then W2 bias.  The W13
// activation is rounded to fp16 before being quantized to E4M3 with UE8M0
// scales per K32 group, matching the GPT-OSS static fallback's intermediate.
#pragma once

#include "moe_experts.cuh" // ExpertSlots, build_table, wait_z_landed
#include <cuda_fp16.h>

namespace static_mk {

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class ZQ,
          class ROUTING>
using GptOssExpertSlotsOf =
    ExpertSlots<slot_at<BUF_SLOTS, 0>(), // h_q: [slot, T, PAD] E4M3
                slot_at<BUF_SLOTS, 1>(), // h_q scales: [slot, KT2, 512]
                SELF::counter,
                slot_at<MAP_SLOTS, 0>(),
                slot_at<MAP_SLOTS, 1>(),
                slot_at<MAP_SLOTS, 2>(),
                slot_at<MAP_SLOTS, 3>(),
                slot_at<MAP_SLOTS, 4>(),
                slot_at<MAP_SLOTS, 5>(),
                slot_at<MAP_SLOTS, 6>(),
                slot_at<MAP_SLOTS, 7>(),
                slot_at<MAP_SLOTS, 8>(),
                slot_at<MAP_SLOTS, 9>(),
                slot_at<MAP_SLOTS, 10>(),
                slot_at<MAP_SLOTS, 11>(),
                slot_at<MAP_SLOTS, 12>(),
                exchange_offset[ZQ::buf],
                ROUTING::v[1], // E from sm90_gate params {H,E,K,eps}
                ROUTING::v[2], // K
                PARAMS::v[1],  // padded hidden width / W13 K
                PARAMS::v[1]>; // padded intermediate width

namespace gpt_oss_moe_detail {

// Every W13 item covers 64 gate rows and the matching 64 up rows.  The input
// map's piece index is expert-major, then (paired 64-row tile, K128 tile).
template <class RES>
__device__ __forceinline__ TileJob
    w13_job(Maps const &maps, int expert, int tile, int zq_map, int zqkt_map) {
  TileJob j{};
  j.kind = 1;
  j.natoms = RES::KT_LAT;
  j.nst = (RES::KT_LAT + 1) / 2;
  j.k0 = 0;
  j.wmap = &maps.m[RES::W13];
  j.sfmap = &maps.m[RES::W13SF];
  j.wmap2 = &maps.m[RES::W13X2];
  j.sfmap2 = &maps.m[RES::W13SFX2];
  j.amap = &maps.m[zq_map];
  j.amap2 = &maps.m[zqkt_map];
  j.tile0 = j.chunk0 = expert * (RES::MT13 * RES::KT_LAT) + tile * RES::KT_LAT;
  return j;
}

template <class RES>
__device__ __forceinline__ void
    route_table(int const *routing, float const *topk_w, int tokens) {
  int *const cnt_e = RES::cnt_e();
  int *const sel = RES::sel();
  float *const wsel = RES::wsel();
  for (int e = threadIdx.x; e < RES::NE; e += 256) {
    cnt_e[e] = 0;
  }
  __syncthreads();

  // sm90_gate_layer writes routing[E,T] with selected rank k+1.  The table
  // builder expects one expert id and one route weight for each (token,k).
  for (int i = threadIdx.x; i < RES::NPAIR; i += 256) {
    int const t = i / RES::K, k = i % RES::K;
    int expert = 0;
    bool found = false;
    for (int e = 0; e < RES::NE; ++e) {
      if (routing[e * tokens + t] == k + 1) {
        expert = e;
        found = true;
        break;
      }
    }
    sel[i] = expert;
    wsel[i] = found ? topk_w[t * RES::K + k] : 0.f;
    if (found) {
      atomicExch(&cnt_e[expert], 1);
    }
  }
  __syncthreads();
  build_table<RES>();
}

} // namespace gpt_oss_moe_detail

// GPT-OSS task: wait for packed h2, build the route table from rank-coded
// routing, run W13 and W2 from one CTA per SM, then write the final residual
// result to the explicit model tensor slot.
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class ZQ,
          class ROUTING,
          class... REST>
__device__ __forceinline__ void run_gpt_oss_moe_experts(Maps const &maps,
                                                        G const &g,
                                                        KernelLocals &L,
                                                        StaticTask const &) {
  constexpr int H = PARAMS::v[0], PAD = PARAMS::v[1];
  constexpr int HQ = slot_at<BUF_SLOTS, 0>(), HSF = slot_at<BUF_SLOTS, 1>(),
                PARTIAL = slot_at<BUF_SLOTS, 2>(),
                ROUTE = slot_at<BUF_SLOTS, 3>(),
                TOPK_W = slot_at<BUF_SLOTS, 4>(), X1 = slot_at<BUF_SLOTS, 5>(),
                B13 = slot_at<BUF_SLOTS, 6>(), B2 = slot_at<BUF_SLOTS, 7>(),
                OUT = slot_at<BUF_SLOTS, 8>();
  using RES =
      GptOssExpertSlotsOf<SELF, PARAMS, BUF_SLOTS, MAP_SLOTS, ZQ, ROUTING>;
  static_assert(H == 2880 && PAD == 2944 && RES::LAT == PAD && RES::IR == PAD,
                "gpt_oss_moe_experts: GPT-OSS-120B uses H=2880, PAD=2944");
  static_assert(
      T == 8 && RES::K == 4 && RES::NE == 128,
      "gpt_oss_moe_experts: current static route table is T=8,K=4,E=128");
  static_assert(RES::MT13 == 46 && RES::KT_LAT == 23 && RES::KT2 == 23 &&
                    RES::OT2 == 23,
                "gpt_oss_moe_experts: expected 46 paired W13 tiles and 23 "
                "K/output tiles");
  static_assert(
      SELF::counter >= 0 && ROUTE >= 0 && ZQ::buf >= 0 && HQ >= 0 && HSF >= 0 &&
          PARTIAL >= 0 && TOPK_W >= 0 && X1 >= 0 && B13 >= 0 && B2 >= 0 &&
          OUT >= 0,
      "gpt_oss_moe_experts: missing node counter or input/scratch slots");
  static_assert(
      exchange_bytes[ZQ::buf] == zq_bytes(PAD),
      "gpt_oss_moe_experts: ZQ must be the packed E4M3/UE8M0 exchange bank");

  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int const tokens = T;
  int const *routing = buf_at<int const>(g, ROUTE);
  float const *topk_w = buf_at<float const>(g, TOPK_W);
  __nv_bfloat16 const *x1 = buf_at<__nv_bfloat16 const>(g, X1);
  __nv_bfloat16 const *b13 = buf_at<__nv_bfloat16 const>(g, B13);
  __nv_bfloat16 const *b2 = buf_at<__nv_bfloat16 const>(g, B2);
  __nv_bfloat16 *output = buf_at<__nv_bfloat16>(g, OUT);
  uint8_t *hq = buf_at<uint8_t>(g, HQ);
  uint8_t *hsf = buf_at<uint8_t>(g, HSF);
  float *partials = buf_at<float>(g, PARTIAL);
  int *const eid = RES::eid(), *const ntok = RES::ntok(),
             *const tok = RES::tok(), *const start = RES::start();
  unsigned char *const kk = RES::kk();
  float *acc_s = reinterpret_cast<float *>(rt_sm() + RING_BYTES);

  // The exchange producer uses 0xFF prefill and writes each 16-byte piece;
  // waiting for the pieces also covers its packed per-token scale chunks.
  __syncthreads();
  for (int e = threadIdx.x; e < RES::NE; e += 256) {
    RES::cnt_e()[e] = 0;
  }
  wait_z_landed<RES>(g);
  asm volatile("fence.proxy.async.global;" ::: "memory");
  if constexpr (ROUTING::counter >= 0) {
    if (threadIdx.x == 0) {
      cnt_wait(cnt_now(g) + ROUTING::counter,
               ROUTING::x * ROUTING::y * ROUTING::z);
    }
    __syncthreads();
  }

  // Stage the input's per-K128 scale chunks once in the free TMEM columns.
  // The map pair is set 0/set 1, just like the input activation maps.
  if (threadIdx.x == 0) {
    ring_reinit(true);
    mbar_expect(rt.misc0, RES::KT_LAT * SF_CHUNK);
    tma3(&maps.m[RES::ZQ_SF + dbuf_par],
         rt.misc0,
         L.base + OFF_W,
         0,
         0,
         0,
         EVICT_LAST);
  }
  __syncthreads();
  mbar_wait(rt.misc0, 0);
  fence_after();
  if (warp == 1 && lane == 0) {
    for (int kt = 0; kt < RES::KT_LAT; ++kt) {
      cp_sf(L.tb + 72 + 4 * kt, L.base + OFF_W + kt * SF_CHUNK);
    }
    tc_commit(rt.misc0 + 16);
  }
  mbar_wait(rt.misc0 + 16, 0);
  fence_after();
  __syncthreads();

  gpt_oss_moe_detail::route_table<RES>(routing, topk_w, tokens);
  int const n_live = s_ints[1], n13 = n_live * RES::MT13;
  if (threadIdx.x == 0) {
    L.gl = L.gi0 = L.gi1 = L.wt = 0;
  }
  __syncthreads();
  int gl = 0, gi0 = 0, gi1 = 0, wt = 0;

  // W13: one queue item per (live expert slot, paired 64-row feature tile).
  // Warps 0/7 load even/odd ring stages, warps 1/6 issue the two TCGEN05
  // accumulators, and warps 2..5 quantize the gated activation into h_q.
  for (;;) {
    if (threadIdx.x == 0) {
      s_ints[2] = (int)atom_inc_lane0(cnt_now(g) + RES::W2NEXT);
    }
    __syncthreads();
    int const idx = s_ints[2];
    if (idx >= n13) {
      break;
    }
    int const slot = idx / RES::MT13, mt = idx % RES::MT13;
    int const expert = eid[slot];
    TileJob const j = gpt_oss_moe_detail::w13_job<RES>(
        maps, expert, mt, RES::ZQ + dbuf_par, RES::ZQ_KT + dbuf_par);
    int const a = wt & 3, par = (wt >> 2) & 1;
    if ((warp == 1 || warp == 6) && wt >= 4) {
      mbar_wait(rt.acc_empty0 + 8 * a, ((wt >> 2) - 1) & 1);
    }
    if (warp == 0 && lane == 0) {
      load_job<128, 8192, true>(j, L.base, gl, 0);
    } else if (warp == 7 && lane == 0) {
      load_job<128, 8192, true>(j, L.base, gl, 1);
    } else if (warp == 1 || warp == 6) {
      int const iss = (warp == 1) ? 0 : 1;
      int &gi = (iss == 0) ? gi0 : gi1;
      issue_job_warp(j, L.base, L.tb, iss, gi, a, L.tb + 72);
    } else if (warp >= 2 && warp <= 5) {
      float v[T];
      int const row = drain_acc(L.tb, a, par, v);
      if (row >= 0) {
#pragma unroll
        for (int t = 0; t < T; ++t) {
          acc_s[row * T + t] = v[t];
        }
      }
      asm volatile("bar.sync 1, 128;" ::: "memory");
      if (row >= 0 && row < 64) {
        int const f = mt * 64 + row;
        int const ebase = expert * 2 * PAD;
        float gate[T], up[T], hidden[T], amax[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
          float const bg = f < H ? __bfloat162float(b13[ebase + f]) : 0.f;
          float const bu = f < H ? __bfloat162float(b13[ebase + PAD + f]) : 0.f;
          gate[t] = fminf(acc_s[row * T + t] + bg, 7.0f);
          up[t] = fminf(fmaxf(acc_s[(64 + row) * T + t] + bu, -7.0f), 7.0f);
          hidden[t] =
              (up[t] + 1.0f) * gate[t] / (1.0f + __expf(-1.702f * gate[t]));
          // GPT-OSS reference materializes its activated intermediate in fp16
          // before feeding the MXFP8 expert operand.
          hidden[t] = __half2float(__float2half_rn(hidden[t]));
          amax[t] = fabsf(hidden[t]);
        }
#pragma unroll
        for (int d = 16; d > 0; d >>= 1)
#pragma unroll
          for (int t = 0; t < T; ++t) {
            amax[t] = fmaxf(amax[t], __shfl_xor_sync(0xffffffffu, amax[t], d));
          }
        uint8_t q[T], scale[T];
#pragma unroll
        for (int t = 0; t < T; ++t) {
          int ex = ceil_log2_bits(fmaxf(amax[t], 1.0e-30f) * (1.0f / 448.0f));
          ex = ex < -127 ? -127 : (ex > 127 ? 127 : ex);
          float const inv = __uint_as_float((uint32_t)(127 - ex) << 23);
          q[t] = cvt_e4m3_nv(hidden[t] * inv);
          scale[t] = (uint8_t)(ex + 127);
        }
#pragma unroll
        for (int t = 0; t < T; ++t) {
          hq[((size_t)slot * T + t) * PAD + f] = q[t];
          if (lane == 0) {
            int const kt = f >> 7, k32 = (f >> 5) & 3;
            hsf[((size_t)slot * RES::KT2 + kt) * SF_CHUNK + t * 16 + k32] =
                scale[t];
          }
        }
      }
      asm volatile("bar.sync 1, 128;" ::: "memory");
    }
    __syncthreads();
    ++wt;
  }

  // All W13 activations/scales are visible before any W2 TMA reads them.
  if (threadIdx.x == 0) {
    cnt_add(cnt_now(g) + SELF::counter + 1, 1);
  }
  cnt_wait(cnt_now(g) + SELF::counter + 1, NSM);
  __syncthreads();

  // Start a fresh ring epoch for W2.  W2's 23 K tiles reuse stage 0 and its
  // scale columns only after the previous TCGEN05 MMAs have retired.
  if (threadIdx.x == 0) {
    ring_reinit(true);
    L.gl = L.gi0 = L.gi1 = L.wt = 0;
  }
  __syncthreads();
  wt = 0;
  int w2_tile_base = 0;
  int const n2 = n_live * RES::OT2;
  for (;;) {
    if (threadIdx.x == 0) {
      s_ints[2] = (int)atom_inc_lane0(cnt_now(g) + RES::W2NEXT + 1);
    }
    __syncthreads();
    int const idx = s_ints[2];
    if (idx >= n2) {
      break;
    }
    int const slot = idx / RES::OT2, ot = idx % RES::OT2;
    int const expert = eid[slot], a = wt & 3, par = (wt >> 2) & 1;
    if ((warp == 1 || warp == 6) && wt >= 4) {
      mbar_wait(rt.acc_empty0 + 8 * a, ((wt >> 2) - 1) & 1);
    }

    if (warp == 0 && lane == 0) {
      int const tile0 = expert * (RES::OT2 * RES::KT2) + ot * RES::KT2;
      for (int kt = 0; kt < RES::KT2; ++kt) {
        // K tile seq in ring stage seq % SMAX: up to SMAX tiles in flight (the
        // stage's weight piece, its scales, the slot's activation and scales)
        int const seq = w2_tile_base + kt, st = seq % SMAX;
        if (seq >= SMAX) {
          mbar_wait(rt.empty0 + 8 * st, ((seq / SMAX) - 1) & 1);
        }
        uint32_t const full = rt.full0 + 8 * st;
        uint32_t const stage = L.base + st * FSTAGE;
        uint32_t const sf = L.base + OFF_WSF + st * 2 * SF_CHUNK;
        mbar_expect(full, 8192 + SF_CHUNK + T * 128 + SF_CHUNK);
        tma3(&maps.m[RES::W2], full, stage, 0, 0, tile0 + kt, EVICT_FIRST);
        tma3(&maps.m[RES::W2SF], full, sf, 0, 0, tile0 + kt, EVICT_FIRST);
        tma3(&maps.m[RES::HQ_KT],
             full,
             stage + W_STAGE,
             0,
             slot * T,
             kt,
             EVICT_LAST);
        tma3(&maps.m[RES::HQ_SF],
             full,
             sf + SF_CHUNK,
             0,
             0,
             slot * RES::KT2 + kt,
             EVICT_LAST);
      }
    } else if (warp == 1 || warp == 6) {
      for (int kt = 0; kt < RES::KT2; ++kt) {
        int const seq = w2_tile_base + kt, st = seq % SMAX;
        mbar_wait(rt.full0 + 8 * st, (seq / SMAX) & 1);
        fence_after();
        bool const one = issuer_lane<true>();
        if (warp == 6) {
          if (one) {
            mbar_arrive(rt.empty0 + 8 * st);
          }
        } else {
          if (one) {
            // the stage's own scale columns (272 + 8 st: clear of the W13
            // scales at 232 .. 271 and the accumulators), so a stage's copy
            // never overwrites scales an earlier stage's MMAs still read
            uint32_t const sfa = L.tb + 272 + 8 * st, sfb = sfa + 4;
            uint32_t const stage = L.base + st * FSTAGE;
            uint32_t const wsf = L.base + OFF_WSF + st * 2 * SF_CHUNK;
            cp_sf(sfa, wsf);
            cp_sf(sfb, wsf + SF_CHUNK);
            uint64_t const dw = mkdesc(stage), dh = mkdesc(stage + W_STAGE);
            mma_mx(
                L.tb + T * a, dw, dh, idesc_mx(0), kt == 0 ? 0u : 1u, sfa, sfb);
            mma_mx(L.tb + T * a, dw + 2, dh + 2, idesc_mx(1), 1u, sfa, sfb);
            mma_mx(L.tb + T * a, dw + 4, dh + 4, idesc_mx(2), 1u, sfa, sfb);
            mma_mx(L.tb + T * a, dw + 6, dh + 6, idesc_mx(3), 1u, sfa, sfb);
            tc_commit(rt.empty0 + 8 * st);
          }
        }
      }
      if (warp == 1 && issuer_lane<true>()) {
        tc_commit(rt.acc_full0 + 8 * a);
      } else if (warp == 6 && issuer_lane<true>()) {
        mbar_arrive(rt.acc_full0 + 8 * a);
      }
    } else if (warp >= 2 && warp <= 5) {
      int const nt = ntok[slot], row = (warp & 3) * 32 + lane;
      float v[8];
      mbar_wait(rt.acc_full0 + 8 * a, par);
      fence_after();
      tmem_ld8(L.tb + 8 * a + ((uint32_t)((warp & 3) * 32) << 16), v);
      fence_before();
      mbar_arrive(rt.acc_empty0 + 8 * a);
      for (int i = start[slot]; i < start[slot] + nt; ++i) {
        int const t = tok[i];
        float vt = v[0];
#pragma unroll
        for (int k = 1; k < T; ++k) {
          vt = (t == k) ? v[k] : vt;
        }
        int const col = ot * 128 + row;
        if (col < H) {
          partials[((size_t)t * RES::K + kk[i]) * H + col] =
              vt + __bfloat162float(b2[(size_t)expert * PAD + col]);
        }
      }
    }
    __syncthreads();
    ++wt;
    w2_tile_base += RES::KT2;
  }

  if (threadIdx.x == 0) {
    cnt_add(cnt_now(g) + SELF::counter + 2, 1);
  }
  cnt_wait(cnt_now(g) + SELF::counter + 2, NSM);
  __syncthreads();

  // Deterministic k-order accumulation mirrors the existing allreduce_send
  // contract and writes each output vector once, including the residual.
  size_t const elements = (size_t)T * H;
  for (size_t i = (size_t)L.sm_id * 256 + threadIdx.x; i < elements;
       i += (size_t)NSM * 256) {
    int const t = (int)(i / H), c = (int)(i - (size_t)t * H);
    float sum = 0.f;
#pragma unroll
    for (int k = 0; k < RES::K; ++k) {
      sum += RES::wsel()[t * RES::K + k] *
             partials[((size_t)t * RES::K + k) * H + c];
    }
    output[i] = __float2bfloat16(__bfloat162float(x1[i]) + sum);
  }
  __syncthreads();
  L.ring_relaid = true;
  if (threadIdx.x == 0) {
    cnt_add(cnt_now(g) + SELF::counter, 1);
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class ZQ,
          class ROUTING,
          class... REST>
constexpr int smem_gpt_oss_moe_experts() {
  return RING_BYTES + 128 * T * sizeof(float);
}

} // namespace static_mk
