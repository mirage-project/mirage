// tasks/mla_vgate.cuh (HIP, gfx950) -- K3 MLA after the attention, one head:
// v = attn . w_vc, the output gate (math: mpk-kernels tests/mla_reference.py
// core()). Task (x, y) = head x, tokens [TG y, TG y + TG) of grid (heads, T / TG, 1) (TG <= 16; T = 8: one group):
//   attn   = mla_attn's output of this head ([T][heads x KV] bf16, polled)
//   v      = bf16(attn . w_vc[h])             w_vc given as [heads][VD][KV] bf16 (in LDS
//            from the task's start)
//   gate   = bf16(g_proj's K parts added) (a gemm_tile node's K parts in slots,
//            [P][T][heads x VD] fp32, 0xFF-prefilled, polled)
//   o_in   = bf16(v * bf16(sigmoid(gate)))    -> o_in [T][heads x VD] bf16 (the
//            node's buffer, 0xFF-prefilled: o_proj polls it)
#pragma once
#include "../core.cuh"
#include "mla_norm.cuh" // mla_st4_dev
#include "mla_attn.cuh" // mla_a::CH


namespace static_mk {

// one 16-B piece per lane from global memory into LDS at lds + 16 lane (lds: the
// same in every lane; nt: the weights are read once per launch)
// o_in staged in LDS, then each token's 256 B of the head in 16-B stores (kda38: MLA 37 rows 48.0 .. 48.8 vs 51.3 ..
// 52.6 us like for like with 4-B stores, each instruction 8 tokens x 4 row groups apart; MI355X 10-13 kda27: the next
// task on a vgate CU started 3.5 .. 4.7 us after it, o_proj saw o_in 5.2 us after the last vgate)
__device__ __forceinline__ void vgate_lds_dma16(void const *src, void *lds) {
  __builtin_amdgcn_global_load_lds((__attribute__((address_space(1))) void *)(const_cast<void *>(src)),
                                   (__attribute__((address_space(3))) void *)(lds), 16, 0, 2);
}

// NSV > 0 (MPK_MLA_COMB_VGATE: mla_attn_decode params[5] = 3, NSV = its KV splits): attn of this head combined here from the
// splits' fp32 partials and (m, l) (part: [T][NSV][HEADS KV + 2 HEADS]), each token's once its active splits counted (scnt) - the
// SPREAD combine's order, so the same bits as split 0's combine; also stored to attn (the harness checks it)
template <int HEADS, int KV, int VD, int NPART, int TG, int NSV = 0>
__device__ __forceinline__ void mla_vgate_task(KernelLocals &L,
                                               __hip_bfloat16 const *attn,  // [T][HEADS x KV]
                                               __hip_bfloat16 const *w_vc,  // [HEADS][VD][KV]
                                               float const *gparts,         // [NPART][T][HEADS x VD]
                                               __hip_bfloat16 *o_in,        // [T][HEADS x VD]
                                               int h,
                                               int t0,                      // the group's first token
                                               float const *part = nullptr,
                                               long long const *seq_lens = nullptr,
                                               uint32_t const *scnt = nullptr) {
  static_assert(KV == 512 && VD == 128 && TG <= 16 && T % TG == 0, "mla_vgate: K3's shapes (thread maps below)");
  int const tid = task_tid_all(), lane = tid & 63, wave = tid >> 6, tok = lane % 16, grp = lane / 16;
  char *const sm = L.sm;
  // LDS: w_vc^T [VD][KV] bf16 (16-B chunk c of row d at chunk c ^ (d % 16)), attn
  // [T][KV] bf16, the gate [T][VD] fp32
  __hip_bfloat16 *const wv = reinterpret_cast<__hip_bfloat16 *>(sm);
  __hip_bfloat16 *const at = reinterpret_cast<__hip_bfloat16 *>(sm + KV * VD * 2);
  float *const sg = reinterpret_cast<float *>(at + TG * KV);
  __syncthreads();
  {
    // w_vc^T into LDS ([VD][KV] in the framework's layout: row d, 64 chunks of 16
    // B; chunk c of row d at chunk c ^ (d % 16)) by global -> LDS copies, one
    // row per wave instruction (lane l: chunk l ^ (d % 16)), every one issued now:
    // they land while the gate and attn are polled (MI355X 10-13: through
    // registers, 4 loads at a time, ~8 round trips, done after attn landed)
    static_assert(KV / 8 == 64, "mla_vgate: one row of w_vc per wave copy");
    for (int d = wave; d < VD; d += 4) {
      vgate_lds_dma16(w_vc + ((size_t)h * VD + d) * KV + 8 * (lane ^ (d & 15)), wv + d * KV);
    }
  }
  // the gate: item v -> token v / 32, columns 4 (v % 32) .. + 3
  for (int v = tid; v < TG * 32; v += 256) {
    int const t = v >> 5, c = 4 * (v & 31);
    float4 zp[NPART];
    bool all;
    do {
#pragma unroll
      for (int p = 0; p < NPART; p++) {
        zp[p] = as_float4(ld16_at<AUX_DEV>(gparts, (uint32_t)((((size_t)p * T + t0 + t) * HEADS * VD + h * VD + c) * 4))); // (all in flight)
      }
      all = true;
#pragma unroll
      for (int p = 0; p < NPART; p++) {
        all = all && validf4(zp[p]);
      }
      if (!all) {
        tail_pause();
      }
    } while (!all);
    float z[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int p = 0; p < NPART; p++) {
      z[0] += zp[p].x;
      z[1] += zp[p].y;
      z[2] += zp[p].z;
      z[3] += zp[p].w;
    }
#pragma unroll
    for (int e = 0; e < 4; e++) {
      float const gv = round_bf16(z[e]);
      sg[t * VD + c + e] = round_bf16(1.f / (1.f + __expf(-gv)));
    }
  }
  if constexpr (NSV > 0) {
    auto act_of = [&](int x) {
      int const sk = (int)seq_lens[x], nch = (sk + mla_a::CH - 1) / mla_a::CH;
      return NSV < nch ? NSV : nch;
    };
    if (tid < TG) {
      cnt_wait_in(scnt + t0 + tid, (uint32_t)act_of(t0 + tid));
    }
    __syncthreads();
    constexpr int PW = HEADS * KV + 2 * HEADS;
    __amdgpu_buffer_rsrc_t const rp = buf_rsrc(const_cast<float *>(part));
    for (int v = tid; v < TG * KV / 4; v += 256) { // token v / (KV / 4), columns 4 (v % (KV / 4)) .. + 3 of head h
      int const t = v / (KV / 4), c = 4 * (v % (KV / 4)), x = t0 + t, act = act_of(x);
      v2u mlw[NSV];
      uint4 pv[NSV];
#pragma unroll
      for (int q = 0; q < NSV; q++) {
        if (q < act) {
          mlw[q] = __builtin_bit_cast(v2u, __builtin_amdgcn_raw_buffer_load_b64(
                                               rp, (uint32_t)((((size_t)x * NSV + q) * PW + HEADS * KV + 2 * h) * 4), 0, AUX_DEV));
          pv[q] = ld16_at<AUX_DEV>(part, (uint32_t)((((size_t)x * NSV + q) * PW + h * KV + c) * 4));
        }
      }
      float M = __uint_as_float(mlw[0].x);
#pragma unroll
      for (int q = 1; q < NSV; q++) {
        if (q < act) {
          M = fmaxf(M, __uint_as_float(mlw[q].x));
        }
      }
      float const m0 = __uint_as_float(mlw[0].x), f0 = m0 > -1.0e38f ? __expf(m0 - M) : 0.f;
      float l = __uint_as_float(mlw[0].y) * f0;
      float o0 = __uint_as_float(pv[0].x) * f0, o1 = __uint_as_float(pv[0].y) * f0, o2 = __uint_as_float(pv[0].z) * f0,
            o3 = __uint_as_float(pv[0].w) * f0;
#pragma unroll
      for (int q = 1; q < NSV; q++) {
        if (q < act) {
          float const mq = __uint_as_float(mlw[q].x), f = mq > -1.0e38f ? __expf(mq - M) : 0.f;
          l += __uint_as_float(mlw[q].y) * f;
          o0 = fmaf(__uint_as_float(pv[q].x), f, o0);
          o1 = fmaf(__uint_as_float(pv[q].y), f, o1);
          o2 = fmaf(__uint_as_float(pv[q].z), f, o2);
          o3 = fmaf(__uint_as_float(pv[q].w), f, o3);
        }
      }
      float const inv = 1.f / l;
      uint2 const w = make_uint2(pack_bf16x2(o0 * inv, o1 * inv), pack_bf16x2(o2 * inv, o3 * inv));
      *reinterpret_cast<uint2 *>(at + t * KV + c) = w;
      asm volatile("global_store_dwordx2 %0, %1, off sc1\n s_nop 1" ::"v"(const_cast<__hip_bfloat16 *>(attn) + (size_t)x * HEADS * KV + h * KV + c),
                   "v"(w) : "memory");
    }
  } else {
  // attn of this head (polled): TG x KV / 8 vectors, rounds of at most 2 per thread in flight together (T = 8: one)
  constexpr int NAV = TG * KV / 8, APER = (NAV + 255) / 256, AR = APER < 2 ? APER : 2;
#pragma unroll 1
  for (int i0 = 0; i0 < APER; i0 += AR) {
    uint4 av[AR];
    for (;;) {
      bool all = true;
#pragma unroll
      for (int i = 0; i < AR; i++) {
        int const v = tid + 256 * (i0 + i), t = v / (KV / 8), c = 8 * (v % (KV / 8));
        if (v < NAV) {
          av[i] = ld16_at<AUX_DEV>(attn, (uint32_t)(((size_t)(t0 + t) * HEADS * KV + h * KV + c) * 2));
          all = all && valid16w(av[i]);
        }
      }
      if (all) {
        break;
      }
      tail_pause();
    }
#pragma unroll
    for (int i = 0; i < AR; i++) {
      int const v = tid + 256 * (i0 + i), t = v / (KV / 8), c = 8 * (v % (KV / 8));
      if (v < NAV) {
        *reinterpret_cast<uint4 *>(at + t * KV + c) = av[i];
      }
    }
  }
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); // this wave's w_vc copies landed (before the barrier)
  __syncthreads();
  // v[t][d] = bf16(sum_l attn[t][l] w_vc[l][d]) by MFMA 16x16x32 bf16: wave w
  // takes rows d of tiles 2 w, 2 w + 1 (A: w_vc^T rows, K = l; B: the tokens'
  // attn; D: lane -> token l % 16, register r -> row 4 (l / 16) + r); o_in =
  // bf16(v * sigmoid gate)
  {
    v4f acc[2] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll 4
    for (int st = 0; st < KV / 32; st++) {
      v4f b = {0.f, 0.f, 0.f, 0.f};
      if (tok < TG) {
        b = *reinterpret_cast<v4f const *>(at + tok * KV + 32 * st + 8 * grp);
      }
#pragma unroll
      for (int i = 0; i < 2; i++) {
        int const d = 16 * (2 * wave + i) + tok, ch = (4 * st + grp) ^ (d & 15);
        v4f const a = *reinterpret_cast<v4f const *>(wv + d * KV + 8 * ch);
        acc[i] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, a), __builtin_bit_cast(v8bf, b),
                                                         acc[i], 0, 0, 0);
      }
    }
    __syncthreads(); // every wave's reads of attn (at) done: o_in's rows staged there
    uint2 *const ob = reinterpret_cast<uint2 *>(at); // [TG][VD] bf16
    if (tok < TG) {
#pragma unroll
      for (int i = 0; i < 2; i++) {
        int const d0 = 16 * (2 * wave + i) + 4 * grp;
        float o[4];
#pragma unroll
        for (int r = 0; r < 4; r++) {
          o[r] = round_bf16(acc[i][r]) * sg[tok * VD + d0 + r];
        }
        ob[(tok * VD + d0) / 4] = make_uint2(pack_bf16x2(o[0], o[1]), pack_bf16x2(o[2], o[3]));
      }
    }
    __syncthreads();
    for (int v = tid; v < TG * VD / 8; v += 256) { // token v / 16, columns 8 (v % 16) .. + 7: each token's 256 B of this
      // head in 16 16-B stores
      int const t = v / (VD / 8), c = 8 * (v % (VD / 8));
      st16_dev(o_in + (size_t)(t0 + t) * HEADS * VD + h * VD + c, *reinterpret_cast<uint4 const *>(ob + (t * VD + c) / 4));
    }
  }
  __syncthreads();
}

// task (x, y) = head x, token group y; A = mla_attn (its buffer), GP = g_proj (a gemm_tile's K parts
// in slots); params = [HEADS, KV, VD]; buffer slots = w_vc
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class A,
          class GP,
          class... IN>
__device__ __forceinline__ void run_mla_vgate(Maps const &,
                                              G const &g,
                                              KernelLocals &L,
                                              StaticTask const &tk) {
  static_assert(A::buf >= 0, "mla_vgate: mla_attn's (or mla_attn_decode's) output");
  static_assert(GP::buf >= 0 && GP::v[0] == COMBINE_SLOTS, "mla_vgate: g_proj's K parts in slots");
  static_assert(SELF::buf >= 0 && SELF::x == PARAMS::v[0], "mla_vgate: one task per head, its output");
  constexpr int NSV = []() constexpr { // (params[3]: MPK_MLA_COMB_VGATE, the attention's KV splits; buffer slots 1, 2 = its
    // seq_lens and partials)
    if constexpr (sizeof(PARAMS::v) / sizeof(int) > 3) {
      return PARAMS::v[3];
    } else {
      return 0;
    }
  }();
  float const *part = nullptr;
  long long const *seq = nullptr;
  uint32_t const *scnt = nullptr;
  if constexpr (NSV > 0) {
    static_assert(A::counter >= 0, "mla_vgate NSV: the attention's per-request counters");
    seq = buf_at<long long const>(g, slot_at<BUF_SLOTS, 1>());
    part = buf_at<float const>(g, slot_at<BUF_SLOTS, 2>());
    scnt = cnt_of(g) + A::counter;
  }
  mla_vgate_task<PARAMS::v[0], PARAMS::v[1], PARAMS::v[2], GP::y, T / SELF::y, NSV>(
      L,
      buf_at<__hip_bfloat16 const>(g, A::buf),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<float const>(g, GP::buf),
      buf_at<__hip_bfloat16>(g, SELF::buf),
      tk.x,
      tk.y * (T / SELF::y),
      part,
      seq,
      scnt);
  if constexpr (SELF::counter >= 0) { // (mla_vgate_layer counted: o_proj waits for the count)
    task_post(cnt_of(g) + SELF::counter, 1);
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_mla_vgate() {
  return PARAMS::v[1] * PARAMS::v[2] * 2 + (T / SELF::y) * PARAMS::v[1] * 2 + (T / SELF::y) * PARAMS::v[2] * 4;
}

} // namespace static_mk
