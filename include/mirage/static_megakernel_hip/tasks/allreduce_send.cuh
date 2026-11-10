// tasks/allreduce_send.cuh (HIP) -- as include/mirage/static_megakernel/tasks/
// allreduce_send.cuh: the sending half of an all-reduce of a [T][W] partial
// sum over the GPUs: task x of grid (NSLICE, 1, 1) sends slice x of this GPU's
// partial sum (after every task of the node that writes it: its counter) as
// bf16 into slot `rank` of the node's output on every GPU (an exchange buffer,
// [GPUS][T][W] bf16): one system-scope store per GPU per 16 B (push16; CUDA:
// one multicast store). The adding half: sum_gpus, sum_rmsnorm
// (allreduce_land_bf16).
#pragma once
#include "../core.cuh"

namespace static_mk {

#ifndef STATIC_RBLK_PAR
#define STATIC_RBLK_PAR 0
#endif
// STATIC_MOE_R_RS 1 (default 0; the MoE tail not folded: sum_rmsnorm after R): R reduce-scattered by token - the R send pushes
// token t's row only to GPU t / (T / GPUS) (its owner) instead of every GPU, 1/GPUS of the bytes per link (MI355X T 64: 458 KB per
// link, ~13 us from the experts' end to R sent); sum_rmsnorm.cuh then sends the owner's Rn row to the other GPUs
#ifndef STATIC_MOE_R_RS
#define STATIC_MOE_R_RS 0
#endif
// CM (the LDS hand-off, core.cuh ARS_HAND_ON): slice x = columns [x W / NSLICE, ..) of every token instead of a run of the
// token-major vectors, so a slice reads one 128-column block of glm_moe_experts' R and goes once that block is counted

// src = the producer's output, fp32 [T x GROUP][W], complete once its NDONE
// tasks are counted in done_cnt (the layer's hand-off protocol: cnt_wait_in,
// ld16_in): task x of NSLICE takes slice x of the T x W / 8 bf16 vectors (token
// t: its GROUP rows added in order) and sends each into slot g.rank of the
// output (OFF) of every GPU. RS >= 0: re-arm slice x of exchange buffer RS.
// (MI355X 10-11: polling R itself, 0xFF-prefilled, instead of the counter, so a
// slice could go as soon as its rows were written, was slower: solo e2e 101 ->
// 108 us, the 256 slices' polls loaded R again and again during moe's tail)
// RBLK (R in blocks, the producer moe_experts): instead of all NDONE
// tasks, the slice waits for the 128-column blocks it reads (rblk: moe's block
// counters, core.cuh MOE_MISC_RBLK), each complete at n_live x 8 tiles.
template <int NSLICE, int NDONE, int GROUP, int W, size_t OFF, int RS,
          bool RBLK = false, bool CM = false, bool HAND = false, bool PM = false>
__device__ __forceinline__ void allreduce_send_task(G const &g,
                                                    float const *src,
                                                    uint32_t *done_cnt,
                                                    uint32_t const *rblk,
                                                    uint32_t *self_cnt, // (the landing: this node's counter)
                                                    int x,
                                                    char *lds_hand, // (the LDS hand-off: the CU's LDS)
                                                    __hip_bfloat16 const *pf_res, // (prefetch, or null: hc_post_mix's residual [T][4][W])
                                                    float const *pf_fn STAGE_STAMP_PARAM) { // (and its fn [24][4][W])
  if (pf_res != nullptr) { // (ARS prefetch: hc_post_mix task x's lines into this XCD's L2 while the experts run; it runs next here)
    constexpr int HCn = 4, NM = 24;
    int const c0 = 32 * x, tid = (int)STATIC_TIDX;
    unsigned acc = 0;
    for (int q = tid; q < T * HCn + NM * HCn; q += 256) {
      unsigned const *a = q < T * HCn ? reinterpret_cast<unsigned const *>(pf_res + ((size_t)q * W + c0))
                                      : reinterpret_cast<unsigned const *>(pf_fn + ((size_t)(q - T * HCn) * W + c0));
      acc ^= *a; // (a plain load: the line kept in the L2)
    }
    asm volatile("" ::"v"(acc));
  }
#if STATIC_RBLK_PAR
  // (STATIC_RBLK_PAR: wave 0, lane j polls block j's counter if the slice reads it, all blocks' loads in flight together;
  // the default: thread 0 polls the blocks one after another, ~1 us per device-scope load, up to 14 per slice at T = 64)
  if constexpr (RBLK) {
    constexpr int C8_ = W / 8, NB_ = T * C8_, NBLK_ = C8_ / 16;
    static_assert(NBLK_ <= 64, "STATIC_RBLK_PAR: a lane per 128-column block");
    if (STATIC_TIDX < 64) {
      int const lo_ = (int)((long long)NB_ * x / NSLICE),
                hi_ = (int)((long long)NB_ * (x + 1) / NSLICE);
      int const j = STATIC_TIDX, t0_ = lo_ / C8_, t1_ = (hi_ - 1) / C8_;
      // block j's vectors in token row t: [t C8 + 16 j, t C8 + 16 j + 16)
      auto in_row = [&](int t) { return t * C8_ + 16 * j < hi_ && t * C8_ + 16 * j + 16 > lo_; };
      bool const need = j < NBLK_ && hi_ > lo_ && (t1_ - t0_ >= 2 || in_row(t0_) || in_row(t1_));
      uint32_t nl;
      while ((nl = cnt_ld(rblk)) == 0) {
        __builtin_amdgcn_s_sleep(1);
      }
      bool ok = !need;
      while (!__all(ok)) {
        if (!ok) {
          ok = cnt_ld(rblk + RBLK_STRIDE * (1 + j)) >= nl * 8;
        }
        if (!__all(ok)) {
          __builtin_amdgcn_s_sleep(4);
        }
      }
      asm volatile("" ::: "memory");
    }
  } else if (STATIC_TIDX == 0) {
    cnt_wait_in(done_cnt, NDONE);
  }
  if (STATIC_TIDX == 0) {
    STAGE_STAMP(true)
  }
#else
  if (STATIC_TIDX == 0) {
    if constexpr (RBLK) {
      constexpr int C8_ = W / 8, NB_ = T * C8_;
      int const lo_ = (int)((long long)NB_ * x / NSLICE),
                hi_ = (int)((long long)NB_ * (x + 1) / NSLICE);
      uint32_t nl;
      while ((nl = cnt_ld(rblk)) == 0) {
        __builtin_amdgcn_s_sleep(1);
      }
      int prev = -1;
      constexpr int VPS_ = C8_ / NSLICE; // (CM: vectors per token per slice)
      for (int i = CM ? x * VPS_ : lo_; i < (CM ? (x + 1) * VPS_ : hi_); i++) {
        int const b = (i % C8_) / 16;
        if (b != prev) {
          while (cnt_ld(rblk + RBLK_STRIDE * (1 + b)) < nl * 8) {
            __builtin_amdgcn_s_sleep(4); // (a poll every ~0.3 us: these lines
          }                              // take every W2 unit's add)
          asm volatile("" ::: "memory");
          prev = b;
        }
      }
    } else {
      cnt_wait_in(done_cnt, NDONE);
    }
    STAGE_STAMP(true)
  }
#endif
  __syncthreads();
  if (STATIC_TIDX == 0) {
    STAGE_STAMP_AT(1)
  }
  constexpr int C8 = W / 8; // 16-B bf16 vectors per token
  constexpr int VPS = C8 / NSLICE; // (CM: vectors per token per slice)
  static_assert(!CM || VPS * NSLICE == C8, "allreduce_send CM: whole column slices");
  int const nb = T * C8, blo = (int)((long long)nb * x / NSLICE),
            bhi = (int)((long long)nb * (x + 1) / NSLICE);
  auto vec_of = [&](int li) { // (the slice's li-th vector: token-major run, or CM: token li / VPS, columns x VPS + li % VPS)
    return CM ? (li / VPS) * C8 + x * VPS + li % VPS : blo + li;
  };
  int const nsl = CM ? T * VPS : bhi - blo;
  uint4 own_v = make_uint4(0u, 0u, 0u, 0u); // (this thread's own partial, kept for the landing: at most one vector a thread)
  static_assert((T * (W / 8) + NSLICE - 1) / NSLICE <= 256, "allreduce_send: one own vector a thread");
  for (int li = STATIC_TIDX; li < nsl; li += 256) {
    int const i = vec_of(li);
    int const t = i / C8, c = i - t * C8;
    size_t const dst =
        OFF + (size_t)g.rank * ((size_t)T * W * 2) + ((size_t)t * C8 + c) * 16;
    // (PM: a store gemm's K parts, part-major [GROUP][T][W]; else GROUP rows per token, [T][GROUP][W])
    uint32_t const rows = (uint32_t)(((size_t)t * (PM ? 1 : GROUP) * W + 8 * c) * 4);
    constexpr uint32_t KSTR = (uint32_t)((PM ? (size_t)T * W : (size_t)W) * 4);
    float4 p0 = make_float4(0.f, 0.f, 0.f, 0.f), p1 = p0;
#pragma unroll
    for (int k = 0; k < GROUP; k++) { // the rows of token t in order k = 0 ..
      float4 const a = as_float4(ld16_in_at(src, rows + (uint32_t)k * KSTR)),
                   b = as_float4(
                       ld16_in_at(src, rows + (uint32_t)k * KSTR + 16));
      p0.x += a.x;
      p0.y += a.y;
      p0.z += a.z;
      p0.w += a.w;
      p1.x += b.x;
      p1.y += b.y;
      p1.z += b.z;
      p1.w += b.w;
    }
#if STATIC_MOE_R_RS
    if constexpr (RBLK) {
      static_assert(T % GPUS == 0, "STATIC_MOE_R_RS: whole tokens per GPU");
      int const ow = t / (T / GPUS);
      unsigned char *pr = g.peer[0];
#pragma unroll
      for (int q = 1; q < GPUS; q++) {
        pr = ow == q ? g.peer[q] : pr;
      }
      st16_sys((SOLO ? g.rv : pr) + dst + rg_set(), pack_bf16x8(p0, p1));
      continue;
    }
#endif
    if constexpr (!SOLO) {
      if (T >= STATIC_ARS_LAND_MIN_T) { // (the other GPUs only; the own partial kept for the landing below)
        uint4 const pv = pack_bf16x8(p0, p1);
        own_v = pv;
#pragma unroll
        for (int r = 0; r < GPUS; r++) {
          if (r != g.rank) {
            st16_sys(g.peer[r] + dst + rg_set(), pv);
          }
        }
        continue;
      }
    }
    push16(g, dst, pack_bf16x8(p0, p1));
  }
  // (the landing, T >= STATIC_ARS_LAND_MIN_T: the same threads poll this slice's GPUS slots (as amd/bench/hop_big.hip's senders do), add them in GPU
  // order (fp32), round to bf16 and write the sum into this GPU's own slot (only this GPU reads it); then the task counts
  // itself: hc_post_mix loads the sums from the local slot after the count. MI355X 11-08 glmmoe78: hc_post_mix's own wait for
  // the pushed sums cost ~27 / 24 us of the T 64 / 32 MoE kernel (normal build), the bare hop 10 / 4.6 us)
  if constexpr (!SOLO && T >= STATIC_ARS_LAND_MIN_T) {
    for (int li = STATIC_TIDX; li < nsl; li += 256) {
      int const i = vec_of(li);
      float sm8[8];
      // (the others' slots polled, the own partial from registers, added in GPU order)
      constexpr size_t RK = (size_t)T * W * 2;
      unsigned char const *const row = g.rv + OFF + rg_set();
      uint4 u[GPUS];
      for (;;) {
        bool all = true;
#pragma unroll
        for (int r = 0; r < GPUS; r++) {
          u[r] = r == g.rank ? own_v : ld16_at<AUX_SYS>(row, (uint32_t)(r * RK + (size_t)i * 16));
          all = all && valid16w(u[r]);
        }
        if (all) {
          break;
        }
        __builtin_amdgcn_s_sleep(1);
        asm volatile("" ::: "memory");
      }
#pragma unroll
      for (int k = 0; k < 8; k++) {
        sm8[k] = 0.f;
      }
#pragma unroll
      for (int r = 0; r < GPUS; r++) {
        uint32_t const wd[4] = {u[r].x, u[r].y, u[r].z, u[r].w};
#pragma unroll
        for (int k = 0; k < 4; k++) {
          sm8[2 * k] += bf16_lo(wd[k]);
          sm8[2 * k + 1] += bf16_hi(wd[k]);
        }
      }
      uint4 const sv = make_uint4(pack_bf16x2(sm8[0], sm8[1]), pack_bf16x2(sm8[2], sm8[3]), pack_bf16x2(sm8[4], sm8[5]),
                                  pack_bf16x2(sm8[6], sm8[7]));
      if constexpr (HAND) { // (for hc_post_mix task x, next on this CU: [T][VPS] vectors in LDS)
        *reinterpret_cast<uint4 *>(lds_hand + ARS_HAND_OFF + (size_t)li * 16) = sv;
      } else {
        st16_dev(g.rv + OFF + (size_t)g.rank * ((size_t)T * W * 2) + (size_t)i * 16 + rg_set(), sv);
      }
    }
    if constexpr (HAND) {
      __syncthreads();
    }
    task_post(self_cnt, 1);
  }
#if defined(STATIC_TIMING_BUILD) && defined(STATIC_ARS_ACK_STAMP)
  // (STATIC_ARS_ACK_STAMP, the timing build: stage stamp 1 = this task's pushes acknowledged by every GPU, each thread's stores
  // waited for: a diagnosis of when R / S land vs when the receivers see them)
  wait_mem();
  __syncthreads();
  if (STATIC_TIDX == 0) {
    STAGE_STAMP_AT(1)
  }
#endif
  __syncthreads();
}

template <class PARAMS, int I>
constexpr int ars_param_or0() {
  if constexpr (sizeof(PARAMS::v) / sizeof(int) > I + 1) {
    return PARAMS::v[I];
  } else {
    return 0;
  }
}
template <bool RBLK, class SLOTS, size_t ROFF = MOE_MISC_RBLK>
__device__ __forceinline__ uint32_t const *ars_rblk(G const &g) {
  if constexpr (RBLK) { // (ROFF 0: glm_moe_experts' own blocks buffer)
    return reinterpret_cast<uint32_t const *>(
        buf_at<unsigned char>(g, slot_at<SLOTS, 1>()) + ROFF);
  } else {
    return nullptr;
  }
}

// (params[3] 1: the prefetch tensors in slots 2 / 3, else null - if constexpr: no buf_at of a missing slot)
template <class PARAMS, class BUF_SLOTS, int I, class P>
__device__ __forceinline__ P *ars_pf_ptr(G const &g) {
  if constexpr (ars_param_or0<PARAMS, 3>() == 1) {
    return buf_at<P>(g, slot_at<BUF_SLOTS, I>());
  } else {
    return nullptr;
  }
}
// allreduce_send task x: slice x of SELF::x of its input (the output of node
// SRC, complete once all SRC's tasks are counted) to every GPU. PARAMS {W (the
// row width), rows per token in the input (GROUP: added in order)}; BUF_SLOTS
// {an exchange buffer to re-arm once SRC is done, or -1}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SRC,
          class... REST>
__device__ __forceinline__ void run_allreduce_send(Maps const &,
                                                   G const &g,
                                                   KernelLocals &L,
                                                   StaticTask const &tk) {
  constexpr int W = PARAMS::v[0], GROUP = PARAMS::v[1];
  // params[2] 3: a store gemm's K parts (params[1] of them, part-major); params[2] 1: the producer is moe_experts, BUF_SLOTS[1] its misc scratch
  // (R in blocks: wait for the slice's blocks)
  constexpr bool RBLK = ars_param_or0<PARAMS, 2>() == 1 || ars_param_or0<PARAMS, 2>() == 2; // (2: glm_moe_experts' blocks)
  static_assert(!RBLK || W % 128 == 0, "R_BLOCKS: whole 128-column blocks");
  constexpr bool HAND = ARS_HAND_ON((ars_param_or0<PARAMS, 2>()), SELF::x, W); // (the LDS hand-off: column-major slices)
  static_assert(W % 8 == 0 && GROUP >= 1,
                "allreduce_send: params {row width, rows per token}");
  static_assert(SRC::buf >= 0 && SRC::counter >= 0,
                "allreduce_send: its input is a node's output buffer, the node "
                "counts its tasks");
  static_assert(
      SELF::buf >= 0 && exchange_bytes[SELF::buf] == (size_t)GPUS * T * W * 2,
      "allreduce_send: its output, [GPUS][T][W] bf16 in the exchange region");
  allreduce_send_task<SELF::x,
                      SRC::x * SRC::y * SRC::z,
                      GROUP,
                      W,
                      exchange_offset[SELF::buf],
                      slot_at<BUF_SLOTS, 0>(),
                      RBLK,
                      HAND,
                      HAND, ars_param_or0<PARAMS, 2>() == 3>(
      g,
      buf_at<float const>(g, SRC::buf),
      cnt_of(g) + SRC::counter,
      ars_rblk<RBLK, BUF_SLOTS, ars_param_or0<PARAMS, 2>() == 2 ? 0 : MOE_MISC_RBLK>(g),
      SELF::counter >= 0 ? cnt_of(g) + SELF::counter : nullptr,
      tk.x,
      L.sm,
      ars_pf_ptr<PARAMS, BUF_SLOTS, 2, __hip_bfloat16 const>(g),
      ars_pf_ptr<PARAMS, BUF_SLOTS, 3, float const>(g) STAGE_STAMP_ARG(L.stage_stamps));
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_allreduce_send() {
  return ARS_HAND_ON((ars_param_or0<PARAMS, 2>()), SELF::x, PARAMS::v[0]) ? ARS_HAND_OFF + T * PARAMS::v[0] / SELF::x * 2
                                                                           : 0; // (the LDS hand-off region)
}

} // namespace static_mk
