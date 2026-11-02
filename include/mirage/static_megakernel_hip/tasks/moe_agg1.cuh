// tasks/moe_agg1.cuh (HIP, gfx950) -- the end of a K3 MoE layer with the next
// layer's aggregation 1, one task per token (no exchange between CUs; the
// chunked version, tasks/moe_attn_residual.cuh, spent ~5.6 us in its exchange):
//   y   = bf16(bf16(o + S) + prefix)   (residual_add: o the latent_up rows of
//         the GPU that owns the columns, sum_send's layout in the exchange
//         region; S sum_gpus's buffer; both polled, 0xFF before each launch)
//   info[3] != 0: y also into bank row info[0] (SGLang's fused write)
//   c   = q * gain (fp32)
//   for src in the n bank rows (info[0]; strides info[1], info[2]) and y:
//         s = (c . src) / sqrt(mean(src^2) + eps)
//   w   = softmax(s);  h = sum w src (fp32)
//   out = bf16(h rsqrt(mean(h^2) + eps) ow)   (h's sum of squares from h, as
//         SGLang ROCm's attn_res_hip)
// Task x of grid (T, 1, 1) = token x; thread i: columns 4 i + 1024 j .. + 3,
// j < H / 1024. The bank rows' terms are computed before o and S land.
#pragma once
#include "attn_residual.cuh"
#include "sum_send.cuh" // the layout of its input o

namespace static_mk {

constexpr int AGG1_NBMAX = 8; // bank rows at most (SGLang's K3 bank: 8 rows per token)
// a barrier for LDS only: __syncthreads also waits for every store issued before it (vmcnt), and the y stores before
// the first exchange of sums took ~9 us to complete (MI355X 10-15, kda178 timing build: y computed 85.4 -> weights 94.6)
__device__ __forceinline__ void agg1_lds_bar() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}
// 8 bytes at device scope (POST: a node of this kernel reads out / y / the bank row; s_nop: the gfx950 hazard of a VALU
// write to the store's data right after an inline-asm store, runtime.cuh st16_dev)
__device__ __forceinline__ void agg1_st8_dev(void *p, uint2 v) {
  v2u const w = {v.x, v.y};
  asm volatile("global_store_dwordx2 %0, %1, off sc1\n s_nop 1" ::"v"(p), "v"(w) : "memory");
}

template <size_t OOFF, int H, bool POST = false, bool MIX = false>
__device__ __forceinline__ void moe_agg1_task(G const &g,
                                              KernelLocals &L,
                                              __hip_bfloat16 *bank,
                                              __hip_bfloat16 const *prefix,
                                              __hip_bfloat16 const *q,
                                              __hip_bfloat16 const *gain,
                                              __hip_bfloat16 const *ow,
                                              __hip_bfloat16 *y,
                                              int const *info,
                                              float eps,
                                              __hip_bfloat16 *out,
                                              __hip_bfloat16 const *ssum,
                                              int t STAGE_STAMP_PARAM,
                                              uint32_t const *pre_cnt = nullptr,
                                              uint32_t pre_need = 0,
                                              uint32_t *post_cnt = nullptr,
                                              __hip_bfloat16 *pfx = nullptr,
                                              __hip_bfloat16 *bare = nullptr,
                                              __hip_bfloat16 *mix = nullptr) {
  constexpr int NG = H / 1024, NV = 2 * AGG1_NBMAX + 2;
  static_assert(H % 1024 == 0 && (H / GPUS) % 4 == 0, "moe_agg1: 4 columns per thread per 1024");
  int const tid = task_tid_all(), lane = tid & 63, wave = tid >> 6;
  int const nb = info[0], wr = info[3];
  size_t const bs_tok = (size_t)info[1], bs_row = (size_t)info[2];
  float *const red = reinterpret_cast<float *>(L.sm); // [4 waves][NV], then [4] h's sums of squares
  // the bank rows in LDS ([row][j][thread], 8 B each: each thread reads back only its own; registers: the kernel spilled)
  uint2 *const bkl = reinterpret_cast<uint2 *>(L.sm + 1024);
  __syncthreads(); // the LDS may still be read by the previous task
  auto col = [&](int j) { return 4 * tid + 1024 * j; };
  // 0. what needs no input: prefix (pre_cnt: a node of this kernel writes it, input_wait: its count first), c, ow and the
  // bank rows
  if (pre_cnt != nullptr) {
    if (tid == 0) {
      cnt_wait_in(pre_cnt, pre_need);
    }
    __syncthreads();
  }
  uint2 pre2[NG], ow2[NG];
  float cc[NG][4];
#pragma unroll
  for (int j = 0; j < NG; j++) {
    pre2[j] = ar_ld8_at<AUX_DEV>(prefix, (uint32_t)(((size_t)t * H + col(j)) * 2)); // (device scope: another CU or kernel wrote it)
    ow2[j] = *reinterpret_cast<uint2 const *>(ow + col(j));
    uint2 const q2 = *reinterpret_cast<uint2 const *>(q + col(j)), g2 = *reinterpret_cast<uint2 const *>(gain + col(j));
    cc[j][0] = bf16_lo(q2.x) * bf16_lo(g2.x);
    cc[j][1] = bf16_hi(q2.x) * bf16_hi(g2.x);
    cc[j][2] = bf16_lo(q2.y) * bf16_lo(g2.y);
    cc[j][3] = bf16_hi(q2.y) * bf16_hi(g2.y);
  }
  auto at = [](uint2 v, int e) { return (e & 1) ? bf16_hi((&v.x)[e / 2]) : bf16_lo((&v.x)[e / 2]); };
  // the bank rows' c . b and b . b (each summed over the wave, lane 0 into red), each row into LDS
#if STATIC_AGG1_RB > 1
  // (STATIC_AGG1_RB rows' loads in flight together: one row at a time waited for each row's ~14 KB, up to 8 round trips in
  // series - at T 64 the task starts ~3 us before o lands and this loop was its ~15 us)
#pragma unroll 1
  for (int b0 = 0; b0 < nb; b0 += STATIC_AGG1_RB) {
    uint2 bk[STATIC_AGG1_RB][NG];
#pragma unroll
    for (int i = 0; i < STATIC_AGG1_RB; i++) {
      int const b = min(b0 + i, nb - 1); // (past the last row: the last row again, not used)
#pragma unroll
      for (int j = 0; j < NG; j++) {
        bk[i][j] = *reinterpret_cast<uint2 const *>(bank + t * bs_tok + b * bs_row + col(j));
      }
    }
#pragma unroll
    for (int i = 0; i < STATIC_AGG1_RB; i++) {
      int const b = b0 + i;
      if (b < nb) {
        float d = 0.f, s = 0.f;
#pragma unroll
        for (int j = 0; j < NG; j++) {
#pragma unroll
          for (int e = 0; e < 4; e++) {
            d = fmaf(cc[j][e], at(bk[i][j], e), d);
            s = fmaf(at(bk[i][j], e), at(bk[i][j], e), s);
          }
          bkl[(b * NG + j) * 256 + tid] = bk[i][j];
        }
        d = wave_sum64(d);
        s = wave_sum64(s);
        if (lane == 0) {
          red[wave * NV + 2 * b] = d;
          red[wave * NV + 2 * b + 1] = s;
        }
      }
    }
  }
#else
#pragma unroll 1
  for (int b = 0; b < nb; b++) {
    {
      uint2 bk[NG];
#pragma unroll
      for (int j = 0; j < NG; j++) {
        bk[j] = *reinterpret_cast<uint2 const *>(bank + t * bs_tok + b * bs_row + col(j));
      }
      float d = 0.f, s = 0.f;
#pragma unroll
      for (int j = 0; j < NG; j++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
          d = fmaf(cc[j][e], at(bk[j], e), d);
          s = fmaf(at(bk[j], e), at(bk[j], e), s);
        }
        bkl[(b * NG + j) * 256 + tid] = bk[j];
      }
      d = wave_sum64(d);
      s = wave_sum64(s);
      if (lane == 0) {
        red[wave * NV + 2 * b] = d;
        red[wave * NV + 2 * b + 1] = s;
      }
    }
  }
#endif
  // 1. o and S of this thread's columns (all loads in flight; re-loaded until written)
  uint2 ou[NG], su[NG];
  bool okv[NG];
#pragma unroll
  for (int j = 0; j < NG; j++) {
    okv[j] = false;
  }
  for (;;) {
    bool all = true;
#pragma unroll
    for (int j = 0; j < NG; j++) {
      if (!okv[j]) {
        int const c = col(j), r = c / (H / GPUS), cr = c - r * (H / GPUS);
        bool const skip_o = SOLO && r != g.rank; // (SOLO: another GPU's o: zeros)
        ou[j] = skip_o ? make_uint2(0u, 0u)
                       : ar_ld8_at<AUX_SYS>(g.rv, (uint32_t)(OOFF + (size_t)r * o_rank_bytes(H) +
                                                              (size_t)t * (H / GPUS) * 2 + cr * 2 + rg_set()));
        su[j] = ar_ld8_at<AUX_DEV>(ssum, (uint32_t)(((size_t)t * H + c) * 2));
      }
    }
#pragma unroll
    for (int j = 0; j < NG; j++) {
      okv[j] = okv[j] || (ou[j].x != 0xFFFFFFFFu && ou[j].y != 0xFFFFFFFFu && su[j].x != 0xFFFFFFFFu &&
                          su[j].y != 0xFFFFFFFFu);
      all = all && okv[j];
    }
    if (__all(all)) {
      break;
    }
    tail_pause();
  }
  float u[NG][4];
  uint2 y2[NG]; // y (bf16), stored after out
  uint2 bare2[POST ? NG : 1]; // (POST: bf16(o + S), stored after out's count)
#pragma unroll
  for (int j = 0; j < NG; j++) {
    if constexpr (POST) { // (a 10th slot: the MoE's output without the prefix, bf16(o + S): SGLang's addend where the
                          // next aggregation adds the residual itself)
      float b4[4];
#pragma unroll
      for (int e = 0; e < 4; e++) {
        b4[e] = round_bf16(at(ou[j], e) + at(su[j], e));
        u[j][e] = round_bf16(at(pre2[j], e) + b4[e]);
      }
      bare2[j] = make_uint2(pack_bf16x2(b4[0], b4[1]), pack_bf16x2(b4[2], b4[3]));
    } else {
#pragma unroll
      for (int e = 0; e < 4; e++) {
        u[j][e] = round_bf16(at(pre2[j], e) + round_bf16(at(ou[j], e) + at(su[j], e)));
      }
    }
    y2[j] = make_uint2(pack_bf16x2(u[j][0], u[j][1]), pack_bf16x2(u[j][2], u[j][3]));
  }
  // 2. c . y and y . y, then every source's sums over the 4 waves
  {
    float d = 0.f, s = 0.f;
#pragma unroll
    for (int j = 0; j < NG; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        d = fmaf(cc[j][e], u[j][e], d);
        s = fmaf(u[j][e], u[j][e], s);
      }
    d = wave_sum64(d);
    s = wave_sum64(s);
    if (lane == 0) {
      red[wave * NV + 2 * AGG1_NBMAX] = d;
      red[wave * NV + 2 * AGG1_NBMAX + 1] = s;
    }
  }
  agg1_lds_bar();
#if STATIC_STAMP_SET == 0 || STATIC_STAMP_SET == 5
  if (tid == 0) { // (timing build: stage stamp 0 = every wave's o and S landed, y computed)
    STAGE_STAMP_AT(0)
  }
#endif
  // the scores and softmax (every thread; source AGG1_NBMAX = y)
  float wts[AGG1_NBMAX + 1];
  {
    float mx = -3.0e38f;
#pragma unroll
    for (int s_ = 0; s_ <= AGG1_NBMAX; s_++) {
      bool const live = s_ == AGG1_NBMAX || s_ < nb;
      float const dt = red[2 * s_] + red[NV + 2 * s_] + red[2 * NV + 2 * s_] + red[3 * NV + 2 * s_],
                  sq = red[2 * s_ + 1] + red[NV + 2 * s_ + 1] + red[2 * NV + 2 * s_ + 1] + red[3 * NV + 2 * s_ + 1];
      wts[s_] = live ? dt * rsqrtf(sq / (float)H + eps) : -3.0e38f;
      mx = fmaxf(mx, wts[s_]);
    }
    float den = 0.f;
#pragma unroll
    for (int s_ = 0; s_ <= AGG1_NBMAX; s_++) {
      bool const live = s_ == AGG1_NBMAX || s_ < nb;
      wts[s_] = live ? __expf(wts[s_] - mx) : 0.f;
      den += wts[s_];
    }
    float const inv = 1.f / den;
#pragma unroll
    for (int s_ = 0; s_ <= AGG1_NBMAX; s_++) {
      wts[s_] *= inv;
    }
  }
  // 3. h (fp32; in u's place), its sum of squares over the token, out. Row by row, the row's 7 LDS reads in flight
  // together (per element, under the row's branch, each read waited alone: ~9 us)
#pragma unroll
  for (int j = 0; j < NG; j++)
#pragma unroll
    for (int e = 0; e < 4; e++) {
      u[j][e] *= wts[AGG1_NBMAX];
    }
#pragma unroll 1
  for (int b = 0; b < nb; b++) {
    float const wb = wts[0] * (b == 0) + wts[1] * (b == 1) + wts[2] * (b == 2) + wts[3] * (b == 3) +
                     wts[4] * (b == 4) + wts[5] * (b == 5) + wts[6] * (b == 6) + wts[7] * (b == 7);
    uint2 r[NG];
#pragma unroll
    for (int j = 0; j < NG; j++) {
      r[j] = bkl[(b * NG + j) * 256 + tid];
    }
#pragma unroll
    for (int j = 0; j < NG; j++)
#pragma unroll
      for (int e = 0; e < 4; e++) {
        u[j][e] = fmaf(wb, at(r[j], e), u[j][e]);
      }
  }
  float hs = 0.f;
#pragma unroll
  for (int j = 0; j < NG; j++)
#pragma unroll
    for (int e = 0; e < 4; e++) {
      hs = fmaf(u[j][e], u[j][e], hs);
    }
  hs = wave_sum64(hs);
  if (lane == 0) {
    red[4 * NV + wave] = hs;
  }
  agg1_lds_bar();
  float const rstd = rsqrtf((red[4 * NV] + red[4 * NV + 1] + red[4 * NV + 2] + red[4 * NV + 3]) / (float)H + eps);
#if STATIC_STAMP_SET == 0 || STATIC_STAMP_SET == 5
  if (tid == 0) { // (stage stamp 1: the weights and h's RMS known)
    STAGE_STAMP_AT(1)
  }
#endif
#pragma unroll
  for (int j = 0; j < NG; j++) {
    float o4[4];
#pragma unroll
    for (int e = 0; e < 4; e++) {
      o4[e] = u[j][e] * rstd * at(ow2[j], e);
    }
    if constexpr (POST) {
      agg1_st8_dev(out + (size_t)t * H + col(j), make_uint2(pack_bf16x2(o4[0], o4[1]), pack_bf16x2(o4[2], o4[3])));
    } else {
      *reinterpret_cast<uint2 *>(out + (size_t)t * H + col(j)) =
          make_uint2(pack_bf16x2(o4[0], o4[1]), pack_bf16x2(o4[2], o4[3]));
    }
  }
  if constexpr (POST) { // out complete: count 0 (the next attention's x); then the rest, count 1
    task_post(post_cnt, 1);
  }
#pragma unroll
  for (int j = 0; j < NG; j++) {
    if constexpr (POST) {
      if (bare != nullptr) {
        agg1_st8_dev(bare + (size_t)t * H + col(j), bare2[j]);
      }
      agg1_st8_dev(y + (size_t)t * H + col(j), y2[j]);
      if (wr != 0) {
        agg1_st8_dev(bank + t * bs_tok + nb * bs_row + col(j), y2[j]);
      }
      if (pfx != nullptr) { // the next attention's prefix: y, or zero after a block write (y is then the bank row)
        agg1_st8_dev(pfx + (size_t)t * H + col(j), wr != 0 ? make_uint2(0u, 0u) : y2[j]);
      }
    } else {
      *reinterpret_cast<uint2 *>(y + (size_t)t * H + col(j)) = y2[j];
      if (wr != 0) { // the block-write layer's bank row nb
        *reinterpret_cast<uint2 *>(bank + t * bs_tok + nb * bs_row + col(j)) = y2[j];
      }
    }
  }
  if constexpr (POST) { // (y, the bank row, agg_prefix, agg_bare complete: count 1, attn_residual's wait)
    if constexpr (MIX) { // (an 11th slot: the pre-norm mixture h in bf16, SGLang's aggregate_stream for its dspark capture;
                          // no reader in this kernel; before count 1, whose wait covers them: stores left after a task's
                          // end delay the CU's next task's loads, vmcnt counting in order; rows info[5] apart when set:
                          // a column slice of SGLang's packed capture buffer)
      size_t const ms = info[5] > 0 ? (size_t)info[5] : (size_t)H;
#pragma unroll
      for (int j = 0; j < NG; j++) {
        *reinterpret_cast<uint2 *>(mix + (size_t)t * ms + col(j)) =
            make_uint2(pack_bf16x2(u[j][0], u[j][1]), pack_bf16x2(u[j][2], u[j][3]));
      }
    }
    task_post(post_cnt + 1, 1);
    return;
  }
  // (red is read by every thread above before the next task's writes; an LDS barrier: no wait for the stores above,
  // the kernel's end waits for them)
  __syncthreads();
}

// task x = token x; SSUM: the addend's node (sum_gpus), O: the input's node (o in
// the exchange region, sum_send's layout); params = [H, eps bits]; buffer slots
// = bank, prefix, q, gain, ow, y, info, out. A node with a counter (moe_agg1_layer
// posted=True: out is its output, read by a node of this kernel, e.g. the next
// layer's attention in a MoE + attention kernel): device-scope stores, then the
// task's count; a 9th slot: the next attention's prefix (y, or zero when the bank
// row was written)
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SSUM,
          class O,
          class RES,
          class... IN>
__device__ __forceinline__ void run_moe_agg1(Maps const &,
                                             G const &g,
                                             KernelLocals &L,
                                             StaticTask const &tk) {
  constexpr int H = PARAMS::v[0];
  static_assert(SSUM::buf >= 0 && O::buf >= 0 && exchange_bytes[O::buf] == (size_t)GPUS * o_rank_bytes(H),
                "moe_agg1: S from sum_gpus's buffer, o from sum_send's layout");
  static_assert(SELF::x == T && SELF::y == 1, "moe_agg1: one task per token");
  moe_agg1_task<exchange_offset[O::buf], H, (SELF::counter >= 0), (sizeof(BUF_SLOTS::v) / sizeof(int) > 11)>(
      g, L,
      buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 4>()),
      buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 5>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 6>()),
      __int_as_float(PARAMS::v[1]),
      buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 7>()),
      buf_at<__hip_bfloat16 const>(g, SSUM::buf),
      tk.x STAGE_STAMP_ARG(L.stage_stamps),
      RES::counter >= 0 ? cnt_of(g) + RES::counter : nullptr, // (RES: the prefix's node, e.g. input_wait)
      (uint32_t)(RES::x * RES::y * RES::z),
      SELF::counter >= 0 ? cnt_of(g) + SELF::counter : nullptr,
      sizeof(BUF_SLOTS::v) / sizeof(int) > 9 ? buf_at<__hip_bfloat16>(g, BUF_SLOTS::v[sizeof(BUF_SLOTS::v) / sizeof(int) > 9 ? 8 : 0])
                                              : nullptr,
      sizeof(BUF_SLOTS::v) / sizeof(int) > 10 ? buf_at<__hip_bfloat16>(g, BUF_SLOTS::v[sizeof(BUF_SLOTS::v) / sizeof(int) > 10 ? 9 : 0])
                                               : nullptr,
      sizeof(BUF_SLOTS::v) / sizeof(int) > 11 ? buf_at<__hip_bfloat16>(g, BUF_SLOTS::v[sizeof(BUF_SLOTS::v) / sizeof(int) > 11 ? 10 : 0])
                                               : nullptr);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_moe_agg1() {
  return 1024 + AGG1_NBMAX * (PARAMS::v[0] / 1024) * 256 * 8; // red (< 1 KB), the bank rows (8 x 7168 x 2 B: 112 KB)
}

} // namespace static_mk
