// tasks/act_quant.cuh -- DeepSeek-V4.1's act_quant (inference/kernel.py
// act_quant_kernel, block 32, scale_fmt "ue8m0": power-of-2 scales) of token
// t's row of x [T][K] bf16: per 32 columns amax = max(|x|, 1e-4), ex =
// ceil(log2(amax * fp32(1 / 448))), x_q = e4m3(x * 2^-ex) (= x / 2^ex exactly;
// satfinite = the clamp to +-448), its scale the e8m0 byte ex + 127. The
// output (the node's buffer, in this GPU's memory): x_q [T][K] e4m3, then the
// scale chunks [K / 128][512] gemm_fp8_tile reads: line t of chunk kb (bytes
// 16 t .. 16 t + 15) holds row t's 4 scale bytes in bytes 0..3, byte j the
// scale of columns 128 kb + 32 j .. + 31 (the MMA's scale id j,
// gemm_fp8_tile.cuh idesc_mx8); every other byte of the chunk 0. Task t writes
// row t of x_q and lines t, t + 8, t + 16, t + 24 of every chunk (the chunk's
// 32 lines over the T = 8 tasks), so each launch writes every byte; then the
// node's counter += 1 (its readers wait for T). K = the node's params[0]. x is
// a graph input (read by pointer), or a gemm_fp8_tile node's K parts in slots
// (wo_a's output before wo_b: model.py L790, wo_b's linear quantizes its
// input): added in part order and rounded to bf16 (the GEMM's bf16 output)
// first.
#pragma once
#include "../core.cuh"

namespace static_mk {

// x rounded to bf16 (to nearest even), as a float: where the official's bf16
// tensor arithmetic rounds
__device__ __forceinline__ float bf16r(float x) {
  return __bfloat162float(__float2bfloat16_rn(x));
}

// the output's layout for x of width K: x_q bytes, then the scale chunks (its
// size: static_megakernel.py zq_bytes)
__host__ __device__ constexpr size_t xq_scales_at(int K) {
  return (size_t)T * K;
}

// the scale exponent ex of a 32-column group, amax its max |x|: amax / 2^ex <=
// 448 (e4m3 max); amax times fp32(1 / 448), as kernel.py. amax >= 1e-4: ex in
// [-22, 120]
__device__ __forceinline__ int xq_exponent(float amax) {
  return ceil_log2_bits(fmaxf(amax, 1.0e-4f) * (1.0f / 448.0f));
}
// v[0..3] x 2^-ex as e4m3 (exact before the rounding), v[0] in the low byte
__device__ __forceinline__ uint32_t xq_pack4(float const *v, int ex) {
  float const sc = __uint_as_float((uint32_t)(127 - ex) << 23); // 2^-ex
  return (uint32_t)cvt_e4m3_nv(v[0] * sc) |
         ((uint32_t)cvt_e4m3_nv(v[1] * sc) << 8) |
         ((uint32_t)cvt_e4m3_nv(v[2] * sc) << 16) |
         ((uint32_t)cvt_e4m3_nv(v[3] * sc) << 24);
}
// a 128-column block's 4 scale bytes, from the LANES lanes that hold it (lanes
// LANES b .. LANES b + LANES - 1 of the warp, in column order; the first lane
// of each quarter holds group g's ex): byte g = ex of group g + 127. Every lane
// of the warp calls it. act_quant: 8 lanes of 16 columns; q_norm: 16 of 8
template <int LANES>
__device__ __forceinline__ uint32_t xq_scale_word(int ex) {
  static_assert(LANES == 8 || LANES == 16 || LANES == 32,
                "xq_scale_word: whole groups per lane quarter");
  int const base = (threadIdx.x & 31) & ~(LANES - 1);
  uint32_t s4 = 0;
#pragma unroll
  for (int g = 0; g < 4; g++) {
    s4 |= (uint32_t)(__shfl_sync(0xffffffffu, ex, base + g * (LANES / 4)) + 127)
          << (8 * g);
  }
  return s4;
}
// line t + 8 j (j < 4) of block kb's scale chunk in the output (out: x_q of
// width K, then the chunks): line t row t's 4 scale bytes s4, lines t + 8,
// t + 16, t + 24 zeros
static_assert(T == 8 && SF_CHUNK == 32 * 16,
              "xq_scale_line: the T = 8 tasks (one per token) write the "
              "chunk's 32 lines of 16 B, 4 each");
__device__ __forceinline__ void xq_scale_line(
    unsigned char *out, int K, int kb, int t, int j, uint32_t s4) {
  *reinterpret_cast<uint4 *>(out + xq_scales_at(K) + (size_t)kb * SF_CHUNK +
                             (t + 8 * j) * 16) =
      make_uint4(j == 0 ? s4 : 0u, 0u, 0u, 0u);
}

// columns c .. c + 7 of token t's row of a gemm_fp8_tile node's output in
// slots (fp32 [P][T][K], 0xFF before the launch: polled, all 2 P loads in
// flight at once), its P K parts added in part order into x (not rounded).
// dry: one round of the loads, not waited for (a dry pass: the code runs once)
// REARM (serving mode, STATIC_RESET_IN_KERNEL): 0xFF back into what add_parts8
// read (columns c .. c + 7 of row t of each part), once used: the one reader of
// those values re-arms them for the next launch's poll (config.cuh)
template <int K, int P>
__device__ __forceinline__ void rearm_parts8(float const *part, int t, int c) {
  if constexpr (REARM) {
#pragma unroll
    for (int p = 0; p < P; p++) {
      float const *src = part + ((size_t)p * T + t) * K + c;
      rearm16(src);
      rearm16(src + 4);
    }
  }
}
template <int K, int P>
__device__ __forceinline__ void add_parts8(
    float const *part, int t, int c, float (&x)[8], bool dry = false) {
  float4 v[P][2];
  bool ok[P];
#pragma unroll
  for (int p = 0; p < P; p++) {
    ok[p] = false;
  }
  bool all;
  do {
    all = true;
#pragma unroll
    for (int p = 0; p < P; p++) {
      if (!ok[p]) {
        float const *src = part + ((size_t)p * T + t) * K + c;
        v[p][0] = ldf4_poll(src);
        v[p][1] = ldf4_poll(src + 4);
      }
    }
#pragma unroll
    for (int p = 0; p < P; p++) {
      if (!ok[p]) {
        ok[p] = validf4(v[p][0]) && validf4(v[p][1]);
      }
      all = all && ok[p];
    }
    if (!all && !dry) {
      poll_pause(STATIC_POLL_NS);
    }
  } while (!all && !dry);
#pragma unroll
  for (int h = 0; h < 2; h++) {
    x[4 * h] = v[0][h].x;
    x[4 * h + 1] = v[0][h].y;
    x[4 * h + 2] = v[0][h].z;
    x[4 * h + 3] = v[0][h].w;
#pragma unroll
    for (int p = 1; p < P; p++) { // in part order
      x[4 * h] += v[p][h].x;
      x[4 * h + 1] += v[p][h].y;
      x[4 * h + 2] += v[p][h].z;
      x[4 * h + 3] += v[p][h].w;
    }
  }
}

template <int K, int NC>
__device__ __forceinline__ void act_quant_task(__nv_bfloat16 const *x,
                                               unsigned char *out,
                                               uint32_t *done_cnt,
                                               int t,
                                               int cy STAGE_STAMP_PARAM) {
  // columns [cy NC, (cy + 1) NC): thread i columns 16 i .. 16 i + 15 of each
  // pass of 4096; 2 threads = one 32-column group, 8 threads = one 128-column
  // block, j = the thread's place in the block. NC % 512 == 0: a warp is in a
  // pass whole. A loop (unroll 1): this code runs cold once per SM. timing
  // build: stamp [0] thread 0's first loads landed, [1] its stores issued
  static_assert(NC % 512 == 0 && K % NC == 0,
                "act_quant: whole warps of 128-column blocks");
  int const j = threadIdx.x & 7;
  for (int c = cy * NC + 16 * threadIdx.x; c < (cy + 1) * NC; c += 16 * 256) {
    uint4 const u[2] = {
        *reinterpret_cast<uint4 const *>(x + (size_t)t * K + c),
        *reinterpret_cast<uint4 const *>(x + (size_t)t * K + c + 8)};
    float v[16];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      __nv_bfloat162 const *b = reinterpret_cast<__nv_bfloat162 const *>(&u[h]);
#pragma unroll
      for (int k = 0; k < 4; k++) {
        float2 const f = __bfloat1622float2(b[k]);
        v[8 * h + 2 * k] = f.x;
        v[8 * h + 2 * k + 1] = f.y;
      }
    }
    if (threadIdx.x == 0 && c == cy * NC) {
      STAGE_STAMP_AT(0)
    }
    float amax = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; i++) {
      amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1)); // the group's
    int const ex = xq_exponent(amax);
    *reinterpret_cast<uint4 *>(out + (size_t)t * K + c) =
        make_uint4(xq_pack4(v, ex),
                   xq_pack4(v + 4, ex),
                   xq_pack4(v + 8, ex),
                   xq_pack4(v + 12, ex));
    uint32_t const s4 = xq_scale_word<8>(ex); // 8 threads = a 128-column block
    if (j < 4) {
      xq_scale_line(out, K, c / 128, t, j, s4);
    }
  }
  if (threadIdx.x == 0) {
    STAGE_STAMP_AT(1)
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt, 1); // release: the CTA's stores before (the barrier)
  }
}

// act_quant of columns [cy NC, (cy + 1) NC) of token t's row from a
// gemm_fp8_tile node's P K parts: thread i columns 8 i .. 8 i + 7 of each pass
// of 2048 (4 threads a 32-column group, 16 a 128-column block, as q_norm), the
// parts added in part order -> bf16 -> the same formula and layout as
// act_quant_task. warm = false (this node's first task on the SM): first a dry
// pass, the same code with one round of loads and no stores, while
// the GEMM's parts are on their way; one copy of the code (unroll 1)
template <int K, int P, int NC>
__device__ __forceinline__ void
    act_quant_parts_task(float const *part,
                         unsigned char *out,
                         uint32_t *done_cnt,
                         int t,
                         int cy,
                         bool warm STAGE_STAMP_PARAM) {
  // timing build: stamp [0] the dry pass done, [1] thread 0's parts landed
  static_assert(NC % 2048 == 0 && K % NC == 0,
                "act_quant (K parts): whole passes of 2048 columns, 8 per "
                "thread");
  int const l = threadIdx.x & 31;
#pragma unroll 1
  for (int pass = warm ? 1 : 0; pass < 2; pass++) {
    bool const real = pass == 1;
    if (real && threadIdx.x == 0) {
      STAGE_STAMP_AT(0)
    }
    for (int c = cy * NC + 8 * threadIdx.x; c < (cy + 1) * NC; c += 8 * 256) {
      float y[8];
      add_parts8<K, P>(part, t, c, y, !real);
      if (real && threadIdx.x == 0) {
        STAGE_STAMP_AT(1)
      }
      float amax = 0.f;
#pragma unroll
      for (int k = 0; k < 8; k++) {
        y[k] = __bfloat162float(__float2bfloat16_rn(y[k])); // the GEMM's output
        amax = fmaxf(amax, fabsf(y[k]));
      }
      for (int d = 1; d < 4; d <<= 1) { // over the 4 threads of the group
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
      }
      int const ex = xq_exponent(amax);
      uint2 const q = make_uint2(xq_pack4(y, ex), xq_pack4(y + 4, ex));
      uint32_t const s4 = xq_scale_word<16>(ex); // 16 threads: a 128-col block
      if (real) {
        *reinterpret_cast<uint2 *>(out + (size_t)t * K + c) = q;
        if ((l & 15) < 4) {
          xq_scale_line(out, K, c / 128, t, l & 15, s4);
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt, 1); // release: the CTA's stores before (the barrier)
  }
  // the parts re-armed after the count (its release does not wait for these
  // stores; the next writers are the next launch's)
  for (int c = cy * NC + 8 * threadIdx.x; c < (cy + 1) * NC; c += 8 * 256) {
    rearm_parts8<K, P>(part, t, c);
  }
}

// act_quant of h = SwiGLU(gate, up) of token t (model.py Expert.forward with
// weights None: the shared expert), from a gemm_fp8_tile node's P K parts
// [P][T][2 I] (gate in columns [0, I), up in [I, 2 I)): gate and up added in
// part order and rounded to bf16 (the GEMM's bf16 outputs), up clamped to
// [-L, L] and gate to <= L (L: LIMIT_BITS as fp32), h = bf16(silu(gate) * up)
// (silu as torch: g / (1 + expf(-g))), then act_quant_task's formula into
// act_quant's layout of width K: columns [I, K) zero (the next GEMM's K
// padding). Thread i columns 8 i .. 8 i + 7 of each pass of 2048 (as
// act_quant_parts_task); warm: its dry pass
template <int K, int I, int P, int LIMIT_BITS>
__device__ __forceinline__ void act_quant_swiglu_task(float const *part,
                                                      unsigned char *out,
                                                      uint32_t *done_cnt,
                                                      int t,
                                                      bool warm
                                                          STAGE_STAMP_PARAM) {
  // timing build: stamp [0] the dry pass done, [1] thread 0's parts landed
  static_assert(K % 256 == 0 && I % 32 == 0 && I <= K,
                "act_quant (SwiGLU): whole warps of 8 columns per thread, "
                "whole 32-column groups of h");
  float const lim = __int_as_float(LIMIT_BITS);
  int const l = threadIdx.x & 31;
#pragma unroll 1
  for (int pass = warm ? 1 : 0; pass < 2; pass++) {
    bool const real = pass == 1;
    if (real && threadIdx.x == 0) {
      STAGE_STAMP_AT(0)
    }
    for (int c = 8 * threadIdx.x; c < K; c += 8 * 256) {
      float y[8];
      if (c < I) { // a group is all h or all padding (I % 32 == 0)
        float gt[8], up[8];
        add_parts8<2 * I, P>(part, t, c, gt, !real);
        add_parts8<2 * I, P>(part, t, I + c, up, !real);
        if (real && threadIdx.x == 0) {
          STAGE_STAMP_AT(1)
        }
#pragma unroll
        for (int k = 0; k < 8; k++) {
          float const g = fminf(bf16r(gt[k]), lim);
          float const u = fminf(fmaxf(bf16r(up[k]), -lim), lim);
          y[k] = bf16r(g / (1.0f + expf(-g)) * u);
        }
      } else {
#pragma unroll
        for (int k = 0; k < 8; k++) {
          y[k] = 0.f;
        }
      }
      float amax = 0.f;
#pragma unroll
      for (int k = 0; k < 8; k++) {
        amax = fmaxf(amax, fabsf(y[k]));
      }
      for (int d = 1; d < 4; d <<= 1) { // over the 4 threads of the group
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
      }
      int const ex = xq_exponent(amax);
      uint2 const q = make_uint2(xq_pack4(y, ex), xq_pack4(y + 4, ex));
      uint32_t const s4 = xq_scale_word<16>(ex); // 16 threads: a 128-col block
      if (real) {
        *reinterpret_cast<uint2 *>(out + (size_t)t * K + c) = q;
        if ((l & 15) < 4) {
          xq_scale_line(out, K, c / 128, t, l & 15, s4);
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt, 1); // release: the CTA's stores before (the barrier)
  }
  for (int c = 8 * threadIdx.x; c < I; c += 8 * 256) { // after the count
    rearm_parts8<2 * I, P>(part, t, c);
    rearm_parts8<2 * I, P>(part, t, I + c);
  }
}

// act_quant task (t, y) of grid (T, C, 1): token t, columns [y K / C,
// (y + 1) K / C); x a graph input (BUF_SLOTS {x, a graph tensor [T][K] bf16})
// or a gemm_fp8_tile's output in slots (X::y K parts, X::v[2] = K; BUF_SLOTS
// {}); warm: below (the dry pass; true for a
// graph input's node, which has none). PARAMS {K}, or {K, I, L as fp32 bits}:
// h = SwiGLU of a gemm_fp8_tile's gate | up K parts (X::v[2] = 2 I; grid (T,
// 1, 1); act_quant_swiglu_task)
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class X,
          class... REST>
__device__ __forceinline__ void run_act_quant(Maps const &,
                                              G const &g,
                                              KernelLocals &L,
                                              StaticTask const &tk,
                                              bool warm = true) {
  constexpr int K = PARAMS::v[0];
  constexpr int I = param_or0<PARAMS, 1>(); // SwiGLU: gate's width; 0: none
  static_assert(I > 0 || K % 512 == 0,
                "act_quant: whole warps of 128-column blocks "
                "(16 columns per thread, 8 threads per block)");
  static_assert(SELF::x == T && SELF::z == 1 && K % SELF::y == 0,
                "act_quant: grid (T, column slices)");
  static_assert(SELF::buf >= 0 && exchange_bytes[SELF::buf] == 0 &&
                    SELF::counter >= 0,
                "act_quant: its output in this GPU's memory, a counter of its "
                "tasks");
  if constexpr (I > 0) { // h = SwiGLU of the gate | up K parts
    static_assert(X::buf >= 0 && X::v[0] == COMBINE_SLOTS &&
                      X::v[2] == 2 * I && X::y >= 1 && X::y <= 10 &&
                      SELF::y == 1,
                  "act_quant (SwiGLU): a GEMM's 2 I columns in at most 10 K "
                  "parts in slots, one task per token");
    act_quant_swiglu_task<K, I, X::y, param_at<PARAMS, 2>()>(
        buf_at<float const>(g, X::buf),
        buf_at<unsigned char>(g, SELF::buf),
        g.cnt + SELF::counter,
        tk.x,
        warm STAGE_STAMP_ARG(L.stage_stamps));
  } else if constexpr (X::buf < 0) { // x: a graph input
    act_quant_task<K, K / SELF::y>(
        buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
        buf_at<unsigned char>(g, SELF::buf),
        g.cnt + SELF::counter,
        tk.x,
        tk.y STAGE_STAMP_ARG(L.stage_stamps));
  } else {
    static_assert(X::v[0] == COMBINE_SLOTS && X::v[2] == K && X::y >= 1 &&
                      X::y <= 10,
                  "act_quant: a GEMM's K parts in slots, at most 10 (in "
                  "registers, 8 values each)");
    act_quant_parts_task<K, X::y, K / SELF::y>(
        buf_at<float const>(g, X::buf),
        buf_at<unsigned char>(g, SELF::buf),
        g.cnt + SELF::counter,
        tk.x,
        tk.y,
        warm STAGE_STAMP_ARG(L.stage_stamps));
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_act_quant() {
  return 0;
}

} // namespace static_mk
