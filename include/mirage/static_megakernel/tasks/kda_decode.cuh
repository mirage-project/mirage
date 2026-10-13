// tasks/kda_decode.cuh -- one decode step of KDA (Kimi Delta Attention, the
// linear attention of GLM-5.3-Flash) for one (request, head): task (t, h) of
// grid (T, HL, 1), following vLLM's glm5next kda.py + fused_recurrent_kda:
//   1. add the merged projection's NSPLIT K parts (a gemm_tile node in slots,
//      columns [q HL D | k HL D | v HL D | b HL | f_a D | g_a D], this GPU's
//      heads), rounded to bf16 as vLLM's bf16 GEMM output
//   2. short causal conv (width 4, fp32 weights) + SiLU on the head's 3 D
//      q / k / v channels; the conv state [NR][3 HL D][3] (bf16, oldest first)
//      is updated in place
//   3. q, k: L2-normalised (eps 1e-6), q x D^-1/2; beta = sigmoid(b);
//      forget gate g = lb sigmoid(e^A_log (f_b f_a + dt_bias)) per key dim;
//      output gate z = g_b g_a
//   4. the delta rule on the state S [V = D][K = D] fp32 (vLLM's layout):
//      S = S diag(e^g); S += beta (v - S k) k^T; o = S q
//   5. o (bf16) -> RMSNorm(o) w sigmoid(z) -> the node's output o [T][HL D]
// then the node's counter += 1 (the o_proj GEMM waits for all T x HL tasks),
// then the new state row is stored (the next step's launch reads it).
// The state rows of the T requests are rows[t] (an int32 graph tensor).
// Before the projection is polled: the head's f_b / g_b rows go to shared
// memory (cp.async), the state row S, the conv weights and state and the small
// values to registers.
#pragma once
#include "../core.cuh"

namespace static_mk {

__device__ __forceinline__ float kda_bf16r(float v) {
  return __bfloat162float(__float2bfloat16(v));
}
__device__ __forceinline__ float kda_sigmoid(float z) {
  return 1.0f / (1.0f + expf(-z));
}
// the NSPLIT K parts of 4 consecutive columns of row t, loaded at once, each
// re-polled until it has landed (0xFF prefill), added in order, rounded to bf16
template <int NSPLIT, int NPROJ>
__device__ __forceinline__ float4 kda_proj4(float const *proj, int t, int col) {
  float4 v[NSPLIT];
#pragma unroll
  for (int p = 0; p < NSPLIT; p++) {
    v[p] = ldf4_poll(proj + ((size_t)p * T + t) * NPROJ + col);
  }
  float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int p = 0; p < NSPLIT; p++) {
    while (!validf4(v[p])) {
      v[p] = ldf4_poll(proj + ((size_t)p * T + t) * NPROJ + col);
    }
    acc.x += v[p].x;
    acc.y += v[p].y;
    acc.z += v[p].z;
    acc.w += v[p].w;
  }
  return make_float4(
      kda_bf16r(acc.x), kda_bf16r(acc.y), kda_bf16r(acc.z), kda_bf16r(acc.w));
}
// 16 B global -> shared without registers; kda_cp_wait: this thread's copies
// landed
__device__ __forceinline__ void kda_cp16(void *dst, void const *src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(su32(dst)),
               "l"(src)
               : "memory");
}
__device__ __forceinline__ void kda_cp_wait() {
  asm volatile("cp.async.wait_all;" ::: "memory");
}

constexpr int KDA_WROW = 128 + 8; // bf16 per f_b / g_b row in shared memory:
                                  // rows 16 B apart in the banks
constexpr int KDA_SCRATCH_FLOATS = 10 * 128 + 16;
constexpr int KDA_SMEM_BYTES = 2 * 128 * KDA_WROW * 2 + KDA_SCRATCH_FLOATS * 4;

template <int NSPLIT, int NPROJ, int HL, int D>
__device__ __forceinline__ void
    kda_decode_task(float const *proj,
                    float const *conv_w,
                    __nv_bfloat16 const *fb,
                    __nv_bfloat16 const *gb,
                    float const *small,
                    __nv_bfloat16 *conv_state,
                    float *rec,
                    int const *rows,
                    __nv_bfloat16 *out,
                    uint32_t *done_cnt,
                    int t,
                    int h,
                    float lower_bound,
                    float norm_eps,
                    char *scratch STAGE_STAMP_PARAM) {
  constexpr int HLD = HL * D;
  constexpr int BO = 3 * HLD, FAO = BO + HL, GAO = FAO + D;
  static_assert(D == 128,
                "kda_decode: head dim 128 (256 threads = 128 rows x 2)");
  static_assert(
      HL % 4 == 0 && NPROJ % 4 == 0 && NPROJ >= GAO + D,
      "kda_decode: the merged projection [q | k | v | b | f_a | g_a]");
  int const tid = threadIdx.x;
  // [2 D][KDA_WROW]: the f_b rows of head h, then its g_b rows
  __nv_bfloat16 *w_s = reinterpret_cast<__nv_bfloat16 *>(scratch);
  float *xq = reinterpret_cast<float *>(w_s + 2 * D * KDA_WROW);
  float *xk = xq + D, *xv = xk + D;
  float *fa = xv + D, *ga = fa + D, *qn = ga + D, *kn = qn + D, *eg = kn + D;
  float *og = eg + D, *ov = og + D, *misc = ov + D; // misc [16]
  int const row = rows[t];
  // 0. what does not wait for the projection, to registers first: row v of S
  // (thread pair (2v, 2v + 1): thread 2v + hf has the 4-key groups 2i + hf, so
  // the pair's shared-memory reads below hit different banks)
  int const v = tid >> 1, hf = tid & 1;
  float4 *sp =
      reinterpret_cast<float4 *>(rec + (((size_t)row * HL + h) * D + v) * D);
  float s[64];
#pragma unroll
  for (int i = 0; i < 16; i++) {
    float4 const u = sp[2 * i + hf];
    s[4 * i] = u.x;
    s[4 * i + 1] = u.y;
    s[4 * i + 2] = u.z;
    s[4 * i + 3] = u.w;
  }
  // the conv: thread tid < 96 has 4 channels of q | k | v
  int const seg = tid >> 5, j = (tid & 31) * 4;
  int const ch = seg * HLD + h * D + j; // local conv channel of q | k | v
  __nv_bfloat16 *st = conv_state + ((size_t)row * (3 * HLD) + ch) * 3;
  float4 cw[4];
  float cs[12];
  if (tid < 96) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
      cw[k] = reinterpret_cast<float4 const *>(conv_w + (size_t)ch * 4)[k];
    }
#pragma unroll
    for (int k = 0; k < 12; k++) {
      cs[k] = __bfloat162float(st[k]);
    }
  }
  float const alog = small[HLD + h];
  float const dtb = tid < D ? small[h * D + tid] : 0.f;
  float const onw = tid < D ? small[HLD + HL + tid] : 0.f;
  // the ring area is free once every warp has left the previous task (a
  // GEMM's MMAs read it asynchronously): f_b rows of head h, then g_b's (row r
  // of either: thread r of the gates below)
  __syncthreads();
#pragma unroll
  for (int i = 0; i < 2 * D * D / 8 / 256; i++) {
    int const c = tid + 256 * i, r = c / (D / 8), k = c % (D / 8);
    kda_cp16(w_s + r * KDA_WROW + 8 * k,
             (r < D ? fb : gb) + ((size_t)h * D + r % D) * D + 8 * k);
  }
  // 1-2. projection parts -> bf16 -> short conv + SiLU (q, k, v), f_a, g_a, b
  if (tid < 96) {
    float4 const x4 = kda_proj4<NSPLIT, NPROJ>(proj, t, ch);
    float const xr[4] = {x4.x, x4.y, x4.z, x4.w};
    float *dst = (seg == 0 ? xq : (seg == 1 ? xk : xv)) + j;
#pragma unroll
    for (int k = 0; k < 4; k++) {
      float4 const wk = cw[k];
      float o = wk.x * cs[3 * k] + wk.y * cs[3 * k + 1] + wk.z * cs[3 * k + 2] +
                wk.w * xr[k];
      o = o * kda_sigmoid(o);
      dst[k] = kda_bf16r(o);
      st[3 * k] = __float2bfloat16(cs[3 * k + 1]);
      st[3 * k + 1] = __float2bfloat16(cs[3 * k + 2]);
      st[3 * k + 2] = __float2bfloat16(xr[k]);
    }
  } else if (tid < 128) {
    int const jj = (tid - 96) * 4;
    *reinterpret_cast<float4 *>(fa + jj) =
        kda_proj4<NSPLIT, NPROJ>(proj, t, FAO + jj);
  } else if (tid < 160) {
    int const jj = (tid - 128) * 4;
    *reinterpret_cast<float4 *>(ga + jj) =
        kda_proj4<NSPLIT, NPROJ>(proj, t, GAO + jj);
  } else if (tid == 160) {
    float4 const b4 = kda_proj4<NSPLIT, NPROJ>(proj, t, BO + (h & ~3));
    float const b[4] = {b4.x, b4.y, b4.z, b4.w};
    misc[0] = kda_sigmoid(b[h & 3]);
  }
  kda_cp_wait();
  __syncthreads();
  if (tid == 0) {
    STAGE_STAMP_AT(0)
  } // timing build: the projection landed, the conv done
  // 3a. forget gate (threads 0 .. D - 1, key dim k) and output gate (threads
  // D .. 2D - 1, value dim v): 128-long dots with this head's f_b / g_b rows
  {
    bool const fgate = tid < D;
    int const r = fgate ? tid : tid - D;
    __nv_bfloat16 const *wrow = w_s + tid * KDA_WROW;
    float const *a = fgate ? fa : ga;
    float acc = 0.f;
#pragma unroll 4
    for (int m = 0; m < D; m += 8) {
      uint4 const u = *reinterpret_cast<uint4 const *>(wrow + m);
      __nv_bfloat16 const *wb = reinterpret_cast<__nv_bfloat16 const *>(&u);
#pragma unroll
      for (int i = 0; i < 8; i++) {
        acc += __bfloat162float(wb[i]) * a[m + i];
      }
    }
    acc = kda_bf16r(acc); // vLLM: a bf16 GEMM output
    if (fgate) {
      float const z = expf(alog) * (acc + dtb);
      eg[r] = expf(lower_bound / (1.0f + expf(-z)));
    } else {
      og[r] = acc;
    }
  }
  // 3b. L2 norms of q (warp 0) and k (warp 1)
  if (tid < 64) {
    int const lane = tid & 31;
    float const *src = tid < 32 ? xq : xk;
    float sq = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      sq += src[lane * 4 + i] * src[lane * 4 + i];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, o);
    }
    if (lane == 0) {
      misc[1 + (tid >> 5)] = sq;
    }
  }
  __syncthreads();
  if (tid < D) {
    qn[tid] = xq[tid] / sqrtf(misc[1] + 1e-6f) * (1.0f / sqrtf((float)D));
  } else {
    kn[tid - D] = xk[tid - D] / sqrtf(misc[2] + 1e-6f);
  }
  __syncthreads();
  if (tid == 0) {
    STAGE_STAMP_AT(1)
  } // timing build: the gates and the normalised q, k
  // 4. the delta rule on row v of S (in registers since step 0)
  {
    float4 const *eg4 = reinterpret_cast<float4 const *>(eg);
    float4 const *kn4 = reinterpret_cast<float4 const *>(kn);
    float4 const *qn4 = reinterpret_cast<float4 const *>(qn);
    float d = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) {
      float4 const e = eg4[2 * i + hf], k = kn4[2 * i + hf];
      float const ea[4] = {e.x, e.y, e.z, e.w}, ka[4] = {k.x, k.y, k.z, k.w};
#pragma unroll
      for (int c = 0; c < 4; c++) {
        s[4 * i + c] *= ea[c];
        d += s[4 * i + c] * ka[c];
      }
    }
    d += __shfl_xor_sync(0xffffffffu, d, 1);
    float const delta = (xv[v] - d) * misc[0];
    float o = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) {
      float4 const k = kn4[2 * i + hf], q = qn4[2 * i + hf];
      float const ka[4] = {k.x, k.y, k.z, k.w}, qa[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
      for (int c = 0; c < 4; c++) {
        s[4 * i + c] += ka[c] * delta;
        o += s[4 * i + c] * qa[c];
      }
    }
    o += __shfl_xor_sync(0xffffffffu, o, 1);
    if ((tid & 1) == 0) {
      ov[v] = kda_bf16r(o); // vLLM's core_attn_out is bf16
    }
  }
  __syncthreads();
  // 5. gated RMSNorm of o
  if (tid < D) {
    float sq = ov[tid] * ov[tid];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, o);
    }
    if ((tid & 31) == 0) {
      misc[4 + (tid >> 5)] = sq;
    }
  }
  __syncthreads();
  if (tid < D) {
    float const r =
        rsqrtf((misc[4] + misc[5] + misc[6] + misc[7]) / (float)D + norm_eps);
    float const y = ov[tid] * r * onw * kda_sigmoid(og[tid]);
    out[(size_t)t * HLD + h * D + tid] = __float2bfloat16(y);
  }
  __syncthreads();
  if (tid == 0) {
    cnt_add(done_cnt, 1); // release: o (and the conv state)
  }
  // 6. the new row v of S, after the release: only the next step reads it
#pragma unroll
  for (int i = 0; i < 16; i++) {
    sp[2 * i + hf] =
        make_float4(s[4 * i], s[4 * i + 1], s[4 * i + 2], s[4 * i + 3]);
  }
}

// kda_decode task (token x, head y). PROJ: the merged projection's gemm_tile
// node (COMBINE_SLOTS, its params {mode, K, N = NPROJ, polled}); PARAMS {HL, D,
// gate lower bound, o_norm eps (fp32 bits)}; BUF_SLOTS {conv weight [3 HL
// D][4] fp32, f_b [HL D][D] bf16, g_b [HL D][D] bf16, small (dt_bias [HL D],
// A_log [HL], o_norm weight [D]) fp32, conv state [NR][3 HL D][3] bf16,
// recurrent state [NR][HL][D][D] fp32, rows [T] int32}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class PROJ,
          class... IN>
__device__ __forceinline__ void run_kda_decode(Maps const &,
                                               G const &g,
                                               KernelLocals &L,
                                               StaticTask const &tk) {
  constexpr int HL = PARAMS::v[0], D = PARAMS::v[1];
  static_assert(PROJ::buf >= 0 && PROJ::v[0] == COMBINE_SLOTS &&
                    PROJ::v[3] == 0,
                "kda_decode: its input is a gemm_tile node's output in slots");
  static_assert(SELF::x == T && SELF::y == HL && SELF::buf >= 0 &&
                    SELF::counter >= 0,
                "kda_decode: grid (T, HL), its output o and its counter");
  kda_decode_task<PROJ::y, PROJ::v[2], HL, D>(
      buf_at<float const>(g, PROJ::buf),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 4>()),
      buf_at<float>(g, slot_at<BUF_SLOTS, 5>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 6>()),
      buf_at<__nv_bfloat16>(g, SELF::buf),
      g.cnt + SELF::counter,
      tk.x,
      tk.y,
      __int_as_float(PARAMS::v[2]),
      __int_as_float(PARAMS::v[3]),
      L.sm STAGE_STAMP_ARG(L.stage_stamps));
}
// dynamic shared memory: the f_b / g_b rows and the scratch, inside the ring
// area
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_kda_decode() {
  static_assert(KDA_SMEM_BYTES <= RING_BYTES,
                "kda_decode: its shared memory in the ring area");
  return RING_BYTES;
}

} // namespace static_mk
