// tasks/kda_decode.cuh -- KDA (Kimi Delta Attention, the linear attention of
// GLM-5.3-Flash) as vLLM's glm5next kda.py + fused_recurrent_kda compute it, on
// SGLang's KDA pool (per state row the conv window [3][3 HL D] bf16, oldest
// first, and the state [HL][V][K] fp32). Task (r, h) of grid (T / RP, HL, 1):
// head h of requests RP r .. RP r + RP - 1 of a decode step, one token each
// (RP = 1, or 2 / 4 / 8: T = 16 / 32 / 64 in 128 tasks), request q's state row
// rows[RP r + q]; the window and the state are updated in place
// Per token:
//   1. add the merged projection's NSPLIT K parts (a gemm_tile node in slots,
//      columns [q HL D | k HL D | v HL D | b HL | f_a D | g_a D], this GPU's
//      heads), rounded to bf16 as vLLM's bf16 GEMM output
//   2. short causal conv (width 4, fp32 weights) + SiLU on the head's 3 D
//      q / k / v channels over the window's 3 inputs before the token; a decode
//      step's window becomes [its 2 newest inputs, the token's input]
//   3. q, k: L2-normalised (eps 1e-6), q x D^-1/2; beta = sigmoid(b);
//      forget gate g = lb sigmoid(e^A_log (f_b f_a + dt_bias)) per key dim;
//      output gate z = g_b g_a
//   4. the delta rule on the state S [V = D][K = D] fp32 (vLLM's layout):
//      S = S diag(e^g); S += beta (v - S k) k^T; o = S q
//   5. o (bf16) -> RMSNorm(o) w sigmoid(z) -> the node's output o [T][HL D]
// then the node's counter += 1 (the o_proj GEMM waits for all tasks), then the
// state is stored (only the next step's launch reads it). A request whose slot
// is < 0 is idle (a padded batch): o = 0, no state read or written.
// Before the projection is polled: the head's f_b / g_b rows go to shared
// memory (cp.async), the initial state S, the conv weights and state and the
// small values to registers.
#pragma once
#include "../core.cuh"

namespace static_mk {

__device__ __forceinline__ float kda_bf16r(float v) {
  return __bfloat162float(__float2bfloat16(v));
}
__device__ __forceinline__ float kda_sigmoid(float z) {
  return 1.0f / (1.0f + expf(-z));
}
// 4 consecutive columns of rows t0 .. t0 + N - 1: the NSPLIT K parts of all N
// rows loaded at once and re-polled together until every one has landed (0xFF
// prefill; a round of re-polls is one round trip, not one per part: the parts
// loaded before the GEMM's stores landed are all stale at once), added in
// order, rounded to bf16
template <int NSPLIT, int NPROJ, int N>
__device__ __forceinline__ void kda_proj4(float const *proj, int t0, int col,
                                          float4 (&x)[N]) {
  float4 v[N][NSPLIT];
#pragma unroll
  for (int i = 0; i < N; i++) {
#pragma unroll
    for (int p = 0; p < NSPLIT; p++) {
      v[i][p] = ldf4_poll(proj + ((size_t)p * T + t0 + i) * NPROJ + col);
    }
  }
  bool all = true;
#pragma unroll
  for (int i = 0; i < N; i++) {
#pragma unroll
    for (int p = 0; p < NSPLIT; p++) {
      all = all && validf4(v[i][p]);
    }
  }
#pragma unroll 1
  while (!all) {
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int p = 0; p < NSPLIT; p++) {
        if (!validf4(v[i][p])) {
          v[i][p] = ldf4_poll(proj + ((size_t)p * T + t0 + i) * NPROJ + col);
        }
      }
    }
    all = true;
#pragma unroll
    for (int i = 0; i < N; i++) {
#pragma unroll
      for (int p = 0; p < NSPLIT; p++) {
        all = all && validf4(v[i][p]);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < N; i++) {
    float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int p = 0; p < NSPLIT; p++) {
      acc.x += v[i][p].x;
      acc.y += v[i][p].y;
      acc.z += v[i][p].z;
      acc.w += v[i][p].w;
    }
    x[i] = make_float4(kda_bf16r(acc.x), kda_bf16r(acc.y), kda_bf16r(acc.z),
                       kda_bf16r(acc.w));
  }
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
// mma.sync m16n8k16 (bf16 in, fp32 accumulate): A from 16 row-major rows
// (ldmatrix x4), B from 8 [n][k] rows (ldmatrix x2)
__device__ __forceinline__ void kda_ldsm_x4(uint32_t (&r)[4],
                                            __nv_bfloat16 const *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(su32(p))
               : "memory");
}
__device__ __forceinline__ void kda_ldsm_x2(uint32_t (&r)[2],
                                            __nv_bfloat16 const *p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(su32(p))
               : "memory");
}
__device__ __forceinline__ void
kda_mma16816(uint32_t const (&a)[4], uint32_t const (&b)[2], float (&c)[4]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
                 "r"(b[1]));
}

// The delta rule's thread layout: thread tid has rows 4 b .. 4 b + 3 (b =
// tid / 8) of S, and of each row the 4 key chunks 32 j + 4 g .. 32 j + 4 g + 3
// (g = tid % 8, j < 4): s[16 r + 4 j + c] = S[4 b + r][32 j + 4 g + c]. The 8
// lanes of a row block (lanes 8 m .. 8 m + 7 of a warp) read 8 different 16 B
// chunks of a token's vectors (no bank conflict, 4x less shared-memory traffic
// than a row per thread pair) and add their dot products with 3 shuffles.
// One delta-rule step on the thread's part of S. eg, kn, qn: the token's decay,
// normalised k and q [D] (shared memory); xv: v of the thread's 4 rows; beta:
// sigmoid(b); o of the 4 rows to o (on every lane of the row block).
__device__ __forceinline__ void kda_rule(float (&s)[64], float const *eg,
                                         float const *kn, float const *qn,
                                         float4 xv, float beta, int g,
                                         float (&o)[4]) {
  float e[16], k[16], q[16];
#pragma unroll
  for (int j = 0; j < 4; j++) {
    float4 const e4 = *reinterpret_cast<float4 const *>(eg + 32 * j + 4 * g);
    float4 const k4 = *reinterpret_cast<float4 const *>(kn + 32 * j + 4 * g);
    float4 const q4 = *reinterpret_cast<float4 const *>(qn + 32 * j + 4 * g);
    float const ea[4] = {e4.x, e4.y, e4.z, e4.w};
    float const ka[4] = {k4.x, k4.y, k4.z, k4.w};
    float const qa[4] = {q4.x, q4.y, q4.z, q4.w};
#pragma unroll
    for (int c = 0; c < 4; c++) {
      e[4 * j + c] = ea[c];
      k[4 * j + c] = ka[c];
      q[4 * j + c] = qa[c];
    }
  }
  // the dot products in 4 partial sums each: short dependency chains
  float d[4];
#pragma unroll
  for (int r = 0; r < 4; r++) {
    float p[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < 16; c++) {
      s[16 * r + c] *= e[c];
      p[c & 3] += s[16 * r + c] * k[c];
    }
    d[r] = (p[0] + p[1]) + (p[2] + p[3]);
  }
#pragma unroll
  for (int m = 1; m < 8; m <<= 1) {
#pragma unroll
    for (int r = 0; r < 4; r++) {
      d[r] += __shfl_xor_sync(0xffffffffu, d[r], m);
    }
  }
  float const xr[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
  for (int r = 0; r < 4; r++) {
    float const delta = (xr[r] - d[r]) * beta;
    float p[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int c = 0; c < 16; c++) {
      s[16 * r + c] += k[c] * delta;
      p[c & 3] += s[16 * r + c] * q[c];
    }
    o[r] = (p[0] + p[1]) + (p[2] + p[3]);
  }
#pragma unroll
  for (int m = 1; m < 8; m <<= 1) {
#pragma unroll
    for (int r = 0; r < 4; r++) {
      o[r] += __shfl_xor_sync(0xffffffffu, o[r], m);
    }
  }
}
// the thread's part of S from / to a [D][LD] fp32 matrix m (one head's state
// slot in global memory, LD = D; the shared-memory copy, LD = KDA_SROW)
template <int LD>
__device__ __forceinline__ void kda_state_load(float (&s)[64], float const *m,
                                               int b, int g) {
#pragma unroll
  for (int r = 0; r < 4; r++) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float4 const u = *reinterpret_cast<float4 const *>(m + (4 * b + r) * LD +
                                                         32 * j + 4 * g);
      s[16 * r + 4 * j] = u.x;
      s[16 * r + 4 * j + 1] = u.y;
      s[16 * r + 4 * j + 2] = u.z;
      s[16 * r + 4 * j + 3] = u.w;
    }
  }
}
template <int LD>
__device__ __forceinline__ void kda_state_store(float *m, float const (&s)[64],
                                                int b, int g) {
#pragma unroll
  for (int r = 0; r < 4; r++) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      *reinterpret_cast<float4 *>(m + (4 * b + r) * LD + 32 * j + 4 * g) =
          make_float4(s[16 * r + 4 * j], s[16 * r + 4 * j + 1],
                      s[16 * r + 4 * j + 2], s[16 * r + 4 * j + 3]);
    }
  }
}

constexpr int KDA_WROW = 128 + 8; // bf16 per f_b / g_b row in shared memory:
                                  // rows 16 B apart in the banks
constexpr int KDA_SROW = 128 + 4; // fp32 per row of a state in shared memory
// shared memory: the f_b / g_b rows [2 D][KDA_WROW] bf16; f_a and g_a of the
// tokens ([8][KDA_WROW] bf16 each: the MMA's B, 8 token columns); per token
// q, k, v, the decay, the output gate and o ([RP][D] fp32 each); 32 small
// values; when a task has more than one request, [D][KDA_SROW] fp32 (the next
// request's state) and the other requests' conv windows
template <int RP> // the task's requests (one token each)
__host__ __device__ constexpr int kda_smem_bytes() {
  return (2 * 128 + 2 * 8) * KDA_WROW * 2 + (6 * RP * 128 + 32) * 4 +
         (RP > 1 ? 128 * KDA_SROW * 4 : 0) + (RP - 1) * 96 * 12 * 2;
}

template <int NSPLIT, int NPROJ, int HL, int D, int RP = 1>
__device__ __forceinline__ void kda_decode_task(
    float const *proj, float const *conv_w, __nv_bfloat16 const *fb,
    __nv_bfloat16 const *gb, float const *small, __nv_bfloat16 *conv_state,
    float *rec, int const *rows, __nv_bfloat16 *out, uint32_t *done_cnt, int r,
    int h, float lower_bound, float norm_eps, char *scratch STAGE_STAMP_PARAM) {
  constexpr int HLD = HL * D;
  constexpr int BO = 3 * HLD, FAO = BO + HL, GAO = FAO + D;
  // the conv window of a state row: 3 entries [3][3 HL D]; element (channel c,
  // entry e): c CS + e ES
  constexpr int CL = 3, CS = 1, ES = 3 * HLD;
  static_assert(D == 128,
                "kda_decode: head dim 128 (256 threads: 32 row blocks x 8)");
  static_assert(
      HL % 4 == 0 && NPROJ % 4 == 0 && NPROJ >= GAO + D,
      "kda_decode: the merged projection [q | k | v | b | f_a | g_a]");
  // RP: requests per task (2 / 4 / 8: a decode step of T = 16 / 32 / 64 in 128
  // tasks, one wave of the SMs; the head's f_b / g_b rows and steps 1 .. 3
  // shared, each request's conv window and state its own, the delta rules one
  // after the other: request q + 1's state copied to shared memory while
  // request q's rule runs; requests 1 .. RP - 1's conv windows in shared memory
  // from step 0). NTK: the task's tokens (rows t0 .. t0 + NTK - 1)
  constexpr int NTK = RP;
  static_assert(RP == 1 || RP == 2 || RP == 4 || RP == 8,
                "kda_decode: 1, 2, 4 or 8 requests per task (the gates' MMA: 8 "
                "token columns)");
  int const tid = threadIdx.x;
  int const t0 = r * NTK;    // the task's first row
  int const slot = rows[t0]; // the state row of its first request
  // RP > 1: each request's slot, whether it is live (an idle request of a
  // task: its o = 0, its state and conv window untouched)
  int slotq[RP];
  bool liveq[RP];
  bool any = false;
#pragma unroll
  for (int q = 0; q < RP; q++) {
    slotq[q] = q == 0 ? slot : rows[t0 + q];
    liveq[q] = slotq[q] >= 0;
    any = any || liveq[q];
  }
  bool const live0 = liveq[0];
  int const warp = tid >> 5, lane = tid & 31;
  // [2 D][KDA_WROW]: the f_b rows of head h, then its g_b rows; [8][KDA_WROW]:
  // f_a of the tokens, then [8][KDA_WROW] g_a (rows NTK .. 7 unused)
  __nv_bfloat16 *w_s = reinterpret_cast<__nv_bfloat16 *>(scratch);
  __nv_bfloat16 *fab = w_s + 2 * D * KDA_WROW, *gab = fab + 8 * KDA_WROW;
  // per token [NTK][D]: q and k (normalised in place), v, the decay e^g, the
  // output gate, o
  float *xq = reinterpret_cast<float *>(gab + 8 * KDA_WROW);
  float *xk = xq + NTK * D, *xv = xk + NTK * D, *eg = xv + NTK * D;
  float *og = eg + NTK * D, *ov = og + NTK * D;
  float *beta = ov + NTK * D; // [NTK]
  float *nrm = beta + 8;      // [2 NTK]: |q_i|^2, |k_i|^2
  float *onrm = nrm + 16;     // [NTK]: |o_i|^2
  float *s0 = onrm + 8;       // [D][KDA_SROW]: RP > 1: request q + 1's state
                              // while request q's rule runs
  // RP > 1: requests 1 .. RP - 1's conv windows, [RP - 1][96 threads][12] bf16
  __nv_bfloat16 *cwin = reinterpret_cast<__nv_bfloat16 *>(s0 + D * KDA_SROW);
  // an idle request (a padded batch: its slot < 0): o = 0, no state touched
  if (!any) {
    for (int v = tid; v < NTK * D / 8; v += 256) { // whole 16-B stores (as o)
      *reinterpret_cast<uint4 *>(out + (size_t)(t0 + v / (D / 8)) * HLD +
                                 h * D + (v % (D / 8)) * 8) =
          make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();
    if (tid == 0) {
      cnt_add(done_cnt, 1);
    }
    return;
  }
  // 0. what does not wait for the projection, to registers first: the
  // thread's part of S (kda_rule's layout)
  int const b = tid >> 3, g = tid & 7;
  float s[64];
  if (RP == 1 || live0) {
    kda_state_load<D>(s, rec + ((size_t)slot * HL + h) * D * D, b, g);
  } else {
#pragma unroll
    for (int k = 0; k < 64; k++) {
      s[k] = 0.f;
    }
  }
  // the conv: thread tid < 96 has 4 channels of q | k | v; cs[3 k ..] the 3
  // inputs before this step's first token of channel ch + k
  int const seg = tid >> 5, j = (tid & 31) * 4;
  int const ch = seg * HLD + h * D + j; // local conv channel of q | k | v
  __nv_bfloat16 *st =
      conv_state + (size_t)slot * (3 * HLD) * CL + (size_t)ch * CS;
  // request q's conv row (RP > 1: of an idle request, row 0, never read)
  auto stq = [&](int q) {
    return conv_state + (size_t)(slotq[q] < 0 ? 0 : slotq[q]) * (3 * HLD) * CL +
           (size_t)ch * CS;
  };
  float4 cw[4];
  float cs[12];
  // RP > 1: requests 1 .. RP - 1's windows as loaded (to cwin after the sync
  // below: until then the ring area may still be the previous task's)
  __nv_bfloat16 wq[RP > 1 ? RP - 1 : 1][12];
  if (tid < 96) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
      cw[k] = reinterpret_cast<float4 const *>(conv_w + (size_t)ch * 4)[k];
    }
#pragma unroll
    for (int k = 0; k < 4; k++) {
#pragma unroll
      for (int e = 0; e < 3; e++) {
        cs[3 * k + e] =
            RP == 1 || live0 ? __bfloat162float(st[k * CS + e * ES]) : 0.f;
      }
    }
#pragma unroll
    for (int q = 1; q < RP; q++) { // the other requests' windows
      __nv_bfloat16 const *sq = stq(q);
#pragma unroll
      for (int k = 0; k < 4; k++) {
#pragma unroll
        for (int e = 0; e < 3; e++) {
          wq[q - 1][3 * k + e] =
              liveq[q] ? sq[k * CS + e * ES] : __float2bfloat16(0.f);
        }
      }
    }
  }
  float const alog = small[HLD + h];
  float const dtb = tid < D ? small[h * D + tid] : 0.f;
  float const onw = small[HLD + HL + (tid & (D - 1))];
  // the ring area is free once every warp has left the previous task (a
  // GEMM's MMAs read it asynchronously): f_b rows of head h, then g_b's (the
  // A rows of the gates' MMA below)
  __syncthreads();
  if (RP > 1 && tid < 96) {
#pragma unroll
    for (int q = 1; q < RP; q++) {
#pragma unroll
      for (int k = 0; k < 12; k++) {
        cwin[((q - 1) * 96 + tid) * 12 + k] = wq[q - 1][k];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 2 * D * D / 8 / 256; i++) {
    int const c = tid + 256 * i, rw = c / (D / 8), k = c % (D / 8);
    kda_cp16(w_s + rw * KDA_WROW + 8 * k,
             (rw < D ? fb : gb) + ((size_t)h * D + rw % D) * D + 8 * k);
  }
  // RP > 1: request q's state into s0 (16-B copies, waited by kda_cp_wait)
  auto state_to_s0 = [&](int q) {
    if (liveq[q]) {
      float const *src = rec + ((size_t)slotq[q] * HL + h) * D * D;
#pragma unroll
      for (int i = 0; i < D * D / 4 / 256; i++) {
        int const c = tid + 256 * i, rw = c / (D / 4), k = c % (D / 4);
        kda_cp16(s0 + rw * KDA_SROW + 4 * k, src + (size_t)rw * D + 4 * k);
      }
    }
  };
  if constexpr (RP > 1) { // request 1's state, in flight with the f_b / g_b
                          // rows (step 2's wait)
    state_to_s0(1);
  }
  // 1. projection parts -> bf16, all NTK tokens: q | k | v (threads 0 ..
  // 95, 4 channels each; raw, to their shared-memory rows), f_a (96 .. 127)
  // and g_a (128 .. 159) (bf16 values, stored exactly), beta (160 .. 160 +
  // NTK - 1)
  if (tid < 160) {
    int const col = tid < 96 ? ch : (tid < 128 ? FAO : GAO) + (tid & 31) * 4;
    float4 x4[NTK];
    kda_proj4<NSPLIT, NPROJ, NTK>(proj, t0, col, x4);
    if (tid == 0) {
      STAGE_STAMP_AT(0)
    } // timing build: the projection landed (thread 0's columns)
    if (tid < 96) {
      float *dst = (seg == 0 ? xq : (seg == 1 ? xk : xv)) + j;
#pragma unroll
      for (int i = 0; i < NTK; i++) {
        *reinterpret_cast<float4 *>(dst + i * D) = x4[i];
      }
    } else {
      __nv_bfloat16 *dst = (tid < 128 ? fab : gab) + (tid & 31) * 4;
#pragma unroll
      for (int i = 0; i < NTK; i++) {
        __nv_bfloat162 *d2 =
            reinterpret_cast<__nv_bfloat162 *>(dst + i * KDA_WROW);
        d2[0] = __floats2bfloat162_rn(x4[i].x, x4[i].y);
        d2[1] = __floats2bfloat162_rn(x4[i].z, x4[i].w);
      }
    }
  } else if (tid < 160 + NTK) { // beta of token tid - 160
    float4 b4[1];
    kda_proj4<NSPLIT, NPROJ, 1>(proj, t0 + tid - 160, BO + (h & ~3), b4);
    float const bv[4] = {b4[0].x, b4[0].y, b4[0].z, b4[0].w};
    beta[tid - 160] = kda_sigmoid(bv[h & 3]);
  }
  // 2. short conv + SiLU of the thread's 4 channels, token by token, in
  // place (the window keeps the raw inputs in registers); the conv state's
  // new entries
  if (tid < 96) {
    float *dst = (seg == 0 ? xq : (seg == 1 ? xk : xv)) + j;
#pragma unroll 1
    for (int q = 0; q < RP; q++) { // RP > 1: request q, its window and conv row
      if (q > 0) {
#pragma unroll
        for (int k = 0; k < 12; k++) {
          cs[k] = __bfloat162float(cwin[((q - 1) * 96 + tid) * 12 + k]);
        }
      }
      bool const wr = RP == 1 || liveq[q]; // its conv row written
      // the new window (entry e = x_{e - 2}: the pool's 2 newest inputs,
      // then the token's raw input) to the conv state (its 3 entries)
      __nv_bfloat16 *wd = q == 0 ? st : stq(q);
      auto put = [&](int e, int k, float v) {
        if (wr) {
          wd[k * CS + e * ES] = __float2bfloat16(v);
        }
      };
#pragma unroll
      for (int k = 0; k < 4; k++) {
        put(0, k, cs[3 * k + 1]);
        put(1, k, cs[3 * k + 2]);
      }
      {
        int const tq = q; // the task's token
        float4 const x = *reinterpret_cast<float4 const *>(dst + tq * D);
        float const xr[4] = {x.x, x.y, x.z, x.w};
        float y[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
          float4 const wk = cw[k];
          float o = wk.x * cs[3 * k] + wk.y * cs[3 * k + 1] +
                    wk.z * cs[3 * k + 2] + wk.w * xr[k];
          o = o * kda_sigmoid(o);
          y[k] = kda_bf16r(o);
          put(2, k, xr[k]);
          cs[3 * k] = cs[3 * k + 1];
          cs[3 * k + 1] = cs[3 * k + 2];
          cs[3 * k + 2] = xr[k];
        }
        *reinterpret_cast<float4 *>(dst + tq * D) =
            make_float4(y[0], y[1], y[2], y[3]);
      }
    } // q
  }
  kda_cp_wait();
  __syncthreads();
  // 3a. the gates of every token on tensor cores: the rows [f_b | g_b] (2 D
  // x D) times [f_a | g_a] (D x 8 token columns). Warp w has rows 32 w ..
  // 32 w + 31: w < 4 the forget gate of key dims 32 w .., else the output
  // gate of value dims 32 (w - 4) ..
  {
    __nv_bfloat16 const *xb = warp < 4 ? fab : gab;
    float c[2][4] = {};
#pragma unroll
    for (int kk = 0; kk < D; kk += 16) {
      uint32_t b[2];
      kda_ldsm_x2(b, xb + (lane & 7) * KDA_WROW + kk + ((lane >> 3) & 1) * 8);
#pragma unroll
      for (int mt = 0; mt < 2; mt++) {
        uint32_t a[4];
        kda_ldsm_x4(a, w_s + (32 * warp + 16 * mt + (lane & 15)) * KDA_WROW +
                           kk + (lane >> 4) * 8);
        kda_mma16816(a, b, c[mt]);
      }
    }
    // c[mt][e]: row 32 w + 16 mt + lane / 4 + 8 (e / 2), token 2 (lane % 4)
    // + e % 2; to eg (forget gate) / og (output gate) as they are, finished
    // below
    float *gt = warp < 4 ? eg : og;
#pragma unroll
    for (int mt = 0; mt < 2; mt++) {
#pragma unroll
      for (int e = 0; e < 4; e++) {
        int const rr = 32 * (warp & 3) + 16 * mt + (lane >> 2) + 8 * (e >> 1);
        int const i = 2 * (lane & 3) + (e & 1);
        if (i < NTK) {
          gt[i * D + rr] = c[mt][e];
        }
      }
    }
  }
  // 3b. L2 norms of q_i and k_i: vector 2i (q_i) and 2i + 1 (k_i) on warp
  // (vector mod 8)
  for (int jv = warp; jv < 2 * NTK; jv += 8) {
    float const *src = ((jv & 1) ? xk : xq) + (jv >> 1) * D;
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
      nrm[jv] = sq;
    }
  }
  __syncthreads();
  // 3c. token by token: thread r < D the forget gate of key dim r and
  // q_i[r] (L2-normalised, x D^-1/2), thread D + r the output gate of value
  // dim r and k_i[r] (L2-normalised)
#pragma unroll 1
  for (int i = 0; i < NTK; i++) {
    int const rr = tid & (D - 1);
    if (tid < D) {
      float const g = kda_bf16r(eg[i * D + rr]); // vLLM: a bf16 GEMM output
      float const z = expf(alog) * (g + dtb);
      float const lg = lower_bound / (1.0f + expf(-z)); // the log decay
      float const qn =
          xq[i * D + rr] / sqrtf(nrm[2 * i] + 1e-6f) * (1.0f / sqrtf((float)D));
      eg[i * D + rr] = expf(lg);
      xq[i * D + rr] = qn;
    } else {
      og[i * D + rr] = kda_bf16r(og[i * D + rr]);
      xk[i * D + rr] = xk[i * D + rr] / sqrtf(nrm[2 * i + 1] + 1e-6f);
    }
  }
  __syncthreads();
  if (tid == 0) {
    STAGE_STAMP_AT(1)
  } // timing build: the gates and the normalised q, k
  {
    // 4. the delta rule on the thread's part of S (in registers since step
    // 0), token by token (RP 2: request by request: the first one's state to
    // its slot, the second one's from shared memory)
#pragma unroll 1
    for (int q = 0; q < RP; q++) {
      if (q > 0) { // request q - 1's state to its slot, request q's from s0
        if (liveq[q - 1]) {
          kda_state_store<D>(rec + ((size_t)slotq[q - 1] * HL + h) * D * D, s,
                             b, g);
        }
        kda_state_load<KDA_SROW>(s, s0, b, g);
        if (q + 1 < RP) { // s0 read by every thread: request q + 1's into it
          __syncthreads();
          state_to_s0(q + 1);
        }
      }
      {
        int const tq = q; // the task's token
        float o[4];
        kda_rule(s, eg + tq * D, xk + tq * D, xq + tq * D,
                 *reinterpret_cast<float4 const *>(xv + tq * D + 4 * b),
                 beta[tq], g, o);
        if (g == 0) { // vLLM's core_attn_out is bf16
          *reinterpret_cast<float4 *>(ov + tq * D + 4 * b) =
              make_float4(kda_bf16r(o[0]), kda_bf16r(o[1]), kda_bf16r(o[2]),
                          kda_bf16r(o[3]));
        }
      }
      if (q > 0 && q + 1 < RP) { // request q + 1's state landed (every
                                 // thread's copies)
        kda_cp_wait();
        __syncthreads();
      }
    } // q
  }
  __syncthreads();
  // 5. gated RMSNorm of o_i (warp i: |o_i|^2)
  if (warp < NTK) {
    float sq = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      float const x = ov[warp * D + lane * 4 + i];
      sq += x * x;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      sq += __shfl_xor_sync(0xffffffffu, sq, o);
    }
    if (lane == 0) {
      onrm[warp] = sq;
    }
  }
  __syncthreads();
#pragma unroll 1
  for (int e = tid; e < NTK * D; e += 256) { // e % D = tid % D: onw's dim
    int const i = e / D;
    float const rn = rsqrtf(onrm[i] / (float)D + norm_eps);
    ov[e] = ov[e] * rn * onw * kda_sigmoid(og[e]); // in place: ov[e] is
                                                   // this thread's alone
  }
  __syncthreads();
  // o in whole 16-B stores of 8 bf16: an o_proj GEMM may poll it per 16 B
  // (gemm_tile.cuh POLLED; the node's buffer is 0xFF before each launch)
  for (int v = tid; v < NTK * D / 8; v += 256) {
    float4 const *y = reinterpret_cast<float4 const *>(ov + 8 * v);
    int const tq = v / (D / 8); // RP > 1: an idle request's o = 0
    bool const idle = RP > 1 && !liveq[tq];
    *reinterpret_cast<uint4 *>(out + (size_t)(t0 + tq) * HLD + h * D +
                               (v % (D / 8)) * 8) =
        idle ? make_uint4(0u, 0u, 0u, 0u) : pack_bf16x8(y[0], y[1]);
  }
  __syncthreads();
  if (tid == 0) {
    cnt_add(done_cnt, 1); // release: o (and the conv state)
  }
  // 6. the state, after the release: only the next step reads it, from the
  // registers
  if (RP == 1 || liveq[RP - 1]) { // the last request's (RP > 1: the others
                                  // went to their slots in step 4)
    kda_state_store<D>(rec + ((size_t)rows[t0 + NTK - 1] * HL + h) * D * D, s,
                       b, g);
  }
}

// kda_decode task (x, head y); grid (T / RP, HL), RP requests per task (one
// token each). PROJ: the merged projection's gemm_tile node (COMBINE_SLOTS,
// its params {mode, K, N = NPROJ, polled}); PARAMS {HL, D, gate lower bound,
// o_norm eps (fp32 bits), 1 (the conv layout [NR][3][3 HL D])}; BUF_SLOTS
// {conv weight [3 HL D][4] fp32, f_b [HL D][D] bf16, g_b [HL D][D] bf16, small
// (dt_bias [HL D], A_log [HL], o_norm weight [D]) fp32, conv state bf16,
// recurrent state slots [NS][HL][D][D] fp32, rows [T] int32 (the state slot of
// each request)}
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS,
          class PROJ, class... IN>
__device__ __forceinline__ void run_kda_decode(Maps const &, G const &g,
                                               KernelLocals &L,
                                               StaticTask const &tk) {
  constexpr int HL = PARAMS::v[0], D = PARAMS::v[1], RP = T / SELF::x;
  static_assert(PROJ::buf >= 0 && PROJ::v[0] == COMBINE_SLOTS &&
                    PROJ::v[3] == 0,
                "kda_decode: its input is a gemm_tile node's output in slots");
  static_assert(SELF::x * RP == T && SELF::y == HL && SELF::buf >= 0 &&
                    SELF::counter >= 0,
                "kda_decode: grid (requests, HL), its output o, its counter");
  static_assert(PARAMS::v[4] == 1,
                "kda_decode: the conv layout [NR][3][3 HL D]");
  kda_decode_task<PROJ::y, PROJ::v[2], HL, D, RP>(
      buf_at<float const>(g, PROJ::buf),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 4>()),
      buf_at<float>(g, slot_at<BUF_SLOTS, 5>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 6>()),
      buf_at<__nv_bfloat16>(g, SELF::buf), cnt_now(g) + SELF::counter, tk.x,
      tk.y, __int_as_float(PARAMS::v[2]), __int_as_float(PARAMS::v[3]),
      L.sm STAGE_STAMP_ARG(L.stage_stamps));
}
// dynamic shared memory: the f_b / g_b rows and the scratch, from the ring
// area's base (T = 64, RP 8: past the ring area's end)
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS,
          class... IN>
constexpr int smem_kda_decode() {
  constexpr int need = kda_smem_bytes<T / SELF::x>(); // RP requests per task
  static_assert(smem_launch_bytes(need) <= 232448,
                "kda_decode: its shared memory in the launch's 227 KB");
  return need > RING_BYTES ? need : RING_BYTES;
}

} // namespace static_mk
