// tasks/kda_decode.cuh -- KDA (Kimi Delta Attention, the linear attention of
// GLM-5.3-Flash) as vLLM's glm5next kda.py + fused_recurrent_kda compute it, on
// SGLang's KDA pool (per state row the conv window [3][3 HL D] bf16, oldest
// first, and the state [HL][V][K] fp32). Task (r, h) of grid (T / (NT RP), HL,
// 1): head h of
//   a decode step (NT = 1): requests RP r .. RP r + RP - 1, one token each
//   (RP = 1, or 2 / 4: T = 16 / 32 in 128 tasks), request q's state row
//   rows[RP r + q]; the window and the state are updated in place
//   SERVE (NT > 1): a speculative verify step, request r's NT tokens in order
//   (below)
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
// SERVE (params[5] = 1; a serving framework's speculative verify step, NT > 1:
// SGLang's KDA target-verify contract with ReplaySSM): every request has NV =
// nvalid[0] (<= NT) draft tokens; request r's are rows r NV .. r NV + NV - 1
// (the framework's verify batch as it is: R requests x NV rows, no gather),
// the rows R NV .. T - 1 are idle (o = 0; task r zeroes NT - NV of them). The
// conv window (3 entries) and the state are read from the framework's pools
// at slot rows[r] (one entry per request) and NOT written back. Per token
// t < NV the kernel writes what the framework's commit replays after the
// verdict (kda_replayssm_spec_decode.py's exact fold: the same decay, delta
// and rank-1 steps as this kernel's, so the folded state of the accepted
// prefix is the state this kernel had after that token):
//   ring_v [slots][HL][L][D]   v after the conv and SiLU (bf16, before the
//                              delta rule)
//   ring_k [slots][HL][L][D]   k after the conv and SiLU (bf16, before the L2
//                              norm)
//   ring_g [slots][HL][L][D]   the gate's natural-log decay lb sigmoid(e^A_log
//                              (f_b f_a + dt_bias)) fp32
//   ring_beta [slots][HL][L]   sigmoid(b) fp32
//   at position t (L = params[7], the ring length); and the conv window after
//   token t to the framework's per-token windows: icw [rows][DRAFT][3][3 HL D]
//   bf16 at row icw_row[r] (DRAFT = params[6]: the framework's draft tokens,
//   NV <= DRAFT; SGLang's intermediate_conv_window, causal_conv1d's
//   SAVE_INTERMEDIATE): the window after token t is its entries (t, j) =
//   x_{t - 2 + j}, j = 0, 1, 2 (the pool window's 2 newest inputs where
//   t - 2 + j < 0, else that token's raw input), t < NV
// The verify build's accepted tensor (buffer slot 7) is not read.
// SERVE computes the NV tokens' outputs in one chunk instead of the delta
// rule token by token (the state is read only, so it is never formed): with
// G_t = sum_{s <= t} g_s (the cumulative log decay, per key dim), the
// decayed vectors k~_t = e^{G_t} k_t, q~_t = e^{G_t} q_t and the masked
// products A[s][t] = sum_d k_s k_t e^{G_t - G_s} (s < t), B[s][t] = sum_d k_s
// q_t e^{G_t - G_s} (s <= t), the sequential rule S_t = S_{t-1} diag(e^{g_t})
// + u_t k_t^T, u_t = beta_t (v_t - S_{t-1} diag(e^{g_t}) k_t), o_t = S_t q_t
// unrolls to u_t = beta_t (v_t - S_0 k~_t - sum_{s < t} A[s][t] u_s) and
// o_t = S_0 q~_t + sum_{s <= t} B[s][t] u_s: one pass of S_0 over the 2 NV
// vectors (the thread's part of S_0 in registers, 8 lanes reduced), then a
// per-row forward substitution (DRAFT x DRAFT). Token by token the same
// arithmetic in the same order per token, so the results match the
// sequential form to fp32 rounding.
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
__device__ __forceinline__ void
    kda_proj4(float const *proj, int t0, int col, float4 (&x)[N]) {
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
    x[i] = make_float4(
        kda_bf16r(acc.x), kda_bf16r(acc.y), kda_bf16r(acc.z), kda_bf16r(acc.w));
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
__device__ __forceinline__ void kda_mma16816(uint32_t const (&a)[4],
                                             uint32_t const (&b)[2],
                                             float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
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
__device__ __forceinline__ void kda_rule(float (&s)[64],
                                         float const *eg,
                                         float const *kn,
                                         float const *qn,
                                         float4 xv,
                                         float beta,
                                         int g,
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
__device__ __forceinline__ void
    kda_state_load(float (&s)[64], float const *m, int b, int g) {
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
__device__ __forceinline__ void
    kda_state_store(float *m, float const (&s)[64], int b, int g) {
#pragma unroll
  for (int r = 0; r < 4; r++) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      *reinterpret_cast<float4 *>(m + (4 * b + r) * LD + 32 * j + 4 * g) =
          make_float4(s[16 * r + 4 * j],
                      s[16 * r + 4 * j + 1],
                      s[16 * r + 4 * j + 2],
                      s[16 * r + 4 * j + 3]);
    }
  }
}

constexpr int KDA_WROW = 128 + 8; // bf16 per f_b / g_b row in shared memory:
                                  // rows 16 B apart in the banks
constexpr int KDA_SROW = 128 + 4; // fp32 per row of a state in shared memory
// shared memory: the f_b / g_b rows [2 D][KDA_WROW] bf16; f_a and g_a of the
// tokens ([8][KDA_WROW] bf16 each: the MMA's B, 8 token columns); per token
// q, k, v, the decay, the output gate and o ([NT][D] fp32 each); 32 small
// values; when a task has more than one token, [D][KDA_SROW] fp32 (SERVE: the
// chunk form's scratch; RP > 1: the next request's state)
template <int NT, int RP = 1> // NT: the task's tokens; RP: its requests
__host__ __device__ constexpr int kda_smem_bytes() {
  return (2 * 128 + 2 * 8) * KDA_WROW * 2 + (6 * NT * 128 + 32) * 4 +
         (NT > 1 ? 128 * KDA_SROW * 4 : 0) + (RP - 1) * 96 * 12 * 2;
}
// SERVE: the chunk form's scratch in the initial state's area: k~, q~, G, P, O,
// u [DRAFT][D] and A, B [DRAFT][DRAFT] fp32
__host__ __device__ constexpr int kda_serve_scratch_floats(int draft) {
  return 6 * draft * 128 + 2 * draft * draft;
}

struct KdaServe { // SERVE: the framework's buffers (the header comment)
  int const *nvalid, *icw_row;
  __nv_bfloat16 *icw, *ring_v, *ring_k;
  float *ring_g, *ring_beta;
};

template <int NSPLIT,
          int NPROJ,
          int HL,
          int D,
          int NT,
          int SERVE = 0,
          int DRAFT = 0,
          int RL = 0,
          int RP = 1>
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
                    int r,
                    int h,
                    float lower_bound,
                    float norm_eps,
                    KdaServe const sv,
                    char *scratch STAGE_STAMP_PARAM) {
  constexpr int HLD = HL * D;
  constexpr int BO = 3 * HLD, FAO = BO + HL, GAO = FAO + D;
  // the conv window of a state row: 3 entries [3][3 HL D] (SERVE: read only;
  // the per-token windows go to icw); element (channel c, entry e): c CS + e ES
  constexpr int CL = 3, CS = 1, ES = 3 * HLD;
  static_assert(
      SERVE ? NT > 1 && DRAFT >= 1 && DRAFT <= NT && RL >= DRAFT : NT == 1,
      "kda_decode: one token per request in a decode step; SERVE: NT "
      "> 1 tokens per request, DRAFT <= NT draft tokens, a ring of at "
      "least DRAFT");
  static_assert(
      !SERVE || kda_serve_scratch_floats(DRAFT) <= 128 * KDA_SROW,
      "kda_decode SERVE: the chunk form's scratch in the state's area");
  static_assert(D == 128,
                "kda_decode: head dim 128 (256 threads: 32 row blocks x 8)");
  static_assert(
      HL % 4 == 0 && NPROJ % 4 == 0 && NPROJ >= GAO + D,
      "kda_decode: the merged projection [q | k | v | b | f_a | g_a]");
  static_assert(NT >= 1 && NT <= 8, "kda_decode: 1 .. 8 tokens per request");
  // RP: requests per task (2 / 4: a decode step of T = 16 / 32 in 128 tasks,
  // one wave of the SMs; the head's f_b / g_b rows and steps 1 .. 3 shared,
  // each request's conv window and state its own, the delta rules one after
  // the other: request q + 1's state copied to shared memory while request q's
  // rule runs; requests 1 .. RP - 1's conv windows in shared memory from step
  // 0). NTK: the task's tokens (rows t0 .. t0 + NTK - 1)
  constexpr int NTK = NT * RP;
  static_assert(RP == 1 || ((RP == 2 || RP == 4) && !SERVE),
                "kda_decode: 2 or 4 requests per task only in a decode step");
  int const tid = threadIdx.x;
  constexpr int R = T / NT; // requests per launch
  // SERVE: the requests' draft tokens nv (all alike), request r's rows from
  // r nv, its slot rows[r]; else rows r NT .. and the slot of its first token
  int const nv = SERVE ? min(sv.nvalid[0], NT) : NT;
  int const t0 = SERVE ? r * nv : r * NTK; // the task's first row
  int const slot = rows[SERVE ? r : t0];
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
  // the output row of the request's local row i (SERVE, i >= nv: this task's
  // share of the idle rows after every request's tokens)
  auto out_row = [&](int i) {
    return (!SERVE || i < nv) ? t0 + i : R * nv + r * (NT - nv) + (i - nv);
  };
  int const warp = tid >> 5, lane = tid & 31;
  // [2 D][KDA_WROW]: the f_b rows of head h, then its g_b rows; [8][KDA_WROW]:
  // f_a of the tokens, then [8][KDA_WROW] g_a (rows NT .. 7 unused)
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
  float *s0 = onrm + 8;       // [D][KDA_SROW]: SERVE's scratch (below); RP > 1:
                        // request q + 1's state while request q's rule runs
  // RP > 1: requests 1 .. RP - 1's conv windows, [RP - 1][96 threads][12] bf16
  __nv_bfloat16 *cwin = reinterpret_cast<__nv_bfloat16 *>(s0 + D * KDA_SROW);
  // an idle request (a padded batch: its slot < 0): o = 0, no state touched
  if (!any) {
    for (int v = tid; v < NTK * D / 8; v += 256) { // whole 16-B stores (as o)
      *reinterpret_cast<uint4 *>(out + (size_t)out_row(v / (D / 8)) * HLD +
                                 h * D + (v % (D / 8)) * 8) =
          make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();
    if (tid == 0) {
      cnt_add(done_cnt, 1);
    }
    return;
  }
  // SERVE: the request's row of the framework's per-token conv windows
  __nv_bfloat16 *icw = nullptr;
  if constexpr (SERVE) {
    icw = sv.icw + (size_t)sv.icw_row[r] * (DRAFT * 3) * (3 * HLD);
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
  float cs[12], cs0[12]; // cs0: as loaded (the conv shifts cs; SERVE's dry pass
                         // restarts from cs0)
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
        cs0[3 * k + e] =
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
  // SERVE: the chunk form's scratch (3c, 4) in s0's area
  constexpr int DR = SERVE ? DRAFT : 1;
  float *kt = s0, *qt = kt + DR * D, *cG = qt + DR * D; // k~, q~, G [DRAFT][D]
  float *Pm = cG + DR * D, *Om = Pm + DR * D;           // S_0 k~_t, S_0 q~_t
  float *us = Om + DR * D;                              // u_t [DRAFT][D]
  float *Am = us + DR * D, *Bm = Am + DR * DR;          // [DRAFT][DRAFT] (s, t)
  // Steps 1 .. 4 run in one pass; SERVE first runs pass 0: step 4 (the chunk
  // form) dry, on one token (nvp), on whatever the shared memory holds, no
  // store to global memory, while the projection is on its way: the code of a
  // task runs once per SM, and cold instruction fetch costs ~7 ns an
  // instruction; the dry pass loads the code (as topk_route's dry pass). A
  // longer dry pass (steps 2 .. 5 too) measured slower: it then ends after the
  // projection has landed. One copy of the code (unroll 1); real: pass 1
#pragma unroll 1
  for (int pass = SERVE ? 0 : 1; pass < 2; pass++) {
    bool const real = pass == 1;
    int const nvp = real ? nv : 1; // the tokens of this pass
#pragma unroll
    for (int k = 0; k < 12; k++) {
      cs[k] = cs0[k];
    }
    if (real) {
      // 1. projection parts -> bf16, all NT tokens: q | k | v (threads 0 .. 95,
      // 4 channels each; raw, to their shared-memory rows), f_a (96 .. 127) and
      // g_a (128 .. 159) (bf16 values, stored exactly), beta (160 .. 160 + NT -
      // 1)
      if (tid < 160) {
        int const col =
            tid < 96 ? ch : (tid < 128 ? FAO : GAO) + (tid & 31) * 4;
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
        if constexpr (SERVE) {
          if (tid - 160 < nv && tid - 160 < RL) {
            sv.ring_beta[((size_t)slot * HL + h) * RL + tid - 160] =
                beta[tid - 160];
          }
        }
      }
      // 2. short conv + SiLU of the thread's 4 channels, token by token, in
      // place (the window keeps the raw inputs in registers); the conv state's
      // new entries
      if (tid < 96) {
        float *dst = (seg == 0 ? xq : (seg == 1 ? xk : xv)) + j;
#pragma unroll 1
        for (int q = 0; q < RP;
             q++) { // RP > 1: request q, its window and conv row
          if (q > 0) {
#pragma unroll
            for (int k = 0; k < 12; k++) {
              cs[k] = __bfloat162float(cwin[((q - 1) * 96 + tid) * 12 + k]);
            }
          }
          bool const wr = RP == 1 || liveq[q]; // its conv row written
          // the new window (entry e = x_{e - 2}: the pool's 2 newest inputs,
          // then the tokens' raw inputs) to the conv state (a decode step: its
          // 3 entries) or to the framework's per-token windows (SERVE: the
          // window after token t < nv is (t, j) = entry t + j, j = 0, 1, 2)
          __nv_bfloat16 *wd = SERVE ? icw + ch : (q == 0 ? st : stq(q));
          constexpr int WS = CS, WE = ES;
          auto put = [&](int e, int k, float v) {
            if (!real || !wr) {
              return;
            }
            __nv_bfloat16 const b = __float2bfloat16(v);
            if constexpr (SERVE) {
#pragma unroll
              for (int j = 0; j < 3; j++) {
                int const t = e - j;
                if (t >= 0 && t < nv) {
                  wd[k * WS + (t * 3 + j) * WE] = b;
                }
              }
            } else {
              wd[k * WS + e * WE] = b;
            }
          };
#pragma unroll
          for (int k = 0; k < 4; k++) {
            put(0, k, cs[3 * k + 1]);
            put(1, k, cs[3 * k + 2]);
          }
#pragma unroll 1
          for (int i = 0; i < nvp; i++) {
            int const tq = q * NT + i; // the task's token
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
              put(2 + i, k, xr[k]);
              cs[3 * k] = cs[3 * k + 1];
              cs[3 * k + 1] = cs[3 * k + 2];
              cs[3 * k + 2] = xr[k];
            }
            *reinterpret_cast<float4 *>(dst + tq * D) =
                make_float4(y[0], y[1], y[2], y[3]);
            if constexpr (SERVE) { // the ring: k and v after the conv, before
                                   // the norm and the delta rule
              if (real && seg > 0 && i < RL) {
                __nv_bfloat16 *rg = (seg == 1 ? sv.ring_k : sv.ring_v) +
                                    (((size_t)slot * HL + h) * RL + i) * D + j;
                __nv_bfloat162 *r2 = reinterpret_cast<__nv_bfloat162 *>(rg);
                r2[0] = __floats2bfloat162_rn(y[0], y[1]);
                r2[1] = __floats2bfloat162_rn(y[2], y[3]);
              }
            }
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
          kda_ldsm_x2(b,
                      xb + (lane & 7) * KDA_WROW + kk + ((lane >> 3) & 1) * 8);
#pragma unroll
          for (int mt = 0; mt < 2; mt++) {
            uint32_t a[4];
            kda_ldsm_x4(a,
                        w_s + (32 * warp + 16 * mt + (lane & 15)) * KDA_WROW +
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
            int const rr =
                32 * (warp & 3) + 16 * mt + (lane >> 2) + 8 * (e >> 1);
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
      // dim r and k_i[r] (L2-normalised). SERVE (the chunk form, the header
      // comment): thread r < D also k_i[r], the cumulative log decay G_i[r] and
      // the decayed k~, q~ (in s0's area: kt, qt, cG [DRAFT][D]), of the nv
      // tokens
#pragma unroll 1
      for (int i = 0; i < (SERVE ? nvp : NTK); i++) {
        int const rr = tid & (D - 1);
        if (tid < D) {
          float const g = kda_bf16r(eg[i * D + rr]); // vLLM: a bf16 GEMM output
          float const z = expf(alog) * (g + dtb);
          float const lg = lower_bound / (1.0f + expf(-z)); // the log decay
          float const qn = xq[i * D + rr] / sqrtf(nrm[2 * i] + 1e-6f) *
                           (1.0f / sqrtf((float)D));
          if constexpr (SERVE) {
            if (real && i < RL) {
              sv.ring_g[(((size_t)slot * HL + h) * RL + i) * D + rr] = lg;
            }
            float const G = (i > 0 ? cG[(i - 1) * D + rr] : 0.f) + lg;
            float const e = expf(G);
            float const kn = xk[i * D + rr] / sqrtf(nrm[2 * i + 1] + 1e-6f);
            cG[i * D + rr] = G;
            xq[i * D + rr] = qn;
            xk[i * D + rr] = kn;
            qt[i * D + rr] = qn * e;
            kt[i * D + rr] = kn * e;
          } else {
            eg[i * D + rr] = expf(lg);
            xq[i * D + rr] = qn;
          }
        } else {
          og[i * D + rr] = kda_bf16r(og[i * D + rr]);
          if constexpr (!SERVE) {
            xk[i * D + rr] = xk[i * D + rr] / sqrtf(nrm[2 * i + 1] + 1e-6f);
          }
        }
      }
      __syncthreads();
      if (real && tid == 0) {
        STAGE_STAMP_AT(1)
      } // timing build: the gates and the normalised q, k
    }   // real: steps 1 .. 3
    if constexpr (SERVE) {
      // 4. the chunk form (the header comment) on the thread's part of S_0 (in
      // registers since step 0), in compact loops (the dry pass above). 4a.
      // A[s][t], B[s][t] of the pairs s <= t < nv: warp w the pairs w, w + 8,
      // .. (t outer), a lane 4 key dims
      constexpr int NPAIR = DRAFT * (DRAFT + 1) / 2;
      for (int p = warp; p < NPAIR; p += 8) {
        int t = 0;
        while ((t + 1) * (t + 2) / 2 <= p) {
          t++;
        }
        int const s2 = p - t * (t + 1) / 2;
        if (t >= nvp) {
          continue; // warp-uniform
        }
        float4 const ks =
            *reinterpret_cast<float4 const *>(xk + s2 * D + 4 * lane);
        float4 const kt4 =
            *reinterpret_cast<float4 const *>(xk + t * D + 4 * lane);
        float4 const qt4 =
            *reinterpret_cast<float4 const *>(xq + t * D + 4 * lane);
        float4 const gs =
            *reinterpret_cast<float4 const *>(cG + s2 * D + 4 * lane);
        float4 const gt =
            *reinterpret_cast<float4 const *>(cG + t * D + 4 * lane);
        float const e0 = expf(gt.x - gs.x), e1 = expf(gt.y - gs.y),
                    e2 = expf(gt.z - gs.z), e3 = expf(gt.w - gs.w);
        float const w0 = ks.x * e0, w1 = ks.y * e1, w2 = ks.z * e2,
                    w3 = ks.w * e3;
        float av =
            s2 < t ? w0 * kt4.x + w1 * kt4.y + w2 * kt4.z + w3 * kt4.w : 0.f;
        float bv = w0 * qt4.x + w1 * qt4.y + w2 * qt4.z + w3 * qt4.w;
#pragma unroll
        for (int m = 16; m > 0; m >>= 1) {
          av += __shfl_xor_sync(0xffffffffu, av, m);
          bv += __shfl_xor_sync(0xffffffffu, bv, m);
        }
        if (lane == 0) {
          Am[s2 * DR + t] = av;
          Bm[s2 * DR + t] = bv;
        }
      }
      // 4b. S_0 k~_t and S_0 q~_t (t < nv): token by token, the thread's 4
      // rows x 16 keys, the 8 lanes of a row block reduced, lane 0's 4 rows to
      // Pm / Om (one copy of the loop body: warm after the first token)
      // two vectors an iteration (k~_m and q~_m: independent chains)
#pragma unroll 1
      for (int m = 0; m < nvp; m++) {
        float d[2][4] = {};
#pragma unroll
        for (int j = 0; j < 4; j++) {
          float4 const u0 =
              *reinterpret_cast<float4 const *>(kt + m * D + 32 * j + 4 * g);
          float4 const u1 =
              *reinterpret_cast<float4 const *>(qt + m * D + 32 * j + 4 * g);
          float const uu[2][4] = {{u0.x, u0.y, u0.z, u0.w},
                                  {u1.x, u1.y, u1.z, u1.w}};
#pragma unroll
          for (int rr = 0; rr < 4; rr++) {
#pragma unroll
            for (int c = 0; c < 4; c++) {
              float const sv_ = s[16 * rr + 4 * j + c];
              d[0][rr] += sv_ * uu[0][c];
              d[1][rr] += sv_ * uu[1][c];
            }
          }
        }
#pragma unroll
        for (int mask = 1; mask < 8; mask <<= 1) {
#pragma unroll
          for (int rr = 0; rr < 4; rr++) {
            d[0][rr] += __shfl_xor_sync(0xffffffffu, d[0][rr], mask);
            d[1][rr] += __shfl_xor_sync(0xffffffffu, d[1][rr], mask);
          }
        }
        if (g == 0) {
          *reinterpret_cast<float4 *>(Pm + m * D + 4 * b) =
              make_float4(d[0][0], d[0][1], d[0][2], d[0][3]);
          *reinterpret_cast<float4 *>(Om + m * D + 4 * b) =
              make_float4(d[1][0], d[1][1], d[1][2], d[1][3]);
        }
      }
      __syncthreads();
      // 4c. thread v < D: value row v of every token: u_t = beta_t (v_t - P_t -
      // sum_{s < t} A[s][t] u_s), o_t = O_t + sum_{s <= t} B[s][t] u_s (bf16:
      // vLLM's core_attn_out); u in shared memory (us, the thread's own column)
      if (tid < D) {
#pragma unroll 1
        for (int t = 0; t < nvp; t++) {
          float a = xv[t * D + tid] - Pm[t * D + tid];
#pragma unroll 1
          for (int s2 = 0; s2 < t; s2++) {
            a -= Am[s2 * DR + t] * us[s2 * D + tid];
          }
          float const ut = beta[t] * a;
          us[t * D + tid] = ut;
          float o = Om[t * D + tid] + Bm[t * DR + t] * ut;
#pragma unroll 1
          for (int s2 = 0; s2 < t; s2++) {
            o += Bm[s2 * DR + t] * us[s2 * D + tid];
          }
          ov[t * D + tid] = kda_bf16r(o);
        }
      }
      __syncthreads();
    } else {
      // 4. the delta rule on the thread's part of S (in registers since step
      // 0), token by token (RP 2: request by request: the first one's state to
      // its slot, the second one's from shared memory)
#pragma unroll 1
      for (int q = 0; q < RP; q++) {
        if (q > 0) { // request q - 1's state to its slot, request q's from s0
          if (liveq[q - 1]) {
            kda_state_store<D>(
                rec + ((size_t)slotq[q - 1] * HL + h) * D * D, s, b, g);
          }
          kda_state_load<KDA_SROW>(s, s0, b, g);
          if (q + 1 < RP) { // s0 read by every thread: request q + 1's into it
            __syncthreads();
            state_to_s0(q + 1);
          }
        }
#pragma unroll 1
        for (int i = 0; i < NT; i++) {
          int const tq = q * NT + i; // the task's token
          float o[4];
          kda_rule(s,
                   eg + tq * D,
                   xk + tq * D,
                   xq + tq * D,
                   *reinterpret_cast<float4 const *>(xv + tq * D + 4 * b),
                   beta[tq],
                   g,
                   o);
          if (g == 0) { // vLLM's core_attn_out is bf16
            *reinterpret_cast<float4 *>(ov + tq * D + 4 * b) =
                make_float4(kda_bf16r(o[0]),
                            kda_bf16r(o[1]),
                            kda_bf16r(o[2]),
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
    if (!real) {
      continue; // the dry pass ends here (every thread)
    }
  }                      // pass
  if constexpr (SERVE) { // the idle tokens: o = 0
    for (int e = tid; e < (NT - nv) * D; e += 256) {
      ov[nv * D + e] = 0.f;
      og[nv * D + e] = 0.f;
    }
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
    bool const idle = RP > 1 && !liveq[tq / NT];
    *reinterpret_cast<uint4 *>(out + (size_t)out_row(tq) * HLD + h * D +
                               (v % (D / 8)) * 8) =
        idle ? make_uint4(0u, 0u, 0u, 0u) : pack_bf16x8(y[0], y[1]);
  }
  __syncthreads();
  if (tid == 0) {
    cnt_add(done_cnt, 1); // release: o (and the conv state)
  }
  // 6. the state, after the release: only the next step reads it, from the
  // registers. (SERVE: the framework's commit folds the accepted tokens from
  // the ring: nothing written here.)
  if constexpr (SERVE) {
    return;
  }
  if (RP == 1 || liveq[RP - 1]) { // the last request's (RP > 1: the others
                                  // went to their slots in step 4)
    kda_state_store<D>(
        rec + ((size_t)rows[t0 + NTK - 1] * HL + h) * D * D, s, b, g);
  }
}

// kda_decode task (x, head y); grid (T / NTASK, HL), NTASK tokens per task (a
// decode step: RP = NTASK requests; SERVE: one request's NT = NTASK tokens).
// PROJ: the merged projection's gemm_tile node (COMBINE_SLOTS, its params
// {mode, K, N = NPROJ, polled}); PARAMS {HL, D, gate lower bound, o_norm eps
// (fp32 bits), 1 (the conv layout [NR][3][3 HL D])[, SERVE (0 / 1), DRAFT, RL
// (the header comment)]}; BUF_SLOTS {conv weight [3 HL D][4] fp32, f_b [HL
// D][D] bf16, g_b [HL D][D] bf16, small (dt_bias [HL D], A_log [HL], o_norm
// weight [D]) fp32, conv state bf16, recurrent state slots [NS][HL][D][D] fp32,
// rows [T] int32 (the state slot of each request), then with SERVE accepted
// [R] int32 (not read), nvalid [R] int32 (entry 0: every request's draft
// tokens), icw_row [R] int32, icw bf16, ring_v, ring_k bf16, ring_g, ring_beta
// fp32}
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
  constexpr int HL = PARAMS::v[0], D = PARAMS::v[1], NTASK = T / SELF::x;
  // tokens per task: a verify step's NT tokens of one request (its slots go on
  // past 7), or RP requests' one token each (a decode step: 7 slots)
  constexpr int NSLOT = (int)(sizeof(BUF_SLOTS::v) / sizeof(int)) - 1;
  constexpr int RP = NTASK > 1 && NSLOT == 7 ? NTASK : 1, NT = NTASK / RP;
  static_assert(PROJ::buf >= 0 && PROJ::v[0] == COMBINE_SLOTS &&
                    PROJ::v[3] == 0,
                "kda_decode: its input is a gemm_tile node's output in slots");
  static_assert(SELF::x * NTASK == T && SELF::y == HL && SELF::buf >= 0 &&
                    SELF::counter >= 0,
                "kda_decode: grid (requests, HL), its output o, its counter");
  // SERVE and its DRAFT, RL: params 5 .. 7 when given (StaticParams<..>::v has
  // one entry more than the params: v[5] is 0 with 5 params; 6 and 7 are
  // beyond it then)
  constexpr int NP = (int)(sizeof(PARAMS::v) / sizeof(int)) - 1;
  constexpr int SERVE = NP > 5 ? PARAMS::v[5] : 0;
  constexpr int DRAFT = NP > 6 ? PARAMS::v[6] : 0,
                RL = NP > 7 ? PARAMS::v[7] : 0;
  static_assert(PARAMS::v[4] == 1,
                "kda_decode: the conv layout [NR][3][3 HL D]");
  KdaServe sv = {};
  if constexpr (SERVE) {
    sv.nvalid = buf_at<int const>(g, slot_at<BUF_SLOTS, 8>());
    sv.icw_row = buf_at<int const>(g, slot_at<BUF_SLOTS, 9>());
    sv.icw = buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 10>());
    sv.ring_v = buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 11>());
    sv.ring_k = buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 12>());
    sv.ring_g = buf_at<float>(g, slot_at<BUF_SLOTS, 13>());
    sv.ring_beta = buf_at<float>(g, slot_at<BUF_SLOTS, 14>());
  }
  kda_decode_task<PROJ::y, PROJ::v[2], HL, D, NT, SERVE, DRAFT, RL, RP>(
      buf_at<float const>(g, PROJ::buf),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 4>()),
      buf_at<float>(g, slot_at<BUF_SLOTS, 5>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 6>()),
      buf_at<__nv_bfloat16>(g, SELF::buf),
      cnt_now(g) + SELF::counter,
      tk.x,
      tk.y,
      __int_as_float(PARAMS::v[2]),
      __int_as_float(PARAMS::v[3]),
      sv,
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
  // RP as run_kda_decode decides it: a decode step's tokens per task (7 slots)
  // are requests
  constexpr int NTASK = T / SELF::x;
  constexpr int RP =
      NTASK > 1 && (int)(sizeof(BUF_SLOTS::v) / sizeof(int)) - 1 == 7 ? NTASK
                                                                      : 1;
  static_assert(kda_smem_bytes<NTASK, RP>() <= RING_BYTES,
                "kda_decode: its shared memory in the ring area");
  return RING_BYTES;
}

} // namespace static_mk
