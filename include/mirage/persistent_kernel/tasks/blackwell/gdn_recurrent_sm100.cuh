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
#pragma once
// `gdn_slot_is_first_chunk` (the shared step==0 slot predicate) lives in the
// conv task's header; both GDN tasks use it, so pull it in explicitly rather
// than relying on task_header.cuh's include order.
#include "gdn_conv1d_sm100.cuh"
#include "tasks/common/common_header.cuh"

// Gated-DeltaNet recurrence + fused gated RMSNorm/SiLU epilogue, with a
// persistent per-request-slot fp32 recurrent-state pool.  This is the MPK port
// of Qwen3.5's GDN core (docs/qwen35/v1-architecture.md 3.2, row 6 of the 2.2
// op table; docs/qwen35/vllm-graph.md 2.1.4 + 2.1.6).
//
// One task == one (v-head hv, request slot).  It owns the WHOLE per-layer chain
// between the conv output and the out_proj input:
//
//   q,k L2 norm -> q scaling -> gating (g, beta) -> delta-rule state update
//   -> readout o = S q -> gated RMSNorm(o)*w * SiLU(z) -> bf16 out
//
// Math, per token t of the chunk (i_h = hv / (NUM_V_HEADS/NUM_K_HEADS), the GVA
// mapping - two v-heads share one q/k head):
//
//   q    = qkv_c[t, i_h*Dk : +Dk]
//   k    = qkv_c[t, KEY_DIM + i_h*Dk : +Dk]
//   v    = qkv_c[t, 2*KEY_DIM + hv*Dv : +Dv]
//   z    = z[t, hv*Dv : +Dv]
//   q   <- l2norm(q) * Dk^-0.5
//   k   <- l2norm(k)
//   g    = -exp(A_log[hv]) * softplus(a[t,hv] + dt_bias[hv])
//   beta = sigmoid(b[t,hv])
//   S   <- S * exp(g)            (elementwise, BEFORE the dot)
//   delta = (v - S k) * beta
//   S   <- S + delta (x) k
//   o    = S q
//   out[t, hv*Dv : +Dv] = (norm_w * rmsnorm(o)) * silu(z)
//
// NUMERIC TARGET - the rounding order is HF's, not vLLM's and not the
// architecture doc's.  v1-architecture.md 1 pins HF `transformers` as the
// numeric target: where implementations disagree in low-order bits, HF wins.
// Measured against the M2-I3 oracle (real layer-0 checkpoint tensors, see
// `test_gdn_recurrent_oracle.py`) THREE of the decisions below are genuine
// discriminators, not cosmetics:
//
//   1. q/k L2 norm runs in BF16, not fp32.  transformers' `l2norm` is applied
//      to the bf16 q/k BEFORE `torch_recurrent_gated_delta_rule` upcasts to
//      fp32, so every step of it (the squares, the sum, the +eps, the rsqrt and
//      the final scaling) round-trips through bf16.  vLLM's Triton kernel
//      normalizes in fp32 instead.  Using the fp32 order misses 1448 of 4096
//      bf16 elements of `o` and puts 2.3e-2 of absolute error into S.
//   2. `o` is rounded to BF16 before the gated norm sees it -
//      `torch_recurrent_gated_delta_rule` ends with `.to(initial_dtype)`.
//   3. Inside the gated norm the NORMALIZED value is rounded to bf16 BEFORE it
//      is multiplied by the (fp32) norm weight -
//      `Qwen3_5MoeRMSNormGated.forward` does `weight * hidden_states.to(
//      input_dtype)`.  The architecture doc's all-fp32 formula misses 1046 of
//      4096 bf16 elements.
//
// With those three in place the kernel is BIT-EXACT against the oracle on
// `beta`, `decay_g`, `core_attn_out` and `gated_norm_out` for the decode step.
//
// The fp32 STATE is not bit-exact against HF, and cannot be.  Exactly two
// mechanisms account for the difference, both measured, neither a model
// semantic:
//
//   M1  FMA contraction.  nvcc fuses `acc += s * k[c]` into a single-rounding
//       FMA; torch rounds the multiply and the add separately.  (Confirmed by
//       rebuilding this kernel with -fmad=false, which makes it bit-exact
//       against a plain multiply-then-add reference - state included.)
//   M2  Association order of the 128-term fp32 dot product.  torch's
//       `.sum(dim=-2)` order is an internal TensorIterator decomposition that
//       no natural CUDA order reproduces (measured: sequential, pairwise, and
//       2/4/8/16/32/64-lane strided and blocked orders all differ).
//
// A torch reference carrying BOTH mechanisms agrees with this kernel
// bit-for-bit on o, y and the fp32 state, decode and prefill - which is what
// proves there is no third, unexplained source of deviation
// (`test_gdn_recurrent_oracle.py`, the EXACT-ORDER checks).  FMA is kept on: it
// rounds once instead of twice, and empirically lands CLOSER to HF than the
// unfused order does.  The reduction uses the most accurate natural order
// (32-lane strided partials + a shuffle tree).
//
// The resulting deviation is bounded, not compounding: the recurrence
// multiplies S by exp(g) <= 1 every step, so old rounding error is damped
// rather than accumulated.  Measured over a 128-step decode chain the relative
// deviation saturates at ~1.3e-5 of |S|_rms by step ~32 and stays flat, and
// vs the real oracle a decode step lands at 4.8e-7 absolute (3.5e-6 of |S|).
//
// -use_fast_math.  The MEGAKERNEL is compiled with -use_fast_math (see the
// nvcc line in persistent_kernel.py), which rewrites expf/log1pf/rsqrtf to
// their approximate intrinsics; the standalone unit-test build is not.  This
// matters here because several intermediates are deliberately rounded to bf16,
// which turns a ~1e-6 fp32 perturbation into a whole bf16 ULP with probability
// ~ 1e-6 / 2^-8.  Measured A/B over a 256-token carried chain (1,048,576 bf16
// outputs) against the same torch reference:
//
//     no fast-math    o flips 144/1048576 (0.0137%)   S 9.8e-7 of |S|rms
//     -use_fast_math  o flips 165/1048576 (0.0157%)   S 1.5e-6 of |S|rms
//
// i.e. fast-math contributes ~21 extra flips per million; the dominant term is
// the fp32 association order compounding over the chain, which no build flag
// removes.  On a SINGLE decode step from a true state (the oracle case) both
// builds reproduce `core_attn_out` and `gated_norm_out` bit-exactly.  The one
// intermediate fast-math measurably costs is `g` itself (relative ~6.5e-7),
// which is internal.  Left as-is deliberately: the accurate `expf`/`log1pf`
// here are the SAME libdevice functions torch calls, so they are bit-exact
// whenever the build does not rewrite them, and forcing accuracy under
// fast-math would need an undocumented libdevice call for no measurable gain.
//
// Layout contract:
//   qkv_c  [num_tokens, QKV_STRIDE]  bf16  packed [q | k | v], conv output
//   ba     [num_tokens, BA_STRIDE]   bf16  packed [b | a], BA_STRIDE = 2*Hv
//   ad     [2, NUM_V_HEADS]          fp32  row 0 = A_log, row 1 = dt_bias
//   state  [slots, Hv, HEAD_V_DIM, HEAD_K_DIM]  fp32, read AND written
//   z      [num_tokens, Z_STRIDE]    bf16  the output gate
//   norm_w [HEAD_V_DIM]              fp32  ones-init, NOT Gemma (no 1+w)
//   out    [num_tokens, OUT_STRIDE]  bf16
//
// STATE LAYOUT: S is stored [v][k] (v-major, k contiguous), which is the
// TRANSPOSE of HF's `[k_head_dim, v_head_dim]` recurrent_states.  This is a
// deliberate choice: with k contiguous, both reductions (S k and S q) and the
// rank-1 update are row-local, so one warp owns a whole v-row, reads 32
// consecutive floats per step (no bank conflicts, no padding) and needs no
// cross-row communication.  The pool is private to this task pair, so the
// layout is ours to choose; only the oracle test transposes.
//
// Lifecycle (v1-architecture.md 3.3): identical to the conv task.  `zero_state`
// is the kernel-side `runtime_config.step[req] == 0` predicate - the first
// prefill chunk of whichever request holds the slot treats its state as zero
// instead of loading it, and writes the updated state back unconditionally, so
// slot reuse re-zeros implicitly.  A slot with Q_LEN == 0 (parked) returns
// before touching its state.
//
// Parallelism: grid = (NUM_V_HEADS, max_num_batched_requests, 1) per
// v1-architecture.md 2.2 row 6 - `task_metadata.kv_idx` = the v-head,
// `task_metadata.request_id` = the slot.  blockIdx is NEVER used for identity;
// it is the worker id.  Unlike the conv FIR, the recurrence is strictly
// sequential in t, so a chunk's tokens cannot be split - the parallel axes are
// (head, slot) only.

namespace kernel {

// Block-wide fp32 sum.  The LEADING barrier is load-bearing: `red` is reused by
// every reduction in the token loop, so no thread may write it while another is
// still reading the previous result.
__device__ __forceinline__ float gdn_block_sum(float acc, float *red) {
  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;
  int const num_warps = (blockDim.x + 31) >> 5;
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) {
    acc += __shfl_xor_sync(0xffffffff, acc, m);
  }
  __syncthreads();
  if (lane == 0) {
    red[warp] = acc;
  }
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < num_warps; w++) {
    total += red[w];
  }
  return total;
}

// transformers' `l2norm`, run in the tensor's OWN dtype (bf16) exactly as HF
// does, then handed to the caller already widened to fp32 (and optionally
// scaled - the recurrence multiplies q by Dk^-0.5 right after the fp32 cast).
//
//   inv = rsqrt((x*x).sum(-1) + eps)   x <- x * inv
//
// On a bf16 tensor torch rounds to bf16 after EACH of those ops while
// accumulating the sum itself in fp32; that is what the casts below reproduce.
template <typename T, int DIM>
__device__ __forceinline__ void gdn_l2norm_bf16(T const *__restrict__ src,
                                                float *__restrict__ dst,
                                                float *__restrict__ red,
                                                float post_scale,
                                                float eps) {
  int const tid = threadIdx.x;
  int const nthreads = blockDim.x;
  float acc = 0.0f;
  for (int i = tid; i < DIM; i += nthreads) {
    float x = static_cast<float>(src[i]);
    // `x * x` on a bf16 tensor rounds every square back to bf16.
    acc += static_cast<float>(T(x * x));
  }
  // `.sum(-1)` accumulates in fp32 but RETURNS bf16.
  float const sum_bf = static_cast<float>(T(gdn_block_sum(acc, red)));
  float const eps_bf = static_cast<float>(T(sum_bf + eps));
  float const inv = static_cast<float>(T(rsqrtf(eps_bf)));
  for (int i = tid; i < DIM; i += nthreads) {
    float const xn = static_cast<float>(T(static_cast<float>(src[i]) * inv));
    dst[i] = xn * post_scale;
  }
  __syncthreads();
}

template <typename T,
          int NUM_V_HEADS,
          int NUM_K_HEADS,
          int HEAD_K_DIM,
          int HEAD_V_DIM,
          int QKV_STRIDE,
          int BA_STRIDE,
          int Z_STRIDE,
          int OUT_STRIDE>
__device__ __forceinline__ void
    gdn_recurrent_sm100_task_impl(void const *qkv_ptr,
                                  void const *ba_ptr,
                                  void const *alog_dtbias_ptr,
                                  void *state_ptr,
                                  void const *z_ptr,
                                  void const *norm_w_ptr,
                                  void *out_ptr,
                                  int hv,
                                  int q_len,
                                  bool zero_state,
                                  // Optional: the pre-epilogue readout `o`,
                                  // laid out like `out`.  The oracle dumps it
                                  // (`gdn.core_attn_out`) but the fused task
                                  // otherwise consumes it in registers, so this
                                  // is how the unit test observes it.  Left
                                  // null by the generated task code.
                                  void *o_debug_ptr = nullptr) {
  static_assert(NUM_V_HEADS % NUM_K_HEADS == 0,
                "GVA needs an integer v-heads-per-k-head ratio");
  static_assert(HEAD_K_DIM % 32 == 0,
                "one warp walks a k-row 32 lanes at a time");
  static_assert(BA_STRIDE >= 2 * NUM_V_HEADS, "ba packs [b | a]");
  constexpr int KEY_DIM = NUM_K_HEADS * HEAD_K_DIM;
  constexpr int VAL_BASE = 2 * KEY_DIM;
  constexpr int STATE_ELEMS = HEAD_V_DIM * HEAD_K_DIM;
  constexpr int K_UNROLL = HEAD_K_DIM / 32;
  constexpr float EPS = 1e-6f;
  // Softplus threshold: F.softplus(beta=1, threshold=20) passes x through
  // unchanged above 20 rather than evaluating log1p(exp(x)).
  constexpr float SOFTPLUS_THRESHOLD = 20.0f;

  // An inactive slot (Q_LEN == 0 from qo_indptr) must not touch its state, so a
  // parked request's recurrent state survives untouched.
  if (q_len <= 0) {
    return;
  }

  int const tid = threadIdx.x;
  int const nthreads = blockDim.x;
  int const lane = tid & 31;
  int const warp = tid >> 5;
  int const num_warps = nthreads >> 5;

  extern __shared__ __align__(16) char smem[];
  float *s_state = reinterpret_cast<float *>(smem); // [HEAD_V_DIM][HEAD_K_DIM]
  float *s_k = s_state + STATE_ELEMS;               // [HEAD_K_DIM]
  float *s_q = s_k + HEAD_K_DIM;                    // [HEAD_K_DIM]
  float *s_o = s_q + HEAD_K_DIM;                    // [HEAD_V_DIM]
  float *s_red = s_o + HEAD_V_DIM;                  // [32]

  T const *__restrict__ d_qkv = static_cast<T const *>(qkv_ptr);
  T const *__restrict__ d_ba = static_cast<T const *>(ba_ptr);
  float const *__restrict__ d_ad = static_cast<float const *>(alog_dtbias_ptr);
  float *__restrict__ d_state = static_cast<float *>(state_ptr);
  T const *__restrict__ d_z = static_cast<T const *>(z_ptr);
  float const *__restrict__ d_norm_w = static_cast<float const *>(norm_w_ptr);
  T *__restrict__ d_out = static_cast<T *>(out_ptr);

  int const i_h = hv / (NUM_V_HEADS / NUM_K_HEADS);
  // `-exp(A_log)` and the query scale are loop invariants.  The scale is
  // computed in DOUBLE and rounded once, which is bit-identical to torch's
  // `1 / (head_k_dim ** 0.5)` (a float64 expression handed to a fp32 tensor);
  // `rsqrtf` would not be.
  float const neg_exp_a_log = -expf(d_ad[hv]);
  float const dt_bias = d_ad[NUM_V_HEADS + hv];
  float const q_scale =
      static_cast<float>(1.0 / sqrt(static_cast<double>(HEAD_K_DIM)));

  // Load this slot's state, or treat it as zero on the request's first chunk.
  if (zero_state) {
    for (int i = tid; i < STATE_ELEMS; i += nthreads) {
      s_state[i] = 0.0f;
    }
  } else {
    for (int i = tid; i < STATE_ELEMS; i += nthreads) {
      s_state[i] = d_state[i];
    }
  }
  __syncthreads();

  // The recurrence is sequential in t: a chunk's tokens are walked in order,
  // carrying S in shared memory across the whole chunk.
  for (int t = 0; t < q_len; t++) {
    T const *__restrict__ qkv_row = d_qkv + static_cast<size_t>(t) * QKV_STRIDE;

    gdn_l2norm_bf16<T, HEAD_K_DIM>(
        qkv_row + i_h * HEAD_K_DIM, s_q, s_red, q_scale, EPS);
    gdn_l2norm_bf16<T, HEAD_K_DIM>(
        qkv_row + KEY_DIM + i_h * HEAD_K_DIM, s_k, s_red, 1.0f, EPS);

    // Gating scalars.  `g` is fully fp32 (A_log is an fp32 parameter and HF
    // upcasts `a`); `beta` is a bf16-NATIVE sigmoid - HF calls `b.sigmoid()`
    // on the bf16 tensor with no `.float()` first, so it round-trips.
    float const b_val =
        static_cast<float>(d_ba[static_cast<size_t>(t) * BA_STRIDE + hv]);
    float const a_val = static_cast<float>(
        d_ba[static_cast<size_t>(t) * BA_STRIDE + NUM_V_HEADS + hv]);
    float const beta = static_cast<float>(T(1.0f / (1.0f + expf(-b_val))));
    float const x = a_val + dt_bias;
    float const softplus = (x > SOFTPLUS_THRESHOLD) ? x : log1pf(expf(x));
    float const decay = expf(neg_exp_a_log * softplus);

    // One warp per v-row.  Within a row lane l accumulates k = l, l+32, ...
    // and a shuffle tree combines the 32 partials - the most accurate natural
    // association order (see the header note).
    T const *__restrict__ v_row = qkv_row + VAL_BASE + hv * HEAD_V_DIM;
    for (int v = warp; v < HEAD_V_DIM; v += num_warps) {
      float *__restrict__ row = s_state + static_cast<size_t>(v) * HEAD_K_DIM;

      // Decay FIRST, elementwise, then contract with k: HF applies
      // `S = S * exp(g)` as its own op, so each element is rounded before it
      // enters the dot product.
      float kv = 0.0f;
#pragma unroll
      for (int u = 0; u < K_UNROLL; u++) {
        int const c = lane + u * 32;
        float const s = row[c] * decay;
        row[c] = s;
        kv += s * s_k[c];
      }
#pragma unroll
      for (int m = 16; m > 0; m >>= 1) {
        kv += __shfl_xor_sync(0xffffffff, kv, m);
      }

      float const delta = (static_cast<float>(v_row[v]) - kv) * beta;

      // Rank-1 update and readout share one pass over the row.
      float o = 0.0f;
#pragma unroll
      for (int u = 0; u < K_UNROLL; u++) {
        int const c = lane + u * 32;
        float const s = row[c] + s_k[c] * delta;
        row[c] = s;
        o += s * s_q[c];
      }
#pragma unroll
      for (int m = 16; m > 0; m >>= 1) {
        o += __shfl_xor_sync(0xffffffff, o, m);
      }
      if (lane == 0) {
        s_o[v] = o;
      }
    }
    __syncthreads();

    // Fused epilogue: RMSNormGated(o, z) * norm_w, per head, replacing a
    // separate RMSNormGated task (v1-architecture.md 3.2).  Rounding order is
    // HF's `Qwen3_5MoeRMSNormGated.forward`, see decisions 2 and 3 above.
    T *__restrict__ d_o_debug = static_cast<T *>(o_debug_ptr);
    float acc = 0.0f;
    for (int i = tid; i < HEAD_V_DIM; i += nthreads) {
      float const ob = static_cast<float>(T(s_o[i]));
      s_o[i] = ob;
      if (d_o_debug != nullptr) {
        d_o_debug[static_cast<size_t>(t) * OUT_STRIDE + i] = T(ob);
      }
      acc += ob * ob;
    }
    float const variance =
        gdn_block_sum(acc, s_red) / static_cast<float>(HEAD_V_DIM);
    float const inv_rms = rsqrtf(variance + EPS);
    for (int i = tid; i < HEAD_V_DIM; i += nthreads) {
      float const x_hat = static_cast<float>(T(s_o[i] * inv_rms));
      float const y = d_norm_w[i] * x_hat;
      float const zf =
          static_cast<float>(d_z[static_cast<size_t>(t) * Z_STRIDE + i]);
      // torch's SiLU is `x * sigmoid(x)`, not `x / (1 + exp(-x))`.
      float const silu = zf * (1.0f / (1.0f + expf(-zf)));
      d_out[static_cast<size_t>(t) * OUT_STRIDE + i] = T(y * silu);
    }
  }

  // The updated state is written back unconditionally (including on the
  // zero_state path, which is how a slot's first chunk initialises the pool).
  __syncthreads();
  for (int i = tid; i < STATE_ELEMS; i += nthreads) {
    d_state[i] = s_state[i];
  }
}

} // namespace kernel
