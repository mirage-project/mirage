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
#include <cuda_fp8.h>
// `gdn_slot_is_first_chunk` (the shared step==0 slot predicate) lives in the
// conv task's header; both GDN tasks use it, so pull it in explicitly rather
// than relying on task_header.cuh's include order.
#include "gdn_conv1d_sm100.cuh"
#include "tasks/common/common_header.cuh"

// cp.async (`__pipeline_memcpy_async`) for the decode fast path's row ring.
#include <cuda_pipeline.h>

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

// Block-wide MAX, the same shape as gdn_block_sum above.  M4-I9 flag C uses it
// for the fp8 group amax.  Unlike the sum this one is EXACT and
// order-independent -- fmaxf has no rounding -- so the fused quantize's scale
// cannot differ from the standalone quantize's whatever the reduction shape is.
__device__ __forceinline__ float gdn_block_max(float acc, float *red) {
  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;
  int const num_warps = (blockDim.x + 31) >> 5;
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) {
    acc = fmaxf(acc, __shfl_xor_sync(0xffffffff, acc, m));
  }
  __syncthreads();
  if (lane == 0) {
    red[warp] = acc;
  }
  __syncthreads();
  float total = red[0];
  for (int w = 1; w < num_warps; w++) {
    total = fmaxf(total, red[w]);
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
          int OUT_STRIDE,
          // M4-I9 flag C, same contract as the decode-split impl: defaulted OFF
          // and the extra pointers defaulted null so the pre-M4-I9 call site
          // emits identical generated text.
          bool FUSE_QUANT = false,
          bool WRITE_OUT = true,
          int S_STRIDE = OUT_STRIDE / 128>
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
                                  void *o_debug_ptr = nullptr,
                                  void *q_out_ptr = nullptr,
                                  void *s_out_ptr = nullptr) {
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
    // M4-I9 flag C -- the SAME fusion as the decode-split epilogue, applied per
    // token here because this path's epilogue lives inside the token loop. It
    // MUST be here as well as there: with the flag on, `out`'s consumer reads
    // the fp8 pair, so a prefill chunk that produced only bf16 would hand the
    // projection stale bytes.
    float amax = 1e-10f;
    for (int i = tid; i < HEAD_V_DIM; i += nthreads) {
      float const x_hat = static_cast<float>(T(s_o[i] * inv_rms));
      float const y = d_norm_w[i] * x_hat;
      float const zf =
          static_cast<float>(d_z[static_cast<size_t>(t) * Z_STRIDE + i]);
      // torch's SiLU is `x * sigmoid(x)`, not `x / (1 + exp(-x))`.
      float const silu = zf * (1.0f / (1.0f + expf(-zf)));
      T const out_v = T(y * silu);
      if constexpr (WRITE_OUT) {
        d_out[static_cast<size_t>(t) * OUT_STRIDE + i] = out_v;
      }
      if constexpr (FUSE_QUANT) {
        float const ov = static_cast<float>(out_v);
        s_o[i] = ov;
        amax = fmaxf(fabsf(ov), amax);
      }
    }
    if constexpr (FUSE_QUANT) {
      static_assert(HEAD_V_DIM == 128,
                    "flag C assumes a v-head is exactly one 128-element fp8 "
                    "scale group");
      float group_max = gdn_block_max(amax, s_red);
      group_max = fmaxf(group_max, 1e-10f);
      float const y_scale = group_max / 448.0f;
      if (tid == 0) {
        static_cast<float *>(s_out_ptr)[static_cast<size_t>(t) * S_STRIDE] =
            y_scale;
      }
      __nv_fp8_e4m3 *const d_q = static_cast<__nv_fp8_e4m3 *>(q_out_ptr);
      for (int i = tid; i < HEAD_V_DIM; i += nthreads) {
        float const quant_val = fminf(fmaxf(s_o[i] / y_scale, -448.0f), 448.0f);
        d_q[static_cast<size_t>(t) * OUT_STRIDE + i] = __nv_fp8_e4m3(quant_val);
      }
      __syncthreads(); // s_o is reused by the next token's readout
    }
  }

  // The updated state is written back unconditionally (including on the
  // zero_state path, which is how a slot's first chunk initialises the pool).
  __syncthreads();
  for (int i = tid; i < STATE_ELEMS; i += nthreads) {
    d_state[i] = s_state[i];
  }
}

// ===========================================================================
// DECODE FAST PATH (q_len == 1, carried state).
// ===========================================================================
//
// Ported from the ferret `gdn-recurrent-decode-vllm-beat` winner, workspace2
// tag v010 (dafe5ec).  That loop gated EVERY iteration on an integer `memcmp`
// of BOTH `out` (bf16) and the updated `state` (fp32) against a frozen verbatim
// copy of `gdn_recurrent_sm100_task_impl` above, so the standalone kernel is
// bit-exact against the golden path by construction; this port keeps every
// floating-point expression, per-lane element ownership, FMA chain order and
// shuffle-tree association identical to it.  Standalone (v010, no fast-math):
// 686.24 / 3039.35 / 4262.50 GB/s at bs 1/8/16 vs the vLLM/SGLang Triton
// reference's 768.89 / 3722.48 / 4350.66.
//
// WHY IT IS FASTER, and what changes under MPK dispatch:
//
//   (a) State stays in REGISTERS, never staged in shared memory.  The golden
//       path reads the whole 64 KiB `S` into smem, works on it there and writes
//       it back: 128 KiB of extra smem traffic and two block-wide barriers per
//       token.  Each v-row's recurrence (decay, S k, rank-1 update, S q) is
//       independent of every other row and touches `row[lane + u*32]` exactly
//       once, so the row can be loaded straight to registers, updated, and
//       stored - identical values, no round trip.
//   (b) Warp-level q/k L2 norms replace two block-wide reductions per token
//       (`gdn_l2norm_bf16_warp`; the exactness argument is with it).
//   (c) A cp.async row-staging ring hides the per-row DRAM latency.
//   (d) The v-row axis is SPLIT across SPLIT cooperating tasks.
//
// (d) is the load-bearing MPK-dispatch decision.  MPK runs ONE persistent
// worker CTA per SM (`persistent_kernel.cuh`: 256 threads, the full
// MAX_DYNAMIC_SHARED_MEMORY_SIZE arena, one task at a time), so a task's own
// occupancy is fixed at 1 and the standalone kernel's "8 blocks/SM" argument
// does NOT carry over - but the WIDTH argument does, and harder: at bs1 the
// GDN stage offers only NUM_V_HEADS == 32 tasks, so 116 of 148 SMs sit idle.
// Splitting the v-rows multiplies the task count by SPLIT.
//
// MPK expresses a grid split as SEPARATE TASKS, exactly as
// TASK_PAGED_ATTENTION_SPLIT_KV_SM100 does (`runtime.cc`: the task-creation
// loop walks bid.z and stamps a split index into `task_metadata`).  So this
// task's `bgraph.grid_dim.z` becomes SPLIT, bid.z becomes
// `task_metadata.merge_task_offset`, and each task owns rows
// [split_idx*ROWS, (split_idx+1)*ROWS).
//
// The epilogue (gated RMSNorm over all HEAD_V_DIM readouts) needs every row,
// so the split tasks deposit raw fp32 `o` partials in a scratch buffer and the
// LAST-ARRIVING task runs the verbatim golden epilogue in place, selected by a
// `__threadfence()` + `atomicAdd` arrival counter that self-resets.
//
// THIS IS SOUND UNDER MPK'S PERSISTENT DISPATCH, and the argument matters:
//   * It is NOT a barrier.  No task ever waits for a peer - each one bumps the
//     counter and either returns or runs the epilogue.  So the SPLIT tasks do
//     NOT need to be co-resident, which is exactly the property a persistent
//     work-queue scheduler cannot guarantee.  (A per-token cross-task barrier
//     WOULD deadlock here; that is why prefill, whose epilogue is inside the
//     token loop, keeps the unsplit golden path.)
//   * `__threadfence()` is device scope and the megakernel is a SINGLE launch,
//     so the release/acquire pair covers every worker CTA.
//   * The event that unblocks `out_proj` counts ALL SPLIT tasks, so the
//     epilogue (which runs inside the last of them) always precedes any
//     consumer.
//   * The self-reset is safe because the counter is indexed by (slot, hv) only
//     and GDN layer L+1 is transitively downstream of GDN layer L, so no two
//     uses of a counter word are ever in flight together.
//
// SCRATCH LAYOUT.  `MAX_INPUTS_PER_TASK` is 7 and the task already takes 6
// inputs + 1 output, so the o-partials and the counter share ONE fp32 buffer
// rather than costing a global struct-size change:
//
//   split_scratch [num_slots, num_v_heads, HEAD_V_DIM + 8]  fp32
//       [.. , 0 : HEAD_V_DIM]  raw fp32 `o` partials
//       [.. , HEAD_V_DIM]      arrival counter, read as `unsigned int`
//       [.. , HEAD_V_DIM+1 :]  padding, keeps every row 16 B aligned
//
// Zero-initialised at allocation (torch.zeros gives the 0u counter) and
// self-resetting thereafter.  SPLIT == 1 skips the buffer entirely: the single
// task keeps `o` in shared memory exactly like the golden path.
template <int HEAD_V_DIM>
constexpr int gdn_split_scratch_stride() {
  return HEAD_V_DIM + 8;
}

// One v-row of the decode recurrence.  Expressions, per-lane k ownership
// (c = lane + u*32), the u = 0..K_UNROLL-1 FMA chain and the 5-step shfl_xor
// tree are identical to the golden token loop above; only WHERE the state
// values come from (a register/smem staging slot instead of an inline
// `row[c]` read) differs, which cannot change any FP result.
template <int K_UNROLL>
__device__ __forceinline__ void
    gdn_decode_process_row(float const (&ld)[K_UNROLL],
                           float v_val,
                           float decay,
                           float beta,
                           float const *__restrict__ s_k,
                           float const *__restrict__ s_q,
                           int lane,
                           float *__restrict__ row,
                           float *__restrict__ o_slot) {
  float sreg[K_UNROLL];
  float kv = 0.0f;
#pragma unroll
  for (int u = 0; u < K_UNROLL; u++) {
    int const c = lane + u * 32;
    float const s = ld[u] * decay;
    sreg[u] = s;
    kv += s * s_k[c];
  }
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) {
    kv += __shfl_xor_sync(0xffffffff, kv, m);
  }

  float const delta = (v_val - kv) * beta;

  float o = 0.0f;
#pragma unroll
  for (int u = 0; u < K_UNROLL; u++) {
    int const c = lane + u * 32;
    float const s = sreg[u] + s_k[c] * delta;
    row[c] = s;
    o += s * s_q[c];
  }
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) {
    o += __shfl_xor_sync(0xffffffff, o, m);
  }
  if (lane == 0) {
    *o_slot = o;
  }
}

// transformers' `l2norm` again, but reduced inside ONE warp.
//
// BIT-EXACT vs the block-wide `gdn_l2norm_bf16` above, and this is a proof,
// not an observation.  Golden runs it with NUM_THREADS == 256 over DIM == 128,
// so thread i handles element i and threads 128..255 contribute acc == +0.0f.
// Its `gdn_block_sum` butterfly leaves warp w holding R_w, the shuffle-tree sum
// of T(x*x) over elements [32w, 32w+32), and red[4..7] == +0.0f; the serial
// tail is then total = ((((((((0+R0)+R1)+R2)+R3)+0)+0)+0)+0).  Adding +0.0f to
// a non-negative finite fp32 is exact, so that equals (((0+R0)+R1)+R2)+R3.
// The warp version reproduces each R_g from the SAME 32 values in the SAME
// lanes through the SAME 5-step butterfly, and accumulates them in golden's
// left-to-right red[] order - identical association, no barriers.  Everything
// downstream (the T() roundings, rsqrtf, the post_scale) is unchanged.
//
// The second pass reuses the first pass's `src` values from registers instead
// of re-reading them: `qkv` is const for the whole task and no kernel in the
// launch writes it, so the re-read would return identical bits.
template <typename T, int DIM>
__device__ __forceinline__ void
    gdn_l2norm_bf16_warp(T const *__restrict__ src,
                         float *__restrict__ dst,
                         float post_scale,
                         float eps) {
  static_assert(DIM % 32 == 0, "one warp walks the row 32 lanes at a time");
  constexpr int G = DIM / 32;
  int const lane = threadIdx.x & 31;
  float xs[G];
  float total = 0.0f;
#pragma unroll
  for (int g = 0; g < G; g++) {
    float const x = static_cast<float>(src[g * 32 + lane]);
    xs[g] = x;
    float acc = static_cast<float>(T(x * x));
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) {
      acc += __shfl_xor_sync(0xffffffff, acc, m);
    }
    total += acc; // == golden's serial red[0]+red[1]+...
  }
  float const sum_bf = static_cast<float>(T(total));
  float const eps_bf = static_cast<float>(T(sum_bf + eps));
  float const inv = static_cast<float>(T(rsqrtf(eps_bf)));
#pragma unroll
  for (int g = 0; g < G; g++) {
    float const xn = static_cast<float>(T(xs[g] * inv));
    dst[g * 32 + lane] = xn * post_scale;
  }
}

// Decode step for one (v-head, request slot, v-row split).  The caller
// guarantees q_len == 1 and zero_state == false; every other case goes to
// `gdn_recurrent_sm100_task_impl` above.  Pointers arrive pre-offset exactly
// as they do for that function (`task_register.cc`), plus `split_scratch_ptr`
// already advanced to this (slot, hv) row.
template <typename T,
          int NUM_V_HEADS,
          int NUM_K_HEADS,
          int HEAD_K_DIM,
          int HEAD_V_DIM,
          int QKV_STRIDE,
          int BA_STRIDE,
          int Z_STRIDE,
          int OUT_STRIDE,
          int SPLIT,
          int DEPTH,
          int NUM_THREADS,
          // M4-I9 flag C: fuse the fp32-block-scale FP8 quantize of `out` into
          // this task's epilogue.  Defaulted OFF and the extra pointers
          // defaulted to nullptr so the pre-M4-I9 call site emits the SAME
          // generated text -- gate 1c's byte-identical requirement.
          bool FUSE_QUANT = false,
          bool WRITE_OUT = true>
__device__ __forceinline__ void
    gdn_recurrent_sm100_decode_split_impl(void const *qkv_ptr,
                                          void const *ba_ptr,
                                          void const *alog_dtbias_ptr,
                                          void *state_ptr,
                                          void const *z_ptr,
                                          void const *norm_w_ptr,
                                          void *out_ptr,
                                          void *split_scratch_ptr,
                                          int hv,
                                          int split_idx,
                                          void *q_out_ptr = nullptr,
                                          void *s_out_ptr = nullptr) {
  static_assert(NUM_V_HEADS % NUM_K_HEADS == 0,
                "GVA needs an integer v-heads-per-k-head ratio");
  static_assert(HEAD_K_DIM % 32 == 0,
                "one warp walks a k-row 32 lanes at a time");
  static_assert(BA_STRIDE >= 2 * NUM_V_HEADS, "ba packs [b | a]");
  static_assert(SPLIT >= 1 && HEAD_V_DIM % SPLIT == 0,
                "the v-row split must divide HEAD_V_DIM");
  static_assert(DEPTH >= 2 && DEPTH <= 4, "cp.async ring size");
  static_assert(NUM_THREADS % 32 == 0 && NUM_THREADS >= 96,
                "needs >= 3 warps (q norm, k norm, z staging)");
  constexpr int NUM_WARPS = NUM_THREADS / 32;
  constexpr int KEY_DIM = NUM_K_HEADS * HEAD_K_DIM;
  constexpr int VAL_BASE = 2 * KEY_DIM;
  constexpr int K_UNROLL = HEAD_K_DIM / 32;
  constexpr int ROWS = HEAD_V_DIM / SPLIT;
  constexpr float EPS = 1e-6f;
  constexpr float SOFTPLUS_THRESHOLD = 20.0f;

  int const tid = threadIdx.x;
  int const lane = tid & 31;
  int const warp = tid >> 5;
  int const v0 = split_idx * ROWS;

  // Shared memory comes from the megakernel's dynamic arena (the standalone
  // kernel used static __shared__, which would blow the worker's static
  // budget on top of MAX_DYNAMIC_SHARED_MEMORY_SIZE).
  extern __shared__ __align__(16) char smem[];
  float *s_k = reinterpret_cast<float *>(smem);      // [HEAD_K_DIM]
  float *s_q = s_k + HEAD_K_DIM;                     // [HEAD_K_DIM]
  float *s_o = s_q + HEAD_K_DIM;                     // [HEAD_V_DIM]
  float *s_red = s_o + HEAD_V_DIM;                   // [32]
  float *s_rows = s_red + 32;         // [DEPTH][NUM_WARPS][HEAD_K_DIM]
  T *s_z = reinterpret_cast<T *>(s_rows + (size_t)DEPTH * NUM_WARPS *
                                              HEAD_K_DIM); // [HEAD_V_DIM]
  int *s_is_last = reinterpret_cast<int *>(s_z + HEAD_V_DIM); // [1]

  T const *__restrict__ qkv_row = static_cast<T const *>(qkv_ptr);
  T const *__restrict__ d_ba = static_cast<T const *>(ba_ptr);
  float const *__restrict__ d_ad = static_cast<float const *>(alog_dtbias_ptr);
  float *__restrict__ d_state = static_cast<float *>(state_ptr);
  T const *__restrict__ d_z = static_cast<T const *>(z_ptr);
  float const *__restrict__ d_norm_w = static_cast<float const *>(norm_w_ptr);
  T *__restrict__ d_out = static_cast<T *>(out_ptr);
  float *__restrict__ o_task = static_cast<float *>(split_scratch_ptr);

  int const i_h = hv / (NUM_V_HEADS / NUM_K_HEADS);
  float const neg_exp_a_log = -expf(d_ad[hv]);
  float const dt_bias = d_ad[NUM_V_HEADS + hv];
  float const q_scale =
      static_cast<float>(1.0 / sqrt(static_cast<double>(HEAD_K_DIM)));

  T const *__restrict__ v_row = qkv_row + VAL_BASE + hv * HEAD_V_DIM;
  // With SPLIT == 1 the readout never leaves the block, so keep it in smem
  // exactly like the golden path and skip the scratch round trip entirely.
  float *__restrict__ o_dst = (SPLIT == 1) ? s_o : o_task;

  // Stage the epilogue's z gate at block ENTRY (warp 2 is otherwise idle until
  // the l2norm barrier).  Bytes are copied verbatim; the float conversion still
  // happens at the same place in the epilogue.
  if (warp == 2) {
    for (int i = lane; i < HEAD_V_DIM; i += 32) {
      s_z[i] = d_z[i];
    }
  }

  if constexpr (ROWS <= NUM_WARPS) {
    // Narrow slices (one v-row per warp): hoist the gating scalars and each
    // warp's single state row above the l2norm barrier so the two independent
    // DRAM-latency chains overlap instead of serialising.  Pure instruction
    // scheduling - every FP expression and its order is untouched.
    float const b_val =
        static_cast<float>(d_ba[static_cast<size_t>(hv)]);
    float const a_val =
        static_cast<float>(d_ba[static_cast<size_t>(NUM_V_HEADS + hv)]);
    float const beta = static_cast<float>(T(1.0f / (1.0f + expf(-b_val))));
    float const x = a_val + dt_bias;
    float const softplus = (x > SOFTPLUS_THRESHOLD) ? x : log1pf(expf(x));
    float const decay = expf(neg_exp_a_log * softplus);

    int const v_first = warp;
    float pre[K_UNROLL];
    float v_pre = 0.0f;
    if (v_first < ROWS) {
      float const *__restrict__ row0 =
          d_state + static_cast<size_t>(v0 + v_first) * HEAD_K_DIM;
#pragma unroll
      for (int u = 0; u < K_UNROLL; u++) {
        pre[u] = row0[lane + u * 32];
      }
      v_pre = static_cast<float>(v_row[v0 + v_first]);
    }

    if (warp == 0) {
      gdn_l2norm_bf16_warp<T, HEAD_K_DIM>(
          qkv_row + i_h * HEAD_K_DIM, s_q, q_scale, EPS);
    } else if (warp == 1) {
      gdn_l2norm_bf16_warp<T, HEAD_K_DIM>(
          qkv_row + KEY_DIM + i_h * HEAD_K_DIM, s_k, 1.0f, EPS);
    }
    __syncthreads();

    if (v_first < ROWS) {
      gdn_decode_process_row<K_UNROLL>(
          pre,
          v_pre,
          decay,
          beta,
          s_k,
          s_q,
          lane,
          d_state + static_cast<size_t>(v0 + v_first) * HEAD_K_DIM,
          o_dst + (v0 + v_first));
      for (int v = v_first + NUM_WARPS; v < ROWS; v += NUM_WARPS) {
        int const rv = v0 + v;
        float *__restrict__ row = d_state + static_cast<size_t>(rv) * HEAD_K_DIM;
        float ld[K_UNROLL];
#pragma unroll
        for (int u = 0; u < K_UNROLL; u++) {
          ld[u] = row[lane + u * 32];
        }
        gdn_decode_process_row<K_UNROLL>(ld,
                                         static_cast<float>(v_row[rv]),
                                         decay,
                                         beta,
                                         s_k,
                                         s_q,
                                         lane,
                                         row,
                                         o_dst + rv);
      }
    }
  } else {
    // Wide slices: l2norm first, then the cp.async-staged row loop.  Each
    // warp's first row copy is issued BEFORE the barrier - cp.async consumes no
    // registers, so the block's first rows of state traffic overlap the l2norm
    // DRAM chain instead of serialising behind __syncthreads().
    constexpr int AHEAD = DEPTH - 1;
    // Ring slot `j` for this warp.  Written as K_UNROLL consecutive floats per
    // lane (coalesced global side, and the one cp.async transaction size the
    // hardware takes) and read back as lane + u*32 (bank-conflict-free smem
    // side) - the transpose is deliberate.
    constexpr int CP_BYTES = (int)sizeof(float) * K_UNROLL;
    static_assert(CP_BYTES == 4 || CP_BYTES == 8 || CP_BYTES == 16,
                  "cp.async takes 4, 8 or 16 B per thread, so this path needs "
                  "head_k_dim in {32, 64, 128}");
    float *const my_ring = s_rows + (size_t)warp * HEAD_K_DIM;
    constexpr int RING_STRIDE = NUM_WARPS * HEAD_K_DIM;
    {
      int const pv = warp;
      if (pv < ROWS) {
        float const *__restrict__ rj =
            d_state + static_cast<size_t>(v0 + pv) * HEAD_K_DIM;
        __pipeline_memcpy_async(
            my_ring + lane * K_UNROLL, rj + lane * K_UNROLL, CP_BYTES);
      }
      __pipeline_commit();
    }
    if (warp == 0) {
      gdn_l2norm_bf16_warp<T, HEAD_K_DIM>(
          qkv_row + i_h * HEAD_K_DIM, s_q, q_scale, EPS);
    } else if (warp == 1) {
      gdn_l2norm_bf16_warp<T, HEAD_K_DIM>(
          qkv_row + KEY_DIM + i_h * HEAD_K_DIM, s_k, 1.0f, EPS);
    }
    __syncthreads();

    float const b_val = static_cast<float>(d_ba[static_cast<size_t>(hv)]);
    float const a_val =
        static_cast<float>(d_ba[static_cast<size_t>(NUM_V_HEADS + hv)]);
    float const beta = static_cast<float>(T(1.0f / (1.0f + expf(-b_val))));
    float const x = a_val + dt_bias;
    float const softplus = (x > SOFTPLUS_THRESHOLD) ? x : log1pf(expf(x));
    float const decay = expf(neg_exp_a_log * softplus);

#pragma unroll
    for (int j = 1; j < AHEAD; j++) { // deeper lookahead, post-barrier issue
      int const pv = warp + j * NUM_WARPS;
      if (pv < ROWS) {
        float const *__restrict__ rj =
            d_state + static_cast<size_t>(v0 + pv) * HEAD_K_DIM;
        __pipeline_memcpy_async(my_ring + (size_t)j * RING_STRIDE +
                                    lane * K_UNROLL,
                                rj + lane * K_UNROLL,
                                CP_BYTES);
      }
      __pipeline_commit();
    }

    int it = 0;
    for (int v = warp; v < ROWS; v += NUM_WARPS, it++) {
      int const pv = v + AHEAD * NUM_WARPS;
      if (pv < ROWS) {
        float const *__restrict__ rp =
            d_state + static_cast<size_t>(v0 + pv) * HEAD_K_DIM;
        __pipeline_memcpy_async(my_ring +
                                    (size_t)((it + AHEAD) % DEPTH) *
                                        RING_STRIDE +
                                    lane * K_UNROLL,
                                rp + lane * K_UNROLL,
                                CP_BYTES);
      }
      __pipeline_commit();          // one group per iteration (may be empty)
      __pipeline_wait_prior(AHEAD); // current row's copy complete
      __syncwarp();
      float const *__restrict__ buf =
          my_ring + (size_t)(it % DEPTH) * RING_STRIDE;
      float const v_val = static_cast<float>(v_row[v0 + v]);
      float ld[K_UNROLL];
#pragma unroll
      for (int u = 0; u < K_UNROLL; u++) {
        ld[u] = buf[lane + u * 32];
      }
      // Ring-slot reuse is race-free without an extra sync: the first
      // full-mask __shfl_xor_sync inside gdn_decode_process_row reconverges the
      // warp AFTER every lane has read its slice into registers, and the next
      // overwrite of this slot is only issued on a later iteration.
      gdn_decode_process_row<K_UNROLL>(
          ld,
          v_val,
          decay,
          beta,
          s_k,
          s_q,
          lane,
          d_state + static_cast<size_t>(v0 + v) * HEAD_K_DIM,
          o_dst + (v0 + v));
    }
  }

  // ---- epilogue: the last-arriving split task runs it verbatim ----
  __syncthreads(); // this task's state / o writes are issued
  if constexpr (SPLIT > 1) {
    unsigned int *const ctr = reinterpret_cast<unsigned int *>(
        o_task + HEAD_V_DIM); // counter word, see the layout note above
    if (tid == 0) {
      __threadfence(); // release: publish this task's writes
      unsigned int const old = atomicAdd(ctr, 1u);
      bool const last = (old == (unsigned int)(SPLIT - 1));
      *s_is_last = last ? 1 : 0;
      if (last) {
        *ctr = 0u;       // self-reset for the next use of this (slot, hv)
        __threadfence(); // acquire: pull peer tasks' o partials
      }
    }
    __syncthreads();
    if (*s_is_last == 0) {
      return;
    }
    for (int i = tid; i < HEAD_V_DIM; i += NUM_THREADS) {
      s_o[i] = o_task[i];
    }
    __syncthreads();
  }

  // Verbatim golden epilogue: same rounding order, same gdn_block_sum
  // association, same NUM_THREADS so the warp-partial layout matches.
  float acc = 0.0f;
  for (int i = tid; i < HEAD_V_DIM; i += NUM_THREADS) {
    float const ob = static_cast<float>(T(s_o[i]));
    s_o[i] = ob;
    acc += ob * ob;
  }
  float const variance =
      gdn_block_sum(acc, s_red) / static_cast<float>(HEAD_V_DIM);
  float const inv_rms = rsqrtf(variance + EPS);
  // M4-I9 flag C.  HEAD_V_DIM is 128 on this checkpoint, i.e. EXACTLY one fp8
  // scale group, and this task owns the whole of it: the split tasks deposit
  // partials and only the last-arriving one reaches here.  So the quantize the
  // standalone task would do over `out`'s group `hv` can be done here with no
  // cross-task reduction and no change of grouping.
  //
  // BIT-EXACT BY CONSTRUCTION.  `out_v` is the same `T(y * silu)` this loop
  // already stores, so the value quantized is the same bf16 the standalone
  // quantize would read back out of global; the amax is over the same 128
  // elements seeded with the same `eps`, reduced with fmaxf, which is exact and
  // order-independent; and `y_scale`, the scale store and
  // `fp8(clamp(orig / y_scale, min, max))` are the same expressions as
  // `per_token_group_quantize_fp8_task_impl`'s f32-scale branch.  No rounding
  // position moves, so the CAST-POSITION RULE does not apply.
  float amax = 1e-10f; // the quantize's `eps` seed
  for (int i = tid; i < HEAD_V_DIM; i += NUM_THREADS) {
    float const x_hat = static_cast<float>(T(s_o[i] * inv_rms));
    float const y = d_norm_w[i] * x_hat;
    float const zf = static_cast<float>(s_z[i]); // staged at entry
    float const silu = zf * (1.0f / (1.0f + expf(-zf)));
    T const out_v = T(y * silu);
    if constexpr (WRITE_OUT) {
      d_out[i] = out_v;
    }
    if constexpr (FUSE_QUANT) {
      float const ov = static_cast<float>(out_v);
      s_o[i] = ov; // s_o is free from here; restage for the second pass
      amax = fmaxf(fabsf(ov), amax);
    }
  }
  if constexpr (FUSE_QUANT) {
    static_assert(HEAD_V_DIM == 128,
                  "flag C assumes a v-head is exactly one 128-element fp8 "
                  "scale group; a different HEAD_V_DIM would need the group "
                  "loop the standalone quantize has");
    float group_max = gdn_block_max(amax, s_red);
    group_max = fmaxf(group_max, 1e-10f);
    float const y_scale = group_max / 448.0f;
    if (tid == 0) {
      static_cast<float *>(s_out_ptr)[0] = y_scale;
    }
    __nv_fp8_e4m3 *const d_q = static_cast<__nv_fp8_e4m3 *>(q_out_ptr);
    for (int i = tid; i < HEAD_V_DIM; i += NUM_THREADS) {
      float const quant_val = fminf(fmaxf(s_o[i] / y_scale, -448.0f), 448.0f);
      d_q[i] = __nv_fp8_e4m3(quant_val);
    }
  }
}

// Dynamic-smem bytes the decode fast path carves out of the worker arena.
template <typename T,
          int HEAD_K_DIM,
          int HEAD_V_DIM,
          int DEPTH,
          int NUM_THREADS>
constexpr size_t gdn_decode_split_smem_bytes() {
  return sizeof(float) * ((size_t)2 * HEAD_K_DIM + HEAD_V_DIM + 32 +
                          (size_t)DEPTH * (NUM_THREADS / 32) * HEAD_K_DIM) +
         sizeof(T) * HEAD_V_DIM + sizeof(int);
}

} // namespace kernel
