// Qwen3.8 GDN decode: one value head of one request, on SGLang's
// [slot, value_head, value_dim, key_dim] FP32 temporal cache.
#pragma once
#include "../core.cuh"
#include "mirage/persistent_kernel/tasks/common/qwen35_gdn.cuh"

namespace static_mk {

// The stock packed Triton decode emits mul.f32 by log2(e) followed by
// ex2.approx.f32 for tl.exp, and sqrt.approx.ftz.f32 for tl.sqrt. CUDA's
// expf/sqrtf use different precision paths without --use_fast_math, which
// perturbs the FP32 recurrent cache even when the BF16 output still matches.
__device__ __forceinline__ float qwen_gdn_triton_exp(float value) {
  float exponent = __fmul_rn(value, __int_as_float(0x3fb8aa3b));
  float result;
  asm volatile("ex2.approx.f32 %0, %1;" : "=f"(result) : "f"(exponent));
  return result;
}

__device__ __forceinline__ float qwen_gdn_triton_sqrt(float value) {
  float result;
  asm volatile("sqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ float qwen_gdn_triton_rsqrt(float value) {
  float result;
  asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ float qwen_gdn_triton_div(float numerator,
                                                     float denominator) {
  float result;
  asm volatile("div.full.f32 %0, %1, %2;"
               : "=f"(result) : "f"(numerator), "f"(denominator));
  return result;
}

// SGLang's packed Triton kernel uses one warp with four adjacent K values
// per lane. Its local f32x2 reduction is (p0 + p2) + (p1 + p3), followed by
// butterfly shuffles with XOR 16, 8, 4, 2, 1. All 32 lanes must call this.
__device__ __forceinline__ float qwen_gdn_triton_warp_sum(
    float p0, float p1, float p2, float p3) {
  float sum = __fadd_rn(__fadd_rn(p0, p2), __fadd_rn(p1, p3));
#pragma unroll
  for (int offset = 16; offset >= 1; offset >>= 1)
    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffff, sum, offset));
  return sum;
}

template <typename NormT>
__device__ __forceinline__ void qwen_gdn_sglang_norm_gate(
    __nv_bfloat16 const *raw, __nv_bfloat16 const *z,
    char const *norm_weight, __nv_bfloat16 *output) {
  // B8 SGLang's gated RMSNorm assigns 16 lanes per 128-element row and eight
  // adjacent elements per lane. Its tl.sum first reduces the even and odd
  // positions separately, then XOR-shuffles lanes 8, 4, 2, 1. The 32-lane
  // Qwen/HF reduction order can cross BF16 rounding boundaries here.
  __shared__ float inv_rms;
  int const i = threadIdx.x;
  if (i < 32) {
    int const base = (i & 15) * 8;
    float p[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float const value = float(raw[base + j]);
      p[j] = __fmul_rn(value, value);
    }
    float const even = __fadd_rn(__fadd_rn(p[0], p[2]),
                                 __fadd_rn(p[4], p[6]));
    float const odd = __fadd_rn(__fadd_rn(p[1], p[3]),
                                __fadd_rn(p[5], p[7]));
    float ss = __fadd_rn(even, odd);
#pragma unroll
    for (int offset = 8; offset >= 1; offset >>= 1)
      ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffff, ss, offset));
    if (i == 0)
      inv_rms = qwen_gdn_triton_rsqrt(__fadd_rn(
          qwen_gdn_triton_div(ss, 128.f), 1.e-6f));
  }
  __syncthreads();
  if (i < 128) {
    auto weight = reinterpret_cast<NormT const *>(norm_weight);
    float const value = float(raw[i]);
    float const gate = float(z[i]);
    float const normalized = __fmul_rn(value, inv_rms);
    float const weighted = __fmul_rn(normalized, float(weight[i]));
    float const sigmoid = qwen_gdn_triton_div(
        1.f, __fadd_rn(1.f, qwen_gdn_triton_exp(-gate)));
    float const silu = __fmul_rn(gate, sigmoid);
    output[i] = __nv_bfloat16(__fmul_rn(weighted, silu));
  }
  __syncthreads();
}

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS,
          class CONV, class QKVZ, class AB, class... REST>
__device__ __forceinline__ void run_qwen_gdn_delta(
    Maps const &, G const &g, KernelLocals &L, StaticTask const &tk) {
  static_assert(T == 8 && SELF::x == 48 && SELF::y == 8,
                "Qwen GDN static recurrence supports B8 only");
  static_assert(CONV::buf >= 0 && CONV::counter >= 0 &&
                    (QKVZ::x == 0 || (QKVZ::buf >= 0 && QKVZ::counter >= 0)) &&
                    (AB::x == 0 || (AB::buf >= 0 && AB::counter >= 0)) &&
                    SELF::counter >= 0,
                "GDN producers need output buffers and counters");
  int const head = tk.x, row = tk.y;
  if (threadIdx.x == 0) {
    cnt_wait(cnt_now(g) + CONV::counter + row, CONV::x);
    if constexpr (QKVZ::x != 0)
      cnt_wait(cnt_now(g) + QKVZ::counter, QKVZ::x * QKVZ::y);
    if constexpr (AB::x != 0)
      cnt_wait(cnt_now(g) + AB::counter, AB::x * AB::y);
  }
  __syncthreads();
#ifdef STATIC_TIMING_BUILD
  // Timing builds only: separate producer wait from recurrence and gated norm.
  if (threadIdx.x == 0) L.stage_stamps[0] = gtime();
#endif

  auto output = buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 4>()) +
                row * 6144 + head * 128;
  int const slot = buf_at<int const>(g, slot_at<BUF_SLOTS, 3>())[row];
  if (slot <= 0) {
    if (threadIdx.x < 128) output[threadIdx.x] = __float2bfloat16(0.f);
    __syncthreads();
    if (threadIdx.x == 0) cnt_add(cnt_now(g) + SELF::counter, 1);
    return;
  }

  auto qkv = buf_at<__nv_bfloat16 const>(g, CONV::buf) + row * 10240;
  __nv_bfloat16 const *qkvz, *ab_base;
  if constexpr (QKVZ::x == 0)
    qkvz = buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 5>());
  else
    qkvz = buf_at<__nv_bfloat16 const>(g, QKVZ::buf);
  if constexpr (AB::x == 0)
    ab_base = buf_at<__nv_bfloat16 const>(
        g, slot_at<BUF_SLOTS, 5 + (QKVZ::x == 0 ? 1 : 0)>());
  else
    ab_base = buf_at<__nv_bfloat16 const>(g, AB::buf);
  auto z = qkvz + row * 16384 + 10240 + head * 128;
  // The hybrid graph binds SGLang's [B,A] 96-column projection directly;
  // full static graphs retain their padded [A,B,zero] 128-column layout.
  auto ab = ab_base + row * (PARAMS::v[1] == 1 ? 96 : 128);
  auto parameters = buf_at<float const>(g, slot_at<BUF_SLOTS, 0>());
  auto norm_weight = buf_at<char const>(g, slot_at<BUF_SLOTS, 1>());
  auto state = buf_at<float>(g, slot_at<BUF_SLOTS, 2>()) +
               (static_cast<size_t>(slot) * 48 + head) * 128 * 128;

  // SGLang's packed decode normalizes Q/K in FP32. The original MPK Qwen
  // task rounds intermediate norms to BF16 for HF parity; those casts must
  // not carry over to this SGLang-integrated path.
  __shared__ float q[128], k[128], q_norm, k_norm, decay, beta;
  __shared__ __nv_bfloat16 raw[128];
  int const key_head = head / 3;
  auto qx = qkv + key_head * 128;
  auto kx = qkv + 2048 + key_head * 128;
  auto vx = qkv + 4096 + head * 128;
  if (threadIdx.x < 32) {
    // #blocked<{sizePerThread=[4],threadsPerWarp=[32]}> from the stock
    // packed kernel: one lane owns four adjacent K entries.
    int const base = 4 * threadIdx.x;
    float q_products[4], k_products[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float const qv = float(qx[base + j]);
      float const kv = float(kx[base + j]);
      q_products[j] = __fmul_rn(qv, qv);
      k_products[j] = __fmul_rn(kv, kv);
    }
    float qs = __fadd_rn(__fadd_rn(q_products[0], q_products[2]),
                         __fadd_rn(q_products[1], q_products[3]));
    float ks = __fadd_rn(__fadd_rn(k_products[0], k_products[2]),
                         __fadd_rn(k_products[1], k_products[3]));
#pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
      qs = __fadd_rn(qs, __shfl_xor_sync(0xffffffff, qs, offset));
      ks = __fadd_rn(ks, __shfl_xor_sync(0xffffffff, ks, offset));
    }
    if (threadIdx.x == 0) {
      // The packed Triton decode divides by sqrt(sum + eps), rather than
      // multiplying by a reciprocal square root as the HF helper does.
      q_norm = qwen_gdn_triton_sqrt(__fadd_rn(qs, 1.e-6f));
      k_norm = qwen_gdn_triton_sqrt(__fadd_rn(ks, 1.e-6f));
      float const a = float(ab[PARAMS::v[1] == 1 ? 48 + head : head]);
      float const dt = a + parameters[48 + head];
      float const softplus = dt <= 20.f ? logf(1.f + qwen_gdn_triton_exp(dt)) : dt;
      decay = qwen_gdn_triton_exp(-qwen_gdn_triton_exp(parameters[head]) * softplus);
      float const b = float(ab[PARAMS::v[1] == 1 ? head : 48 + head]);
      beta = float(__nv_bfloat16(qwen_gdn_triton_div(
          1.f, 1.f + qwen_gdn_triton_exp(-b))));
    }
  }
  __syncthreads();
  if (threadIdx.x < 128) {
    int const i = threadIdx.x;
    // Match the float32 kernel parameter produced by SGLang's Python
    // (128 ** -0.5); rsqrtf may round this constant differently.
    q[i] = __fmul_rn(qwen_gdn_triton_div(float(qx[i]), q_norm),
                     __int_as_float(0x3db504f3));
    k[i] = qwen_gdn_triton_div(float(kx[i]), k_norm);
  }
  __syncthreads();
#ifdef QWEN_GDN_Q_ONLY
  // Diagnostic build: each group of three value heads has one writer for its
  // shared query head. Reinterpret the BF16 output allocation as FP32, so the
  // existing graph interface can expose exact normalized Q bits and q_norm.
  if (head % 3 == 0 && threadIdx.x < 128) {
    auto q_dump = reinterpret_cast<float *>(
        buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 4>())) + row * 3072;
    q_dump[key_head * 128 + threadIdx.x] = q[threadIdx.x];
    if (threadIdx.x == 0) q_dump[2048 + key_head] = q_norm;
  }
  __syncthreads();
  if (threadIdx.x == 0) cnt_add(cnt_now(g) + SELF::counter, 1);
  return;
#endif
  // The 256-thread task has eight warps. Each warp owns one complete V row at
  // a time, with four adjacent K values per lane as in stock Triton. Keeping
  // the decayed state in registers avoids a second global read for output.
  int const warp = threadIdx.x >> 5;
  int const lane = threadIdx.x & 31;
  int const k0 = 4 * lane;
  // Q is invariant across this warp's value rows. Keep its four lane-local
  // elements live across the recurrence instead of reloading shared memory.
  float q_values[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) q_values[j] = q[k0 + j];
#pragma unroll 4
  for (int v = warp; v < 128; v += 8) {
    float *sv = state + v * 128 + k0;
    // Each lane owns four consecutive FP32 state values.  The address is
    // 16-byte aligned, so load/store the row segment as one vector while
    // retaining the per-value arithmetic and reduction order.
    float4 const state_values = *reinterpret_cast<float4 const *>(sv);
    float const old_state[4] = {
        state_values.x, state_values.y, state_values.z, state_values.w};
    float decayed[4], prediction_parts[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      decayed[j] = __fmul_rn(old_state[j], decay);
      prediction_parts[j] = __fmul_rn(decayed[j], k[k0 + j]);
    }
    float const prediction = qwen_gdn_triton_warp_sum(
        prediction_parts[0], prediction_parts[1],
        prediction_parts[2], prediction_parts[3]);
    float const correction = __fmul_rn(float(vx[v]) - prediction, beta);
    float output_parts[4], updated_state[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      updated_state[j] = __fmaf_rn(k[k0 + j], correction, decayed[j]);
      output_parts[j] = __fmul_rn(updated_state[j], q_values[j]);
    }
    *reinterpret_cast<float4 *>(sv) = make_float4(
        updated_state[0], updated_state[1],
        updated_state[2], updated_state[3]);
    float const value = qwen_gdn_triton_warp_sum(
        output_parts[0], output_parts[1],
        output_parts[2], output_parts[3]);
    if (lane == 0) raw[v] = __nv_bfloat16(value);
  }
  __syncthreads();
#ifdef STATIC_TIMING_BUILD
  if (threadIdx.x == 0) L.stage_stamps[1] = gtime();
#endif
#ifdef QWEN_GDN_RAW_ONLY
  // Diagnostic build: expose the BF16 recurrence output before gated RMSNorm.
  if (threadIdx.x < 128) output[threadIdx.x] = raw[threadIdx.x];
  __syncthreads();
#else
  if constexpr (PARAMS::v[0] == 1) {
    qwen_gdn_sglang_norm_gate<float>(
        raw, z, norm_weight, output);
  } else {
    qwen_gdn_sglang_norm_gate<__nv_bfloat16>(
        raw, z, norm_weight, output);
  }
#endif
  if (threadIdx.x == 0) cnt_add(cnt_now(g) + SELF::counter, 1);
}

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS,
          class... IN>
constexpr int smem_qwen_gdn_delta() { return 0; }

} // namespace static_mk
