// GPU numeric check for moe_mxfp4_sm100_task_impl.
//
//   nvcc -O3 -std=c++17 -arch=sm_100a --expt-relaxed-constexpr --expt-extended-lambda \
//     -I <repo>/include -I <repo>/deps/cutlass/include \
//     numeric_sm100.cu -o /tmp/mxfp4_numeric && /tmp/mxfp4_numeric

#include "mirage/persistent_kernel/tasks/blackwell/moe_mxfp4_sm100.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

constexpr int BATCH = 8;
constexpr int TOPK = 4;
constexpr int K = 2880;

#define CHECK(cmd)                                                             \
  do {                                                                         \
    cudaError_t e = (cmd);                                                     \
    if (e != cudaSuccess) {                                                    \
      std::printf("%s: %s\n", #cmd, cudaGetErrorString(e));                    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void w13(cute::bfloat16_t const *input, uint8_t const *blocks, uint8_t const *scales,
                    int32_t const *routing, int32_t const *mask, cute::bfloat16_t const *bias,
                    cute::bfloat16_t *output, int expert_offset, bool with_bias) {
  if (with_bias) {
    kernel::moe_mxfp4_sm100_task_impl<BATCH, 128, 5760, K, 4, TOPK, 8, 5760, true, false>(
        input, blocks, scales, routing, mask, bias, output, expert_offset);
  } else {
    kernel::moe_mxfp4_sm100_task_impl<BATCH, 128, 5760, K, 4, TOPK, 8, 5760, true, true>(
        input, blocks, scales, routing, mask, bias, output, expert_offset);
  }
}

__global__ void w13_both(cute::bfloat16_t const *input, uint8_t const *blocks, uint8_t const *scales,
                         int32_t const *routing, int32_t const *mask, cute::bfloat16_t *output) {
  kernel::moe_mxfp4_sm100_task_impl<BATCH, 128, 5760, K, 4, TOPK, 1, 5760, true, true>(
      input, blocks, scales, routing, mask, nullptr, output, 0);
}

__global__ void w2(cute::bfloat16_t const *input, uint8_t const *blocks, uint8_t const *scales,
                   int32_t const *routing, int32_t const *mask, cute::bfloat16_t *output) {
  kernel::moe_mxfp4_sm100_task_impl<BATCH, 64, 2880, K, 4, TOPK, 8, 2880, false, true>(
      input, blocks, scales, routing, mask, nullptr, output, 0);
}

static double raw_dot(float const *act, uint8_t const *wrow, uint8_t const *wscale) {
  double acc = 0.0;
  for (int gk = 0; gk < K; gk += 32) {
    float vals[32];
    float amax = 0.f;
    for (int k = 0; k < 32; ++k) {
      vals[k] = act[gk + k];
      amax = fmaxf(amax, fabsf(vals[k]));
    }
    int sb = kernel::mxfp4::quantize_ue8m0(amax);
    float inv = amax > 0.f ? 1.f / kernel::mxfp4::ue8m0(sb) : 0.f;
    float ascale = kernel::mxfp4::ue8m0(sb);
    float ws = kernel::mxfp4::ue8m0(wscale[gk / 32]);
    for (int k = 0; k < 32; ++k) {
      uint8_t byte = wrow[(gk + k) / 2];
      int wn = (k & 1) ? (byte >> 4) : (byte & 15);
      int an = kernel::mxfp4::quantize_e2m1(vals[k] * inv);
      acc += double(kernel::mxfp4::e2m1(an) * ascale) * double(kernel::mxfp4::e2m1(wn) * ws);
    }
  }
  return acc;
}

static float quant_dot(float const *act, uint8_t const *wrow, uint8_t const *wscale) {
  return float(cute::bfloat16_t(float(raw_dot(act, wrow, wscale))));
}

static void fill_weights(std::vector<uint8_t> &blocks, std::vector<uint8_t> &scales, int rows, int orig,
                         int expert) {
  for (int n = 0; n < rows; ++n) {
    int row = expert * orig + n;
    for (int k = 0; k < K; k += 2) {
      int n0 = (row * 3 + k) & 15;
      int n1 = (row + k) & 15;
      blocks[row * (K / 2) + k / 2] = uint8_t(n0 | (n1 << 4));
    }
    for (int s = 0; s < K / 32; ++s) {
      scales[row * (K / 32) + s] = uint8_t(127 + ((n + s) & 3));
    }
  }
}

static int launch(void (*fn)(cute::bfloat16_t const *, uint8_t const *, uint8_t const *, int32_t const *,
                             int32_t const *, cute::bfloat16_t const *, cute::bfloat16_t *, int, bool),
                  cute::bfloat16_t const *in, uint8_t const *blocks, uint8_t const *scales, int32_t const *routing,
                  int32_t const *mask, cute::bfloat16_t const *bias, cute::bfloat16_t *out, int expert_offset,
                  bool with_bias) {
  int smem = 96 * 1024;
  CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  fn<<<1, 256, smem>>>(in, blocks, scales, routing, mask, bias, out, expert_offset, with_bias);
  CHECK(cudaDeviceSynchronize());
  return 0;
}

int main() {
  constexpr int ORIG13 = 5760;
  constexpr int N13 = 128;
  constexpr int ORIG2 = 2880;
  constexpr int N2 = 64;
  constexpr float kSentinel = 32.f;

  std::vector<float> act(BATCH * K);
  for (int t = 0; t < BATCH; ++t) {
    for (int k = 0; k < K; ++k) {
      act[t * K + k] = ((t * 3 + k * 5) % 17) * 0.25f - 1.0f;
    }
  }
  std::vector<cute::bfloat16_t> input13(BATCH * K);
  for (size_t i = 0; i < act.size(); ++i) {
    input13[i] = cute::bfloat16_t(act[i]);
  }

  std::vector<uint8_t> blocks13(2 * ORIG13 * (K / 2)), scales13(2 * ORIG13 * (K / 32));
  fill_weights(blocks13, scales13, N13, ORIG13, 0);
  fill_weights(blocks13, scales13, N13, ORIG13, 1);
  std::vector<cute::bfloat16_t> bias(2 * ORIG13);
  for (int n = 0; n < N13; ++n) {
    bias[n] = cute::bfloat16_t(0.5f * ((n % 5) - 2));
    bias[ORIG13 + n] = cute::bfloat16_t(-0.25f);
  }

  std::vector<int32_t> routing(2 * BATCH, 0), mask(5, 0);
  for (int t = 0; t < BATCH; ++t) {
    routing[t] = 1;
  }
  routing[3] = 0;
  mask[0] = 0;
  mask[4] = 1;

  cute::bfloat16_t *dI, *dO, *dBias;
  uint8_t *dB, *dS;
  int32_t *dR, *dM;
  CHECK(cudaMalloc(&dI, BATCH * TOPK * K * sizeof(cute::bfloat16_t)));
  CHECK(cudaMalloc(&dO, BATCH * TOPK * ORIG13 * sizeof(cute::bfloat16_t)));
  CHECK(cudaMalloc(&dBias, bias.size() * sizeof(cute::bfloat16_t)));
  CHECK(cudaMalloc(&dB, blocks13.size()));
  CHECK(cudaMalloc(&dS, scales13.size()));
  CHECK(cudaMalloc(&dR, routing.size() * 4));
  CHECK(cudaMalloc(&dM, mask.size() * 4));
  CHECK(cudaMemcpy(dI, input13.data(), input13.size() * sizeof(cute::bfloat16_t), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, blocks13.data(), blocks13.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dS, scales13.data(), scales13.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dBias, bias.data(), bias.size() * sizeof(cute::bfloat16_t), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dR, routing.data(), routing.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dM, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice));

  auto check = [&](char const *name, int n_out, int orig, int expert, int slot_1, bool with_bias,
                   bool unrouted_token3) {
    std::vector<cute::bfloat16_t> got(BATCH * TOPK * orig);
    CHECK(cudaMemcpy(got.data(), dO, got.size() * sizeof(cute::bfloat16_t), cudaMemcpyDeviceToHost));
    double max_abs = 0;
    int n = 0;
    for (int t = 0; t < BATCH; ++t) {
      bool skip = unrouted_token3 && t == 3;
      for (int c = 0; c < n_out; ++c) {
        float g = float(got[(static_cast<size_t>(t) * TOPK + (slot_1 - 1)) * orig + c]);
        if (skip) {
          max_abs = fmax(max_abs, fabs(double(g) - double(kSentinel)));
          continue;
        }
        uint8_t const *wrow = blocks13.data() + (static_cast<size_t>(expert) * orig + c) * (K / 2);
        uint8_t const *wscale = scales13.data() + (static_cast<size_t>(expert) * orig + c) * (K / 32);
        float ref = with_bias ? float(cute::bfloat16_t(float(raw_dot(act.data() + t * K, wrow, wscale)) +
                                                       float(bias[expert * orig + c])))
                              : quant_dot(act.data() + t * K, wrow, wscale);
        max_abs = fmax(max_abs, fabs(double(g) - double(ref)));
        ++n;
      }
    }
    bool ok = max_abs < 1e-3;
    std::printf("%s %s max_abs=%g\n", ok ? "PASS" : "FAIL", name, max_abs);
    return ok ? 0 : 1;
  };

  int failed = 0;
  std::vector<cute::bfloat16_t> sent(BATCH * TOPK * ORIG13, cute::bfloat16_t(kSentinel));
  CHECK(cudaMemcpy(dO, sent.data(), sent.size() * sizeof(cute::bfloat16_t), cudaMemcpyHostToDevice));
  if (launch(w13, dI, dB, dS, dR, dM, nullptr, dO, 0, false)) {
    return 1;
  }
  failed += check("w13 unrouted token", N13, ORIG13, 0, 1, false, true);

  CHECK(cudaMemcpy(dO, sent.data(), sent.size() * sizeof(cute::bfloat16_t), cudaMemcpyHostToDevice));
  routing[3] = 1;
  CHECK(cudaMemcpy(dR, routing.data(), routing.size() * 4, cudaMemcpyHostToDevice));
  if (launch(w13, dI, dB, dS, dR, dM, dBias, dO, 0, true)) {
    return 1;
  }
  failed += check("w13 bias", N13, ORIG13, 0, 1, true, false);

  // Two experts, one CTA, distinct topk slots.
  for (int t = 0; t < BATCH; ++t) {
    routing[t] = 1;
    routing[BATCH + t] = 2;
  }
  mask[0] = 0;
  mask[1] = 1;
  mask[4] = 2;
  CHECK(cudaMemcpy(dR, routing.data(), routing.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dM, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dO, sent.data(), sent.size() * sizeof(cute::bfloat16_t), cudaMemcpyHostToDevice));
  int smem = 96 * 1024;
  CHECK(cudaFuncSetAttribute(w13_both, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  w13_both<<<1, 256, smem>>>(dI, dB, dS, dR, dM, dO);
  CHECK(cudaDeviceSynchronize());
  failed += check("w13 expert0 slot1", N13, ORIG13, 0, 1, false, false);
  failed += check("w13 expert1 slot2", N13, ORIG13, 1, 2, false, false);

  // Offset 1 skips expert 0.
  CHECK(cudaMemcpy(dO, sent.data(), sent.size() * sizeof(cute::bfloat16_t), cudaMemcpyHostToDevice));
  if (launch(w13, dI, dB, dS, dR, dM, nullptr, dO, 1, false)) {
    return 1;
  }
  {
    std::vector<cute::bfloat16_t> got(BATCH * TOPK * ORIG13);
    CHECK(cudaMemcpy(got.data(), dO, got.size() * sizeof(cute::bfloat16_t), cudaMemcpyDeviceToHost));
    double untouched = 0, used = 0;
    for (int t = 0; t < BATCH; ++t) {
      for (int c = 0; c < N13; ++c) {
        float slot1 = float(got[(static_cast<size_t>(t) * TOPK + 0) * ORIG13 + c]);
        untouched = fmax(untouched, fabs(double(slot1) - double(kSentinel)));
        uint8_t const *wrow = blocks13.data() + (static_cast<size_t>(1) * ORIG13 + c) * (K / 2);
        uint8_t const *wscale = scales13.data() + (static_cast<size_t>(1) * ORIG13 + c) * (K / 32);
        float ref = quant_dot(act.data() + t * K, wrow, wscale);
        float slot2 = float(got[(static_cast<size_t>(t) * TOPK + 1) * ORIG13 + c]);
        used = fmax(used, fabs(double(slot2) - double(ref)));
      }
    }
    bool ok = untouched < 1e-3 && used < 1e-3;
    std::printf("%s w13 offset skips expert0 untouched=%g expert1=%g\n", ok ? "PASS" : "FAIL", untouched, used);
    failed += ok ? 0 : 1;
  }

  // W2 reads the per-slot activation, not the shared W13 input.
  std::vector<cute::bfloat16_t> input2(BATCH * TOPK * K);
  std::vector<float> act2(BATCH * TOPK * K);
  for (int t = 0; t < BATCH; ++t) {
    for (int k = 0; k < K; ++k) {
      float v = ((t * 9 + k) % 13) * 0.5f - 2.f;
      act2[(t * TOPK + 0) * K + k] = v;
      input2[(t * TOPK + 0) * K + k] = cute::bfloat16_t(v);
    }
  }
  std::vector<uint8_t> blocks2(ORIG2 * (K / 2)), scales2(ORIG2 * (K / 32));
  for (int n = 0; n < N2; ++n) {
    for (int k = 0; k < K; k += 2) {
      blocks2[n * (K / 2) + k / 2] = uint8_t(((n + k) & 15) | (((n * 2 + k) & 15) << 4));
    }
    for (int s = 0; s < K / 32; ++s) {
      scales2[n * (K / 32) + s] = 130;
    }
  }
  for (int t = 0; t < BATCH; ++t) {
    routing[t] = 1;
  }
  mask[0] = 0;
  mask[4] = 1;
  CHECK(cudaMemcpy(dI, input2.data(), input2.size() * sizeof(cute::bfloat16_t), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, blocks2.data(), blocks2.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dS, scales2.data(), scales2.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dR, routing.data(), routing.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dM, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice));
  std::vector<cute::bfloat16_t> sent2(BATCH * TOPK * ORIG2, cute::bfloat16_t(kSentinel));
  CHECK(cudaMemcpy(dO, sent2.data(), sent2.size() * sizeof(cute::bfloat16_t), cudaMemcpyHostToDevice));
  CHECK(cudaFuncSetAttribute(w2, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  w2<<<1, 256, smem>>>(dI, dB, dS, dR, dM, dO);
  CHECK(cudaDeviceSynchronize());
  {
    std::vector<cute::bfloat16_t> got(BATCH * TOPK * ORIG2);
    CHECK(cudaMemcpy(got.data(), dO, got.size() * sizeof(cute::bfloat16_t), cudaMemcpyDeviceToHost));
    double max_abs = 0, past = 0;
    for (int t = 0; t < BATCH; ++t) {
      for (int c = 0; c < ORIG2; ++c) {
        float g = float(got[(static_cast<size_t>(t) * TOPK) * ORIG2 + c]);
        if (c >= N2) {
          past = fmax(past, fabs(double(g) - double(kSentinel)));
          continue;
        }
        float ref = quant_dot(act2.data() + (t * TOPK) * K, blocks2.data() + c * (K / 2),
                              scales2.data() + c * (K / 32));
        max_abs = fmax(max_abs, fabs(double(g) - double(ref)));
      }
    }
    bool ok = max_abs < 1e-3 && past < 1e-3;
    std::printf("%s w2 slice max_abs=%g past_slice=%g\n", ok ? "PASS" : "FAIL", max_abs, past);
    failed += ok ? 0 : 1;
  }

  std::printf("%s\n", failed ? "FAILED" : "ALL PASS");
  return failed ? 1 : 0;
}
