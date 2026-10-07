// TMA weight load for moe_mxfp4_sm100. Rows are padded to a 128-byte stride
// so the 128B swizzle matches the MMA operand.
//
//   nvcc -O3 -std=c++17 -arch=sm_100a --expt-relaxed-constexpr --expt-extended-lambda \
//     -I <repo>/include -I <repo>/deps/cutlass/include tma_sm100.cu -lcuda -o /tmp/mxfp4_tma

#include "mirage/persistent_kernel/tasks/blackwell/moe_mxfp4_sm100.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

constexpr int BATCH = 8;
constexpr int TOPK = 4;
constexpr int K = 2880;
constexpr int ROW = 1536;
constexpr int ORIG = 5760;
constexpr int NSLICE = 128;

#define CHECK(cmd)                                                             \
  do {                                                                         \
    cudaError_t e = (cmd);                                                     \
    if (e != cudaSuccess) {                                                    \
      std::printf("%s: %s\n", #cmd, cudaGetErrorString(e));                    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void launch(cute::bfloat16_t const *input, uint8_t const *blocks, uint8_t const *scales,
                       int32_t const *routing, int32_t const *mask, cute::bfloat16_t *output,
                       CUtensorMap const *tma, int expert_offset) {
  kernel::moe_mxfp4_sm100_task_impl<BATCH, NSLICE, ORIG, K, 4, TOPK, 4, ORIG, true, true, ROW>(
      input, blocks, scales, routing, mask, nullptr, output, expert_offset, tma);
}

__global__ void launch_grid(cute::bfloat16_t const *input, uint8_t const *blocks, uint8_t const *scales,
                            int32_t const *routing, int32_t const *mask, cute::bfloat16_t *output,
                            CUtensorMap const *tmas) {
  kernel::moe_mxfp4_sm100_task_impl<BATCH, NSLICE, ORIG, K, 4, TOPK, 4, ORIG, true, true, ROW>(
      input, blocks, scales, routing, mask, nullptr, output, static_cast<int>(blockIdx.x),
      tmas + blockIdx.y);
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

static int make_desc(CUtensorMap *desc, void *base, int rows) {
  uint64_t shape[5] = {ROW, static_cast<uint64_t>(rows), 1, 1, 1};
  uint64_t stride[4] = {ROW, 0, 0, 0};
  uint32_t box[5] = {128, 128, 1, 1, 1};
  uint32_t elem[5] = {1, 1, 1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, 5, base, shape, stride, box,
                                      elem, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    char const *s = nullptr;
    cuGetErrorString(r, &s);
    std::printf("cuTensorMapEncodeTiled: %s\n", s ? s : "?");
    return 1;
  }
  return 0;
}

int main() {
  CHECK(cudaFree(0));
  constexpr int nexp = 4;
  std::vector<float> act(BATCH * K);
  for (int t = 0; t < BATCH; ++t) {
    for (int k = 0; k < K; ++k) {
      act[t * K + k] = ((t * 3 + k * 5) % 17) * 0.25f - 1.0f;
    }
  }
  std::vector<cute::bfloat16_t> input(act.size());
  for (size_t i = 0; i < act.size(); ++i) {
    input[i] = cute::bfloat16_t(act[i]);
  }

  std::vector<uint8_t> blocks(static_cast<size_t>(nexp) * ORIG * ROW, 0);
  std::vector<uint8_t> scales(static_cast<size_t>(nexp) * ORIG * (K / 32));
  std::vector<uint8_t> tight(NSLICE * (K / 2));
  for (int e = 0; e < nexp; ++e) {
    for (int n = 0; n < NSLICE; ++n) {
      int row = e * ORIG + n;
      for (int k = 0; k < K; k += 2) {
        uint8_t byte = uint8_t(((row * 3 + k) & 15) | (((row + k) & 15) << 4));
        blocks[static_cast<size_t>(row) * ROW + k / 2] = byte;
        if (e == 0) {
          tight[n * (K / 2) + k / 2] = byte;
        }
      }
      for (int s = 0; s < K / 32; ++s) {
        scales[static_cast<size_t>(row) * (K / 32) + s] = uint8_t(127 + ((n + s) & 3));
      }
    }
  }
  std::vector<int32_t> routing(nexp * BATCH, 1), mask(5, 0);
  mask[4] = nexp;
  for (int e = 0; e < nexp; ++e) {
    mask[e] = e;
    for (int t = 0; t < BATCH; ++t) {
      routing[e * BATCH + t] = e + 1;
    }
  }

  cute::bfloat16_t *dI, *dO;
  uint8_t *dB, *dS;
  int32_t *dR, *dM;
  CHECK(cudaMalloc(&dI, input.size() * sizeof(cute::bfloat16_t)));
  CHECK(cudaMalloc(&dO, static_cast<size_t>(BATCH) * TOPK * ORIG * sizeof(cute::bfloat16_t)));
  CHECK(cudaMalloc(&dB, blocks.size()));
  CHECK(cudaMalloc(&dS, scales.size()));
  CHECK(cudaMalloc(&dR, routing.size() * 4));
  CHECK(cudaMalloc(&dM, mask.size() * 4));
  CHECK(cudaMemcpy(dI, input.data(), input.size() * sizeof(cute::bfloat16_t), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, blocks.data(), blocks.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dS, scales.data(), scales.size(), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dR, routing.data(), routing.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dM, mask.data(), mask.size() * 4, cudaMemcpyHostToDevice));

  CUtensorMap host_desc;
  if (make_desc(&host_desc, dB, nexp * ORIG)) {
    return 1;
  }
  CUtensorMap *dDesc;
  CHECK(cudaMalloc(&dDesc, sizeof(CUtensorMap)));
  CHECK(cudaMemcpy(dDesc, &host_desc, sizeof(CUtensorMap), cudaMemcpyHostToDevice));

  // More than half of the 228KB SM budget, so one CTA owns the SM's TMEM.
  int smem = 164 * 1024;
  CHECK(cudaFuncSetAttribute(launch, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  launch<<<1, 256, smem>>>(dI, dB, dS, dR, dM, dO, dDesc, 0);
  CHECK(cudaDeviceSynchronize());

  std::vector<cute::bfloat16_t> got(static_cast<size_t>(BATCH) * TOPK * ORIG);
  CHECK(cudaMemcpy(got.data(), dO, got.size() * sizeof(cute::bfloat16_t), cudaMemcpyDeviceToHost));
  double max_abs = 0;
  for (int t = 0; t < BATCH; ++t) {
    for (int c = 0; c < NSLICE; ++c) {
      float g = float(got[(static_cast<size_t>(t) * TOPK) * ORIG + c]);
      float ref = float(cute::bfloat16_t(float(
          raw_dot(act.data() + t * K, tight.data() + c * (K / 2), scales.data() + c * (K / 32)))));
      max_abs = std::max(max_abs, std::abs(double(g) - double(ref)));
    }
  }
  std::printf("%s tma w13 max_abs=%g\n", max_abs < 1e-3 ? "PASS" : "FAIL", max_abs);

  uint8_t *scratch;
  CHECK(cudaMalloc(&scratch, 256ull << 20));
  cudaEvent_t a, b;
  CHECK(cudaEventCreate(&a));
  CHECK(cudaEventCreate(&b));
  std::vector<float> samples;
  for (int i = 0; i < 20; ++i) {
    CHECK(cudaMemset(scratch, 0, 256ull << 20));
    CHECK(cudaEventRecord(a));
    launch<<<1, 256, smem>>>(dI, dB, dS, dR, dM, dO, dDesc, 0);
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    samples.push_back(ms * 1000.f);
  }
  std::sort(samples.begin(), samples.end());
  std::printf("grid1 cold median %.1f us\n", samples[samples.size() / 2]);

  std::vector<CUtensorMap> slices(45);
  for (int s = 0; s < 45; ++s) {
    void *base = dB + static_cast<size_t>(s) * NSLICE * ROW;
    if (make_desc(&slices[s], base, nexp * ORIG - s * NSLICE)) {
      return 1;
    }
  }
  CUtensorMap *dSlices;
  CHECK(cudaMalloc(&dSlices, slices.size() * sizeof(CUtensorMap)));
  CHECK(cudaMemcpy(dSlices, slices.data(), slices.size() * sizeof(CUtensorMap), cudaMemcpyHostToDevice));
  CHECK(cudaFuncSetAttribute(launch_grid, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  samples.clear();
  for (int i = 0; i < 20; ++i) {
    CHECK(cudaMemset(scratch, 0, 256ull << 20));
    CHECK(cudaEventRecord(a));
    launch_grid<<<dim3(4, 45), 256, smem>>>(dI, dB, dS, dR, dM, dO, dSlices);
    CHECK(cudaEventRecord(b));
    CHECK(cudaEventSynchronize(b));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, a, b));
    samples.push_back(ms * 1000.f);
  }
  std::sort(samples.begin(), samples.end());
  std::printf("grid 4x45 cold median %.1f us\n", samples[samples.size() / 2]);
  return max_abs < 1e-3 ? 0 : 1;
}
