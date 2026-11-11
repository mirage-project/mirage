// Microbenchmark: matrix-instruction (MFMA) throughput on MI355X (gfx950) for the formats the K3 layers use.
//
// moe_experts multiplies MXFP8 activations by MXFP4 weights; the rest of the layer is BF16. On gfx950 the MX formats go through
// the scaled MFMA (v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4: A and B formats chosen per instruction, one E8M0 scale per
// 32 K-values). Measured: every CU runs 4 waves (one per SIMD), each a loop of independent MFMAs into 4 accumulators (so
// back-to-back MFMAs do not wait on each other); TFLOPS = 2*M*N*K * count / launch time. Cases:
//   bf16 16x16x32         v_mfma_f32_16x16x32_bf16
//   fp8 x fp4 16x16x128   A e4m3, B e2m1 (the expert GEMM's mix)
//   fp8 x fp8 16x16x128   both e4m3
//   fp4 x fp4 16x16x128   both e2m1
//   fp8 x fp4 32x32x64
// The question for moe_experts: does fp8 x fp4 run at the fp8 rate or the fp4 rate on this GPU.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 mfma_rate.hip -o mfma_rate
// Run:   ./mfma_rate [iters=20000]     -> one line per case, then a JSON summary on the last line
#include <string>

#include "common.h"

typedef int v8i __attribute__((ext_vector_type(8)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef float v16f __attribute__((ext_vector_type(16)));
typedef __bf16 v8bf __attribute__((ext_vector_type(8)));

enum Fmt { FP8 = 0, BF8 = 1, FP6 = 2, BF6 = 3, FP4 = 4 };  // the scaled MFMA's cbsz / blgp format codes

template <int FA, int FB>
__global__ void scaled16_kernel(const v8i *in, float *out, int iters) {
  v8i a = in[threadIdx.x % 64], b = in[64 + threadIdx.x % 64];
  int sa = 127 + threadIdx.x % 3, sb = 127;  // E8M0 scales (127 = 2^0)
  v4f c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
  for (int i = 0; i < iters; ++i) {
    c0 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c0, FA, FB, 0, sa, 0, sb);
    c1 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c1, FA, FB, 0, sa, 0, sb);
    c2 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c2, FA, FB, 0, sa, 0, sb);
    c3 = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, c3, FA, FB, 0, sa, 0, sb);
  }
  v4f c = c0 + c1 + c2 + c3;
  out[blockIdx.x * blockDim.x + threadIdx.x] = c[0] + c[1] + c[2] + c[3];
}

template <int FA, int FB>
__global__ void scaled32_kernel(const v8i *in, float *out, int iters) {
  v8i a = in[threadIdx.x % 64], b = in[64 + threadIdx.x % 64];
  int sa = 127, sb = 127;
  v16f c0 = {0}, c1 = {0};
  for (int i = 0; i < iters; ++i) {
    c0 = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c0, FA, FB, 0, sa, 0, sb);
    c1 = __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4(a, b, c1, FA, FB, 0, sa, 0, sb);
  }
  v16f c = c0 + c1;
  float s = 0;
  for (int j = 0; j < 16; ++j) s += c[j];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void bf16_kernel(const v8i *in, float *out, int iters) {
  v8bf a = __builtin_bit_cast(v8bf, *(const v4f *)&in[threadIdx.x % 64]);
  v8bf b = __builtin_bit_cast(v8bf, *(const v4f *)&in[64 + threadIdx.x % 64]);
  v4f c0 = {0}, c1 = {0}, c2 = {0}, c3 = {0};
  for (int i = 0; i < iters; ++i) {
    c0 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c0, 0, 0, 0);
    c1 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c1, 0, 0, 0);
    c2 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c2, 0, 0, 0);
    c3 = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, c3, 0, 0, 0);
  }
  v4f c = c0 + c1 + c2 + c3;
  out[blockIdx.x * blockDim.x + threadIdx.x] = c[0] + c[1] + c[2] + c[3];
}

int main(int argc, char **argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 20000;
  CHECK(hipSetDevice(0));
  print_device(0);
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int ncu = prop.multiProcessorCount, threads = 256;
  const struct {
    const char *name;
    void (*k)(const v8i *, float *, int);
    double flops_per_mfma;
    int mfma_per_iter;
  } cases[] = {
      {"bf16 16x16x32", bf16_kernel, 2.0 * 16 * 16 * 32, 4},
      {"fp8 x fp4 16x16x128", scaled16_kernel<FP8, FP4>, 2.0 * 16 * 16 * 128, 4},
      {"fp8 x fp8 16x16x128", scaled16_kernel<FP8, FP8>, 2.0 * 16 * 16 * 128, 4},
      {"fp4 x fp4 16x16x128", scaled16_kernel<FP4, FP4>, 2.0 * 16 * 16 * 128, 4},
      {"fp8 x fp4 32x32x64", scaled32_kernel<FP8, FP4>, 2.0 * 32 * 32 * 64, 2},
  };
  v8i *in;
  float *out;
  CHECK(hipMalloc(&in, 128 * sizeof(v8i)));
  CHECK(hipMemset(in, 0x11, 128 * sizeof(v8i)));
  CHECK(hipMalloc(&out, (size_t)ncu * threads * sizeof(float)));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  std::string json;
  for (auto &c : cases) {
    std::vector<double> tf;
    for (int rep = 0; rep < 4; ++rep) {
      CHECK(hipEventRecord(e0));
      hipLaunchKernelGGL(c.k, dim3(ncu), dim3(threads), 0, 0, in, out, iters);
      CHECK(hipEventRecord(e1));
      CHECK(hipDeviceSynchronize());
      float ms;
      CHECK(hipEventElapsedTime(&ms, e0, e1));
      double waves = (double)ncu * threads / 64;
      if (rep > 0) tf.push_back(waves * iters * c.mfma_per_iter * c.flops_per_mfma / (ms * 1e-3) / 1e12);
    }
    printf("%-22s %.0f TFLOPS (dense, median of %zu launches)\n", c.name, median(tf), tf.size());
    char row[160];
    snprintf(row, sizeof(row), "%s{\"case\": \"%s\", \"tflops\": %.1f}", json.empty() ? "" : ", ", c.name, median(tf));
    json += row;
  }
  printf("{\"bench\": \"mfma_rate\", \"iters\": %d, \"rows\": [%s]}\n", iters, json.c_str());
  return 0;
}
