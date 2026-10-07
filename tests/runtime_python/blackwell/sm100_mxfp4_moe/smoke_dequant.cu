// Host+device check of the MXFP4 dequant the SM100 expert GEMM uses.
// Runs on the machine's GPU (sm_90 is enough: no tcgen05 here).
//
//   nvcc -O2 -std=c++17 -arch=sm_90a \
//     -I <repo>/include smoke_dequant.cu -o /tmp/mxfp4_smoke && /tmp/mxfp4_smoke

#include "mirage/persistent_kernel/tasks/blackwell/mxfp4.cuh"

#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstdint>

__global__ void gemm_kernel(const __nv_bfloat16 *__restrict__ x,
                            const uint8_t *__restrict__ blocks,
                            const uint8_t *__restrict__ scales,
                            const __nv_bfloat16 *__restrict__ bias,
                            __nv_bfloat16 *__restrict__ y,
                            int batch, int n, int k) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int token = blockIdx.y;
  if (col >= n || token >= batch) {
    return;
  }
  const uint8_t *row_b = blocks + static_cast<size_t>(col) * (k / 2);
  const uint8_t *row_s = scales + static_cast<size_t>(col) * (k / 32);
  const __nv_bfloat16 *xv = x + static_cast<size_t>(token) * k;
  float acc = 0.f;
  for (int i = 0; i < k; ++i) {
    acc += __bfloat162float(xv[i]) * kernel::mxfp4::dequant(row_b, row_s, i);
  }
  if (bias) {
    acc += __bfloat162float(bias[col]);
  }
  y[static_cast<size_t>(token) * n + col] = __float2bfloat16(acc);
}

static float ref_dequant(const uint8_t *bytes, const uint8_t *scales, int k) {
  return kernel::mxfp4::dequant(bytes, scales, k);
}

int main() {
  constexpr int B = 4, N = 128, K = 256;
  std::vector<uint8_t> blocks(N * (K / 2)), scales(N * (K / 32));
  std::vector<__nv_bfloat16> x(B * K), bias(N), y(B * N), y_ref(B * N);
  for (int i = 0; i < (int)blocks.size(); ++i) {
    blocks[i] = static_cast<uint8_t>((i * 17 + 3) & 0xff);
  }
  for (int i = 0; i < (int)scales.size(); ++i) {
    scales[i] = static_cast<uint8_t>(120 + (i % 15));
  }
  for (int i = 0; i < B * K; ++i) {
    x[i] = __float2bfloat16(((i % 9) - 4) * 0.25f);
  }
  for (int i = 0; i < N; ++i) {
    bias[i] = __float2bfloat16(((i % 5) - 2) * 0.5f);
  }

  // Nibble order and the 32-wide scale group, including a negative E2M1 code.
  {
    uint8_t byte = 0x1A; // high=1 -> 0.5, low=10 -> -1.0
    uint8_t sc = 128;    // 2^(128-127) = 2
    float even = ref_dequant(&byte, &sc, 0);
    float odd = ref_dequant(&byte, &sc, 1);
    if (even != -2.f || odd != 1.f) {
      std::printf("FAIL nibble/scale even=%g odd=%g\n", even, odd);
      return 1;
    }
  }

  for (int t = 0; t < B; ++t) {
    for (int col = 0; col < N; ++col) {
      float acc = 0.f;
      for (int k = 0; k < K; ++k) {
        acc += __bfloat162float(x[t * K + k]) *
               ref_dequant(blocks.data() + col * (K / 2),
                           scales.data() + col * (K / 32), k);
      }
      acc += __bfloat162float(bias[col]);
      y_ref[t * N + col] = __float2bfloat16(acc);
    }
  }

  uint8_t *d_blocks = nullptr, *d_scales = nullptr;
  __nv_bfloat16 *d_x = nullptr, *d_bias = nullptr, *d_y = nullptr;
  cudaMalloc(&d_blocks, blocks.size());
  cudaMalloc(&d_scales, scales.size());
  cudaMalloc(&d_x, x.size() * sizeof(__nv_bfloat16));
  cudaMalloc(&d_bias, bias.size() * sizeof(__nv_bfloat16));
  cudaMalloc(&d_y, y.size() * sizeof(__nv_bfloat16));
  cudaMemcpy(d_blocks, blocks.data(), blocks.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_scales, scales.data(), scales.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias.data(), bias.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
  gemm_kernel<<<dim3((N + 63) / 64, B), 64>>>(d_x, d_blocks, d_scales, d_bias, d_y, B, N, K);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::printf("FAIL cuda %s\n", cudaGetErrorString(err));
    return 1;
  }
  cudaMemcpy(y.data(), d_y, y.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);

  int mismatches = 0;
  float max_abs = 0.f;
  for (int i = 0; i < B * N; ++i) {
    float a = __bfloat162float(y[i]);
    float b = __bfloat162float(y_ref[i]);
    float d = fabsf(a - b);
    if (d > max_abs) {
      max_abs = d;
    }
    if (d > 0.f) {
      ++mismatches;
    }
  }
  if (mismatches) {
    std::printf("FAIL gemm mismatches=%d max_abs=%g\n", mismatches, max_abs);
    return 1;
  }
  std::printf("PASS mxfp4 dequant+gemm  B=%d N=%d K=%d  (bit-exact bf16)\n", B, N, K);
  return 0;
}
