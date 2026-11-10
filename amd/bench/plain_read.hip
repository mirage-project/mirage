#include <hip/hip_runtime.h>
#include <cstdio>
__global__ void rd(const uint4 *p, size_t n, unsigned *sink) {
  unsigned acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { uint4 v = p[i]; acc ^= v.x ^ v.w; }
  if (acc == 0x12345u) sink[0] = acc;
}
int main() {
  size_t GB = 1ull << 30; uint4 *a, *b; unsigned *s;
  hipMalloc(&a, GB); hipMalloc(&b, GB); hipMalloc(&s, 4); hipMemset(a, 1, GB); hipMemset(b, 2, GB);
  hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
  for (int r = 0; r < 3; ++r) {
    rd<<<1024, 256>>>(b, GB / 16, s);
    hipEventRecord(e0); rd<<<1024, 256>>>(a, GB / 16, s); hipEventRecord(e1); hipDeviceSynchronize();
    float ms; hipEventElapsedTime(&ms, e0, e1);
    printf("plain read 1 GB: %.1f us = %.2f TB/s (%s)\n", ms * 1e3, GB / (ms * 1e-3) / 1e12, hipGetErrorString(hipGetLastError()));
  }
}
