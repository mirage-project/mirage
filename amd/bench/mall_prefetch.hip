// amd/bench/mall_prefetch.hip -- does a buffer read once just before a kernel (the 256 MB memory-side cache, MALL) make that
// kernel's reads of it faster? One GPU. Per round: stream X (512 MB, evicts W from the L2s and the MALL), optionally read W
// once (the prefetch), then time a read of W (all CUs, 16-B loads, as gemm_tile's weight loads).
//   hipcc -O3 --offload-arch=gfx950 amd/bench/mall_prefetch.hip -o /tmp/mallp && /tmp/mallp [W MB=70] [rounds=20] [gap MB=0]
// gap MB > 0: between the prefetch and the timed read, a stream of a third buffer Y of that size (the kernel in between)
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                                                                  \
  do {                                                                                                                         \
    hipError_t e_ = (x);                                                                                                       \
    if (e_ != hipSuccess) {                                                                                                    \
      printf("HIP error %s at %d\n", hipGetErrorString(e_), __LINE__);                                                         \
      exit(1);                                                                                                                 \
    }                                                                                                                          \
  } while (0)

// 8 independent 16-B loads in flight per thread (U = 8), then the remainder one at a time
__global__ void __launch_bounds__(256) rd(uint4 const *p, size_t n, unsigned *sink) {
  unsigned acc = 0;
  size_t const st = (size_t)gridDim.x * 256;
  size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
  for (; i + 7 * st < n; i += 8 * st) {
    uint4 v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) {
      v[k] = p[i + k * st]; // (plain loads: cached as the weight loads are)
    }
#pragma unroll
    for (int k = 0; k < 8; k++) {
      acc ^= v[k].x ^ v[k].y ^ v[k].z ^ v[k].w;
    }
  }
  for (; i < n; i += st) {
    uint4 const v = p[i];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) {
    sink[0] = acc;
  }
}

int main(int argc, char **argv) {
  size_t const wmb = argc > 1 ? atoi(argv[1]) : 70, rounds = argc > 2 ? atoi(argv[2]) : 20, gmb = argc > 3 ? atoi(argv[3]) : 0;
  size_t const xb = 512ull << 20, wb = wmb << 20, gb = gmb << 20;
  uint4 *x, *w, *y = nullptr;
  unsigned *sink;
  CK(hipMalloc(&x, xb));
  CK(hipMalloc(&w, wb));
  if (gb) {
    CK(hipMalloc(&y, gb));
    CK(hipMemset(y, 1, gb));
  }
  CK(hipMalloc(&sink, 64));
  CK(hipMemset(x, 1, xb));
  CK(hipMemset(w, 2, wb));
  int ncu = 0;
  CK(hipDeviceGetAttribute(&ncu, hipDeviceAttributeMultiprocessorCount, 0));
  int const grid = ncu * 4;
  hipEvent_t a, b;
  CK(hipEventCreate(&a));
  CK(hipEventCreate(&b));
  for (int pre = 0; pre < 2; pre++) {
    std::vector<float> us;
    for (size_t r = 0; r < rounds + 2; r++) {
      rd<<<grid, 256>>>(x, xb / 16, sink);
      if (pre) {
        rd<<<grid, 256>>>(w, wb / 16, sink);
      }
      if (gb) {
        rd<<<grid, 256>>>(y, gb / 16, sink);
      }
      CK(hipEventRecord(a));
      rd<<<grid, 256>>>(w, wb / 16, sink);
      CK(hipEventRecord(b));
      CK(hipEventSynchronize(b));
      float ms = 0;
      CK(hipEventElapsedTime(&ms, a, b));
      if (r >= 2) {
        us.push_back(ms * 1e3f);
      }
    }
    std::sort(us.begin(), us.end());
    float const med = us[us.size() / 2];
    printf("W %zu MB, %s%s: read median %.1f us (min %.1f, max %.1f) = %.2f TB/s\n", wmb,
           pre ? "read once just before" : "cold (after a 512 MB stream)", gb ? " + a stream between" : "", med, us[0],
           us.back(), wb / (med * 1e-6) / 1e12);
  }
  return 0;
}
