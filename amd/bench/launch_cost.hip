// Microbenchmark: what a kernel's shape adds to its launch cost in a HIP graph on MI355X (gfx950). Question (10-24, dec251 /
// dec255): an empty fused library chained in a graph costs ~7.1 us per launch (span 0.5) while plain empty kernels in a graph run
// ~2.5 us apart (launch_gap.hip); the fused kernels have one 256-thread workgroup per CU, ~150 KB of LDS each, a few KB of kernel
// arguments by value (Maps + G) and, when they spill, a scratch segment. Which of these costs dispatch / completion time?
//
// N kernels of the same variant back to back in one HIP graph; each stamps (s_memrealtime) its first workgroup's start and its last
// workgroup's end (as launch_gap.hip). Reported per variant: the median gap between kernels, and the graph's wall time / N (events).
// Variants (grid = one 256-thread workgroup per CU, body ~1 us):
//   plain    no LDS, 16 B of arguments
//   lds      ~150 KB of dynamic LDS
//   karg     a 3 KB argument struct by value
//   scratch  a per-thread array indexed at run time (private segment, ScratchSize > 0)
//   all      lds + karg + scratch
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 launch_cost.hip -o launch_cost
// Run:   ./launch_cost [kernels=200]
#include <functional>
#include <string>

#include "common.h"

struct BigArg {
  unsigned w[768]; // 3 KB
};

__device__ __forceinline__ void body(unsigned long long *starts, unsigned long long *ends, int i, unsigned long long ticks) {
  if (threadIdx.x != 0) return;
  unsigned long long const t0 = realtime();
  atomicMin(&starts[i], t0);
  while (realtime() - t0 < ticks) __builtin_amdgcn_s_sleep(1);
  atomicMax(&ends[i], realtime());
}

template <bool LDS, bool KARG, bool SCRATCH>
__global__ void __launch_bounds__(256, 1) cost_kernel(unsigned long long *starts, unsigned long long *ends, int i,
                                                      unsigned long long ticks, BigArg big, unsigned *sink) {
  extern __shared__ char lds[];
  if constexpr (LDS) {
    if (threadIdx.x == 0) lds[0] = 0;
  }
  if constexpr (KARG) {
    if (big.w[threadIdx.x & 767] == 0x9e3779b9u) sink[0] = 1;
  }
  if constexpr (SCRATCH) {
    volatile unsigned a[64];
    for (int k = 0; k < 64; k++) a[k] = k ^ threadIdx.x;
    if (a[(threadIdx.x * 7 + i) & 63] == 0x9e3779b9u) sink[1] = 1;
  }
  body(starts, ends, i, ticks);
}

template <int NW>
__global__ void __launch_bounds__(256, 1) cost_scr(unsigned long long *starts, unsigned long long *ends, int i, unsigned long long ticks,
                                                   unsigned *sink) {
  volatile unsigned a[NW]; // NW x 4 B of scratch per thread
  for (int k = 0; k < NW; k++) a[k] = k ^ threadIdx.x;
  if (a[(threadIdx.x * 7 + i) % NW] == 0x9e3779b9u) sink[1] = 1;
  body(starts, ends, i, ticks);
}
__device__ __noinline__ unsigned callee_scr(int i) { // scratch only in a callee
  volatile unsigned a[8];
  for (int k = 0; k < 8; k++) a[k] = k ^ threadIdx.x;
  return a[(threadIdx.x * 7 + i) & 7];
}
__global__ void __launch_bounds__(256, 1) cost_callee(unsigned long long *starts, unsigned long long *ends, int i, unsigned long long ticks,
                                                      unsigned *sink) {
  if (callee_scr(i) == 0x9e3779b9u) sink[1] = 1;
  body(starts, ends, i, ticks);
}

template <bool LDS, bool KARG, bool SCRATCH>
__global__ void __launch_bounds__(256, 1) cost_small(unsigned long long *starts, unsigned long long *ends, int i,
                                                     unsigned long long ticks, unsigned *sink) {
  extern __shared__ char lds[];
  if constexpr (LDS) {
    if (threadIdx.x == 0) lds[0] = 0;
  }
  if constexpr (SCRATCH) {
    volatile unsigned a[64];
    for (int k = 0; k < 64; k++) a[k] = k ^ threadIdx.x;
    if (a[(threadIdx.x * 7 + i) & 63] == 0x9e3779b9u) sink[1] = 1;
  }
  body(starts, ends, i, ticks);
}

int main(int argc, char **argv) {
  int const n = argc > 1 ? atoi(argv[1]) : 200;
  CHECK(hipSetDevice(0));
  print_device(0);
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int const grid = prop.multiProcessorCount;
  double const tpn = wall_ticks_per_ns(0);
  unsigned long long const ticks = (unsigned long long)(1000.0 * tpn);
  int const LDSB = 150 * 1024;
  set_max_lds((const void *)cost_kernel<true, false, false>, LDSB);
  set_max_lds((const void *)cost_kernel<true, true, true>, LDSB);
  set_max_lds((const void *)cost_small<true, false, false>, LDSB);
  unsigned long long *starts, *ends;
  unsigned *sink;
  CHECK(hipMalloc(&starts, n * 8));
  CHECK(hipMalloc(&ends, n * 8));
  CHECK(hipMalloc(&sink, 16));
  BigArg big;
  for (int k = 0; k < 768; k++) big.w[k] = k;
  hipStream_t s;
  CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  struct V {
    const char *name;
    std::function<void(int)> launch;
  };
  std::vector<V> vs = {
      {"plain", [&](int i) { hipLaunchKernelGGL((cost_small<false, false, false>), dim3(grid), dim3(256), 0, s, starts, ends, i, ticks, sink); }},
      {"lds", [&](int i) { hipLaunchKernelGGL((cost_small<true, false, false>), dim3(grid), dim3(256), LDSB, s, starts, ends, i, ticks, sink); }},
      {"karg", [&](int i) { hipLaunchKernelGGL((cost_kernel<false, true, false>), dim3(grid), dim3(256), 0, s, starts, ends, i, ticks, big, sink); }},
      {"scratch", [&](int i) { hipLaunchKernelGGL((cost_small<false, false, true>), dim3(grid), dim3(256), 0, s, starts, ends, i, ticks, sink); }},
      {"scr32B", [&](int i) { hipLaunchKernelGGL((cost_scr<8>), dim3(grid), dim3(256), 0, s, starts, ends, i, ticks, sink); }},
      {"scr16B", [&](int i) { hipLaunchKernelGGL((cost_scr<4>), dim3(grid), dim3(256), 0, s, starts, ends, i, ticks, sink); }},
      {"callee", [&](int i) { hipLaunchKernelGGL(cost_callee, dim3(grid), dim3(256), 0, s, starts, ends, i, ticks, sink); }},
      {"all", [&](int i) { hipLaunchKernelGGL((cost_kernel<true, true, true>), dim3(grid), dim3(256), LDSB, s, starts, ends, i, ticks, big, sink); }},
  };
  for (auto &v : vs) {
    hipGraph_t g;
    hipGraphExec_t exec;
    CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    for (int i = 0; i < n; ++i) v.launch(i);
    CHECK(hipStreamEndCapture(s, &g));
    CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
    CHECK(hipGraphDestroy(g));
    std::vector<double> gaps, spans, per;
    for (int rep = 0; rep < 4; ++rep) { // rep 0 warms up
      CHECK(hipMemsetAsync(starts, 0xff, n * 8, s));
      CHECK(hipMemsetAsync(ends, 0, n * 8, s));
      CHECK(hipEventRecord(e0, s));
      CHECK(hipGraphLaunch(exec, s));
      CHECK(hipEventRecord(e1, s));
      CHECK(hipStreamSynchronize(s));
      float ms;
      CHECK(hipEventElapsedTime(&ms, e0, e1));
      std::vector<unsigned long long> st(n), en(n);
      CHECK(hipMemcpy(st.data(), starts, n * 8, hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(en.data(), ends, n * 8, hipMemcpyDeviceToHost));
      if (rep == 0) continue;
      per.push_back(ms * 1e6 / n);
      for (int i = 0; i + 1 < n; ++i) gaps.push_back(((double)st[i + 1] - (double)en[i]) / tpn);
      for (int i = 0; i < n; ++i) spans.push_back(((double)en[i] - (double)st[i]) / tpn);
    }
    printf("%-8s grid %d x 256: gap between kernels %.0f ns, span %.0f ns, graph time per kernel %.0f ns\n", v.name, grid,
           median(gaps), median(spans), median(per));
    CHECK(hipGraphExecDestroy(exec));
  }
  return 0;
}
