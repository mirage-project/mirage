// poll_depth.hip -- how long a poller takes to see a value another CU stored (device scope), with D polls in flight
// (each poll a relaxed agent-scope load, the next issued S x 64 cycles after the previous; D = 1: one at a time, as the
// megakernel's poll loops). Workgroup 0 stores trial i's value at a pseudo-random time and stamps it; workgroup 1 polls
// and stamps when it sees it; delay = the two stamps' difference (s_memrealtime, one clock for the whole GPU). With
// load = 1 the other workgroups stream reads from a 2 GB buffer meanwhile (HBM busy, as during a layer).
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 poll_depth.hip -o poll_depth
// Run:   ./poll_depth [trials=400]        -> one line per (D, S, load): median / p90 delay in ns
#include "common.h"

constexpr int MAXT = 1024;
typedef unsigned v4u __attribute__((ext_vector_type(4)));

__device__ __forceinline__ unsigned ld_agent(unsigned const *p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

template <int D>
__device__ __forceinline__ void poll_for(unsigned const *p, unsigned want, int S) {
  unsigned r[D];
#pragma unroll
  for (int k = 0; k < D; k++) {
    r[k] = ld_agent(p);
    for (int s = 0; s < S; s++) __builtin_amdgcn_s_sleep(1);
  }
  for (;;) {
#pragma unroll
    for (int k = 0; k < D; k++) {
      if (r[k] == want) return;
      r[k] = ld_agent(p);
      if (S == 0) {
        __builtin_amdgcn_s_sleep(1);
      } else {
        for (int s = 0; s < S; s++) __builtin_amdgcn_s_sleep(1);
      }
    }
  }
}

template <int D>
__global__ void probe(unsigned *data, unsigned *ready, unsigned long long *tw, unsigned long long *tr, int trials, int S,
                      int load, v4u const *big, size_t nbig, unsigned *stop, unsigned *sink) {
  extern __shared__ char lds[];
  if (threadIdx.x == 0) lds[0] = 0;
  int const wg = blockIdx.x;
  if (wg >= 2) { // load generators
    if (!load) return;
    v4u acc = {0, 0, 0, 0};
    size_t i = (size_t)wg * blockDim.x + threadIdx.x, step = (size_t)gridDim.x * blockDim.x;
    while (__hip_atomic_load(stop, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == 0) {
#pragma unroll 4
      for (int k = 0; k < 64; k++) {
        v4u v = __builtin_nontemporal_load(big + i);
        acc.x ^= v.x;
        i += step;
        if (i >= nbig) i -= nbig;
      }
    }
    if (acc.x == 0x12345678u) sink[0] = acc.x;
    return;
  }
  if (threadIdx.x != 0) return;
  for (int t = 0; t < trials; t++) {
    if (wg == 0) { // writer
      while (ld_agent(ready + t) == 0) __builtin_amdgcn_s_sleep(1);
      unsigned const h = (unsigned)t * 2654435761u;
      for (unsigned s = 0; s < (h >> 26); s++) __builtin_amdgcn_s_sleep(2); // 0 .. 63 x 128 cycles
      tw[t] = realtime();
      __hip_atomic_store(data + 32 * t, (unsigned)t + 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    } else { // reader
      __hip_atomic_store(ready + t, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      poll_for<D>(data + 32 * t, (unsigned)t + 1, S);
      tr[t] = realtime();
    }
  }
  if (wg == 1) __hip_atomic_store(stop, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

int main(int argc, char **argv) {
  int const trials = argc > 1 ? std::min(atoi(argv[1]), MAXT) : 400;
  print_device(0);
  double const tpn = wall_ticks_per_ns(0);
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  int const ncu = p.multiProcessorCount;
  unsigned *data, *ready, *stop, *sink;
  unsigned long long *tw, *tr;
  v4u *big;
  size_t const nbig = (2ull << 30) / 16;
  CHECK(hipMalloc(&data, MAXT * 128));
  CHECK(hipMalloc(&ready, MAXT * 4));
  CHECK(hipMalloc(&stop, 4));
  CHECK(hipMalloc(&sink, 4));
  CHECK(hipMalloc(&tw, MAXT * 8));
  CHECK(hipMalloc(&tr, MAXT * 8));
  CHECK(hipMalloc(&big, nbig * 16));
  CHECK(hipMemset(big, 1, nbig * 16));
  set_max_lds((const void *)probe<1>, ONE_PER_CU_LDS);
  set_max_lds((const void *)probe<2>, ONE_PER_CU_LDS);
  set_max_lds((const void *)probe<4>, ONE_PER_CU_LDS);
  for (int load = 0; load < 2; load++)
    for (int D : {1, 2, 4})
      for (int S : {0, 4, 8, 16}) {
        if (D == 1 && S > 0) continue;
        CHECK(hipMemset(data, 0, MAXT * 128));
        CHECK(hipMemset(ready, 0, MAXT * 4));
        CHECK(hipMemset(stop, 0, 4));
        void (*k)(unsigned *, unsigned *, unsigned long long *, unsigned long long *, int, int, int, v4u const *, size_t,
                  unsigned *, unsigned *) = D == 1 ? probe<1> : D == 2 ? probe<2> : probe<4>;
        hipLaunchKernelGGL(k, dim3(ncu), dim3(256), ONE_PER_CU_LDS, 0, data, ready, tw, tr, trials, S, load, big, nbig, stop,
                           sink);
        CHECK(hipDeviceSynchronize());
        std::vector<unsigned long long> a(trials), b(trials);
        CHECK(hipMemcpy(a.data(), tw, trials * 8, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(b.data(), tr, trials * 8, hipMemcpyDeviceToHost));
        std::vector<double> d;
        for (int t = 8; t < trials; t++) d.push_back((double)(b[t] - a[t]) / tpn);
        std::sort(d.begin(), d.end());
        printf("load %d D %d S %2d: delay median %.0f ns, p10 %.0f, p90 %.0f\n", load, D, S, median(d), d[d.size() / 10],
               d[d.size() * 9 / 10]);
      }
  return 0;
}
