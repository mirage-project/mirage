// l2_handoff.hip -- can a CU see another CU's plain store through their die's shared L2, and how fast, by load kind
// (gfx950). Workgroup 0 stores trial i's value with a plain store (no scope bits: its XCD's L2, write-back); workgroup 8
// (the same XCD when workgroups go round-robin over the 8 XCDs; both print their xcc_id) polls it with one of:
//   0: load sc0                     (workgroup scope)
//   1: buffer_inv sc0, plain load
//   2: buffer_inv sc1, plain load
//   3: load nt
//   4: load sc1                     (device scope: past L2; the plain store may not be there yet)
// A poll gives up after 20000 tries (counted as "never"). delay = reader's stamp - writer's stamp (s_memrealtime).
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 l2_handoff.hip -o l2_handoff
// Run:   ./l2_handoff [trials=200]
#include "common.h"

constexpr int MAXT = 512;

template <int KIND>
__device__ __forceinline__ unsigned poll_load(unsigned const *p) {
  unsigned v;
  if constexpr (KIND == 0) {
    asm volatile("global_load_dword %0, %1, off sc0\n\ts_waitcnt vmcnt(0)" : "=v"(v) : "v"(p) : "memory");
  } else if constexpr (KIND == 1) {
    asm volatile("buffer_inv sc0\n\tglobal_load_dword %0, %1, off\n\ts_waitcnt vmcnt(0)" : "=v"(v) : "v"(p) : "memory");
  } else if constexpr (KIND == 2) {
    asm volatile("buffer_inv sc1\n\tglobal_load_dword %0, %1, off\n\ts_waitcnt vmcnt(0)" : "=v"(v) : "v"(p) : "memory");
  } else if constexpr (KIND == 3) {
    asm volatile("global_load_dword %0, %1, off nt\n\ts_waitcnt vmcnt(0)" : "=v"(v) : "v"(p) : "memory");
  } else {
    asm volatile("global_load_dword %0, %1, off sc1\n\ts_waitcnt vmcnt(0)" : "=v"(v) : "v"(p) : "memory");
  }
  return v;
}

template <int KIND>
__global__ void probe(unsigned *data, unsigned *ready, unsigned long long *tw, unsigned long long *tr, unsigned *xcc,
                      int trials) {
  extern __shared__ char lds[];
  if (threadIdx.x == 0) lds[0] = 0;
  int const wg = blockIdx.x;
  if ((wg != 0 && wg != 8) || threadIdx.x != 0) return;
  if (wg == 0) xcc[0] = xcc_id();
  if (wg == 8) xcc[1] = xcc_id();
  for (int t = 0; t < trials; t++) {
    unsigned *const p = data + 64 * t;
    if (wg == 0) {
      while (__hip_atomic_load(ready + t, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == 0) __builtin_amdgcn_s_sleep(1);
      unsigned const h = (unsigned)t * 2654435761u;
      for (unsigned s = 0; s < (h >> 27); s++) __builtin_amdgcn_s_sleep(2);
      tw[t] = realtime();
      asm volatile("global_store_dword %0, %1, off" ::"v"(p), "v"((unsigned)t + 1) : "memory");
    } else {
      (void)poll_load<KIND>(p); // the line in this CU's L1 / the L2 before the store (the stale case)
      __hip_atomic_store(ready + t, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      int n = 0;
      while (poll_load<KIND>(p) != (unsigned)t + 1 && ++n < 20000) __builtin_amdgcn_s_sleep(1);
      tr[t] = n < 20000 ? realtime() : 0;
    }
  }
}

int main(int argc, char **argv) {
  int const trials = argc > 1 ? std::min(atoi(argv[1]), MAXT) : 200;
  print_device(0);
  double const tpn = wall_ticks_per_ns(0);
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, 0));
  unsigned *data, *ready, *xcc;
  unsigned long long *tw, *tr;
  CHECK(hipMalloc(&data, MAXT * 256));
  CHECK(hipMalloc(&ready, MAXT * 4));
  CHECK(hipMalloc(&xcc, 8));
  CHECK(hipMalloc(&tw, MAXT * 8));
  CHECK(hipMalloc(&tr, MAXT * 8));
  typedef void (*K)(unsigned *, unsigned *, unsigned long long *, unsigned long long *, unsigned *, int);
  K const ks[5] = {probe<0>, probe<1>, probe<2>, probe<3>, probe<4>};
  char const *names[5] = {"load sc0", "buffer_inv sc0 + load", "buffer_inv sc1 + load", "load nt", "load sc1"};
  for (int k = 0; k < 5; k++) {
    set_max_lds((const void *)ks[k], ONE_PER_CU_LDS);
    CHECK(hipMemset(data, 0, MAXT * 256));
    CHECK(hipMemset(ready, 0, MAXT * 4));
    hipLaunchKernelGGL(ks[k], dim3(p.multiProcessorCount), dim3(64), ONE_PER_CU_LDS, 0, data, ready, tw, tr, xcc, trials);
    CHECK(hipDeviceSynchronize());
    std::vector<unsigned long long> a(trials), b(trials);
    unsigned x[2];
    CHECK(hipMemcpy(a.data(), tw, trials * 8, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(b.data(), tr, trials * 8, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(x, xcc, 8, hipMemcpyDeviceToHost));
    std::vector<double> d;
    int never = 0;
    for (int t = 4; t < trials; t++) {
      if (!b[t]) {
        never++;
        continue;
      }
      d.push_back((double)(b[t] - a[t]) / tpn);
    }
    std::sort(d.begin(), d.end());
    printf("%-24s writer xcc %u reader xcc %u: seen %zu, never %d; delay median %.0f ns, p10 %.0f, p90 %.0f\n", names[k],
           x[0], x[1], d.size(), never, d.empty() ? 0.0 : median(d), d.empty() ? 0.0 : d[d.size() / 10],
           d.empty() ? 0.0 : d[d.size() * 9 / 10]);
  }
  return 0;
}
