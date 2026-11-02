// Microbenchmark: what one dependency hand-off costs on MI355X (gfx950).
//
// The static megakernel's tasks wait on counters in global memory: the producer's stores, then a release; the consumer polls,
// then an acquire. Measured as a ping-pong between two workgroups (one round trip = two hand-offs), for the placements a plan
// can produce:
//   same die     both workgroups on one XCD (one shared 4 MB L2)
//   other die    different XCDs (the flag crosses the Infinity Fabric)
//   other GPU    workgroup a on GPU 0, b on GPU 1, the flag in GPU 1's memory (xGMI, system scope)
// and three ways to post and poll (agent scope):
//   release      store-release (gfx950: buffer_wbl2 = write back this XCD's L2, then the store) / relaxed polls + one acquire
//   relaxed      plain atomic store, no L2 write-back: the flag's own cost (enough for a flag; not for data written before it)
//   nt poll      store-release / polls with non-temporal loads (Fleet's cross-XCD polling)
// Also: where each workgroup of a full launch runs (die), whether that is the same in a second launch, and the clock's step.
//
// Every workgroup holds 96 KB of LDS, so at most one runs per CU and all of them are resident at once (a persistent launch).
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 sync_latency.hip -o sync_latency
// Run:   ./sync_latency [iters=10000] [pairs per kind=4]      -> one line per measurement, then a JSON summary on the last line
#include <map>

#include "common.h"

enum Post { POST_RELEASE = 0, POST_RELAXED = 1 };
enum Poll { POLL_ATOMIC = 0, POLL_NT = 1 };

// where[wg] = {XCC id, HW_ID}
__global__ void where_kernel(unsigned *where) {
  extern __shared__ char lds[];
  if (threadIdx.x == 0) {
    lds[0] = 0;
    where[2 * blockIdx.x] = xcc_id();
    where[2 * blockIdx.x + 1] = hw_id();
  }
}

// Poll, then one acquire fence (at agent scope on gfx950 an acquire is an L2 invalidate, buffer_inv sc1)
template <int SCOPE, int POLL>
__device__ inline void wait_for(unsigned *flag, unsigned value) {
  if constexpr (POLL == POLL_NT) {
    while (__builtin_nontemporal_load(flag) < value) __builtin_amdgcn_s_sleep(0);
  } else {
    while (__hip_atomic_load(flag, __ATOMIC_RELAXED, SCOPE) < value) __builtin_amdgcn_s_sleep(0);
  }
  if constexpr (SCOPE == __HIP_MEMORY_SCOPE_SYSTEM) {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
  } else {
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
  }
}

template <int SCOPE, int POST>
__device__ inline void post(unsigned *flag, unsigned value) {
  __hip_atomic_store(flag, value, POST == POST_RELEASE ? __ATOMIC_RELEASE : __ATOMIC_RELAXED, SCOPE);
}

// a posts 2i+1 and waits for 2i+2; b waits for 2i+1 and posts 2i+2; a stamps the start and the end. role: 0 both a and b in
// this launch; 1 only a; 2 only b (two GPUs: one launch per GPU).
template <int SCOPE, int POST, int POLL>
__global__ void pingpong_kernel(unsigned *flag, int a, int b, int iters, int role, unsigned long long *t) {
  extern __shared__ char lds[];
  if (threadIdx.x != 0) return;
  lds[0] = 0;
  bool is_a = blockIdx.x == a && role != 2, is_b = blockIdx.x == b && role != 1;
  if (is_a) {
    post<SCOPE, POST>(flag, 1);  // warm-up: both sides polling before the timed round trips
    wait_for<SCOPE, POLL>(flag, 2);
    unsigned long long t0 = realtime();
    for (int i = 1; i <= iters; ++i) {
      post<SCOPE, POST>(flag, 2 * i + 1);
      wait_for<SCOPE, POLL>(flag, 2 * i + 2);
    }
    t[0] = realtime() - t0;
  } else if (is_b) {
    for (int i = 0; i <= iters; ++i) {
      wait_for<SCOPE, POLL>(flag, 2 * i + 1);
      post<SCOPE, POST>(flag, 2 * i + 2);
    }
  }
}

// back-to-back clock reads: the smallest nonzero step (ticks) and the span of n reads
__global__ void clock_kernel(unsigned long long *out, int n) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  unsigned long long prev = realtime(), first = prev, min_step = ~0ull;
  for (int i = 0; i < n; ++i) {
    unsigned long long now = realtime();
    if (now != prev && now - prev < min_step) min_step = now - prev;
    prev = now;
  }
  out[0] = min_step;
  out[1] = prev - first;
}

struct Sample {
  const char *kind, *variant;
  int a, b;
  double ns_round_trip;
};

using Kernel = void (*)(unsigned *, int, int, int, int, unsigned long long *);

static double run_pair(Kernel k, unsigned *flag, int a, int b, int nwg, int iters, double tpn, unsigned long long *t) {
  CHECK(hipMemset(flag, 0, sizeof(unsigned)));
  hipLaunchKernelGGL(k, dim3(nwg), dim3(64), ONE_PER_CU_LDS, 0, flag, a, b, iters, 0, t);
  CHECK(hipDeviceSynchronize());
  unsigned long long ticks;
  CHECK(hipMemcpy(&ticks, t, sizeof(ticks), hipMemcpyDeviceToHost));
  return ticks / tpn / iters;
}

int main(int argc, char **argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 10000, per_kind = argc > 2 ? atoi(argv[2]) : 4;
  int ndev;
  CHECK(hipGetDeviceCount(&ndev));
  CHECK(hipSetDevice(0));
  print_device(0);
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int nwg = prop.multiProcessorCount;
  double tpn = wall_ticks_per_ns(0);
  const struct {
    const char *name;
    Kernel k;
  } variants[] = {
      {"release", pingpong_kernel<__HIP_MEMORY_SCOPE_AGENT, POST_RELEASE, POLL_ATOMIC>},
      {"relaxed", pingpong_kernel<__HIP_MEMORY_SCOPE_AGENT, POST_RELAXED, POLL_ATOMIC>},
      {"nt poll", pingpong_kernel<__HIP_MEMORY_SCOPE_AGENT, POST_RELEASE, POLL_NT>},
  };
  Kernel sys = pingpong_kernel<__HIP_MEMORY_SCOPE_SYSTEM, POST_RELEASE, POLL_ATOMIC>;
  set_max_lds((const void *)where_kernel, ONE_PER_CU_LDS);
  set_max_lds((const void *)sys, ONE_PER_CU_LDS);
  for (auto &v : variants) set_max_lds((const void *)v.k, ONE_PER_CU_LDS);

  // 1. the clock
  unsigned long long *t;
  CHECK(hipMalloc(&t, 4 * sizeof(unsigned long long)));
  hipLaunchKernelGGL(clock_kernel, dim3(1), dim3(64), 0, 0, t, 100000);
  CHECK(hipDeviceSynchronize());
  unsigned long long clk[2];
  CHECK(hipMemcpy(clk, t, sizeof(clk), hipMemcpyDeviceToHost));
  printf("clock: smallest step %llu ticks = %.1f ns; 100000 back-to-back reads span %.1f us\n", clk[0], clk[0] / tpn,
         clk[1] / tpn / 1e3);

  // 2. where each workgroup of a full launch runs, in two launches
  unsigned *where;
  CHECK(hipMalloc(&where, 2 * nwg * sizeof(unsigned)));
  std::vector<unsigned> w1(2 * nwg), w2(2 * nwg);
  for (auto *w : {&w1, &w2}) {
    hipLaunchKernelGGL(where_kernel, dim3(nwg), dim3(64), ONE_PER_CU_LDS, 0, where);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(w->data(), where, 2 * nwg * sizeof(unsigned), hipMemcpyDeviceToHost));
  }
  std::map<unsigned, std::vector<int>> by_xcc;
  int moved = 0;
  for (int i = 0; i < nwg; ++i) {
    by_xcc[w1[2 * i]].push_back(i);
    moved += w1[2 * i] != w2[2 * i];
  }
  printf("workgroup -> XCC: %zu dies;", by_xcc.size());
  for (auto &[x, wgs] : by_xcc)
    printf(" XCC %u: %zu wgs (first %d %d %d)", x, wgs.size(), wgs[0], wgs.size() > 1 ? wgs[1] : -1, wgs.size() > 2 ? wgs[2] : -1);
  printf("\nworkgroups on another die in the second launch: %d of %d\n", moved, nwg);

  // 3. same die / other die, GPU 0, each variant
  unsigned *flag;
  CHECK(hipMalloc(&flag, 256));
  std::vector<Sample> samples;
  std::vector<unsigned> dies;
  for (auto &[x, wgs] : by_xcc) dies.push_back(x);
  for (auto &v : variants) {
    for (int p = 0; p < per_kind; ++p) {
      auto &wgs = by_xcc[dies[p % dies.size()]];
      if (wgs.size() < 2) break;
      int a = wgs[0], b = wgs[1 + p % (wgs.size() - 1)];
      samples.push_back({"same die", v.name, a, b, run_pair(v.k, flag, a, b, nwg, iters, tpn, t)});
    }
    for (int p = 0; p < per_kind && dies.size() >= 2; ++p) {
      int a = by_xcc[dies[0]][p % by_xcc[dies[0]].size()], b = by_xcc[dies[1 + p % (dies.size() - 1)]][0];
      samples.push_back({"other die", v.name, a, b, run_pair(v.k, flag, a, b, nwg, iters, tpn, t)});
    }
  }

  // 4. other GPU: a on GPU 0, b on GPU 1, the flag in GPU 1's memory; one launch per GPU, system scope
  if (ndev >= 2) {
    int can;
    CHECK(hipDeviceCanAccessPeer(&can, 0, 1));
    if (can) {
      CHECK(hipDeviceEnablePeerAccess(1, 0));
      CHECK(hipSetDevice(1));
      set_max_lds((const void *)sys, ONE_PER_CU_LDS);
      unsigned *rflag;
      unsigned long long *t1;
      hipStream_t s0, s1;
      CHECK(hipMalloc(&rflag, 256));
      CHECK(hipMalloc(&t1, 4 * sizeof(unsigned long long)));
      CHECK(hipStreamCreate(&s1));
      CHECK(hipSetDevice(0));
      CHECK(hipStreamCreate(&s0));
      for (int p = 0; p < per_kind; ++p) {
        CHECK(hipSetDevice(1));
        CHECK(hipMemset(rflag, 0, sizeof(unsigned)));
        CHECK(hipDeviceSynchronize());
        hipLaunchKernelGGL(sys, dim3(nwg), dim3(64), ONE_PER_CU_LDS, s1, rflag, -1, p, iters, 2, t1);
        CHECK(hipSetDevice(0));
        hipLaunchKernelGGL(sys, dim3(nwg), dim3(64), ONE_PER_CU_LDS, s0, rflag, p, -1, iters, 1, t);
        CHECK(hipStreamSynchronize(s0));
        CHECK(hipSetDevice(1));
        CHECK(hipStreamSynchronize(s1));
        CHECK(hipSetDevice(0));
        unsigned long long ticks;
        CHECK(hipMemcpy(&ticks, t, sizeof(ticks), hipMemcpyDeviceToHost));
        samples.push_back({"other GPU", "release", p, p, ticks / tpn / iters});
      }
    } else {
      printf("GPU 0 cannot access GPU 1's memory: no other-GPU measurement\n");
    }
  }

  for (auto &s : samples)
    printf("%-10s %-8s wg %3d <-> wg %3d: round trip %.0f ns (one hand-off %.0f ns)\n", s.kind, s.variant, s.a, s.b,
           s.ns_round_trip, s.ns_round_trip / 2);
  printf("{\"bench\": \"sync_latency\", \"iters\": %d, \"clock_step_ns\": %.1f, \"dies\": %zu, \"moved\": %d, \"samples\": [", iters,
         clk[0] / tpn, by_xcc.size(), moved);
  for (size_t i = 0; i < samples.size(); ++i)
    printf("%s{\"kind\": \"%s\", \"variant\": \"%s\", \"a\": %d, \"b\": %d, \"round_trip_ns\": %.1f}", i ? ", " : "",
           samples[i].kind, samples[i].variant, samples[i].a, samples[i].b, samples[i].ns_round_trip);
  printf("]}\n");
  return 0;
}
