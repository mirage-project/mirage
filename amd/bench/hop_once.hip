// hop_once.hip (11-09): one xGMI all-to-all push after a quiet period, as the kernels' all-reduces (one burst, then nothing on
// the links), instead of hop_big.hip's back-to-back rounds. Per round: a cross-GPU barrier (workgroup 0 of every GPU writes the
// round into every GPU's flag slot with system-scope stores and polls all of them; the other workgroups poll the local copy),
// then every workgroup spins QUIET_US, then pushes its [T][R] bf16 block into its slot of every GPU's receive buffer (two sets,
// 0xFF prefilled, the other set reset with st16_dev first: the kernels' exchange protocol) and polls its share until no word
// is 0xFFFFFFFF. Time per round and GPU: the latest workgroup's (landed - pushes started), wall clock. MI355X 11-09 glmprof28:
// in the kernel the receivers see the pushed data ~12 us after every sender's pushes were acknowledged, at T 16 / 32 / 64.
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 -DT_ROWS=64 hop_once.hip -o hop_once      Run: ./hop_once [gpus=4] [rounds=50]
#include <string>

#include "common.h"

#ifndef T_ROWS
#define T_ROWS 64
#endif
constexpr int MAX_GPUS = 8, T = T_ROWS, N = 4096;
constexpr size_t SLOT = (size_t)T * N * 2;

struct Peers {
  unsigned char *recv[MAX_GPUS]; // GPU p's receive buffer: [2 sets][G slots][T N] bf16
  unsigned *flag[MAX_GPUS];      // GPU p's barrier flags: [G] (written by GPU g at slot g)
};

typedef unsigned v4u_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ void st16_sys(void *p, unsigned v) {
  v4u_t const w = {v, v, v, v};
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1" ::"v"(p), "v"(w) : "memory");
}
__device__ __forceinline__ void st16_dev(void *p, unsigned v) {
  v4u_t const w = {v, v, v, v};
  asm volatile("global_store_dwordx4 %0, %1, off sc1\n s_nop 1" ::"v"(p), "v"(w) : "memory");
}
__device__ __forceinline__ v4u_t ld16_sys_nw(void const *p) {
  v4u_t w;
  asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1" : "=v"(w) : "v"(p) : "memory");
  return w;
}
__device__ __forceinline__ unsigned ld32_sys(unsigned const *p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

__global__ void hop_once(Peers peers, int me, int gpus, int senders, int rounds, int quiet_ticks, unsigned *local_go,
                         unsigned long long *t_sum, unsigned long long *t_max, unsigned *fail) {
  int const wg = blockIdx.x, R = N / senders;
  size_t const n16 = SLOT / 16, lo = n16 * wg / senders, hi = n16 * (wg + 1) / senders;
  for (int it = 1; it <= rounds; ++it) {
    size_t const SET = (size_t)(it & 1) * gpus * SLOT, OTHER = (size_t)((it + 1) & 1) * gpus * SLOT;
    // the other set's share reset (landed in the previous round by this workgroup)
    for (int r = 0; r < gpus; r++)
      for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) st16_dev(peers.recv[me] + OTHER + (size_t)r * SLOT + i * 16, ~0u);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    __syncthreads();
    // cross-GPU barrier
    if (wg == 0 && threadIdx.x == 0) {
      for (int r = 0; r < gpus; r++) __hip_atomic_store(peers.flag[r] + me, (unsigned)it, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      for (int r = 0; r < gpus; r++)
        while (ld32_sys(peers.flag[me] + r) < (unsigned)it) __builtin_amdgcn_s_sleep(1);
      __hip_atomic_store(local_go, (unsigned)it, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    if (threadIdx.x == 0)
      while (__hip_atomic_load(local_go, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) < (unsigned)it) __builtin_amdgcn_s_sleep(1);
    __syncthreads();
    // quiet period
    unsigned long long const q0 = wall_clock64();
    while (wall_clock64() - q0 < (unsigned long long)quiet_ticks) __builtin_amdgcn_s_sleep(8);
    __syncthreads();
    unsigned long long const t0 = wall_clock64();
    // push (every thread, each vector to every GPU in turn: the finisher's pattern)
    for (int v = threadIdx.x; v < T * R / 8; v += blockDim.x)
      for (int r = 0; r < gpus; r++)
        st16_sys(peers.recv[r] + SET + (size_t)me * SLOT + (size_t)wg * T * R * 2 + (size_t)v * 16, (unsigned)it);
    // land this workgroup's share of every slot
    for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      unsigned done = 0, all = (1u << gpus) - 1;
      long long spins = 0;
      while (done != all) {
        v4u_t w[MAX_GPUS];
#pragma unroll
        for (int r = 0; r < MAX_GPUS; r++)
          if (r < gpus && !(done & (1u << r))) w[r] = ld16_sys_nw(peers.recv[me] + SET + (size_t)r * SLOT + i * 16);
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
        for (int r = 0; r < MAX_GPUS; r++)
          if (r < gpus && !(done & (1u << r)) && w[r].x != ~0u && w[r].y != ~0u && w[r].z != ~0u && w[r].w != ~0u) done |= 1u << r;
        if (done == all) break;
        if (++spins > (1ll << 18)) {
          atomicAdd(fail, 1u);
          break;
        }
        __builtin_amdgcn_s_sleep(1);
      }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      unsigned long long const dt = wall_clock64() - t0;
      atomicMax(&t_max[it], dt);
    }
  }
  (void)t_sum;
}

int main(int argc, char **argv) {
  int ndev;
  CHECK(hipGetDeviceCount(&ndev));
  int const gpus = argc > 1 ? std::min(atoi(argv[1]), ndev) : std::min(ndev, 4);
  int const rounds = argc > 2 ? atoi(argv[2]) : 50;
  setvbuf(stdout, nullptr, _IOLBF, 0);
  print_device(0);
  Peers peers;
  std::vector<unsigned *> go(gpus), fail(gpus);
  std::vector<unsigned long long *> tmax(gpus), tsum(gpus);
  std::vector<hipStream_t> streams(gpus);
  for (int g = 0; g < gpus; ++g) {
    CHECK(hipSetDevice(g));
    for (int p = 0; p < gpus; ++p)
      if (p != g) CHECK(hipDeviceEnablePeerAccess(p, 0));
    CHECK(hipMalloc(&peers.recv[g], 2 * gpus * SLOT));
    CHECK(hipMalloc(&peers.flag[g], 64 * 4));
    CHECK(hipMalloc(&go[g], 4));
    CHECK(hipMalloc(&fail[g], 4));
    CHECK(hipMalloc(&tmax[g], (rounds + 1) * 8));
    CHECK(hipMalloc(&tsum[g], 8));
    CHECK(hipStreamCreateWithFlags(&streams[g], hipStreamNonBlocking));
    set_max_lds((const void *)hop_once, ONE_PER_CU_LDS);
  }
  double const tpn = wall_ticks_per_ns(0);
  for (int quiet_us : {0, 5, 20, 50})
    for (int senders : {64, 128}) {
      for (int g = 0; g < gpus; ++g) {
        CHECK(hipSetDevice(g));
        CHECK(hipMemset(peers.recv[g], 0xFF, 2 * gpus * SLOT));
        CHECK(hipMemset(peers.flag[g], 0, 64 * 4));
        CHECK(hipMemset(go[g], 0, 4));
        CHECK(hipMemset(fail[g], 0, 4));
        CHECK(hipMemset(tmax[g], 0, (rounds + 1) * 8));
        CHECK(hipDeviceSynchronize());
      }
      int const qt = (int)(quiet_us * 1000.0 * tpn);
      for (int g = 0; g < gpus; ++g) {
        CHECK(hipSetDevice(g));
        hipLaunchKernelGGL(hop_once, dim3(senders), dim3(256), ONE_PER_CU_LDS, streams[g], peers, g, gpus, senders, rounds, qt, go[g],
                           tsum[g], tmax[g], fail[g]);
        CHECK(hipGetLastError());
      }
      std::vector<double> per; // per GPU: median over rounds 2.. of the latest workgroup's push -> landed
      unsigned fails = 0;
      for (int g = 0; g < gpus; ++g) {
        CHECK(hipSetDevice(g));
        CHECK(hipStreamSynchronize(streams[g]));
        std::vector<unsigned long long> h(rounds + 1);
        unsigned f;
        CHECK(hipMemcpy(h.data(), tmax[g], (rounds + 1) * 8, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(&f, fail[g], 4, hipMemcpyDeviceToHost));
        fails += f;
        std::vector<double> v;
        for (int it = 2; it <= rounds; it++) v.push_back(h[it] / tpn / 1e3);
        per.push_back(median(v));
      }
      printf("T %d, %zu KB per peer, %3d senders, quiet %2d us: push -> landed (latest workgroup, median of rounds) per GPU", T,
             SLOT / 1024, senders, quiet_us);
      for (double x : per) printf(" %.2f", x);
      printf(" us%s\n", fails ? " POLLS GAVE UP" : "");
    }
  return 0;
}
