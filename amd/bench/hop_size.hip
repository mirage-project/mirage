// Microbenchmark: one xGMI all-to-all hop on MI355X by payload size (push_pattern.hip's tile16 pattern, payload n columns).
// Every GPU sends a [T][n] bf16 block into its slot of every GPU's receive buffer (T n 2 B per peer); S sender workgroups (one
// per CU), workgroup w owning columns [R w, R w + R) (R = n / S), its T x R block contiguous, 16-B system-scope stores from
// one wave; then each workgroup polls its share of the slots of its own buffer until every word holds the iteration (the data
// is its own flag, system-scope loads), then the next iteration. An iteration = one hop of the slowest sender. n = 7168: the
// o_proj all-to-all now (115 KB per peer); n = 896: a reduce-scatter's share (one eighth of the row, 14 KB per peer).
// Polls (PAR): 0 = each of the 8 slots of a vector polled one after the other (8 round trips in a row: the first version,
// whose ~4 us per hop measured that, not xGMI: amd/bench/ping_pong.hip, a one-way signal between two GPUs is ~0.6 us);
// 1 = the 8 slots' loads in flight together, re-issued only for the slots not landed (as the kernels' landing loops).
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 hop_size.hip -o hop_size
// Run:   ./hop_size [gpus=8] [iters=200]     -> one line per case, then a JSON summary on the last line
#include <string>

#include "common.h"

constexpr int MAX_GPUS = 8, T = 8, NMAX = 7168;

struct Peers {
  unsigned char *recv[MAX_GPUS];  // recv[p] = GPU p's receive buffer: [G][T NMAX] bf16 (slot g = from GPU g)
};

typedef unsigned v4u_t __attribute__((ext_vector_type(4)));

__device__ __forceinline__ void st16_sys(void *p, unsigned v) {
  v4u_t const w = {v, v, v, v};
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1" ::"v"(p), "v"(w) : "memory");
}
__device__ __forceinline__ v4u_t load16_sys(void const *p) {
  v4u_t w;
  asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1\n s_waitcnt vmcnt(0)" : "=v"(w) : "v"(p) : "memory");
  return w;
}

__device__ __forceinline__ v4u_t load16_sys_nw(void const *p) { // (no wait: the caller waits)
  v4u_t w;
  asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1" : "=v"(w) : "v"(p) : "memory");
  return w;
}

template <int PAR>
__global__ void hop_kernel(Peers peers, int me, int gpus, int n, int senders, int iters, unsigned long long *t, unsigned *fail) {
  extern __shared__ char lds[];
  if (threadIdx.x == 0) lds[0] = 0;
  int const wg = blockIdx.x, lane = threadIdx.x & 63, R = n / senders;
  size_t const SLOT = (size_t)T * NMAX * 2, n16 = (size_t)T * n * 2 / 16, lo = n16 * wg / senders, hi = n16 * (wg + 1) / senders;
  unsigned long long const t0 = realtime();
  for (int it = 1; it <= iters; ++it) {
    if (__hip_atomic_load(fail, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0) break;
    if (threadIdx.x < 64) {
      for (int r = 0; r < gpus; r++) {
        unsigned char *const base = peers.recv[r] + (size_t)me * SLOT;
        for (int v = lane; v < T * R / 8; v += 64) st16_sys(base + (size_t)wg * T * R * 2 + (size_t)v * 16, (unsigned)it);
      }
    }
    if constexpr (PAR == 1) {
      for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        unsigned done = 0, all = (1u << gpus) - 1;
        long long spins = 0;
        while (done != all) {
          v4u_t w[MAX_GPUS];
#pragma unroll
          for (int r = 0; r < MAX_GPUS; r++) {
            if (r < gpus && !(done & (1u << r))) w[r] = load16_sys_nw(peers.recv[me] + (size_t)r * SLOT + i * 16);
          }
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
          for (int r = 0; r < MAX_GPUS; r++) {
            if (r < gpus && !(done & (1u << r)) && w[r].x >= (unsigned)it && w[r].y >= (unsigned)it && w[r].z >= (unsigned)it &&
                w[r].w >= (unsigned)it)
              done |= 1u << r;
          }
          if (done == all) break;
          if (++spins > (1ll << 16)) {
            atomicAdd(fail, 1u);
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        }
      }
    } else
    for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
      for (int r = 0; r < gpus; r++) {
        void const *p = peers.recv[me] + (size_t)r * SLOT + i * 16;
        long long spins = 0;
        for (;;) {
          v4u_t const w = load16_sys(p);
          if (w.x >= (unsigned)it && w.y >= (unsigned)it && w.z >= (unsigned)it && w.w >= (unsigned)it) break;
          if (++spins > (1ll << 16)) {
            atomicAdd(fail, 1u);
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicMax(&t[0], realtime() - t0);
}

int main(int argc, char **argv) {
  int ndev;
  CHECK(hipGetDeviceCount(&ndev));
  int const gpus = argc > 1 ? std::min(atoi(argv[1]), ndev) : std::min(ndev, MAX_GPUS);
  int const iters = argc > 2 ? atoi(argv[2]) : 200;
  setvbuf(stdout, nullptr, _IOLBF, 0);
  print_device(0);
  size_t const slot = (size_t)T * NMAX * 2;
  Peers peers;
  std::vector<unsigned long long *> t(gpus);
  std::vector<unsigned *> fail(gpus);
  std::vector<hipStream_t> streams(gpus);
  for (int g = 0; g < gpus; ++g) {
    CHECK(hipSetDevice(g));
    for (int p = 0; p < gpus; ++p) {
      if (p == g) continue;
      int can;
      CHECK(hipDeviceCanAccessPeer(&can, g, p));
      if (!can) {
        printf("{\"bench\": \"hop_size\", \"error\": \"GPU %d cannot access GPU %d\"}\n", g, p);
        return 0;
      }
      CHECK(hipDeviceEnablePeerAccess(p, 0));
    }
    CHECK(hipMalloc(&peers.recv[g], gpus * slot));
    CHECK(hipMalloc(&t[g], 16));
    CHECK(hipMalloc(&fail[g], 4));
    CHECK(hipStreamCreateWithFlags(&streams[g], hipStreamNonBlocking));
    set_max_lds((const void *)hop_kernel<0>, ONE_PER_CU_LDS);
    set_max_lds((const void *)hop_kernel<1>, ONE_PER_CU_LDS);
  }
  double const tpn = wall_ticks_per_ns(0);
  std::string json;
  int const cases[][2] = {{7168, 224}, {7168, 112}, {3584, 224}, {3584, 112}, {1792, 224}, {1792, 112}, {1792, 56},
                          {896, 112}, {896, 56},   {896, 28},   {448, 56},   {448, 28},   {128, 16},  {64, 8}};
  for (int par : {1, 0})
  for (auto const &c : cases) {
    int const n = c[0], senders = c[1];
    for (int g = 0; g < gpus; ++g) {
      CHECK(hipSetDevice(g));
      CHECK(hipMemset(peers.recv[g], 0, gpus * slot));
      CHECK(hipMemset(t[g], 0, 16));
      CHECK(hipMemset(fail[g], 0, 4));
      CHECK(hipDeviceSynchronize());
    }
    for (int g = 0; g < gpus; ++g) {
      CHECK(hipSetDevice(g));
      if (par)
        hipLaunchKernelGGL(hop_kernel<1>, dim3(senders), dim3(256), ONE_PER_CU_LDS, streams[g], peers, g, gpus, n, senders, iters,
                           t[g], fail[g]);
      else
        hipLaunchKernelGGL(hop_kernel<0>, dim3(senders), dim3(256), ONE_PER_CU_LDS, streams[g], peers, g, gpus, n, senders, iters,
                           t[g], fail[g]);
      CHECK(hipGetLastError());
    }
    std::vector<double> per;
    unsigned fails = 0;
    for (int g = 0; g < gpus; ++g) {
      CHECK(hipSetDevice(g));
      CHECK(hipStreamSynchronize(streams[g]));
      unsigned long long ticks;
      unsigned f;
      CHECK(hipMemcpy(&ticks, t[g], 8, hipMemcpyDeviceToHost));
      CHECK(hipMemcpy(&f, fail[g], 4, hipMemcpyDeviceToHost));
      per.push_back(ticks / tpn / iters);
      fails += f;
    }
    double const worst = *std::max_element(per.begin(), per.end());
    printf("%d GPUs, %6d B per peer (n %4d), %3d senders (%3d columns each), polls %s: one hop %.2f us (slowest GPU; median "
           "%.2f)%s\n", gpus, T * n * 2, n, senders, n / senders, par ? "parallel" : "serial  ", worst / 1e3, median(per) / 1e3,
           fails ? " POLLS GAVE UP" : "");
    char row[200];
    snprintf(row, sizeof(row), "%s{\"n\": %d, \"senders\": %d, \"par\": %d, \"slowest_us\": %.3f, \"median_us\": %.3f, \"fails\": %u}",
             json.empty() ? "" : ", ", n, senders, par, worst / 1e3, median(per) / 1e3, fails);
    json += row;
  }
  printf("{\"bench\": \"hop_size\", \"gpus\": %d, \"iters\": %d, \"rows\": [%s]}\n", gpus, iters, json.c_str());
  return 0;
}
