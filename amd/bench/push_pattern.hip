// Microbenchmark: the KDA / MLA o_proj all-to-all over xGMI on MI355X as the layer does it (gemm_tile COMBINE_PUSH_ALL ->
// attn_residual's polls), by store pattern. Every GPU sends its partial [T][N] bf16 (8 x 7168: 114688 B) into its slot of
// every GPU's receive buffer; S sender workgroups (one per CU), workgroup w owning rows [R w, R w + R) (R = N / S); then each
// workgroup polls its share of the 8 slots of its own buffer until every word holds the iteration (the data is its own flag,
// system-scope loads), then the next iteration. An iteration = one hop of the slowest sender (the GPUs run in lockstep through
// the data). Timed per GPU with s_memrealtime over `iters` iterations.
// Patterns (all system-scope stores, sc0 sc1):
//   0 gemm8:  the layer's now: layout [T][N]; one wave; lane l (token l % 16 < T, rows R w + 16 i + 4 (l / 16) .. + 3) one
//             8-B store per 16-row tile i (32 lanes per instruction: T segments of 32 B)
//   1 row16:  layout [T][N]; vector v = token v / (R / 8), rows 8 (v % (R / 8)) .. + 7: 16-B stores (T segments of 2 R B)
//   2 tile16: layout [S][T][R] (a workgroup's T x R block contiguous): 16-B stores (one 2 T R-B segment)
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 push_pattern.hip -o push_pattern
// Run:   ./push_pattern [gpus=8] [iters=200]     -> one line per case, then a JSON summary on the last line
#include <string>

#include "common.h"

constexpr int MAX_GPUS = 8, T = 8, N = 7168;

struct Peers {
  unsigned char *recv[MAX_GPUS];  // recv[p] = GPU p's receive buffer: [G][T N] bf16 (slot g = from GPU g)
};

typedef unsigned v4u_t __attribute__((ext_vector_type(4)));
typedef unsigned v2u_t __attribute__((ext_vector_type(2)));

__device__ __forceinline__ void st16_sys(void *p, unsigned v) {
  v4u_t const w = {v, v, v, v};
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1" ::"v"(p), "v"(w) : "memory");
}
__device__ __forceinline__ void st8_sys(void *p, unsigned v) {
  v2u_t const w = {v, v};
  asm volatile("global_store_dwordx2 %0, %1, off sc0 sc1" ::"v"(p), "v"(w) : "memory");
}
__device__ __forceinline__ v4u_t load16_sys(void const *p) {
  v4u_t w;
  asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1\n s_waitcnt vmcnt(0)" : "=v"(w) : "v"(p) : "memory");
  return w;
}

template <int PAT>
__global__ void push_kernel(Peers peers, int me, int gpus, int senders, int iters, unsigned long long *t, unsigned *fail) {
  extern __shared__ char lds[];
  if (threadIdx.x == 0) lds[0] = 0;
  int const wg = blockIdx.x, lane = threadIdx.x & 63, R = N / senders;
  constexpr size_t SLOT = (size_t)T * N * 2;
  size_t const n16 = SLOT / 16, lo = n16 * wg / senders, hi = n16 * (wg + 1) / senders;  // this workgroup's polled share
  unsigned long long const t0 = realtime();
  for (int it = 1; it <= iters; ++it) {
    if (__hip_atomic_load(fail, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0) break;
    if (threadIdx.x < 64) {  // one wave sends (the layer: the result's wave)
      for (int r = 0; r < gpus; r++) {
        unsigned char *const base = peers.recv[r] + (size_t)me * SLOT;
        if (PAT == 0) {
          int const tok = lane % 16;
          if (tok < T) {
            for (int i = 0; i < R / 16; i++) st8_sys(base + ((size_t)tok * N + R * wg + 16 * i + 4 * (lane / 16)) * 2, (unsigned)it);
          }
        } else {
          for (int v = lane; v < T * R / 8; v += 64) {   // 16-B vector v of the workgroup's T x R block
            size_t const off = PAT == 1 ? ((size_t)(v / (R / 8)) * N + R * wg + 8 * (v % (R / 8))) * 2
                                        : (size_t)wg * T * R * 2 + (size_t)v * 16;
            st16_sys(base + off, (unsigned)it);
          }
        }
      }
    }
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
  size_t const slot = (size_t)T * N * 2;
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
        printf("{\"bench\": \"push_pattern\", \"error\": \"GPU %d cannot access GPU %d\"}\n", g, p);
        return 0;
      }
      CHECK(hipDeviceEnablePeerAccess(p, 0));
    }
    CHECK(hipMalloc(&peers.recv[g], gpus * slot));
    CHECK(hipMalloc(&t[g], 16));
    CHECK(hipMalloc(&fail[g], 4));
    CHECK(hipStreamCreateWithFlags(&streams[g], hipStreamNonBlocking));
    set_max_lds((const void *)push_kernel<0>, ONE_PER_CU_LDS);
    set_max_lds((const void *)push_kernel<1>, ONE_PER_CU_LDS);
    set_max_lds((const void *)push_kernel<2>, ONE_PER_CU_LDS);
  }
  double const tpn = wall_ticks_per_ns(0);
  char const *names[3] = {"gemm8", "row16", "tile16"};
  std::string json;
  for (int senders : {224, 112, 56}) {   // R = 32, 64, 128 rows per workgroup
    for (int pat : {0, 1, 2}) {
      for (int g = 0; g < gpus; ++g) {
        CHECK(hipSetDevice(g));
        CHECK(hipMemset(peers.recv[g], 0, gpus * slot));
        CHECK(hipMemset(t[g], 0, 16));
        CHECK(hipMemset(fail[g], 0, 4));
        CHECK(hipDeviceSynchronize());
      }
      for (int g = 0; g < gpus; ++g) {
        CHECK(hipSetDevice(g));
        auto k = pat == 0 ? push_kernel<0> : pat == 1 ? push_kernel<1> : push_kernel<2>;
        hipLaunchKernelGGL(k, dim3(senders), dim3(256), ONE_PER_CU_LDS, streams[g], peers, g, gpus, senders, iters, t[g], fail[g]);
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
      printf("%d GPUs, %zu B per GPU, %3d senders (%3d rows each), pattern %-6s: one hop %.2f us (slowest GPU; median %.2f)%s\n",
             gpus, slot, senders, N / senders, names[pat], worst / 1e3, median(per) / 1e3, fails ? " POLLS GAVE UP" : "");
      char row[200];
      snprintf(row, sizeof(row), "%s{\"senders\": %d, \"pattern\": \"%s\", \"slowest_us\": %.3f, \"median_us\": %.3f, \"fails\": %u}",
               json.empty() ? "" : ", ", senders, names[pat], worst / 1e3, median(per) / 1e3, fails);
      json += row;
    }
  }
  printf("{\"bench\": \"push_pattern\", \"gpus\": %d, \"iters\": %d, \"rows\": [%s]}\n", gpus, iters, json.c_str());
  return 0;
}
