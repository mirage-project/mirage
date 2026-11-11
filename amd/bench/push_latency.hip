// Microbenchmark: one hop of the MoE layer's all-reduce over xGMI on MI355X, as the layer does it (allreduce_send ->
// latent_up's polls): every GPU pushes its message (B bytes) into its slot of every GPU's receive buffer with 16-byte
// system-scope stores, and polls the 8 slots of its own buffer until all landed; the data is its own flag (no fence, no
// counter: each 4-byte word carries the iteration number). xgmi_exchange.hip measures the flag-based exchange instead.
//
// S workgroups per GPU (one per CU): workgroup w owns 16-byte vectors [n w / S, n (w + 1) / S) of the message; per
// iteration it stores them to the 8 GPUs, then polls the same vectors of the 8 slots of its GPU's buffer until each word
// holds the iteration, then goes on. So an iteration is one hop of the slowest of the 8 senders of a vector (the GPUs run
// in lockstep through the data itself). Timed per GPU with s_memrealtime over `iters` iterations.
// Store kinds: sys (sc0 sc1, the layer's st16_sys), nt (nt sc0 sc1). The polls: sc0 sc1 loads. (A plain "nt" store
// has no system scope: it may stay in the sender's L2, and the receiver never sees it.)
// Background: the CUs not sending stream 1 GB of HBM with non-temporal loads during the iterations (bg 1), as moe's W2
// tail does on some CUs while the first R slices go out.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 push_latency.hip -o push_latency
// Run:   ./push_latency [gpus=8] [iters=200]     -> one line per case, then a JSON summary on the last line
#include <string>

#include "common.h"

constexpr int MAX_GPUS = 8;

struct Peers {
  uint4 *recv[MAX_GPUS];  // recv[p] = GPU p's receive buffer: [G][n16] (slot g = from GPU g)
};

typedef unsigned v4u_t __attribute__((ext_vector_type(4)));
typedef int v4i_t __attribute__((ext_vector_type(4)));

template <int KIND>
__device__ __forceinline__ void store16(uint4 *p, unsigned v) {
  v4u_t const w = {v, v, v, v};
  if constexpr (KIND == 0) {
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1" ::"v"(p), "v"(w) : "memory");
  } else {
    asm volatile("global_store_dwordx4 %0, %1, off nt sc0 sc1" ::"v"(p), "v"(w) : "memory");
  }
}
__device__ __forceinline__ v4u_t load16_sys(uint4 const *p) {
  v4u_t w;
  asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1\n s_waitcnt vmcnt(0)" : "=v"(w) : "v"(p) : "memory");
  return w;
}

template <int KIND>
__global__ void push_kernel(Peers peers, int me, int gpus, int n16, int senders, int iters, int bg, const v4i_t *bgbuf,
                            size_t bg16, unsigned long long *t, unsigned *fail, unsigned *sink) {
  extern __shared__ char lds[];
  if (threadIdx.x == 0) lds[0] = 0;
  int const wg = blockIdx.x;
  if (wg >= senders) {  // background readers (bg 1): stream bgbuf until the senders are done
    if (!bg) return;
    int const nr = gridDim.x - senders, r = wg - senders;
    size_t const chunk = (bg16 + nr - 1) / nr, lo = r * chunk, hi = lo + chunk < bg16 ? lo + chunk : bg16;
    unsigned acc = 0;
    for (size_t i = lo + threadIdx.x;; i += 8 * 256) {  // round and round its chunk until the senders are done
      if (i >= hi) i = lo + threadIdx.x;
      v4i_t v[8];
#pragma unroll
      for (int j = 0; j < 8; j++) v[j] = i + j * 256 < hi ? __builtin_nontemporal_load(bgbuf + i + j * 256) : v4i_t{0, 0, 0, 0};
#pragma unroll
      for (int j = 0; j < 8; j++) acc ^= (unsigned)v[j][j & 3];
      if (__hip_atomic_load(&t[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0) break;
    }
    if (acc == 0x9e3779b9u) sink[0] = acc;
    return;
  }
  int const lo = (int)((long long)n16 * wg / senders), hi = (int)((long long)n16 * (wg + 1) / senders);
  unsigned long long const t0 = realtime();
  for (int it = 1; it <= iters; ++it) {
    if (__hip_atomic_load(fail, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0) break;  // a poll gave up: the case is void
    for (int i = lo + (int)threadIdx.x; i < hi; i += blockDim.x) {
#pragma unroll
      for (int r = 0; r < MAX_GPUS; r++) {
        if (r < gpus) store16<KIND>(peers.recv[r] + (size_t)me * n16 + i, (unsigned)it);
      }
    }
    for (int i = lo + (int)threadIdx.x; i < hi; i += blockDim.x) {
      for (int r = 0; r < gpus; r++) {
        uint4 const *p = peers.recv[me] + (size_t)r * n16 + i;
        long long spins = 0;
        for (;;) {
          v4u_t const w = load16_sys(p);
          if (w.x >= (unsigned)it && w.y >= (unsigned)it && w.z >= (unsigned)it && w.w >= (unsigned)it) break;
          if (++spins > (1ll << 16)) {  // ~tens of ms: give up, count it
            atomicAdd(fail, 1u);
            break;
          }
          __builtin_amdgcn_s_sleep(1);
        }
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    atomicMax(&t[0], realtime() - t0);
    __hip_atomic_store(&t[1], 1ull, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
}

int main(int argc, char **argv) {
  int ndev;
  CHECK(hipGetDeviceCount(&ndev));
  int const gpus = argc > 1 ? std::min(atoi(argv[1]), ndev) : std::min(ndev, MAX_GPUS);
  int const iters = argc > 2 ? atoi(argv[2]) : 200;
  setvbuf(stdout, nullptr, _IOLBF, 0);  // a line at a time: a run cut short keeps the cases done
  print_device(0);
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  int const ncu = prop.multiProcessorCount;
  const size_t max_bytes = 64 << 10, BG = 1ull << 30;
  Peers peers;
  std::vector<unsigned long long *> t(gpus);
  std::vector<unsigned *> fail(gpus), sink(gpus);
  std::vector<v4i_t *> bgbuf(gpus);
  std::vector<hipStream_t> streams(gpus);
  for (int g = 0; g < gpus; ++g) {
    CHECK(hipSetDevice(g));
    for (int p = 0; p < gpus; ++p) {
      if (p == g) continue;
      int can;
      CHECK(hipDeviceCanAccessPeer(&can, g, p));
      if (!can) {
        printf("{\"bench\": \"push_latency\", \"error\": \"GPU %d cannot access GPU %d\"}\n", g, p);
        return 0;
      }
      CHECK(hipDeviceEnablePeerAccess(p, 0));
    }
    CHECK(hipMalloc(&peers.recv[g], gpus * max_bytes));
    CHECK(hipMalloc(&t[g], 16));
    CHECK(hipMalloc(&fail[g], 4));
    CHECK(hipMalloc(&sink[g], 16));
    CHECK(hipMalloc(&bgbuf[g], BG));
    CHECK(hipMemset(bgbuf[g], 1, BG));
    CHECK(hipStreamCreateWithFlags(&streams[g], hipStreamNonBlocking));
    set_max_lds((const void *)push_kernel<0>, ONE_PER_CU_LDS);
    set_max_lds((const void *)push_kernel<1>, ONE_PER_CU_LDS);
  }
  double const tpn = wall_ticks_per_ns(0);
  std::string json;
  // 57344 B: R, 8 tokens x 3584 bf16; 14336 B: latent_up's output block of a GPU, 8 x 896 bf16
  for (size_t bytes : {(size_t)57344, (size_t)14336}) {
    for (int kind : {0, 1}) {
      for (int bg : {0, 1}) {
        for (int senders : {8, 32, 64, 128, 256}) {
          if (bg && senders == ncu) continue;
          int const n16 = (int)(bytes / 16);
          for (int g = 0; g < gpus; ++g) {
            CHECK(hipSetDevice(g));
            CHECK(hipMemset(peers.recv[g], 0, gpus * max_bytes));
            CHECK(hipMemset(t[g], 0, 16));
            CHECK(hipMemset(fail[g], 0, 4));
            CHECK(hipDeviceSynchronize());
          }
          int const grid = bg ? ncu : senders;
          for (int g = 0; g < gpus; ++g) {
            CHECK(hipSetDevice(g));
            if (kind == 0)
              hipLaunchKernelGGL(push_kernel<0>, dim3(grid), dim3(256), ONE_PER_CU_LDS, streams[g], peers, g, gpus, n16, senders,
                                 iters, bg, bgbuf[g], BG / 16, t[g], fail[g], sink[g]);
            else
              hipLaunchKernelGGL(push_kernel<1>, dim3(grid), dim3(256), ONE_PER_CU_LDS, streams[g], peers, g, gpus, n16, senders,
                                 iters, bg, bgbuf[g], BG / 16, t[g], fail[g], sink[g]);
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
          printf("%d GPUs, %5zu B per GPU, store %-3s, background %s, %3d sender CUs: one hop %.2f us (slowest GPU; median %.2f)%s\n",
                 gpus, bytes, kind ? "nt" : "sys", bg ? "HBM stream" : "none", senders, worst / 1e3, median(per) / 1e3,
                 fails ? " POLLS GAVE UP" : "");
          char row[240];
          snprintf(row, sizeof(row),
                   "%s{\"bytes\": %zu, \"store\": \"%s\", \"bg\": %d, \"senders\": %d, \"slowest_us\": %.3f, \"median_us\": %.3f, "
                   "\"fails\": %u}",
                   json.empty() ? "" : ", ", bytes, kind ? "nt" : "sys", bg, senders, worst / 1e3, median(per) / 1e3, fails);
          json += row;
        }
      }
    }
  }
  printf("{\"bench\": \"push_latency\", \"gpus\": %d, \"iters\": %d, \"rows\": [%s]}\n", gpus, iters, json.c_str());
  return 0;
}
