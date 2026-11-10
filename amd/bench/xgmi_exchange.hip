// Microbenchmark: an all-to-all exchange between the GPUs of one MI355X node over xGMI, in the MoE layer's shape.
//
// On B300 the layer's exchanges use NVLS multicast (one store reaches all 8 GPUs). MI355X has no multicast: each GPU stores
// its block into every peer's receive buffer through peer pointers, then posts a flag on each peer (system-scope release);
// each GPU waits until all G-1 peers' flags arrived. One exchange per iteration, `iters` iterations in one launch (the flag
// value counts iterations), timed per GPU with s_memrealtime (the GPUs' clocks are not compared with each other).
// Sizes: 28 KB (8 tokens x 3584 MXFP8 values, the latent z), 112 KB (8 x 7168 bf16, a hidden-size row block), 1 MB.
// Senders: `senders` workgroups per GPU split the peers and the bytes (16-byte stores).
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 xgmi_exchange.hip -o xgmi_exchange
// Run:   ./xgmi_exchange [gpus=all] [iters=200]       -> one line per case, then a JSON summary on the last line
#include <string>

#include "common.h"

constexpr int MAX_GPUS = 8;

struct Peers {
  uint4 *recv[MAX_GPUS];     // recv[p] = GPU p's receive buffer: [G][bytes] (slot g = from GPU g)
  unsigned *flag[MAX_GPUS];  // flag[p] = GPU p's arrival counter
};

__global__ void exchange_kernel(Peers peers, const uint4 *src, int me, int gpus, size_t bytes, int senders, int iters,
                                unsigned long long *t) {
  extern __shared__ char lds[];
  if (threadIdx.x == 0) lds[0] = 0;
  int wg = blockIdx.x;
  if (wg >= senders) return;
  size_t n16 = bytes / 16;
  unsigned long long t0 = realtime();
  for (int it = 1; it <= iters; ++it) {
    // each peer's block is cut into `senders` parts; the (peer, part) pairs k = 0 .. (gpus-1)*senders-1 are dealt round-robin
    // over the workgroups (gpus-1 pairs each)
    for (int k = wg; k < (gpus - 1) * senders; k += senders) {
      int peer_i = k % (gpus - 1), part = k / (gpus - 1), parts = senders;
      int peer = peer_i >= me ? peer_i + 1 : peer_i;
      size_t chunk = (n16 + parts - 1) / parts, lo = part * chunk, hi = lo + chunk < n16 ? lo + chunk : n16;
      uint4 *dst = peers.recv[peer] + (size_t)me * n16;
      for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) dst[i] = src[i];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      // one post per (peer, part) pair served: each GPU's flag grows by (gpus-1) * senders per iteration
      __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
      for (int k = wg; k < (gpus - 1) * senders; k += senders) {
        int peer_i = k % (gpus - 1);
        int peer = peer_i >= me ? peer_i + 1 : peer_i;
        __hip_atomic_fetch_add(peers.flag[peer], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
      }
      if (wg == 0) {
        unsigned want = (unsigned)it * (gpus - 1) * senders;
        while (__hip_atomic_load(peers.flag[me], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < want) __builtin_amdgcn_s_sleep(0);
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "");
      }
    }
    // the other sender workgroups wait for workgroup 0 to see the iteration complete before reusing the buffers
    if (threadIdx.x == 0 && wg != 0) {
      unsigned want = (unsigned)it * (gpus - 1) * senders;
      while (__hip_atomic_load(peers.flag[me], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM) < want) __builtin_amdgcn_s_sleep(0);
    }
    __syncthreads();
  }
  if (wg == 0 && threadIdx.x == 0) t[0] = realtime() - t0;
}

int main(int argc, char **argv) {
  int ndev;
  CHECK(hipGetDeviceCount(&ndev));
  int gpus = argc > 1 ? std::min(atoi(argv[1]), ndev) : std::min(ndev, MAX_GPUS);
  int iters = argc > 2 ? atoi(argv[2]) : 200;
  print_device(0);
  if (gpus < 2) {
    printf("{\"bench\": \"xgmi_exchange\", \"error\": \"needs 2+ GPUs, have %d\"}\n", ndev);
    return 0;
  }
  const size_t sizes[] = {28 << 10, 112 << 10, 1 << 20};
  const size_t max_bytes = 1 << 20;
  Peers peers;
  std::vector<uint4 *> src(gpus);
  std::vector<unsigned long long *> t(gpus);
  std::vector<hipStream_t> streams(gpus);
  for (int g = 0; g < gpus; ++g) {
    CHECK(hipSetDevice(g));
    for (int p = 0; p < gpus; ++p) {
      if (p == g) continue;
      int can;
      CHECK(hipDeviceCanAccessPeer(&can, g, p));
      if (!can) {
        printf("{\"bench\": \"xgmi_exchange\", \"error\": \"GPU %d cannot access GPU %d\"}\n", g, p);
        return 0;
      }
      CHECK(hipDeviceEnablePeerAccess(p, 0));
    }
    CHECK(hipMalloc(&peers.recv[g], gpus * max_bytes));
    CHECK(hipMalloc(&peers.flag[g], 256));
    CHECK(hipMalloc(&src[g], max_bytes));
    CHECK(hipMemset(src[g], g + 1, max_bytes));
    CHECK(hipMalloc(&t[g], sizeof(unsigned long long)));
    CHECK(hipStreamCreateWithFlags(&streams[g], hipStreamNonBlocking));
    set_max_lds((const void *)exchange_kernel, ONE_PER_CU_LDS);
  }
  double tpn = wall_ticks_per_ns(0);
  std::string json;
  for (size_t bytes : sizes) {
    for (int senders : {7, 14, 28, 56, 112}) {
      for (int g = 0; g < gpus; ++g) {
        CHECK(hipSetDevice(g));
        CHECK(hipMemset(peers.flag[g], 0, 256));
        CHECK(hipDeviceSynchronize());
      }
      for (int g = 0; g < gpus; ++g) {
        CHECK(hipSetDevice(g));
        hipLaunchKernelGGL(exchange_kernel, dim3(senders), dim3(256), ONE_PER_CU_LDS, streams[g], peers, src[g], g, gpus, bytes,
                           senders, iters, t[g]);
      }
      std::vector<double> per;
      for (int g = 0; g < gpus; ++g) {
        CHECK(hipSetDevice(g));
        CHECK(hipStreamSynchronize(streams[g]));
        unsigned long long ticks;
        CHECK(hipMemcpy(&ticks, t[g], sizeof(ticks), hipMemcpyDeviceToHost));
        per.push_back(ticks / tpn / iters);
      }
      double worst = *std::max_element(per.begin(), per.end());
      printf("%d GPUs, %4zu KB to each peer, %3d sender workgroups: one exchange %.2f us (slowest GPU; median GPU %.2f us)\n", gpus,
             bytes >> 10, senders, worst / 1e3, median(per) / 1e3);
      char row[200];
      snprintf(row, sizeof(row), "%s{\"gpus\": %d, \"bytes\": %zu, \"senders\": %d, \"slowest_us\": %.3f, \"median_us\": %.3f}",
               json.empty() ? "" : ", ", gpus, bytes, senders, worst / 1e3, median(per) / 1e3);
      json += row;
    }
  }
  printf("{\"bench\": \"xgmi_exchange\", \"iters\": %d, \"rows\": [%s]}\n", iters, json.c_str());
  return 0;
}
