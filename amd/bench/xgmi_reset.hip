// Microbenchmark: one xGMI hop of data-as-flag (as push_latency.hip) when the receiver resets its own receive buffer to 0xFF in
// the kernel before the senders write it, as the layer does (core.cuh kernel_begin: each launch resets the next launch's
// exchange set with st16_dev). Question (MI355X 10-24, dec256 / dec257): every R push was acked by all GPUs ~8-10 us before
// latent_up's polls saw the data, whenever the polls started; does a reset by the receiver's own stores leave a copy that the
// system-scope polls read until it is evicted?
//
// 8 GPUs in lockstep through the data; S workgroups per GPU (one per CU), workgroup w owns 16-byte vectors [n w / S, n (w+1) / S)
// of the message. Two buffer sets, iteration i uses set i & 1. Per iteration, each workgroup:
//   1. resets its vectors of set (i + 1) & 1 of its own buffer (every GPU's slot) to 0xFF with store kind RK, waits for them
//      (RK 0: no reset, the set keeps iteration i - 1's values; 1: sc1 stores (st16_dev); 2: sc0 sc1 (st16_sys); 3: plain stores)
//   2. stores its vectors of the message (every word = i) into slot `me` of set i & 1 of every GPU (sc0 sc1, as push16)
//   3. polls its vectors of every slot of set i & 1 of its own buffer (sc0 sc1 loads) until every word is i
// An iteration = one hop of the slowest sender; timed per GPU with s_memrealtime over `iters` iterations.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 xgmi_reset.hip -o xgmi_reset
// Run:   ./xgmi_reset [gpus=8] [iters=200]
#include <string>

#include "common.h"

constexpr int MAX_GPUS = 8;

struct Peers {
  uint4 *recv[MAX_GPUS];  // recv[p] = GPU p's receive buffer: [2 sets][G slots][n16]
};

typedef unsigned v4u_t __attribute__((ext_vector_type(4)));

template <int RK>
__device__ __forceinline__ void reset16(uint4 *p) {
  v4u_t const w = {0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu};
  if constexpr (RK == 1) {
    asm volatile("global_store_dwordx4 %0, %1, off sc1\n s_nop 1" ::"v"(p), "v"(w) : "memory");
  } else if constexpr (RK == 2) {
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n s_nop 1" ::"v"(p), "v"(w) : "memory");
  } else if constexpr (RK == 3) {
    asm volatile("global_store_dwordx4 %0, %1, off\n s_nop 1" ::"v"(p), "v"(w) : "memory");
  }
}
__device__ __forceinline__ void push16_sys(uint4 *p, unsigned v) {
  v4u_t const w = {v, v, v, v};
  asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1\n s_nop 1" ::"v"(p), "v"(w) : "memory");
}
__device__ __forceinline__ v4u_t load16_sys(uint4 const *p) {
  v4u_t w;
  asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1\n s_waitcnt vmcnt(0)" : "=v"(w) : "v"(p) : "memory");
  return w;
}

template <int RK>
__global__ void xr_kernel(Peers peers, int me, int gpus, int n16, int iters, unsigned long long *t, unsigned *fail) {
  extern __shared__ char lds[];
  if (threadIdx.x == 0) lds[0] = 0;
  int const wg = blockIdx.x, nw = gridDim.x;
  int const lo = (int)((long long)n16 * wg / nw), hi = (int)((long long)n16 * (wg + 1) / nw);
  size_t const slot = (size_t)n16, set = (size_t)gpus * n16;
  unsigned long long const t0 = realtime();
  for (int it = 1; it <= iters; ++it) {
    if (__hip_atomic_load(fail, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != 0) break;
    size_t const s = (size_t)(it & 1), sn = (size_t)((it + 1) & 1);
    if constexpr (RK != 0) {  // 1. the next iteration's set of this GPU's buffer to 0xFF
      for (int i = lo + (int)threadIdx.x; i < hi; i += blockDim.x) {
        for (int r = 0; r < gpus; r++) reset16<RK>(peers.recv[me] + sn * set + r * slot + i);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    }
    for (int i = lo + (int)threadIdx.x; i < hi; i += blockDim.x) {  // 2. this GPU's message into every GPU's slot `me`
#pragma unroll
      for (int r = 0; r < MAX_GPUS; r++) {
        if (r < gpus) push16_sys(peers.recv[r] + s * set + (size_t)me * slot + i, (unsigned)it);
      }
    }
    for (int i = lo + (int)threadIdx.x; i < hi; i += blockDim.x) {  // 3. every GPU's message from this GPU's set s
      for (int r = 0; r < gpus; r++) {
        uint4 const *p = peers.recv[me] + s * set + r * slot + i;
        long long spins = 0;
        for (;;) {
          v4u_t const w = load16_sys(p);
          if (w.x == (unsigned)it && w.y == (unsigned)it && w.z == (unsigned)it && w.w == (unsigned)it) break;
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
  const size_t max_bytes = 128 << 10;
  Peers peers;
  std::vector<unsigned long long *> t(gpus);
  std::vector<unsigned *> fail(gpus);
  std::vector<hipStream_t> streams(gpus);
  void (*ks[4])(Peers, int, int, int, int, unsigned long long *, unsigned *) = {xr_kernel<0>, xr_kernel<1>, xr_kernel<2>,
                                                                                xr_kernel<3>};
  for (int g = 0; g < gpus; ++g) {
    CHECK(hipSetDevice(g));
    for (int p = 0; p < gpus; ++p) {
      if (p == g) continue;
      int can;
      CHECK(hipDeviceCanAccessPeer(&can, g, p));
      if (!can) {
        printf("GPU %d cannot access GPU %d\n", g, p);
        return 0;
      }
      CHECK(hipDeviceEnablePeerAccess(p, 0));
    }
    CHECK(hipMalloc(&peers.recv[g], 2 * gpus * max_bytes));
    CHECK(hipMalloc(&t[g], 16));
    CHECK(hipMalloc(&fail[g], 4));
    CHECK(hipStreamCreateWithFlags(&streams[g], hipStreamNonBlocking));
    for (int k = 0; k < 4; k++) set_max_lds((const void *)ks[k], ONE_PER_CU_LDS);
  }
  double const tpn = wall_ticks_per_ns(0);
  char const *rk_name[4] = {"none", "sc1 (st16_dev)", "sc0 sc1 (sys)", "plain"};
  for (size_t bytes : {(size_t)114688, (size_t)14336}) {
    for (int rk : {0, 1, 2, 3}) {
      for (int senders : {64, 128}) {
        int const n16 = (int)(bytes / 16);
        for (int g = 0; g < gpus; ++g) {
          CHECK(hipSetDevice(g));
          CHECK(hipMemset(peers.recv[g], 0, 2 * gpus * max_bytes));
          CHECK(hipMemset(t[g], 0, 16));
          CHECK(hipMemset(fail[g], 0, 4));
          CHECK(hipDeviceSynchronize());
        }
        for (int g = 0; g < gpus; ++g) {
          CHECK(hipSetDevice(g));
          hipLaunchKernelGGL(ks[rk], dim3(senders), dim3(256), ONE_PER_CU_LDS, streams[g], peers, g, gpus, n16, iters, t[g], fail[g]);
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
        double worst = 0;
        for (double v : per) worst = std::max(worst, v);
        printf("%d GPUs, %6zu B per GPU, reset %-15s, %3d CUs: one hop (incl. the reset) %.2f us (slowest GPU; median %.2f)%s\n",
               gpus, bytes, rk_name[rk], senders, worst / 1e3, median(per) / 1e3, fails ? " POLLS GAVE UP" : "");
      }
    }
  }
  return 0;
}
