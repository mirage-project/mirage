// Microbenchmark: one CU's read rate by the loads' scope bits, and by LDS DMA (global_load_lds), on MI355X. Question (10-24, dec261):
// latent_up's first poll pass over R (64 KB per CU of system-scope 16-B loads, sc0 sc1) returned ~5.5 us after its weights had
// landed (~12 GB/s per CU), while plain loads reach ~25-36 GB/s per CU (read_bw.hip). Which way reads data another GPU wrote fastest?
//
// Each of C workgroups (one per CU, 256 threads) reads its own 512 KB chunk of a buffer once, D 16-B loads in flight per thread
// (issued together, then one wait), after 1 GB read elsewhere (the L2s and the Infinity Cache hold none of it). Kinds:
//   0 plain, 1 sc1 (device scope), 2 sc0 sc1 (system scope, the layer's polls), 3 nt,
//   4 LDS DMA plain, 5 LDS DMA sc0 sc1 (system), 6 LDS DMA sc1 (each wave 1 KB per instruction into its LDS area)
// Reported: GB/s per CU = bytes / (the workgroup's s_memrealtime span), median over the workgroups, median of 5 runs.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 read_sys.hip -o read_sys
// Run:   ./read_sys
#include "common.h"

typedef unsigned v4u_t __attribute__((ext_vector_type(4)));

template <int KIND>
__device__ __forceinline__ v4u_t ld(uint4 const *p) {
  v4u_t w;
  if constexpr (KIND == 0) {
    asm volatile("global_load_dwordx4 %0, %1, off" : "=v"(w) : "v"(p) : "memory");
  } else if constexpr (KIND == 1) {
    asm volatile("global_load_dwordx4 %0, %1, off sc1" : "=v"(w) : "v"(p) : "memory");
  } else if constexpr (KIND == 2) {
    asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1" : "=v"(w) : "v"(p) : "memory");
  } else {
    asm volatile("global_load_dwordx4 %0, %1, off nt" : "=v"(w) : "v"(p) : "memory");
  }
  return w;
}

template <int KIND, int D>
__global__ void __launch_bounds__(256, 1) rd_kernel(uint4 const *buf, size_t chunk16, unsigned long long *t, unsigned *sink) {
  extern __shared__ __attribute__((aligned(1024))) char lds[];
  int const w = blockIdx.x, tid = threadIdx.x, wave = tid >> 6, lane = tid & 63;
  uint4 const *p = buf + (size_t)w * chunk16;
  __syncthreads();
  unsigned long long const t0 = realtime();
  unsigned acc = 0;
  if constexpr (KIND < 4) {
    for (size_t i = tid; i < chunk16; i += (size_t)256 * D) {
      v4u_t v[D];
#pragma unroll
      for (int j = 0; j < D; j++) {
        size_t const k = i + (size_t)256 * j;
        v[j] = ld<KIND>(p + (k < chunk16 ? k : tid));
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
      for (int j = 0; j < D; j++) acc ^= v[j].x ^ v[j].w;
    }
  } else {
    constexpr int AUX = KIND == 4 ? 0 : (KIND == 5 ? 17 : 16);
    char *const area = lds + wave * (D * 1024);   // wave's D KB
    for (size_t i = tid; i < chunk16; i += (size_t)256 * D) {
#pragma unroll
      for (int j = 0; j < D; j++) {
        size_t const k = i + (size_t)256 * j;
        __builtin_amdgcn_global_load_lds((__attribute__((address_space(1))) void *)(const_cast<uint4 *>(p + (k < chunk16 ? k : tid))),
                                         (__attribute__((address_space(3))) void *)(area + j * 1024), 16, 0, AUX);
      }
      asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
      acc ^= reinterpret_cast<unsigned const *>(area)[lane];
    }
  }
  unsigned long long const t1 = realtime();
  if (tid == 0) t[w] = t1 - t0;
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  CHECK(hipSetDevice(0));
  print_device(0);
  size_t const CH = 512 << 10, BG = 1ull << 30;
  int const MAXC = 256;
  uint4 *buf, *bg;
  unsigned long long *t;
  unsigned *sink;
  CHECK(hipMalloc(&buf, CH * MAXC));
  CHECK(hipMalloc(&bg, BG));
  CHECK(hipMalloc(&t, MAXC * 8));
  CHECK(hipMalloc(&sink, 16));
  CHECK(hipMemset(buf, 1, CH * MAXC));
  CHECK(hipMemset(bg, 2, BG));
  double const tpn = wall_ticks_per_ns(0);
  char const *names[7] = {"plain", "sc1 (dev)", "sc0 sc1 (sys)", "nt", "LDS DMA plain", "LDS DMA sys", "LDS DMA sc1"};
  using K = void (*)(uint4 const *, size_t, unsigned long long *, unsigned *);
  K ks[7][3] = {{rd_kernel<0, 4>, rd_kernel<0, 8>, rd_kernel<0, 16>}, {rd_kernel<1, 4>, rd_kernel<1, 8>, rd_kernel<1, 16>},
                {rd_kernel<2, 4>, rd_kernel<2, 8>, rd_kernel<2, 16>}, {rd_kernel<3, 4>, rd_kernel<3, 8>, rd_kernel<3, 16>},
                {rd_kernel<4, 4>, rd_kernel<4, 8>, rd_kernel<4, 16>}, {rd_kernel<5, 4>, rd_kernel<5, 8>, rd_kernel<5, 16>},
                {rd_kernel<6, 4>, rd_kernel<6, 8>, rd_kernel<6, 16>}};
  int const DS[3] = {4, 8, 16};
  for (int k = 0; k < 7; k++)
    for (int d = 0; d < 3; d++) set_max_lds((const void *)ks[k][d], 96 * 1024);
  for (int cus : {16, 64, 256}) {
    for (int k = 0; k < 7; k++) {
      for (int d = 0; d < 3; d++) {
        std::vector<double> runs;
        for (int r = 0; r < 5; r++) {
          // 1 GB read elsewhere first: the L2s and the Infinity Cache hold none of buf
          hipLaunchKernelGGL((rd_kernel<0, 16>), dim3(256), dim3(256), 96 * 1024, 0, bg, BG / 16 / 256, t, sink);
          hipLaunchKernelGGL(ks[k][d], dim3(cus), dim3(256), 96 * 1024, 0, buf, CH / 16, t, sink);
          CHECK(hipDeviceSynchronize());
          std::vector<unsigned long long> h(cus);
          CHECK(hipMemcpy(h.data(), t, cus * 8, hipMemcpyDeviceToHost));
          std::vector<double> rate;
          for (int c = 0; c < cus; c++) rate.push_back((double)CH / (h[c] / tpn));   // bytes per ns = GB/s
          runs.push_back(median(rate));
        }
        printf("%3d CUs, %-14s, %2d x 16 B in flight per thread (%3d KB per CU): %6.1f GB/s per CU\n", cus, names[k], DS[d],
               DS[d] * 4, median(runs));
      }
    }
  }
  return 0;
}
