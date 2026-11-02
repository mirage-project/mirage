// Microbenchmark: does data a kernel reads stay in the XCD's L2 for the next kernel, and how much faster does the next
// kernel read it? (For an attention kernel prefetching the next MoE kernel's front weights during its idle tail.)
// Kernel A (256 workgroups, one per CU) reads a slice of R KB per workgroup (workgroup w: slice w); kernel B (256
// workgroups) reads slice w again in workgroup w and times it (per workgroup: first load issued -> all landed, KB per us),
// and records its XCD; A records its XCD too, so the run says whether w lands on the same die in both. Cases: B after A
// (stream order), B after A + a 1 GB stream read by a third kernel (L2 / MALL evicted), B with no A (cold).
// Loads: 16 B per lane, 4 waves x 64 lanes, Q KB in flight per wave (plain or nt).
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 l2_carry.hip -o l2_carry
// Run:   ./l2_carry
#include <string>

#include "common.h"

typedef int v4i_t __attribute__((ext_vector_type(4)));

template <bool NT>
__global__ void read_kernel(v4i_t const *buf, int slice16, unsigned long long *t, unsigned *xcd, unsigned *sink) {
  int const w = blockIdx.x;
  v4i_t const *p = buf + (size_t)w * slice16;
  __shared__ unsigned long long t0s;
  __syncthreads();
  if (threadIdx.x == 0) t0s = realtime();
  __syncthreads();
  unsigned acc = 0;
  // 8 loads in flight per thread per round (16 B each): 32 KB per wave... per round 256 threads x 8 x 16 B = 32 KB
  for (int i = threadIdx.x; i < slice16; i += 256 * 8) {
    v4i_t v[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      int const k = i + j * 256;
      if (NT) {
        v[j] = k < slice16 ? __builtin_nontemporal_load(p + k) : v4i_t{0, 0, 0, 0};
      } else {
        v[j] = k < slice16 ? p[k] : v4i_t{0, 0, 0, 0};
      }
    }
#pragma unroll
    for (int j = 0; j < 8; j++) acc ^= (unsigned)v[j][j & 3];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    t[w] = realtime() - t0s;
    xcd[w] = xcc_id();
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

__global__ void flush_kernel(v4i_t const *big, size_t n16, unsigned *sink) {
  unsigned acc = 0;
  for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < n16; i += (size_t)gridDim.x * 256) {
    acc ^= (unsigned)__builtin_nontemporal_load(big + i)[0];
  }
  if (acc == 0x9e3779b9u) sink[0] = acc;
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  print_device(0);
  CHECK(hipSetDevice(0));
  int const WG = 256;
  size_t const BIG = 1ull << 30;
  v4i_t *buf, *big;
  unsigned long long *t;
  unsigned *xa, *xb, *sink;
  CHECK(hipMalloc(&buf, (size_t)WG * 512 * 1024));
  CHECK(hipMalloc(&big, BIG));
  CHECK(hipMemset(buf, 1, (size_t)WG * 512 * 1024));
  CHECK(hipMemset(big, 2, BIG));
  CHECK(hipMalloc(&t, WG * 8));
  CHECK(hipMalloc(&xa, WG * 4));
  CHECK(hipMalloc(&xb, WG * 4));
  CHECK(hipMalloc(&sink, 16));
  double const tpn = wall_ticks_per_ns(0);
  std::vector<unsigned long long> th(WG);
  std::vector<unsigned> ha(WG), hb(WG);
  for (int kb : {32, 64, 128, 256}) {
    int const slice16 = kb * 1024 / 16;
    for (int nt : {0, 1}) {
      for (int mode : {0, 1, 2}) { // 0: A then B; 1: A, flush, B; 2: flush, B (cold)
        double med = 0;
        std::vector<double> rates;
        int same = 0;
        for (int rep = 0; rep < 5; rep++) {
          flush_kernel<<<1024, 256>>>(big, BIG / 16, sink);
          if (mode != 2) {
            if (nt) read_kernel<true><<<WG, 256>>>(buf, slice16, t, xa, sink);
            else read_kernel<false><<<WG, 256>>>(buf, slice16, t, xa, sink);
          }
          if (mode == 1) flush_kernel<<<1024, 256>>>(big, BIG / 16, sink);
          if (nt) read_kernel<true><<<WG, 256>>>(buf, slice16, t, xb, sink);
          else read_kernel<false><<<WG, 256>>>(buf, slice16, t, xb, sink);
          CHECK(hipDeviceSynchronize());
          CHECK(hipMemcpy(th.data(), t, WG * 8, hipMemcpyDeviceToHost));
          CHECK(hipMemcpy(ha.data(), xa, WG * 4, hipMemcpyDeviceToHost));
          CHECK(hipMemcpy(hb.data(), xb, WG * 4, hipMemcpyDeviceToHost));
          std::vector<double> us;
          for (int w = 0; w < WG; w++) us.push_back(th[w] / tpn / 1e3);
          med = median(us);
          rates.push_back(kb / med);
          if (mode != 2) {
            same = 0;
            for (int w = 0; w < WG; w++) same += ha[w] == hb[w];
          }
        }
        printf("slice %3d KB per WG, %s loads, %-22s: B's read med %.2f us per WG (%.1f GB/s per CU, median of 5 runs)%s\n", kb,
               nt ? "nt   " : "plain", mode == 0 ? "after A (same slices)" : mode == 1 ? "after A + 1 GB flush" : "cold",
               med, median(rates), mode != 2 ? (std::string(", WG on the same XCD in A and B: ") + std::to_string(same) + "/256").c_str() : "");
      }
    }
  }
  return 0;
}
