// Microbenchmark: inside ONE kernel, does a CU that reads its next GEMM tasks' weights while idle (into its XCD's L2) then
// stream them faster? The question (MI355X 10-16): in a MoE + KDA kernel ~200 CUs wait ~20 us during the MoE's tail
// (hops) with their first projection task's slice in LDS; their 2nd and 3rd tasks (128 KB each) stream after x lands
// (the projection ends ~18 us after x). l2_carry.hip showed the L2 does not carry data across kernels; this is within one.
// 256 workgroups x 4 waves (96 KB LDS: one per CU); workgroup w owns slice w of S KB (contiguous; wave v a quarter).
// Each wave streams its quarter as gemm_tile does (1 KB per wave per step, 16 steps in flight, nt loads, aux 2).
// Variants (1 GB read before each launch, so L2 and MALL start empty):
//   0 cold     the timed pass only
//   1 warm     a pre-read of the slice (plain loads, aux 0, results discarded), 10 us wait, then the timed pass
//   2 warm nt  the pre-read with nt loads (aux 2)
//   3 warm+bg  as 1, and after the pre-read the odd workgroups stream 128 KB of other data each (nt; 16 MB in all), as the
//              MoE tail's other reads would
// Reported: the timed pass per workgroup (first load issued -> all 4 waves' last load used), median / p90 / max over the 256
// workgroups (median of 9 launches), GB/s per CU at the median.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 l2_warm.hip -o l2_warm
// Run:   ./l2_warm
#include "common.h"

typedef float v4f __attribute__((ext_vector_type(4)));
constexpr int STG = 16;

__device__ __forceinline__ v4f ldb(__amdgpu_buffer_rsrc_t rs, unsigned off, int aux) {
  return aux ? __builtin_bit_cast(v4f, __builtin_amdgcn_raw_buffer_load_b128(rs, off, 0, 2))
             : __builtin_bit_cast(v4f, __builtin_amdgcn_raw_buffer_load_b128(rs, off, 0, 0));
}

// one wave's stream of `steps` 1 KB pieces from byte offset base (rolling window of STG loads)
__device__ __forceinline__ v4f stream(__amdgpu_buffer_rsrc_t rs, unsigned base, int steps, int aux) {
  int const lane = threadIdx.x & 63;
  v4f acc = {0.f, 0.f, 0.f, 0.f}, fr[STG];
#pragma unroll
  for (int i = 0; i < STG; i++) fr[i] = ldb(rs, base + (unsigned)(i * 1024 + lane * 16), aux);
#pragma unroll 1
  for (int s0 = 0; s0 < steps - STG; s0 += STG) {
#pragma unroll
    for (int i = 0; i < STG; i++) {
      acc += fr[i];
      fr[i] = ldb(rs, base + (unsigned)((s0 + i + STG) * 1024 + lane * 16), aux);
      __builtin_amdgcn_sched_barrier(0);
    }
  }
#pragma unroll
  for (int i = 0; i < STG; i++) acc += fr[i];
  return acc;
}

template <int V>
__global__ void __launch_bounds__(256, 1) kern(float const *w, float const *bg, int skb, unsigned long long *t, float *out) {
  extern __shared__ char sm[];
  if (threadIdx.x == 0) sm[0] = 0;
  int const wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
  __amdgpu_buffer_rsrc_t const rs = __builtin_amdgcn_make_buffer_rsrc(const_cast<float *>(w), 0, 0x7FFFFFFF, 0x00020000);
  int const steps = skb / 4; // 1 KB pieces per wave
  unsigned const base = (unsigned)(((size_t)blockIdx.x * skb + (size_t)wave * (skb / 4)) * 1024);
  v4f acc = {0.f, 0.f, 0.f, 0.f};
  if constexpr (V >= 1) {
    acc += stream(rs, base, steps, V == 2 ? 2 : 0);
    if constexpr (V == 3) {
      if (blockIdx.x & 1) {
        __amdgpu_buffer_rsrc_t const rb = __builtin_amdgcn_make_buffer_rsrc(const_cast<float *>(bg), 0, 0x7FFFFFFF, 0x00020000);
        acc += stream(rb, (unsigned)(((size_t)blockIdx.x * 128 + wave * 32) * 1024), 32, 2); // (steps: a multiple of STG)
      }
    }
    unsigned long long const tw = realtime() + 1000; // 10 us
    while (realtime() < tw) {
    }
    __syncthreads();
  }
  __shared__ unsigned long long t0s;
  if (threadIdx.x == 0) t0s = realtime();
  __syncthreads();
  acc += stream(rs, base, steps, 2);
  asm volatile("" ::"v"(acc));
  __syncthreads();
  if (threadIdx.x == 0) t[blockIdx.x] = realtime() - t0s;
  if (lane == 0 && acc[0] == 1234.5f) out[blockIdx.x] = acc[1];
}

__global__ void flush_k(const int *p, size_t n, int *sink) {
  int acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) acc += p[i];
  if (acc == 0x7FFFFFFF) *sink = acc;
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  CHECK(hipSetDevice(0));
  print_device(0);
  double const tpn = wall_ticks_per_ns(0);
  size_t const wbytes = (size_t)256 * 512 * 1024, bgbytes = (size_t)256 * 128 * 1024;
  float *w, *bg, *out;
  unsigned long long *t;
  int *flush;
  size_t const flush_n = (size_t)256 << 20;
  CHECK(hipMalloc(&w, wbytes));
  CHECK(hipMemset(w, 0, wbytes));
  CHECK(hipMalloc(&bg, bgbytes));
  CHECK(hipMemset(bg, 0, bgbytes));
  CHECK(hipMalloc(&out, 256 * 4));
  CHECK(hipMalloc(&t, 256 * 8));
  CHECK(hipMalloc(&flush, flush_n * 4));
  CHECK(hipMemset(flush, 1, flush_n * 4));
  void (*ks[4])(float const *, float const *, int, unsigned long long *, float *) = {kern<0>, kern<1>, kern<2>, kern<3>};
  char const *nm[4] = {"cold", "warm (plain pre-read)", "warm (nt pre-read)", "warm + bg stream"};
  for (auto k : ks) set_max_lds((const void *)k, ONE_PER_CU_LDS);
  for (int skb : {64, 128, 256, 384}) {
    for (int v = 0; v < 4; v++) {
      std::vector<double> md, p9, mx;
      for (int rep = 0; rep < 10; rep++) {
        hipLaunchKernelGGL(flush_k, dim3(1024), dim3(256), 0, 0, flush, flush_n, flush + 1);
        hipLaunchKernelGGL(ks[v], dim3(256), dim3(256), ONE_PER_CU_LDS, 0, (float const *)w, (float const *)bg, skb, t, out);
        CHECK(hipDeviceSynchronize());
        std::vector<unsigned long long> h(256);
        CHECK(hipMemcpy(h.data(), t, 256 * 8, hipMemcpyDeviceToHost));
        std::vector<double> us;
        for (auto x : h) us.push_back(x / tpn / 1e3);
        std::sort(us.begin(), us.end());
        if (rep) {
          md.push_back(us[128]);
          p9.push_back(us[230]);
          mx.push_back(us[255]);
        }
      }
      double const m = median(md);
      printf("%3d KB per CU (%5.1f MB per XCD), %-22s timed pass median %6.2f p90 %6.2f max %6.2f us; %5.1f GB/s per CU\n",
             skb, skb * 32 / 1024.0, nm[v], m, median(p9), median(mx), skb * 1024 / (m * 1e3));
    }
  }
  return 0;
}
