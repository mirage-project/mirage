// Microbenchmark: what does a megakernel task pay the first time its code runs on a CU in a launch (instruction fetch)?
// A task's code runs once or a few times per CU per launch (the latency chains: conv, recurrence, o_proj, attn_residual,
// sum_gpus, latent_up, moe_agg1): if each launch starts with a cold instruction cache, every chain step pays its fetch.
// One wave per workgroup, one workgroup per CU (96 KB LDS). The block: N KB of straight-line code (independent
// v_fma_f32 on 8 registers, 8 B each), run twice; s_memtime (shader clocks) around each pass. Cases: N = 4, 16, 32 KB;
// 1 workgroup or 256 at once; then a second launch of the same kernel (cold again after a launch?). Reported: pass 1 / pass
// 2 shader clocks (median over the workgroups), the difference in ns at the measured clock.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 icache.hip -o icache
// Run:   ./icache
#include "common.h"

#define FMA8 \
  asm volatile("v_fma_f32 %0, %0, %8, %8\n v_fma_f32 %1, %1, %8, %8\n v_fma_f32 %2, %2, %8, %8\n v_fma_f32 %3, %3, %8, %8\n" \
               " v_fma_f32 %4, %4, %8, %8\n v_fma_f32 %5, %5, %8, %8\n v_fma_f32 %6, %6, %8, %8\n v_fma_f32 %7, %7, %8, %8"  \
               : "+v"(a0), "+v"(a1), "+v"(a2), "+v"(a3), "+v"(a4), "+v"(a5), "+v"(a6), "+v"(a7) : "v"(b));

template <int KB>
__device__ __forceinline__ void block(float &a0, float &a1, float &a2, float &a3, float &a4, float &a5, float &a6, float &a7,
                                      float b) {
#pragma unroll
  for (int i = 0; i < KB * 1024 / 64; i++) { // 8 instructions x 8 B = 64 B per FMA8
    FMA8
  }
}

template <int KB>
__global__ void __launch_bounds__(64, 1) kern(unsigned long long *t, float *sink) {
  extern __shared__ char lds[];
  if (threadIdx.x == 0) lds[0] = 0;
  float a0 = threadIdx.x, a1 = 1, a2 = 2, a3 = 3, a4 = 4, a5 = 5, a6 = 6, a7 = 7, b = 1.0001f;
  unsigned long long c0 = clock64();
  block<KB>(a0, a1, a2, a3, a4, a5, a6, a7, b);
  unsigned long long c1 = clock64();
  block<KB>(a0, a1, a2, a3, a4, a5, a6, a7, b);
  unsigned long long c2 = clock64();
  if (threadIdx.x == 0) {
    t[blockIdx.x * 2] = c1 - c0;
    t[blockIdx.x * 2 + 1] = c2 - c1;
  }
  if (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 == 1.2345f) *sink = a0;
}

template <int KB>
void run(unsigned long long *dt, float *sink, int wgs, int launches) {
  void (*k)(unsigned long long *, float *) = kern<KB>;
  set_max_lds((const void *)k, ONE_PER_CU_LDS);
  for (int l = 0; l < launches; l++) {
    hipLaunchKernelGGL(k, dim3(wgs), dim3(64), ONE_PER_CU_LDS, 0, dt, sink);
    CHECK(hipDeviceSynchronize());
    std::vector<unsigned long long> h(wgs * 2);
    CHECK(hipMemcpy(h.data(), dt, wgs * 16, hipMemcpyDeviceToHost));
    std::vector<double> p1, p2;
    for (int w = 0; w < wgs; w++) {
      p1.push_back((double)h[w * 2]);
      p2.push_back((double)h[w * 2 + 1]);
    }
    double const m1 = median(p1), m2 = median(p2);
    printf("%2d KB code, %3d workgroups, launch %d: pass 1 %8.0f clocks, pass 2 %8.0f, +%.0f clocks (%.2f us at 2.4 GHz)%s\n", KB,
           wgs, l, m1, m2, m1 - m2, (m1 - m2) / 2400.0, l == 0 ? " (first launch of this kernel)" : "");
  }
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  CHECK(hipSetDevice(0));
  print_device(0);
  unsigned long long *dt;
  float *sink;
  CHECK(hipMalloc(&dt, 256 * 16));
  CHECK(hipMalloc(&sink, 64));
  for (int wgs : {1, 256}) {
    run<4>(dt, sink, wgs, 3);
    run<16>(dt, sink, wgs, 3);
    run<32>(dt, sink, wgs, 3);
  }
  return 0;
}
