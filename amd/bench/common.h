// Shared helpers for the AMD (gfx950) microbenchmarks: error checks, which die / CU a workgroup runs on, the clocks.
#pragma once
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                                                    \
  do {                                                                                              \
    hipError_t e_ = (x);                                                                            \
    if (e_ != hipSuccess) {                                                                         \
      fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_));             \
      exit(1);                                                                                      \
    }                                                                                               \
  } while (0)

// 96 KB of LDS per workgroup: at most one workgroup per CU, so a launch of (number of CUs) workgroups is all resident at once
constexpr int ONE_PER_CU_LDS = 96 * 1024;

// the XCD (die) this wave runs on
__device__ inline unsigned xcc_id() {
  unsigned v;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(v));
  return v;
}

// HW_ID: wave, SIMD, CU, shader array and shader engine ids of this wave (field layout: CDNA4 ISA guide)
__device__ inline unsigned hw_id() {
  unsigned v;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_HW_ID)" : "=s"(v));
  return v;
}

// s_memrealtime: the constant-rate clock (hipDeviceAttributeWallClockRate kHz); s_memtime: the shader clock (changes with
// the GPU's clock speed; finer)
__device__ inline unsigned long long realtime() { return wall_clock64(); }
__device__ inline unsigned long long shader_clock() { return clock64(); }

inline double wall_ticks_per_ns(int dev) {
  int khz;
  CHECK(hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, dev));
  return khz / 1e6;
}

inline void set_max_lds(const void *kernel, int bytes) {
  CHECK(hipFuncSetAttribute(kernel, hipFuncAttributeMaxDynamicSharedMemorySize, bytes));
}

inline double median(std::vector<double> v) {
  if (v.empty()) return 0;
  std::sort(v.begin(), v.end());
  return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}

inline void print_device(int dev) {
  hipDeviceProp_t p;
  CHECK(hipGetDeviceProperties(&p, dev));
  int ndev;
  CHECK(hipGetDeviceCount(&ndev));
  printf("device %d of %d: %s (%s), %d CUs, LDS per block %zu KB, L2 %d KB, clock %d MHz, wall clock %.0f MHz\n", dev, ndev,
         p.name, p.gcnArchName, p.multiProcessorCount, p.sharedMemPerBlock / 1024, p.l2CacheSize / 1024, p.clockRate / 1000,
         wall_ticks_per_ns(dev) * 1e3);
}
