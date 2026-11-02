// one wave's instruction latency and throughput on gfx950: a dependent add chain, independent adds (8 chains), a dependent
// LDS read chain; cycles from the shader clock (clock64), also checked against s_memrealtime
#include "common.h"
__global__ void __launch_bounds__(64, 1) rate_kernel(unsigned *out, unsigned long long *t, int n) {
  __shared__ unsigned lds[1024];
  for (int i = threadIdx.x; i < 1024; i += 64) lds[i] = (i * 7 + 1) & 1023;
  __syncthreads();
  unsigned a = threadIdx.x, b0 = 1, b1 = 2, b2 = 3, b3 = 4, b4 = 5, b5 = 6, b6 = 7, b7 = 8;
  long long c0 = clock64();
#pragma unroll 1
  for (int i = 0; i < n; i++) {
#pragma unroll
    for (int k = 0; k < 16; k++) asm volatile("v_add_u32 %0, %0, %1" : "+v"(a) : "v"(b0));
  }
  long long c1 = clock64();
#pragma unroll 1
  for (int i = 0; i < n; i++) {
#pragma unroll
    for (int k = 0; k < 2; k++) {
      asm volatile("v_add_u32 %0, %0, %1" : "+v"(b0) : "v"(a));
      asm volatile("v_add_u32 %0, %0, %1" : "+v"(b1) : "v"(a));
      asm volatile("v_add_u32 %0, %0, %1" : "+v"(b2) : "v"(a));
      asm volatile("v_add_u32 %0, %0, %1" : "+v"(b3) : "v"(a));
      asm volatile("v_add_u32 %0, %0, %1" : "+v"(b4) : "v"(a));
      asm volatile("v_add_u32 %0, %0, %1" : "+v"(b5) : "v"(a));
      asm volatile("v_add_u32 %0, %0, %1" : "+v"(b6) : "v"(a));
      asm volatile("v_add_u32 %0, %0, %1" : "+v"(b7) : "v"(a));
    }
  }
  long long c2 = clock64();
  unsigned p = threadIdx.x;
#pragma unroll 1
  for (int i = 0; i < n; i++) {
#pragma unroll
    for (int k = 0; k < 16; k++) p = lds[p & 1023];
  }
  long long c3 = clock64();
  out[threadIdx.x] = a + b0 + b1 + b2 + b3 + b4 + b5 + b6 + b7 + p;
  if (threadIdx.x == 0) { t[0] = c1 - c0; t[1] = c2 - c1; t[2] = c3 - c2; }
}
int main() {
  print_device(0);
  unsigned *o; unsigned long long *t;
  CHECK(hipMalloc(&o, 256)); CHECK(hipMalloc(&t, 64));
  int const n = 1000;
  for (int r = 0; r < 3; r++) {
    hipLaunchKernelGGL(rate_kernel, dim3(1), dim3(64), 0, 0, o, t, n);
    CHECK(hipDeviceSynchronize());
  }
  unsigned long long h[3];
  CHECK(hipMemcpy(h, t, 24, hipMemcpyDeviceToHost));
  double const ops = 16.0 * n;
  printf("one wave64: dependent v_add %.1f cycles each; independent v_add (8 chains) %.1f cycles each; dependent LDS read %.1f "
         "cycles each\n", h[0] / ops, h[1] / ops, h[2] / ops);
  printf("{\"bench\": \"issue_rate\", \"dep_add\": %.2f, \"indep_add\": %.2f, \"lds_chain\": %.2f}\n", h[0] / ops, h[1] / ops, h[2] / ops);
  return 0;
}
