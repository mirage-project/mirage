// lds_tr_check.hip -- what ds_read_b64_tr_b16 (gfx950) returns per lane.
// LDS holds a 64 x 64 matrix of u16, value = 64 r + c (row r, column c). One
// wave; per pattern, lane l passes a byte address A(l); the 4 u16 it gets back
// are printed as (row, column) pairs.
//   pattern 0: A(l) = 128 (l % 16) + 8 (l / 16)   (lane -> row l % 16, 4-column group l / 16)
//   pattern 1: A(l) = 128 l                       (lane -> row l, column 0)
//   pattern 2: A(l) = 8 l                         (row 0 .. 7: lane -> 4 consecutive columns)
// hipcc --offload-arch=gfx950 -O3 amd/bench/lds_tr_check.hip -o /tmp/lds_tr_check && /tmp/lds_tr_check
#include <hip/hip_runtime.h>
#include <cstdio>

typedef short v4s __attribute__((ext_vector_type(4)));

__global__ void probe(unsigned short *out, int pattern) {
  __shared__ unsigned short m[64 * 64];
  int const l = threadIdx.x;
  for (int i = l; i < 64 * 64; i += 64) {
    m[i] = (unsigned short)i;
  }
  __syncthreads();
  unsigned a = pattern == 0 ? 128u * (l % 16) + 8u * (l / 16) : pattern == 1 ? 128u * l : 8u * l;
  v4s v = __builtin_amdgcn_ds_read_tr16_b64_v4i16(
      (__attribute__((address_space(3))) v4s *)((__attribute__((address_space(3))) char *)m + a));
  for (int e = 0; e < 4; e++) {
    out[l * 4 + e] = (unsigned short)v[e];
  }
}

int main() {
  unsigned short *d, h[256];
  (void)hipMalloc(&d, 512);
  for (int p = 0; p < 3; p++) {
    hipLaunchKernelGGL(probe, dim3(1), dim3(64), 0, 0, d, p);
    (void)hipMemcpy(h, d, 512, hipMemcpyDeviceToHost);
    printf("pattern %d (lane: (row,col) x 4)\n", p);
    for (int l = 0; l < 64; l++) {
      printf(" %2d:", l);
      for (int e = 0; e < 4; e++) {
        printf(" (%d,%d)", h[l * 4 + e] / 64, h[l * 4 + e] % 64);
      }
      printf((l % 4 == 3) ? "\n" : " |");
    }
  }
  return 0;
}
