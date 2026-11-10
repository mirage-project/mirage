// sysload.hip (11-09): the time of one pass of hc_post_mix's step-2 loads at T 64 - WG workgroups (one per CU) x 256 threads, each
// thread 4 16-B loads (one per GPU slot, slots S apart) of an already written 2 MB buffer, then s_waitcnt - by the loads' cache
// bits (3 sc0 sc1 = the kernels' landing loads, 2 sc1, 0 none) and by who wrote the buffer last (local: a kernel on this GPU with
// sc0 sc1 stores; remote: a kernel on GPU 1 with sc0 sc1 stores into GPU 0's buffer). Every pass reads the same lines again
// (as a landing loop's rounds); pass 0 alone is the first touch. MI355X 11-09 glmprof23: in the kernel this step took ~14 us
// at T 64 for data that thread 0 alone saw ~1 us after it landed.
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 sysload.hip -o sysload       Run: ./sysload
#include "common.h"

typedef unsigned v4u_t __attribute__((ext_vector_type(4)));
constexpr int T = 64, H = 4096, SLOTS = 4, PASSES = 8;
constexpr size_t SLOT = (size_t)T * H * 2; // 512 KB

template <int SC>
__device__ __forceinline__ v4u_t ld16(void const *p) {
  v4u_t w;
  if constexpr (SC == 3) {
    asm volatile("global_load_dwordx4 %0, %1, off sc0 sc1" : "=v"(w) : "v"(p) : "memory");
  } else if constexpr (SC == 2) {
    asm volatile("global_load_dwordx4 %0, %1, off sc1" : "=v"(w) : "v"(p) : "memory");
  } else {
    asm volatile("global_load_dwordx4 %0, %1, off" : "=v"(w) : "v"(p) : "memory");
  }
  return w;
}

__global__ void fill(unsigned char *buf, unsigned v) { // sc0 sc1 stores, as the pushes
  size_t const n16 = SLOTS * SLOT / 16;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    v4u_t const w = {v, v, v, v};
    asm volatile("global_store_dwordx4 %0, %1, off sc0 sc1" ::"v"(buf + i * 16), "v"(w) : "memory");
  }
}

template <int SC>
__global__ void readk(unsigned char const *buf, unsigned long long *t, unsigned *sink, int wgs) {
  // workgroup w, thread j: vector v = w 256 + j of every slot (wgs x 256 vectors = the slot's T H / 8 at wgs 128)
  size_t const v = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned acc = 0;
  for (int p = 0; p < PASSES; p++) {
    __syncthreads();
    unsigned long long const t0 = wall_clock64();
    v4u_t w[SLOTS];
#pragma unroll
    for (int r = 0; r < SLOTS; r++) w[r] = ld16<SC>(buf + (size_t)r * SLOT + v * 16);
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
#pragma unroll
    for (int r = 0; r < SLOTS; r++) acc ^= w[r].x ^ w[r].w;
    __syncthreads();
    if (threadIdx.x == 0) atomicMax(&t[p], wall_clock64() - t0);
  }
  if (acc == 0x12345678u) atomicAdd(sink, 1u);
}

int main() {
  int ndev;
  CHECK(hipGetDeviceCount(&ndev));
  setvbuf(stdout, nullptr, _IOLBF, 0);
  print_device(0);
  unsigned char *buf;
  unsigned long long *t;
  unsigned *sink;
  CHECK(hipSetDevice(0));
  CHECK(hipMalloc(&buf, SLOTS * SLOT));
  CHECK(hipMalloc(&t, PASSES * 8));
  CHECK(hipMalloc(&sink, 4));
  bool const remote_ok = ndev > 1;
  if (remote_ok) {
    CHECK(hipSetDevice(1));
    CHECK(hipDeviceEnablePeerAccess(0, 0));
    CHECK(hipSetDevice(0));
  }
  double const tpn = wall_ticks_per_ns(0);
  int const wgs = (int)(SLOT / 16 / 256); // 128
  for (int writer = 0; writer < (remote_ok ? 2 : 1); writer++)
    for (int sc : {3, 2, 0})
      for (int rep = 0; rep < 2; rep++) {
        CHECK(hipSetDevice(writer));
        hipLaunchKernelGGL(fill, dim3(256), dim3(256), 0, 0, buf, 0x3f803f80u + rep);
        CHECK(hipDeviceSynchronize());
        CHECK(hipSetDevice(0));
        CHECK(hipMemset(t, 0, PASSES * 8));
        CHECK(hipDeviceSynchronize());
        if (sc == 3) hipLaunchKernelGGL(readk<3>, dim3(wgs), dim3(256), 0, 0, buf, t, sink, wgs);
        if (sc == 2) hipLaunchKernelGGL(readk<2>, dim3(wgs), dim3(256), 0, 0, buf, t, sink, wgs);
        if (sc == 0) hipLaunchKernelGGL(readk<0>, dim3(wgs), dim3(256), 0, 0, buf, t, sink, wgs);
        CHECK(hipDeviceSynchronize());
        unsigned long long h[PASSES];
        CHECK(hipMemcpy(h, t, sizeof(h), hipMemcpyDeviceToHost));
        printf("writer %s, loads %s, %d WGs x 256 threads x %d 16-B loads (%zu KB): pass us", writer ? "GPU 1 (remote)" : "GPU 0 (local) ",
               sc == 3 ? "sc0 sc1" : sc == 2 ? "sc1    " : "plain  ", wgs, SLOTS, SLOTS * SLOT / 1024);
        for (int p = 0; p < PASSES; p++) printf(" %.2f", h[p] / tpn / 1e3);
        printf("\n");
      }
  return 0;
}
