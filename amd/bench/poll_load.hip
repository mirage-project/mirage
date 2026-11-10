// Microbenchmark: how much slower is a device-scope hand-off between two CUs while the other CUs stream reads, by how much
// they keep in flight? The question behind it (MI355X 10-16): weight preloads into LDS during a MoE layer's tail (a chain of
// cross-CU / cross-GPU hand-offs) lost four times; device-scope loads bypass the L2 (MALL / HBM), so they may queue behind
// the bulk reads.
//
// 256 workgroups (96 KB LDS: one per CU). Workgroups 0 and 1 (two XCDs when workgroups go round-robin over them; printed)
// ping-pong: thread 0 of 0 stores i to flag a (agent-scope atomic store), 1 polls a until i, stores i to flag b, 0 polls b
// until i; ITERS round trips. The other 254 workgroups (4 waves each) read a 1 GB buffer (16 B per lane per load, nt like
// the GEMM's weight loads) with D loads per wave issued together, then used, until 0 sets stop. Cases D = 0 (idle), 1, 2, 4,
// 8, 16 (1 .. 64 KB per CU in flight). Reported (median of 7 runs): the round trip (us), the readers' rate (TB/s).
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 poll_load.hip -o poll_load
// Run:   ./poll_load
#include "common.h"

typedef unsigned v4u __attribute__((ext_vector_type(4)));
constexpr int ITERS = 400;

struct Sh {
  unsigned a, b, stop, pad;
  unsigned long long t_pp, bytes, t_rd0, t_rd1;
  unsigned xcc0, xcc1;
};

template <int D>
__device__ void reader(Sh *s, unsigned const *big, size_t n16, unsigned *sink) {
  __amdgpu_buffer_rsrc_t const rs = __builtin_amdgcn_make_buffer_rsrc(const_cast<unsigned *>(big), 0, 0x7FFFFFFF, 0x00020000);
  int const wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
  size_t const rw = (size_t)(blockIdx.x - 2) * 4 + wave, nrw = 254 * 4;
  unsigned x = 0;
  unsigned long long done = 0;
  size_t step = 0;
  for (int it = 0;; it++) {
    if ((it & 7) == 0 && __hip_atomic_load(&s->stop, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
      break;
    }
    v4u f[D];
#pragma unroll
    for (int d = 0; d < D; d++) {
      size_t const piece = ((rw + nrw * (step + d)) * 64 + lane) % n16; // 1 KB pieces, waves interleaved
      f[d] = __builtin_bit_cast(v4u, __builtin_amdgcn_raw_buffer_load_b128(rs, (unsigned)((piece * 16) & 0x7FFFFFF0u), 0, 2));
    }
#pragma unroll
    for (int d = 0; d < D; d++) {
      x ^= f[d][0];
    }
    step += D;
    done += D;
  }
  if (lane == 0) {
    atomicAdd(&s->bytes, done * 1024ull);
  }
  if (x == 0x12345678u) {
    *sink = x;
  }
}

template <int D>
__global__ void __launch_bounds__(256, 1) kern(Sh *s, unsigned const *big, size_t n16, unsigned *sink) {
  extern __shared__ char lds[];
  if (threadIdx.x == 0) {
    lds[0] = 0;
  }
  if (blockIdx.x >= 2) {
    if (D > 0) {
      if (threadIdx.x == 0 && blockIdx.x == 2) {
        s->t_rd0 = wall_clock64();
      }
      reader<D == 0 ? 1 : D>(s, big, n16, sink);
    } else if (threadIdx.x == 0) {
      while (!__hip_atomic_load(&s->stop, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
        __builtin_amdgcn_s_sleep(8);
      }
    }
    return;
  }
  if (threadIdx.x != 0) {
    return;
  }
  unsigned xc;
  asm volatile("s_getreg_b32 %0, hwreg(HW_REG_XCC_ID)" : "=s"(xc));
  if (blockIdx.x == 0) {
    s->xcc0 = xc;
    // let the readers ramp up first (5 us)
    unsigned long long const t_wait = wall_clock64() + 500;
    while (wall_clock64() < t_wait) {
    }
    unsigned long long const t0 = wall_clock64();
    for (unsigned i = 1; i <= ITERS; i++) {
      __hip_atomic_store(&s->a, i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      while (__hip_atomic_load(&s->b, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != i) {
      }
    }
    s->t_pp = wall_clock64() - t0;
    s->t_rd1 = wall_clock64();
    __hip_atomic_store(&s->stop, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  } else {
    s->xcc1 = xc;
    for (unsigned i = 1; i <= ITERS; i++) {
      while (__hip_atomic_load(&s->a, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) != i) {
      }
      __hip_atomic_store(&s->b, i, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  CHECK(hipSetDevice(0));
  print_device(0);
  size_t const nbytes = (size_t)1 << 30, n16 = nbytes / 16;
  unsigned *big, *sink;
  Sh *s;
  CHECK(hipMalloc(&big, nbytes));
  CHECK(hipMemset(big, 1, nbytes));
  CHECK(hipMalloc(&sink, 64));
  CHECK(hipMalloc(&s, sizeof(Sh)));
  double const tpn = wall_ticks_per_ns(0);
  void (*ks[6])(Sh *, unsigned const *, size_t, unsigned *) = {kern<0>, kern<1>, kern<2>, kern<4>, kern<8>, kern<16>};
  int const ds[6] = {0, 1, 2, 4, 8, 16};
  for (auto k : ks) {
    set_max_lds((const void *)k, ONE_PER_CU_LDS);
  }
  for (int c = 0; c < 6; c++) {
    std::vector<double> rtt, bw;
    Sh h{};
    for (int rep = 0; rep < 7; rep++) {
      CHECK(hipMemset(s, 0, sizeof(Sh)));
      CHECK(hipDeviceSynchronize());
      hipLaunchKernelGGL(ks[c], dim3(256), dim3(256), ONE_PER_CU_LDS, 0, s, (unsigned const *)big, n16, sink);
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(&h, s, sizeof(Sh), hipMemcpyDeviceToHost));
      rtt.push_back(h.t_pp / tpn / 1e3 / ITERS);
      double const dt = (h.t_rd1 - h.t_rd0) / tpn; // ns (the readers run a little longer than this)
      bw.push_back(ds[c] ? h.bytes / dt / 1e3 : 0.0);
    }
    printf("readers D=%2d loads per wave (%2d KB per CU): device-scope round trip %.3f us (median of 7), readers %.2f TB/s; "
           "ping-pong XCDs %u / %u\n", ds[c], 4 * ds[c], median(rtt), median(bw), h.xcc0, h.xcc1);
  }
  return 0;
}
