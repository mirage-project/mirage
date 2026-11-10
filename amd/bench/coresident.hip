// Microbenchmark: does a second kernel's workgroups, waiting on CUs the first kernel freed, slow the first kernel's
// remaining workgroups? (The KDA -> MoE overlap, amd/tests/test_overlap.py: KDA's tail ended ~5 us later with the MoE kernel
// resident, even with every MoE workgroup only polling one word.)
// Kernel A: 256 workgroups (one per CU: 96 KB LDS each); workgroups w % 4 != 0 exit at once, w % 4 == 0 run WORK: a
// dependent chain (compute: FMAs; memory: device-scope loads of a 4-KB line ring; mixed; L2 hits: plain loads of a 2-MB
// table, in the L2 after the first touches) and record its wall time and
// shader clocks. Kernel B (its stream waits until A's 256 are resident): 256 workgroups, each polls a flag A's last
// workgroup sets (s_sleep SLEEP between polls) or, with SLEEP < 0, only sleeps 60 us (no memory).
// Cases: A alone; A + B with SLEEP 1 / 8 / 64 / no-memory. Reported per case (median of 9 runs): A's busy workgroups'
// time (median, max) and shader MHz.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 coresident.hip -o coresident
// Run:   ./coresident
#include <string>

#include "common.h"

struct Sh {
  unsigned a_resident, a_done, flag, pad, half;
  unsigned long long t[64], clk[64]; // A's busy workgroups: wall ticks, shader clocks
};

template <int WORK>
__global__ void kern_a(Sh *s, unsigned const *ring, int iters, unsigned const *big) {
  extern __shared__ char lds[];
  if (threadIdx.x == 0) {
    lds[0] = 0;
    __hip_atomic_fetch_add(&s->a_resident, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  int const w = blockIdx.x;
  if (w % 4 != 0) {
    return;
  }
  __syncthreads();
  unsigned long long const t0 = realtime(), c0 = shader_clock();
  float a = threadIdx.x * 1e-3f, b = 1.0001f;
  unsigned idx = (threadIdx.x * 16 + w * 64) & 1023;
  for (int i = 0; i < iters; i++) {
    if (i == iters / 2 && w == 0 && threadIdx.x == 0) { // B is launched from here on (its stream waits for it)
      __hip_atomic_store(&s->half, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    if (WORK == 0 || WORK == 2) {
#pragma unroll
      for (int k = 0; k < 64; k++) {
        a = fmaf(a, b, 0.5f);
      }
    }
    if (WORK == 1 || WORK == 2) {
      idx = __hip_atomic_load(ring + idx, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    if (WORK == 3) { // plain loads of a 2-MB ring per workgroup group: L2 hits after the first round
      idx = big[(idx + (unsigned)w * 4096u) & (512u * 1024u - 1u)];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    s->t[w / 4] = realtime() - t0;
    s->clk[w / 4] = shader_clock() - c0;
    if (a == 12345.f) s->pad = idx;
    if (__hip_atomic_fetch_add(&s->a_done, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == 63) {
      __hip_atomic_store(&s->flag, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
  }
  if (idx == 0xFFFFFFFFu) s->pad = (unsigned)a;
}

__global__ void kern_b(Sh *s, int sleep, unsigned long long until) {
  extern __shared__ char lds[];
  if (threadIdx.x == 0) {
    lds[0] = 0;
    if (sleep >= 0) {
      while (__hip_atomic_load(&s->flag, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) == 0) {
        if (sleep >= 64) __builtin_amdgcn_s_sleep(64);
        else if (sleep >= 8) __builtin_amdgcn_s_sleep(8);
        else __builtin_amdgcn_s_sleep(1);
      }
    } else {
      unsigned long long const end = realtime() + until;
      while (realtime() < end) __builtin_amdgcn_s_sleep(64);
    }
  }
  __syncthreads();
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  print_device(0);
  CHECK(hipSetDevice(0));
  set_max_lds((const void *)kern_a<0>, ONE_PER_CU_LDS);
  set_max_lds((const void *)kern_a<1>, ONE_PER_CU_LDS);
  set_max_lds((const void *)kern_a<2>, ONE_PER_CU_LDS);
  set_max_lds((const void *)kern_a<3>, ONE_PER_CU_LDS);
  unsigned *big;
  CHECK(hipMalloc(&big, 2u << 20));
  {
    std::vector<unsigned> hb(512 * 1024);
    for (size_t i = 0; i < hb.size(); i++) hb[i] = (unsigned)((i * 2654435761u + 977u) & (512u * 1024u - 1u)) & ~15u;
    CHECK(hipMemcpy(big, hb.data(), 2u << 20, hipMemcpyHostToDevice));
  }
  set_max_lds((const void *)kern_b, ONE_PER_CU_LDS);
  Sh *s;
  unsigned *ring;
  CHECK(hipMalloc(&s, sizeof(Sh)));
  CHECK(hipMalloc(&ring, 1024 * 4));
  std::vector<unsigned> hr(1024);
  for (int i = 0; i < 1024; i++) hr[i] = (i * 16 + 16) & 1023 | (i & 15); // a ring over lines
  CHECK(hipMemcpy(ring, hr.data(), 4096, hipMemcpyHostToDevice));
  hipStream_t sa, sb;
  CHECK(hipStreamCreateWithFlags(&sa, hipStreamNonBlocking));
  CHECK(hipStreamCreateWithFlags(&sb, hipStreamNonBlocking));
  double const tpn = wall_ticks_per_ns(0);
  char const *wn[4] = {"compute", "memory", "mixed", "L2 hits"};
  int const iters[4] = {400, 600, 320, 300};
  for (int work = 0; work < 4; work++) {
    for (int bc : {-2, 1, 8, 64, -1}) { // -2: A alone; -1: B sleeps without memory
      std::vector<double> med, mx, mhz;
      for (int rep = 0; rep < 9; rep++) {
        CHECK(hipMemset(s, 0, sizeof(Sh)));
        CHECK(hipDeviceSynchronize());
        void (*ka)(Sh *, unsigned const *, int, unsigned const *) =
            work == 0 ? kern_a<0> : work == 1 ? kern_a<1> : work == 2 ? kern_a<2> : kern_a<3>;
        hipLaunchKernelGGL(ka, dim3(256), dim3(256), ONE_PER_CU_LDS, sa, s, (unsigned const *)ring, iters[work],
                           (unsigned const *)big);
        if (bc != -2) {
          CHECK(hipStreamWaitValue32(sb, &s->a_resident, 256u, hipStreamWaitValueGte, 0xFFFFFFFFu));
          CHECK(hipStreamWaitValue32(sb, &s->half, 1u, hipStreamWaitValueGte, 0xFFFFFFFFu));
          hipLaunchKernelGGL(kern_b, dim3(256), dim3(256), ONE_PER_CU_LDS, sb, s, bc, (unsigned long long)(60e3 * tpn));
        }
        CHECK(hipDeviceSynchronize());
        Sh h;
        CHECK(hipMemcpy(&h, s, sizeof(Sh), hipMemcpyDeviceToHost));
        std::vector<double> us, f;
        for (int k = 0; k < 64; k++) {
          us.push_back(h.t[k] / tpn / 1e3);
          f.push_back(h.clk[k] / (h.t[k] / tpn) * 1e3);
        }
        med.push_back(median(us));
        mx.push_back(*std::max_element(us.begin(), us.end()));
        mhz.push_back(median(f));
      }
      printf("%-7s A%s: busy WG time median %.2f us, max %.2f us, shader clock %.0f MHz (median of 9 runs)\n", wn[work],
             bc == -2 ? " alone            " : bc == -1 ? " + B (no-memory)  " : (std::string(" + B poll sleep ") + std::to_string(bc) + (bc < 10 ? " " : "")).c_str(),
             median(med), median(mx), median(mhz));
    }
  }
  return 0;
}
