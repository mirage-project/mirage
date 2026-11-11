// Microbenchmark: can a CU see another CU's store through their shared XCD L2, and with which load bits? The question (10-27): W13 -> W2
// h_q in the moe_experts task goes through device-scope (sc1) stores and loads, which bypass the L2; with no W2 steals every reader of an
// expert's h_q runs on the producer's XCD, so plain stores + L2 loads could serve it - if a polling load does not keep returning a stale
// L1 (vL1D) copy.
// Case: 2 workgroups on the same XCD (picked by HW_REG_XCC_ID; grid 64, the first two arrivals on XCD 0 play, the others leave). The
// consumer first loads the line (so its L1 / L2 hold the old value), then the producer, after a delay, stores a new value with store
// bits SB; the consumer polls with load bits LB (s_sleep 1 between polls) until it sees the value or 200 us pass. Prints per (SB, LB)
// whether the value was seen and the median latency (consumer saw - producer stored, s_memrealtime) over 20 rounds.
// Bits (gfx950 buffer aux): 0 plain, 1 sc0, 16 sc1, 17 sc0 sc1; stores the same encoding.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 xcd_poll.hip -o xcd_poll
// Run:   ./xcd_poll
#include "common.h"

template <int SB, int LB>
__global__ void __launch_bounds__(64) poll_kernel(unsigned *buf, unsigned *slot, unsigned long long *t, unsigned round) {
  __shared__ int role;
  if (threadIdx.x == 0) {
    role = -1;
    if (xcc_id() == 0) {
      unsigned const r = atomicAdd(&slot[0], 1u);
      role = r < 2 ? (int)r : -1;
    }
  }
  __syncthreads();
  if (role < 0 || threadIdx.x != 0) return;
  __amdgpu_buffer_rsrc_t const rs = __builtin_amdgcn_make_buffer_rsrc(buf, 0, 4096, 0x00020000);
  unsigned const want = 1000u + round;
  if (role == 1) { // consumer: warm the line, signal, poll
    unsigned v = __builtin_amdgcn_raw_buffer_load_b32(rs, 0, 0, LB);
    __hip_atomic_store(&slot[1], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    unsigned long long const t0 = realtime();
    while (v != want && realtime() - t0 < 20000ull) { // (100 MHz: 200 us)
      __builtin_amdgcn_s_sleep(1);
      v = __builtin_amdgcn_raw_buffer_load_b32(rs, 0, 0, LB);
    }
    t[1] = v == want ? realtime() : 0ull;
  } else { // producer: wait for the consumer's warm-up, then store
    while (__hip_atomic_load(&slot[1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == 0u) __builtin_amdgcn_s_sleep(1);
    unsigned long long const s = realtime();
    while (realtime() - s < 200ull) __builtin_amdgcn_s_sleep(1); // (2 us)
    t[0] = realtime();
    __builtin_amdgcn_raw_buffer_store_b32(want, rs, 0, 0, SB);
    __builtin_amdgcn_s_waitcnt(0);
  }
}

template <int SB, int LB>
void run(unsigned *buf, unsigned *slot, unsigned long long *t, double tpn) {
  std::vector<double> lat;
  int seen = 0;
  for (unsigned r = 0; r < 20; r++) {
    CHECK(hipMemset(slot, 0, 64));
    CHECK(hipMemset(buf, 0, 4096));
    CHECK(hipMemset(t, 0, 16));
    CHECK(hipDeviceSynchronize());
    poll_kernel<SB, LB><<<64, 64>>>(buf, slot, t, r);
    CHECK(hipDeviceSynchronize());
    unsigned long long h[2];
    CHECK(hipMemcpy(h, t, 16, hipMemcpyDeviceToHost));
    if (h[1]) {
      seen++;
      lat.push_back((double)(h[1] - h[0]) / tpn / 1e3);
    }
  }
  printf("store bits %2d, load bits %2d: seen %2d / 20, latency median %s us\n", SB, LB, seen,
         lat.empty() ? "-" : std::to_string(median(lat)).substr(0, 6).c_str());
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  print_device(0);
  CHECK(hipSetDevice(0));
  unsigned *buf, *slot;
  unsigned long long *t;
  CHECK(hipMalloc(&buf, 4096));
  CHECK(hipMalloc(&slot, 64));
  CHECK(hipMalloc(&t, 16));
  double const tpn = wall_ticks_per_ns(0);
  run<0, 0>(buf, slot, t, tpn);
  run<0, 1>(buf, slot, t, tpn);
  run<0, 16>(buf, slot, t, tpn);
  run<1, 1>(buf, slot, t, tpn);
  run<16, 0>(buf, slot, t, tpn);
  run<16, 1>(buf, slot, t, tpn);
  run<16, 16>(buf, slot, t, tpn);
  return 0;
}
