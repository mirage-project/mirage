// Microbenchmark: can a second kernel start on CUs the first one frees, before the first one ends (an AMD stand-in for
// CUDA's programmatic dependent launch: the MoE kernel's front starting on the attention kernel's idle CUs)?
// Kernel A: 256 workgroups (one per CU: 96 KB LDS each); workgroup w spins until A_T0 + (w % 4 == 0 ? 40 : 10) us
// (a quarter of the CUs busy long, the rest freed early). Kernel B, launched on another stream right after A: NB
// workgroups (96 KB LDS); each, on entry, reads A's resident count (A's workgroups add 1 on entry): if A is not all
// resident (< 256), it exits at once (it may hold a CU A still needs); else it claims a worker slot (atomic, < 256) and
// records its start time, or exits when the slots are taken. Reported: B's slot claims over time vs A's end, so whether
// B's workers start while A still runs (on the CUs A freed). Plain streams, then the same captured in a HIP graph.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 overlap.hip -o overlap
// Run:   ./overlap
#include <string>

#include "common.h"

struct Sh {
  unsigned a_resident, b_slots, b_exits;
  unsigned long long t0;          // A's first workgroup's entry (set once)
  unsigned long long a_end[256];  // A workgroup end times
  unsigned long long b_start[256]; // B worker slot start times
};

__global__ void kern_a(Sh *s) {
  extern __shared__ char lds[];
  if (threadIdx.x == 0) {
    lds[0] = 0;
    unsigned long long const now = realtime();
    atomicCAS(&s->t0, 0ull, now);
    atomicAdd(&s->a_resident, 1u);
  }
  __syncthreads();
  int const w = blockIdx.x;
  unsigned long long const t0 = __hip_atomic_load(&s->t0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  unsigned long long const until = t0 + (unsigned long long)((w % 4 == 0 ? 40 : 10) * 100); // 100 MHz wall clock
  while (realtime() < until) __builtin_amdgcn_s_sleep(8);
  if (threadIdx.x == 0) s->a_end[w] = realtime();
}

__global__ void kern_b(Sh *s) {
  extern __shared__ char lds[];
  __shared__ int slot;
  if (threadIdx.x == 0) {
    lds[0] = 0;
    slot = -1;
    if (__hip_atomic_load(&s->a_resident, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) >= 256u) {
      unsigned const k = atomicAdd(&s->b_slots, 1u);
      if (k < 256u) {
        slot = (int)k;
        s->b_start[k] = realtime();
      }
    }
    if (slot < 0) atomicAdd(&s->b_exits, 1u);
  }
  __syncthreads();
  if (slot < 0) return;
  // a worker: a little work so it holds its CU for 5 us
  unsigned long long const until = realtime() + 500;
  while (realtime() < until) __builtin_amdgcn_s_sleep(8);
}

// B2: 256 workgroups, no claim protocol (its stream waited until A's workgroups were all resident: hipStreamWaitValue32)
__global__ void kern_b2(Sh *s) {
  extern __shared__ char lds[];
  if (threadIdx.x == 0) {
    lds[0] = 0;
    unsigned const k = atomicAdd(&s->b_slots, 1u);
    if (k < 256u) s->b_start[k] = realtime();
  }
  __syncthreads();
  unsigned long long const until = realtime() + 500;
  while (realtime() < until) __builtin_amdgcn_s_sleep(8);
}

static void report(Sh const &h, double tpn, const char *tag) {
  unsigned long long a_last = 0;
  for (int w = 0; w < 256; w++) a_last = std::max(a_last, h.a_end[w]);
  int before = 0, nb = std::min(h.b_slots, 256u);
  std::vector<double> st;
  for (int k = 0; k < nb; k++) {
    double const us = ((double)h.b_start[k] - (double)h.t0) / tpn / 1e3;
    st.push_back(us);
    before += h.b_start[k] < a_last;
  }
  std::sort(st.begin(), st.end());
  printf("%s: A's last workgroup ends %.1f us after A's first entry; B workers %d (exits %u), started before A's end: %d; "
         "B start us (first / 64th / 128th / last): %.1f / %.1f / %.1f / %.1f\n",
         tag, ((double)a_last - (double)h.t0) / tpn / 1e3, nb, h.b_exits, before, st.empty() ? -1 : st[0],
         st.size() > 63 ? st[63] : -1, st.size() > 127 ? st[127] : -1, st.empty() ? -1 : st.back());
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  print_device(0);
  CHECK(hipSetDevice(0));
  set_max_lds((const void *)kern_a, ONE_PER_CU_LDS);
  set_max_lds((const void *)kern_b, ONE_PER_CU_LDS);
  Sh *s;
  CHECK(hipMalloc(&s, sizeof(Sh)));
  hipStream_t sa, sb;
  CHECK(hipStreamCreateWithFlags(&sa, hipStreamNonBlocking));
  CHECK(hipStreamCreateWithFlags(&sb, hipStreamNonBlocking));
  double const tpn = wall_ticks_per_ns(0);
  Sh h;
  for (int nb : {512, 1024, 2048}) {
    for (int rep = 0; rep < 3; rep++) {
      CHECK(hipMemset(s, 0, sizeof(Sh)));
      CHECK(hipDeviceSynchronize());
      hipLaunchKernelGGL(kern_a, dim3(256), dim3(256), ONE_PER_CU_LDS, sa, s);
      hipLaunchKernelGGL(kern_b, dim3(nb), dim3(256), ONE_PER_CU_LDS, sb, s);
      CHECK(hipDeviceSynchronize());
      CHECK(hipMemcpy(&h, s, sizeof(Sh), hipMemcpyDeviceToHost));
      std::string tag = "streams, B " + std::to_string(nb) + " workgroups, run " + std::to_string(rep);
      report(h, tpn, tag.c_str());
    }
  }
  // B2 after a stream wait on A's resident count (no exits)
  for (int rep = 0; rep < 3; rep++) {
    CHECK(hipMemset(s, 0, sizeof(Sh)));
    CHECK(hipDeviceSynchronize());
    hipLaunchKernelGGL(kern_a, dim3(256), dim3(256), ONE_PER_CU_LDS, sa, s);
    CHECK(hipStreamWaitValue32(sb, &s->a_resident, 256u, hipStreamWaitValueGte, 0xFFFFFFFFu));
    hipLaunchKernelGGL(kern_b2, dim3(256), dim3(256), ONE_PER_CU_LDS, sb, s);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(&h, s, sizeof(Sh), hipMemcpyDeviceToHost));
    std::string tag = "streams + wait value, B2 256 workgroups, run " + std::to_string(rep);
    report(h, tpn, tag.c_str());
  }
  {
    hipGraph_t graph;
    hipGraphExec_t exec;
    hipEvent_t fork, join;
    CHECK(hipEventCreateWithFlags(&fork, hipEventDisableTiming));
    CHECK(hipEventCreateWithFlags(&join, hipEventDisableTiming));
    CHECK(hipStreamBeginCapture(sa, hipStreamCaptureModeGlobal));
    CHECK(hipEventRecord(fork, sa));
    CHECK(hipStreamWaitEvent(sb, fork, 0));
    hipLaunchKernelGGL(kern_a, dim3(256), dim3(256), ONE_PER_CU_LDS, sa, s);
    hipError_t const we = hipStreamWaitValue32(sb, &s->a_resident, 256u, hipStreamWaitValueGte, 0xFFFFFFFFu);
    printf("hipStreamWaitValue32 under capture: %s\n", hipGetErrorString(we));
    hipLaunchKernelGGL(kern_b2, dim3(256), dim3(256), ONE_PER_CU_LDS, sb, s);
    CHECK(hipEventRecord(join, sb));
    CHECK(hipStreamWaitEvent(sa, join, 0));
    hipError_t const ce = hipStreamEndCapture(sa, &graph);
    printf("end capture: %s\n", hipGetErrorString(ce));
    if (we == hipSuccess && ce == hipSuccess) {
      CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
      for (int rep = 0; rep < 3; rep++) {
        CHECK(hipMemset(s, 0, sizeof(Sh)));
        CHECK(hipDeviceSynchronize());
        CHECK(hipGraphLaunch(exec, sa));
        CHECK(hipStreamSynchronize(sa));
        CHECK(hipMemcpy(&h, s, sizeof(Sh), hipMemcpyDeviceToHost));
        std::string tag = "graph + wait value, B2 256 workgroups, run " + std::to_string(rep);
        report(h, tpn, tag.c_str());
      }
    }
    (void)hipGetLastError();
  }
  // the same in a HIP graph: fork / join with events
  for (int nb : {1024}) {
    hipGraph_t graph;
    hipGraphExec_t exec;
    hipEvent_t fork, join;
    CHECK(hipEventCreateWithFlags(&fork, hipEventDisableTiming));
    CHECK(hipEventCreateWithFlags(&join, hipEventDisableTiming));
    CHECK(hipStreamBeginCapture(sa, hipStreamCaptureModeGlobal));
    CHECK(hipEventRecord(fork, sa));
    CHECK(hipStreamWaitEvent(sb, fork, 0));
    hipLaunchKernelGGL(kern_a, dim3(256), dim3(256), ONE_PER_CU_LDS, sa, s);
    hipLaunchKernelGGL(kern_b, dim3(nb), dim3(256), ONE_PER_CU_LDS, sb, s);
    CHECK(hipEventRecord(join, sb));
    CHECK(hipStreamWaitEvent(sa, join, 0));
    CHECK(hipStreamEndCapture(sa, &graph));
    CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (int rep = 0; rep < 3; rep++) {
      CHECK(hipMemset(s, 0, sizeof(Sh)));
      CHECK(hipDeviceSynchronize());
      CHECK(hipGraphLaunch(exec, sa));
      CHECK(hipStreamSynchronize(sa));
      CHECK(hipMemcpy(&h, s, sizeof(Sh), hipMemcpyDeviceToHost));
      std::string tag = "graph, B " + std::to_string(nb) + " workgroups, run " + std::to_string(rep);
      report(h, tpn, tag.c_str());
    }
  }
  return 0;
}
