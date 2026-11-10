// Microbenchmark: the time between two dependent kernels on MI355X (gfx950), plain stream launches and a HIP graph.
//
// HIP has no programmatic dependent launch (PDL): a kernel starts only after the previous one in the stream has finished.
// The gap between kernel i's last workgroup ending and kernel i+1's first workgroup starting is what a megakernel removes at
// each operator boundary. N small kernels run back to back; each stamps (s_memrealtime) its first start and its last end:
//   start[i] = min over workgroups of the first instruction's time (atomicMin)
//   end[i]   = max over workgroups of the last instruction's time (atomicMax)
//   gap[i]   = start[i + 1] - end[i]
// For grids of 1 workgroup and of one workgroup per CU (96 KB LDS each, the persistent kernel's shape), with a tiny body
// (spin `work_ns`), launched (a) one by one on a stream and (b) as one HIP graph replay. Medians are printed.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 launch_gap.hip -o launch_gap
// Run:   ./launch_gap [kernels=200] [work_ns=2000]       -> one line per case, then a JSON summary on the last line
#include <string>

#include "common.h"

__global__ void stamped_kernel(unsigned long long *starts, unsigned long long *ends, int i, unsigned long long work_ticks) {
  extern __shared__ char lds[];
  if (threadIdx.x != 0) return;
  unsigned long long t0 = realtime();
  lds[0] = 0;
  atomicMin(&starts[i], t0);
  while (realtime() - t0 < work_ticks) __builtin_amdgcn_s_sleep(1);
  atomicMax(&ends[i], realtime());
}

struct Case {
  const char *mode;
  int grid;
  double gap_ns, span_ns;
};

int main(int argc, char **argv) {
  int n = argc > 1 ? atoi(argv[1]) : 200;
  double work_ns = argc > 2 ? atof(argv[2]) : 2000;
  CHECK(hipSetDevice(0));
  print_device(0);
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  double tpn = wall_ticks_per_ns(0);
  unsigned long long work_ticks = (unsigned long long)(work_ns * tpn);
  set_max_lds((const void *)stamped_kernel, ONE_PER_CU_LDS);
  unsigned long long *starts, *ends;
  CHECK(hipMalloc(&starts, n * sizeof(unsigned long long)));
  CHECK(hipMalloc(&ends, n * sizeof(unsigned long long)));
  hipStream_t s;
  CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  std::vector<Case> cases;
  for (int grid : {1, prop.multiProcessorCount}) {
    int lds = grid == 1 ? 0 : ONE_PER_CU_LDS;
    auto launch_all = [&] {
      for (int i = 0; i < n; ++i)
        hipLaunchKernelGGL(stamped_kernel, dim3(grid), dim3(64), lds, s, starts, ends, i, work_ticks);
    };
    hipGraphExec_t exec = nullptr;
    for (const char *mode : {"stream", "graph"}) {
      bool graph = std::string(mode) == "graph";
      if (graph) {
        hipGraph_t g;
        CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
        launch_all();
        CHECK(hipStreamEndCapture(s, &g));
        CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
        CHECK(hipGraphDestroy(g));
      }
      std::vector<double> gaps, spans;
      for (int rep = 0; rep < 3; ++rep) {  // rep 0 warms up; reps 1-2 are kept
        CHECK(hipMemsetAsync(starts, 0xff, n * sizeof(unsigned long long), s));
        CHECK(hipMemsetAsync(ends, 0, n * sizeof(unsigned long long), s));
        if (graph) {
          CHECK(hipGraphLaunch(exec, s));
        } else {
          launch_all();
        }
        CHECK(hipStreamSynchronize(s));
        std::vector<unsigned long long> st(n), en(n);
        CHECK(hipMemcpy(st.data(), starts, n * sizeof(unsigned long long), hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(en.data(), ends, n * sizeof(unsigned long long), hipMemcpyDeviceToHost));
        if (rep == 0) continue;
        for (int i = 0; i + 1 < n; ++i) gaps.push_back(((double)st[i + 1] - (double)en[i]) / tpn);
        for (int i = 0; i < n; ++i) spans.push_back(((double)en[i] - (double)st[i]) / tpn);
      }
      cases.push_back({mode, grid, median(gaps), median(spans)});
      printf("%-6s grid %3d: gap between kernels %.0f ns (median of %zu), kernel span %.0f ns (body %.0f ns)\n", mode, grid,
             median(gaps), gaps.size(), median(spans), work_ns);
    }
    CHECK(hipGraphExecDestroy(exec));
  }
  printf("{\"bench\": \"launch_gap\", \"kernels\": %d, \"work_ns\": %.0f, \"cases\": [", n, work_ns);
  for (size_t i = 0; i < cases.size(); ++i)
    printf("%s{\"mode\": \"%s\", \"grid\": %d, \"gap_ns\": %.1f, \"span_ns\": %.1f}", i ? ", " : "", cases[i].mode, cases[i].grid,
           cases[i].gap_ns, cases[i].span_ns);
  printf("]}\n");
  return 0;
}
