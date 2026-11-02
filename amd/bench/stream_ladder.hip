// Microbenchmark: why do the megakernel's weight streams (gemm_tile's K loop, the KDA projection: ~4.6 TB/s; moe_experts:
// 4.2-4.6) run slower than a plain reader with the same loads in flight (poll_load.hip: 16 x 1 KB per wave, 7.3 TB/s)?
// One persistent kernel, 256 workgroups x 4 waves (96 KB LDS: one per CU); the GPU reads 96 MB in total (the KDA projection:
// 384 KB per CU), 1 GB read before each launch. Variants, each a step from the reader towards gemm_tile:
//   0 burst      16 loads per wave issued together, then all used (s_waitcnt 0); addresses interleaved over the waves
//   1 rolling    16 in flight, one used and one issued per step (vmcnt 15), interleaved addresses
//   2 chunks     rolling, each wave its own contiguous run (32 KB = one projection task's tile row), runs 1024 waves apart
//   3 gemm       2 + per step: the activation fragment from LDS (ds_read_b128, lgkmcnt 0) and one MFMA 16x16x32 bf16
//   4 tasks      3 cut into 3 tasks per CU (each a fresh 16-load ramp and a drain, a barrier between: as gemm_tile)
//   5 tasks+pf   4, but each task's drain steps issue the next task's first loads (the window carried over the boundary)
// Reported: us (events, median of 15) and TB/s for each.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 stream_ladder.hip -o stream_ladder
// Run:   ./stream_ladder
#include "common.h"

typedef float v4f __attribute__((ext_vector_type(4)));
typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
constexpr int STG = 16, STEPS = 96, TASKS = 3, TSTEPS = STEPS / TASKS; // per wave: 96 KB in 3 tasks of 32 KB

__device__ __forceinline__ v4f ld(__amdgpu_buffer_rsrc_t rs, unsigned off) {
  return __builtin_bit_cast(v4f, __builtin_amdgcn_raw_buffer_load_b128(rs, off, 0, 2));
}

template <int V>
__global__ void __launch_bounds__(256, 1) kern(float const *w, float *out) {
  extern __shared__ char sm[];
  int const wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
  int const gw = blockIdx.x * 4 + wave; // 0 .. 1023
  __amdgpu_buffer_rsrc_t const rs = __builtin_amdgcn_make_buffer_rsrc(const_cast<float *>(w), 0, 0x7FFFFFFF, 0x00020000);
  // the byte offset of this wave's step s (1 KB pieces, 16 B per lane)
  auto off = [&](int s) -> unsigned {
    if (V <= 1) return (unsigned)((((size_t)s * 1024 + gw) * 64 + lane) * 16);               // interleaved
    return (unsigned)((((size_t)gw * STEPS + s) * 64 + lane) * 16);                          // contiguous runs per wave
  };
  for (int i = threadIdx.x; i < 1024; i += 256) {
    reinterpret_cast<float *>(sm)[i] = 0.001f * i;
  }
  __syncthreads();
  v4f acc = {0.f, 0.f, 0.f, 0.f};
  if constexpr (V == 0) {
    for (int s0 = 0; s0 < STEPS; s0 += STG) {
      v4f f[STG];
#pragma unroll
      for (int i = 0; i < STG; i++) f[i] = ld(rs, off(s0 + i));
#pragma unroll
      for (int i = 0; i < STG; i++) acc += f[i];
    }
  } else {
    auto use = [&](v4f f, int s) {
      if constexpr (V >= 3) {
        v4f const b = *reinterpret_cast<v4f const *>(sm + ((s * 64 + lane) & 255) * 16);
        acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, f), __builtin_bit_cast(v8bf, b), acc, 0, 0, 0);
      } else {
        acc += f;
      }
    };
    if constexpr (V <= 3) {
      v4f fr[STG];
#pragma unroll
      for (int i = 0; i < STG; i++) fr[i] = ld(rs, off(i));
#pragma unroll 1
      for (int s0 = 0; s0 < STEPS - STG; s0 += STG) {
#pragma unroll
        for (int i = 0; i < STG; i++) {
          use(fr[i], s0 + i);
          fr[i] = ld(rs, off(s0 + i + STG));
          __builtin_amdgcn_sched_barrier(0);
        }
      }
#pragma unroll
      for (int i = 0; i < STG; i++) use(fr[i], STEPS - STG + i);
    } else {
      // 3 tasks of TSTEPS steps; V 5: the last STG steps of a task issue the next task's first loads
      v4f fr[STG];
#pragma unroll
      for (int i = 0; i < STG; i++) fr[i] = ld(rs, off(i));
#pragma unroll 1
      for (int t = 0; t < TASKS; t++) {
        int const b0 = t * TSTEPS;
#pragma unroll 1
        for (int s0 = 0; s0 < TSTEPS - STG; s0 += STG) {
#pragma unroll
          for (int i = 0; i < STG; i++) {
            use(fr[i], b0 + s0 + i);
            fr[i] = ld(rs, off(b0 + s0 + i + STG));
            __builtin_amdgcn_sched_barrier(0);
          }
        }
#pragma unroll
        for (int i = 0; i < STG; i++) {
          use(fr[i], b0 + TSTEPS - STG + i);
          if (V == 5 && t + 1 < TASKS) {
            fr[i] = ld(rs, off(b0 + TSTEPS + i));
            __builtin_amdgcn_sched_barrier(0);
          }
        }
        // the task's end: its output, a barrier (gemm_tile: the K parts' sums / the next task's LDS)
        if (lane < 16) out[((size_t)gw * TASKS + t) * 16 + lane] = acc[0];
        __syncthreads();
        if (V == 4 && t + 1 < TASKS) {
#pragma unroll
          for (int i = 0; i < STG; i++) fr[i] = ld(rs, off(b0 + TSTEPS + i));
        }
      }
    }
  }
  if (lane < 16 && acc[1] == 12345.f) out[lane] = acc[2];
}

__global__ void flush_k(const int *p, size_t n, int *sink) {
  int acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) acc += p[i];
  if (acc == 0x7FFFFFFF) *sink = acc;
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  CHECK(hipSetDevice(0));
  print_device(0);
  size_t const wbytes = (size_t)1024 * STEPS * 1024; // 96 MB
  float *w, *out;
  int *flush;
  size_t const flush_n = (size_t)256 << 20;
  CHECK(hipMalloc(&w, wbytes));
  CHECK(hipMemset(w, 0, wbytes));
  CHECK(hipMalloc(&out, (size_t)1024 * TASKS * 16 * 4));
  CHECK(hipMalloc(&flush, flush_n * 4));
  CHECK(hipMemset(flush, 1, flush_n * 4));
  void (*ks[6])(float const *, float *) = {kern<0>, kern<1>, kern<2>, kern<3>, kern<4>, kern<5>};
  char const *names[6] = {"0 burst", "1 rolling", "2 chunks", "3 gemm", "4 tasks", "5 tasks+pf"};
  for (auto k : ks) set_max_lds((const void *)k, ONE_PER_CU_LDS);
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  for (int v = 0; v < 6; v++) {
    std::vector<double> us;
    for (int rep = 0; rep < 16; rep++) {
      hipLaunchKernelGGL(flush_k, dim3(1024), dim3(256), 0, 0, flush, flush_n, flush + 1);
      CHECK(hipEventRecord(e0, 0));
      hipLaunchKernelGGL(ks[v], dim3(256), dim3(256), ONE_PER_CU_LDS, 0, (float const *)w, out);
      CHECK(hipEventRecord(e1, 0));
      CHECK(hipEventSynchronize(e1));
      float ms;
      CHECK(hipEventElapsedTime(&ms, e0, e1));
      if (rep) us.push_back(ms * 1e3);
    }
    CHECK(hipGetLastError());
    double const m = median(us);
    printf("%-11s 96 MB: %7.2f us, %5.2f TB/s (median of 15, 1 GB read before each)\n", names[v], m, wbytes / (m * 1e-6) / 1e12);
  }
  return 0;
}
