// Microbenchmark: does a weight stream read faster when an earlier kernel already pulled it into the Infinity Cache (MALL,
// 256 MB, memory side: a kernel launch does not empty it)? The question behind it: the KDA projection (Wproj 6400 x 7168
// bf16, 91.8 MB, grid [100, 7] = 700 tasks of 64 rows x 1024 columns) ends ~20 us after the go vs a ~15 us floor at the
// per-CU read rate, while most CUs idle in the last ~10 us of the MoE kernel before it.
//
// GEMM: the gemm_load.hip tile (shuffled weights, 8 tokens, MFMA 16x16x32 bf16), persistent: 256 workgroups (96 KB LDS: one
// per CU), workgroup w runs tasks w, w + 256, ... (as the megakernel's round-robin lists). Warm kernel: W workgroups load the
// weight fragments of the GEMM's tasks [0, n) (default cache policy; values xor-ed, never used), nothing else.
// Cases (each: 1 GB read first, median of 20):
//   cold                           flush -> GEMM
//   warm all 700 tasks (256 WGs)   flush -> warm -> GEMM
//   warm first 256 tasks           flush -> warm (256 / 64 WGs) -> GEMM          (each CU's first task)
//   warm first 512 tasks           flush -> warm (256 WGs) -> GEMM
//   warm all, then 64 MB other     flush -> warm -> read 64 MB elsewhere -> GEMM  (the L2s refilled: MALL hits only)
// GEMM hints: default and nt (STATIC_GEMM_SHUF_AUX 2). Reported: GEMM us (events), warm kernel us. Output checked equal to
// the cold run's.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 mall_warm.hip -o mall_warm
// Run:   ./mall_warm
#include <cstring>
#include <random>
#include <string>

#include "common.h"

typedef float v4f __attribute__((ext_vector_type(4)));
typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
constexpr int T = 8, STG = 8, N = 6400, K = 7168, KP = 1024, NX = N / 64, PARTS = K / KP, TASKS = NX * PARTS;
constexpr int S = KP / 32;

__device__ inline unsigned frag_off(int task, int wave, int lane, int s) {
  int const tx = task % NX, ty = task / NX, tile = tx * 4 + wave;
  return (unsigned)((((size_t)tile * (K / 32) + ty * (KP / 32) + s) * 64 + lane) * 16);
}

template <int AUX>
__global__ void __launch_bounds__(256, 1) gemm_k(const uint16_t *w, const uint16_t *act, float *out) {
  extern __shared__ char sm[];
  int const wave = threadIdx.x >> 6, lane = threadIdx.x & 63, tok = lane % 16, g = lane / 16;
  __amdgpu_buffer_rsrc_t const rs = __builtin_amdgcn_make_buffer_rsrc(const_cast<uint16_t *>(w), 0, 0x7FFFFFFF, 0x00020000);
  for (int task = blockIdx.x; task < TASKS; task += gridDim.x) {
    int const tx = task % NX, ty = task / NX, k0 = KP * ty;
    v4f frag[STG];
#pragma unroll
    for (int s = 0; s < STG; s++)
      frag[s] = __builtin_bit_cast(v4f, __builtin_amdgcn_raw_buffer_load_b128(rs, frag_off(task, wave, lane, s), 0, AUX));
    __syncthreads();
    for (int v = threadIdx.x; v < T * KP / 8; v += 256) {
      int const t = v / (KP / 8), c = (v % (KP / 8)) * 8;
      *reinterpret_cast<uint4 *>(sm + ((size_t)t * KP + c) * 2) = *reinterpret_cast<const uint4 *>(act + (size_t)t * K + k0 + c);
    }
    __syncthreads();
    v4f acc = {0.f, 0.f, 0.f, 0.f};
    auto step = [&](int s, v4f f) {
      v4f b = {0.f, 0.f, 0.f, 0.f};
      if (tok < T) b = *reinterpret_cast<const v4f *>(sm + ((size_t)tok * KP + 32 * s + 8 * g) * 2);
      acc = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, f), __builtin_bit_cast(v8bf, b), acc, 0, 0, 0);
    };
#pragma unroll 1
    for (int k = 0; k < S - STG; k += STG) {
#pragma unroll
      for (int s = 0; s < STG; s++) {
        step(k + s, frag[s]);
        frag[s] = __builtin_bit_cast(v4f, __builtin_amdgcn_raw_buffer_load_b128(rs, frag_off(task, wave, lane, k + s + STG), 0, AUX));
      }
    }
#pragma unroll
    for (int s = 0; s < STG; s++) step(S - STG + s, frag[s]);
    if (tok < T) {
      int const row = 64 * tx + 16 * wave + 4 * g;
      *reinterpret_cast<v4f *>(out + ((size_t)ty * T + tok) * N + row) = acc;
    }
  }
}

// loads tasks [0, n)'s fragments (16 per wave in flight), xor-ed into a value never used
__global__ void __launch_bounds__(256, 1) warm_k(const uint16_t *w, int n, unsigned *sink) {
  int const wave = threadIdx.x >> 6, lane = threadIdx.x & 63;
  __amdgpu_buffer_rsrc_t const rs = __builtin_amdgcn_make_buffer_rsrc(const_cast<uint16_t *>(w), 0, 0x7FFFFFFF, 0x00020000);
  unsigned x = 0;
  for (int task = blockIdx.x; task < n; task += gridDim.x) {
#pragma unroll 1
    for (int s0 = 0; s0 < S; s0 += 16) {
      v4f f[16];
#pragma unroll
      for (int s = 0; s < 16; s++)
        f[s] = __builtin_bit_cast(v4f, __builtin_amdgcn_raw_buffer_load_b128(rs, frag_off(task, wave, lane, s0 + s), 0, 0));
#pragma unroll
      for (int s = 0; s < 16; s++) x ^= __builtin_bit_cast(unsigned, f[s][0]);
    }
  }
  if (x == 0x12345678u) *sink = x;
}

__global__ void read_k(const int *p, size_t n, int *sink) {
  int acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) acc += p[i];
  if (acc == 0x7FFFFFFF) *sink = acc;
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  CHECK(hipSetDevice(0));
  print_device(0);
  std::mt19937 rng(3);
  std::uniform_int_distribution<int> mant(0, 127), sgn(0, 1), ex(118, 126);
  std::vector<uint16_t> wh((size_t)N * K), ah((size_t)T * K);
  for (auto &x : wh) x = (uint16_t)((sgn(rng) << 15) | (ex(rng) << 7) | mant(rng));
  for (auto &x : ah) x = (uint16_t)((sgn(rng) << 15) | (ex(rng) << 7) | mant(rng));
  uint16_t *dw, *da;
  float *dout;
  int *flush, *other;
  unsigned *sink;
  size_t const flush_n = (size_t)256 << 20, other_n = (size_t)16 << 20;  // 1 GB, 64 MB of ints
  CHECK(hipMalloc(&dw, wh.size() * 2));
  CHECK(hipMalloc(&da, ah.size() * 2));
  CHECK(hipMalloc(&dout, (size_t)PARTS * T * N * 4));
  CHECK(hipMalloc(&flush, flush_n * 4));
  CHECK(hipMalloc(&other, other_n * 4));
  CHECK(hipMalloc(&sink, 64));
  CHECK(hipMemset(flush, 1, flush_n * 4));
  CHECK(hipMemset(other, 1, other_n * 4));
  CHECK(hipMemcpy(dw, wh.data(), wh.size() * 2, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(da, ah.data(), ah.size() * 2, hipMemcpyHostToDevice));
  set_max_lds((const void *)gemm_k<0>, ONE_PER_CU_LDS);
  set_max_lds((const void *)gemm_k<2>, ONE_PER_CU_LDS);
  set_max_lds((const void *)warm_k, ONE_PER_CU_LDS);
  hipEvent_t e[4];
  for (auto &x : e) CHECK(hipEventCreate(&x));
  struct Case {
    const char *name;
    int warm_tasks, warm_wgs;
    bool other;
  };
  Case const cases[] = {{"cold", 0, 0, false},
                        {"warm all 700 tasks (256 WGs)", TASKS, 256, false},
                        {"warm first 256 tasks (256 WGs)", 256, 256, false},
                        {"warm first 256 tasks (64 WGs)", 256, 64, false},
                        {"warm first 512 tasks (256 WGs)", 512, 256, false},
                        {"warm all, then 64 MB other", TASKS, 256, true}};
  std::vector<float> ref((size_t)PARTS * T * N), o(ref.size());
  bool have_ref = false;
  std::string json;
  for (int aux : {2, 0}) {
    for (Case const &c : cases) {
      std::vector<double> gus, wus;
      for (int rep = 0; rep < 21; rep++) {
        hipLaunchKernelGGL(read_k, dim3(1024), dim3(256), 0, 0, flush, flush_n, flush + 1);
        CHECK(hipEventRecord(e[0], 0));
        if (c.warm_tasks) hipLaunchKernelGGL(warm_k, dim3(c.warm_wgs), dim3(256), ONE_PER_CU_LDS, 0, dw, c.warm_tasks, sink);
        CHECK(hipEventRecord(e[1], 0));
        if (c.other) hipLaunchKernelGGL(read_k, dim3(1024), dim3(256), 0, 0, other, other_n, other + 1);
        CHECK(hipEventRecord(e[2], 0));
        if (aux == 2) hipLaunchKernelGGL(gemm_k<2>, dim3(256), dim3(256), ONE_PER_CU_LDS, 0, dw, da, dout);
        else hipLaunchKernelGGL(gemm_k<0>, dim3(256), dim3(256), ONE_PER_CU_LDS, 0, dw, da, dout);
        CHECK(hipEventRecord(e[3], 0));
        CHECK(hipEventSynchronize(e[3]));
        float g, wm;
        CHECK(hipEventElapsedTime(&g, e[2], e[3]));
        CHECK(hipEventElapsedTime(&wm, e[0], e[1]));
        if (rep) {
          gus.push_back(g * 1e3);
          wus.push_back(wm * 1e3);
        }
      }
      CHECK(hipGetLastError());
      CHECK(hipMemcpy(o.data(), dout, o.size() * 4, hipMemcpyDeviceToHost));
      if (!have_ref) {
        ref = o;
        have_ref = true;
      }
      bool const same = memcmp(ref.data(), o.data(), o.size() * 4) == 0;
      double const g = median(gus), wm = c.warm_tasks ? median(wus) : 0;
      printf("GEMM %s: %-32s GEMM %6.2f us (%.2f TB/s), warm kernel %6.2f us; output same as the first run: %s\n",
             aux ? "nt     " : "default", c.name, g, (double)N * K * 2 / (g * 1e-6) / 1e12, wm, same ? "yes" : "NO");
      char row[200];
      snprintf(row, sizeof(row), "%s{\"aux\": %d, \"case\": \"%s\", \"gemm_us\": %.3f, \"warm_us\": %.3f, \"same\": %d}",
               json.empty() ? "" : ", ", aux, c.name, g, wm, same ? 1 : 0);
      json += row;
    }
  }
  printf("{\"bench\": \"mall_warm\", \"rows\": [%s]}\n", json.c_str());
  return 0;
}
