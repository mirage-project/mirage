// Microbenchmark: the bf16 GEMM tile of tasks/gemm_tile.cuh (8 tokens, weight rows x K, MFMA 16x16x32) alone, to see how fast
// its weight stream is by weight layout and cache hint (MI355X 10-11: in the layer a 128-row task of the router read ~19 GB/s
// per CU).
//
// Weight layouts:
//   row-major  the model's [N][K]: a wave's load = 16 rows x 64 B (lane l: row l % 16, K 8 (l / 16) .. + 7 of the step), rows
//              K * 2 bytes apart (what gemm_tile.cuh reads today)
//   shuffled   [N / 16][K / 32][64 lanes][16 B]: the same fragments, a wave's load = 1 KB contiguous (as the MoE weights)
// Hints: default, nt. Task = 64 NT rows x KP columns (NT row tiles of 16 per wave, 4 waves), GEMM_STAGES K steps of 32 in flight
// per wave, activation slice in LDS, fp32 partial sums out per K part. Every variant's output is compared with the row-major
// default one (bit-identical: same fragments, same order) and a few values with the CPU.
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 gemm_load.hip -o gemm_load
// Run:   ./gemm_load   -> per shape, grid, layout, hint: us (median of 20, 1 GB read before each) and TB/s of weight bytes
#include <cmath>
#include <cstring>
#include <random>
#include <string>

#include "common.h"

typedef float v4f __attribute__((ext_vector_type(4)));
typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
constexpr int T = 8, STAGES = 8;

constexpr int gemm_stages(int S) {
  for (int d = STAGES < S ? STAGES : S; d > 1; d--)
    if (S % d == 0) return d;
  return 1;
}

template <int LAYOUT, int HINT, int NT, int KP>
__global__ void __launch_bounds__(256, 1)
    gemm_k(const uint16_t *w, const uint16_t *act, float *out, int N, int K) {
  extern __shared__ char sm[];
  constexpr int S = KP / 32, STG = gemm_stages(S), AUX = HINT ? 2 : 0;
  int const nx = N / (64 * NT), tx = blockIdx.x % nx, ty = blockIdx.x / nx;
  int const row0 = 64 * NT * tx, k0 = KP * ty;
  for (int v = threadIdx.x; v < T * KP / 8; v += 256) {
    int const t = v / (KP / 8), c = (v % (KP / 8)) * 8;
    *reinterpret_cast<uint4 *>(sm + ((size_t)t * KP + c) * 2) = *reinterpret_cast<const uint4 *>(act + (size_t)t * K + k0 + c);
  }
  __syncthreads();
  int const wave = threadIdx.x >> 6, lane = threadIdx.x & 63, tok = lane % 16, g = lane / 16;
  __amdgpu_buffer_rsrc_t const rs = __builtin_amdgcn_make_buffer_rsrc(const_cast<uint16_t *>(w), 0, 0x7FFFFFFF, 0x00020000);
  auto off = [&](int i, int s) -> unsigned {
    int const tile = row0 / 16 + NT * wave + i;
    if (LAYOUT == 0) return (unsigned)((((size_t)(16 * tile + lane % 16)) * K + k0 + 32 * s + 8 * g) * 2);
    return (unsigned)((((size_t)tile * (K / 32) + k0 / 32 + s) * 64 + lane) * 16);
  };
  auto wfrag = [&](int i, int s) {
    return __builtin_bit_cast(v4f, __builtin_amdgcn_raw_buffer_load_b128(rs, off(i, s), 0, AUX));
  };
  auto bfrag = [&](int s) {
    v4f b = {0.f, 0.f, 0.f, 0.f};
    if (tok < T) b = *reinterpret_cast<const v4f *>(sm + ((size_t)tok * KP + 32 * s + 8 * g) * 2);
    return b;
  };
  v4f acc[NT], frag[STG][NT];
  for (int i = 0; i < NT; i++) acc[i] = v4f{0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int s = 0; s < STG; s++)
#pragma unroll
    for (int i = 0; i < NT; i++) frag[s][i] = wfrag(i, s);
  auto step = [&](int s, v4f const *f) {
    v8bf const b = __builtin_bit_cast(v8bf, bfrag(s));
#pragma unroll
    for (int i = 0; i < NT; i++)
      acc[i] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, f[i]), b, acc[i], 0, 0, 0);
  };
#pragma unroll 1
  for (int k = 0; k < S - STG; k += STG) {
#pragma unroll
    for (int s = 0; s < STG; s++) {
      step(k + s, frag[s]);
#pragma unroll
      for (int i = 0; i < NT; i++) frag[s][i] = wfrag(i, k + s + STG);
    }
  }
#pragma unroll
  for (int s = 0; s < STG; s++) step(S - STG + s, frag[s]);
  if (tok < T)
    for (int i = 0; i < NT; i++) {
      int const row = row0 + 16 * (NT * wave + i) + 4 * g;
      *reinterpret_cast<v4f *>(out + ((size_t)ty * T + tok) * N + row) = acc[i];
    }
}

using Kern = void (*)(const uint16_t *, const uint16_t *, float *, int, int);

template <int NT, int KP>
void variants(Kern *k) {
  k[0] = gemm_k<0, 0, NT, KP>;
  k[1] = gemm_k<0, 1, NT, KP>;
  k[2] = gemm_k<1, 0, NT, KP>;
  k[3] = gemm_k<1, 1, NT, KP>;
}

// empties L2 and the Infinity Cache: every thread reads its ints of a 1 GB buffer
__global__ void flush_k(const int *p, size_t n, int *sink) {
  int acc = 0;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) acc += p[i];
  if (acc == 0x7FFFFFFF) *sink = acc;
}

static float bf2f(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}

struct Case {
  int N, K, NT, KP;
};

int main() {
  CHECK(hipSetDevice(0));
  print_device(0);
  std::mt19937 rng(3);
  std::uniform_int_distribution<int> mant(0, 127), sgn(0, 1), ex(118, 126);
  // the layer's GEMMs: router 896 x 7168, latent_down 3584 x 7168, shared gate_up 1536 x 7168, shared down 7168 x 768, latent_up
  // (per GPU) 896 x 3584; grids as (rows per task, K parts)
  std::vector<Case> cases = {{3584, 7168, 2, 896}, {3584, 7168, 1, 896},  {3584, 7168, 1, 1792}, {3584, 7168, 2, 512},
                             {896, 7168, 2, 512},  {896, 7168, 1, 512},   {896, 7168, 1, 256},   {1536, 7168, 2, 896},
                             {1536, 7168, 1, 896}, {7168, 768, 2, 768},   {7168, 768, 1, 768},   {896, 3584, 1, 512}};
  int const maxN = 7168, maxK = 7168;
  std::vector<uint16_t> wh((size_t)maxN * maxK), ws(wh.size()), ah((size_t)T * maxK);
  for (auto &x : wh) x = (uint16_t)((sgn(rng) << 15) | (ex(rng) << 7) | mant(rng));
  for (auto &x : ah) x = (uint16_t)((sgn(rng) << 15) | (ex(rng) << 7) | mant(rng));
  uint16_t *dw, *dws, *da;
  float *dout, *dref;
  int *flush;
  size_t const flush_n = (size_t)256 << 20;   // 1 GB of ints read before each launch (L2 and the 256 MB Infinity Cache)
  CHECK(hipMalloc(&dw, wh.size() * 2));
  CHECK(hipMalloc(&dws, ws.size() * 2));
  CHECK(hipMalloc(&da, ah.size() * 2));
  CHECK(hipMalloc(&dout, (size_t)64 * T * maxN * 4));
  CHECK(hipMalloc(&dref, (size_t)64 * T * maxN * 4));
  CHECK(hipMalloc(&flush, flush_n * 4));
  CHECK(hipMemset(flush, 1, flush_n * 4));
  CHECK(hipMemcpy(da, ah.data(), ah.size() * 2, hipMemcpyHostToDevice));
  hipEvent_t e0, e1;
  CHECK(hipEventCreate(&e0));
  CHECK(hipEventCreate(&e1));
  const char *names[4] = {"row-major", "row-major nt", "shuffled", "shuffled nt"};
  std::string json;
  for (const Case &c : cases) {
    // this shape's weight: row-major and shuffled
    for (int r = 0; r < c.N; r++)
      for (int k = 0; k < c.K; k++) {
        uint16_t const v = wh[(size_t)r * maxK + k];
        int const tile = r / 16, s = k / 32, lane = (r % 16) + 16 * ((k % 32) / 8), e = k % 8;
        ws[(((size_t)tile * (c.K / 32) + s) * 64 + lane) * 8 + e] = v;
      }
    std::vector<uint16_t> wr((size_t)c.N * c.K);
    for (int r = 0; r < c.N; r++) memcpy(&wr[(size_t)r * c.K], &wh[(size_t)r * maxK], c.K * 2);
    CHECK(hipMemcpy(dw, wr.data(), wr.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dws, ws.data(), (size_t)c.N * c.K * 2, hipMemcpyHostToDevice));
    std::vector<uint16_t> ac((size_t)T * c.K);
    for (int t = 0; t < T; t++) memcpy(&ac[(size_t)t * c.K], &ah[(size_t)t * maxK], c.K * 2);
    CHECK(hipMemcpy(da, ac.data(), ac.size() * 2, hipMemcpyHostToDevice));
    Kern k[4];
    if (c.NT == 2 && c.KP == 896) variants<2, 896>(k);
    else if (c.NT == 1 && c.KP == 896) variants<1, 896>(k);
    else if (c.NT == 1 && c.KP == 1792) variants<1, 1792>(k);
    else if (c.NT == 2 && c.KP == 512) variants<2, 512>(k);
    else if (c.NT == 1 && c.KP == 512) variants<1, 512>(k);
    else if (c.NT == 1 && c.KP == 256) variants<1, 256>(k);
    else if (c.NT == 2 && c.KP == 768) variants<2, 768>(k);
    else variants<1, 768>(k);
    int const parts = c.K / c.KP, tasks = c.N / (64 * c.NT) * parts, lds = T * c.KP * 2;
    size_t const out_n = (size_t)parts * T * c.N;
    std::vector<float> ref;
    for (int v = 0; v < 4; v++) {
      set_max_lds((const void *)k[v], lds);
      const uint16_t *wv = v < 2 ? dw : dws;
      std::vector<double> us;
      for (int rep = 0; rep < 21; rep++) {
        hipLaunchKernelGGL(flush_k, dim3(1024), dim3(256), 0, 0, flush, flush_n, flush + 1);
        CHECK(hipEventRecord(e0, 0));
        hipLaunchKernelGGL(k[v], dim3(tasks), dim3(256), lds, 0, wv, da, dout, c.N, c.K);
        CHECK(hipEventRecord(e1, 0));
        CHECK(hipEventSynchronize(e1));
        float ms;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        if (rep) us.push_back(ms * 1e3);
      }
      CHECK(hipGetLastError());
      std::vector<float> o(out_n);
      CHECK(hipMemcpy(o.data(), dout, out_n * 4, hipMemcpyDeviceToHost));
      bool same = true;
      if (v == 0) {
        ref = o;
      } else {
        same = memcmp(ref.data(), o.data(), out_n * 4) == 0;
      }
      // a few sums vs the CPU (all K parts added)
      double maxrel = 0;
      for (int r = 0; r < c.N; r += c.N / 7)
        for (int t = 0; t < T; t += 3) {
          double cpu = 0, gpu = 0;
          for (int kk = 0; kk < c.K; kk++) cpu += (double)bf2f(wr[(size_t)r * c.K + kk]) * bf2f(ac[(size_t)t * c.K + kk]);
          for (int q = 0; q < parts; q++) gpu += o[((size_t)q * T + t) * c.N + r];
          maxrel = std::max(maxrel, std::fabs(gpu - cpu) / (std::fabs(cpu) + 1e-3));
        }
      double const m = median(us), tbs = (double)c.N * c.K * 2 / (m * 1e-6) / 1e12;
      printf("N %5d K %5d, %3d-row tasks x %2d K parts = %4d tasks, %-13s: %7.2f us, %5.2f TB/s; same as row-major %s, "
             "vs CPU max rel %.1e\n", c.N, c.K, 64 * c.NT, parts, tasks, names[v], m, tbs, same ? "yes" : "NO", maxrel);
      char row[200];
      snprintf(row, sizeof(row), "%s{\"N\": %d, \"K\": %d, \"rows\": %d, \"parts\": %d, \"variant\": \"%s\", \"us\": %.3f, "
               "\"same\": %d}", json.empty() ? "" : ", ", c.N, c.K, 64 * c.NT, parts, names[v], m, same ? 1 : 0);
      json += row;
    }
  }
  printf("{\"bench\": \"gemm_load\", \"rows\": [%s]}\n", json.c_str());
  return 0;
}
