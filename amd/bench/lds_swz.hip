// Microbenchmark: the B-fragment reads of gemm_wave from an activation slice in LDS at T >= 16 (16 tokens per MFMA column
// tile), rows KA * 2 bytes apart (KA = 896 / 1024 / 1536: latent_down, the KDA projection, o_proj at T 64): plain rows,
// rows padded by 16 B (STATIC_GEMM_ACT_PAD) and the XOR swizzle (STATIC_GEMM_ACT_SWZ: piece p of row r at p ^ (r & 15)).
// MI355X dec190 / dec192 (fused KDA T 64): padded rows 382-384 us vs 388-392 plain, the swizzle 389-396: is the swizzle's
// address pattern conflict-free on gfx950's LDS like the padded one?
// One workgroup of 4 waves; per wave its weight fragments in LDS [S][64][16 B]; the activation [16 tokens][KA] bf16 (padded:
// KA + 8). Per step: the weight read, NC = 1 or 4 column tiles' B reads (tokens 16 c + l % 16 of a 64-token slice), NC MFMAs.
// Reported: shader clocks of the loop (median of 4 waves, median of 101 launches).
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 lds_swz.hip -o lds_swz
// Run:   ./lds_swz
#include <cmath>

#include "common.h"

typedef float v4f __attribute__((ext_vector_type(4)));
typedef __bf16 v8bf __attribute__((ext_vector_type(8)));

// V: 0 plain, 1 padded, 2 swizzled; NC column tiles (tokens 16 NC)
template <int S, int KA0, int NC, int V>
__global__ void __launch_bounds__(256, 1) kern(float *out, unsigned long long *clk) {
  extern __shared__ char sm[];
  int const wave = threadIdx.x >> 6, lane = threadIdx.x & 63, tok = lane % 16, g = lane / 16;
  constexpr int KA = V == 1 ? KA0 + 8 : KA0, TT = 16 * NC;
  char *const wl = sm + (size_t)wave * S * 1024;
  char *const act = sm + (size_t)4 * S * 1024; // [TT][KA] bf16
  for (int i = threadIdx.x; i < 4 * S * 256; i += 256) {
    reinterpret_cast<unsigned *>(sm)[i] = 0x3C003C00u ^ (i & 0xFF);
  }
  for (int i = threadIdx.x; i < TT * KA / 2; i += 256) {
    reinterpret_cast<unsigned *>(act)[i] = 0x3C003C00u ^ ((i * 7) & 0xFF);
  }
  __syncthreads();
  auto wf = [&](int s) { return *reinterpret_cast<v4f const *>(wl + (size_t)s * 1024 + lane * 16); };
  auto bf = [&](int s, int c) {
    int const r = 16 * c + tok;
    if constexpr (V == 2) {
      return *reinterpret_cast<v4f const *>(act + (size_t)r * KA * 2 + (((4 * s + g) ^ (r & 15)) * 16));
    }
    return *reinterpret_cast<v4f const *>(act + ((size_t)r * KA + 32 * s + 8 * g) * 2);
  };
  auto mf = [](v4f a, v4f b, v4f c) {
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, a), __builtin_bit_cast(v8bf, b), c, 0, 0, 0);
  };
  v4f acc[NC];
#pragma unroll
  for (int c = 0; c < NC; c++) acc[c] = v4f{0.f, 0.f, 0.f, 0.f};
  unsigned long long const c0 = clock64();
#pragma unroll 4
  for (int s = 0; s < S; s++) {
    v4f const w = wf(s);
#pragma unroll
    for (int c = 0; c < NC; c++) acc[c] = mf(w, bf(s, c), acc[c]);
  }
  float r = 0.f;
#pragma unroll
  for (int c = 0; c < NC; c++) r += acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3];
  asm volatile("" ::"v"(r));
  unsigned long long const c1 = clock64();
  if (lane == 0) clk[wave] = c1 - c0;
  out[wave * 64 + lane] = r;
}

template <int S, int KA0, int NC>
void shape() {
  int const lds = 4 * S * 1024 + 16 * NC * (KA0 + 8) * 2;
  void (*ks[3])(float *, unsigned long long *) = {kern<S, KA0, NC, 0>, kern<S, KA0, NC, 1>, kern<S, KA0, NC, 2>};
  float *out;
  unsigned long long *clk;
  CHECK(hipMalloc(&out, 256 * 4));
  CHECK(hipMalloc(&clk, 4 * 8));
  char const *nm[3] = {"plain", "padded 16 B", "XOR swizzle"};
  for (int v = 0; v < 3; v++) {
    set_max_lds((const void *)ks[v], lds);
    std::vector<double> c;
    for (int rep = 0; rep < 101; rep++) {
      hipLaunchKernelGGL(ks[v], dim3(1), dim3(256), lds, 0, out, clk);
      CHECK(hipDeviceSynchronize());
      unsigned long long h[4];
      CHECK(hipMemcpy(h, clk, 32, hipMemcpyDeviceToHost));
      std::vector<double> w4(h, h + 4);
      c.push_back(median(w4));
    }
    printf("S %2d KA %4d (rows %4d B) tokens %2d: %-12s %7.0f clocks\n", S, KA0, KA0 * 2, 16 * NC, nm[v], median(c));
  }
  CHECK(hipFree(out));
  CHECK(hipFree(clk));
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  CHECK(hipSetDevice(0));
  print_device(0);
  shape<16, 896, 1>();   // (LDS: 4 waves x S KB of weights + 16 NC rows x (KA + 8) x 2 B <= 160 KB)
  shape<16, 896, 2>();
  shape<8, 896, 4>();
  shape<16, 1024, 2>();
  shape<16, 1536, 1>();
  shape<16, 512, 4>();
  return 0;
}
