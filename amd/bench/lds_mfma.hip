// Microbenchmark: how long does gemm_tile's K loop take once its weight slice is in LDS (the STATIC_GEMM_LDSW path: a task
// whose slice was copied there while it waited, e.g. o_proj or latent_up)? MI355X 10-16: the ISA of the K loop waits
// lgkmcnt(0) for each step's activation read and chains every MFMA on one accumulator, so a step costs about the LDS
// latency plus the MFMA latency; preloading the weights then saves less than the stream time.
// One workgroup of 4 waves (as a task), its LDS: per wave [S][64 lanes][16 B] of weight fragments, the activation
// [8 tokens][KS] bf16. Variants of the loop over S steps (one row tile per wave, NT = 1):
//   0 as gemm_wave's WLDS path: per step the weight read, the activation read, one MFMA on acc (unroll 4)
//   1 reads in chunks of 8 steps (16 ds_read_b128 issued together), MFMAs on 2 accumulators (even / odd steps)
//   2 as 1 with 4 accumulators
//   3 as 0, the activation rows padded by 16 B (rows KS * 2 bytes apart put the 8 tokens' fragments on the same LDS banks)
//   4 as 1, padded
// Reported: shader clocks for the loop (median of the 4 waves, median of 101 launches; wave entry -> after the last MFMA's
// result is used), and the results' check (all variants: the same sums up to fp32 order).
//
// Build: hipcc -std=c++17 --offload-arch=gfx950 -O3 lds_mfma.hip -o lds_mfma
// Run:   ./lds_mfma
#include <cmath>

#include "common.h"

typedef float v4f __attribute__((ext_vector_type(4)));
typedef __bf16 v8bf __attribute__((ext_vector_type(8)));
constexpr int T = 8;

template <int S, int V>
__global__ void __launch_bounds__(256, 1) kern(float *out, unsigned long long *clk) {
  extern __shared__ char sm[];
  int const wave = threadIdx.x >> 6, lane = threadIdx.x & 63, tok = lane % 16, g = lane / 16;
  constexpr int KS = S * 32, KA = (V >= 3) ? KS + 8 : KS; // (the activation's row pitch in elements)
  char *const wl = sm + (size_t)wave * S * 1024;              // this wave's weights
  char *const act = sm + (size_t)4 * S * 1024;                 // [T][KS] bf16
  for (int i = threadIdx.x; i < 4 * S * 256; i += 256) {       // fill (weights: small values)
    reinterpret_cast<unsigned *>(sm)[i] = 0x3C003C00u ^ (i & 0xFF);
  }
  for (int i = threadIdx.x; i < T * KA / 2; i += 256) {
    reinterpret_cast<unsigned *>(act)[i] = 0x3C003C00u ^ ((i * 7) & 0xFF);
  }
  __syncthreads();
  auto wf = [&](int s) { return *reinterpret_cast<v4f const *>(wl + (size_t)s * 1024 + lane * 16); };
  auto bf = [&](int s) { return *reinterpret_cast<v4f const *>(act + ((size_t)(tok & (T - 1)) * KA + 32 * s + 8 * g) * 2); };
  auto mf = [](v4f a, v4f b, v4f c) {
    return __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, a), __builtin_bit_cast(v8bf, b), c, 0, 0, 0);
  };
  v4f acc = {0.f, 0.f, 0.f, 0.f};
  unsigned long long const c0 = clock64();
  if constexpr (V == 0 || V == 3) {
#pragma unroll 4
    for (int s = 0; s < S; s++) {
      v4f const w = wf(s);
      acc = mf(w, bf(s), acc);
    }
  } else {
    constexpr int NA = V == 2 ? 4 : 2, CH = 8;
    v4f a[NA];
#pragma unroll
    for (int j = 0; j < NA; j++) a[j] = v4f{0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int s0 = 0; s0 < S; s0 += CH) {
      v4f w[CH], b[CH];
#pragma unroll
      for (int i = 0; i < CH; i++) {
        if (s0 + i < S) {
          w[i] = wf(s0 + i);
          b[i] = bf(s0 + i);
        }
      }
#pragma unroll
      for (int i = 0; i < CH; i++) {
        if (s0 + i < S) a[i % NA] = mf(w[i], b[i], a[i % NA]);
      }
    }
#pragma unroll
    for (int j = 0; j < NA; j++) acc += a[j];
  }
  float const r = acc[0] + acc[1] + acc[2] + acc[3]; // (the results used: the loop's last MFMA has to finish)
  asm volatile("" ::"v"(r));
  unsigned long long const c1 = clock64();
  if (lane == 0) clk[blockIdx.x * 4 + wave] = c1 - c0;
  out[(blockIdx.x * 4 + wave) * 64 + lane] = r;
}

template <int S>
void shape() {
  int const lds = 4 * S * 1024 + T * (S * 32 + 8) * 2;
  void (*ks[5])(float *, unsigned long long *) = {kern<S, 0>, kern<S, 1>, kern<S, 2>, kern<S, 3>, kern<S, 4>};
  float *out;
  unsigned long long *clk;
  CHECK(hipMalloc(&out, 256 * 64 * 4));
  CHECK(hipMalloc(&clk, 4 * 8));
  std::vector<float> ref(256);
  for (int v = 0; v < 5; v++) {
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
    std::vector<float> o(256);
    CHECK(hipMemcpy(o.data(), out, 256 * 4, hipMemcpyDeviceToHost));
    double md = 0;
    if (v == 0) ref = o;
    for (int i = 0; i < 256; i++) md = std::max(md, (double)std::fabs(o[i] - ref[i]) / (std::fabs(ref[i]) + 1e-6));
    char const *nm[5] = {"as gemm_wave (WLDS)", "chunks of 8, 2 acc", "chunks of 8, 4 acc", "as 0, act padded", "as 1, act padded"};
    printf("S = %2d steps (%5d LDS bytes): %-22s %6.0f clocks (%.2f us at 2.4 GHz); vs variant 0 max rel %.1e\n", S, lds, nm[v],
           median(c), median(c) / 2400.0, md);
  }
  CHECK(hipFree(out));
  CHECK(hipFree(clk));
}

int main() {
  setvbuf(stdout, nullptr, _IOLBF, 0);
  CHECK(hipSetDevice(0));
  print_device(0);
  shape<24>();
  shape<32>();
  shape<16>();
  return 0;
}
