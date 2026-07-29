// M4-I2 GATE 1 (primary) -- golden-vs-ferret-fast bit-exactness for task 279,
// compiled by the SHIPPED megakernel toolchain.
//
// WHY THIS EXISTS AND NOT ONLY THE TORCH TEST. The megakernel JIT resolves nvcc
// off PATH (persistent_kernel.py: shutil.which("nvcc")) and every driver on this
// box pins /usr/local/cuda-12.8, so what ships is nvcc 12.8 + -use_fast_math.
// The box's torch is 2.13.0+cu130, and torch.utils.cpp_extension REFUSES to
// build against a different CUDA major, so the pybind harness in
// tests/runtime_python/... can only be built with nvcc 13.0. That harness is
// still worth running (it also carries the pre-existing tolerance test and the
// real-checkpoint test), but it cannot certify the shipped compiler. This file
// has no torch dependency, so it compiles under 12.8 with the shipped flags.
//
// WHAT IT CHECKS. For each shipped Qwen3.5 dense projection and each decode
// batch size, the whole projection is computed twice from identical inputs:
//   arm G: OUTPUT_SIZE/128 CTAs running linear_fp8_blockscale_task_impl_golden
//   arm F: OUTPUT_SIZE/slice CTAs running the dispatcher (ferret v011 fast path)
// and the bf16 results are compared BIT-WISE. A sub-block slice lies inside one
// 128x128 scale block, so each fast task is handed its CONTAINING block row --
// the same arithmetic the builder produces by row-replicating weight_scale.
//
// TWO DATA REGIMES, because they falsify different things:
//
//   E (exact-by-construction, the ferret task's own bar): fp8 values are small
//     integers and scales are n/8, so every intermediate -- the unscaled tile
//     sum, the scale product, the promoted accumulator -- is exact in fp32.
//     Bit-exactness is then INDEPENDENT of FMA contraction and of within-tile
//     summation order, so a mismatch means a real port defect: wrong bytes,
//     wrong fragment indexing, wrong scale row, a missed slice, a ring-depth
//     race. THIS REGIME MUST BE BIT-EXACT.
//
//   R (random, deliberately NOT exact): magnitudes are realistic and the sums
//     round. Bit-exactness here additionally requires the two paths to be
//     contracted identically by the compiler. That is a compiler scheduling
//     property, not part of the numerics contract, so a 1-ULP diff here is
//     reported rather than treated as failure -- AC-3 is the arbiter for
//     whether it reaches the tokens. Reported so the claim is calibrated.
//
// Build (both lanes):
//   nvcc -O3 -std=c++17 -arch=sm_100a -DMIRAGE_GRACE_BLACKWELL \
//        -DMIRAGE_BACKEND_USE_CUDA -DMPK_TARGET_CC=100 \
//        -I<mirage>/include/mirage/persistent_kernel \
//        -I<mirage>/include/mirage/persistent_kernel/tasks \
//        -I<mirage>/include -I<mirage>/deps/cutlass/include \
//        [-use_fast_math] --expt-relaxed-constexpr \
//        bitexact_standalone.cu -o bitexact
#include "blackwell/linear_fp8_blockscale_sm100.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using bfloat16 = type::bfloat16_t;

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e__ = (x);                                                      \
    if (e__ != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e__), __FILE__,  \
              __LINE__);                                                        \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

// One CTA per N_SLICE-row weight slice, exactly MPK's dispatch. FORCE_GOLDEN
// pins the pre-M4-I2 path at the 128-row slice.
template <int M, int N, int K, int N_SLICE, bool RESIDUAL, bool FORCE_GOLDEN>
__global__ void __launch_bounds__(WORKER_NUM_THREADS)
    projection(uint8_t const *__restrict__ a_q,
               float const *__restrict__ a_s,
               uint8_t const *__restrict__ b_q,
               float const *__restrict__ b_s,
               bfloat16 const *__restrict__ res,
               bfloat16 *__restrict__ out) {
  int const nb = blockIdx.x;
  uint8_t const *w = b_q + (size_t)nb * N_SLICE * K;
  float const *ws = b_s + (size_t)(nb * N_SLICE / 128) * (K / 128);
  bfloat16 const *r = RESIDUAL ? res + (size_t)nb * N_SLICE : nullptr;
  bfloat16 *o = out + (size_t)nb * N_SLICE;
  if constexpr (FORCE_GOLDEN) {
    kernel::linear_fp8_blockscale_task_impl_golden<bfloat16, M, N_SLICE, K, N,
                                                   RESIDUAL>(a_q, a_s, w, ws, r,
                                                             o);
  } else {
    kernel::linear_fp8_blockscale_task_impl<bfloat16, M, N_SLICE, K, N,
                                            RESIDUAL>(a_q, a_s, w, ws, r, o);
  }
}

static uint32_t rng = 0x12345678u;
static uint32_t rnd() {
  rng ^= rng << 13;
  rng ^= rng >> 17;
  rng ^= rng << 5;
  return rng;
}

template <int M, int N, int K, int N_SLICE, bool RESIDUAL, bool FORCE_GOLDEN>
static void launch(uint8_t const *a_q, float const *a_s, uint8_t const *b_q,
                   float const *b_s, bfloat16 const *res, bfloat16 *out) {
  constexpr int smem =
      FORCE_GOLDEN ? kernel::linear_fp8_blockscale::smem_bytes(M)
                   : kernel::linear_fp8_blockscale::task_smem_bytes(M, K,
                                                                   N_SLICE);
  auto *entry = projection<M, N, K, N_SLICE, RESIDUAL, FORCE_GOLDEN>;
  CUDA_CHECK(cudaFuncSetAttribute(
      entry, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  entry<<<N / N_SLICE, WORKER_NUM_THREADS, smem>>>(a_q, a_s, b_q, b_s, res, out);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

struct Verdict {
  size_t mismatches;
  int max_ulp;
  size_t unwritten;
};

template <int M, int N, int K, int N_SLICE, bool RESIDUAL>
static Verdict run_case(bool exact_regime) {
  size_t const ab = (size_t)M * K, bb = (size_t)N * K;
  int const KT = K / 128, NB = N / 128;
  std::vector<uint8_t> h_a(ab), h_b(bb);
  std::vector<float> h_as((size_t)M * KT), h_bs((size_t)NB * KT);
  std::vector<__nv_bfloat16> h_res((size_t)M * N);

  for (size_t i = 0; i < ab; ++i) {
    // regime E: small integers {-2..2}. regime R: k*0.5 in [-3.5, 3.5], all
    // exactly representable in e4m3 but whose PRODUCTS sum inexactly at K>=512.
    float v = exact_regime ? (float)((int)(rnd() % 5) - 2)
                           : 0.5f * (float)((int)(rnd() % 15) - 7);
    __nv_fp8_e4m3 q(v);
    h_a[i] = *reinterpret_cast<uint8_t *>(&q);
  }
  for (size_t i = 0; i < bb; ++i) {
    float v = exact_regime ? (float)((int)(rnd() % 5) - 2)
                           : 0.5f * (float)((int)(rnd() % 15) - 7);
    __nv_fp8_e4m3 q(v);
    h_b[i] = *reinterpret_cast<uint8_t *>(&q);
  }
  for (auto &s : h_as) {
    s = exact_regime ? (float)(1 + (int)(rnd() % 16)) / 8.0f
                     : (float)(1 + (int)(rnd() % 4096)) * 1.7e-5f;
  }
  for (auto &s : h_bs) {
    s = exact_regime ? (float)(1 + (int)(rnd() % 16)) / 8.0f
                     : (float)(1 + (int)(rnd() % 4096)) * 1.3e-5f;
  }
  for (auto &r : h_res) {
    r = __nv_bfloat16(exact_regime
                          ? (float)((int)(rnd() % 511) - 255) / 64.0f
                          : (float)((int)(rnd() % 65535) - 32767) * 3.1e-4f);
  }

  uint8_t *d_a, *d_b;
  float *d_as, *d_bs;
  __nv_bfloat16 *d_res, *d_g, *d_f;
  CUDA_CHECK(cudaMalloc(&d_a, ab));
  CUDA_CHECK(cudaMalloc(&d_b, bb));
  CUDA_CHECK(cudaMalloc(&d_as, h_as.size() * 4));
  CUDA_CHECK(cudaMalloc(&d_bs, h_bs.size() * 4));
  CUDA_CHECK(cudaMalloc(&d_res, h_res.size() * 2));
  CUDA_CHECK(cudaMalloc(&d_g, (size_t)M * N * 2));
  CUDA_CHECK(cudaMalloc(&d_f, (size_t)M * N * 2));
  CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), ab, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bb, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_as, h_as.data(), h_as.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_bs, h_bs.data(), h_bs.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), h_res.size() * 2,
                        cudaMemcpyHostToDevice));
  // 0xEE poison: a slice no task writes cannot pass by matching a zero buffer.
  CUDA_CHECK(cudaMemset(d_g, 0xEE, (size_t)M * N * 2));
  CUDA_CHECK(cudaMemset(d_f, 0xEE, (size_t)M * N * 2));

  launch<M, N, K, 128, RESIDUAL, true>(
      d_a, d_as, d_b, d_bs, reinterpret_cast<bfloat16 const *>(d_res),
      reinterpret_cast<bfloat16 *>(d_g));
  launch<M, N, K, N_SLICE, RESIDUAL, false>(
      d_a, d_as, d_b, d_bs, reinterpret_cast<bfloat16 const *>(d_res),
      reinterpret_cast<bfloat16 *>(d_f));

  std::vector<uint16_t> g((size_t)M * N), f((size_t)M * N);
  CUDA_CHECK(cudaMemcpy(g.data(), d_g, g.size() * 2, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(f.data(), d_f, f.size() * 2, cudaMemcpyDeviceToHost));

  Verdict v{0, 0, 0};
  for (size_t i = 0; i < g.size(); ++i) {
    if (g[i] == 0xEEEE) {
      ++v.unwritten;
    }
    if (f[i] == 0xEEEE) {
      ++v.unwritten;
    }
    if (g[i] != f[i]) {
      ++v.mismatches;
      int ulp = (int)g[i] - (int)f[i];
      if (ulp < 0) {
        ulp = -ulp;
      }
      if (ulp > v.max_ulp) {
        v.max_ulp = ulp;
      }
      if (v.mismatches <= 3) {
        fprintf(stderr, "    [%zu,%zu] golden=0x%04x fast=0x%04x\n", i / N,
                i % N, g[i], f[i]);
      }
    }
  }
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_as));
  CUDA_CHECK(cudaFree(d_bs));
  CUDA_CHECK(cudaFree(d_res));
  CUDA_CHECK(cudaFree(d_g));
  CUDA_CHECK(cudaFree(d_f));
  return v;
}

static int fails_E = 0, total = 0, worst_ulp_R = 0;
static size_t mism_R = 0;

#define CASE(LABEL, M, N, K, SLICE, RES)                                       \
  do {                                                                         \
    for (int regime = 0; regime < 2; ++regime) {                               \
      bool const ex = (regime == 0);                                           \
      Verdict v = run_case<M, N, K, SLICE, RES>(ex);                           \
      bool const bad = v.mismatches || v.unwritten;                            \
      printf("%-16s N=%-5d K=%-5d slice=%-4d bs=%-3d %s  %s", LABEL, N, K,     \
             SLICE, M, ex ? "E" : "R",                                         \
             bad ? "DIFF" : "BIT-EXACT");                                      \
      if (bad) {                                                               \
        printf("  mismatch=%zu maxulp=%d unwritten=%zu", v.mismatches,         \
               v.max_ulp, v.unwritten);                                        \
      }                                                                        \
      printf("\n");                                                            \
      ++total;                                                                 \
      if (ex && bad) {                                                         \
        ++fails_E;                                                             \
      }                                                                        \
      if (!ex) {                                                               \
        mism_R += v.mismatches;                                                \
        if (v.max_ulp > worst_ulp_R) {                                         \
          worst_ulp_R = v.max_ulp;                                             \
        }                                                                      \
      }                                                                        \
    }                                                                          \
  } while (0)

// Every shipped Qwen3.5 dense fp8 call site x every decode batch size. Slices
// must match builder.py's FP8_DENSE_N_SLICE.
#define ALL_SHAPES(M)                                                          \
  CASE("gdn_in_proj_qkv", M, 8192, 2048, 64, false);                           \
  CASE("gdn_in_proj_z", M, 4096, 2048, 32, false);                             \
  CASE("attn_qkvg_proj", M, 9216, 2048, 64, false);                            \
  CASE("out_proj/o_proj", M, 2048, 4096, 16, true);                            \
  CASE("shared_gate_up", M, 1024, 2048, 32, false);                            \
  CASE("shared_down", M, 2048, 512, 64, false);

int main() {
  printf("regime E = exact-by-construction (MUST be bit-exact)\n");
  printf("regime R = random/inexact (1-ULP diffs are compiler FMA "
         "contraction, reported not failed)\n\n");
  ALL_SHAPES(1)
  ALL_SHAPES(2)
  ALL_SHAPES(4)
  ALL_SHAPES(8)
  ALL_SHAPES(16)
  printf("\nregime E: %d/%d cases bit-exact\n", total / 2 - fails_E, total / 2);
  printf("regime R: %zu differing elements, worst |ULP| = %d\n", mism_R,
         worst_ulp_R);
  if (fails_E) {
    printf("GATE1_STANDALONE: FAIL (%d exact-regime cases not bit-exact)\n",
           fails_E);
    return 1;
  }
  printf("GATE1_STANDALONE: PASS\n");
  return 0;
}
