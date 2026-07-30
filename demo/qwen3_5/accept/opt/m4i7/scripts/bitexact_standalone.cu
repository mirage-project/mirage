// M4-I7 gate 1: bit-exactness of every moe_fp8_blockscale fast path against the
// FROZEN golden body, with NO torch dependency so it can be built by the SHIPPED
// JIT toolchain (nvcc 12.8) in both flag lanes, including -use_fast_math.
//
// Why a torch-free harness exists at all: the megakernel JIT resolves nvcc off
// PATH (12.8 on this box) but the box's torch is +cu130 and
// torch.utils.cpp_extension refuses a CUDA-major mismatch, so the pybind test
// can only certify 13.0. (M4-I2 hit the same wall.)
//
// WHAT IS COMPARED. Each (family, geometry, n_live, regime) case is computed:
//   * golden, one CTA, expert_offset=0 / expert_stride=1  -- the reference
//   * moe_impl_path<PATH> for PATH = 0, 1, 2, at the MPK dispatch geometry
//     (expert_offset = blockIdx.x, expert_stride = gridDim.x = 128, which is
//     exactly what task_register.cc emits: grid.x = min(num_experts, mbt*topk))
//   * the shipped dispatcher moe_fp8_blockscale_task_impl at that same geometry
// and every one must be BITWISE equal to golden. Outputs are poisoned with 0xEE
// first, so an unwritten element cannot pass by matching a zeroed buffer, and
// non-routed (token, slot) rows must still read 0xEE afterwards.
//
// TWO DATA REGIMES, because they falsify different things:
//   E  small-integer fp8 bytes + n/8 scales -- the product is exact in fp32, so
//      bit-exactness is independent of FMA contraction and a mismatch is a real
//      port defect.
//   R  full-range random -- bit-exactness additionally requires identical
//      compiler contraction across the two bodies. Reported, and expected,
//      because the promotion sequence is identical by construction.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t e_ = (x);                                                      \
    if (e_ != cudaSuccess) {                                                    \
      printf("CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      exit(2);                                                                  \
    }                                                                           \
  } while (0)

// WORKER_NUM_THREADS / NUM_THREADS_PER_WARP / bfloat16 come from the task
// header's own includes (common/worker_config.h, common/common_header.cuh), so
// this harness sees exactly the megakernel's values.
#define CP_ASYNC_SM80_ENABLED 1

#include "blackwell/moe_fp8_blockscale_sm100.cuh"

// the megakernel TU gets this from the transpiler preamble
using bfloat16 = __nv_bfloat16;

constexpr int BATCH = 16;
constexpr int NUM_TOPK = 8;
constexpr int NUM_EXPERTS = 256;
// task_register.cc: expert_stride = bgraph.grid_dim.x, and builder.py sets
// grid.x = min(num_experts, max_num_batched_tokens * topk) = min(256, 16*8).
constexpr int MPK_GRID_X = 128;

// ---------------- launchers ----------------
template <bool W13, int OUT_N, int ORIG_N, int K>
__global__ void __launch_bounds__(WORKER_NUM_THREADS)
    k_golden(void const *a, void const *as, void const *w, void const *ws,
             void const *rt, void const *mk, void *o) {
  kernel::golden::moe_fp8_blockscale_task_impl<bfloat16, BATCH, NUM_TOPK,
                                              NUM_EXPERTS, OUT_N, ORIG_N, K,
                                              W13>(a, as, w, ws, rt, mk, o, 0,
                                                   1);
}

template <bool W13, int OUT_N, int ORIG_N, int K, int PATH>
__global__ void __launch_bounds__(WORKER_NUM_THREADS)
    k_path(void const *a, void const *as, void const *w, void const *ws,
           void const *rt, void const *mk, void *o) {
  kernel::moe_impl_path<bfloat16, BATCH, NUM_TOPK, NUM_EXPERTS, OUT_N, ORIG_N,
                        K, W13, PATH>(a, as, w, ws, rt, mk, o,
                                      (int)blockIdx.x, (int)gridDim.x);
}

template <bool W13, int OUT_N, int ORIG_N, int K>
__global__ void __launch_bounds__(WORKER_NUM_THREADS)
    k_dispatch(void const *a, void const *as, void const *w, void const *ws,
               void const *rt, void const *mk, void *o) {
  kernel::moe_fp8_blockscale_task_impl<bfloat16, BATCH, NUM_TOPK, NUM_EXPERTS,
                                       OUT_N, ORIG_N, K, W13>(
      a, as, w, ws, rt, mk, o, (int)blockIdx.x, (int)gridDim.x);
}

// ---------------- host data ----------------
struct Buffers {
  uint8_t *a, *w;
  float *as, *ws;
  int32_t *rt, *mk;
  __nv_bfloat16 *o;
  size_t o_elems;
  int nact;
  std::vector<int32_t> h_rt;
};

static uint8_t f2e4m3(float v) {
  __nv_fp8_storage_t s = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  return (uint8_t)s;
}

// regime E: small integers, exactly representable in e4m3 and exact in fp32
// products; regime R: full-range random.
static float draw(std::mt19937 &g, bool exact) {
  if (exact) {
    static const float v[] = {-8, -6, -4, -3, -2, -1, 1, 2, 3, 4, 6, 8};
    return v[g() % 12];
  }
  std::uniform_real_distribution<float> d(-2.0f, 2.0f);
  return d(g);
}

template <bool W13, int N, int K>
static Buffers make(int n_live, bool exact, uint32_t seed) {
  std::mt19937 g(seed);
  Buffers b{};
  size_t const a_rows = W13 ? (size_t)BATCH : (size_t)BATCH * NUM_TOPK;
  size_t const kt = K / 128;

  std::vector<uint8_t> ha(a_rows * K);
  std::vector<float> has(a_rows * kt);
  for (auto &x : ha) {
    x = f2e4m3(draw(g, exact));
  }
  for (size_t i = 0; i < has.size(); ++i) {
    has[i] = exact ? (float)((g() % 8) + 1) / 8.0f
                   : std::uniform_real_distribution<float>(0.01f, 0.5f)(g);
  }

  // Only the ACTIVATED experts' weights are read, but allocate the full tensor
  // so the pointer arithmetic is the shipped one.
  std::vector<uint8_t> hw((size_t)NUM_EXPERTS * N * K);
  std::vector<float> hws((size_t)NUM_EXPERTS * (N / 128) * kt);
  for (auto &x : hw) {
    x = f2e4m3(draw(g, exact));
  }
  for (size_t i = 0; i < hws.size(); ++i) {
    hws[i] = exact ? (float)((g() % 8) + 1) / 8.0f
                   : std::uniform_real_distribution<float>(0.01f, 0.5f)(g);
  }

  // routing[expert, token] = slot + 1 for routed pairs, 0 otherwise; each of the
  // n_live tokens draws NUM_TOPK DISTINCT experts (top-k has no repeats).
  std::vector<int32_t> hrt((size_t)NUM_EXPERTS * BATCH, 0);
  std::vector<int32_t> hmk(NUM_EXPERTS + 1, 0);
  std::vector<uint8_t> seen(NUM_EXPERTS, 0);
  for (int t = 0; t < n_live; ++t) {
    std::vector<uint8_t> used(NUM_EXPERTS, 0);
    for (int s = 0; s < NUM_TOPK; ++s) {
      int e;
      do {
        e = (int)(g() % NUM_EXPERTS);
      } while (used[e]);
      used[e] = 1;
      hrt[(size_t)e * BATCH + t] = s + 1;
      seen[e] = 1;
    }
  }
  int nact = 0;
  for (int e = 0; e < NUM_EXPERTS; ++e) {
    if (seen[e]) {
      hmk[nact++] = e;
    }
  }
  hmk[NUM_EXPERTS] = nact;
  b.nact = nact;
  b.h_rt = hrt;

  b.o_elems = (size_t)BATCH * NUM_TOPK * N;
  CUDA_CHECK(cudaMalloc(&b.a, ha.size()));
  CUDA_CHECK(cudaMalloc(&b.as, has.size() * 4));
  CUDA_CHECK(cudaMalloc(&b.w, hw.size()));
  CUDA_CHECK(cudaMalloc(&b.ws, hws.size() * 4));
  CUDA_CHECK(cudaMalloc(&b.rt, hrt.size() * 4));
  CUDA_CHECK(cudaMalloc(&b.mk, hmk.size() * 4));
  CUDA_CHECK(cudaMalloc(&b.o, b.o_elems * 2));
  CUDA_CHECK(cudaMemcpy(b.a, ha.data(), ha.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(b.as, has.data(), has.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(b.w, hw.data(), hw.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(b.ws, hws.data(), hws.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(b.rt, hrt.data(), hrt.size() * 4, cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(b.mk, hmk.data(), hmk.size() * 4, cudaMemcpyHostToDevice));
  return b;
}

static void freeb(Buffers &b) {
  cudaFree(b.a);
  cudaFree(b.as);
  cudaFree(b.w);
  cudaFree(b.ws);
  cudaFree(b.rt);
  cudaFree(b.mk);
  cudaFree(b.o);
}

static int g_fail = 0, g_cases = 0;

static void poison(Buffers &b) {
  CUDA_CHECK(cudaMemset(b.o, 0xEE, b.o_elems * 2));
}

static std::vector<uint16_t> grab(Buffers &b) {
  std::vector<uint16_t> h(b.o_elems);
  CUDA_CHECK(cudaMemcpy(h.data(), b.o, b.o_elems * 2, cudaMemcpyDeviceToHost));
  return h;
}

// Verify (a) every routed element was written (not 0xEEEE) and (b) every
// non-routed (token, slot) row is untouched.
template <int N>
static void audit_coverage(Buffers &b, std::vector<uint16_t> const &h,
                           char const *tag) {
  size_t unwritten = 0, clobbered = 0;
  std::vector<uint8_t> routed((size_t)BATCH * NUM_TOPK, 0);
  for (int e = 0; e < NUM_EXPERTS; ++e) {
    for (int t = 0; t < BATCH; ++t) {
      int s = b.h_rt[(size_t)e * BATCH + t];
      if (s > 0) {
        routed[(size_t)t * NUM_TOPK + (s - 1)] = 1;
      }
    }
  }
  for (int t = 0; t < BATCH; ++t) {
    for (int s = 0; s < NUM_TOPK; ++s) {
      size_t base = ((size_t)t * NUM_TOPK + s) * N;
      for (int n = 0; n < N; ++n) {
        bool poisoned = h[base + n] == 0xEEEE;
        if (routed[(size_t)t * NUM_TOPK + s] && poisoned) {
          ++unwritten;
        }
        if (!routed[(size_t)t * NUM_TOPK + s] && !poisoned) {
          ++clobbered;
        }
      }
    }
  }
  if (unwritten || clobbered) {
    printf("      [FAIL] %s coverage: unwritten=%zu clobbered=%zu\n", tag,
           unwritten, clobbered);
    ++g_fail;
  }
}

template <bool W13, int OUT_N, int ORIG_N, int K>
static void run_case(int n_live, bool exact, uint32_t seed, char const *label) {
  Buffers b = make<W13, ORIG_N, K>(n_live, exact, seed);
  constexpr int N_SLICES = ORIG_N / OUT_N;
  constexpr size_t WS_STRIDE = (size_t)(OUT_N / 128) * (K / 128);
  constexpr size_t W_STRIDE = (size_t)OUT_N * K;
  constexpr int GOLD_SMEM = kernel::golden::moe_fp8_blockscale::smem_bytes(BATCH);
  int const smem = mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE;

  printf("    %s  n_live=%2d nact=%3d regime=%c  OUT_N=%4d slices=%d\n", label,
         n_live, b.nact, exact ? 'E' : 'R', OUT_N, N_SLICES);

  // ---- golden reference: one CTA per N slice, offset 0 / stride 1 ----
  poison(b);
  CUDA_CHECK(cudaFuncSetAttribute(k_golden<W13, OUT_N, ORIG_N, K>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  GOLD_SMEM));
  for (int sl = 0; sl < N_SLICES; ++sl) {
    k_golden<W13, OUT_N, ORIG_N, K><<<1, WORKER_NUM_THREADS, GOLD_SMEM>>>(
        b.a, b.as, b.w + sl * W_STRIDE, b.ws + sl * WS_STRIDE, b.rt, b.mk,
        (void *)(b.o + (size_t)sl * OUT_N));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> ref = grab(b);
  audit_coverage<ORIG_N>(b, ref, "golden");

  auto cmp = [&](char const *tag, std::vector<uint16_t> const &got) {
    ++g_cases;
    size_t diff = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      if (ref[i] != got[i]) {
        ++diff;
      }
    }
    printf("      %-9s %s%s\n", tag, diff == 0 ? "bit-exact" : "MISMATCH ",
           diff == 0 ? "" : (std::to_string(diff) + " differing").c_str());
    if (diff) {
      ++g_fail;
    }
  };

  // PATHID is a literal so the launch expression names the function directly;
  // a function POINTER cannot be launched with <<<>>>.
#define RUN_PATH(PATHID)                                                       \
  do {                                                                         \
    poison(b);                                                                 \
    CUDA_CHECK(cudaFuncSetAttribute(                                           \
        k_path<W13, OUT_N, ORIG_N, K, PATHID>,                                 \
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem));                   \
    for (int sl = 0; sl < N_SLICES; ++sl) {                                    \
      k_path<W13, OUT_N, ORIG_N, K, PATHID>                                    \
          <<<MPK_GRID_X, WORKER_NUM_THREADS, smem>>>(                          \
              b.a, b.as, b.w + sl * W_STRIDE, b.ws + sl * WS_STRIDE, b.rt,     \
              b.mk, (void *)(b.o + (size_t)sl * OUT_N));                       \
    }                                                                          \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
    std::vector<uint16_t> got = grab(b);                                       \
    audit_coverage<ORIG_N>(b, got, "PATH" #PATHID);                            \
    cmp("PATH" #PATHID, got);                                                  \
  } while (0)

  RUN_PATH(0);
  if constexpr (kernel::moe_fp8_blockscale_fast::path_admissible(BATCH, 1,
                                                                OUT_N, K, W13)) {
    RUN_PATH(1);
  }
  if constexpr (kernel::moe_fp8_blockscale_fast::path_admissible(BATCH, 2,
                                                                OUT_N, K, W13)) {
    RUN_PATH(2);
  }
#undef RUN_PATH

  poison(b);
  CUDA_CHECK(cudaFuncSetAttribute(k_dispatch<W13, OUT_N, ORIG_N, K>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  smem));
  for (int sl = 0; sl < N_SLICES; ++sl) {
    k_dispatch<W13, OUT_N, ORIG_N, K><<<MPK_GRID_X, WORKER_NUM_THREADS, smem>>>(
        b.a, b.as, b.w + sl * W_STRIDE, b.ws + sl * WS_STRIDE, b.rt, b.mk,
        (void *)(b.o + (size_t)sl * OUT_N));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint16_t> gd = grab(b);
  audit_coverage<ORIG_N>(b, gd, "dispatch");
  cmp("dispatch", gd);

  freeb(b);
}

int main(int argc, char **argv) {
  int dev = 0;
  CUDA_CHECK(cudaSetDevice(dev));
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  printf("device: %s  SMs=%d  smem/block(optin)=%d\n", p.name,
         p.multiProcessorCount, (int)p.sharedMemPerBlockOptin);
  printf("MAX_DYNAMIC_SHARED_MEMORY_SIZE = %d\n",
         mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE);
  printf("smem_bytes_k: w13 PATH0=%d PATH1=%d PATH2=%d | w2 PATH0=%d PATH1=%d "
         "PATH2=%d\n",
         kernel::moe_fp8_blockscale_fast::smem_bytes_k(16, 0, 2048),
         kernel::moe_fp8_blockscale_fast::smem_bytes_k(16, 1, 2048),
         kernel::moe_fp8_blockscale_fast::smem_bytes_k(16, 2, 2048),
         kernel::moe_fp8_blockscale_fast::smem_bytes_k(16, 0, 512),
         kernel::moe_fp8_blockscale_fast::smem_bytes_k(16, 1, 512),
         kernel::moe_fp8_blockscale_fast::smem_bytes_k(16, 2, 512));

  int const lives[] = {1, 2, 4, 8, 16};
  for (int r = 0; r < 2; ++r) {
    bool exact = (r == 0);
    printf("\n=== regime %c ===\n", exact ? 'E' : 'R');
    for (int i = 0; i < 5; ++i) {
      uint32_t seed = 1000u + 17u * i + 7u * r;
      // unit-test geometry: the whole per-expert N in one task
      run_case<true, 1024, 1024, 2048>(lives[i], exact, seed, "w13 full-N ");
      run_case<false, 2048, 2048, 512>(lives[i], exact, seed, "w2  full-N ");
      // shipped MPK geometry: moe_n_splits = 2
      run_case<true, 512, 1024, 2048>(lives[i], exact, seed, "w13 split2 ");
      run_case<false, 1024, 2048, 512>(lives[i], exact, seed, "w2  split2 ");
    }
  }
  printf("\n%d arms compared, %d failures\n", g_cases, g_fail);
  printf(g_fail ? "GATE1: FAIL\n" : "GATE1: PASS\n");
  return g_fail ? 1 : 0;
}
