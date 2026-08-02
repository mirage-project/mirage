// UMMA layout oracle for the Blackwell (sm_100) transpiler backend.
//
// WHY THIS EXISTS
// ---------------
// This models the *generated* kernel's MMA path while letting you vary one
// thing at a time (the smem fill layout, the swizzle parameters). It is what
// finally located the swizzle bug that made every generated matmul return the
// right values at wrong positions: sweeping <B,M,S> against a PyTorch-
// equivalent reference singled out Swizzle<3,3,3>, while every attempt to
// reason forward about CuTe layout algebra was wrong. See the verified/limits
// note below before trusting a result.
//
// It iterates in ~30s (one nvcc invocation) instead of a full Mirage rebuild
// plus Python round-trip, so prefer it for any question of the form "which
// layout/swizzle/descriptor does the UMMA actually read?".
//
// WHAT IS VERIFIED IN THIS PRESERVED COPY
//   * The descriptor dump (start / LBO / SBO / layout_type per k-block) matches
//     the bitfields in cute/arch/mma_sm100_desc.hpp.
//   * CAND 4 (A zeroed -> all-zero output) confirms the MMA really reads the
//     smem you filled. An early version of this harness "worked" while its
//     output did not depend on the fill at all -- always run this check.
//   * The sweep DISCRIMINATES correctly: of all <B,M,S>, only Swizzle<3,3,3>
//     produces a correct row 0; every other combination is wrong by element 0
//     or 1. That is the signal that identified the transpiler's swizzle bug.
//
// KNOWN LIMITATION -- NOT AN END-TO-END ORACLE AS PRESERVED.
// In the live scratch version the winning sweep entry reached maxabs=0.0. This
// copy reaches a correct row 0 but diverges from row 1 (maxabs != 0), so some
// epilogue detail differs from the generated kernel. Reverting the LA/LB/LC
// swizzle to <3,3,4> was tried and is NOT the cause. Until that is tracked
// down, use this for operand-layout discrimination and descriptor inspection,
// and confirm any end-to-end claim against the real generated kernel (dump it
// via KNGraph.compile()["code"]) rather than against this harness alone.
//
// BUILD / RUN
//   nvcc -std=c++17 -O3 -arch=sm_100a -DMIRAGE_BACKEND_USE_CUDA \
//        --expt-relaxed-constexpr -I deps/cutlass/include \
//        -I include/mirage/transpiler/runtime -lcublas \
//        -o /tmp/umma_harness tools/blackwell/umma_layout_harness.cu
//   /tmp/umma_harness
//
// It also decodes the UMMA smem descriptors (start/LBO/SBO/layout_type) per
// k-block; see cute/arch/mma_sm100_desc.hpp for the bitfield definitions.
#define NUM_GPUS 1
#define USE_NVSHMEM false
#define MIRAGE_BLACKWELL
#include "runtime.h"
#include <cstdio>
using namespace cute;
// runtime.h declares these for the generated-kernel entry path; unused here.
static void _init() {}
void _execute_mugraph(std::vector<void const*>, std::vector<void*>, void*, CUstream_st*, void*) {}

static constexpr int M = 128, K = 64, N = 64;

// CAND selects how the A operand is written into smem.
template <int CAND, int SB = 3, int SM = 3, int SS = 3>
__global__ void __launch_bounds__(128)
    harness(bfloat16_t *out, bfloat16_t const *Ag, bfloat16_t const *Bg) {
  int thread_idx = threadIdx.x;
  static constexpr int NUM_THREADS = 128;
  auto cluster_shape = make_shape(Int<1>{}, Int<1>{}, Int<1>{});
  uint32_t elect_one_warp = (threadIdx.x / 32 == 0);
  bool elect_one_cta = true;
  extern __shared__ char buf[];
  uint64_t *mma_barrier_ptr = (uint64_t *)(buf + 16);
  uint32_t *tmem_base_ptr = (uint32_t *)(buf + 0);
  bfloat16_t *sA = (bfloat16_t *)(buf + 1024);
  bfloat16_t *sB = (bfloat16_t *)(buf + 1024 + 16384);
  bfloat16_t *sC = (bfloat16_t *)(buf + 1024 + 16384 + 8192);
  *((uint128_t *)buf) = 0ul;

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  if (elect_one_warp) {
    tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns, tmem_base_ptr);
  }
  __syncthreads();
  if (elect_one_warp && cute::elect_one_sync()) {
    cute::initialize_barrier(*mma_barrier_ptr, 1);
  }
  auto tiled_mma = cutlass::gemm::collective::detail::sm100_make_1sm_trivial_tiled_mma<
      bfloat16_t, bfloat16_t, float, Shape<Int<M>, Int<N>>,
      decltype(cluster_shape), UMMA::Major::K, UMMA::Major::MN>();
  auto mma_tiler = make_shape(tile_size<0>(tiled_mma), tile_size<1>(tiled_mma),
                              tile_size<2>(tiled_mma) * _4{});
  using LA = decltype(composition(Swizzle<3,3,4>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using LB = decltype(composition(Swizzle<3,3,4>{}, Layout<Shape<Int<64>, Int<64>>,  Stride<Int<1>, Int<64>>>{}));
  using LC = decltype(composition(Swizzle<3,3,4>{}, Layout<Shape<Int<64>, Int<128>>, Stride<Int<1>, Int<64>>>{}));
  using Kern = tb::Blackwell_Matmul<bfloat16_t, true, false, LA, LB, LC, NUM_THREADS,
      0, false, false, false, false, 2, decltype(cluster_shape),
      decltype(tiled_mma), decltype(mma_tiler)>;
  auto acc = Kern::get_mma_tC(blockIdx.x, blockIdx.y, *tmem_base_ptr);
  __syncthreads();

  auto pipeA = typename Kern::DstPipeLayout_A{};
  auto pipeB = typename Kern::DstPipeLayout_B{};
  auto atomA = typename Kern::SmemLayoutAtom_A{};
  auto tileA = tile_to_shape(atomA, make_shape(Int<M>{}, Int<K>{}));

  // Fill exactly as InputNonChunkedSyncCopy does: linear index into the
  // destination layout, paired with the same linear index into the gmem tile
  // layout the transpiler emits. CAND 0 must therefore reproduce the real
  // generated kernel; if it does not, the harness is not a valid oracle.
  // NOTE: do NOT reintroduce a composition of DstPipeLayout with a linearizing
  // layout here. That drops the smem_ptr_flag[16b] bit-granularity semantics and
  // silently yields different addresses -- it was tried and is wrong. Evaluate
  // the pipe layout directly.
  auto pipeA_ = typename Kern::DstPipeLayout_A{};
  auto pipeB_ = typename Kern::DstPipeLayout_B{};
  auto srcA = Layout<Shape<Int<K>, Int<M>>, Stride<_1, Int<K>>>{};   // (k,m):(1,K)
  auto srcB = Layout<Shape<Int<N>, Int<K>>, Stride<_1, Int<N>>>{};   // (n,k):(1,N)
  for (int i = thread_idx; i < M * K; i += NUM_THREADS) {
    int off;
    if      (CAND == 10) { int m = i % M, k = i / M; off = Swizzle<SB,SM,SS>{}(K * m + k); }
    else if (CAND == 0) off = pipeA_(idx2crd(i, shape(pipeA_)));
    else if (CAND == 2) off = tileA(idx2crd(i, shape(tileA)));
    else                off = LA{}(i);
    sA[off] = (CAND == 4) ? bfloat16_t(0.f) : Ag[srcA(i)];
  }
  for (int i = thread_idx; i < K * N; i += NUM_THREADS) {
    int boff;
    if (CAND == 10) { int n = i % N, k = i / N; boff = Swizzle<SB,SM,SS>{}(n + N * k); }
    else            boff = pipeB_(idx2crd(i, shape(pipeB_)));
    sB[boff] = Bg[srcB(i)];
  }
  __syncthreads();

  if (CAND == 0 && thread_idx == 0) {
    // Decode the UMMA smem descriptors the MMA actually issues.
    auto cta_mma = tiled_mma.get_slice(_0{});
    Tensor tCsA_ = make_tensor(make_smem_ptr(sA), typename Kern::DstPipeLayout_A{});
    Tensor tCsB_ = make_tensor(make_smem_ptr(sB), typename Kern::DstPipeLayout_B{});
    Tensor tCrA_ = cta_mma.make_fragment_A(tCsA_);
    Tensor tCrB_ = cta_mma.make_fragment_B(tCsB_);
    for (int kb = 0; kb < 2; ++kb) {
      uint64_t da = *reinterpret_cast<uint64_t const*>(&(tCrA_(_,_,kb,0).data()));
      uint64_t db = *reinterpret_cast<uint64_t const*>(&(tCrB_(_,_,kb,0).data()));
      // UMMA desc: [13:0] start>>4, [29:16] LBO>>4, [45:32] SBO>>4, [52:49] base, [63:61] swizzle
      printf("  kblk %d  A: start=%llu LBO=%llu SBO=%llu sw=%llu | B: start=%llu LBO=%llu SBO=%llu sw=%llu\n",
        kb,
        (unsigned long long)((da)&0x3FFF), (unsigned long long)((da>>16)&0x3FFF),
        (unsigned long long)((da>>32)&0x3FFF), (unsigned long long)((da>>61)&0x7),
        (unsigned long long)((db)&0x3FFF), (unsigned long long)((db>>16)&0x3FFF),
        (unsigned long long)((db>>32)&0x3FFF), (unsigned long long)((db>>61)&0x7));
    }
  }
  for (uint32_t f = 0; f < 1; f++) {
    if (elect_one_cta && elect_one_warp) {
      Kern::run(acc, sA, sB, f, tiled_mma, 0);
      cutlass::arch::umma_arrive(mma_barrier_ptr);
    }
  }
  cute::wait_barrier(*mma_barrier_ptr, 0);
  Kern::write_tC_to_sC(sC, acc, thread_idx);
  tb::wg_sync<128>(8);
  auto lc = LC{};
  auto dstC = Layout<Shape<Int<N>, Int<M>>, Stride<_1, Int<N>>>{};
  for (int i = thread_idx; i < M * N; i += NUM_THREADS) {
    out[dstC(i)] = sC[lc(i)];
  }
  __syncthreads();
  if (elect_one_warp) {
    tmem_allocator.release_allocation_lock();
    tmem_allocator.free(*tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns);
  }
}

template <int CAND, int SB = 3, int SM = 3, int SS = 3> void go(bfloat16_t *dO, bfloat16_t *dA, bfloat16_t *dB, float *hRef) {
  size_t smem = 1024 + 16384 + 8192 + 16384 + 1024;
  cudaFuncSetAttribute(harness<CAND,SB,SM,SS>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaMemset(dO, 0, M * N * sizeof(bfloat16_t));
  harness<CAND,SB,SM,SS><<<1, 128, smem>>>(dO, dA, dB);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("CAND %d: CUDA error %s\n", CAND, cudaGetErrorString(e)); return; }
  std::vector<bfloat16_t> h(M * N);
  cudaMemcpy(h.data(), dO, M * N * sizeof(bfloat16_t), cudaMemcpyDeviceToHost);
  double maxabs = 0; int firstbad = -1;
  for (int i = 0; i < M * N; ++i) {
    double d = std::abs(float(h[i]) - hRef[i]);
    if (d > maxabs) maxabs = d;
    if (d > 0.5 && firstbad < 0) firstbad = i;
  }
  printf("CAND %d sw<%d,%d,%d>: maxabs=%.3f firstbad=(%d,%d)  row0[:6]=[%.0f %.0f %.0f %.0f %.0f %.0f]\n",
         CAND, SB, SM, SS, maxabs, firstbad < 0 ? -1 : firstbad / N, firstbad < 0 ? -1 : firstbad % N,
         float(h[0]), float(h[1]), float(h[2]), float(h[3]), float(h[4]), float(h[5]));
}

int main() {
  std::vector<bfloat16_t> hA(M * K), hB(K * N);
  std::vector<float> hRef(M * N, 0.f);
  for (int m = 0; m < M; ++m) for (int k = 0; k < K; ++k) hA[m*K+k] = bfloat16_t(float((m * 7 + k) % 13));
  for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) hB[k*N+n] = bfloat16_t(float(k == n ? 1 : 0));
  for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
    float s = 0; for (int k = 0; k < K; ++k) s += float(hA[m*K+k]) * float(hB[k*N+n]);
    hRef[m*N+n] = s;
  }
  bfloat16_t *dA, *dB, *dO;
  cudaMalloc(&dA, hA.size()*2); cudaMalloc(&dB, hB.size()*2); cudaMalloc(&dO, M*N*2);
  cudaMemcpy(dA, hA.data(), hA.size()*2, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), hB.size()*2, cudaMemcpyHostToDevice);
  go<0>(dO, dA, dB, hRef.data());
  go<2>(dO, dA, dB, hRef.data());
  go<3>(dO, dA, dB, hRef.data());
  go<4>(dO, dA, dB, hRef.data());  // sanity: A all zeros -> output must be all zeros
  // <B,M,S> sweep: the exact experiment that identified Swizzle<3,3,3>.
  go<10,1,3,3>(dO, dA, dB, hRef.data());
  go<10,1,3,4>(dO, dA, dB, hRef.data());
  go<10,1,4,3>(dO, dA, dB, hRef.data());
  go<10,1,4,4>(dO, dA, dB, hRef.data());
  go<10,2,3,3>(dO, dA, dB, hRef.data());
  go<10,2,3,4>(dO, dA, dB, hRef.data());
  go<10,2,4,3>(dO, dA, dB, hRef.data());
  go<10,2,4,4>(dO, dA, dB, hRef.data());
  go<10,3,3,3>(dO, dA, dB, hRef.data());
  go<10,3,3,4>(dO, dA, dB, hRef.data());
  go<10,3,4,3>(dO, dA, dB, hRef.data());
  go<10,3,4,4>(dO, dA, dB, hRef.data());
  return 0;
}
