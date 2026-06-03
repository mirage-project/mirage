// Qwen3-8B decode-phase linear projections at M=16 on B200
// tcgen05 cg1 persistent kernel with swapab trick for small M
// C[M, N] = A[M, K] @ W[N, K]^T  (+ residual for O, Down)
// Via swapab: C^T[N, M] = W[N, K] @ A[K, M]^T
//   MMA "A" operand = W[N, K] (large, row-major)
//   MMA "B" operand = A[M, K] (small, row-major, treated as A^T)
//   MMA "C" result  = C^T[N, M_padded]

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cublas_v2.h>

// ── Helpers ──

constexpr int WARP_SIZE = 32;

__host__ __device__ inline constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

template <typename T>
__device__ inline T warp_uniform(T x) { return __shfl_sync(0xFFFFFFFF, x, 0); }

__device__ inline uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
    "{\n\t.reg .pred %%px;\n\t"
    "elect.sync _|%%px, %1;\n\t"
    "@%%px mov.s32 %0, 1;\n\t}"
    : "+r"(pred) : "r"(0xFFFFFFFF));
  return pred;
}

__device__ inline void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ inline void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile(
    "{\n\t.reg .pred P1;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
    "@P1 bra.uni DONE;\n\t"
    "bra.uni LAB_WAIT;\n\t"
    "DONE:\n\t}"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks));
}

__device__ inline void mbarrier_arrive_expect_tx(int mbar_addr, int size) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
              :: "r"(mbar_addr), "r"(size) : "memory");
}

__device__ inline void mbarrier_arrive(int mbar_addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(mbar_addr) : "memory");
}

__device__ inline void tma_3d_load(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1 "
              "[%0], [%1, {%2, %3, %4}], [%5];"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr)
              : "memory");
}

__device__ inline void tma_3d_load_l2(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, uint64_t hint) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
              "[%0], [%1, {%2, %3, %4}], [%5], %6;"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "l"(hint)
              : "memory");
}

// L2 cache hint constants
constexpr uint64_t L2_EVICT_FIRST  = 0x12F0000000000000ULL;  // streaming, evict first
constexpr uint64_t L2_EVICT_LAST   = 0x14F0000000000000ULL;  // keep in L2, evict last
constexpr uint64_t L2_EVICT_NORMAL = 0x16F0000000000000ULL;  // normal eviction

__device__ inline void tcgen05_commit(int mbar_addr) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
              :: "r"(mbar_addr) : "memory");
}

__device__ inline void tcgen05_mma(int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t idesc, int enable_d) {
  asm volatile(
    "{\n\t.reg .pred p;\n\t"
    "setp.ne.b32 p, %4, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
    :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_d));
}

__device__ inline constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; }

inline void check_cu(CUresult err) {
  if (err == CUDA_SUCCESS) return;
  const char *msg;
  cuGetErrorString(err, &msg);
  fprintf(stderr, "CUDA driver error: %s\n", msg);
  exit(1);
}

inline void check_cuda(cudaError_t err) {
  if (err == cudaSuccess) return;
  fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
  exit(1);
}

// ── Kernel Constants ──

constexpr int NUM_WARPS = 6;  // 4 epilogue + 1 TMA + 1 MMA
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
constexpr int BLOCK_M = 128;   // covers N_real dimension (after swap)
constexpr int BLOCK_N = 16;    // covers M_real dimension (M_real=16, exact fit)
constexpr int BLOCK_K = 128;   // 2 z-slices of 64 each
constexpr int MMA_K = 16;
constexpr int NUM_STAGES = 6;

// SMEM descriptor template: SBO = 8*128, 128B swizzle
constexpr uint64_t SMEM_DESC = (desc_encode(8 * 128) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);

// MMA instruction descriptor: M=128, N=16, FP32 accum, BF16 inputs
constexpr uint32_t I_DESC = (1U << 4U)       // dtype=FP32
                          | (1U << 7U)        // atype=BF16
                          | (1U << 10U)       // btype=BF16
                          | ((uint32_t)BLOCK_N >> 3U << 17U)    // MMA_N
                          | ((uint32_t)BLOCK_M >> 4U << 24U);   // MMA_M

constexpr int W_SIZE = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);  // 32768
constexpr int A_SIZE = BLOCK_N * BLOCK_K * sizeof(nv_bfloat16);  // 4096
constexpr int AB_PER_STAGE = W_SIZE + A_SIZE;                     // 36864

// ── Main Kernel ──

// W_L2_HINT: 0=EVICT_FIRST (streaming), 1=EVICT_NORMAL (spatial L2 reuse)
template <bool HAS_RESIDUAL, int M_REAL = 16, int SPLIT_K = 1, int W_L2_HINT = 0>
__global__
__launch_bounds__(TB_SIZE, 1)
void swapab_cg1_kernel(
  const __grid_constant__ CUtensorMap W_tmap,   // "A" for MMA: W[N, K]
  const __grid_constant__ CUtensorMap A_tmap,    // "B" for MMA: A[M_pad, K]
  nv_bfloat16 *C_ptr,          // output C[M_real, N_real] row-major
  const nv_bfloat16 *res_ptr,  // residual (same layout as C), nullptr if !HAS_RESIDUAL
  int N_real, int K,
  int M_mma,                   // = N_real (after swap)
  float *workspace             // only used when SPLIT_K > 1
) {
  const int tid = threadIdx.x;
  const int bid = warp_uniform(blockIdx.x);
  const int num_bids = warp_uniform(gridDim.x);
  const int warp_id = warp_uniform(tid / WARP_SIZE);
  const int lane_id = tid % WARP_SIZE;

  const int grid_m = M_mma / BLOCK_M;

  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

  // Mbarrier layout in dynamic shared memory (after pipeline data)
  const int tma_mbar_addr = smem + AB_PER_STAGE * NUM_STAGES;
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
  const int epilogue_mbar_addr = mainloop_mbar_addr + 2 * 8;

  // Static shared for TMEM base address
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int tmem_addr_s;

  // Prefetch TMA descriptors
  if (warp_id == 0 && elect_sync()) {
    asm volatile("prefetch.tensormap [%0];" :: "l"(&W_tmap));
    asm volatile("prefetch.tensormap [%0];" :: "l"(&A_tmap));
  }

  // Initialize barriers (warp 4) and allocate TMEM (warp 5)
  if (warp_id == 4 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++) {
      mbarrier_init(tma_mbar_addr + i * 8, 1);   // 1 TMA thread arrives
      mbarrier_init(mma_mbar_addr + i * 8, 1);   // 1 MMA commit arrives
    }
    for (int i = 0; i < 2; i++) {
      mbarrier_init(mainloop_mbar_addr + i * 8, 1);              // 1 MMA commit
      mbarrier_init(epilogue_mbar_addr + i * 8, 4);  // 4 elected threads (one per epilogue warp)
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  else if (warp_id == 5) {
    // All threads in warp 5 participate in TMEM allocation
    const int addr = static_cast<int>(__cvta_generic_to_shared(&tmem_addr_s));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                :: "r"(addr), "r"(BLOCK_N * 2));  // 32 columns (double-buffered)
  }

  __syncthreads();
  const int taddr = tmem_addr_s;

  // Persistent scheduling
  const int num_spatial_tiles = grid_m;
  const int total_k_iters = K / BLOCK_K;
  const int iters_per_slice = total_k_iters / SPLIT_K;
  const int num_tiles = num_spatial_tiles * SPLIT_K;

  if (warp_id == 4) {
    // ── TMA Warp ──
    if (elect_sync()) {
      int tma_stage = 0;
      int mma_phase = 1;  // initial MMA mbar parity is 0; wait for !0 = succeeds immediately

      for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
        const int spatial_idx = this_bid % num_spatial_tiles;
        const int k_slice = this_bid / num_spatial_tiles;
        const int k_start = k_slice * iters_per_slice;
        const int off_m = spatial_idx * BLOCK_M;

        for (int i = 0; i < iters_per_slice; i++) {
          const int iter_k = k_start + i;

          // Wait for MMA to release this SMEM buffer
          mbarrier_wait(mma_mbar_addr + tma_stage * 8, mma_phase);

          const int mbar_addr = tma_mbar_addr + tma_stage * 8;
          const int W_smem = smem + tma_stage * AB_PER_STAGE;
          const int A_smem = W_smem + W_SIZE;
          const int z_coord = iter_k * (BLOCK_K / 64);

          // Load W and A with L2 cache hints
          constexpr uint64_t W_HINT = (W_L2_HINT == 0) ? L2_EVICT_FIRST : L2_EVICT_NORMAL;
          tma_3d_load_l2(W_smem, &W_tmap, 0, off_m, z_coord, mbar_addr, W_HINT);
          tma_3d_load_l2(A_smem, &A_tmap, 0, 0, z_coord, mbar_addr, L2_EVICT_LAST);
          mbarrier_arrive_expect_tx(mbar_addr, W_SIZE + A_SIZE);

          tma_stage = (tma_stage + 1) % NUM_STAGES;
          if (tma_stage == 0) mma_phase ^= 1;
        }
      }
    }
  }
  else if (warp_id == 5) {
    // ── MMA Warp ──
    if (elect_sync()) {
      int tma_stage = 0;
      int tma_phase = 0;
      int mainloop_stage = 0;
      int epilogue_phase = 1;  // initial epilogue mbar parity is 0; wait for !0 = succeeds

      for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
        // Wait for epilogue to release TMEM buffer
        mbarrier_wait(epilogue_mbar_addr + mainloop_stage * 8, epilogue_phase);

        for (int i = 0; i < iters_per_slice; i++) {
          const int W_smem = smem + tma_stage * AB_PER_STAGE;
          const int A_smem = W_smem + W_SIZE;
          const int tmem = taddr + mainloop_stage * BLOCK_N;

          uint64_t a_desc = SMEM_DESC | (W_smem >> 4);
          uint64_t b_desc = SMEM_DESC | (A_smem >> 4);

          // Wait for TMA to fill this SMEM buffer
          mbarrier_wait(tma_mbar_addr + tma_stage * 8, tma_phase);
          asm volatile("tcgen05.fence::after_thread_sync;");

          // First z-slice: 4 MMA steps of MMA_K=16 each
          tcgen05_mma(tmem, a_desc, b_desc, I_DESC, i);  // i==0 → zero init
          for (int k2 = 1; k2 < 64 / MMA_K; k2++) {
            a_desc += (32 >> 4);
            b_desc += (32 >> 4);
            tcgen05_mma(tmem, a_desc, b_desc, I_DESC, 1);
          }

          // Second z-slice (k1=1)
          for (int k1 = 1; k1 < BLOCK_K / 64; k1++) {
            uint64_t a2 = SMEM_DESC | ((W_smem + k1 * BLOCK_M * 128) >> 4);
            uint64_t b2 = SMEM_DESC | ((A_smem + k1 * BLOCK_N * 128) >> 4);
            for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
              tcgen05_mma(tmem, a2, b2, I_DESC, 1);
              a2 += (32 >> 4);
              b2 += (32 >> 4);
            }
          }

          // Signal MMA done with this SMEM buffer
          tcgen05_commit(mma_mbar_addr + tma_stage * 8);

          tma_stage = (tma_stage + 1) % NUM_STAGES;
          if (tma_stage == 0) tma_phase ^= 1;
        }

        // Signal mainloop done → epilogue can read TMEM
        tcgen05_commit(mainloop_mbar_addr + mainloop_stage * 8);
        mainloop_stage = (mainloop_stage + 1) % 2;
        if (mainloop_stage == 0) epilogue_phase ^= 1;
      }
    }
  }
  else {
    // ── Epilogue Warps (0-3) ──
    int mainloop_stage = 0;
    int mainloop_phase = 0;

    for (int this_bid = bid; this_bid < num_tiles; this_bid += num_bids) {
      const int spatial_idx = this_bid % num_spatial_tiles;
      const int k_slice = this_bid / num_spatial_tiles;
      const int bid_m = spatial_idx;

      // Wait for MMA to complete this tile
      mbarrier_wait(mainloop_mbar_addr + mainloop_stage * 8, mainloop_phase);
      asm volatile("tcgen05.fence::after_thread_sync;");

      const int n_real = bid_m * BLOCK_M + warp_id * 32 + lane_id;

      if (n_real < N_real) {
        // Read 16 fp32 columns from TMEM (one per M_real value)
        const int t_row = warp_id * 32;
        const int t_col = taddr + mainloop_stage * BLOCK_N;
        const int t_addr = (t_row << 16) + t_col;

        float f[16];
        asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x16.b32\n"
          "  {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
          : "=f"(f[0]), "=f"(f[1]), "=f"(f[2]), "=f"(f[3]),
            "=f"(f[4]), "=f"(f[5]), "=f"(f[6]), "=f"(f[7]),
            "=f"(f[8]), "=f"(f[9]), "=f"(f[10]), "=f"(f[11]),
            "=f"(f[12]), "=f"(f[13]), "=f"(f[14]), "=f"(f[15])
          : "r"(t_addr));
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        if constexpr (SPLIT_K == 1) {
          // Direct output: convert to bf16 first, then add residual (matches cuBLAS + separate add)
          if constexpr (HAS_RESIDUAL) {
            #pragma unroll
            for (int m = 0; m < M_REAL; m++) {
              // Round GEMM to bf16 first (matches cuBLAS output precision)
              nv_bfloat16 gemm_bf16 = __float2bfloat16(f[m]);
              f[m] = __bfloat162float(gemm_bf16) + __bfloat162float(res_ptr[m * N_real + n_real]);
            }
          }
          #pragma unroll
          for (int m = 0; m < M_REAL; m++) {
            nv_bfloat16 val = __float2bfloat16(f[m]);
            asm volatile("st.global.L1::no_allocate.b16 [%0], %1;"
              :: "l"(C_ptr + m * N_real + n_real), "h"(*(uint16_t*)&val) : "memory");
          }
        } else {
          // Split-K: write fp32 partial sums to workspace
          float *ws_base = workspace + k_slice * M_REAL * N_real;
          #pragma unroll
          for (int m = 0; m < M_REAL; m++) {
            asm volatile("st.global.L1::no_allocate.b32 [%0], %1;"
              :: "l"(ws_base + m * N_real + n_real), "f"(f[m]) : "memory");
          }
        }
      }

      // Signal epilogue completion (elected thread only - tcgen05.ld.sync ensures all done)
      if (elect_sync()) {
        mbarrier_arrive(epilogue_mbar_addr + mainloop_stage * 8);
      }

      mainloop_stage = (mainloop_stage + 1) % 2;
      if (mainloop_stage == 0) mainloop_phase ^= 1;
    }
  }

  // All warps sync before TMEM dealloc
  __syncthreads();
  if (warp_id == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                :: "r"(taddr), "r"(BLOCK_N * 2));
  }
}

// ── Split-K Reduction Kernel ──

template <int M_REAL, bool HAS_RESIDUAL, int SPLIT_K>
__global__ void splitk_reduce_kernel(
  float *workspace,           // float[SPLIT_K * M_REAL * N_real]
  nv_bfloat16 *C_ptr,        // output [M_REAL, N_real]
  const nv_bfloat16 *res_ptr, // residual or nullptr
  int N_real
) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = M_REAL * N_real;
  if (idx >= total) return;

  float sum = 0.0f;
  #pragma unroll
  for (int s = 0; s < SPLIT_K; s++) {
    sum += workspace[s * total + idx];
  }

  if constexpr (HAS_RESIDUAL) {
    sum += __bfloat162float(res_ptr[idx]);
  }

  C_ptr[idx] = __float2bfloat16(sum);
}

// ── Host Launch ──

template <bool HAS_RESIDUAL, int SPLIT_K = 1, int W_L2_HINT = 0>
void launch_kernel(
  const nv_bfloat16 *W_ptr,   // weight [N_real, K] row-major
  const nv_bfloat16 *A_ptr,   // input [M_real, K] row-major (padded to BLOCK_N rows)
  nv_bfloat16 *C_ptr,         // output [M_real, N_real] row-major
  const nv_bfloat16 *res_ptr, // residual or nullptr
  int M_real, int N_real, int K,
  float *workspace = nullptr
) {
  const int M_mma = N_real;  // after swap

  CUtensorMap W_tmap, A_tmap;

  auto init_tmap = [&](CUtensorMap *tmap, const nv_bfloat16 *ptr,
                       uint64_t global_height, uint32_t shared_height, bool l2_promote = false) {
    constexpr uint32_t rank = 3;
    uint64_t globalDim[rank]       = {64, global_height, (uint64_t)K / 64};
    uint64_t globalStrides[rank-1] = {(uint64_t)K * sizeof(nv_bfloat16), 128};
    uint32_t boxDim[rank]          = {64, shared_height, (uint32_t)BLOCK_K / 64};
    uint32_t elementStrides[rank]  = {1, 1, 1};
    check_cu(cuTensorMapEncodeTiled(
      tmap, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, (void *)ptr,
      globalDim, globalStrides, boxDim, elementStrides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      l2_promote ? CU_TENSOR_MAP_L2_PROMOTION_L2_128B : CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  };

  // Promote W to L2 when it fits (< 96 MB L2 cache)
  bool promote_w = ((size_t)N_real * K * 2 <= 96ULL * 1024 * 1024);
  init_tmap(&W_tmap, W_ptr, N_real, BLOCK_M, promote_w);
  init_tmap(&A_tmap, A_ptr, BLOCK_N, BLOCK_N, true);  // always promote A (tiny)

  const int num_spatial_tiles = M_mma / BLOCK_M;
  const int num_tiles = num_spatial_tiles * SPLIT_K;
  int grid = std::min(148, num_tiles);

  constexpr int mbar_size = (NUM_STAGES * 2 + 4) * 8 + 16;  // extra padding
  int smem_size = AB_PER_STAGE * NUM_STAGES + mbar_size;

  auto kern = swapab_cg1_kernel<HAS_RESIDUAL, 16, SPLIT_K, W_L2_HINT>;
  check_cuda(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

  kern<<<grid, TB_SIZE, smem_size>>>(W_tmap, A_tmap, C_ptr, res_ptr,
                                      N_real, K, M_mma, workspace);
  check_cuda(cudaGetLastError());

  // Split-K reduction
  if constexpr (SPLIT_K > 1) {
    const int total_elems = 16 * N_real;
    const int reduce_threads = 256;
    const int reduce_blocks = cdiv(total_elems, reduce_threads);
    splitk_reduce_kernel<16, HAS_RESIDUAL, SPLIT_K>
      <<<reduce_blocks, reduce_threads>>>(workspace, C_ptr, res_ptr, N_real);
    check_cuda(cudaGetLastError());
  }
}

// ── Benchmark Harness ──

struct GemmConfig {
  const char *name;
  int M, N, K;
  bool has_residual;
};

int main() {
  check_cuda(cudaSetDevice(0));

  GemmConfig configs[] = {
    {"QKV",    16, 6144,   4096,  false},
    {"O",      16, 4096,   4096,  true},
    {"GateUp", 16, 24576,  4096,  false},
    {"Down",   16, 4096,   12288, true},
    {"LMHead", 16, 153600, 4096,  false},
  };

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);

  // L2 flush buffer (128 MB > 96 MB L2)
  size_t flush_sz = 128 * 1024 * 1024;
  char *flush_buf;
  check_cuda(cudaMalloc(&flush_buf, flush_sz));

  // Split-K config
  constexpr int QKV_SPLIT_K = 1;     // 48 tiles, no split-K overhead
  constexpr int O_SPLIT_K = 8;       // 32*8=256 tiles, more parallelism
  constexpr int GATEUP_SPLIT_K = 1;  // 192 tiles already
  constexpr int DOWN_SPLIT_K = 4;    // 32*4=128 tiles, residual fused in reduction
  constexpr int LMHEAD_SPLIT_K = 1;  // 1200 tiles already

  // Workspace for split-K
  size_t max_ws_size = std::max({
    (size_t)QKV_SPLIT_K * 16 * 6144,
    (size_t)O_SPLIT_K * 16 * 4096,
    (size_t)DOWN_SPLIT_K * 16 * 4096
  }) * sizeof(float);
  float *splitk_workspace;
  check_cuda(cudaMalloc(&splitk_workspace, max_ws_size));

  printf("KERNEL_RESULT {");
  bool first_kr = true;

  double ref_tflops[5];
  double ker_tflops[5];

  for (int ci = 0; ci < 5; ci++) {
    auto &cfg = configs[ci];
    int M = cfg.M, N = cfg.N, K = cfg.K;
    double flops = 2.0 * M * N * K;

    // Allocate
    nv_bfloat16 *dA, *dW, *dC, *dC_ref, *dRes;
    size_t padM = BLOCK_N;  // 16

    check_cuda(cudaMalloc(&dA, (size_t)padM * K * sizeof(nv_bfloat16)));
    check_cuda(cudaMemset(dA, 0, (size_t)padM * K * sizeof(nv_bfloat16)));
    check_cuda(cudaMalloc(&dW, (size_t)N * K * sizeof(nv_bfloat16)));
    check_cuda(cudaMalloc(&dC, (size_t)M * N * sizeof(nv_bfloat16)));
    check_cuda(cudaMalloc(&dC_ref, (size_t)M * N * sizeof(nv_bfloat16)));

    if (cfg.has_residual)
      check_cuda(cudaMalloc(&dRes, (size_t)M * N * sizeof(nv_bfloat16)));
    else
      dRes = nullptr;

    // Initialize random data
    {
      size_t nA = (size_t)M * K;
      size_t nW = (size_t)N * K;
      size_t nC = (size_t)M * N;
      nv_bfloat16 *hA = new nv_bfloat16[nA];
      nv_bfloat16 *hW = new nv_bfloat16[nW];
      srand(42 + ci);
      for (size_t i = 0; i < nA; i++) hA[i] = __float2bfloat16((rand() % 201 - 100) / 100.0f);
      for (size_t i = 0; i < nW; i++) hW[i] = __float2bfloat16((rand() % 201 - 100) / 100.0f);
      check_cuda(cudaMemcpy(dA, hA, nA * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
      check_cuda(cudaMemcpy(dW, hW, nW * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
      if (cfg.has_residual) {
        nv_bfloat16 *hRes = new nv_bfloat16[nC];
        for (size_t i = 0; i < nC; i++) hRes[i] = __float2bfloat16((rand() % 201 - 100) / 100.0f);
        check_cuda(cudaMemcpy(dRes, hRes, nC * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
        delete[] hRes;
      }
      delete[] hA;
      delete[] hW;
    }

    // ── cuBLAS reference ──
    {
      float alpha = 1.0f, beta = 0.0f;
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                   N, M, K,
                   &alpha,
                   dW, CUDA_R_16BF, K,
                   dA, CUDA_R_16BF, K,
                   &beta,
                   dC_ref, CUDA_R_16BF, N,
                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
      check_cuda(cudaDeviceSynchronize());

      for (int i = 0; i < 20; i++) {
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                     &alpha, dW, CUDA_R_16BF, K, dA, CUDA_R_16BF, K,
                     &beta, dC_ref, CUDA_R_16BF, N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
      }
      check_cuda(cudaDeviceSynchronize());

      cudaEvent_t ev0, ev1;
      cudaEventCreate(&ev0);
      cudaEventCreate(&ev1);
      float times[100];
      for (int i = 0; i < 100; i++) {
        check_cuda(cudaMemset(flush_buf, 0, flush_sz));
        check_cuda(cudaDeviceSynchronize());
        cudaEventRecord(ev0);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                     &alpha, dW, CUDA_R_16BF, K, dA, CUDA_R_16BF, K,
                     &beta, dC_ref, CUDA_R_16BF, N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&times[i], ev0, ev1);
      }
      std::sort(times, times + 100);
      ref_tflops[ci] = flops / (times[50] * 1e-3) / 1e12;
      cudaEventDestroy(ev0);
      cudaEventDestroy(ev1);
    }

    // ── Our kernel ──
    check_cuda(cudaMemset(dC, 0, (size_t)M * N * sizeof(nv_bfloat16)));

    auto run_kernel = [&]() {
      if (ci == 0) {
        launch_kernel<false, QKV_SPLIT_K>(dW, dA, dC, nullptr, M, N, K, splitk_workspace);
      } else if (ci == 1) {
        launch_kernel<true, O_SPLIT_K>(dW, dA, dC, dRes, M, N, K, splitk_workspace);
      } else if (ci == 2) {
        launch_kernel<false, GATEUP_SPLIT_K>(dW, dA, dC, nullptr, M, N, K);
      } else if (ci == 3) {
        launch_kernel<true, DOWN_SPLIT_K>(dW, dA, dC, dRes, M, N, K, splitk_workspace);
      } else {
        launch_kernel<false, LMHEAD_SPLIT_K, 1>(dW, dA, dC, nullptr, M, N, K);  // EVICT_NORMAL for LMHead
      }
    };

    run_kernel();
    check_cuda(cudaDeviceSynchronize());

    // Verify
    {
      size_t nC = (size_t)M * N;
      nv_bfloat16 *hC = new nv_bfloat16[nC];
      nv_bfloat16 *hRef = new nv_bfloat16[nC];
      nv_bfloat16 *hRes = nullptr;
      check_cuda(cudaMemcpy(hC, dC, nC * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
      check_cuda(cudaMemcpy(hRef, dC_ref, nC * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
      if (cfg.has_residual) {
        hRes = new nv_bfloat16[nC];
        check_cuda(cudaMemcpy(hRes, dRes, nC * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
      }

      float max_err = 0;
      int err_count = 0;
      for (size_t i = 0; i < nC; i++) {
        float got = __bfloat162float(hC[i]);
        float ref_val = __bfloat162float(hRef[i]);
        if (cfg.has_residual) {
          ref_val += __bfloat162float(hRes[i]);
          // Round to bf16 for fair comparison (our kernel fuses residual with fp32 GEMM)
          ref_val = __bfloat162float(__float2bfloat16(ref_val));
        }
        float e = fabsf(got - ref_val) / fmaxf(fmaxf(fabsf(ref_val), fabsf(got)), 1.0f);
        if (e > max_err) max_err = e;
        if (e > 5e-3f) err_count++;
      }
      fprintf(stderr, "%s: max_err=%.6f errs=%d/%zu\n", cfg.name, max_err, err_count, nC);
      delete[] hC;
      delete[] hRef;
      if (hRes) delete[] hRes;
    }

    // Benchmark
    for (int i = 0; i < 20; i++) run_kernel();
    check_cuda(cudaDeviceSynchronize());

    {
      cudaEvent_t ev0, ev1;
      cudaEventCreate(&ev0);
      cudaEventCreate(&ev1);
      float times[100];
      for (int i = 0; i < 100; i++) {
        check_cuda(cudaMemset(flush_buf, 0, flush_sz));
        check_cuda(cudaDeviceSynchronize());
        cudaEventRecord(ev0);
        run_kernel();
        cudaEventRecord(ev1);
        cudaEventSynchronize(ev1);
        cudaEventElapsedTime(&times[i], ev0, ev1);
      }
      std::sort(times, times + 100);
      ker_tflops[ci] = flops / (times[50] * 1e-3) / 1e12;
      cudaEventDestroy(ev0);
      cudaEventDestroy(ev1);
    }

    if (!first_kr) printf(", ");
    printf("\"%s\": %.2f", cfg.name, ker_tflops[ci]);
    first_kr = false;

    cudaFree(dA); cudaFree(dW); cudaFree(dC); cudaFree(dC_ref);
    if (dRes) cudaFree(dRes);
  }
  printf("}\n");

  printf("KERNEL_RESULT_REFERENCE {");
  for (int ci = 0; ci < 5; ci++) {
    if (ci > 0) printf(", ");
    printf("\"%s\": %.2f", configs[ci].name, ref_tflops[ci]);
  }
  printf("}\n");

  cudaFree(flush_buf);
  cudaFree(splitk_workspace);
  cublasDestroy(handle);
  return 0;
}
