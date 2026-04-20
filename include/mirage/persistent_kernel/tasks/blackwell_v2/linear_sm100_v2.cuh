// Qwen3-8B decode-phase linear for v2 runtime
// Adapted from tests/runtime_python/blackwell/linear_c.cu (CTA_GROUP=1 swapab)
//
// Changes from source:
//   - __global__ → __device__ __noinline__
//   - blockIdx.x, gridDim.x → bid, num_bids parameters
//   - __syncthreads() → bar.sync 1, TB_SIZE
//   - Thread guard for v2's 8-compute-warp layout (only 6 warps used)

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace kernel {
namespace linear_v2 {


// ── Kernel Constants ──
constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 6;                     // 4 epilogue + 1 TMA + 1 MMA
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;   // 192 threads
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 16;
constexpr int BLOCK_K = 128;
constexpr int MMA_K = 16;
constexpr int NUM_STAGES = 6;

constexpr int W_SIZE = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);  // 32768
constexpr int A_SIZE = BLOCK_N * BLOCK_K * sizeof(nv_bfloat16);  // 4096
constexpr int AB_PER_STAGE = W_SIZE + A_SIZE;                    // 36864

// SMEM layout: [pipeline data | mbarriers]
// mbar region: NUM_STAGES*2 tma+mma mbars + 2 mainloop + 2 epilogue = 16*8 = 128 bytes + padding
constexpr int MBAR_SIZE = (NUM_STAGES * 2 + 4) * 8 + 16;
constexpr int TOTAL_SMEM = AB_PER_STAGE * NUM_STAGES + MBAR_SIZE;

// ── Helpers ──
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
    asm volatile(
        "{\n\t.reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, 0x989680;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t}"
        :: "r"(mbar_addr), "r"(phase));
}

__device__ inline void mbarrier_arrive_expect_tx(int mbar_addr, int size) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(mbar_addr), "r"(size) : "memory");
}

__device__ inline void mbarrier_arrive(int mbar_addr) {
    asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(mbar_addr) : "memory");
}

__device__ inline void tma_3d_load_l2(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, uint64_t hint) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
                 "[%0], [%1, {%2, %3, %4}], [%5], %6;"
                 :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "l"(hint)
                 : "memory");
}

constexpr uint64_t L2_EVICT_FIRST  = 0x12F0000000000000ULL;
constexpr uint64_t L2_EVICT_LAST   = 0x14F0000000000000ULL;
constexpr uint64_t L2_EVICT_NORMAL = 0x16F0000000000000ULL;

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

constexpr uint64_t SMEM_DESC = (desc_encode(8 * 128) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);

constexpr uint32_t I_DESC = (1U << 4U)
                          | (1U << 7U)
                          | (1U << 10U)
                          | ((uint32_t)BLOCK_N >> 3U << 17U)
                          | ((uint32_t)BLOCK_M >> 4U << 24U);

// Named barrier 1 for TB_SIZE threads (replaces __syncthreads)
// Barrier 0 is implicit __syncthreads (full blockDim).
__device__ inline void task_barrier() {
    asm volatile("bar.sync 1, %0;" :: "n"(TB_SIZE));
}

// ── Main Task Function ──
// One TaskDesc = one output tile. tile_idx encodes both spatial_idx and k_slice:
//   spatial_idx = tile_idx % num_spatial_tiles
//   k_slice     = tile_idx / num_spatial_tiles
// Called once per tile by v2 runtime (dispatch reads tile_idx from task_metadata).
// NOTE: pass CUtensorMap BY POINTER so runtime dispatch can forward GMEM
// tensormap pointers directly. cp.async.bulk.tensor requires the descriptor
// in .const / .param / .global — not stack. Passing by const ref lets the
// compiler copy a 64B CUtensorMap to stack, which fails at runtime.
template <bool HAS_RESIDUAL, int M_REAL = 16, int SPLIT_K = 1, int W_L2_HINT = 0,
          int TILES_PER_TASK = 1>
__device__ __noinline__ void linear_task(
    CUtensorMap const *W_tmap_ptr,    // W[N, K] TMA descriptor (GMEM)
    CUtensorMap const *A_tmap_ptr,    // A[M, K] TMA descriptor (GMEM or const)
    nv_bfloat16 *C_ptr,               // output C[M_real, N_real]
    nv_bfloat16 const *res_ptr,       // residual, or nullptr
    int N_real, int K,
    int tile_idx,                     // single tile index (v1 convention)
    float *workspace                  // split-K partial sums, or nullptr
) {
    // v2 compute warp guard: v2 has 8 compute warps but linear uses only 6.
    // TODO(gap #2, v2_gap_analysis.md): per-warp register reallocation.
    // ptxas ignores setmaxnreg unless it is the FIRST instruction of a
    // __noinline__ function (warning C7508: "unable to determine register
    // count at entry"). To get 232 regs for the epilogue without spills, this
    // function must split into linear_epilogue_task / linear_tma_task /
    // linear_mma_task __noinline__ callees, each beginning with its own
    // setmaxnreg. Current version stays at the default 128 regs → epilogue
    // may spill, but correctness is unaffected.
    if (threadIdx.x >= TB_SIZE) return;
    const int tid = threadIdx.x;
    const int warp_id = warp_uniform(tid / WARP_SIZE);
    const int lane_id = tid % WARP_SIZE;

    const int M_mma = N_real;  // after swap
    const int grid_m = M_mma / BLOCK_M;

    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

    const int tma_mbar_addr = smem + AB_PER_STAGE * NUM_STAGES;
    const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
    const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;
    const int epilogue_mbar_addr = mainloop_mbar_addr + 2 * 8;

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ int tmem_addr_s;

    // Prefetch TMA descriptors
    if (warp_id == 0 && elect_sync()) {
        asm volatile("prefetch.tensormap [%0];" :: "l"(W_tmap_ptr));
        asm volatile("prefetch.tensormap [%0];" :: "l"(A_tmap_ptr));
    }

    // Init barriers (warp 4) and allocate TMEM (warp 5)
    if (warp_id == 4 && elect_sync()) {
        for (int i = 0; i < NUM_STAGES; i++) {
            mbarrier_init(tma_mbar_addr + i * 8, 1);
            mbarrier_init(mma_mbar_addr + i * 8, 1);
        }
        for (int i = 0; i < 2; i++) {
            mbarrier_init(mainloop_mbar_addr + i * 8, 1);
            mbarrier_init(epilogue_mbar_addr + i * 8, 4 * WARP_SIZE);
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    else if (warp_id == 5) {
        const int addr = static_cast<int>(__cvta_generic_to_shared(&tmem_addr_s));
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(addr), "r"(BLOCK_N * 2));
    }

    task_barrier();  // was __syncthreads()
    const int taddr = tmem_addr_s;

    const int num_spatial_tiles = grid_m;
    const int total_k_iters = K / BLOCK_K;
    const int iters_per_slice = total_k_iters / SPLIT_K;
    const int num_tiles = num_spatial_tiles * SPLIT_K;

    // Bounds check: dispatch could pass invalid tile_idx (base).
    // With TILES_PER_TASK, this task processes tiles [tile_idx, tile_idx+TILES_PER_TASK).
    // Compute actual count here (clamp to num_tiles).
    if (tile_idx >= num_tiles) {
        task_barrier();
        if (warp_id == 0 && elect_sync()) {
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                         :: "r"(taddr), "r"(BLOCK_N * 2));
        }
        return;
    }
    const int tiles_left = num_tiles - tile_idx;
    const int tiles_to_process = (TILES_PER_TASK < tiles_left) ? TILES_PER_TASK : tiles_left;

    if (warp_id == 4) {
        // ── TMA Warp ── persistent stage/phase across all tiles in this task
        if (elect_sync()) {
            int tma_stage = 0;
            int mma_phase = 1;

            for (int t = 0; t < tiles_to_process; t++) {
                const int cur_tile_idx = tile_idx + t;
                const int cur_spatial_idx = cur_tile_idx % num_spatial_tiles;
                const int cur_k_slice = cur_tile_idx / num_spatial_tiles;
                const int cur_k_start = cur_k_slice * iters_per_slice;
                const int cur_off_m = cur_spatial_idx * BLOCK_M;

                for (int i = 0; i < iters_per_slice; i++) {
                    const int iter_k = cur_k_start + i;

                    mbarrier_wait(mma_mbar_addr + tma_stage * 8, mma_phase);

                    const int mbar_addr = tma_mbar_addr + tma_stage * 8;
                    const int W_smem = smem + tma_stage * AB_PER_STAGE;
                    const int A_smem = W_smem + W_SIZE;
                    const int z_coord = iter_k * (BLOCK_K / 64);

                    constexpr uint64_t W_HINT = (W_L2_HINT == 0) ? L2_EVICT_FIRST : L2_EVICT_NORMAL;
                    tma_3d_load_l2(W_smem, W_tmap_ptr, 0, cur_off_m, z_coord, mbar_addr, W_HINT);
                    tma_3d_load_l2(A_smem, A_tmap_ptr, 0, 0, z_coord, mbar_addr, L2_EVICT_LAST);
                    mbarrier_arrive_expect_tx(mbar_addr, W_SIZE + A_SIZE);

                    tma_stage = (tma_stage + 1) % NUM_STAGES;
                    if (tma_stage == 0) mma_phase ^= 1;
                }
            }
        }
    }
    else if (warp_id == 5) {
        // ── MMA Warp ── persistent tma_stage/tma_phase; epilogue_phase flips per tile
        if (elect_sync()) {
            int tma_stage = 0;
            int tma_phase = 0;
            const int mainloop_stage = 0;
            int epilogue_phase = 1;

            for (int t = 0; t < tiles_to_process; t++) {
                mbarrier_wait(epilogue_mbar_addr + mainloop_stage * 8, epilogue_phase);

                for (int i = 0; i < iters_per_slice; i++) {
                    const int W_smem = smem + tma_stage * AB_PER_STAGE;
                    const int A_smem = W_smem + W_SIZE;
                    const int tmem = taddr + mainloop_stage * BLOCK_N;

                    uint64_t a_desc = SMEM_DESC | (W_smem >> 4);
                    uint64_t b_desc = SMEM_DESC | (A_smem >> 4);

                    mbarrier_wait(tma_mbar_addr + tma_stage * 8, tma_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");

                    // First z-slice: 4 MMA steps of MMA_K=16. i=0 resets accumulator per tile.
                    tcgen05_mma(tmem, a_desc, b_desc, I_DESC, i);
                    for (int k2 = 1; k2 < 64 / MMA_K; k2++) {
                        a_desc += (32 >> 4);
                        b_desc += (32 >> 4);
                        tcgen05_mma(tmem, a_desc, b_desc, I_DESC, 1);
                    }

                    for (int k1 = 1; k1 < BLOCK_K / 64; k1++) {
                        uint64_t a2 = SMEM_DESC | ((W_smem + k1 * BLOCK_M * 128) >> 4);
                        uint64_t b2 = SMEM_DESC | ((A_smem + k1 * BLOCK_N * 128) >> 4);
                        for (int k2 = 0; k2 < 64 / MMA_K; k2++) {
                            tcgen05_mma(tmem, a2, b2, I_DESC, 1);
                            a2 += (32 >> 4);
                            b2 += (32 >> 4);
                        }
                    }

                    tcgen05_commit(mma_mbar_addr + tma_stage * 8);

                    tma_stage = (tma_stage + 1) % NUM_STAGES;
                    if (tma_stage == 0) tma_phase ^= 1;
                }

                tcgen05_commit(mainloop_mbar_addr + mainloop_stage * 8);
                epilogue_phase ^= 1;  // flip for next tile
            }
        }
    }
    else {
        // ── Epilogue Warps (0-3) ── mainloop_phase flips per tile
        const int mainloop_stage = 0;
        int mainloop_phase = 0;

        for (int t = 0; t < tiles_to_process; t++) {
            const int cur_tile_idx = tile_idx + t;
            const int cur_spatial_idx = cur_tile_idx % num_spatial_tiles;
            const int cur_k_slice = cur_tile_idx / num_spatial_tiles;
            const int bid_m = cur_spatial_idx;

            mbarrier_wait(mainloop_mbar_addr + mainloop_stage * 8, mainloop_phase);
            asm volatile("tcgen05.fence::after_thread_sync;");

            const int n_real = bid_m * BLOCK_M + warp_id * 32 + lane_id;

            if (n_real < N_real) {
                const int t_col = taddr + mainloop_stage * BLOCK_N;
                const int t_addr = (warp_id * 32 << 16) + t_col;

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
                    if constexpr (HAS_RESIDUAL) {
                        #pragma unroll
                        for (int m = 0; m < M_REAL; m++) {
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
                    float *ws_base = workspace + cur_k_slice * M_REAL * N_real;
                    #pragma unroll
                    for (int m = 0; m < M_REAL; m++) {
                        asm volatile("st.global.L1::no_allocate.b32 [%0], %1;"
                            :: "l"(ws_base + m * N_real + n_real), "f"(f[m]) : "memory");
                    }
                }
            }

            mbarrier_arrive(epilogue_mbar_addr + mainloop_stage * 8);
            mainloop_phase ^= 1;  // flip for next tile
        }
    }

    task_barrier();  // was __syncthreads()
    if (warp_id == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(taddr), "r"(BLOCK_N * 2));
    }
}

// Split-K reduction task (used when SPLIT_K > 1)
// Uses the v2 compute pool (NUM_COMPUTE_WARPS*32=256 threads), NOT blockDim.x.
// Non-compute warps skip this function via the dispatch path.
template <int M_REAL, bool HAS_RESIDUAL, int SPLIT_K>
__device__ __forceinline__ void splitk_reduce_task(
    float *workspace,
    nv_bfloat16 *C_ptr,
    nv_bfloat16 const *res_ptr,
    int N_real,
    int bid, int num_bids
) {
    constexpr int EFF_THREADS = 8 * 32;  // NUM_COMPUTE_WARPS * 32
    if (threadIdx.x >= EFF_THREADS) return;
    const int total = M_REAL * N_real;
    for (int idx = bid * EFF_THREADS + threadIdx.x;
         idx < total;
         idx += num_bids * EFF_THREADS) {
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
}

} // namespace linear_v2
} // namespace kernel
