// Qwen3-8B decode-phase linear for v2 runtime
// Adapted from tests/runtime_python/blackwell/linear_c.cu (CTA_GROUP=1 swapab)
//
// Changes from source:
//   - __global__ → __device__ __noinline__
//   - blockIdx.x, gridDim.x → bid, num_bids parameters
//   - Phase 3.4: split into role-specific functions (loader/launcher/consumer);
//     cross-warp sync is via op-private mbarriers in dynamic_semaphores
//     (no bar.sync), and SMEM is addressed through planner per-stage regions.

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

#include "mirage/persistent_kernel/runtime_header.h"
#include "mirage/persistent_kernel/tasks/blackwell_v2/linear_sm100_v2_spec.h"

namespace kernel {
namespace linear_v2 {


// ── Kernel Constants ──
constexpr int WARP_SIZE = 32;
constexpr int BLOCK_M = 128;
constexpr int BLOCK_N = 16;
constexpr int BLOCK_K = 128;
constexpr int MMA_K = 16;
constexpr int NUM_STAGES = 6;

constexpr int W_SIZE = BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);  // 32768
constexpr int A_SIZE = BLOCK_N * BLOCK_K * sizeof(nv_bfloat16);  // 4096

// SMEM layout (Phase 3.2): one region per pipeline stage holding W (32 KB)
// and A (4 KB) in separate planner regions, plus a tiny scratch region for
// cross-role publishing of the TMEM address. Region bases are fetched at
// runtime via task_desc->smem_region_offset(REGION_*). The 24 mbarriers
// live in RuntimeSMEM::dynamic_semaphores at SEM_OP_BASE.. — addressed via
// the `dyn_sem_base` arg the runtime passes in.
//
// Phase 4.1: tma_mbar is split into per-stage W and A mbarriers so weight
// TMAs can complete independently of activation TMAs (which require the
// cross-SM dependency wait).
//
// Per-task SEM ordinals (relative to dyn_sem_base):
//   [+0  ..+5 ]  W_tma_mbar      (count=1,  loader→launcher, W only)
//   [+6  ..+11]  A_tma_mbar      (count=1,  loader→launcher, A only)
//   [+12 ..+17]  mma_mbar        (count=1,  launcher→loader, "stage K MMA done")
//   [+18 ..+19]  mainloop_mbar   (count=1,  launcher→consumer)
//   [+20 ..+21]  epilogue_mbar   (count=4*WARP_SIZE, consumer→launcher)
//   [+22      ]  tmem_ready      (count=1,  launcher→consumer)
//   [+23      ]  consumer_done   (count=4*WARP_SIZE, consumer→launcher)
constexpr int SEM_W_TMA_BASE     = 0;
constexpr int SEM_A_TMA_BASE     = NUM_STAGES;          // 6
constexpr int SEM_MMA_BASE       = 2 * NUM_STAGES;      // 12
constexpr int SEM_MAINLOOP_BASE  = 3 * NUM_STAGES;      // 18
constexpr int SEM_EPILOGUE_BASE  = 3 * NUM_STAGES + 2;  // 20
constexpr int SEM_TMEM_READY     = 3 * NUM_STAGES + 4;  // 22
constexpr int SEM_CONSUMER_DONE  = 3 * NUM_STAGES + 5;  // 23
constexpr int NUM_OP_SEMS        = 3 * NUM_STAGES + 6;  // 24

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

// ── Role-Split Task Functions (Phase 3.4) ──
// linear_task is split into three __noinline__ functions, one per role warp.
// All three are dispatched independently by the v2 runtime; cross-role sync
// is via the 18 op-private mbarriers in dynamic_semaphores[slot][SEM_OP_BASE..]
// (init'd by the controller's init_semaphores body). The math is unchanged
// from the pre-split linear_task — only the warp-branching glue went away.
//
// One TaskDesc = one output tile. tile_idx encodes both spatial_idx and k_slice:
//   spatial_idx = tile_idx % num_spatial_tiles
//   k_slice     = tile_idx / num_spatial_tiles
//
// NOTE: pass CUtensorMap BY POINTER so runtime dispatch can forward GMEM
// tensormap pointers directly. cp.async.bulk.tensor requires the descriptor
// in .const / .param / .global — not stack.

// ── Loader role (warp 4) ── prefetch tensormaps + TMA loop.
//
// Phase 4.2 + 5b: dep wait inline before A TMAs; per-page cross-task
// waits inline before each W and A TMA on the FIRST iter using each
// stage. Combined with codegen page-protocol prefix (still active),
// the per-page waits are redundant but idempotent.
template <int SPLIT_K = 1, int W_L2_HINT = 0, int TILES_PER_TASK = 1>
__device__ __noinline__ void linear_loader_task(
    mirage::runtime::TaskDesc const *task_desc,
    mirage::runtime_v2::RuntimeSMEM *runtime_smem,
    mirage::runtime::RuntimeConfig const &runtime_config,
    CUtensorMap const *W_tmap_ptr,
    CUtensorMap const *A_tmap_ptr,
    int N_real, int K,
    int tile_idx,
    int instruction_index,
    int iter_num,
    int dyn_sem_base
) {
    // 32 threads of warp 4 enter; the original kernel only ever did
    // single-thread issuance from warp 4, so elect one and let the rest exit.
    if (!elect_sync()) return;

    // Prefetch TMA descriptors.
    asm volatile("prefetch.tensormap [%0];" :: "l"(W_tmap_ptr));
    asm volatile("prefetch.tensormap [%0];" :: "l"(A_tmap_ptr));

    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

    const int W_tma_mbar_base = dyn_sem_base + SEM_W_TMA_BASE * 8;
    const int A_tma_mbar_base = dyn_sem_base + SEM_A_TMA_BASE * 8;
    const int mma_mbar_addr   = dyn_sem_base + SEM_MMA_BASE * 8;

    const int M_mma = N_real;
    const int grid_m = M_mma / BLOCK_M;
    const int num_spatial_tiles = grid_m;
    const int total_k_iters = K / BLOCK_K;
    const int iters_per_slice = total_k_iters / SPLIT_K;
    const int num_tiles = num_spatial_tiles * SPLIT_K;

    if (tile_idx >= num_tiles) return;

    const int tiles_left = num_tiles - tile_idx;
    const int tiles_to_process = (TILES_PER_TASK < tiles_left) ? TILES_PER_TASK : tiles_left;

    constexpr uint64_t W_HINT = (W_L2_HINT == 0) ? L2_EVICT_FIRST : L2_EVICT_NORMAL;

    // Re-initialize the intra-task mma_mbar pipeline barriers HERE in the loader,
    // after the page-wait. Root-cause fix: a stray asynchronous
    // tcgen05.commit(mma_mbar[s]) issued by a PRIOR occupant of this ring slot
    // can land AFTER the controller's init_semaphores re-init (the hardware
    // commit targets the mbar address regardless of re-init), flipping the
    // freshly-zeroed phase back to 1. That defeats this task's loader round-0
    // pipeline-fill pass-through (mma_phase=1 expects fresh phase 0) -> the
    // loader blocks forever at that stage -> launcher waits W -> consumers wait
    // MAINLOOP -> deadlock (observed: ~24/64 gate_up tasks, mma_mbar[5] stale).
    // By the time the loader reaches here (after the cross-task page-wait), all
    // such stray commits have already landed, and this re-init is strictly
    // ordered before the launcher's first commit (which needs this task's W),
    // so it permanently clears the stale phase with no race.
    for (int s = 0; s < NUM_STAGES; s++) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     :: "r"(mma_mbar_addr + s * 8));
        // W_tma/A_tma mbars are async-arrived too (TMA hardware delivers bytes
        // via mbarrier_arrive_expect_tx), so a prior occupant's late TMA
        // completion can stray onto them after init — same hazard as mma_mbar.
        // The loader is their arriver, so re-initing here (before its first TMA
        // and before the launcher's first W/A wait) clears any stale phase.
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     :: "r"(W_tma_mbar_base + s * 8));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                     :: "r"(A_tma_mbar_base + s * 8));
    }
    asm volatile("fence.mbarrier_init.release.cluster;");

    int tma_stage = 0;
    int mma_phase = 1;
    bool dep_done = false;

    for (int t = 0; t < tiles_to_process; t++) {
        const int cur_tile_idx = tile_idx + t;
        const int cur_spatial_idx = cur_tile_idx % num_spatial_tiles;
        const int cur_k_slice = cur_tile_idx / num_spatial_tiles;
        const int cur_k_start = cur_k_slice * iters_per_slice;
        const int cur_off_m = cur_spatial_idx * BLOCK_M;

        for (int i = 0; i < iters_per_slice; i++) {
            const int iter_k = cur_k_start + i;

            // Intra-task gate: launcher signals "stage K MMA done; you can
            // refill". For the first NUM_STAGES iters this passes immediately
            // (init parity), so weight TMAs run as soon as their pages are
            // ready cross-task.
            mbarrier_wait(mma_mbar_addr + tma_stage * 8, mma_phase);

            const int W_mbar = W_tma_mbar_base + tma_stage * 8;
            const int A_mbar = A_tma_mbar_base + tma_stage * 8;
            const int W_smem = smem + task_desc->smem_region_offset(
                ::kernel::linear_sm100_v2::REGION_W_0 + tma_stage);
            const int A_smem = smem + task_desc->smem_region_offset(
                ::kernel::linear_sm100_v2::REGION_A_0 + tma_stage);
            const int z_coord = iter_k * (BLOCK_K / 64);

            // Weight TMA.
            tma_3d_load_l2(W_smem, W_tmap_ptr, 0, cur_off_m, z_coord, W_mbar, W_HINT);
            mbarrier_arrive_expect_tx(W_mbar, W_SIZE);

            // Activation TMA: gated by producer event.
            if (!dep_done) {
                mirage::runtime_v2::wait_task_dependency(
                    runtime_config, task_desc, iter_num);
                dep_done = true;
            }

            tma_3d_load_l2(A_smem, A_tmap_ptr, 0, 0, z_coord, A_mbar, L2_EVICT_LAST);
            mbarrier_arrive_expect_tx(A_mbar, A_SIZE);

            tma_stage = (tma_stage + 1) % NUM_STAGES;
            if (tma_stage == 0) mma_phase ^= 1;
        }
    }

}

// ── Launcher role (warp 5) ── tcgen05.alloc + publish tmem_addr + MMA loop
// + wait consumer_done + tcgen05.dealloc.
//
// Phase 4.3: launcher releases all 14 pages lane-parallel right after the
// MMA loop ends — earlier than the codegen consumer-suffix would (which
// runs at end of consumer body). Loader of the next task can therefore
// start its weight-prefetch TMAs during this task's wait-consumer_done +
// dealloc tail. Linear uses all 14 pages, so no `task_uses_page` check
// is needed; lane K just arrives page K once.
template <int SPLIT_K = 1, int TILES_PER_TASK = 1>
__device__ __noinline__ void linear_launcher_task(
    mirage::runtime::TaskDesc const *task_desc,
    mirage::runtime_v2::RuntimeSMEM *runtime_smem,
    int N_real, int K,
    int tile_idx,
    int dyn_sem_base
) {
    // 32 threads of warp 5 enter. tcgen05.alloc / dealloc are sync.aligned —
    // all 32 lanes participate. The MMA loop itself is single-threaded
    // (issued by an elected lane), matching the pre-split warp-5 branch.
    const int lane_id = threadIdx.x & 31;

    extern __shared__ __align__(1024) char smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));

    const int W_tma_mbar_base      = dyn_sem_base + SEM_W_TMA_BASE * 8;
    const int A_tma_mbar_base      = dyn_sem_base + SEM_A_TMA_BASE * 8;
    const int mma_mbar_addr        = dyn_sem_base + SEM_MMA_BASE * 8;
    const int mainloop_mbar_addr   = dyn_sem_base + SEM_MAINLOOP_BASE * 8;
    const int epilogue_mbar_addr   = dyn_sem_base + SEM_EPILOGUE_BASE * 8;
    const int tmem_ready_mbar_addr = dyn_sem_base + SEM_TMEM_READY * 8;
    const int consumer_done_mbar_addr = dyn_sem_base + SEM_CONSUMER_DONE * 8;

    // Bounds check first: skip alloc/dealloc churn for invalid tiles.
    const int M_mma = N_real;
    const int grid_m = M_mma / BLOCK_M;
    const int num_spatial_tiles = grid_m;
    const int total_k_iters = K / BLOCK_K;
    const int iters_per_slice = total_k_iters / SPLIT_K;
    const int num_tiles = num_spatial_tiles * SPLIT_K;
    if (tile_idx >= num_tiles) {
        // Bounds-fail (padding task): the launcher is the sole role that
        // arrives page_finished. If we just `return`, the declared SMEM pages
        // are never released and the NEXT task on this slot deadlocks waiting
        // for page_ready. Release all declared pages exactly as the normal
        // path does (one arrival per page) before bailing out.
        if (lane_id < MAX_SMEM_PAGES_PER_TASK) {
            mirage::runtime_v2::runtime_finish_page(runtime_smem, lane_id, 1);
        }
        return;
    }

    const int tiles_left = num_tiles - tile_idx;
    const int tiles_to_process = (TILES_PER_TASK < tiles_left) ? TILES_PER_TASK : tiles_left;

    // Allocate TMEM into the planner's scratch region. tcgen05.alloc writes
    // the allocated address as a b32 to [%0] (SMEM scratch).
    // The codegen-emitted loader page-lifecycle prefix already waited
    // the scratch page across-task; no explicit wait needed here.
    const int scratch_smem_addr = smem + task_desc->smem_region_offset(
        ::kernel::linear_sm100_v2::REGION_SCRATCH);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(scratch_smem_addr), "r"(BLOCK_N * 2));

    // Publish: lane 0 arrives SEM_TMEM_READY. The mbar_arrive carries
    // release-cluster semantics so the scratch write becomes visible to
    // consumer warps after their matching wait. All 32 lanes also cache
    // the tmem address into a register here, so the dealloc below does
    // not have to re-read scratch — the codegen-emitted consumer page
    // suffix may have freed the scratch page by that point.
    const int taddr = *reinterpret_cast<int *>(
        smem_ptr + task_desc->smem_region_offset(
            ::kernel::linear_sm100_v2::REGION_SCRATCH));
    if (lane_id == 0) {
        // Re-init mainloop_mbar here (before publishing tmem_ready, which gates
        // the consumer's first mainloop wait, and before the launcher's first
        // mainloop commit). Same root-cause fix as the loader's mma_mbar re-init:
        // mainloop_mbar is tcgen05-committed (async) by the launcher and waited
        // by the consumer, so a stray commit from a prior slot occupant can land
        // after the controller's init_semaphores re-init and corrupt the phase,
        // hanging the consumer -> it never arrives consumer_done -> launcher
        // hangs on consumer_done. Clearing it here removes the stale phase.
        for (int s = 0; s < 2; s++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                         :: "r"(mainloop_mbar_addr + s * 8));
            // epilogue_mbar is waited by the launcher (here) and arrived by the
            // consumer after tmem_ready; re-init it too so a stale phase left by
            // a prior slot occupant (e.g. a 1-tile linear leaves epilogue[0] at
            // phase 1) doesn't block this task's fill-wait. count = 4*WARP_SIZE.
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                         :: "r"(epilogue_mbar_addr + s * 8),
                            "r"(4 * WARP_SIZE));
        }
        asm volatile("fence.mbarrier_init.release.cluster;");
        mbarrier_arrive(tmem_ready_mbar_addr);
    }

    if (elect_sync()) {
        int tma_stage = 0;
        int tma_phase = 0;
        int mainloop_stage = 0;
        int epilogue_phase = 1;

        for (int t = 0; t < tiles_to_process; t++) {
            mbarrier_wait(epilogue_mbar_addr + mainloop_stage * 8, epilogue_phase);

            for (int i = 0; i < iters_per_slice; i++) {
                const int W_smem = smem + task_desc->smem_region_offset(
                    ::kernel::linear_sm100_v2::REGION_W_0 + tma_stage);
                const int A_smem = smem + task_desc->smem_region_offset(
                    ::kernel::linear_sm100_v2::REGION_A_0 + tma_stage);
                const int tmem = taddr + mainloop_stage * BLOCK_N;

                uint64_t a_desc = SMEM_DESC | (W_smem >> 4);
                uint64_t b_desc = SMEM_DESC | (A_smem >> 4);

                // Phase 4.1: wait both W and A mbars (split from tma_mbar).
                mbarrier_wait(W_tma_mbar_base + tma_stage * 8, tma_phase);
                mbarrier_wait(A_tma_mbar_base + tma_stage * 8, tma_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");

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
            mainloop_stage = (mainloop_stage + 1) % 2;
            if (mainloop_stage == 0) epilogue_phase ^= 1;
        }
    }

    // BISECT: temporarily restore Phase 4.3 launcher all-pages release.
    if (lane_id < MAX_SMEM_PAGES_PER_TASK) {
        mirage::runtime_v2::runtime_finish_page(runtime_smem, lane_id, 1);
    }
    __syncwarp();

    // Wait for all 4 consumer warps to finish their last tcgen05.wait, so
    // TMEM is no longer in use. Init parity = 0; after 128 arrives, parity
    // flips to 1; lane 0's wait(phase=0) returns when parity != 0.
    if (lane_id == 0) {
        mbarrier_wait(consumer_done_mbar_addr, 0);
    }
    __syncwarp();

    // Dealloc — sync.aligned, all 32 lanes of warp 5. Uses the cached
    // taddr (read above before consumer body could have released the
    // scratch page) so we do not race with the next task's loader.
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(BLOCK_N * 2));

}

// ── Consumer role (warps 0-3) ── wait tmem_ready + epilogue + consumer_done.
template <bool HAS_RESIDUAL, int M_REAL = 16, int SPLIT_K = 1, int TILES_PER_TASK = 1>
__device__ __noinline__ void linear_consumer_task(
    mirage::runtime::TaskDesc const *task_desc,
    nv_bfloat16 *C_ptr,
    nv_bfloat16 const *res_ptr,
    int N_real, int K,
    int tile_idx,
    float *workspace,
    int dyn_sem_base
) {
    // 128 threads of warps 0-3 enter. Each does its own epilogue store; all
    // 128 arrive consumer_done at the end so launcher can dealloc.
    const int warp_id = warp_uniform(threadIdx.x / WARP_SIZE);
    const int lane_id = threadIdx.x & 31;

    const int M_mma = N_real;
    const int grid_m = M_mma / BLOCK_M;
    const int num_spatial_tiles = grid_m;
    const int total_k_iters = K / BLOCK_K;
    const int iters_per_slice = total_k_iters / SPLIT_K;
    const int num_tiles = num_spatial_tiles * SPLIT_K;

    if (tile_idx >= num_tiles) return;

    const int tiles_left = num_tiles - tile_idx;
    const int tiles_to_process = (TILES_PER_TASK < tiles_left) ? TILES_PER_TASK : tiles_left;

    extern __shared__ __align__(1024) char smem_ptr[];

    const int mainloop_mbar_addr     = dyn_sem_base + SEM_MAINLOOP_BASE * 8;
    const int epilogue_mbar_addr     = dyn_sem_base + SEM_EPILOGUE_BASE * 8;
    const int tmem_ready_mbar_addr   = dyn_sem_base + SEM_TMEM_READY * 8;
    const int consumer_done_mbar_addr = dyn_sem_base + SEM_CONSUMER_DONE * 8;

    // Wait for launcher to publish tmem_addr. Init parity 0 + arrive flips
    // to 1 → wait(phase=0) returns. mbar_wait carries acquire-cluster, so
    // the scratch write made by launcher's alloc is visible after wait.
    if (lane_id == 0) {
        mbarrier_wait(tmem_ready_mbar_addr, 0);
    }
    __syncwarp();
    const int taddr = *reinterpret_cast<int *>(
        smem_ptr + task_desc->smem_region_offset(
            ::kernel::linear_sm100_v2::REGION_SCRATCH));

    int mainloop_stage = 0;
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
        mainloop_stage = (mainloop_stage + 1) % 2;
        if (mainloop_stage == 0) mainloop_phase ^= 1;
    }

    // Signal launcher that all 4 consumer warps are done with TMEM.
    // count = 4 * WARP_SIZE = 128; every consumer thread arrives once.
    mbarrier_arrive(consumer_done_mbar_addr);
}

// ── Storer role (warp 6) ── per-stage page release.
//
// Phase 5: storer is the role that arrives page_finished[page] on behalf of
// linear, freeing the launcher of that work (launcher is at the 255-reg
// cliff). Storer is otherwise idle for linear, so we use it as a passive
// release engine.
//
// Mechanism: storer iterates the launcher's stage cycle in lockstep,
// waiting on mma_mbar[stage] each iter (same parity flips that loader
// waits, so adding storer as an additional waiter is free). It counts
// fires per stage; on the last fire of stage K, the W_K pages are no
// longer read by anyone and can be released. A pages are sub-page packed
// across multiple stages — they release when the last contributing stage
// has had its last fire (tracked via per-page remaining counters).
template <int SPLIT_K = 1, int TILES_PER_TASK = 1>
__device__ __noinline__ void linear_storer_task(
    mirage::runtime::TaskDesc const *task_desc,
    mirage::runtime_v2::RuntimeSMEM *runtime_smem,
    int N_real, int K_param,
    int tile_idx,
    int dyn_sem_base
) {
    if (!elect_sync()) return;

    const int M_mma = N_real;
    const int grid_m = M_mma / BLOCK_M;
    const int num_spatial_tiles = grid_m;
    const int total_k_iters = K_param / BLOCK_K;
    const int iters_per_slice = total_k_iters / SPLIT_K;
    const int num_tiles = num_spatial_tiles * SPLIT_K;

    if (tile_idx >= num_tiles) {
        // Bounds-fail: still arrive every page once so parity tracking
        // doesn't drift. Loader of next task expects exactly one arrive
        // per page per task.
        for (int p = 0; p < MAX_SMEM_PAGES_PER_TASK; p++) {
            mirage::runtime_v2::runtime_finish_page(runtime_smem, p, 1);
        }
        return;
    }

    const int tiles_left = num_tiles - tile_idx;
    const int tiles_to_process = (TILES_PER_TASK < tiles_left) ? TILES_PER_TASK : tiles_left;
    const int total_iters = tiles_to_process * iters_per_slice;

    const int mma_mbar_addr = dyn_sem_base + SEM_MMA_BASE * 8;

    // Per-stage tracking: how many fires before stage K is "done forever"
    // in this task. last_use[K] = floor((total_iters - 1 - K) / NS) + 1.
    int last_use[NUM_STAGES];
    int phase_per_stage[NUM_STAGES];
    int fires_per_stage[NUM_STAGES];
    #pragma unroll
    for (int s = 0; s < NUM_STAGES; s++) {
        last_use[s] = (total_iters - 1 - s) / NUM_STAGES + 1;
        phase_per_stage[s] = 0;
        fires_per_stage[s] = 0;
    }

    // Per-page remaining counter for sub-page-packed pages. Each stage K's
    // A region contributes 1 to its physical page's count. When the count
    // hits 0 (all stages on that page have had their last fire), the page
    // can release. W pages get +1 each from their owning stage.
    int page_remaining[MAX_SMEM_PAGES_PER_TASK] = {0};
    #pragma unroll
    for (int s = 0; s < NUM_STAGES; s++) {
        auto const &W = task_desc->smem_regions[
            ::kernel::linear_sm100_v2::REGION_W_0 + s];
        for (int p = 0; p < W.page_count; p++) {
            page_remaining[W.physical_page_start + p]++;
        }
        auto const &A = task_desc->smem_regions[
            ::kernel::linear_sm100_v2::REGION_A_0 + s];
        for (int p = 0; p < A.page_count; p++) {
            page_remaining[A.physical_page_start + p]++;
        }
    }
    // Scratch page also needs an arrive. It piggybacks on whichever page
    // it shares (or its own page if standalone). Bump the count.
    {
        auto const &SC = task_desc->smem_regions[
            ::kernel::linear_sm100_v2::REGION_SCRATCH];
        for (int p = 0; p < SC.page_count; p++) {
            page_remaining[SC.physical_page_start + p]++;
        }
    }

    // Iterate launcher's stage cycle. Each fire decrements the relevant
    // per-page counter; when a counter hits 0, release that page.
    int stage = 0;
    for (int it = 0; it < total_iters; it++) {
        mbarrier_wait(mma_mbar_addr + stage * 8, phase_per_stage[stage]);
        phase_per_stage[stage] ^= 1;
        fires_per_stage[stage]++;

        if (fires_per_stage[stage] == last_use[stage]) {
            // Stage `stage` has had its last MMA in this task. The W and
            // A regions for this stage are no longer read by launcher.
            auto const &W = task_desc->smem_regions[
                ::kernel::linear_sm100_v2::REGION_W_0 + stage];
            for (int p = 0; p < W.page_count; p++) {
                int phys = W.physical_page_start + p;
                if (--page_remaining[phys] == 0) {
                    mirage::runtime_v2::runtime_finish_page(
                        runtime_smem, phys, 1);
                }
            }
            auto const &A = task_desc->smem_regions[
                ::kernel::linear_sm100_v2::REGION_A_0 + stage];
            for (int p = 0; p < A.page_count; p++) {
                int phys = A.physical_page_start + p;
                if (--page_remaining[phys] == 0) {
                    mirage::runtime_v2::runtime_finish_page(
                        runtime_smem, phys, 1);
                }
            }
        }

        stage = (stage + 1) % NUM_STAGES;
    }

    // Scratch's region count was bumped above. Decrement once here, since
    // scratch is already done with by the time MMA loop finishes (launcher
    // and consumer both cached taddr from scratch at task start).
    {
        auto const &SC = task_desc->smem_regions[
            ::kernel::linear_sm100_v2::REGION_SCRATCH];
        for (int p = 0; p < SC.page_count; p++) {
            int phys = SC.physical_page_start + p;
            if (--page_remaining[phys] == 0) {
                mirage::runtime_v2::runtime_finish_page(
                    runtime_smem, phys, 1);
            }
        }
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
