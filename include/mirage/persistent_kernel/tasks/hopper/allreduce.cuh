/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "tasks/common/common_header.cuh"

#ifdef USE_NVSHMEM

// ---------------------------------------------------------------------------
// Self-contained NVLS tile allreduce for the Mirage persistent kernel.
//
// Problem: including <nvshmem.h> / <nvshmemx.h> pulls in hundreds of
// __noinline__ transfer functions from nvshmem_defines.h and
// transfer_device.cuh.  With -rdc=true these inflate register count from
// 166 -> 255, preventing persistent-kernel co-scheduling on B200.
//
// Solution: include ONLY the two lightweight type-definition headers from
// NVSHMEM (no device functions), and inline the ~30 lines of device code
// actually needed for the NVLS ONE_SHOT_PULL bf16 SUM allreduce.
// ---------------------------------------------------------------------------

// Pure struct / type definitions -- no device functions
#include "device_host/nvshmem_types.h"
#include "device_host/nvshmem_tensor.h"

// In NVSHMEM_NO_DEVICE_LIB mode (rdc=false), we define the device state ourselves.
// It's populated at init time by our callback in nvshmemid_hostlib_init_attr.
// In standard mode (rdc=true), it comes from libnvshmem_device.a.
#ifdef NVSHMEM_NO_DEVICE_LIB
__managed__ nvshmemi_device_host_state_t nvshmemi_device_state_d;
#else
extern __device__ nvshmemi_device_host_state_t nvshmemi_device_state_d;
#endif

namespace kernel {

// ========================= constants ========================================
// From nvshmem_common.cuh -- only the handful we need.
static constexpr int MPKAR_SYNC_SIZE       = 27648;           // SYNC_SIZE
static constexpr int MPKAR_NVSHMEMI_SYNC_SIZE = 2 * MPKAR_SYNC_SIZE;
static constexpr int MPKAR_NVSHMEMI_JOB_GPU_LDST = 1 << 1;
static constexpr int MPKAR_NVSHMEMI_CALL_SITE_BARRIER_WARP = 1;

// ========================= team helpers =====================================

// Equivalent to nvshmemi_team_get_sync_counter(team)
static __device__ __forceinline__
long *mpkar_team_get_sync_counter(nvshmemi_team_t *team) {
    return &nvshmemi_device_state_d.sync_counter[2 * team->team_idx];
}

// Minimal version of nvshmemi_team_get_psync for SYNC op only.
// Full version computes offsets for REDUCE/BCAST/FCOLLECT etc. which we
// never use, so we avoid pulling in the dependent macros.
static __device__ __forceinline__
long *mpkar_team_get_psync_sync(nvshmemi_team_t *team) {
    long *team_psync = &nvshmemi_device_state_d.psync_pool[0];
    // We need get_psync_len_per_team() to index into the pool, but that
    // function depends on fcollect thresholds and other env params.
    // The NVSHMEM internal implementation indexes as:
    //   psync_pool[team_idx * get_psync_len_per_team()]
    // We replicate that computation here.

    // --- replicate get_fcollect_psync_len_per_team ---
    size_t fcollect_ll_threshold =
        nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll_threshold;
    size_t fcollect_sync_size =
        (2 * 2 * nvshmemi_device_state_d.npes * fcollect_ll_threshold) / sizeof(long);

    // --- replicate get_fcollect_ll128_psync_len_per_team ---
    size_t fcollect_ll128_threshold =
        nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll128_threshold;
    // NVSHMEMI_FCOLLECT_LL128_CALC_PSYNC_SIZE(x, T):
    //   ROUND_UP(x, 120/sizeof(T)) + sizeof(uint64_t)/sizeof(T) * ROUND_UP_DIV(x, 120/sizeof(T))
    // For T=char, sizeof(T)=1:
    //   ROUND_UP(x, 120) + 8 * ROUND_UP_DIV(x, 120)
    auto round_up_div = [](size_t x, size_t y) -> size_t { return (x + y - 1) / y; };
    auto round_up = [&](size_t x, size_t y) -> size_t { return round_up_div(x, y) * y; };

    size_t fcollect_ll128_sync_size =
        round_up(fcollect_ll128_threshold, 120) +
        8 * round_up_div(fcollect_ll128_threshold, 120);
    fcollect_ll128_sync_size =
        fcollect_ll128_sync_size * 2 * nvshmemi_device_state_d.npes / sizeof(long);

    // --- replicate get_psync_len_per_team ---
    size_t psync_len = (4 * (size_t)MPKAR_NVSHMEMI_SYNC_SIZE +
        nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size / sizeof(long) +
        10 * (size_t)MPKAR_SYNC_SIZE +  // NVSHMEMI_BCAST_SYNC_SIZE
        fcollect_sync_size +
        2 * (size_t)MPKAR_SYNC_SIZE +   // 2 * NVSHMEMI_ALLTOALL_SYNC_SIZE
        fcollect_ll128_sync_size +
        nvshmemi_device_state_d.npes);
    // NVSHMEMI_TEAM_ROUND_UP(ans, 2)
    psync_len = round_up(psync_len, 2);

    team_psync = &nvshmemi_device_state_d.psync_pool[team->team_idx * psync_len];
    // For SYNC op, psync is at offset 0 within the team's region
    return team_psync;
}

// Translate PE index (may wrap) to world PE via pe_mapping.
static __device__ __forceinline__
int mpkar_team_translate_pe(nvshmemi_team_t *team, int pe_idx) {
    return team->pe_mapping[pe_idx % team->size];
}

// ========================= barrier signal ===================================
// P2P volatile store to peer's psync slot.
// Only the P2P path (job_connectivity <= NVSHMEMI_JOB_GPU_LDST) is needed
// for NVLS teams where all GPUs are NVLink-connected.
static __device__ __forceinline__
void mpkar_signal_for_barrier(long *dest, long value, int pe) {
    const void *peer_base_addr =
        (void *)__ldg(
            (const long long unsigned *)nvshmemi_device_state_d.peer_heap_base_p2p + pe);
    volatile long *dest_actual =
        (volatile long *)((char *)(peer_base_addr) +
                          ((char *)dest - (char *)(nvshmemi_device_state_d.heap_base)));
    *dest_actual = value;
}

// ========================= spin-wait ========================================
static __device__ __forceinline__
void mpkar_wait_until_ge(volatile long *addr, long val) {
    while (*addr < val) {
        // spin
    }
}

// ========================= dissemination barrier ============================
// Power-of-2 radix dissemination barrier (block scope).
// Template params: k = radix, logk = log2(k).
// For our use case with are_gpus_p2p_connected && SCOPE==BLOCK, the NVSHMEM
// code sets k = team->size.  For TP=2 => k=2,logk=1; TP=4 => k=4,logk=2;
// TP=8 => k=8,logk=3.
template <int k, int logk>
static __device__ __forceinline__
void mpkar_sync_dissem_pow2_block(nvshmem_team_t team) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int size = teami->size;
    volatile long *sync_counter =
        (volatile long *)mpkar_team_get_sync_counter(teami);
    volatile long *pSync =
        (volatile long *)mpkar_team_get_psync_sync(teami) +
        MPKAR_NVSHMEMI_SYNC_SIZE * (sync_counter[0] % 2);

    int shift;
    int to_nbr_idx, to_nbr;
    int from_nbr_idx, from_nbr;
    int temp = size - 1;
    int phase_num = 0;
    volatile long *counter = sync_counter;

    while (temp) {
        // notify neighbors
        for (int j = threadIdx.x + 1; j <= k - 1; j += blockDim.x) {
            shift = j << phase_num;
            if (shift >= size) break;
            to_nbr_idx = teami->my_pe + shift;
            to_nbr = mpkar_team_translate_pe(teami, to_nbr_idx);
            mpkar_signal_for_barrier(
                (long *)pSync + nvshmemi_device_state_d.mype,
                counter[0], to_nbr);
        }

        // wait for neighbors
        for (int j = threadIdx.x + 1; j <= k - 1; j += blockDim.x) {
            shift = j << phase_num;
            if (shift >= size) break;
            from_nbr_idx = teami->my_pe - shift;
            if (from_nbr_idx < 0) from_nbr_idx = size + from_nbr_idx;
            from_nbr = mpkar_team_translate_pe(teami, from_nbr_idx);
            mpkar_wait_until_ge(pSync + from_nbr, counter[0]);
        }
        temp >>= logk;
        phase_num++;
        __syncthreads();
    }
    if (threadIdx.x == 0) sync_counter[0] += 1;
    __syncthreads();
}

// Generic (non-power-of-2) dissemination barrier for non-strided teams.
static __device__ __forceinline__
void mpkar_sync_dissem_generic_block(nvshmem_team_t team) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int size = teami->size;
    volatile long *sync_counter =
        (volatile long *)mpkar_team_get_sync_counter(teami);
    volatile long *pSync =
        (volatile long *)mpkar_team_get_psync_sync(teami) +
        MPKAR_NVSHMEMI_SYNC_SIZE * (sync_counter[0] % 2);

    int k = min(
        nvshmemi_device_state_d.gpu_coll_env_params_var.barrier_tg_dissem_kval,
        size);
    int my_idx = teami->my_pe;
    int temp = size - 1;
    int num_phases = 0;
    while (temp) { num_phases++; temp /= k; }

    volatile long *counter = sync_counter;
    int pow_k = 1;
    for (int i = 0; i < num_phases; i++) {
        for (int j = threadIdx.x + 1; j <= k - 1; j += blockDim.x) {
            int shift = j * pow_k;
            if (shift >= size) break;
            int to_nbr_idx = (my_idx + shift) % size;
            int to_nbr = teami->pe_mapping[to_nbr_idx];
            mpkar_signal_for_barrier(
                (long *)pSync + nvshmemi_device_state_d.mype,
                counter[0], to_nbr);
        }
        for (int j = threadIdx.x + 1; j <= k - 1; j += blockDim.x) {
            int shift = j * pow_k;
            if (shift >= size) break;
            int from_nbr_idx = my_idx - shift;
            if (from_nbr_idx < 0) from_nbr_idx = size + from_nbr_idx;
            int from_nbr = teami->pe_mapping[from_nbr_idx];
            mpkar_wait_until_ge(pSync + from_nbr, counter[0]);
        }
        pow_k *= k;
        __syncthreads();
    }
    if (threadIdx.x == 0) sync_counter[0] += 1;
    __syncthreads();
}

// Strided team dissemination barrier.
static __device__ __forceinline__
void mpkar_sync_dissem_strided_block(nvshmemi_team_t *teami,
                                     volatile long *pSync,
                                     volatile long *sync_counter) {
    int start = teami->start;
    int stride = teami->stride;
    int size = teami->size;
    int k = min(
        nvshmemi_device_state_d.gpu_coll_env_params_var.barrier_tg_dissem_kval,
        size);
    int my_idx = (nvshmemi_device_state_d.mype - start) / stride;
    int temp = size - 1;
    int num_phases = 0;
    while (temp) { num_phases++; temp /= k; }

    volatile long *counter = sync_counter;
    int pow_k = 1;
    for (int i = 0; i < num_phases; i++) {
        for (int j = threadIdx.x + 1; j <= k - 1; j += blockDim.x) {
            int shift = j * pow_k;
            if (shift >= size) break;
            int to_nbr_idx = (my_idx + shift) % size;
            int to_nbr = start + to_nbr_idx * stride;
            mpkar_signal_for_barrier(
                (long *)pSync + nvshmemi_device_state_d.mype,
                counter[0], to_nbr);
        }
        for (int j = threadIdx.x + 1; j <= k - 1; j += blockDim.x) {
            int shift = j * pow_k;
            if (shift >= size) break;
            int from_nbr_idx = my_idx - shift;
            if (from_nbr_idx < 0) from_nbr_idx = size + from_nbr_idx;
            int from_nbr = start + from_nbr_idx * stride;
            mpkar_wait_until_ge(pSync + from_nbr, counter[0]);
        }
        pow_k *= k;
        __syncthreads();
    }
    if (threadIdx.x == 0) sync_counter[0] += 1;
    __syncthreads();
}

// Top-level sync dispatch (block scope).
// Mirrors nvshmemi_sync_algo_threadgroup<BLOCK>.
static __device__ __forceinline__
void mpkar_sync_block(nvshmem_team_t team) {
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    int size = teami->size;
    int k = min(
        nvshmemi_device_state_d.gpu_coll_env_params_var.barrier_tg_dissem_kval,
        size);
    k = max(k, 2);
    // For P2P-connected teams at block scope, use full-radix (k = size)
    if (teami->are_gpus_p2p_connected) k = size;

    switch (k) {
        case 2:  mpkar_sync_dissem_pow2_block<2,  1>(team); break;
        case 4:  mpkar_sync_dissem_pow2_block<4,  2>(team); break;
        case 8:  mpkar_sync_dissem_pow2_block<8,  3>(team); break;
        case 16: mpkar_sync_dissem_pow2_block<16, 4>(team); break;
        case 32: mpkar_sync_dissem_pow2_block<32, 5>(team); break;
        default: {
            // Non-power-of-2 or stride>0: use generic path
            if (teami->stride > 0) {
                volatile long *sync_counter =
                    (volatile long *)mpkar_team_get_sync_counter(teami);
                volatile long *pSync =
                    (volatile long *)mpkar_team_get_psync_sync(teami) +
                    MPKAR_NVSHMEMI_SYNC_SIZE * (sync_counter[0] % 2);
                mpkar_sync_dissem_strided_block(teami, pSync, sync_counter);
            } else {
                mpkar_sync_dissem_generic_block(team);
            }
            break;
        }
    }
}

// ========================= NVLS multicast pointer ===========================
static __device__ __forceinline__
void *mpkar_mc_ptr(nvshmemi_team_t *team, const void *ptr) {
    if (team == nullptr || team->nvls_rsc_base_ptr == nullptr) {
        return nullptr;
    }
    ptrdiff_t offset =
        (char *)ptr - (char *)nvshmemi_device_state_d.heap_base;
    if (ptr >= nvshmemi_device_state_d.heap_base &&
        offset < (ptrdiff_t)nvshmemi_device_state_d.heap_size &&
        team->nvls_rsc_base_ptr != nullptr) {
        void *mc_addr =
            (void *)__ldg((const long long unsigned *)team->nvls_rsc_base_ptr);
        if (mc_addr != nullptr)
            mc_addr = (void *)((char *)mc_addr + offset);
        return mc_addr;
    }
    return nullptr;
}

// ========================= NVLS ld_reduce PTX ===============================
// bf16: multimem.ld_reduce.global.add.acc::f32.v4.bf16x2
// Loads 16 bytes (8 bf16 values) from multicast address, reduces across GPUs.
static __device__ __forceinline__
void mpkar_nvls_ld_reduce_bf16_v4(uint32_t &r0, uint32_t &r1,
                                  uint32_t &r2, uint32_t &r3,
                                  const int4 *mc_addr) {
    asm("multimem.ld_reduce.global.add.acc::f32.v4.bf16x2 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "l"(mc_addr));
}

// f16: multimem.ld_reduce.global.add.acc::f32.v4.f16x2
static __device__ __forceinline__
void mpkar_nvls_ld_reduce_f16_v4(uint32_t &r0, uint32_t &r1,
                                 uint32_t &r2, uint32_t &r3,
                                 const int4 *mc_addr) {
    asm("multimem.ld_reduce.global.add.acc::f32.v4.f16x2 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "l"(mc_addr));
}

// f32: multimem.ld_reduce.global.add.v4.f32
static __device__ __forceinline__
void mpkar_nvls_ld_reduce_f32_v4(float &r0, float &r1,
                                 float &r2, float &r3,
                                 const int4 *mc_addr) {
    asm("multimem.ld_reduce.global.add.v4.f32 "
        "{%0, %1, %2, %3}, [%4];"
        : "=f"(r0), "=f"(r1), "=f"(r2), "=f"(r3)
        : "l"(mc_addr));
}

// ========================= NVLS one-shot reduce (v4, block scope) ===========
// Type-dispatched NVLS reduce: read from multicast src, write to local dst.
// ONE_SHOT: ld_reduce from MC address, st.global to local HBM dst.
// Handles bf16, f16, f32 via PTX specialization.
template <typename T>
static __device__ __forceinline__
void mpkar_nvls_reduce_v4_block(int4 *__restrict__ dst,
                                const int4 *__restrict__ mc_src,
                                int nelems_v4) {
    for (int j = threadIdx.x; j < nelems_v4; j += blockDim.x) {
        uint32_t u4[4];

        // Dispatch on element type
        if constexpr (sizeof(T) == 2) {
            // bf16 or f16 -- both use the same register type (uint32_t)
            // but different PTX instructions
            if constexpr (cuda::std::is_same<T, __nv_bfloat16>::value) {
                mpkar_nvls_ld_reduce_bf16_v4(u4[0], u4[1], u4[2], u4[3],
                                             mc_src + j);
            } else {
                // half / __half
                mpkar_nvls_ld_reduce_f16_v4(u4[0], u4[1], u4[2], u4[3],
                                            mc_src + j);
            }
            asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
                :: "l"(dst + j),
                   "r"(u4[0]), "r"(u4[1]), "r"(u4[2]), "r"(u4[3]));
        } else {
            // float
            float f4[4];
            mpkar_nvls_ld_reduce_f32_v4(f4[0], f4[1], f4[2], f4[3],
                                        mc_src + j);
            asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
                :: "l"(dst + j),
                   "r"(__float_as_uint(f4[0])),
                   "r"(__float_as_uint(f4[1])),
                   "r"(__float_as_uint(f4[2])),
                   "r"(__float_as_uint(f4[3])));
        }
    }
}

// ========================= public API =======================================
// Drop-in replacement for the old nvshmem_tile_allreduce.
// Same template interface so callers do not need to change.
//
// Template params:
//   T            - element type (__nv_bfloat16, half, float)
//   BATCH_SIZE   - unused (kept for API compat)
//   OUTPUT_SIZE  - contiguous dimension in elements
//   OUTPUT_STRIDE - stride of minor dimension in elements
template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__device__ __forceinline__ void nvshmem_tile_allreduce(void *input_ptr,
                                                       void *output_ptr,
                                                       void *_teams,
                                                       int task_offset,
                                                       int active_tokens) {
    nvshmem_team_t *teams = reinterpret_cast<nvshmem_team_t *>(_teams);
    nvshmem_team_t team = teams[task_offset];

#ifdef MPK_AR_LOCAL_COPY
    // ABLATION: replace AllReduce with local copy (output=input).
    // This isolates whether the bug is in AllReduce or in the MLP computation.
    int const n_elems = OUTPUT_SIZE * active_tokens;
    for (int i = threadIdx.x; i < n_elems; i += blockDim.x) {
        reinterpret_cast<T *>(output_ptr)[i] =
            reinterpret_cast<T *>(input_ptr)[i];
    }
#else
    // --- Phase 1: ensure local data is visible, then cross-GPU barrier ---
    __threadfence();
    mpkar_sync_block(team);

    // --- Phase 2: NVLS multicast ld_reduce -> local store ---
    nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
    void *mc_src = mpkar_mc_ptr(teami, input_ptr);

    // Compute number of int4 (16-byte) elements.
    // For 2D tile with shape [OUTPUT_SIZE, active_tokens] and stride
    // [1, OUTPUT_STRIDE]:
    //   contiguous dimension = OUTPUT_SIZE elements
    //   number of rows = active_tokens
    // When OUTPUT_SIZE == OUTPUT_STRIDE (no padding), the tile is fully
    // contiguous and we can do a single vectorized pass.
    // When OUTPUT_SIZE < OUTPUT_STRIDE, we must iterate row by row.

    static_assert(OUTPUT_SIZE % (16 / sizeof(T)) == 0,
                  "OUTPUT_SIZE must be a multiple of 16/sizeof(T) for v4 NVLS");

    constexpr int ELEMS_PER_V4 = 16 / sizeof(T);  // 8 for bf16, 4 for f32
    constexpr int V4_PER_ROW = OUTPUT_SIZE / ELEMS_PER_V4;
    constexpr int STRIDE_V4 = OUTPUT_STRIDE / ELEMS_PER_V4;

    int4 *dst_v4 = reinterpret_cast<int4 *>(output_ptr);
    const int4 *src_mc_v4 = reinterpret_cast<const int4 *>(mc_src);

    if constexpr (OUTPUT_SIZE == OUTPUT_STRIDE) {
        // Contiguous: one pass over all rows
        int total_v4 = V4_PER_ROW * active_tokens;
        mpkar_nvls_reduce_v4_block<T>(dst_v4, src_mc_v4, total_v4);
    } else {
        // Strided: per-row
        for (int row = 0; row < active_tokens; row++) {
            mpkar_nvls_reduce_v4_block<T>(
                dst_v4 + row * STRIDE_V4,
                src_mc_v4 + row * STRIDE_V4,
                V4_PER_ROW);
        }
    }

    // --- Phase 3: ensure PULL stores are visible locally ---
    // PULL variant stores only to local HBM, so __threadfence() (not
    // __threadfence_system()) is sufficient.
    __threadfence();
    __syncthreads();
#endif
}

} // namespace kernel

#endif // USE_NVSHMEM
