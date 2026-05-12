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
#include "device_host/nvshmem_tensor.h"
#include "device_host/nvshmem_types.h"

// nvshmemi_device_state_d is defined in runtime_header.h (before any NVSHMEM
// headers) to ensure visibility for proxy_device.cuh etc.

namespace kernel {

// ========================= constants ========================================
// From nvshmem_common.cuh -- only the handful we need.
static constexpr int MPKAR_SYNC_SIZE = 27648; // SYNC_SIZE
static constexpr int MPKAR_NVSHMEMI_SYNC_SIZE = 2 * MPKAR_SYNC_SIZE;
static constexpr int MPKAR_NVSHMEMI_JOB_GPU_LDST = 1 << 1;
static constexpr int MPKAR_NVSHMEMI_CALL_SITE_BARRIER_WARP = 1;

// ========================= team helpers =====================================

// Equivalent to nvshmemi_team_get_sync_counter(team)
static __device__ __forceinline__ long *
    mpkar_team_get_sync_counter(nvshmemi_team_t *team) {
  return &nvshmemi_device_state_d.sync_counter[2 * team->team_idx];
}

// Minimal version of nvshmemi_team_get_psync for SYNC op only.
// Full version computes offsets for REDUCE/BCAST/FCOLLECT etc. which we
// never use, so we avoid pulling in the dependent macros.
static __device__ __forceinline__ long *
    mpkar_team_get_psync_sync(nvshmemi_team_t *team) {
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
      (2 * 2 * nvshmemi_device_state_d.npes * fcollect_ll_threshold) /
      sizeof(long);

  // --- replicate get_fcollect_ll128_psync_len_per_team ---
  size_t fcollect_ll128_threshold =
      nvshmemi_device_state_d.gpu_coll_env_params_var.fcollect_ll128_threshold;
  // NVSHMEMI_FCOLLECT_LL128_CALC_PSYNC_SIZE(x, T):
  //   ROUND_UP(x, 120/sizeof(T)) + sizeof(uint64_t)/sizeof(T) * ROUND_UP_DIV(x,
  //   120/sizeof(T))
  // For T=char, sizeof(T)=1:
  //   ROUND_UP(x, 120) + 8 * ROUND_UP_DIV(x, 120)
  auto round_up_div = [](size_t x, size_t y) -> size_t {
    return (x + y - 1) / y;
  };
  auto round_up = [&](size_t x, size_t y) -> size_t {
    return round_up_div(x, y) * y;
  };

  size_t fcollect_ll128_sync_size =
      round_up(fcollect_ll128_threshold, 120) +
      8 * round_up_div(fcollect_ll128_threshold, 120);
  fcollect_ll128_sync_size = fcollect_ll128_sync_size * 2 *
                             nvshmemi_device_state_d.npes / sizeof(long);

  // --- replicate get_psync_len_per_team ---
  size_t psync_len =
      (4 * (size_t)MPKAR_NVSHMEMI_SYNC_SIZE +
       nvshmemi_device_state_d.gpu_coll_env_params_var.reduce_scratch_size /
           sizeof(long) +
       10 * (size_t)MPKAR_SYNC_SIZE + // NVSHMEMI_BCAST_SYNC_SIZE
       fcollect_sync_size +
       2 * (size_t)MPKAR_SYNC_SIZE + // 2 * NVSHMEMI_ALLTOALL_SYNC_SIZE
       fcollect_ll128_sync_size + nvshmemi_device_state_d.npes);
  // NVSHMEMI_TEAM_ROUND_UP(ans, 2)
  psync_len = round_up(psync_len, 2);

  team_psync = &nvshmemi_device_state_d.psync_pool[team->team_idx * psync_len];
  // For SYNC op, psync is at offset 0 within the team's region
  return team_psync;
}

// Translate PE index (may wrap) to world PE via pe_mapping.
static __device__ __forceinline__ int
    mpkar_team_translate_pe(nvshmemi_team_t *team, int pe_idx) {
  return team->pe_mapping[pe_idx % team->size];
}

// ========================= barrier signal ===================================
// P2P volatile store to peer's psync slot.
// Only the P2P path (job_connectivity <= NVSHMEMI_JOB_GPU_LDST) is needed
// for NVLS teams where all GPUs are NVLink-connected.
static __device__ __forceinline__ void
    mpkar_signal_for_barrier(long *dest, long value, int pe) {
  void const *peer_base_addr = (void *)__ldg(
      (long long unsigned const *)nvshmemi_device_state_d.peer_heap_base_p2p +
      pe);
  long volatile *dest_actual =
      (long volatile *)((char *)(peer_base_addr) +
                        ((char *)dest -
                         (char *)(nvshmemi_device_state_d.heap_base)));
  *dest_actual = value;
}

// ========================= spin-wait ========================================
static __device__ __forceinline__ void mpkar_wait_until_ge(long volatile *addr,
                                                           long val) {
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
static __device__ __forceinline__ void
    mpkar_sync_dissem_pow2_block(nvshmem_team_t team) {
  nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
  int size = teami->size;
  long volatile *sync_counter =
      (long volatile *)mpkar_team_get_sync_counter(teami);
  long volatile *pSync = (long volatile *)mpkar_team_get_psync_sync(teami) +
                         MPKAR_NVSHMEMI_SYNC_SIZE * (sync_counter[0] % 2);

  int shift;
  int to_nbr_idx, to_nbr;
  int from_nbr_idx, from_nbr;
  int temp = size - 1;
  int phase_num = 0;
  long volatile *counter = sync_counter;

  while (temp) {
    // notify neighbors
    for (int j = threadIdx.x + 1; j <= k - 1; j += blockDim.x) {
      shift = j << phase_num;
      if (shift >= size) {
        break;
      }
      to_nbr_idx = teami->my_pe + shift;
      to_nbr = mpkar_team_translate_pe(teami, to_nbr_idx);
      mpkar_signal_for_barrier(
          (long *)pSync + nvshmemi_device_state_d.mype, counter[0], to_nbr);
    }

    // wait for neighbors
    for (int j = threadIdx.x + 1; j <= k - 1; j += blockDim.x) {
      shift = j << phase_num;
      if (shift >= size) {
        break;
      }
      from_nbr_idx = teami->my_pe - shift;
      if (from_nbr_idx < 0) {
        from_nbr_idx = size + from_nbr_idx;
      }
      from_nbr = mpkar_team_translate_pe(teami, from_nbr_idx);
      mpkar_wait_until_ge(pSync + from_nbr, counter[0]);
    }
    temp >>= logk;
    phase_num++;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    sync_counter[0] += 1;
  }
  __syncthreads();
}

// Generic (non-power-of-2) dissemination barrier for non-strided teams.
static __device__ __forceinline__ void
    mpkar_sync_dissem_generic_block(nvshmem_team_t team) {
  nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
  int size = teami->size;
  long volatile *sync_counter =
      (long volatile *)mpkar_team_get_sync_counter(teami);
  long volatile *pSync = (long volatile *)mpkar_team_get_psync_sync(teami) +
                         MPKAR_NVSHMEMI_SYNC_SIZE * (sync_counter[0] % 2);

  int k = min(
      nvshmemi_device_state_d.gpu_coll_env_params_var.barrier_tg_dissem_kval,
      size);
  int my_idx = teami->my_pe;
  int temp = size - 1;
  int num_phases = 0;
  while (temp) {
    num_phases++;
    temp /= k;
  }

  long volatile *counter = sync_counter;
  int pow_k = 1;
  for (int i = 0; i < num_phases; i++) {
    for (int j = threadIdx.x + 1; j <= k - 1; j += blockDim.x) {
      int shift = j * pow_k;
      if (shift >= size) {
        break;
      }
      int to_nbr_idx = (my_idx + shift) % size;
      int to_nbr = teami->pe_mapping[to_nbr_idx];
      mpkar_signal_for_barrier(
          (long *)pSync + nvshmemi_device_state_d.mype, counter[0], to_nbr);
    }
    for (int j = threadIdx.x + 1; j <= k - 1; j += blockDim.x) {
      int shift = j * pow_k;
      if (shift >= size) {
        break;
      }
      int from_nbr_idx = my_idx - shift;
      if (from_nbr_idx < 0) {
        from_nbr_idx = size + from_nbr_idx;
      }
      int from_nbr = teami->pe_mapping[from_nbr_idx];
      mpkar_wait_until_ge(pSync + from_nbr, counter[0]);
    }
    pow_k *= k;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    sync_counter[0] += 1;
  }
  __syncthreads();
}

// Strided team dissemination barrier.
static __device__ __forceinline__ void mpkar_sync_dissem_strided_block(
    nvshmemi_team_t *teami, long volatile *pSync, long volatile *sync_counter) {
  int start = teami->start;
  int stride = teami->stride;
  int size = teami->size;
  int k = min(
      nvshmemi_device_state_d.gpu_coll_env_params_var.barrier_tg_dissem_kval,
      size);
  int my_idx = (nvshmemi_device_state_d.mype - start) / stride;
  int temp = size - 1;
  int num_phases = 0;
  while (temp) {
    num_phases++;
    temp /= k;
  }

  long volatile *counter = sync_counter;
  int pow_k = 1;
  for (int i = 0; i < num_phases; i++) {
    for (int j = threadIdx.x + 1; j <= k - 1; j += blockDim.x) {
      int shift = j * pow_k;
      if (shift >= size) {
        break;
      }
      int to_nbr_idx = (my_idx + shift) % size;
      int to_nbr = start + to_nbr_idx * stride;
      mpkar_signal_for_barrier(
          (long *)pSync + nvshmemi_device_state_d.mype, counter[0], to_nbr);
    }
    for (int j = threadIdx.x + 1; j <= k - 1; j += blockDim.x) {
      int shift = j * pow_k;
      if (shift >= size) {
        break;
      }
      int from_nbr_idx = my_idx - shift;
      if (from_nbr_idx < 0) {
        from_nbr_idx = size + from_nbr_idx;
      }
      int from_nbr = start + from_nbr_idx * stride;
      mpkar_wait_until_ge(pSync + from_nbr, counter[0]);
    }
    pow_k *= k;
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    sync_counter[0] += 1;
  }
  __syncthreads();
}

// Top-level sync dispatch (block scope).
// Mirrors nvshmemi_sync_algo_threadgroup<BLOCK>.
static __device__ __forceinline__ void mpkar_sync_block(nvshmem_team_t team) {
  nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
  int size = teami->size;
  int k = min(
      nvshmemi_device_state_d.gpu_coll_env_params_var.barrier_tg_dissem_kval,
      size);
  k = max(k, 2);
  // For P2P-connected teams at block scope, use full-radix (k = size)
  if (teami->are_gpus_p2p_connected) {
    k = size;
  }

  switch (k) {
    case 2:
      mpkar_sync_dissem_pow2_block<2, 1>(team);
      break;
    case 4:
      mpkar_sync_dissem_pow2_block<4, 2>(team);
      break;
    case 8:
      mpkar_sync_dissem_pow2_block<8, 3>(team);
      break;
    case 16:
      mpkar_sync_dissem_pow2_block<16, 4>(team);
      break;
    case 32:
      mpkar_sync_dissem_pow2_block<32, 5>(team);
      break;
    default: {
      // Non-power-of-2 or stride>0: use generic path
      if (teami->stride > 0) {
        long volatile *sync_counter =
            (long volatile *)mpkar_team_get_sync_counter(teami);
        long volatile *pSync =
            (long volatile *)mpkar_team_get_psync_sync(teami) +
            MPKAR_NVSHMEMI_SYNC_SIZE * (sync_counter[0] % 2);
        mpkar_sync_dissem_strided_block(teami, pSync, sync_counter);
      } else {
        mpkar_sync_dissem_generic_block(team);
      }
      break;
    }
  }
}

// ============= Per-task contention-free dissemination barrier ===============
//
// Variant of `mpkar_sync_block` that uses a PRIVATE counter and slot pair
// per task_offset, eliminating the contention measured in the 2026-05-12
// phase-isolation experiment (56 concurrent AR tasks contending on the
// shared `sync_counter[0]` + `pSync[mype]` cost 91 μs/task barrier vs an
// expected ~6 μs for an uncontended dissemination).
//
// Layout in the team's psync_pool region (≥4*SYNC_SIZE longs reserved, of
// which the legacy SYNC op uses only slots 0..2*SYNC_SIZE-1):
//   per_task_region = psync_base + 2 * MPKAR_NVSHMEMI_SYNC_SIZE
//   task_counter[task_offset] = per_task_region + task_offset
//   task_pSync[task_offset, phase&1, pe] = per_task_region + MAX_AR_TASKS
//       + task_offset*size*2 + (phase&1)*size + pe
// Budget: MAX_AR_TASKS=128 → 128 (counters) + 128*8*2=2048 (slots) = 17 KB
// out of ~440 KB unused in the team's reduce/bcast region of psync_pool.
//
// Stationarity precondition (audited 2026-05-12 in
// scratch/ar_rewrite_design.md): each AR call fires the same (team,
// task_offset) on every PE exactly once. DSv3's `_use_prefill` gate,
// `gate_mode` runtime check, and MTP draft loops are all symmetric across
// PEs. Future call sites that asymmetrically skip a single rank MUST fall
// back to the legacy `mpkar_sync_block` path.
//
// Only enabled when compiled with `-DMPK_AR_PER_TASK_BARRIER` (propagated by
// `MPK_AR_PER_TASK_BARRIER=1` in persistent_kernel.py).
static constexpr int MPKAR_PER_TASK_MAX_TASKS = 128;

static __device__ __forceinline__ void
    mpkar_sync_block_per_task(nvshmem_team_t team, int task_offset) {
  nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
  int size = teami->size;

  if (!teami->are_gpus_p2p_connected ||
      task_offset < 0 ||
      task_offset >= MPKAR_PER_TASK_MAX_TASKS) {
    // Conservative fallback for cases this routine isn't designed for
    // (non-P2P team or task_offset out of bounds). DSv3 TP=4 / TP=2 / TP=8
    // teams are always P2P-connected so this branch is dead in practice.
    mpkar_sync_block(team);
    return;
  }

  long volatile *psync_base = (long volatile *)mpkar_team_get_psync_sync(teami);
  long volatile *per_task_region =
      psync_base + 2 * MPKAR_NVSHMEMI_SYNC_SIZE;
  long volatile *task_counter = per_task_region + task_offset;
  long volatile *task_pSync_base =
      per_task_region + MPKAR_PER_TASK_MAX_TASKS + task_offset * size * 2;

  // Atomic increment to claim a unique phase number for THIS task on THIS PE.
  // First call sees counter=0 (psync_pool zero-init by NVSHMEM bootstrap)
  // and signals phase=1. The +1 is critical: pSync slots are also zero-init,
  // so a phase=0 signal would trivially satisfy any peer's wait_until_ge.
  long my_phase = 0;
  if (threadIdx.x == 0) {
    my_phase = (long)atomicAdd(
                   reinterpret_cast<unsigned long long *>(
                       const_cast<long *>(task_counter)),
                   (unsigned long long)1) +
               1;
  }
  __shared__ long s_my_phase;
  if (threadIdx.x == 0) {
    s_my_phase = my_phase;
  }
  __syncthreads();
  my_phase = s_my_phase;

  // 2-buffered slot pair selected by phase parity; slot within pair indexed
  // by source PE (this PE's world mype).
  long volatile *pSync = task_pSync_base + (my_phase & 1) * size;

  // P2P-connected teams use k = size (full-radix dissemination). For TP=4
  // that's 3 signals + 3 waits per phase, 1 phase total. Private slots per
  // task eliminate contention across the 56 callers. Run on lane 0 only —
  // work is latency-bound by NVLink P2P round-trip (~1 μs each).
  int const my_pe = teami->my_pe;
  int const world_my_pe = nvshmemi_device_state_d.mype;
  if (threadIdx.x == 0) {
    int k = size;
    for (int j = 1; j < k; j++) {
      int to_nbr_idx = my_pe + j;
      if (to_nbr_idx >= size) {
        to_nbr_idx -= size;
      }
      int to_nbr = mpkar_team_translate_pe(teami, to_nbr_idx);
      mpkar_signal_for_barrier(
          (long *)(pSync + world_my_pe), my_phase, to_nbr);
    }
    for (int j = 1; j < k; j++) {
      int from_nbr_idx = my_pe - j;
      if (from_nbr_idx < 0) {
        from_nbr_idx += size;
      }
      int from_nbr = mpkar_team_translate_pe(teami, from_nbr_idx);
      mpkar_wait_until_ge(pSync + from_nbr, my_phase);
    }
  }
  __syncthreads();
}

// ========================= NVLS multicast pointer ===========================
static __device__ __forceinline__ void *mpkar_mc_ptr(nvshmemi_team_t *team,
                                                     void const *ptr) {
  if (team == nullptr || team->nvls_rsc_base_ptr == nullptr) {
    return nullptr;
  }
  ptrdiff_t offset = (char *)ptr - (char *)nvshmemi_device_state_d.heap_base;
  if (ptr >= nvshmemi_device_state_d.heap_base &&
      offset < (ptrdiff_t)nvshmemi_device_state_d.heap_size &&
      team->nvls_rsc_base_ptr != nullptr) {
    void *mc_addr =
        (void *)__ldg((long long unsigned const *)team->nvls_rsc_base_ptr);
    if (mc_addr != nullptr) {
      mc_addr = (void *)((char *)mc_addr + offset);
    }
    return mc_addr;
  }
  return nullptr;
}

static __device__ __forceinline__ void *mpkar_peer_ptr(void const *ptr,
                                                       int pe) {
  char const *base =
      static_cast<char const *>(nvshmemi_device_state_d.heap_base);
  char const *addr = static_cast<char const *>(ptr);
  ptrdiff_t offset = addr - base;
  void const *peer_base_addr = (void *)__ldg(
      (long long unsigned const *)nvshmemi_device_state_d.peer_heap_base_p2p +
      pe);
  return (void *)(static_cast<char const *>(peer_base_addr) + offset);
}

// ========================= NVLS ld_reduce PTX ===============================
// bf16: multimem.ld_reduce.global.add.acc::f32.v4.bf16x2
// Loads 16 bytes (8 bf16 values) from multicast address, reduces across GPUs.
static __device__ __forceinline__ void
    mpkar_nvls_ld_reduce_bf16_v4(uint32_t &r0,
                                 uint32_t &r1,
                                 uint32_t &r2,
                                 uint32_t &r3,
                                 int4 const *mc_addr) {
  asm("multimem.ld_reduce.global.add.acc::f32.v4.bf16x2 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "l"(mc_addr));
}

// f16: multimem.ld_reduce.global.add.acc::f32.v4.f16x2
static __device__ __forceinline__ void
    mpkar_nvls_ld_reduce_f16_v4(uint32_t &r0,
                                uint32_t &r1,
                                uint32_t &r2,
                                uint32_t &r3,
                                int4 const *mc_addr) {
  asm("multimem.ld_reduce.global.add.acc::f32.v4.f16x2 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "l"(mc_addr));
}

// f32: multimem.ld_reduce.global.add.v4.f32
static __device__ __forceinline__ void mpkar_nvls_ld_reduce_f32_v4(
    float &r0, float &r1, float &r2, float &r3, int4 const *mc_addr) {
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
static __device__ __forceinline__ void mpkar_nvls_reduce_v4_block(
    int4 *__restrict__ dst, int4 const *__restrict__ mc_src, int nelems_v4) {
  for (int j = threadIdx.x; j < nelems_v4; j += blockDim.x) {
    uint32_t u4[4];

    // Dispatch on element type
    if constexpr (sizeof(T) == 2) {
      // bf16 or f16 -- both use the same register type (uint32_t)
      // but different PTX instructions
      if constexpr (cuda::std::is_same<T, __nv_bfloat16>::value) {
        mpkar_nvls_ld_reduce_bf16_v4(u4[0], u4[1], u4[2], u4[3], mc_src + j);
      } else {
        // half / __half
        mpkar_nvls_ld_reduce_f16_v4(u4[0], u4[1], u4[2], u4[3], mc_src + j);
      }
      asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(dst + j),
          "r"(u4[0]),
          "r"(u4[1]),
          "r"(u4[2]),
          "r"(u4[3]));
    } else {
      // float
      float f4[4];
      mpkar_nvls_ld_reduce_f32_v4(f4[0], f4[1], f4[2], f4[3], mc_src + j);
      asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(dst + j),
          "r"(__float_as_uint(f4[0])),
          "r"(__float_as_uint(f4[1])),
          "r"(__float_as_uint(f4[2])),
          "r"(__float_as_uint(f4[3])));
    }
  }
}

static __device__ __forceinline__ uint32_t
    mpkar_add_bf16x2(uint32_t reduced_bits, uint32_t residual_bits) {
  union PackedBf16x2 {
    uint32_t bits;
    __nv_bfloat162 value;
  };

  PackedBf16x2 reduced{.bits = reduced_bits};
  PackedBf16x2 residual{.bits = residual_bits};
  float2 const reduced_vals = __bfloat1622float2(reduced.value);
  float2 const residual_vals = __bfloat1622float2(residual.value);
  PackedBf16x2 out;
  out.value = __float22bfloat162_rn(make_float2(
      reduced_vals.x + residual_vals.x, reduced_vals.y + residual_vals.y));
  return out.bits;
}

// Same NVLS reduce as above, but adds a local residual before the final store.
// This is the safe fusion point for tensor-parallel MoE residuals: every rank
// contributes only its partial MLP output to NVLS, and residual is added once
// after the cross-rank reduction has completed.
template <typename T>
static __device__ __forceinline__ void
    mpkar_nvls_reduce_add_residual_v4_block(int4 *__restrict__ dst,
                                            int4 const *__restrict__ mc_src,
                                            int4 const *__restrict__ residual,
                                            int nelems_v4) {
  for (int j = threadIdx.x; j < nelems_v4; j += blockDim.x) {
    if constexpr (sizeof(T) == 2) {
      if constexpr (cuda::std::is_same<T, __nv_bfloat16>::value) {
        uint32_t u4[4];
        mpkar_nvls_ld_reduce_bf16_v4(u4[0], u4[1], u4[2], u4[3], mc_src + j);
        int4 const r4 = residual[j];
        u4[0] = mpkar_add_bf16x2(u4[0], static_cast<uint32_t>(r4.x));
        u4[1] = mpkar_add_bf16x2(u4[1], static_cast<uint32_t>(r4.y));
        u4[2] = mpkar_add_bf16x2(u4[2], static_cast<uint32_t>(r4.z));
        u4[3] = mpkar_add_bf16x2(u4[3], static_cast<uint32_t>(r4.w));
        asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(dst + j),
            "r"(u4[0]),
            "r"(u4[1]),
            "r"(u4[2]),
            "r"(u4[3]));
      } else {
        // The persistent DeepSeek path instantiates bf16 only. Keep the
        // non-bf16 path unfused instead of silently doing the wrong conversion.
        uint32_t u4[4];
        mpkar_nvls_ld_reduce_f16_v4(u4[0], u4[1], u4[2], u4[3], mc_src + j);
        asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(dst + j),
            "r"(u4[0]),
            "r"(u4[1]),
            "r"(u4[2]),
            "r"(u4[3]));
      }
    } else {
      float f4[4];
      mpkar_nvls_ld_reduce_f32_v4(f4[0], f4[1], f4[2], f4[3], mc_src + j);
      int4 const r4 = residual[j];
      f4[0] += __uint_as_float(static_cast<uint32_t>(r4.x));
      f4[1] += __uint_as_float(static_cast<uint32_t>(r4.y));
      f4[2] += __uint_as_float(static_cast<uint32_t>(r4.z));
      f4[3] += __uint_as_float(static_cast<uint32_t>(r4.w));
      asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(dst + j),
          "r"(__float_as_uint(f4[0])),
          "r"(__float_as_uint(f4[1])),
          "r"(__float_as_uint(f4[2])),
          "r"(__float_as_uint(f4[3])));
    }
  }
}

// ========================= public API =======================================
// Drop-in replacement for the old nvshmem_tile_allreduce, plus a residual
// variant used to fuse MoE allreduce + residual add.
//
// Template params:
//   T            - element type (__nv_bfloat16, half, float)
//   BATCH_SIZE   - unused (kept for API compat)
//   OUTPUT_SIZE  - contiguous dimension in elements
//   OUTPUT_STRIDE - stride of minor dimension in elements
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE,
          bool ADD_RESIDUAL>
__device__ __forceinline__ void nvshmem_tile_allreduce_impl(void *input_ptr,
                                                            void *residual_ptr,
                                                            void *output_ptr,
                                                            void *_teams,
                                                            int task_offset,
                                                            int active_tokens) {
  nvshmem_team_t *teams = reinterpret_cast<nvshmem_team_t *>(_teams);
  nvshmem_team_t team = teams[task_offset];
  int const num_active_rows = max(0, min(active_tokens, BATCH_SIZE));

  // --- Phase 1: ensure local data is visible, then cross-GPU barrier ---
  __threadfence();
#ifndef MPK_AR_SKIP_BARRIER
#if defined(MPK_AR_PER_TASK_BARRIER)
  // Contention-free per-task slot version (2026-05-12). See
  // `mpkar_sync_block_per_task` comment for stationarity precondition.
  mpkar_sync_block_per_task(team, task_offset);
#else
  mpkar_sync_block(team);
#endif
#endif

  // --- Phase 2: NVLS multicast ld_reduce -> local store ---
  nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
  void *mc_src = mpkar_mc_ptr(teami, input_ptr);
#ifdef MPK_AR_SKIP_REDUCE
  // Skip NVLS reduce entirely; just copy input → output (per-PE, no AR).
  // Used only for measuring barrier-vs-reduce cost; produces wrong output.
  {
    int4 *dst_v4_dbg = reinterpret_cast<int4 *>(output_ptr);
    int4 const *src_v4_dbg = reinterpret_cast<int4 const *>(input_ptr);
    constexpr int ELEMS_PER_V4_DBG = 16 / sizeof(T);
    constexpr int V4_PER_ROW_DBG = OUTPUT_SIZE / ELEMS_PER_V4_DBG;
    int const total_v4_dbg = V4_PER_ROW_DBG * num_active_rows;
    for (int j = threadIdx.x; j < total_v4_dbg; j += blockDim.x) {
      dst_v4_dbg[j] = src_v4_dbg[j];
    }
    __threadfence();
    __syncthreads();
    return;
  }
#endif

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

  constexpr int ELEMS_PER_V4 = 16 / sizeof(T); // 8 for bf16, 4 for f32
  constexpr int V4_PER_ROW = OUTPUT_SIZE / ELEMS_PER_V4;
  constexpr int STRIDE_V4 = OUTPUT_STRIDE / ELEMS_PER_V4;

  int4 *dst_v4 = reinterpret_cast<int4 *>(output_ptr);
  int4 const *src_mc_v4 = reinterpret_cast<int4 const *>(mc_src);
  int4 const *residual_v4 = reinterpret_cast<int4 const *>(residual_ptr);

  if constexpr (OUTPUT_SIZE == OUTPUT_STRIDE) {
    // Contiguous: one pass over all rows
    int total_v4 = V4_PER_ROW * num_active_rows;
    if constexpr (ADD_RESIDUAL) {
      mpkar_nvls_reduce_add_residual_v4_block<T>(
          dst_v4, src_mc_v4, residual_v4, total_v4);
    } else {
      mpkar_nvls_reduce_v4_block<T>(dst_v4, src_mc_v4, total_v4);
    }
  } else {
    // Strided: per-row
    for (int row = 0; row < num_active_rows; row++) {
      if constexpr (ADD_RESIDUAL) {
        mpkar_nvls_reduce_add_residual_v4_block<T>(dst_v4 + row * STRIDE_V4,
                                                   src_mc_v4 + row * STRIDE_V4,
                                                   residual_v4 +
                                                       row * STRIDE_V4,
                                                   V4_PER_ROW);
      } else {
        mpkar_nvls_reduce_v4_block<T>(
            dst_v4 + row * STRIDE_V4, src_mc_v4 + row * STRIDE_V4, V4_PER_ROW);
      }
    }
  }

  // --- Phase 3: ensure PULL stores are visible locally ---
  // PULL variant stores only to local HBM, so __threadfence() (not
  // __threadfence_system()) is sufficient.
  __threadfence();
  __syncthreads();
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__device__ __forceinline__ void nvshmem_tile_allreduce(void *input_ptr,
                                                       void *output_ptr,
                                                       void *_teams,
                                                       int task_offset,
                                                       int active_tokens) {
  nvshmem_tile_allreduce_impl<T, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_STRIDE, false>(
      input_ptr, nullptr, output_ptr, _teams, task_offset, active_tokens);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__device__ __forceinline__ void
    nvshmem_tile_allreduce_with_residual(void *input_ptr,
                                         void *residual_ptr,
                                         void *output_ptr,
                                         void *_teams,
                                         int task_offset,
                                         int active_tokens) {
  nvshmem_tile_allreduce_impl<T, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_STRIDE, true>(
      input_ptr, residual_ptr, output_ptr, _teams, task_offset, active_tokens);
}

} // namespace kernel

#endif // USE_NVSHMEM
