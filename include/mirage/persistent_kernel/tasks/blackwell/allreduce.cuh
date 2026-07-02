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

// multimem.st.global.v4.b32 -- broadcast 16 bytes to the multicast address.
// Used by the reduce-scatter+all-gather variant for the all-gather phase.
static __device__ __forceinline__ void mpkar_nvls_st_v4(
    int4 const *mc_addr, uint32_t v0, uint32_t v1, uint32_t v2, uint32_t v3) {
  asm("multimem.st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(mc_addr),
      "r"(v0),
      "r"(v1),
      "r"(v2),
      "r"(v3));
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

// ========================= P2P fallback reduce (no NVLS) =====================
// Used when the team has no NVLS multicast resource — observed on NVSHMEM
// 3.6.5 under MPK's team_split_strided setup where every team has
// `nvls_rsc_base_ptr == NULL` on the device side (see TP=2 debug session
// Reads this PE's local copy + every peer's heap mirror via
// `peer_heap_base_p2p`, sums in bf16, stores. bf16 only (matches DSv3 AR
// instantiation).
template <typename T>
static __device__ __forceinline__ void
    mpkar_p2p_reduce_v4_block(int4 *__restrict__ dst,
                              void const *local_input_ptr,
                              nvshmemi_team_t *teami,
                              int nelems_v4) {
  static_assert(cuda::std::is_same<T, __nv_bfloat16>::value,
                "P2P AR fallback currently bf16 only.");
  int4 const *local_v4 = reinterpret_cast<int4 const *>(local_input_ptr);
  int const npes = teami->size;
  for (int j = threadIdx.x; j < nelems_v4; j += blockDim.x) {
    int4 acc = local_v4[j];
    for (int p = 0; p < npes; p++) {
      int peer_world_pe = teami->pe_mapping[p];
      if (peer_world_pe == nvshmemi_device_state_d.mype) {
        continue;
      }
      int4 *peer_v4 = reinterpret_cast<int4 *>(
          mpkar_peer_ptr(local_input_ptr, peer_world_pe));
      int4 const peer_val = peer_v4[j];
      acc.x = mpkar_add_bf16x2(static_cast<uint32_t>(acc.x),
                               static_cast<uint32_t>(peer_val.x));
      acc.y = mpkar_add_bf16x2(static_cast<uint32_t>(acc.y),
                               static_cast<uint32_t>(peer_val.y));
      acc.z = mpkar_add_bf16x2(static_cast<uint32_t>(acc.z),
                               static_cast<uint32_t>(peer_val.z));
      acc.w = mpkar_add_bf16x2(static_cast<uint32_t>(acc.w),
                               static_cast<uint32_t>(peer_val.w));
    }
    dst[j] = acc;
  }
}

template <typename T>
static __device__ __forceinline__ void
    mpkar_p2p_reduce_add_residual_v4_block(int4 *__restrict__ dst,
                                           void const *local_input_ptr,
                                           int4 const *__restrict__ residual,
                                           nvshmemi_team_t *teami,
                                           int nelems_v4) {
  static_assert(cuda::std::is_same<T, __nv_bfloat16>::value,
                "P2P AR fallback currently bf16 only.");
  int4 const *local_v4 = reinterpret_cast<int4 const *>(local_input_ptr);
  int const npes = teami->size;
  for (int j = threadIdx.x; j < nelems_v4; j += blockDim.x) {
    int4 acc = local_v4[j];
    for (int p = 0; p < npes; p++) {
      int peer_world_pe = teami->pe_mapping[p];
      if (peer_world_pe == nvshmemi_device_state_d.mype) {
        continue;
      }
      int4 *peer_v4 = reinterpret_cast<int4 *>(
          mpkar_peer_ptr(local_input_ptr, peer_world_pe));
      int4 const peer_val = peer_v4[j];
      acc.x = mpkar_add_bf16x2(static_cast<uint32_t>(acc.x),
                               static_cast<uint32_t>(peer_val.x));
      acc.y = mpkar_add_bf16x2(static_cast<uint32_t>(acc.y),
                               static_cast<uint32_t>(peer_val.y));
      acc.z = mpkar_add_bf16x2(static_cast<uint32_t>(acc.z),
                               static_cast<uint32_t>(peer_val.z));
      acc.w = mpkar_add_bf16x2(static_cast<uint32_t>(acc.w),
                               static_cast<uint32_t>(peer_val.w));
    }
    int4 const r4 = residual[j];
    acc.x = mpkar_add_bf16x2(static_cast<uint32_t>(acc.x),
                             static_cast<uint32_t>(r4.x));
    acc.y = mpkar_add_bf16x2(static_cast<uint32_t>(acc.y),
                             static_cast<uint32_t>(r4.y));
    acc.z = mpkar_add_bf16x2(static_cast<uint32_t>(acc.z),
                             static_cast<uint32_t>(r4.z));
    acc.w = mpkar_add_bf16x2(static_cast<uint32_t>(acc.w),
                             static_cast<uint32_t>(r4.w));
    dst[j] = acc;
  }
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

template <int WORLD_SIZE>
static __device__ __forceinline__ void
    mpkar_nvls_rs_ag_bf16_v4_block(int4 *__restrict__ dst,
                                   int4 const *__restrict__ mc_src,
                                   int4 const *__restrict__ mc_dst,
                                   int stride_v4,
                                   int v4_per_row,
                                   int num_rows,
                                   int my_rank) {
  static_assert(WORLD_SIZE > 0, "WORLD_SIZE must be positive.");
  int const slice_v4 = v4_per_row / WORLD_SIZE;
  int const slice_start = my_rank * slice_v4;
  int const slice_end = slice_start + slice_v4;

  // Phase RS: this rank reduces only its owned slice into the local output.
  for (int row = 0; row < num_rows; row++) {
    int const row_base = row * stride_v4;
    for (int col = threadIdx.x + slice_start; col < slice_end;
         col += blockDim.x) {
      uint32_t u4[4];
      int const idx = row_base + col;
      mpkar_nvls_ld_reduce_bf16_v4(u4[0], u4[1], u4[2], u4[3], mc_src + idx);
      asm("st.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(dst + idx),
          "r"(u4[0]),
          "r"(u4[1]),
          "r"(u4[2]),
          "r"(u4[3]));
    }
  }

  // Phase AG: broadcast this rank's reduced slice to every peer's output.
  for (int row = 0; row < num_rows; row++) {
    int const row_base = row * stride_v4;
    for (int col = threadIdx.x + slice_start; col < slice_end;
         col += blockDim.x) {
      int const idx = row_base + col;
      int4 const val = dst[idx];
      mpkar_nvls_st_v4(mc_dst + idx,
                       static_cast<uint32_t>(val.x),
                       static_cast<uint32_t>(val.y),
                       static_cast<uint32_t>(val.z),
                       static_cast<uint32_t>(val.w));
    }
  }

  // Make the multicast stores visible before the caller's cross-rank barrier.
  __threadfence_system();
}

static __device__ __forceinline__ void
    mpkar_add_residual_bf16_v4_block(int4 *__restrict__ dst,
                                     int4 const *__restrict__ residual,
                                     int stride_v4,
                                     int v4_per_row,
                                     int num_rows) {
  for (int row = 0; row < num_rows; row++) {
    int const row_base = row * stride_v4;
    for (int col = threadIdx.x; col < v4_per_row; col += blockDim.x) {
      int const idx = row_base + col;
      int4 acc = dst[idx];
      int4 const r4 = residual[idx];
      acc.x = mpkar_add_bf16x2(static_cast<uint32_t>(acc.x),
                               static_cast<uint32_t>(r4.x));
      acc.y = mpkar_add_bf16x2(static_cast<uint32_t>(acc.y),
                               static_cast<uint32_t>(r4.y));
      acc.z = mpkar_add_bf16x2(static_cast<uint32_t>(acc.z),
                               static_cast<uint32_t>(r4.z));
      acc.w = mpkar_add_bf16x2(static_cast<uint32_t>(acc.w),
                               static_cast<uint32_t>(r4.w));
      dst[idx] = acc;
    }
  }
}

// Small dispatch wrapper reused by both the baseline and per-tile paths so the
// reduce body stays a single source of truth.
template <typename T, bool ADD_RESIDUAL>
static __device__ __forceinline__ void
    mpkar_nvls_reduce_add_residual_or_plain(int4 *__restrict__ dst,
                                            int4 const *__restrict__ mc_src,
                                            int4 const *__restrict__ residual,
                                            int nelems_v4) {
  if constexpr (ADD_RESIDUAL) {
    mpkar_nvls_reduce_add_residual_v4_block<T>(
        dst, mc_src, residual, nelems_v4);
  } else {
    mpkar_nvls_reduce_v4_block<T>(dst, mc_src, nelems_v4);
  }
}

// ===========================================================================
// ============ Candidate-2: NVLS one-shot + per-tile arrival gate ============
// ===========================================================================
// Env-gated by MPK_DSV3_AR_NVLS_PERTILE (default-OFF ⇒ default build is
// byte-identical). Mechanism (Codex-scoped thread 019f1980 / hang-safety
// cross-checked 019f1f97 + 019f1f9d): the measured AR slowCTA imbalance
// (fast ~6us, slow ~11.5us, rotating straggler) is dominated by the radix-8
// dissemination barrier (mpkar_sync_dissem_pow2_block<8,3>) blocking on the
// globally-slowest-arriving rank FOR THAT SLICE. The NVLS multimem.ld_reduce
// body is near-free. This variant KEEPS the HW ld_reduce but REPLACES the
// radix-8 dissemination relay with a FLAT (direct, single-phase) per-CTA
// per-PE arrival gate — the vLLM barrier_at_start/barrier_at_end pattern,
// specialized to NVLS-pull semantics.
//
// WHY TWO GATES (not opening-only): the AR input buffer `allreduce_buf` is a
// SINGLE symmetric allocation REUSED across all 61 layers x 512 steps (NOT
// per-layer fresh). With NVLS pull, a fast rank would overwrite its slice for
// call k+1 while a slow peer is still ld_reduce-ing call k's data. The
// baseline dodges this because its opening dissemination barrier of k+1
// transitively closes k (a rank can't reach the k+1 barrier until it passed
// k's reduce). A single flat opening flag is data-ready (post-write) and
// therefore CANNOT also be pre-overwrite, so it cannot transitively close the
// previous call (Codex 019f1f9d Q4). Hence we add an explicit CLOSING gate:
//   START gate: publish "my slice input is globally visible" (epoch), wait all
//               8 peers ⇒ safe to NVLS-read the reduced multicast.
//   END   gate: publish "my ld_reduce of this epoch has completed", wait all 8
//               peers ⇒ safe for ANY rank to overwrite its slice for k+1.
//
// HANG-SAFETY (each risk from the wiki's persistent-megakernel list + Codex):
//   * flag-before-data:  __syncthreads() + __threadfence_system() BEFORE the
//     start-flag store; the store itself is st.release.sys → data
//     happens-before flag, visible to peer GPUs over NVLink (device-scope
//     __threadfence()/ld.acquire.gpu are TOO WEAK for peer devices — Codex Q1a).
//   * weak-ld_reduce-as-readiness: readiness comes ONLY from the all-8 acquire
//     of start-flags; the ld_reduce is never treated as its own sync (Q1b).
//   * ABA / epoch reuse: strictly-monotonic 64-bit per-(CTA,gate) epoch, wait
//     >= epoch. start[] and end[] are SEPARATE arrays so start[k+1] can never
//     be misread as end[k] (Q2). 64-bit never wraps over a run.
//   * source-buffer lifetime: the END gate's release/acquire chain
//     (B's ld_reduce(k) -> B release end(k) -> A acquire end(k) -> A
//     overwrite(k+1)) makes B's synchronous ld_reduce complete-before A's
//     reuse. multimem.ld_reduce is a SYNCHRONOUS register-dest load (not an
//     async op), so after the instruction + __syncthreads() + fence the read
//     is complete, not merely issued (Q1). PTX causality then forbids the
//     stale read (Q1).
//   * spin-starvation / co-scheduling: same CTA publishes its OWN flag
//     UNCONDITIONALLY before it spins on peers ⇒ no same-GPU
//     producer-behind-consumer; peers are on OTHER GPUs (Q1d). Requires all
//     grid CTAs resident (persistent megakernel — true here).
//   * unconditional progress / no circular wait: every rank publishes both
//     flags for every (CTA,epoch) before waiting (signal-all-then-wait-all is
//     deadlock-free) (Q1e). Any skipped CTA/rank/epoch ⇒ hard hang, same as
//     the baseline dissemination barrier.
//
// HONEST LEVER BOUND (wiki bounded-claim note + Codex Q2/Q4): the flat gate
// CANNOT remove the causal max_r ready_time[r][slice] slowest-peer term — that
// is a lower bound no algorithm removes. Since the grid is ALREADY per-CTA and
// each 128-elem slice is one upstream matmul burst (no intra-slice temporal
// ordering at bs=1), intra-CTA sub-chunking would decouple NOTHING. The ONLY
// win is (radix-relay depth 3 + pSync coupling) − (2 flat 1-phase gates +
// their 2*(P-1)=14 remote flag stores/CTA fanout). This may be a WIN if the
// relay latency dominates, NULL/REGRESS if the extra closing gate + fanout
// dominates. This is a genuine microbench question — do NOT pre-claim a win.
// ===========================================================================
#ifdef MPK_DSV3_AR_NVLS_PERTILE

// One 64-bit epoch counter per (CTA, PE) per gate. Laid out as:
//   flags[ (cta * NPES + pe) ]                    -> start-gate slot
//   flags[ START_REGION + (cta * NPES + pe) ]     -> end-gate slot
// where START_REGION = grid_size * NPES. Lives in the NVSHMEM symmetric heap
// (so P2P peer stores resolve) and is zero-initialized at kernel init; epochs
// start at 1 so the wait `>= 1` never spuriously passes on the zeroed slots.
// Provided by the builder as runtime_config.ar_pertile_flags (a symmetric
// tensor); see task_register.cc / builder.py wiring notes.

// Publish `epoch` into peer `pe`'s flag slot `slot_idx` with release.sys.
static __device__ __forceinline__ void
    mpkar_pt_flag_release(unsigned long long *flags_base_local,
                          long slot_idx,
                          unsigned long long epoch,
                          int pe) {
  // Resolve the peer's mirror of our local symmetric flags buffer.
  void const *peer_base_addr = (void *)__ldg(
      (long long unsigned const *)nvshmemi_device_state_d.peer_heap_base_p2p +
      pe);
  unsigned long long *peer_flags =
      (unsigned long long *)((char *)(peer_base_addr) +
                             ((char *)flags_base_local -
                              (char *)(nvshmemi_device_state_d.heap_base)));
  unsigned long long *dst = peer_flags + slot_idx;
  // st.release.sys: release-orders all prior stores/fences (incl. the data
  // __threadfence_system()) before this flag becomes visible to peer GPUs.
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(dst),
               "l"(epoch)
               : "memory");
}

// Acquire-load our own local flag slot (written by peer `pe`).
static __device__ __forceinline__ unsigned long long
    mpkar_pt_flag_acquire(unsigned long long const *flags_base_local,
                          long slot_idx) {
  unsigned long long v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(v)
               : "l"(flags_base_local + slot_idx)
               : "memory");
  return v;
}

// One flat all-to-all gate: publish `epoch` to every peer's slot for this
// (CTA, my_team_rank) in `region`, then spin until every peer's slot for this
// (CTA, r) has reached `epoch`. Elected single thread (tid 0) does the I/O;
// caller wraps with __syncthreads() on both sides.
//
// SLOT INDEXING (Codex 019f1fad Q3.1 fix): the publisher writes into the
// PEER's slot at index = MY TEAM-RANK (0..npes-1), and every reader reads its
// own slot at index r (team-rank r). Both sides use TEAM-RANK, never the raw
// global PE id, so the design is correct for strided/non-identity teams
// (TP8 EP2 uses strided teams where pe_mapping[r] != r in general).
//
// PROXY/ALIAS NOTE (Codex 019f1fad Q2): this uses the SAME
// peer_heap_base_p2p + offset P2P mapping as the production-validated baseline
// barrier (mpkar_signal_for_barrier), with strictly-STRONGER ordering
// (st.release.sys / ld.acquire.sys vs the baseline's plain volatile). If the
// baseline's cross-PE message passing is correct on this box (it is — shipped
// AR), this pair is correct a fortiori: identical mapping, stronger fences.
static __device__ __forceinline__ void
    mpkar_pt_flat_gate(unsigned long long *flags_base_local,
                       long region_off,
                       int cta,
                       int npes,
                       nvshmemi_team_t *teami,
                       unsigned long long epoch) {
  int const my_pe = nvshmemi_device_state_d.mype;
  // My rank WITHIN this team (0..npes-1). team_pool teams are contiguous or
  // strided; my_pe == pe_mapping[my_team_rank].
  int my_team_rank = 0;
#pragma unroll 1
  for (int r = 0; r < npes; r++) {
    if (teami->pe_mapping[r] == my_pe) {
      my_team_rank = r;
      break;
    }
  }
  long const my_slot = region_off + (long)cta * npes + my_team_rank;
  // signal-all: write MY team-rank slot into every peer's (and my own) buffer.
  for (int r = 0; r < npes; r++) {
    int peer_world_pe = teami->pe_mapping[r];
    if (peer_world_pe == my_pe) {
      // local publish (no peer store needed, but keep the slot coherent)
      asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(
                       flags_base_local + my_slot),
                   "l"(epoch)
                   : "memory");
      continue;
    }
    mpkar_pt_flag_release(flags_base_local, my_slot, epoch, peer_world_pe);
  }
  // wait-all: my own local slots [cta, 0..npes) are filled by each peer's
  // release (peer of team-rank r wrote into slot index r).
  for (int r = 0; r < npes; r++) {
    long const slot = region_off + (long)cta * npes + r;
    while (mpkar_pt_flag_acquire(flags_base_local, slot) < epoch) {
      // spin — peer is on another GPU; no same-GPU starvation
    }
  }
}

// Per-tile NVLS one-shot with flat start+end arrival gates. Same reduce body
// as mpkar_nvls_reduce{,_add_residual}_v4_block, but the cross-rank rendezvous
// is the flat gate pair instead of the dissemination barrier.
template <typename T, bool ADD_RESIDUAL>
static __device__ __forceinline__ void mpkar_pt_reduce_v4_block(
    int4 *__restrict__ dst,
    int4 const *__restrict__ mc_src,
    int4 const *__restrict__ residual,
    int nelems_v4,
    unsigned long long *flags_base_local,
    long start_region_off,
    long end_region_off,
    int cta,
    nvshmemi_team_t *teami,
    unsigned long long epoch) {
  int const npes = teami->size;

  // --- data-ready fence: local slice globally visible to peer GPUs ---
  __syncthreads();
  __threadfence_system();

  // --- START gate (data-ready) ---
  if (threadIdx.x == 0) {
    mpkar_pt_flat_gate(
        flags_base_local, start_region_off, cta, npes, teami, epoch);
  }
  __syncthreads();

  // --- NVLS ld_reduce (all 8 contributions now globally visible) ---
  mpkar_nvls_reduce_add_residual_or_plain<T, ADD_RESIDUAL>(
      dst, mc_src, residual, nelems_v4);

  // --- reduce-complete fence: ld_reduce is synchronous; order it before the
  //     end-flag so peers observing end(epoch) know our read finished ---
  __syncthreads();
  __threadfence_system();

  // --- END gate (reduce-done ⇒ safe to reuse the buffer for epoch+1) ---
  if (threadIdx.x == 0) {
    mpkar_pt_flat_gate(
        flags_base_local, end_region_off, cta, npes, teami, epoch);
  }
  __syncthreads();
}

// Public per-tile impl. Same template signature as the baseline
// nvshmem_tile_allreduce_impl PLUS a flags pointer (symmetric heap). The
// codegen (task_register.cc) emits a call to THIS function only when the
// builder activated the per-tile path (env MPK_DSV3_AR_NVLS_PERTILE at build
// time), so the default task graph + codegen are byte-identical.
//
// Flag layout (per rank, in the symmetric flags buffer):
//   GRID   = OUTPUT_STRIDE / OUTPUT_SIZE      (number of AR CTAs; 56 for DSv3)
//   NPES   = teami->size                      (8 for TP8)
//   start slots: [0 .. GRID*NPES)             index (cta*NPES + pe)
//   end   slots: [GRID*NPES .. 2*GRID*NPES)   index GRID*NPES + (cta*NPES + pe)
//   total 2*GRID*NPES uint64 (7168 B for DSv3 TP8). Zero-init at kernel init;
//   epochs start at 1.
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE,
          bool ADD_RESIDUAL>
__device__ __forceinline__ void
    nvshmem_tile_allreduce_pertile_impl(void *input_ptr,
                                        void *residual_ptr,
                                        void *output_ptr,
                                        void *_teams,
                                        int task_offset,
                                        int active_tokens,
                                        void *flags_ptr) {
  nvshmem_team_t *teams = reinterpret_cast<nvshmem_team_t *>(_teams);
  nvshmem_team_t team = teams[task_offset];
  int const num_active_rows = max(0, min(active_tokens, BATCH_SIZE));

  nvshmemi_team_t *teami = nvshmemi_device_state_d.team_pool[team];
  void *mc_src = mpkar_mc_ptr(teami, input_ptr);

  static_assert(OUTPUT_SIZE % (16 / sizeof(T)) == 0,
                "OUTPUT_SIZE must be a multiple of 16/sizeof(T) for v4 NVLS");
  static_assert(OUTPUT_STRIDE % OUTPUT_SIZE == 0,
                "per-tile AR requires OUTPUT_STRIDE a multiple of OUTPUT_SIZE "
                "(grid = OUTPUT_STRIDE/OUTPUT_SIZE)");

  constexpr int ELEMS_PER_V4 = 16 / sizeof(T);
  constexpr int V4_PER_ROW = OUTPUT_SIZE / ELEMS_PER_V4;
  constexpr int STRIDE_V4 = OUTPUT_STRIDE / ELEMS_PER_V4;
  // Number of AR CTAs = grid.x, derived from the column partition.
  constexpr int GRID = OUTPUT_STRIDE / OUTPUT_SIZE;

  int4 *dst_v4 = reinterpret_cast<int4 *>(output_ptr);
  int4 const *src_mc_v4 = reinterpret_cast<int4 const *>(mc_src);
  int4 const *residual_v4 = reinterpret_cast<int4 const *>(residual_ptr);

  int const npes = teami->size;
  int const cta = task_offset; // this CTA's slice index (0..GRID-1)
  long const start_region_off = 0;
  long const end_region_off = (long)GRID * npes;

  // Monotonic epoch for this CTA: reuse the per-team sync_counter (already
  // incremented once per collective call — exactly a per-CTA monotonic
  // value). Read it, use (val+1) as this call's epoch so a zeroed flag
  // (epoch-0 slot) never spuriously satisfies wait >= epoch, then bump it so
  // the next call uses the next epoch.
  long volatile *sync_counter =
      (long volatile *)mpkar_team_get_sync_counter(teami);
  unsigned long long const epoch =
      (unsigned long long)(sync_counter[0]) + 1ULL;

  unsigned long long *flags_base_local =
      reinterpret_cast<unsigned long long *>(flags_ptr);

  // per-tile path is defined only for the NVLS + contiguous-per-row case; the
  // no-NVLS P2P fallback keeps the baseline dissemination barrier (handled by
  // the caller selecting the baseline impl when mc_src == nullptr).
  bool const use_nvls = (mc_src != nullptr);

  if (!use_nvls) {
    // Safety fallback: if this team has no NVLS resource, defer to the proven
    // baseline path (dissemination barrier + P2P reduce). Never silently run
    // the per-tile gate without the HW reduce it was designed around.
    __threadfence();
    mpkar_sync_block(team);
    if constexpr (OUTPUT_SIZE == OUTPUT_STRIDE) {
      int total_v4 = V4_PER_ROW * num_active_rows;
      if constexpr (ADD_RESIDUAL) {
        mpkar_p2p_reduce_add_residual_v4_block<T>(
            dst_v4, input_ptr, residual_v4, teami, total_v4);
      } else {
        mpkar_p2p_reduce_v4_block<T>(dst_v4, input_ptr, teami, total_v4);
      }
    } else {
      for (int row = 0; row < num_active_rows; row++) {
        void const *row_input = static_cast<char const *>(input_ptr) +
                                row * STRIDE_V4 * (int)sizeof(int4);
        if constexpr (ADD_RESIDUAL) {
          mpkar_p2p_reduce_add_residual_v4_block<T>(dst_v4 + row * STRIDE_V4,
                                                    row_input,
                                                    residual_v4 +
                                                        row * STRIDE_V4,
                                                    teami,
                                                    V4_PER_ROW);
        } else {
          mpkar_p2p_reduce_v4_block<T>(
              dst_v4 + row * STRIDE_V4, row_input, teami, V4_PER_ROW);
        }
      }
    }
    __threadfence();
    __syncthreads();
    // NOTE: mpkar_sync_block already bumped sync_counter[0] for its own pSync
    // double-buffering, so we do NOT bump again here (the fallback path does
    // not use per-tile epochs). Monotonicity of the epoch is preserved either
    // way; the flags buffer is simply unused on this no-NVLS branch.
    return;
  }

  // NVLS per-tile: START gate → ld_reduce → END gate, per row.
  if constexpr (OUTPUT_SIZE == OUTPUT_STRIDE) {
    int total_v4 = V4_PER_ROW * num_active_rows;
    mpkar_pt_reduce_v4_block<T, ADD_RESIDUAL>(dst_v4,
                                              src_mc_v4,
                                              residual_v4,
                                              total_v4,
                                              flags_base_local,
                                              start_region_off,
                                              end_region_off,
                                              cta,
                                              teami,
                                              epoch);
  } else {
    // Multi-row (mbt>1) is the prefill build; the per-tile lever targets bs=1
    // decode (num_active_rows==1). For safety we gate ALL rows behind a single
    // start/end epoch pair (correct, slightly coarser). At bs=1 this is one
    // row and identical to the fast path.
    __syncthreads();
    __threadfence_system();
    if (threadIdx.x == 0) {
      mpkar_pt_flat_gate(
          flags_base_local, start_region_off, cta, npes, teami, epoch);
    }
    __syncthreads();
    for (int row = 0; row < num_active_rows; row++) {
      if constexpr (ADD_RESIDUAL) {
        mpkar_nvls_reduce_add_residual_v4_block<T>(dst_v4 + row * STRIDE_V4,
                                                   src_mc_v4 + row * STRIDE_V4,
                                                   residual_v4 +
                                                       row * STRIDE_V4,
                                                   V4_PER_ROW);
      } else {
        mpkar_nvls_reduce_v4_block<T>(dst_v4 + row * STRIDE_V4,
                                      src_mc_v4 + row * STRIDE_V4,
                                      V4_PER_ROW);
      }
    }
    __syncthreads();
    __threadfence_system();
    if (threadIdx.x == 0) {
      mpkar_pt_flat_gate(
          flags_base_local, end_region_off, cta, npes, teami, epoch);
    }
    __syncthreads();
  }

  // Bump the per-team epoch counter for the next collective call.
  if (threadIdx.x == 0) {
    sync_counter[0] += 1;
  }
  __syncthreads();
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__device__ __forceinline__ void
    nvshmem_tile_allreduce_pertile(void *input_ptr,
                                   void *output_ptr,
                                   void *_teams,
                                   int task_offset,
                                   int active_tokens,
                                   void *flags_ptr) {
  nvshmem_tile_allreduce_pertile_impl<T,
                                      BATCH_SIZE,
                                      OUTPUT_SIZE,
                                      OUTPUT_STRIDE,
                                      false>(input_ptr,
                                             nullptr,
                                             output_ptr,
                                             _teams,
                                             task_offset,
                                             active_tokens,
                                             flags_ptr);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__device__ __forceinline__ void
    nvshmem_tile_allreduce_pertile_with_residual(void *input_ptr,
                                                 void *residual_ptr,
                                                 void *output_ptr,
                                                 void *_teams,
                                                 int task_offset,
                                                 int active_tokens,
                                                 void *flags_ptr) {
  nvshmem_tile_allreduce_pertile_impl<T,
                                      BATCH_SIZE,
                                      OUTPUT_SIZE,
                                      OUTPUT_STRIDE,
                                      true>(input_ptr,
                                            residual_ptr,
                                            output_ptr,
                                            _teams,
                                            task_offset,
                                            active_tokens,
                                            flags_ptr);
}

#endif // MPK_DSV3_AR_NVLS_PERTILE

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

  constexpr int ELEMS_PER_V4 = 16 / sizeof(T); // 8 for bf16, 4 for f32
  constexpr int V4_PER_ROW = OUTPUT_SIZE / ELEMS_PER_V4;
  constexpr int STRIDE_V4 = OUTPUT_STRIDE / ELEMS_PER_V4;

  int4 *dst_v4 = reinterpret_cast<int4 *>(output_ptr);
  int4 const *src_mc_v4 = reinterpret_cast<int4 const *>(mc_src);
  int4 const *residual_v4 = reinterpret_cast<int4 const *>(residual_ptr);

  // Pick NVLS path when the team has a multicast resource bound, otherwise
  // fall back to the P2P reduce that reads each peer's heap mirror directly.
  // mc_src is NULL whenever the team has no NVLS rsc (observed on NVSHMEM
  // 3.6.5 under MPK's team_split_strided setup — see TP=2 debug session
  // The fallback uses peer_heap_base_p2p, which is populated by
  // NVSHMEM for any p2p-connected GPU pair (no NVLS dependency).
  bool const use_nvls = (mc_src != nullptr);
  if constexpr (OUTPUT_SIZE == OUTPUT_STRIDE) {
    int total_v4 = V4_PER_ROW * num_active_rows;
    if constexpr (ADD_RESIDUAL) {
      if (use_nvls) {
        mpkar_nvls_reduce_add_residual_v4_block<T>(
            dst_v4, src_mc_v4, residual_v4, total_v4);
      } else {
        mpkar_p2p_reduce_add_residual_v4_block<T>(
            dst_v4, input_ptr, residual_v4, teami, total_v4);
      }
    } else {
      if (use_nvls) {
        mpkar_nvls_reduce_v4_block<T>(dst_v4, src_mc_v4, total_v4);
      } else {
        mpkar_p2p_reduce_v4_block<T>(dst_v4, input_ptr, teami, total_v4);
      }
    }
  } else {
    for (int row = 0; row < num_active_rows; row++) {
      void const *row_input = static_cast<char const *>(input_ptr) +
                              row * STRIDE_V4 * (int)sizeof(int4);
      if constexpr (ADD_RESIDUAL) {
        if (use_nvls) {
          mpkar_nvls_reduce_add_residual_v4_block<T>(
              dst_v4 + row * STRIDE_V4,
              src_mc_v4 + row * STRIDE_V4,
              residual_v4 + row * STRIDE_V4,
              V4_PER_ROW);
        } else {
          mpkar_p2p_reduce_add_residual_v4_block<T>(dst_v4 + row * STRIDE_V4,
                                                    row_input,
                                                    residual_v4 +
                                                        row * STRIDE_V4,
                                                    teami,
                                                    V4_PER_ROW);
        }
      } else {
        if (use_nvls) {
          mpkar_nvls_reduce_v4_block<T>(dst_v4 + row * STRIDE_V4,
                                        src_mc_v4 + row * STRIDE_V4,
                                        V4_PER_ROW);
        } else {
          mpkar_p2p_reduce_v4_block<T>(
              dst_v4 + row * STRIDE_V4, row_input, teami, V4_PER_ROW);
        }
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
