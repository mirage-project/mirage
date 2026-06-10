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

// MPK linear task — device PTX helpers (namespace kernel::linear).
//
// These are the single-thread / warp PTX wrappers the linear role functions
// use. This header is the single authority for them; the role functions in
// linear_sm100_v2.cuh pull them from kernel::linear.
//
// Device-only (contains __device__ PTX) — NOT host-includable. The host-safe
// constants live in linear_spec.h, which this header includes.

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cstdint>

#include "mirage/persistent_kernel/tasks/blackwell_v2/linear_spec.h"

namespace kernel {
namespace linear {

template <typename T>
__device__ inline T warp_uniform(T x) {
  return __shfl_sync(0xFFFFFFFF, x, 0);
}

__device__ inline uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile("{\n\t.reg .pred %%px;\n\t"
               "elect.sync _|%%px, %1;\n\t"
               "@%%px mov.s32 %0, 1;\n\t}"
               : "+r"(pred)
               : "r"(0xFFFFFFFF));
  return pred;
}

__device__ inline void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(mbar_addr),
               "r"(count));
}

__device__ inline void mbarrier_wait(int mbar_addr, int phase) {
  asm volatile("{\n\t.reg .pred P1;\n\t"
               "LAB_WAIT:\n\t"
               "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], "
               "%1, 0x989680;\n\t"
               "@P1 bra.uni DONE;\n\t"
               "bra.uni LAB_WAIT;\n\t"
               "DONE:\n\t}" ::"r"(mbar_addr),
               "r"(phase));
}

__device__ inline void mbarrier_arrive_expect_tx(int mbar_addr, int size) {
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::
          "r"(mbar_addr),
      "r"(size)
      : "memory");
}

__device__ inline void mbarrier_arrive(int mbar_addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(
      mbar_addr)
               : "memory");
}

__device__ inline void tma_3d_load_l2(int dst, const void *tmap_ptr, int x,
                                      int y, int z, int mbar_addr,
                                      uint64_t hint) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global."
               "mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
               "[%0], [%1, {%2, %3, %4}], [%5], %6;" ::"r"(dst),
               "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "l"(hint)
               : "memory");
}

__device__ inline void tcgen05_commit(int mbar_addr) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
               "cluster.b64 [%0];" ::"r"(mbar_addr)
               : "memory");
}

__device__ inline void tcgen05_mma(int taddr, uint64_t a_desc, uint64_t b_desc,
                                   uint32_t idesc, int enable_d) {
  asm volatile("{\n\t.reg .pred p;\n\t"
               "setp.ne.b32 p, %4, 0;\n\t"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}" ::"r"(
                   taddr),
               "l"(a_desc), "l"(b_desc), "r"(idesc), "r"(enable_d));
}

// ── Table-driven semaphore init / re-init (driven by CHANNELS) ──────────────
// Device code can't runtime-iterate the host `constexpr` table (odr-use →
// "undefined in device code"), so we access it ONLY via constant channel
// indices with forced-constexpr field extraction — the compiler folds these to
// literals, no array instance needed. The ordinals/counts/policy still live in
// exactly one place (linear_spec.h); these macros just expand it per channel.
//
// _b/_a/_d are constexpr locals → guaranteed compile-time, no array storage.
#define LIN_INIT_FULL(dyn, K)                                                  \
  do {                                                                         \
    constexpr int _d = CHANNELS[K].depth, _b = CHANNELS[K].full_sem_base,      \
                  _a = CHANNELS[K].full_arrivals;                              \
    for (int s = 0; s < _d; s++) mbarrier_init((dyn) + (_b + s) * 8, _a);      \
  } while (0)
#define LIN_INIT_EMPTY(dyn, K)                                                 \
  do {                                                                         \
    constexpr int _d = CHANNELS[K].depth, _b = CHANNELS[K].empty_sem_base,     \
                  _a = CHANNELS[K].empty_arrivals,                             \
                  _sh = CHANNELS[K].shares_empty_with;                         \
    if (_sh < 0)                                                               \
      for (int s = 0; s < _d; s++) mbarrier_init((dyn) + (_b + s) * 8, _a);    \
  } while (0)

// linear_init: full structural init of all op-private mbars for one task; run
// once by the dispatcher (init_semaphores) before the task is published.
// dyn_sem_base = op_sem_base_addr(...) (SMEM addr of this slot's SEM_OP_BASE;
// ordinal k at +k*8).
__device__ __forceinline__ void linear_init(int dyn_sem_base) {
  LIN_INIT_FULL(dyn_sem_base, CH_W);   LIN_INIT_EMPTY(dyn_sem_base, CH_W);
  LIN_INIT_FULL(dyn_sem_base, CH_A);   LIN_INIT_EMPTY(dyn_sem_base, CH_A);
  LIN_INIT_FULL(dyn_sem_base, CH_ACC); LIN_INIT_EMPTY(dyn_sem_base, CH_ACC);
  mbarrier_init(dyn_sem_base + ONESHOT_SEMS[0].sem * 8, ONESHOT_SEMS[0].arrivals);
  mbarrier_init(dyn_sem_base + ONESHOT_SEMS[1].sem * 8, ONESHOT_SEMS[1].arrivals);
  asm volatile("fence.mbarrier_init.release.cluster;");
}

// reinit_for_role: re-init only the edges this role owns at task start (per
// reinit_*_by / OneShotSem.reinit_by), clearing prior-slot async strays. Called
// single-threaded by the owning role (loader's elected lane; mma lane 0).
// MUST be __forceinline__: as a real call, nvcc reorders the mbarrier-init
// fence relative to the surrounding TMA/MMA issue, re-exposing the stale-arrival
// race. Inlining keeps the fence ordered as written.
#define LIN_REINIT_FULL(dyn, K, r)                                             \
  do {                                                                         \
    if (CHANNELS[K].reinit_full_by == (r)) LIN_INIT_FULL(dyn, K);              \
  } while (0)
#define LIN_REINIT_EMPTY(dyn, K, r)                                            \
  do {                                                                         \
    if (CHANNELS[K].reinit_empty_by == (r)) LIN_INIT_EMPTY(dyn, K);            \
  } while (0)
__device__ __forceinline__ void reinit_for_role(Role r, int dyn_sem_base) {
  LIN_REINIT_FULL(dyn_sem_base, CH_W, r);   LIN_REINIT_EMPTY(dyn_sem_base, CH_W, r);
  LIN_REINIT_FULL(dyn_sem_base, CH_A, r);   LIN_REINIT_EMPTY(dyn_sem_base, CH_A, r);
  LIN_REINIT_FULL(dyn_sem_base, CH_ACC, r); LIN_REINIT_EMPTY(dyn_sem_base, CH_ACC, r);
  if (ONESHOT_SEMS[0].reinit_by == r)
    mbarrier_init(dyn_sem_base + ONESHOT_SEMS[0].sem * 8, ONESHOT_SEMS[0].arrivals);
  if (ONESHOT_SEMS[1].reinit_by == r)
    mbarrier_init(dyn_sem_base + ONESHOT_SEMS[1].sem * 8, ONESHOT_SEMS[1].arrivals);
  asm volatile("fence.mbarrier_init.release.cluster;");
}
#undef LIN_REINIT_FULL
#undef LIN_REINIT_EMPTY
#undef LIN_INIT_FULL
#undef LIN_INIT_EMPTY

}  // namespace linear
}  // namespace kernel
