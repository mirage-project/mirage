/* Copyright 2025 CMU
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

namespace kernel {
struct Barrier {
private:
  uint64_t value;

public:
  __device__ inline uint64_t get_value() {
    return value;
  }
};

/*
mbarrier related functions
 */
__device__ static inline void initialize_barrier(
    Barrier &smem_barrier, // 64 bits user-manged barrier in smem
    int thread_count =
        1) // Thread count expected to arrive/wait on this barrier
{
#if defined(MIRAGE_GRACE_HOPPER)
  void const *const barrier_ptr = &smem_barrier;
  uint32_t smem_int_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier_ptr));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smem_int_ptr),
               "r"(thread_count));
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

__device__ static inline void set_barrier_transaction_bytes(
    Barrier &smem_barrier, // 64 bits user-manged barrier in smem
    uint32_t bytes)        // Number of bytes transfered by per TMA transaction
{
#if defined(MIRAGE_GRACE_HOPPER)
  if (lane_id() == 0) {
    void const *const barrier_ptr = &smem_barrier;
    uint32_t smem_int_ptr =
        static_cast<uint32_t>(__cvta_generic_to_shared(barrier_ptr));
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(
            smem_int_ptr),
        "r"(bytes));
  }
#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

__device__ static inline void wait(Barrier &smem_barrier, uint32_t phase) {
#if defined(MIRAGE_GRACE_HOPPER)
  void const *const ptr = &smem_barrier;
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
  asm volatile("{\n"
               ".reg .pred                P1;\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@P1                       bra.uni DONE;\n"
               "bra.uni                   LAB_WAIT;\n"
               "DONE:\n"
               "}\n" ::"r"(mbar_ptr),
               "r"(phase));

#elif defined(__CUDA_ARCH__)
  asm volatile("brkpt;\n" ::);
#endif
}

__device__ static inline void arrive(Barrier &smem_barrier,
                                     uint32_t count = 1) {
  void const *const barrier_ptr = &smem_barrier;
  uint32_t mbar_ptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(barrier_ptr));
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0], %1;\n"
               :
               : "r"(mbar_ptr), "r"(count)
               : "memory");
}
} // namespace kernel
