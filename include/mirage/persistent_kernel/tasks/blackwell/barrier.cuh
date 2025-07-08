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
namespace kernel {
namespace blackwell {

__device__ inline static void init_barrier(uint32_t *smem_ptr,
                                           uint32_t arrive_count) {
  uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile("{\n\t"
               "mbarrier.init.shared::cta.b64 [%1], %0; \n"
               "}"
               :
               : "r"(arrive_count), "r"(smem_addr));
}

__device__ inline static void wait(uint32_t *smem_ptr, uint32_t phase) {
  uint32_t smem_addr =
      static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  // Arbitrarily large timer value after which try-wait expires and re-tries.
  uint32_t ticks = 0x989680;
  asm volatile("{\n\t"
               ".reg .pred       P1; \n\t"
               "LAB_WAIT: \n\t"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
               "@P1 bra DONE; \n\t"
               "bra     LAB_WAIT; \n\t"
               "DONE: \n\t"
               "}"
               :
               : "r"(smem_addr), "r"(phase), "r"(ticks));
}

} // namespace blackwell
} // namespace kernel