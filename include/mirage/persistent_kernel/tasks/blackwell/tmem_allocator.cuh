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

__device__ inline static void tmem_alloc(uint32_t *dst_ptr,
                                         int tmem_alloc_cols) {
  assert(tmem_alloc_cols >= 32 && tmem_alloc_cols % 32 == 0);
  uint32_t dst_intptr =
      static_cast<uint32_t>(__cvta_generic_to_shared(dst_ptr));
  asm volatile(
      "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::
          "r"(dst_intptr),
      "r"(tmem_alloc_cols));
}

__device__ inline static void tmem_dealloc(uint32_t tmem_ptr,
                                           int tmem_alloc_cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32  %0, %1;\n" ::"r"(
                   tmem_ptr),
               "r"(tmem_alloc_cols));
}

__device__ inline static void release_lock() {
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
}

__device__ inline static void before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;\n");
}

__device__ inline static void after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;\n");
}

} // namespace blackwell
} // namespace kernel