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

__device__ __forceinline__ int atom_add_release_gpu_s32(int *addr, int val) {
  int old_val;
  asm volatile("atom.add.release.gpu.s32 %0,[%1],%2;"
               : "=r"(old_val)
               : "l"(addr), "r"(val)
               : "memory");
  return old_val;
}

__device__ __forceinline__ unsigned long long int
    atom_add_release_gpu_u64(unsigned long long int *addr,
                             unsigned long long int val) {
  unsigned long long int old_val;
  asm volatile("atom.add.release.gpu.u64 %0,[%1],%2;"
               : "=l"(old_val)
               : "l"(addr), "l"(val)
               : "memory");
  return old_val;
}

__device__ __forceinline__ unsigned long long int
    atom_cas_release_gpu_u64(unsigned long long int *addr,
                             unsigned long long int cmp,
                             unsigned long long int val) {
  unsigned long long int old_val;
  asm volatile("atom.cas.release.gpu.b64 %0,[%1],%2,%3;"
               : "=l"(old_val)
               : "l"(addr), "l"(cmp), "l"(val)
               : "memory");
  return old_val;
}

__device__ __forceinline__ unsigned long long int
    ld_acquire_gpu_u64(unsigned long long int *addr) {
  unsigned long long int val;
  asm volatile("ld.acquire.gpu.u64 %0, [%1];" : "=l"(val) : "l"(addr));
  return val;
}

__device__ __forceinline__ unsigned long long int
    ld_relaxed_gpu_u64(unsigned long long int *addr) {
  unsigned long long int val;
  asm volatile("ld.relaxed.gpu.u64 %0, [%1];" : "=l"(val) : "l"(addr));
  return val;
}

__device__ __forceinline__ void st_relaxed_gpu_u64(unsigned long long int *addr,
                                                   unsigned long long int val) {
  asm volatile("st.relaxed.gpu.u64 [%0], %1;" : : "l"(addr), "l"(val));
}
