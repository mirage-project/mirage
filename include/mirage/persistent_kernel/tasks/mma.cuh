
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

__device__ static __forceinline__ void
    mma_m16n16k16_bf16bf16bf32(float *C, uint32_t *A, uint32_t *B, float *D) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
               : "r"(A[0]),
                 "r"(A[1]),
                 "r"(A[2]),
                 "r"(A[3]),
                 "r"(B[0]),
                 "r"(B[1]),
                 "f"(C[0]),
                 "f"(C[1]),
                 "f"(C[2]),
                 "f"(C[3]));

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
               : "=f"(D[4]), "=f"(D[5]), "=f"(D[6]), "=f"(D[7])
               : "r"(A[0]),
                 "r"(A[1]),
                 "r"(A[2]),
                 "r"(A[3]),
                 "r"(B[2]),
                 "r"(B[3]),
                 "f"(C[4]),
                 "f"(C[5]),
                 "f"(C[6]),
                 "f"(C[7]));
}
} // namespace kernel
