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


namespace mirage {
  namespace runtime {

template <int BATCH_SIZE, int OUT_DIM>
__device__ __forceinline__ void embedding_kernel(
    bfloat16_t* __restrict__ output,
    const uint16_t* __restrict__ input_ids,
    const bfloat16_t* __restrict__ embedding) 
{   
    for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < BATCH_SIZE * OUT_DIM; i += blockDim.x * gridDim.x) {
        int idx = i / OUT_DIM;
        int off = i % OUT_DIM;
        uint16_t wordIdx = reinterpret_cast<const uint16_t*>(input_ids)[idx];
        output[i] = embedding[wordIdx * OUT_DIM + off];
    }
}

}
}
  