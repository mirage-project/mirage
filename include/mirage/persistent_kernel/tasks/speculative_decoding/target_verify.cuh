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
#include "common.h"
#include "utils.cuh"
namespace kernel {

// simply sequential greedy search
template <int NUM_SPEC_TOKENS>
__device__ __forceinline__ void
    target_verify_greedy_kernel(void const *__restrict__ spec_token_id_ptr,
                         void const *__restrict__ target_token_id_ptr,
                         void *__restrict__ final_output_ptr) {
    int const *__restrict__ spec_token_id = static_cast<int const *>(spec_token_id_ptr);
    int const *__restrict__ target_token_id = static_cast<int const *>(target_token_id_ptr);
    int *__restrict__ accepted_spec_token_num = static_cast<int *>(final_output_ptr);
    for(int i = 0; i < NUM_SPEC_TOKENS; i++) {
        if(spec_token_id_ptr[i] != target_token_id_ptr[i]) {
            accepted_spec_token_num[0] = i;
            return;
        }
    }
    accepted_spec_token_num[0] = NUM_SPEC_TOKENS + 1;
}

} // namespace kernel
