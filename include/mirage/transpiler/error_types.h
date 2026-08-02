/* Copyright 2023-2024 CMU
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

namespace mirage {
namespace transpiler {

enum TranspileErrorType {
  CUDA_T_SUCCESS = 0,
  CUDA_T_INSUFFICIENT_SMEM = 1,
  CUDA_T_LAYOUT_ERROR = 2,
  CUDA_T_CONFIG_ERROR = 3,
  // A matmul result consumed by another op inside the forloop (the fused
  // attention shape). Distinct from CONFIG so the message can say so -- it
  // otherwise surfaced as "block_dim does not match num_warp_groups * 128",
  // which points at the wrong thing entirely.
  CUDA_T_UNSUPPORTED_CHAINED_MATMUL = 4,
  // forloop_range == 1 with PIPELINED (TMA) inputs: the Blackwell producer
  // warpgroup deadlocks in producer_acquire (proven by ablating every
  // consumer wait -- the kernel still hung -- and by plain K-loop attention
  // hanging identically at FL=1). Pipelining buys nothing at one iteration;
  // callers should use forloop_dim=-1, as every working FL=1 graph does.
  CUDA_T_FL1_PIPELINED_DEADLOCK = 5,
  // A non-exp op fused into a Blackwell matmul's epilogue chain:
  // write_tC_to_sC applies only exp (NUM_EXPS_BEFORE_STORE), so any other
  // fused op would silently vanish from the computation (a fused SQUARE did).
  CUDA_T_UNSUPPORTED_FUSED_EPILOGUE = 6,
  CUDA_T_UNKOWN_ERRORS = 999,
};

} // namespace transpiler
} // namespace mirage
