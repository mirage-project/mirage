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
  CUDA_T_UNSUPPORTED_CHAINED_MATMUL = 4,
  CUDA_T_FL1_PIPELINED_DEADLOCK = 5,
  CUDA_T_UNSUPPORTED_FUSED_EPILOGUE = 6,
  CUDA_T_UNKOWN_ERRORS = 999,
};

inline char const *error_type_reason(TranspileErrorType e) {
  switch (e) {
    case CUDA_T_INSUFFICIENT_SMEM:
      return "the task body plans more shared memory than a worker has";
    case CUDA_T_LAYOUT_ERROR:
      return "an operand layout the MMA cannot read -- see the operand_ok "
             "guard in transpiler_tb_blackwell.cc";
    case CUDA_T_CONFIG_ERROR:
      return "an illegal MMA tile or block size";
    case CUDA_T_UNSUPPORTED_CHAINED_MATMUL:
      return "a matmul result feeds another op INSIDE the forloop (chained "
             "matmul / fused attention)";
    case CUDA_T_FL1_PIPELINED_DEADLOCK:
      return "forloop_range == 1 with pipelined inputs deadlocks the producer "
             "warpgroup; search with forloop_range >= 2";
    case CUDA_T_UNSUPPORTED_FUSED_EPILOGUE:
      return "an op consumes a matmul result directly; the epilogue only "
             "supports exp";
    default:
      return "unknown";
  }
}

} // namespace transpiler
} // namespace mirage
