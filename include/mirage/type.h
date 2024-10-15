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

#include "mirage/utils/json_utils.h"
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace mirage {
namespace type {

typedef uint16_t FPType;

enum DataType {
  DT_INT8 = 900,
  DT_UINT16 = 910,
  DT_BFLOAT16 = 920,
  DT_FLOAT16 = 921,
  DT_FLOAT32 = 930,
  DT_DOUBLE = 940,
  DT_UNKNOWN = 999,
};

size_t get_datatype_size(DataType type);

enum KNOperatorType {
  KN_UNKOWN = 1000,
  KN_INPUT_OP = 1001,
  KN_OUTPUT_OP = 1002,
  KN_MATMUL_OP = 1003,
  // ElementUnary
  KN_EXP_OP = 1100,
  KN_SQUARE_OP = 1101,
  KN_SQRT_OP = 1102,
  KN_SILU_OP = 1103,
  // ElementBinary
  KN_ADD_OP = 1200,
  KN_MUL_OP = 1201,
  KN_DIV_OP = 1202,
  // Reduction & Normalization
  KN_REDUCTION_0_OP = 1300,
  KN_REDUCTION_1_OP = 1301,
  KN_REDUCTION_2_OP = 1302,
  KN_RMS_NORM_OP = 1350,
  // Communication
  KN_ALLREDUCE_OP = 1400,
  KN_CUSTOMIZED_OP = 1999,
};

NLOHMANN_JSON_SERIALIZE_ENUM(KNOperatorType,
                             {
                                 {KN_UNKOWN, "kn_unkown"},
                                 {KN_INPUT_OP, "kn_input_op"},
                                 {KN_OUTPUT_OP, "kn_output_op"},
                                 {KN_MATMUL_OP, "kn_matmul_op"},
                                 {KN_REDUCTION_0_OP, "kn_reduction_0_op"},
                                 {KN_REDUCTION_1_OP, "kn_reduction_1_op"},
                                 {KN_REDUCTION_2_OP, "kn_reduction_2_op"},
                                 {KN_RMS_NORM_OP, "kn_rms_norm_op"},
                                 {KN_EXP_OP, "kn_exp_op"},
                                 {KN_SQUARE_OP, "kn_square_op"},
                                 {KN_SQRT_OP, "kn_sqrt_op"},
                                 {KN_SILU_OP, "kn_silu_op"},
                                 {KN_ADD_OP, "kn_add_op"},
                                 {KN_MUL_OP, "kn_mul_op"},
                                 {KN_DIV_OP, "kn_div_op"},
                                 {KN_ALLREDUCE_OP, "kn_allreduce_op"},
                                 {KN_CUSTOMIZED_OP, "kn_customized_op"},
                             })

enum TBOperatorType {
  TB_UNKOWN = 2000,
  TB_INPUT_OP = 2001,
  TB_OUTPUT_OP = 2002,
  TB_MATMUL_OP = 2003,
  // ElementUnary
  TB_EXP_OP = 2100,
  TB_SQUARE_OP = 2101,
  TB_SQRT_OP = 2102,
  TB_SILU_OP = 2103,
  TB_MUL_SCALAR_OP = 2104,
  // ElementBinary
  TB_ADD_OP = 2200,
  TB_MUL_OP = 2201,
  TB_DIV_OP = 2202,
  // Reduction and Normalization
  TB_REDUCTION_FIRST_OP_ID = 2300,
  TB_REDUCTION_0_OP = 2301,
  TB_REDUCTION_1_OP = 2302,
  TB_REDUCTION_2_OP = 2303,
  TB_REDUCTION_0_TO_DIMX_OP = 2304,
  TB_REDUCTION_1_TO_DIMX_OP = 2305,
  TB_REDUCTION_2_TO_DIMX_OP = 2306,
  TB_REDUCTION_LAST_OP_ID = 2349,
  TB_RMS_NORM_OP = 2350,
  // Concat
  TB_CONCAT_FIRST_OP_ID = 2400,
  TB_CONCAT_0_OP = 2400,
  TB_CONCAT_1_OP = 2401,
  TB_CONCAT_2_OP = 2402,
  TB_CONCAT_LAST_OP_ID = 2410,
  TB_CONCAT_THEN_MATMUL_OP = 2411,
  // Forloop Accum
  // LD indicates last dimension
  TB_FORLOOP_ACCUM_FIRST_OP = 2500,
  TB_FORLOOP_ACCUM_NO_RED_OP = 2500,
  TB_FORLOOP_ACCUM_RED_LD_SUM_OP = 2501,
  TB_FORLOOP_ACCUM_RED_LD_MEAN_OP = 2502,
  TB_FORLOOP_ACCUM_RED_LD_RMS_OP = 2503,
  TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP = 2504,
  TB_FORLOOP_ACCUM_LAST_OP = 2599,
  TB_CUSTOMIZED_OP = 2999
};

NLOHMANN_JSON_SERIALIZE_ENUM(
    TBOperatorType,
    {
        {TB_UNKOWN, "tb_unkown"},
        {TB_INPUT_OP, "tb_input_op"},
        {TB_OUTPUT_OP, "tb_output_op"},
        {TB_MATMUL_OP, "tb_matmul_op"},
        {TB_REDUCTION_0_OP, "tb_reduction_0_op"},
        {TB_REDUCTION_1_OP, "tb_reduction_1_op"},
        {TB_REDUCTION_2_OP, "tb_reduction_2_op"},
        {TB_EXP_OP, "tb_exp_op"},
        {TB_SQUARE_OP, "tb_square_op"},
        {TB_SQRT_OP, "tb_sqrt_op"},
        {TB_SILU_OP, "tb_silu_op"},
        {TB_MUL_SCALAR_OP, "tb_mul_scalar_op"},
        {TB_ADD_OP, "tb_add_op"},
        {TB_MUL_OP, "tb_mul_op"},
        {TB_DIV_OP, "tb_div_op"},
        {TB_REDUCTION_0_TO_DIMX_OP, "tb_reduction_0_to_dimx_op"},
        {TB_REDUCTION_1_TO_DIMX_OP, "tb_reduction_1_to_dimx_op"},
        {TB_REDUCTION_2_TO_DIMX_OP, "tb_reduction_2_to_dimx_op"},
        {TB_RMS_NORM_OP, "tb_rms_norm_op"},
        {TB_CONCAT_0_OP, "tb_concat_0_op"},
        {TB_CONCAT_1_OP, "tb_concat_1_op"},
        {TB_CONCAT_2_OP, "tb_concat_2_op"},
        {TB_CONCAT_0_OP, "tb_concat_0_op"},
        {TB_CONCAT_1_OP, "tb_concat_1_op"},
        {TB_CONCAT_2_OP, "tb_concat_2_op"},
        {TB_FORLOOP_ACCUM_NO_RED_OP, "tb_accum_nored_op"},
        {TB_FORLOOP_ACCUM_RED_LD_SUM_OP, "tb_accum_red_ld_sum_op"},
        {TB_FORLOOP_ACCUM_RED_LD_MEAN_OP, "tb_accum_red_ld_mean_op"},
        {TB_FORLOOP_ACCUM_RED_LD_RMS_OP, "tb_accum_red_ld_rms_op"},
        {TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP, "tb_accum_redtox_ld_sum_op"},
        {TB_CUSTOMIZED_OP, "tb_customized_op"},
    })

bool is_threadblock_element_unary(TBOperatorType op_type);

enum ActivationType {
  ACT_UNKOWN = 3000,
  ACT_EXP = 3001,
  ACT_RELU = 3002,
  ACT_GELU = 3003,
  ACT_SILU = 3004,
  ACT_NONE = 3099,
};

enum TBEpilogueType {
  TB_EPILOGUE_NONE = 3100,
  TB_EPILOGUE_ALLREDUCE = 3101,
  TB_EPILOGUE_ALLTOALL = 3102,
  TB_EPILOGUE_INVALID = 3199,
};

} // namespace type
} // namespace mirage
