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
#include <string>
#include <unordered_map>

namespace mirage {
namespace type {

typedef uint16_t FPType;
typedef int64_t GuidType;

// only to be used in create_op in search.cc
inline std::unordered_map<std::string, float> CLAMP_MIN_MAX;

enum DataType {
  // 1-bit types
  // range: 900-909
  // 2-bit types
  // range: 910-919
  // 4-bit types
  // range: 920-929
  DT_INT4 = 920,
  // 8-bit types
  // range(float types): 930-934
  // range(int types): 935-939
  DT_FLOAT8 = 930,
  DT_INT8 = 935,
  // 16-bit types
  // range(float types): 940-944
  // range(int types): 945-949
  DT_BFLOAT16 = 940,
  DT_FLOAT16 = 941,
  DT_UINT16 = 945,
  // 32-bit types
  // range(float type): 950-954
  // range(int type): 955-959
  DT_FLOAT32 = 950,
  // 64-bit types
  // range(float types): 960-964
  DT_DOUBLE = 960,
  DT_UNKNOWN = 999,
};

size_t get_datatype_size(DataType type);
std::string get_datatype_str(DataType dtype);

enum KNOperatorType {
  KN_UNKOWN = 1000,
  KN_INPUT_OP = 1001,
  KN_OUTPUT_OP = 1002,
  KN_MATMUL_OP = 1003,
  // ElementUnary
  KN_EXP_OP = 1100,
  KN_SQUARE_OP = 1101,
  KN_SQRT_OP = 1102,
  KN_MUL_SCALAR_OP = 1103,
  KN_SILU_OP = 1104,
  KN_SIGMOID_OP = 1105,
  KN_GELU_OP = 1106,
  // non-lax elementunary ops
  KN_RELU_OP = 1150,
  KN_CLAMP_OP = 1151,
  KN_LOG_OP = 1160,
  // ElementBinary
  KN_ADD_OP = 1200,
  KN_MUL_OP = 1201,
  KN_DIV_OP = 1202,
  KN_POW_OP = 1203,
  // Reduction & Normalization
  KN_REDUCTION_0_OP = 1300,
  KN_REDUCTION_1_OP = 1301,
  KN_REDUCTION_2_OP = 1302,
  KN_RMS_NORM_OP = 1350,
  // Concat & Split
  KN_CONCAT_FIRST_OP_ID = 1400,
  KN_CONCAT_0_OP = 1400,
  KN_CONCAT_1_OP = 1401,
  KN_CONCAT_2_OP = 1402,
  KN_CONCAT_LAST_OP_ID = 1409,
  KN_SPLIT_FIRST_OP_ID = 1420,
  KN_SPLIT_0_OP = 1420,
  KN_SPLIT_1_OP = 1421,
  KN_SPLIT_2_OP = 1422,
  KN_CHUNK_0_OP = 1423,
  KN_CHUNK_1_OP = 1424,
  KN_CHUNK_2_OP = 1425,
  KN_SPLIT_LAST_OP_ID = 1429,
  // Communication
  KN_ALLREDUCE_OP = 1900,
  KN_CUSTOMIZED_OP = 1999,
};

NLOHMANN_JSON_SERIALIZE_ENUM(KNOperatorType,
                             {
                                 {KN_UNKOWN, "kn_unkown"},
                                 {KN_INPUT_OP, "kn_input_op"},
                                 {KN_OUTPUT_OP, "kn_output_op"},
                                 {KN_MATMUL_OP, "kn_matmul_op"},
                                 {KN_EXP_OP, "kn_exp_op"},
                                 {KN_SQUARE_OP, "kn_square_op"},
                                 {KN_SQRT_OP, "kn_sqrt_op"},
                                 {KN_MUL_SCALAR_OP, "kn_mul_scalar_op"},
                                 {KN_SILU_OP, "kn_silu_op"},
                                 {KN_SIGMOID_OP, "kn_sigmoid_op"},
                                 {KN_GELU_OP, "kn_gelu_op"},
                                 {KN_RELU_OP, "kn_relu_op"},
                                 {KN_CLAMP_OP, "kn_clamp_op"},
                                 {KN_LOG_OP, "kn_log_op"},
                                 {KN_ADD_OP, "kn_add_op"},
                                 {KN_MUL_OP, "kn_mul_op"},
                                 {KN_DIV_OP, "kn_div_op"},
                                 {KN_POW_OP, "kn_pow_op"},
                                 {KN_REDUCTION_0_OP, "kn_reduction_0_op"},
                                 {KN_REDUCTION_1_OP, "kn_reduction_1_op"},
                                 {KN_REDUCTION_2_OP, "kn_reduction_2_op"},
                                 {KN_RMS_NORM_OP, "kn_rms_norm_op"},
                                 {KN_CONCAT_FIRST_OP_ID,
                                  "kn_concat_first_op_id"},
                                 {KN_CONCAT_0_OP, "kn_concat_0_op"},
                                 {KN_CONCAT_1_OP, "kn_concat_1_op"},
                                 {KN_CONCAT_2_OP, "kn_concat_2_op"},
                                 {KN_CONCAT_LAST_OP_ID, "kn_concat_last_op_id"},
                                 {KN_SPLIT_FIRST_OP_ID, "kn_split_first_op_id"},
                                 {KN_SPLIT_0_OP, "kn_split_0_op"},
                                 {KN_SPLIT_1_OP, "kn_split_1_op"},
                                 {KN_SPLIT_2_OP, "kn_split_2_op"},
                                 {KN_CHUNK_0_OP, "kn_chunk_0_op"},
                                 {KN_CHUNK_1_OP, "kn_chunk_1_op"},
                                 {KN_CHUNK_2_OP, "kn_chunk_2_op"},
                                 {KN_SPLIT_LAST_OP_ID, "kn_split_last_op_id"},
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
  TB_MUL_SCALAR_OP = 2103,
  TB_SILU_OP = 2104,
  TB_SIGMOID_OP = 2105,
  TB_GELU_OP = 2106,
  // non-lax elementunary ops
  TB_RELU_OP = 2150,
  TB_CLAMP_OP = 2151,
  TB_LOG_OP = 2160,
  // ElementBinary
  TB_ADD_OP = 2200,
  TB_MUL_OP = 2201,
  TB_DIV_OP = 2202,
  TB_SUB_OP = 2203,
  TB_POW_OP = 2204,
  // Reduction and Normalization
  TB_REDUCTION_FIRST_OP_ID = 2300,
  TB_REDUCTION_0_OP = 2301,
  TB_REDUCTION_1_OP = 2302,
  TB_REDUCTION_2_OP = 2303,
  TB_REDUCTION_0_TO_DIMX_OP = 2304,
  TB_REDUCTION_1_TO_DIMX_OP = 2305,
  TB_REDUCTION_2_TO_DIMX_OP = 2306,
  TB_REDUCTION_0_MAX_OP = 2307,
  TB_REDUCTION_1_MAX_OP = 2308,
  TB_REDUCTION_2_MAX_OP = 2309,
  TB_REDUCTION_LAST_OP_ID = 2349,
  TB_RMS_NORM_OP = 2350,
  // Concat & Split
  TB_CONCAT_FIRST_OP_ID = 2400,
  TB_CONCAT_0_OP = 2400,
  TB_CONCAT_1_OP = 2401,
  TB_CONCAT_2_OP = 2402,
  TB_CONCAT_LAST_OP_ID = 2409,
  TB_CONCAT_THEN_MATMUL_OP = 2411,
  TB_SPLIT_FIRST_OP_ID = 2420,
  TB_SPLIT_0_OP = 2420,
  TB_SPLIT_1_OP = 2421,
  TB_SPLIT_2_OP = 2422,
  TB_SPLIT_LAST_OP_ID = 2429,
  // Forloop Accum
  // LD indicates last dimension
  TB_FORLOOP_ACCUM_FIRST_OP = 2500,
  TB_FORLOOP_ACCUM_NO_RED_OP = 2500,
  TB_FORLOOP_ACCUM_RED_LD_SUM_OP = 2501,
  TB_FORLOOP_ACCUM_RED_LD_MEAN_OP = 2502,
  TB_FORLOOP_ACCUM_RED_LD_RMS_OP = 2503,
  TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP = 2504,
  TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP = 2505,
  TB_FORLOOP_ACCUM_RED_LD_SUM_RESCALE_OP = 2506,
  TB_FORLOOP_ACCUM_MAX_OP = 2507,
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
        {TB_EXP_OP, "tb_exp_op"},
        {TB_SQUARE_OP, "tb_square_op"},
        {TB_SQRT_OP, "tb_sqrt_op"},
        {TB_MUL_SCALAR_OP, "tb_mul_scalar_op"},
        {TB_SILU_OP, "tb_silu_op"},
        {TB_SIGMOID_OP, "tb_sigmoid_op"},
        {TB_GELU_OP, "tb_gelu_op"},
        {TB_RELU_OP, "tb_relu_op"},
        {TB_CLAMP_OP, "tb_clamp_op"},
        {TB_LOG_OP, "tb_log_op"},
        {TB_ADD_OP, "tb_add_op"},
        {TB_MUL_OP, "tb_mul_op"},
        {TB_DIV_OP, "tb_div_op"},
        {TB_SUB_OP, "tb_sub_op"},
        {TB_POW_OP, "tb_pow_op"},
        {TB_REDUCTION_FIRST_OP_ID, "tb_reduction_first_op_id"},
        {TB_REDUCTION_0_OP, "tb_reduction_0_op"},
        {TB_REDUCTION_1_OP, "tb_reduction_1_op"},
        {TB_REDUCTION_2_OP, "tb_reduction_2_op"},
        {TB_REDUCTION_0_TO_DIMX_OP, "tb_reduction_0_to_dimx_op"},
        {TB_REDUCTION_1_TO_DIMX_OP, "tb_reduction_1_to_dimx_op"},
        {TB_REDUCTION_2_TO_DIMX_OP, "tb_reduction_2_to_dimx_op"},
        {TB_REDUCTION_0_MAX_OP, "tb_reduction_0_max_op"},
        {TB_REDUCTION_1_MAX_OP, "tb_reduction_1_max_op"},
        {TB_REDUCTION_2_MAX_OP, "tb_reduction_2_max_op"},
        {TB_REDUCTION_LAST_OP_ID, "tb_reduction_last_op_id"},
        {TB_RMS_NORM_OP, "tb_rms_norm_op"},
        {TB_CONCAT_FIRST_OP_ID, "tb_concat_first_op_id"},
        {TB_CONCAT_0_OP, "tb_concat_0_op"},
        {TB_CONCAT_1_OP, "tb_concat_1_op"},
        {TB_CONCAT_2_OP, "tb_concat_2_op"},
        {TB_CONCAT_LAST_OP_ID, "tb_concat_last_op_id"},
        {TB_CONCAT_THEN_MATMUL_OP, "tb_concat_then_matmul_op"},
        {TB_SPLIT_FIRST_OP_ID, "tb_split_first_op_id"},
        {TB_SPLIT_0_OP, "tb_split_0_op"},
        {TB_SPLIT_1_OP, "tb_split_1_op"},
        {TB_SPLIT_2_OP, "tb_split_2_op"},
        {TB_SPLIT_LAST_OP_ID, "tb_split_last_op_id"},
        {TB_FORLOOP_ACCUM_NO_RED_OP, "tb_forloop_accum_nored_op"},
        {TB_FORLOOP_ACCUM_RED_LD_SUM_OP, "tb_forloop_accum_red_ld_sum_op"},
        {TB_FORLOOP_ACCUM_RED_LD_MEAN_OP, "tb_forloop_accum_red_ld_mean_op"},
        {TB_FORLOOP_ACCUM_RED_LD_RMS_OP, "tb_forloop_accum_red_ld_rms_op"},
        {TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP,
         "tb_forloop_accum_redtox_ld_sum_op"},
        {TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP,
         "tb_forloop_accum_nored_rescale_op"},
        {TB_FORLOOP_ACCUM_RED_LD_SUM_RESCALE_OP,
         "tb_forloop_accum_red_ld_sum_rescale_op"},
        {TB_FORLOOP_ACCUM_MAX_OP, "tb_forloop_accum_max_op"},
        {TB_FORLOOP_ACCUM_LAST_OP, "tb_forloop_accum_last_op"},
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
