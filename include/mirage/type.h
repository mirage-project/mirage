/* Copyright 2023 CMU
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
uint16_t const FP_P = 227;
uint16_t const FP_Q = 113;
uint32_t const FP_EXP_BASE = 3;
uint16_t const FP_PQ = 25651;
// FP_P_MUL_Q_MOD_1 is a multiplier of P and is 1 module Q
uint16_t const FP_P_MUL_Q_MOD_1 = 227;
// FP_Q_MUL_P_MOD_1 is a multiplier of Q and is 1 module P
uint16_t const FP_Q_MUL_P_MOD_1 = 25425;
size_t const MAX_SMEM_SIZE = 96 * 1024; // 96 KB
int const TB_REDUCTION_DIMX = 64;

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
  KN_MATMUL_OP = 1003,
  KN_REDUCTION_0_OP = 1004,
  KN_REDUCTION_1_OP = 1005,
  KN_REDUCTION_2_OP = 1006,
  KN_EXP_OP = 1007,
  KN_ADD_OP = 1008,
  KN_MUL_OP = 1009,
  KN_DIV_OP = 1010,
  KN_CUSTOMIZED_OP = 1999,
};

NLOHMANN_JSON_SERIALIZE_ENUM(KNOperatorType,
                             {
                                 {KN_UNKOWN, "kn_unkown"},
                                 {KN_INPUT_OP, "kn_input_op"},
                                 {KN_MATMUL_OP, "kn_matmul_op"},
                                 {KN_REDUCTION_0_OP, "kn_reduction_0_op"},
                                 {KN_REDUCTION_1_OP, "kn_reduction_1_op"},
                                 {KN_REDUCTION_2_OP, "kn_reduction_2_op"},
                                 {KN_EXP_OP, "kn_exp_op"},
                                 {KN_ADD_OP, "kn_add_op"},
                                 {KN_MUL_OP, "kn_mul_op"},
                                 {KN_DIV_OP, "kn_div_op"},
                                 {KN_CUSTOMIZED_OP, "kn_customized_op"},
                             })

enum TBOperatorType {
  TB_UNKOWN = 2000,
  TB_INPUT_OP = 2001,
  TB_OUTPUT_OP = 2002,
  TB_MATMUL_OP = 2003,
  TB_EXP_OP = 2007,
  TB_ADD_OP = 2008,
  TB_MUL_OP = 2009,
  TB_DIV_OP = 2010,
  TB_REDUCTION_FIRST_OP_ID = 2100,
  TB_REDUCTION_0_OP = 2101,
  TB_REDUCTION_1_OP = 2102,
  TB_REDUCTION_2_OP = 2103,
  TB_REDUCTION_0_TO_DIMX_OP = 2104,
  TB_REDUCTION_1_TO_DIMX_OP = 2105,
  TB_REDUCTION_2_TO_DIMX_OP = 2106,
  TB_REDUCTION_LAST_OP_ID = 2199,
  TB_CONCAT_FIRST_OP_ID = 2200,
  TB_CONCAT_0_OP = 2200,
  TB_CONCAT_1_OP = 2201,
  TB_CONCAT_2_OP = 2202,
  TB_CONCAT_LAST_OP_ID = 2210,
  TB_CONCAT_THEN_MATMUL_OP = 2011,
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
        {TB_ADD_OP, "tb_add_op"},
        {TB_MUL_OP, "tb_mul_op"},
        {TB_DIV_OP, "tb_div_op"},
        {TB_REDUCTION_0_TO_DIMX_OP, "tb_reduction_0_to_dimx_op"},
        {TB_REDUCTION_1_TO_DIMX_OP, "tb_reduction_1_to_dimx_op"},
        {TB_REDUCTION_2_TO_DIMX_OP, "tb_reduction_2_to_dimx_op"},
        {TB_CONCAT_0_OP, "tb_concat_0_op"},
        {TB_CONCAT_1_OP, "tb_concat_1_op"},
        {TB_CONCAT_2_OP, "tb_concat_2_op"},
        {TB_CONCAT_0_OP, "tb_concat_0_op"},
        {TB_CONCAT_1_OP, "tb_concat_1_op"},
        {TB_CONCAT_2_OP, "tb_concat_2_op"},
        {TB_CUSTOMIZED_OP, "tb_customized_op"},
    })

enum ActivationType {
  ACT_UNKOWN = 3000,
  ACT_EXP = 3001,
  ACT_RELU = 3002,
  ACT_GELU = 3003,
  ACT_NONE = 3100
};
} // namespace type
} // namespace mirage
