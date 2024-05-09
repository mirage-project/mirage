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

namespace mirage {
namespace layout {

enum DmemLayout {
  DmemRowMajor = 100,
  DmemColumnMajor = 101,
  DmemUnknownLayout = 199,
};

NLOHMANN_JSON_SERIALIZE_ENUM(DmemLayout,
                             {
                                 {DmemRowMajor, "DmemRowMajor"},
                                 {DmemColumnMajor, "DmemColumnMajor"},
                                 {DmemUnknownLayout, "DmemUnknownLayout"},
                             })

enum SmemLayout {
  SmemRowMajor = 200,
  SmemColumnMajor = 201,
  SmemRowMajorTensorOpMultiplicand_Crosswise16 = 202,
  SmemRowMajorTensorOpMultiplicand_Crosswise32 = 203,
  SmemRowMajorTensorOpMultiplicand_Crosswise64 = 204,
  SmemColumnMajorTensorOpMultiplicand_Crosswise16 = 205,
  SmemColumnMajorTensorOpMultiplicand_Crosswise32 = 206,
  SmemColumnMajorTensorOpMultiplicand_Crosswise64 = 207,
  SmemUnknownLayout = 299,
};

NLOHMANN_JSON_SERIALIZE_ENUM(SmemLayout,
                             {
                                 {SmemRowMajor, "SmemRowMajor"},
                                 {SmemColumnMajor, "SmemColumnMajor"},
                                 {SmemRowMajorTensorOpMultiplicand_Crosswise16, "SmemRowMajorTensorOpMultiplicand_Crosswise16"},
                                 {SmemRowMajorTensorOpMultiplicand_Crosswise32, "SmemRowMajorTensorOpMultiplicand_Crosswise32"},
                                 {SmemRowMajorTensorOpMultiplicand_Crosswise64, "SmemRowMajorTensorOpMultiplicand_Crosswise64"},
                                 {SmemColumnMajorTensorOpMultiplicand_Crosswise16, "SmemColumnMajorTensorOpMultiplicand_Crosswise16"},
                                 {SmemColumnMajorTensorOpMultiplicand_Crosswise32, "SmemColumnMajorTensorOpMultiplicand_Crosswise32"},
                                 {SmemColumnMajorTensorOpMultiplicand_Crosswise64, "SmemColumnMajorTensorOpMultiplicand_Crosswise64"},
                                 {SmemUnknownLayout, "SmemUnknownLayout"},
                             })

} // namespace layout
} // namespace mirage
