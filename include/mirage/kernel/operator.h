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

#include "mirage/kernel/device_tensor.h"
#include "mirage/profile_result.h"
#include "mirage/utils/json_utils.h"
#include <vector>

namespace mirage {
namespace kernel {

class KNOperator {
public:
  KNOperator(mirage::type::KNOperatorType _type);
  KNOperator(mirage::type::KNOperatorType _type, DTensor const &input1);
  KNOperator(mirage::type::KNOperatorType _type,
             DTensor const &input1,
             DTensor const &input2);
  KNOperator(mirage::type::KNOperatorType _type,
             std::vector<DTensor> const &inputs);
  virtual ~KNOperator();
  virtual bool profile(ProfileResult &result) = 0;
  virtual bool fingerprint(void) = 0;
  mirage::type::KNOperatorType op_type;
  std::vector<DTensor> input_tensors;
  std::vector<DTensor> output_tensors;

  virtual operator json() const = 0;
};

class KNInputOp : public KNOperator {
public:
  KNInputOp(std::vector<int> const &dims,
            mirage::type::DataType data_type,
            mirage::layout::DmemLayout layout);
  ~KNInputOp();
  bool profile(ProfileResult &profile);
  bool fingerprint(void);

  operator json() const override;
};

} // namespace kernel
} // namespace mirage
