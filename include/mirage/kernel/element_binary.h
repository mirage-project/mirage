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

#include "mirage/kernel/operator.h"

namespace mirage {
namespace kernel {

class KNElementBinaryOp : public mirage::kernel::KNOperator {
public:
  KNElementBinaryOp(Graph *_graph,
                    DTensor const &input1,
                    DTensor const &input2,
                    mirage::type::KNOperatorType type);
  ~KNElementBinaryOp();
  bool fingerprint(void) override;

  operator json() const override;
};

} // namespace kernel
} // namespace mirage
