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

class KNMatmulOp : public mirage::kernel::KNOperator {
public:
  KNMatmulOp(DTensor const &A, DTensor const &B);
  ~KNMatmulOp();
  bool profile(ProfileResult &profile) override;
  bool fingerprint(void) override;

  operator json() const override;
};

void from_json(json const &j, KNMatmulOp &op);

class MatmulKey {
public:
  MatmulKey(DTensor const &A, DTensor const &B);
  bool operator==(MatmulKey const &b) const;
  DTensor operand_a;
  DTensor operand_b;
};

} // namespace kernel
} // namespace mirage

namespace std {
template <>
struct hash<mirage::kernel::MatmulKey> {
  size_t operator()(mirage::kernel::MatmulKey const &) const;
};
} // namespace std
