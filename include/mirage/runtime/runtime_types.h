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
namespace runtime {

typedef unsigned long long int TaskId;
typedef unsigned long long int EventId;

enum TaskType {
  TASK_TERMINATE = 0,
  TASK_END_OF_ITERATION = 1,
  // compute task starts from 100
  TASK_RMS_NORM_LINEAR = 100,
};

} // namespace runtime
} // namespace mirage
