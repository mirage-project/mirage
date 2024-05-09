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

#include <cublas_v2.h>

namespace mirage {
namespace simulator {
class Simulator {
private:
  static Simulator *singleton;
  Simulator();
  ~Simulator();
  off_t offset;
  size_t work_space_size;
  void *base_ptr;

public:
  static Simulator *get_instance();
  void *allocate(size_t size_in_bytes);
  void free_all();

public:
  cublasHandle_t blas;
};

} // namespace simulator
} // namespace mirage
