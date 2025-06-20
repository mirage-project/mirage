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

#ifdef MIRAGE_BACKEND_USE_CUDA
#include <vector_types.h>
#else
struct dim3 {
  unsigned int x, y, z;

  constexpr dim3(unsigned int _x = 1, unsigned int _y = 1, unsigned int _z = 1)
      : x(_x), y(_y), z(_z) {}
};
struct int3 {
  int x, y, z;

  constexpr int3(int _x = 1, int _y = 1, int _z = 1) : x(_x), y(_y), z(_z) {}
};
#endif
