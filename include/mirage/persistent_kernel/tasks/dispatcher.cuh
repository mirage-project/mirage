/* Copyright 2025 CMU
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
#include "common.h"

namespace kernel {

#define DISPATCH_OUTPUT_SIZE(OUTPUT_SIZE, FUNC, T, ...)                        \
  if ((OUTPUT_SIZE) == 16) {                                                   \
    FUNC<T, 1, 16, 4096>(__VA_ARGS__);                                         \
  } else if ((OUTPUT_SIZE) == 32) {                                            \
    FUNC<T, 1, 32, 4096>(__VA_ARGS__);                                         \
  } else if ((OUTPUT_SIZE) == 64) {                                            \
    FUNC<T, 1, 64, 4096>(__VA_ARGS__);                                         \
  } else {                                                                     \
    printf("Unsupported output size: %d\n", OUTPUT_SIZE);                      \
  }

#define DISPATCH_OUTPUT_SIZE_FOR_RED_SIZE_4K(OUTPUT_SIZE, FUNC, T, ...)        \
  if ((OUTPUT_SIZE) == 16) {                                                   \
    FUNC<T, 1, 16, 4096>(__VA_ARGS__);                                         \
  } else if ((OUTPUT_SIZE) == 32) {                                            \
    FUNC<T, 1, 32, 4096>(__VA_ARGS__);                                         \
  } else if ((OUTPUT_SIZE) == 64) {                                            \
    FUNC<T, 1, 64, 4096>(__VA_ARGS__);                                         \
  } else if ((OUTPUT_SIZE) == 256) {                                           \
    FUNC<T, 1, 256, 4096>(__VA_ARGS__);                                        \
  } else if ((OUTPUT_SIZE) == 1600) {                                          \
    FUNC<T, 1, 1600, 4096>(__VA_ARGS__);                                       \
  } else {                                                                     \
    printf("Unsupported output size: %d\n", OUTPUT_SIZE);                      \
  }

#define DISPATCH_OUTPUT_SIZE_FOR_RED_SIZE_12K(OUTPUT_SIZE, FUNC, T, ...)       \
  if ((OUTPUT_SIZE) == 16) {                                                   \
    FUNC<T, 1, 16, 12288>(__VA_ARGS__);                                        \
  } else if ((OUTPUT_SIZE) == 32) {                                            \
    FUNC<T, 1, 32, 12288>(__VA_ARGS__);                                        \
  } else if ((OUTPUT_SIZE) == 64) {                                            \
    FUNC<T, 1, 64, 12288>(__VA_ARGS__);                                        \
  } else {                                                                     \
    printf("Unsupported output size: %d\n", OUTPUT_SIZE);                      \
  }

} // namespace kernel
