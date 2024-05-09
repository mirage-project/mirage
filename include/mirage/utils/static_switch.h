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

#define INT_SWITCH(VARNAME, CONSTNAME, ...)                                    \
  [&] {                                                                        \
    if (VARNAME <= 4) {                                                        \
      constexpr static int CONSTNAME = 4;                                      \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 8) {                                                 \
      constexpr static int CONSTNAME = 8;                                      \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 16) {                                                \
      constexpr static int CONSTNAME = 16;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 32) {                                                \
      constexpr static int CONSTNAME = 32;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 64) {                                                \
      constexpr static int CONSTNAME = 64;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 128) {                                               \
      constexpr static int CONSTNAME = 128;                                    \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 256) {                                               \
      constexpr static int CONSTNAME = 256;                                    \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      assert(false);                                                           \
    }                                                                          \
  }()

#define WARP_SHAPE_M_SWITCH(VARNAME, CONSTNAME, ...)                           \
  [&] {                                                                        \
    if (VARNAME <= 16) {                                                       \
      constexpr static int CONSTNAME = 16;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 32) {                                                \
      constexpr static int CONSTNAME = 32;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 64) {                                                \
      constexpr static int CONSTNAME = 64;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 128) {                                               \
      constexpr static int CONSTNAME = 128;                                    \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      assert(false);                                                           \
    }                                                                          \
  }()

#define WARP_SHAPE_N_SWITCH(VARNAME, CONSTNAME, ...)                           \
  [&] {                                                                        \
    if (VARNAME <= 8) {                                                        \
      constexpr static int CONSTNAME = 8;                                      \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 16) {                                                \
      constexpr static int CONSTNAME = 16;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 32) {                                                \
      constexpr static int CONSTNAME = 32;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 64) {                                                \
      constexpr static int CONSTNAME = 64;                                     \
      return __VA_ARGS__();                                                    \
    } else if (VARNAME <= 128) {                                                \
      constexpr static int CONSTNAME = 128;                                     \
      return __VA_ARGS__();                                                    \
    } else {                                                                   \
      assert(false);                                                           \
    }                                                                          \
  }()
